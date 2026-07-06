
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS


@METRICS.register_module()
class BoundaryF1Metric(BaseMetric):

    default_prefix = None

    def __init__(self,
                 num_classes: int,
                 ignore_index: int = 255,
                 epsilon: float = 0.02,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.epsilon = float(epsilon)

    @staticmethod
    def _to_numpy_mask(mask) -> np.ndarray:
        """Convert Tensor / PixelData / dict-like mask to numpy int64 [H, W]."""

        # MMSeg output may be dict-like: {'data': tensor}
        if isinstance(mask, dict):
            if 'data' in mask:
                mask = mask['data']
            elif 'sem_seg' in mask:
                mask = mask['sem_seg']
            else:
                raise TypeError(f'Unsupported mask dict keys: {mask.keys()}')

        # PixelData object
        if hasattr(mask, 'data'):
            mask = mask.data

        # Sometimes still dict after .data
        if isinstance(mask, dict):
            if 'data' in mask:
                mask = mask['data']
            else:
                raise TypeError(f'Unsupported nested mask dict keys: {mask.keys()}')

        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()

        mask = np.asarray(mask)

        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        elif mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]

        return mask.astype(np.int64)

    def _boundary_band(self, binary_mask):

        binary_mask = binary_mask.astype(np.uint8)

        if binary_mask.sum() == 0:
            return np.zeros_like(binary_mask, dtype=bool)

        h, w = binary_mask.shape

        dist = max(
            1,
            int(round(
                self.epsilon * max(h, w)
            ))
        )

        k = 2 * dist + 1

        kernel = np.ones((k, k), dtype=np.uint8)

        dilated = cv2.dilate(binary_mask, kernel, iterations=1)
        eroded = cv2.erode(binary_mask, kernel, iterations=1)

        boundary = (dilated != eroded)

        return boundary

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Collect per-image, per-class boundary TP/FP/FN."""
        for data_sample in data_samples:
            pred = self._to_numpy_mask(data_sample['pred_sem_seg'])
            target = self._to_numpy_mask(data_sample['gt_sem_seg'])

            valid = target != self.ignore_index

            class_results: List[Dict[str, int]] = []
            for cls_id in range(self.num_classes):
                pred_mask = (pred == cls_id) & valid
                target_mask = (target == cls_id) & valid

                pred_boundary = self._boundary_band(pred_mask)
                target_boundary = self._boundary_band(target_mask)

                # Keep ignored regions out of boundary matching.
                pred_boundary &= valid
                target_boundary &= valid

                tp = int(np.logical_and(pred_boundary, target_boundary).sum())
                fp = int(np.logical_and(pred_boundary, ~target_boundary).sum())
                fn = int(np.logical_and(~pred_boundary, target_boundary).sum())

                class_results.append(dict(tp=tp, fp=fp, fn=fn))

            self.results.append(class_results)

    def compute_metrics(self, results: List[List[Dict[str, int]]]) -> Dict[str, float]:
        """Aggregate BF1 over images and classes."""
        total_tp = np.zeros(self.num_classes, dtype=np.float64)
        total_fp = np.zeros(self.num_classes, dtype=np.float64)
        total_fn = np.zeros(self.num_classes, dtype=np.float64)

        for image_results in results:
            for cls_id, item in enumerate(image_results):
                total_tp[cls_id] += item['tp']
                total_fp[cls_id] += item['fp']
                total_fn[cls_id] += item['fn']

        bf1_per_class = []
        for cls_id in range(self.num_classes):
            denom = 2 * total_tp[cls_id] + total_fp[cls_id] + total_fn[cls_id]
            # If a class has no predicted and no GT boundary, skip it from mean.
            if denom == 0:
                bf1_per_class.append(np.nan)
            else:
                bf1_per_class.append(2 * total_tp[cls_id] / denom * 100.0)

        bf1_per_class = np.asarray(bf1_per_class, dtype=np.float64)
        metrics = {'mBF1': float(np.nanmean(bf1_per_class))}

        for cls_id, value in enumerate(bf1_per_class):
            if not np.isnan(value):
                metrics[f'BF1.class_{cls_id}'] = float(value)

        return metrics