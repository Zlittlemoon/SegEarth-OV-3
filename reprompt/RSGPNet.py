import torch
from torch import nn
import torch.nn.functional as F
from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

@MODELS.register_module()
class RSGPNetSegmentation(BaseSegmentor):
    def __init__(self, classname_path,
                 device=torch.device('cuda'),
                 prob_thd=0.0,
                 bg_idx=0,
                 slide_stride=0,
                 slide_crop=0,
                 confidence_threshold=0.5,
                 use_sem_seg=True,
                 use_presence_score=True,
                 use_transformer_decoder=True,

                 # mask re-prompt refinement
                 use_mask_reprompt=True,
                 reprompt_class_indices= [3,5,6],
                 reprompt_prob_thd=0.65,
                 reprompt_min_area=100,
                 reprompt_max_area_ratio=0.25,
                 reprompt_max_components=2,
                 reprompt_expand_ratio=0.04,
                 reprompt_boundary_width=3,
                 reprompt_blend=0.25,
                 reprompt_min_consistency_iou=0.50,
                 reprompt_area_ratio_low=0.50,
                 reprompt_area_ratio_high=1.80,

                 **kwargs):
        super().__init__()

        self.device = device
        # Initialize SAM3 model
        model = build_sam3_image_model(
            bpe_path="./sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            checkpoint_path='weights/sam3/sam3.pt',
            device="cuda"
        )
        self.processor = Sam3Processor(model, confidence_threshold=confidence_threshold, device=device)
        self.query_words, self.query_idx = get_cls_idx(classname_path)
        self.num_cls = max(self.query_idx) + 1
        self.num_queries = len(self.query_idx)
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        self.prob_thd = prob_thd
        self.bg_idx = bg_idx
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.confidence_threshold = confidence_threshold
        self.use_sem_seg = use_sem_seg
        self.use_presence_score = use_presence_score
        self.use_transformer_decoder = use_transformer_decoder

        self.use_mask_reprompt = use_mask_reprompt
        self.reprompt_class_indices = reprompt_class_indices
        self.reprompt_prob_thd = reprompt_prob_thd
        self.reprompt_min_area = reprompt_min_area
        self.reprompt_max_area_ratio = reprompt_max_area_ratio
        self.reprompt_max_components = reprompt_max_components
        self.reprompt_expand_ratio = reprompt_expand_ratio
        self.reprompt_boundary_width = reprompt_boundary_width
        self.reprompt_blend = reprompt_blend
        self.reprompt_min_consistency_iou = reprompt_min_consistency_iou
        self.reprompt_area_ratio_low = reprompt_area_ratio_low
        self.reprompt_area_ratio_high = reprompt_area_ratio_high

    def _resize_2d_logit(self, logit, h, w):
        logit = logit.squeeze()

        if logit.shape != (h, w):
            logit = F.interpolate(
                logit.view(1, 1, *logit.shape),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze()

        return logit

    def _should_reprompt(self, query_idx, coarse_logit):
        if not self.use_mask_reprompt:
            return False

        class_id = int(self.query_idx[query_idx].item())

        if self.reprompt_class_indices is not None:
            if class_id not in self.reprompt_class_indices:
                return False

        prob = torch.sigmoid(coarse_logit)
        mask = prob > self.reprompt_prob_thd

        area = mask.sum().item()
        total = mask.numel()
        area_ratio = area / max(total, 1)

        if area < self.reprompt_min_area:
            return False

        if area_ratio > self.reprompt_max_area_ratio:
            return False

        return True

    def _xyxy_to_norm_cxcywh(self, box_xyxy, width, height):
        x1, y1, x2, y2 = box_xyxy

        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        cx = x1 + bw / 2.0
        cy = y1 + bh / 2.0

        return [
            float(cx / width),
            float(cy / height),
            float(bw / width),
            float(bh / height)
        ]

    def _extract_boxes_from_mask(self, prob_map, width, height):
        if cv2 is None:
            return []

        mask = (prob_map.detach().float().cpu().numpy() > self.reprompt_prob_thd).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        boxes = []
        for lab in range(1, num_labels):
            x, y, bw, bh, area = stats[lab]

            if area < self.reprompt_min_area:
                continue

            x1, y1 = float(x), float(y)
            x2, y2 = float(x + bw), float(y + bh)

            expand_x = bw * self.reprompt_expand_ratio
            expand_y = bh * self.reprompt_expand_ratio

            x1 = max(0.0, x1 - expand_x)
            y1 = max(0.0, y1 - expand_y)
            x2 = min(float(width), x2 + expand_x)
            y2 = min(float(height), y2 + expand_y)

            comp_prob = prob_map[int(y):int(y + bh), int(x):int(x + bw)].mean().item()
            boxes.append((comp_prob, area, [x1, y1, x2, y2]))

        boxes = sorted(boxes, key=lambda t: (t[0], t[1]), reverse=True)
        boxes = boxes[:self.reprompt_max_components]

        return [b[-1] for b in boxes]

    def _make_boundary_band(self, prob_map):

        r = int(self.reprompt_boundary_width)
        if r <= 0:
            return torch.ones_like(prob_map, dtype=torch.bool)

        mask = (prob_map > self.reprompt_prob_thd).float().view(1, 1, *prob_map.shape)
        k = 2 * r + 1

        dilated = F.max_pool2d(mask, kernel_size=k, stride=1, padding=r)

        eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=k, stride=1, padding=r)

        band = (dilated - eroded).squeeze() > 0
        return band

    def _collect_instance_logits_from_state(self, inference_state, h, w):

        if 'masks_logits' not in inference_state:
            return None

        if inference_state['masks_logits'].shape[0] == 0:
            return None

        refined_logits = torch.full((h, w), -1e4, device=self.device)

        inst_len = inference_state['masks_logits'].shape[0]
        for inst_id in range(inst_len):
            instance_logits = inference_state['masks_logits'][inst_id].squeeze()
            instance_score = inference_state['object_score'][inst_id]

            instance_logits = self._resize_2d_logit(instance_logits, h, w)
            refined_logits = torch.maximum(refined_logits, instance_logits * instance_score)

        return refined_logits

    def _mask_reprompt_refine(self, image, inference_state, coarse_logit):

        width, height = image.size
        h, w = coarse_logit.shape

        coarse_prob = torch.sigmoid(coarse_logit)
        coarse_mask = coarse_prob > self.reprompt_prob_thd

        coarse_area = coarse_mask.sum().item()
        if coarse_area < self.reprompt_min_area:
            return coarse_logit

        boxes_xyxy = self._extract_boxes_from_mask(coarse_prob, width, height)
        if len(boxes_xyxy) == 0:
            return coarse_logit

        refined_state = inference_state

        for box_xyxy in boxes_xyxy:
            norm_box = self._xyxy_to_norm_cxcywh(box_xyxy, width, height)

            out_state = self.processor.add_geometric_prompt(
                state=refined_state,
                box=norm_box,
                label=True
            )

            if out_state is not None:
                refined_state = out_state

        refined_logit = self._collect_instance_logits_from_state(refined_state, h, w)

        if refined_logit is None:
            return coarse_logit

        refined_prob = torch.sigmoid(refined_logit)
        refined_mask = refined_prob > self.reprompt_prob_thd

        refined_area = refined_mask.sum().item()
        if refined_area < self.reprompt_min_area:
            return coarse_logit

        inter = torch.logical_and(coarse_mask, refined_mask).sum().float()
        union = torch.logical_or(coarse_mask, refined_mask).sum().float()
        consistency_iou = (inter / (union + 1e-6)).item()

        area_ratio = refined_area / max(coarse_area, 1)

        if consistency_iou < self.reprompt_min_consistency_iou:
            return coarse_logit

        if area_ratio < self.reprompt_area_ratio_low or area_ratio > self.reprompt_area_ratio_high:
            return coarse_logit

        boundary_band = self._make_boundary_band(coarse_prob)

        final_logit = coarse_logit.clone()

        useful_band = boundary_band & (refined_prob > coarse_prob)

        final_logit[useful_band] = (
            (1.0 - self.reprompt_blend) * coarse_logit[useful_band]
            + self.reprompt_blend * refined_logit[useful_band]
        )

        return final_logit

    def _inference_single_view(self, image):
        """Inference on a single PIL image or crop patch."""
        w, h = image.size
        seg_logits = torch.zeros((self.num_queries, h, w), device=self.device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            base_state = self.processor.set_image(image)

            for query_idx, query_word in enumerate(self.query_words):
                inference_state = base_state

                self.processor.reset_all_prompts(inference_state)

                inference_state = self.processor.set_text_prompt(
                    state=inference_state,
                    prompt=query_word
                )

                query_logit = torch.zeros((h, w), device=self.device)

                if self.use_transformer_decoder:
                    inst_logit = self._collect_instance_logits_from_state(
                        inference_state, h, w
                    )
                    if inst_logit is not None:
                        query_logit = torch.maximum(query_logit, inst_logit)

                if self.use_sem_seg and 'semantic_mask_logits' in inference_state:
                    semantic_logits = inference_state['semantic_mask_logits']
                    semantic_logits = self._resize_2d_logit(semantic_logits, h, w)
                    query_logit = torch.maximum(query_logit, semantic_logits)

                if self._should_reprompt(query_idx, query_logit):
                    query_logit = self._mask_reprompt_refine(
                        image=image,
                        inference_state=inference_state,
                        coarse_logit=query_logit
                    )

                if self.use_presence_score and "presence_score" in inference_state:
                    query_logit = query_logit * inference_state["presence_score"]

                seg_logits[query_idx] = query_logit

        return seg_logits

    def slide_inference(self, image, stride, crop_size):
        """Inference by sliding-window with overlap using PIL cropping."""
        w_img, h_img = image.size

        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size

        # Initialize accumulators
        preds = torch.zeros((self.num_queries, h_img, w_img), device=self.device)
        count_mat = torch.zeros((1, h_img, w_img), device=self.device)

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)

                # Adjust start points to ensure crop size is valid at boundaries
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                # Crop via PIL
                crop_img = image.crop((x1, y1, x2, y2))

                # Inference on crop
                crop_seg_logit = self._inference_single_view(crop_img)

                # Accumulate results
                preds[:, y1:y2, x1:x2] += crop_seg_logit
                count_mat[:, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0, "Error: Sparse sliding window coverage."

        preds = preds / count_mat
        return preds

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            # Fallback for meta info construction
            batch_img_metas = [
                                  dict(
                                      ori_shape=inputs.shape[2:],
                                      img_shape=inputs.shape[2:],
                                      pad_shape=inputs.shape[2:],
                                      padding_size=[0, 0, 0, 0])
                              ] * inputs.shape[0]

        for i, meta in enumerate(batch_img_metas):
            # Load original image to preserve details for SAM3
            image_path = meta.get('img_path')
            image = Image.open(image_path).convert('RGB')
            ori_shape = meta['ori_shape']

            # Determine inference mode
            if self.slide_crop > 0 and (self.slide_crop < image.size[0] or self.slide_crop < image.size[1]):
                seg_logits = self.slide_inference(image, self.slide_stride, self.slide_crop)
            else:
                seg_logits = self._inference_single_view(image)

            # Resize to original shape if necessary (e.g. padding effects)
            if seg_logits.shape[-2:] != ori_shape:
                seg_logits = F.interpolate(
                    seg_logits.unsqueeze(0),
                    size=ori_shape,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

            # Post-processing
            if self.num_cls != self.num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(self.num_cls, len(self.query_idx), 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]
                seg_pred = seg_logits.argmax(0, keepdim=True)

            seg_pred = torch.argmax(seg_logits, dim=0)

            # Apply probability threshold
            max_vals = seg_logits.max(0)[0]
            seg_pred[max_vals < self.prob_thd] = self.bg_idx

            data_samples[i].set_data({
                'seg_logits': PixelData(**{'data': seg_logits}),
                'pred_sem_seg': PixelData(**{'data': seg_pred.unsqueeze(0)})
            })

        return data_samples

    def _forward(data_samples):
        """
    """

    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """

    def extract_feat(self, inputs):
        """
        """

    def loss(self, inputs, data_samples):
        """
        """


def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(',')
        names_i = [i.strip() for i in names_i]
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices