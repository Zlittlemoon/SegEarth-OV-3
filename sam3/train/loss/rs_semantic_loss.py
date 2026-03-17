from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from sam3.model.model_misc import SAM3Output


class RSSemanticOnlyLoss(nn.Module):
    def __init__(self, loss_fn_semantic_seg: nn.Module, log_prefix: str = "sem") -> None:
        super().__init__()
        self.loss_fn_semantic_seg = loss_fn_semantic_seg
        self.log_prefix = log_prefix

    def forward(self, outputs: Any, targets: Any, *args, **kwargs) -> Dict[str, torch.Tensor]:
        return self.compute_loss(outputs, targets)

    def _unwrap_outputs(self, outputs: Any) -> Dict[str, Any]:
        if isinstance(outputs, SAM3Output):
            if len(outputs) == 0:
                raise ValueError("SAM3Output is empty.")
            last_item = outputs[-1]
            if isinstance(last_item, list):
                if len(last_item) == 0:
                    raise ValueError("Last stage in SAM3Output is empty.")
                last_item = last_item[-1]
            if not isinstance(last_item, dict):
                raise TypeError(f"Expected last SAM3Output item to be dict, got {type(last_item)}")
            return last_item

        if isinstance(outputs, list):
            if len(outputs) == 0:
                raise ValueError("outputs list is empty.")
            last_item = outputs[-1]
            if isinstance(last_item, list):
                if len(last_item) == 0:
                    raise ValueError("Last nested outputs list is empty.")
                last_item = last_item[-1]
            if not isinstance(last_item, dict):
                raise TypeError(f"Expected last outputs item to be dict, got {type(last_item)}")
            return last_item

        if isinstance(outputs, dict):
            return outputs

        raise TypeError(f"Unsupported outputs type: {type(outputs)}")

    def _collate_targets(
        self,
        targets: Any,
        device: torch.device,
        output_hw: tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        H, W = output_hw

        if isinstance(targets, dict):
            if "masks" not in targets:
                raise KeyError(f"Expected 'masks' in targets dict, got keys={list(targets.keys())}")

            masks = targets["masks"]
            if not torch.is_tensor(masks):
                masks = torch.as_tensor(masks, device=device)
            else:
                masks = masks.to(device)

            # 空 mask -> 全零 mask
            if masks.numel() == 0:
                masks = torch.zeros((1, H, W), dtype=torch.bool, device=device)
                num_boxes = torch.zeros((1,), dtype=torch.long, device=device)
                return {"masks": masks, "num_boxes": num_boxes}

            # 统一成 [B, H, W]
            if masks.ndim == 2:
                masks = masks.unsqueeze(0)  # [1, H, W]
            elif masks.ndim == 3:
                if masks.shape[0] == 1:
                    pass
                else:
                    masks = masks.bool().any(dim=0, keepdim=True)  # [1, H, W]
            else:
                raise ValueError(f"Unsupported mask shape in targets dict: {tuple(masks.shape)}")

            num_boxes = torch.ones((masks.shape[0],), dtype=torch.long, device=device)

            return {
                "masks": masks,
                "num_boxes": num_boxes,
            }

        if not isinstance(targets, list):
            raise TypeError(f"Unsupported targets type: {type(targets)}")

        if len(targets) == 0:
            raise ValueError("targets list is empty.")

        batched_masks = []
        batched_num_boxes = []

        for i, t in enumerate(targets):
            if not isinstance(t, dict):
                raise TypeError(f"targets[{i}] should be dict, got {type(t)}")
            if "masks" not in t:
                raise KeyError(f"targets[{i}] missing 'masks', got keys={list(t.keys())}")

            m = t["masks"]
            if not torch.is_tensor(m):
                m = torch.as_tensor(m)

            # 空 mask，例如 shape == (0,)
            if m.numel() == 0:
                m = torch.zeros((H, W), dtype=torch.bool, device=device)
                batched_masks.append(m)
                batched_num_boxes.append(0)
                continue

            # 支持 [H,W] / [1,H,W] / [N,H,W]
            if m.ndim == 2:
                pass
            elif m.ndim == 3:
                if m.shape[0] == 1:
                    m = m[0]
                else:
                    # semantic-only: 多实例 union 成一个语义 mask
                    m = m.bool().any(dim=0)
            else:
                raise ValueError(f"Unsupported mask shape for targets[{i}]: {tuple(m.shape)}")

            batched_masks.append(m.to(device))
            batched_num_boxes.append(1)

        masks = torch.stack(batched_masks, dim=0)  # [B, H, W]
        num_boxes = torch.tensor(batched_num_boxes, dtype=torch.long, device=device)

        return {
            "masks": masks,
            "num_boxes": num_boxes,
        }

    def compute_loss(self, outputs: Any, targets: Any) -> Dict[str, torch.Tensor]:
        outputs = self._unwrap_outputs(outputs)

        if "semantic_seg" not in outputs:
            raise KeyError(
                f"RSSemanticOnlyLoss expects 'semantic_seg' in outputs, got keys={list(outputs.keys())}"
            )

        device = outputs["semantic_seg"].device
        output_hw = tuple(outputs["semantic_seg"].shape[-2:])
        targets = self._collate_targets(targets, device, output_hw)

        loss_dict = self.loss_fn_semantic_seg(outputs, targets)

        cleaned = {}
        for k, v in loss_dict.items():
            cleaned[k] = v if torch.is_tensor(v) else torch.as_tensor(v, device=device)

        if len(cleaned) == 0:
            cleaned[f"{self.log_prefix}_zero"] = outputs["semantic_seg"].sum() * 0.0

        return cleaned