from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from sam3.model.model_misc import SAM3Output


def _align_semantic_targets_to_pred_batch(
    outputs: Dict[str, Any], targets: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Make semantic targets batch-aligned with outputs["semantic_seg"].

    Expected final shape:
      pred:    [B, 1, H, W]
      targets: [B, H, W] or [B, 1, H, W]

    Handles:
      1) target is [1, H, W] but pred batch is B>1 -> repeat to B
      2) target is [N, H, W] with N != B and B==1  -> union to [1, H, W]
      3) target already aligned                     -> keep as-is
    """
    if "semantic_seg" not in outputs:
        return targets

    pred = outputs["semantic_seg"]
    pred_bs = int(pred.shape[0])

    if "masks" not in targets:
        return targets

    masks = targets["masks"]

    # [B,1,H,W] -> [B,H,W]
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    # single target replicated to multi-image batch
    if masks.ndim == 3 and masks.shape[0] == 1 and pred_bs > 1:
        masks = masks.repeat(pred_bs, 1, 1)

    # mismatch fallback
    if masks.ndim == 3 and masks.shape[0] != pred_bs:
        if pred_bs == 1:
            masks = masks.bool().any(dim=0, keepdim=True).to(masks.dtype)
        else:
            raise RuntimeError(
                f"[RSSemanticOnlyLoss] semantic target batch mismatch: "
                f"pred batch={pred_bs}, target masks shape={tuple(masks.shape)}"
            )

    targets["masks"] = masks
    if "semantic_masks" in targets:
        targets["semantic_masks"] = masks

    # SemanticSegCriterion expects num_boxes shape [B]
    if "num_boxes" in targets and torch.is_tensor(targets["num_boxes"]):
        if targets["num_boxes"].numel() != pred_bs:
            targets["num_boxes"] = torch.ones(
                pred_bs, dtype=torch.long, device=masks.device
            )

    return targets


class RSSemanticOnlyLoss(nn.Module):
    def __init__(self, loss_fn_semantic_seg: nn.Module, log_prefix: str = "sem") -> None:
        super().__init__()
        self.loss_fn_semantic_seg = loss_fn_semantic_seg
        self.log_prefix = log_prefix

    def forward(self, outputs: Any, targets: Any, *args, **kwargs) -> Dict[str, torch.Tensor]:
        return self.compute_loss(outputs, targets)
    
    def _dbg_tensor(self, name: str, x: Any) -> None:
        if not torch.is_tensor(x):
            print(f"[DEBUG] {name}: type={type(x)}")
            return

        msg = f"[DEBUG] {name}: shape={tuple(x.shape)} dtype={x.dtype}"
        if x.numel() > 0:
            if x.dtype == torch.bool:
                msg += f" true_ratio={x.float().mean().item():.6f}"
            else:
                msg += (
                    f" min={x.min().item():.6f}"
                    f" max={x.max().item():.6f}"
                    f" mean={x.float().mean().item():.6f}"
                )
        print(msg)

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

        def _pick_mask_tensor(t: Dict[str, Any]) -> torch.Tensor:
            # 优先用 semantic_masks；没有时退回 masks
            if "semantic_masks" in t:
                m = t["semantic_masks"]
            elif "masks" in t:
                m = t["masks"]
            else:
                raise KeyError(
                    f"Expected one of ['semantic_masks', 'masks'], got keys={list(t.keys())}"
                )

            if not torch.is_tensor(m):
                m = torch.as_tensor(m, device=device)
            else:
                m = m.to(device)

            return m

        def _empty_semantic() -> Dict[str, torch.Tensor]:
            masks = torch.zeros((1, H, W), dtype=torch.bool, device=device)
            num_boxes = torch.zeros((1,), dtype=torch.long, device=device)
            return {
                "masks": masks, 
                "semantic_masks": masks,
                "num_boxes": num_boxes
            }

        # ------------------------------------------------------------------
        # Case 1: targets is already a dict
        # ------------------------------------------------------------------
        if isinstance(targets, dict):
            masks = _pick_mask_tensor(targets)

            if masks.numel() == 0:
                return _empty_semantic()

            # normalize to [B, H, W]
            if masks.ndim == 2:
                masks = masks.unsqueeze(0)  # [1, H, W]
            elif masks.ndim == 3:
                # already [B, H, W]
                pass
            elif masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]  # [B,1,H,W] -> [B,H,W]
            else:
                raise ValueError(f"Unsupported mask shape in targets dict: {tuple(masks.shape)}")

            if "num_boxes" in targets and torch.is_tensor(targets["num_boxes"]):
                num_boxes = targets["num_boxes"].to(device)
                if num_boxes.ndim == 0:
                    num_boxes = num_boxes.unsqueeze(0)
                if num_boxes.numel() != masks.shape[0]:
                    num_boxes = torch.ones((masks.shape[0],), dtype=torch.long, device=device)
            else:
                num_boxes = torch.ones((masks.shape[0],), dtype=torch.long, device=device)

            return {
                "masks": masks.bool(),
                "semantic_masks": masks.bool(),
                "num_boxes": num_boxes.long(),
            }

        # ------------------------------------------------------------------
        # Case 2: targets is a list
        # ------------------------------------------------------------------
        if not isinstance(targets, list):
            raise TypeError(f"Unsupported targets type: {type(targets)}")

        if len(targets) == 0:
            raise ValueError("targets list is empty.")

        # ------------------------------------------------------------------
        # Special case: current training path passes a list of length 1,
        # whose first element is ALREADY a batched target dict:
        #   targets[0]["masks"] or ["semantic_masks"] has shape [B,H,W]
        # We must KEEP it batched, instead of union-ing along dim 0.
        # ------------------------------------------------------------------
        if len(targets) == 1 and isinstance(targets[0], dict):
            t0 = targets[0]
            masks = _pick_mask_tensor(t0)

            if masks.numel() == 0:
                return _empty_semantic()

            # already batched semantic masks
            if masks.ndim == 3:
                # [B, H, W]
                num_boxes = t0.get("num_boxes", None)
                if torch.is_tensor(num_boxes):
                    num_boxes = num_boxes.to(device)
                    if num_boxes.ndim == 0:
                        num_boxes = num_boxes.unsqueeze(0)

                    # 关键判断：
                    # 如果 num_boxes 的长度和 masks 第一维相同，
                    # 说明 masks 的第一维是 batch 维，不是 instance 维
                    if num_boxes.numel() == masks.shape[0]:
                        return {
                            "masks": masks.bool(),
                            "semantic_masks": masks.bool(),
                            "num_boxes": num_boxes.long(),
                        }

                # 没有可靠 num_boxes 时，也优先按 batched masks 处理
                return {
                    "masks": masks.bool(),
                    "semantic_masks": masks.bool(),
                    "num_boxes": torch.ones((masks.shape[0],), dtype=torch.long, device=device),
                }

            if masks.ndim == 4 and masks.shape[1] == 1:
                # [B,1,H,W] -> [B,H,W]
                masks = masks[:, 0]
                num_boxes = t0.get("num_boxes", None)
                if torch.is_tensor(num_boxes):
                    num_boxes = num_boxes.to(device)
                    if num_boxes.ndim == 0:
                        num_boxes = num_boxes.unsqueeze(0)
                    if num_boxes.numel() != masks.shape[0]:
                        num_boxes = torch.ones((masks.shape[0],), dtype=torch.long, device=device)
                else:
                    num_boxes = torch.ones((masks.shape[0],), dtype=torch.long, device=device)

                return {
                    "masks": masks.bool(),
                    "semantic_masks": masks.bool(),
                    "num_boxes": num_boxes.long(),
                }

            if masks.ndim == 2:
                masks = masks.unsqueeze(0).bool()
                return {
                    "masks": masks,
                    "semantic_masks": masks,
                    "num_boxes": torch.ones((1,), dtype=torch.long, device=device),
                }

            raise ValueError(
                f"Unsupported single-entry batched target mask shape: {tuple(masks.shape)}"
            )

        # ------------------------------------------------------------------
        # General case: list of per-sample targets
        # Keep old behavior here.
        # ------------------------------------------------------------------
        batched_masks = []
        batched_num_boxes = []

        for i, t in enumerate(targets):
            if not isinstance(t, dict):
                raise TypeError(f"targets[{i}] should be dict, got {type(t)}")

            m = _pick_mask_tensor(t)

            if m.numel() == 0:
                m = torch.zeros((H, W), dtype=torch.bool, device=device)
                batched_masks.append(m)
                batched_num_boxes.append(0)
                continue

            # support [H,W] / [1,H,W] / [N,H,W]
            if m.ndim == 2:
                pass
            elif m.ndim == 3:
                if m.shape[0] == 1:
                    m = m[0]
                else:
                    # 这里才是“单样本多实例 -> semantic union”
                    m = m.bool().any(dim=0)
            elif m.ndim == 4 and m.shape[1] == 1:
                if m.shape[0] == 1:
                    m = m[0, 0]
                else:
                    m = m[:, 0].bool().any(dim=0)
            else:
                raise ValueError(f"Unsupported mask shape for targets[{i}]: {tuple(m.shape)}")

            batched_masks.append(m.to(device).bool())
            batched_num_boxes.append(1)

        masks = torch.stack(batched_masks, dim=0)  # [B,H,W]
        num_boxes = torch.tensor(batched_num_boxes, dtype=torch.long, device=device)

        return {
            "masks": masks.bool(),
            "semantic_masks": masks.bool(),
            "num_boxes": num_boxes.long(),
        }

    def compute_loss(self, outputs: Any, targets: Any) -> Dict[str, torch.Tensor]:
        outputs = self._unwrap_outputs(outputs)

        if "semantic_seg" not in outputs:
            raise KeyError(
                f"RSSemanticOnlyLoss expects 'semantic_seg' in outputs, got keys={list(outputs.keys())}"
            )

        device = outputs["semantic_seg"].device
        output_hw = tuple(outputs["semantic_seg"].shape[-2:])

        # ------------------------------------------------------------------
        # RAW TARGETS
        # ------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("[DEBUG][RAW TARGETS]")
        print("=" * 80)
        print("[DEBUG] raw targets type =", type(targets))

        if isinstance(targets, dict):
            print("[DEBUG] raw target keys =", list(targets.keys()))
            for k in ["masks", "semantic_masks", "num_boxes", "is_valid_mask", "is_exhaustive"]:
                if k in targets:
                    self._dbg_tensor(f"raw targets[{k}]", targets[k])

        elif isinstance(targets, list):
            print("[DEBUG] raw targets list len =", len(targets))
            if len(targets) > 0 and isinstance(targets[0], dict):
                print("[DEBUG] raw targets[0] keys =", list(targets[0].keys()))
                for k in ["masks", "semantic_masks", "num_boxes", "is_valid_mask", "is_exhaustive"]:
                    if k in targets[0]:
                        self._dbg_tensor(f"raw targets[0][{k}]", targets[0][k])

        # ------------------------------------------------------------------
        # COLLATE TARGETS
        # ------------------------------------------------------------------
        targets = self._collate_targets(targets, device, output_hw)

        print("\n[DEBUG][AFTER _collate_targets]")
        self._dbg_tensor("collated targets[masks]", targets["masks"])
        if "semantic_masks" in targets:
            self._dbg_tensor("collated targets[semantic_masks]", targets["semantic_masks"])
        self._dbg_tensor("collated targets[num_boxes]", targets["num_boxes"])

        # ------------------------------------------------------------------
        # ALIGN TARGETS TO PRED BATCH
        # ------------------------------------------------------------------
        targets = _align_semantic_targets_to_pred_batch(outputs, targets)

        print("\n[DEBUG][AFTER _align_semantic_targets_to_pred_batch]")
        self._dbg_tensor("aligned targets[masks]", targets["masks"])
        if "semantic_masks" in targets:
            self._dbg_tensor("aligned targets[semantic_masks]", targets["semantic_masks"])
        self._dbg_tensor("aligned targets[num_boxes]", targets["num_boxes"])

        # ------------------------------------------------------------------
        # PREDICTION
        # ------------------------------------------------------------------
        pred = outputs["semantic_seg"].detach()
        prob = pred.sigmoid()
        pred_bin = (prob > 0.5)

        print("\n[DEBUG][PRED]")
        self._dbg_tensor("pred semantic_seg", pred)
        self._dbg_tensor("pred prob", prob)
        self._dbg_tensor("pred_bin@0.5", pred_bin)

        # ------------------------------------------------------------------
        # UPSAMPLE PRED TO TARGET SIZE FOR FAIR FG-RATIO COMPARISON
        # ------------------------------------------------------------------
        tgt = targets["masks"].detach()
        pred_up = torch.nn.functional.interpolate(
            pred.float(),
            size=tgt.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        prob_up = pred_up.sigmoid()
        pred_up_bin = (prob_up > 0.5)

        print("\n[DEBUG][PRED UPSAMPLED TO TARGET SIZE]")
        self._dbg_tensor("pred_up", pred_up)
        self._dbg_tensor("pred_up prob", prob_up)
        self._dbg_tensor("pred_up_bin@0.5", pred_up_bin)

        print(
            "[DEBUG] compare fg ratio | "
            f"target={tgt.float().mean().item():.6f}, "
            f"pred@288={pred_bin.float().mean().item():.6f}, "
            f"pred@target_size={pred_up_bin.float().mean().item():.6f}"
        )

        pred_up = torch.nn.functional.interpolate(
            outputs["semantic_seg"].float(),
            size=targets["masks"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        pred_bin = (pred_up.sigmoid().squeeze(1) > 0.5)
        tgt_bin = targets["masks"].bool()

        intersection = (pred_bin & tgt_bin).sum(dim=(1, 2))
        union = (pred_bin | tgt_bin).sum(dim=(1, 2))
        iou_each = intersection.float() / (union.float() + 1e-8)

        print("[DEBUG] intersection mean =", intersection.float().mean().item())
        print("[DEBUG] union mean =", union.float().mean().item())
        print("[DEBUG] iou_each mean =", iou_each.mean().item())
        print("[DEBUG] iou_each min/max =", iou_each.min().item(), iou_each.max().item())
        
        # ------------------------------------------------------------------
        # LOSS
        # ------------------------------------------------------------------
        loss_dict = self.loss_fn_semantic_seg(outputs, targets)

        print("\n[DEBUG][LOSS DICT]")
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                print(f"[DEBUG] {k} = {v.detach().float().mean().item():.6f}")
            else:
                print(f"[DEBUG] {k} = {v}")

        cleaned = {}
        for k, v in loss_dict.items():
            cleaned[k] = v if torch.is_tensor(v) else torch.as_tensor(v, device=device)

        if len(cleaned) == 0:
            cleaned[f"{self.log_prefix}_zero"] = outputs["semantic_seg"].sum() * 0.0

        return cleaned