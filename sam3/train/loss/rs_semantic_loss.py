
from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam3.model.model_misc import SAM3Output


def _align_semantic_targets_to_pred_batch(
    outputs: Dict[str, Any], targets: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    if "semantic_seg" not in outputs:
        return targets

    pred = outputs["semantic_seg"]
    pred_bs = int(pred.shape[0])

    if "masks" not in targets:
        return targets

    masks = targets["masks"]

    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    if masks.ndim == 3 and masks.shape[0] == 1 and pred_bs > 1:
        masks = masks.repeat(pred_bs, 1, 1)

    if masks.ndim == 3 and masks.shape[0] != pred_bs:
        if pred_bs == 1:
            masks = masks.bool().any(dim=0, keepdim=True).to(masks.dtype)
        else:
            raise RuntimeError(
                f"[RSSemanticOnlyLoss] semantic target batch mismatch: "
                f"pred batch={pred_bs}, target masks shape={tuple(masks.shape)}"
            )

    targets["masks"] = masks

    if "semantic_masks" in targets and torch.is_tensor(targets["semantic_masks"]):
        sm = targets["semantic_masks"]
        if sm.ndim == 4 and sm.shape[1] == 1:
            sm = sm[:, 0]
        if sm.ndim == 3 and sm.shape[0] == 1 and pred_bs > 1:
            sm = sm.repeat(pred_bs, 1, 1)
        if sm.ndim == 3 and sm.shape[0] != pred_bs:
            if pred_bs == 1:
                sm = sm.bool().any(dim=0, keepdim=True).to(sm.dtype)
            else:
                raise RuntimeError(
                    f"[RSSemanticOnlyLoss] semantic_masks batch mismatch: "
                    f"pred batch={pred_bs}, semantic_masks shape={tuple(sm.shape)}"
                )
        targets["semantic_masks"] = sm

    if "num_boxes" in targets and torch.is_tensor(targets["num_boxes"]):
        if targets["num_boxes"].numel() != pred_bs:
            targets["num_boxes"] = torch.ones(
                pred_bs, dtype=torch.long, device=masks.device
            )

    return targets


def _safe_semantic_masks(targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "semantic_masks" in targets and targets["semantic_masks"] is not None:
        semantic_targets = targets["semantic_masks"]
    else:
        semantic_targets = targets["masks"]

    if semantic_targets.ndim == 4 and semantic_targets.shape[1] == 1:
        semantic_targets = semantic_targets[:, 0]

    if semantic_targets.ndim != 3:
        raise RuntimeError(
            f"[RSSemanticOnlyLoss] expected semantic targets [B,H,W], got {tuple(semantic_targets.shape)}"
        )

    return semantic_targets.bool()


def _compute_bin_iou(pred_bin: torch.Tensor, target_bin: torch.Tensor) -> torch.Tensor:
    pred_bin = pred_bin.bool()
    target_bin = target_bin.bool()

    inter = (pred_bin & target_bin).flatten(1).sum(dim=1).float()
    union = (pred_bin | target_bin).flatten(1).sum(dim=1).float()
    iou = inter / torch.clamp(union, min=1.0)
    return iou.mean()


def _compute_bin_macc(pred_bin: torch.Tensor, target_bin: torch.Tensor) -> torch.Tensor:
    """Binary mean class accuracy (mAcc) over foreground/background."""
    pred_bin = pred_bin.bool()
    target_bin = target_bin.bool()

    pred_flat = pred_bin.flatten(1)
    target_flat = target_bin.flatten(1)

    # Foreground class accuracy: TP / GT_fg
    tp_fg = (pred_flat & target_flat).sum(dim=1).float()
    gt_fg = target_flat.sum(dim=1).float()
    acc_fg = tp_fg / torch.clamp(gt_fg, min=1.0)
    valid_fg = gt_fg > 0

    # Background class accuracy: TN / GT_bg
    tp_bg = ((~pred_flat) & (~target_flat)).sum(dim=1).float()
    gt_bg = (~target_flat).sum(dim=1).float()
    acc_bg = tp_bg / torch.clamp(gt_bg, min=1.0)
    valid_bg = gt_bg > 0

    valid_count = valid_fg.float() + valid_bg.float()
    per_sample_macc = (acc_fg * valid_fg.float() + acc_bg * valid_bg.float()) / torch.clamp(
        valid_count, min=1.0
    )
    return per_sample_macc.mean()


def _fg_ratio(x: torch.Tensor) -> float:
    return x.float().mean().item()


def _fg_ratio_per_sample(x: torch.Tensor):
    return x.float().flatten(1).mean(1)[:8].tolist()


def _threshold_tag(th: float) -> str:
    # 0.3 -> t03, 0.4 -> t04, 0.5 -> t05
    return f"t{int(round(float(th) * 10)):02d}"


def _compute_prob_distribution(prob: torch.Tensor) -> Dict[str, torch.Tensor]:
    flat = prob.float().reshape(-1)
    # torch.quantile on full 1008 maps can be expensive; sample deterministically when huge.
    if flat.numel() > 250000:
        stride = max(1, flat.numel() // 250000)
        flat = flat[::stride]

    return {
        "mean": prob.float().mean(),
        "std": prob.float().std(unbiased=False),
        "p10": torch.quantile(flat, 0.10),
        "p50": torch.quantile(flat, 0.50),
        "p90": torch.quantile(flat, 0.90),
    }


def _aggregate_instance_masks_max_logit(pred_masks: torch.Tensor) -> torch.Tensor:
    """
    Aggregate instance logits into semantic logits using max over query dimension.

    Expected common input shape is [B, Q, H, W]. The output is [B, 1, H, W].
    """
    if not torch.is_tensor(pred_masks):
        raise TypeError(f"pred_masks should be a tensor, got {type(pred_masks)}")

    if pred_masks.ndim == 4:
        # [B, Q, H, W] -> [B, 1, H, W]
        return pred_masks.max(dim=1, keepdim=True).values

    if pred_masks.ndim == 3:
        # Treat as already semantic-like logits [B, H, W].
        return pred_masks.unsqueeze(1)

    if pred_masks.ndim == 2:
        return pred_masks.unsqueeze(0).unsqueeze(0)

    raise ValueError(
        f"Unsupported pred_masks shape for max-logit aggregation: {tuple(pred_masks.shape)}"
    )


class RSSemanticOnlyLoss(nn.Module):
    def __init__(
        self,
        loss_fn_semantic_seg: nn.Module,
        log_prefix: str = "sem",
        calibration_thresholds: Sequence[float] = (0.3, 0.4, 0.5),
        use_instance_max_logit_semantic: bool = False,
    ) -> None:
        super().__init__()
        self.loss_fn_semantic_seg = loss_fn_semantic_seg
        self.log_prefix = log_prefix
        self._debug_print_count = 0
        self._debug_print_limit = 5
        self._debug_context = None
        self.calibration_thresholds = tuple(float(x) for x in calibration_thresholds)
        self.use_instance_max_logit_semantic = bool(use_instance_max_logit_semantic)

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
            out: Dict[str, torch.Tensor] = {}

            masks = targets["masks"]
            if not torch.is_tensor(masks):
                masks = torch.as_tensor(masks, device=device)
            else:
                masks = masks.to(device)

            if masks.numel() == 0:
                # For all-negative batches, packed instance masks can be empty while
                # semantic masks still have one map per query. Prefer semantic batch size.
                if "semantic_masks" in targets and targets["semantic_masks"] is not None:
                    semantic_masks = targets["semantic_masks"]
                    if not torch.is_tensor(semantic_masks):
                        semantic_masks = torch.as_tensor(semantic_masks, device=device)
                    else:
                        semantic_masks = semantic_masks.to(device)
                    if semantic_masks.ndim == 2:
                        semantic_masks = semantic_masks.unsqueeze(0)
                    elif semantic_masks.ndim == 4 and semantic_masks.shape[1] == 1:
                        semantic_masks = semantic_masks[:, 0]
                    if semantic_masks.ndim != 3:
                        raise ValueError(
                            f"Unsupported semantic_masks shape with empty masks: {tuple(semantic_masks.shape)}"
                        )
                    semantic_masks = semantic_masks.bool()
                    out["masks"] = semantic_masks.clone()
                    out["semantic_masks"] = semantic_masks
                    out["num_boxes"] = semantic_masks.flatten(1).any(-1).long()
                else:
                    masks = torch.zeros((1, H, W), dtype=torch.bool, device=device)
                    out["masks"] = masks
                    out["num_boxes"] = torch.zeros((1,), dtype=torch.long, device=device)
                    out["semantic_masks"] = masks.clone()
                return out

            if masks.ndim == 2:
                masks = masks.unsqueeze(0)
            elif masks.ndim == 3:
                pass
            elif masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
            else:
                raise ValueError(f"Unsupported mask shape in targets dict: {tuple(masks.shape)}")

            out["masks"] = masks.bool()

            if "num_boxes" in targets and torch.is_tensor(targets["num_boxes"]):
                num_boxes = targets["num_boxes"].to(device)
                if num_boxes.numel() == masks.shape[0]:
                    out["num_boxes"] = num_boxes.view(-1).long()
                else:
                    out["num_boxes"] = torch.ones((masks.shape[0],), dtype=torch.long, device=device)
            else:
                out["num_boxes"] = torch.ones((masks.shape[0],), dtype=torch.long, device=device)

            if "semantic_masks" in targets and targets["semantic_masks"] is not None:
                semantic_masks = targets["semantic_masks"]
                if not torch.is_tensor(semantic_masks):
                    semantic_masks = torch.as_tensor(semantic_masks, device=device)
                else:
                    semantic_masks = semantic_masks.to(device)

                if semantic_masks.ndim == 2:
                    semantic_masks = semantic_masks.unsqueeze(0)
                elif semantic_masks.ndim == 3:
                    pass
                elif semantic_masks.ndim == 4 and semantic_masks.shape[1] == 1:
                    semantic_masks = semantic_masks[:, 0]
                else:
                    raise ValueError(
                        f"Unsupported semantic_masks shape in targets dict: {tuple(semantic_masks.shape)}"
                    )

                if semantic_masks.shape[0] == 1 and masks.shape[0] > 1:
                    semantic_masks = semantic_masks.repeat(masks.shape[0], 1, 1)
                elif semantic_masks.shape[0] != masks.shape[0]:
                    # In semantic-only training (especially with negative/hard-negative queries),
                    # `masks` can be packed by number of GT objects while `semantic_masks` is
                    # batched by number of queries. When they mismatch, use semantic masks as the
                    # canonical batched supervision to avoid dropping valid negative samples.
                    sem_bs = semantic_masks.shape[0]
                    out["masks"] = semantic_masks.bool().clone()
                    out["num_boxes"] = (
                        semantic_masks.bool().flatten(1).any(-1).long().to(device)
                    )
                    if self._debug_print_count < self._debug_print_limit:
                        print(
                            "[RSSemanticOnlyLoss][WARN] dict-branch batch mismatch; "
                            f"using semantic_masks as canonical targets. "
                            f"masks batch={masks.shape[0]}, semantic_masks batch={sem_bs}"
                        )

                out["semantic_masks"] = semantic_masks.bool()
            else:
                out["semantic_masks"] = out["masks"].clone()

            return out

        if not isinstance(targets, list):
            raise TypeError(f"Unsupported targets type: {type(targets)}")

        if len(targets) == 0:
            raise ValueError("targets list is empty.")

        # IMPORTANT:
        # Trainer can pass `targets` as a singleton list whose first element is already
        # a batched dict (e.g. masks shape [B,H,W]). Detect that case and delegate to
        # the dict path instead of collapsing batch dim with any(dim=0).
        if len(targets) == 1 and isinstance(targets[0], dict):
            t0 = targets[0]
            if "masks" in t0:
                m0 = t0["masks"]
                nb0 = t0.get("num_boxes", None)
                if torch.is_tensor(m0):
                    if (m0.ndim == 3 and torch.is_tensor(nb0) and nb0.numel() > 1) or (
                        m0.ndim == 4 and m0.shape[0] > 1
                    ):
                        return self._collate_targets(t0, device, output_hw)

        batched_masks = []
        batched_num_boxes = []
        batched_semantic_masks = []

        for i, t in enumerate(targets):
            if not isinstance(t, dict):
                raise TypeError(f"targets[{i}] should be dict, got {type(t)}")
            if "masks" not in t:
                raise KeyError(f"targets[{i}] missing 'masks', got keys={list(t.keys())}")

            m = t["masks"]
            if not torch.is_tensor(m):
                m = torch.as_tensor(m)

            if m.numel() == 0:
                m = torch.zeros((H, W), dtype=torch.bool, device=device)
                batched_masks.append(m)
                batched_semantic_masks.append(m.clone())
                batched_num_boxes.append(0)
                continue

            if m.ndim == 2:
                sample_mask = m
            elif m.ndim == 3:
                if m.shape[0] == 1:
                    sample_mask = m[0]
                else:
                    sample_mask = m.bool().any(dim=0)
            elif m.ndim == 4 and m.shape[1] == 1:
                if m.shape[0] == 1:
                    sample_mask = m[0, 0]
                else:
                    sample_mask = m[:, 0].bool().any(dim=0)
            else:
                raise ValueError(f"Unsupported mask shape for targets[{i}]: {tuple(m.shape)}")

            sample_mask = sample_mask.to(device).bool()
            batched_masks.append(sample_mask)

            if "semantic_masks" in t and t["semantic_masks"] is not None:
                sm = t["semantic_masks"]
                if not torch.is_tensor(sm):
                    sm = torch.as_tensor(sm)
                if sm.ndim == 2:
                    sample_sem = sm
                elif sm.ndim == 3:
                    if sm.shape[0] == 1:
                        sample_sem = sm[0]
                    else:
                        sample_sem = sm.bool().any(dim=0)
                elif sm.ndim == 4 and sm.shape[1] == 1:
                    if sm.shape[0] == 1:
                        sample_sem = sm[0, 0]
                    else:
                        sample_sem = sm[:, 0].bool().any(dim=0)
                else:
                    raise ValueError(
                        f"Unsupported semantic_masks shape for targets[{i}]: {tuple(sm.shape)}"
                    )
                batched_semantic_masks.append(sample_sem.to(device).bool())
            else:
                batched_semantic_masks.append(sample_mask.clone())

            num_boxes = t.get("num_boxes", None)
            if torch.is_tensor(num_boxes):
                if num_boxes.numel() == 1:
                    batched_num_boxes.append(int(num_boxes.view(-1)[0].item()))
                else:
                    batched_num_boxes.append(int(num_boxes.view(-1).sum().item()))
            else:
                batched_num_boxes.append(1)

        masks = torch.stack(batched_masks, dim=0)
        semantic_masks = torch.stack(batched_semantic_masks, dim=0)
        num_boxes = torch.tensor(batched_num_boxes, dtype=torch.long, device=device)

        return {
            "masks": masks,
            "semantic_masks": semantic_masks,
            "num_boxes": num_boxes,
        }

    def compute_loss(self, outputs: Any, targets: Any) -> Dict[str, torch.Tensor]:
        outputs = self._unwrap_outputs(outputs)
        outputs_for_loss = dict(outputs)

        if self.use_instance_max_logit_semantic:
            if "pred_masks" not in outputs_for_loss:
                raise KeyError(
                    "use_instance_max_logit_semantic=True requires 'pred_masks' in outputs, "
                    f"got keys={list(outputs_for_loss.keys())}"
                )
            outputs_for_loss["semantic_seg"] = _aggregate_instance_masks_max_logit(
                outputs_for_loss["pred_masks"]
            )

        if "semantic_seg" not in outputs_for_loss:
            raise KeyError(
                f"RSSemanticOnlyLoss expects 'semantic_seg' in outputs, got keys={list(outputs_for_loss.keys())}"
            )

        device = outputs_for_loss["semantic_seg"].device
        pred = outputs_for_loss["semantic_seg"]
        pred_hw = tuple(pred.shape[-2:])

        ctx = getattr(self, "_debug_context", None)
        should_debug = self._debug_print_count < self._debug_print_limit

        targets = self._collate_targets(targets, device, pred_hw)

        if should_debug:
            print("\n[DEBUG][RSSemanticOnlyLoss][AFTER _collate_targets]")
            if ctx is not None:
                print(f"[DEBUG] phase={ctx['phase']} epoch={ctx['epoch']} step={ctx['step']}")
                print("[DEBUG] texts[:8] =", ctx.get("texts"))
                print("[DEBUG] image_ids[:8] =", ctx.get("image_ids"))
                print("[DEBUG] category_ids[:8] =", ctx.get("category_ids"))

            print("[DEBUG] collated targets[masks] shape =", tuple(targets["masks"].shape))
            print("[DEBUG] collated masks batch_fg_ratio =", _fg_ratio(targets["masks"]))
            print("[DEBUG] collated masks per_sample_fg_ratio[:8] =", _fg_ratio_per_sample(targets["masks"]))

            if "semantic_masks" in targets:
                print("[DEBUG] collated semantic_masks shape =", tuple(targets["semantic_masks"].shape))
                print("[DEBUG] collated semantic_masks batch_fg_ratio =", _fg_ratio(targets["semantic_masks"]))
                print("[DEBUG] collated semantic_masks per_sample_fg_ratio[:8] =", _fg_ratio_per_sample(targets["semantic_masks"]))

            if "num_boxes" in targets and torch.is_tensor(targets["num_boxes"]):
                print("[DEBUG] collated num_boxes[:8] =", targets["num_boxes"].detach().cpu().view(-1)[:8].tolist())

        targets = _align_semantic_targets_to_pred_batch(outputs_for_loss, targets)

        if should_debug:
            print("\n[DEBUG][RSSemanticOnlyLoss][AFTER _align_semantic_targets_to_pred_batch]")
            print("[DEBUG] aligned targets[masks] shape =", tuple(targets["masks"].shape))
            print("[DEBUG] aligned masks batch_fg_ratio =", _fg_ratio(targets["masks"]))
            print("[DEBUG] aligned masks per_sample_fg_ratio[:8] =", _fg_ratio_per_sample(targets["masks"]))

            if "semantic_masks" in targets:
                print("[DEBUG] aligned semantic_masks shape =", tuple(targets["semantic_masks"].shape))
                print("[DEBUG] aligned semantic_masks batch_fg_ratio =", _fg_ratio(targets["semantic_masks"]))
                print("[DEBUG] aligned semantic_masks per_sample_fg_ratio[:8] =", _fg_ratio_per_sample(targets["semantic_masks"]))

            if "num_boxes" in targets and torch.is_tensor(targets["num_boxes"]):
                print("[DEBUG] aligned num_boxes[:8] =", targets["num_boxes"].detach().cpu().view(-1)[:8].tolist())

        semantic_targets_1008 = _safe_semantic_masks(targets)

        semantic_targets_288 = (
            F.interpolate(
                semantic_targets_1008.float().unsqueeze(1),
                size=pred_hw,
                mode="nearest",
            )
            .squeeze(1)
            .bool()
        )

        if should_debug:
            print("\n[DEBUG][RSSemanticOnlyLoss][FINAL SEM TARGETS]")
            print("[DEBUG] semantic_targets_1008 shape =", tuple(semantic_targets_1008.shape))
            print("[DEBUG] semantic_targets_1008 batch_fg_ratio =", _fg_ratio(semantic_targets_1008))
            print("[DEBUG] semantic_targets_1008 per_sample_fg_ratio[:8] =", _fg_ratio_per_sample(semantic_targets_1008))
            print("[DEBUG] semantic_targets_288 shape =", tuple(semantic_targets_288.shape))
            print("[DEBUG] semantic_targets_288 batch_fg_ratio =", _fg_ratio(semantic_targets_288))
            print("[DEBUG] semantic_targets_288 per_sample_fg_ratio[:8] =", _fg_ratio_per_sample(semantic_targets_288))

        loss_targets = dict(targets)
        loss_targets["semantic_masks"] = semantic_targets_288
        loss_targets["masks"] = semantic_targets_288

        loss_dict = self.loss_fn_semantic_seg(outputs_for_loss, loss_targets)

        with torch.no_grad():
            pred_prob_288 = pred.float().sigmoid()
            pred_up_1008 = F.interpolate(
                pred.float(),
                size=semantic_targets_1008.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            pred_prob_1008 = pred_up_1008.sigmoid()

            # Keep original 0.5 metrics for backwards compatibility.
            pred_bin_288_t05 = pred_prob_288 > 0.5
            pred_bin_1008_t05 = pred_prob_1008 > 0.5
            miou_288_t05 = _compute_bin_iou(pred_bin_288_t05[:, 0], semantic_targets_288)
            miou_1008_t05 = _compute_bin_iou(pred_bin_1008_t05[:, 0], semantic_targets_1008)
            macc_288_t05 = _compute_bin_macc(pred_bin_288_t05[:, 0], semantic_targets_288)
            macc_1008_t05 = _compute_bin_macc(pred_bin_1008_t05[:, 0], semantic_targets_1008)

            loss_dict["miou_semantic_seg_288"] = miou_288_t05
            loss_dict["miou_semantic_seg_1008"] = miou_1008_t05
            loss_dict["macc_semantic_seg_288"] = macc_288_t05
            loss_dict["macc_semantic_seg_1008"] = macc_1008_t05
            # Explicitly expose t05 as the primary score for monitoring/selection.
            loss_dict["primary_miou_semantic_seg_288"] = miou_288_t05
            loss_dict["primary_miou_semantic_seg_1008"] = miou_1008_t05
            loss_dict["primary_macc_semantic_seg_288"] = macc_288_t05
            loss_dict["primary_macc_semantic_seg_1008"] = macc_1008_t05
            loss_dict["pred_fg_ratio_288"] = pred_bin_288_t05.float().mean()
            loss_dict["pred_fg_ratio_1008"] = pred_bin_1008_t05.float().mean()
            loss_dict["target_fg_ratio_288"] = semantic_targets_288.float().mean()
            loss_dict["target_fg_ratio_1008"] = semantic_targets_1008.float().mean()

            # Probability distribution summary to judge calibration directly.
            prob_stats_288 = _compute_prob_distribution(pred_prob_288)
            prob_stats_1008 = _compute_prob_distribution(pred_prob_1008)
            for name, val in prob_stats_288.items():
                loss_dict[f"pred_prob_{name}_288"] = val
            for name, val in prob_stats_1008.items():
                loss_dict[f"pred_prob_{name}_1008"] = val

            # Multi-threshold calibration view: 0.3 / 0.4 / 0.5 by default.
            threshold_debug_rows = []
            for th in self.calibration_thresholds:
                pred_bin_288 = pred_prob_288 > th
                pred_bin_1008 = pred_prob_1008 > th
                miou_288 = _compute_bin_iou(pred_bin_288[:, 0], semantic_targets_288)
                miou_1008 = _compute_bin_iou(pred_bin_1008[:, 0], semantic_targets_1008)
                macc_288 = _compute_bin_macc(pred_bin_288[:, 0], semantic_targets_288)
                macc_1008 = _compute_bin_macc(pred_bin_1008[:, 0], semantic_targets_1008)
                pred_fg_288 = pred_bin_288.float().mean()
                pred_fg_1008 = pred_bin_1008.float().mean()
                tag = _threshold_tag(th)
                loss_dict[f"miou_semantic_seg_288_{tag}"] = miou_288
                loss_dict[f"miou_semantic_seg_1008_{tag}"] = miou_1008
                loss_dict[f"macc_semantic_seg_288_{tag}"] = macc_288
                loss_dict[f"macc_semantic_seg_1008_{tag}"] = macc_1008
                loss_dict[f"pred_fg_ratio_288_{tag}"] = pred_fg_288
                loss_dict[f"pred_fg_ratio_1008_{tag}"] = pred_fg_1008
                threshold_debug_rows.append(
                    (
                        th,
                        float(pred_fg_288.item()),
                        float(miou_288.item()),
                        float(pred_fg_1008.item()),
                        float(miou_1008.item()),
                    )
                )

            if should_debug:
                print("\n[DEBUG][RSSemanticOnlyLoss][288 DIAG]")
                print(
                    f"[DEBUG] pred_288 shape={tuple(pred.shape)} "
                    f"prob_mean={pred_prob_288.mean().item():.6f} "
                    f"pred_fg@0.5={pred_bin_288_t05.float().mean().item():.6f} "
                    f"target_fg={semantic_targets_288.float().mean().item():.6f} "
                    f"miou_288={miou_288_t05.item():.6f}"
                )
                print("[DEBUG][RSSemanticOnlyLoss][1008 DIAG]")
                print(
                    f"[DEBUG] pred_up_1008 shape={tuple(pred_up_1008.shape)} "
                    f"prob_mean={pred_prob_1008.mean().item():.6f} "
                    f"pred_fg@0.5={pred_bin_1008_t05.float().mean().item():.6f} "
                    f"target_fg={semantic_targets_1008.float().mean().item():.6f} "
                    f"miou_1008={miou_1008_t05.item():.6f}"
                )
                print("[DEBUG][RSSemanticOnlyLoss][PROB DIST]")
                print(
                    f"[DEBUG] 288 mean={prob_stats_288['mean'].item():.6f} std={prob_stats_288['std'].item():.6f} "
                    f"p10={prob_stats_288['p10'].item():.6f} p50={prob_stats_288['p50'].item():.6f} p90={prob_stats_288['p90'].item():.6f}"
                )
                print(
                    f"[DEBUG] 1008 mean={prob_stats_1008['mean'].item():.6f} std={prob_stats_1008['std'].item():.6f} "
                    f"p10={prob_stats_1008['p10'].item():.6f} p50={prob_stats_1008['p50'].item():.6f} p90={prob_stats_1008['p90'].item():.6f}"
                )
                print("[DEBUG][RSSemanticOnlyLoss][THRESH CALIBRATION]")
                for th, fg288, mi288, fg1008, mi1008 in threshold_debug_rows:
                    print(
                        f"[DEBUG] th={th:.1f} | "
                        f"288: pred_fg={fg288:.6f} miou={mi288:.6f} | "
                        f"1008: pred_fg={fg1008:.6f} miou={mi1008:.6f}"
                    )
                self._debug_print_count += 1

        cleaned = {}
        for k, v in loss_dict.items():
            cleaned[k] = v if torch.is_tensor(v) else torch.as_tensor(v, device=device)

        if len(cleaned) == 0:
            cleaned[f"{self.log_prefix}_zero"] = outputs_for_loss["semantic_seg"].sum() * 0.0

        return cleaned
