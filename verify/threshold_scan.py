"""
Step A: class-specific prob_thd scan for SegEarth-OV3.

Motivation (from the oracle/backfill diagnostic):
  A large fraction of FOREGROUND top1 pixels are reverted to background by the
  global prob_thd (OEM pavement: 73% of its backfilled pixels are GT==top1;
  LoveDA water: 52%). Those errors are NOT a top1->top2 rerank problem -- they
  are a thresholding problem: the gate is too aggressive for a few classes.

  This script asks: if we LOWER prob_thd for ONE class at a time (keeping the
  global prob_thd for all other classes), does global mIoU go up, and at what
  cost to neighbouring classes?

Key property that makes this cheap:
  class-specific threshold NEVER changes the argmax. The predicted class of a
  pixel is either its raw top1 (if it passes the gate) or bg (if it doesn't).
  So a pixel's final label depends ONLY on (raw_top1, max_score). We run the
  model ONCE, cache (raw_top1, max_score, gt) for every valid pixel, then sweep
  all thresholds in pure numpy -- no re-inference.

Decision rule (mirrors predict() exactly):
    pred = raw_top1
    thd  = class_thd[raw_top1]   # scanned class uses its own thd, rest = global
    if max_score < thd:  pred = bg_idx

Per-threshold output:
  * mIoU + Δ from baseline
  * IoU[target] + Δ
  * released pixels: (raw_top1==c) & (max_sc >= thd) & (max_sc < global_thd)
    broken into GT==target / GT==bg / GT==other  (the false-positive decomposition)
  * worst neighbour drop (the cost that mIoU averages away)

After all grid values for one class, the best-by-mIoU threshold gets a full
per-class IoU table so you can see the complete trade-off profile.

Pareto criterion for picking the threshold:
  IoU[target] up AND worst-other-Δ ≈ 0.
  Stop lowering once a neighbour class starts dropping meaningfully.

After fixing thresholds, RE-RUN oracle_ceiling.py: gated pixels that come back
as foreground enter the strict low-margin candidate pool and shift the CLIP space.

Run:
    python verify/threshold_scan.py configs/cfg_openearthmap.py \
        --limit 80 --classes pavement --grid 0.0 0.02 0.04 0.06 0.08 0.10

    python verify/threshold_scan.py configs/cfg_openearthmap.py \
        --limit 80 --classes grass building tree --grid 0.0 0.02 0.05 0.08 0.10

    python verify/threshold_scan.py configs/cfg_loveda.py \
        --limit 80 --classes water --grid 0.10 0.20 0.30 0.40 0.50

    python verify/threshold_scan.py configs/cfg_loveda.py \
        --limit 80 --classes agricultural forest --grid 0.10 0.20 0.30 0.40 0.50
"""

import argparse

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmseg.registry import MODELS

import mmseg.datasets
import segearthov3_segmentor_merge
import custom_datasets


def parse_args():
    ap = argparse.ArgumentParser(description="Class-specific prob_thd scan")
    ap.add_argument("config")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--classes", nargs="+", default=None,
                    help="class name(s) to scan independently. Each is scanned "
                         "alone with all other classes held at the global prob_thd. "
                         "Default: every foreground class (background excluded).")
    ap.add_argument("--grid", nargs="+", type=float,
                    default=[0.0, 0.02, 0.05, 0.1, 0.2],
                    help="threshold values to try for the scanned class.")
    ap.add_argument("--cfg-options", nargs="+", action=DictAction, default=None)
    return ap.parse_args()


def iou_from_cm(cm):
    tp = np.diag(cm).astype(np.float64)
    gt_sum = cm.sum(1).astype(np.float64)
    pd_sum = cm.sum(0).astype(np.float64)
    union = gt_sum + pd_sum - tp
    iou = np.full(len(tp), np.nan)
    ok = union > 0
    iou[ok] = tp[ok] / union[ok]
    return iou


def cm_from_pred(g, p, C):
    """Fast confusion matrix via bincount (avoids slow np.add.at)."""
    return np.bincount(g * C + p, minlength=C * C).reshape(C, C)


@torch.no_grad()
def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    init_default_scope(cfg.get("default_scope", "mmseg"))

    with open(cfg.model["classname_path"]) as f:
        class_names = [ln.split(",")[0].strip() for ln in f if ln.strip()]
    C = len(class_names)
    name2idx = {n: i for i, n in enumerate(class_names)}

    global_thd = float(cfg.model.get("prob_thd", 0.0))
    bg_idx     = int(cfg.model.get("bg_idx", 0))

    if args.classes is None:
        scan_classes = [i for i in range(C) if i != bg_idx]
    else:
        scan_classes = []
        for nm in args.classes:
            if nm not in name2idx:
                raise SystemExit(f"unknown class '{nm}'. choices: {class_names}")
            idx = name2idx[nm]
            if idx == bg_idx:
                raise SystemExit(
                    f"'{nm}' is the background class (bg_idx={bg_idx}). "
                    "Scan foreground classes only.")
            scan_classes.append(idx)

    print(f"Config       : {args.config}")
    print(f"Classes ({C}) : {class_names}")
    print(f"global prob_thd={global_thd}  bg_idx={bg_idx}")
    print(f"scan classes : {[class_names[c] for c in scan_classes]}")
    print(f"grid         : {args.grid}")
    print(f"limit        : {args.limit or 'all'} images")
    if any(t > global_thd for t in args.grid):
        print(f"[WARN] grid values > global_thd={global_thd} are STRICTER thresholds "
              f"(raise the gate, released=0), not lowering scans. Keep grid <= global_thd "
              f"for a clean backfill-recovery scan.")
    print()

    model = MODELS.build(cfg.model)
    model.processor.model.eval()
    loader = Runner.build_dataloader(cfg.test_dataloader)

    # ---- single inference pass: cache (raw_top1, max_score, gt) per valid pixel ----
    top1_chunks, max_chunks, gt_chunks = [], [], []
    n_done = 0
    for batch in loader:
        out = model.predict(batch["inputs"], batch["data_samples"])
        for ds in out:
            logits = ds.seg_logits.data.float().cpu()             # (C, H, W)
            gt     = ds.gt_sem_seg.data.squeeze(0).long().cpu()   # (H, W)
            valid  = (gt >= 0) & (gt < C)
            flat   = logits.reshape(C, -1)
            vmask  = valid.reshape(-1)

            mx, top1 = flat.max(0)                                 # (P,), (P,)
            top1_chunks.append(top1[vmask].numpy().astype(np.int32))
            max_chunks.append(mx[vmask].numpy().astype(np.float32))
            gt_chunks.append(gt.reshape(-1)[vmask].numpy().astype(np.int32))

        n_done += len(out)
        if n_done % 50 == 0:
            print(f"  processed {n_done} images ...")
        if args.limit and n_done >= args.limit:
            break

    print(f"\nTotal images : {n_done}")

    raw_top1 = np.concatenate(top1_chunks).astype(np.int64)
    max_sc   = np.concatenate(max_chunks).astype(np.float32)
    gt_all   = np.concatenate(gt_chunks).astype(np.int64)
    del top1_chunks, max_chunks, gt_chunks
    print(f"Valid pixels : {len(gt_all):,}\n")

    def predict_with(thd_vec):
        """thd_vec[c] = prob threshold when raw_top1==c. Returns pred array."""
        pred = raw_top1.copy()
        gate = thd_vec[raw_top1]
        pred[max_sc < gate] = bg_idx
        return pred

    # ---- baseline: global prob_thd for every class ----
    base_thd  = np.full(C, global_thd, dtype=np.float32)
    base_pred = predict_with(base_thd)
    base_cm   = cm_from_pred(gt_all, base_pred, C)
    base_iou  = iou_from_cm(base_cm)
    base_miou = np.nanmean(base_iou) * 100

    print(f"Baseline mIoU (global prob_thd={global_thd}): {base_miou:.2f}%\n")
    print("=" * 90)

    # ---- scan each class independently ----
    for c in scan_classes:
        cname     = class_names[c]
        is_c_top1 = raw_top1 == c
        gated     = is_c_top1 & (max_sc < global_thd)   # currently suppressed to bg

        print(f"\n### scan class '{cname}' (idx {c})")
        print(f"  raw top1=={cname}: {is_c_top1.sum():,} px   "
              f"gated->bg at global thd={global_thd}: {gated.sum():,}   "
              f"of which GT=={cname}: {(gated & (gt_all == c)).sum():,}")

        hdr = (f"\n  {'thd':>6}{'mIoU':>9}{'Δm':>7}"
               f"{'IoU[tgt]':>10}{'Δcls':>7}"
               f"{'released':>10}{'rel_GT%':>8}{'rel_bg%':>8}{'rel_oth%':>9}"
               f"{'worst-Δ(cls)':>18}")
        print(hdr)

        records = []   # (thd, miou, iou_array, rel_n, rel_gt_c, rel_gt_bg)
        for thd in args.grid:
            thd_vec = base_thd.copy()
            thd_vec[c] = float(thd)
            pred = predict_with(thd_vec)
            cm   = cm_from_pred(gt_all, pred, C)
            iou  = iou_from_cm(cm)
            miou = np.nanmean(iou) * 100

            dm   = miou - base_miou
            dcls = (iou[c] - base_iou[c]) * 100

            # pixels released by lowering this class's threshold
            if thd < global_thd:
                released = is_c_top1 & (max_sc >= thd) & (max_sc < global_thd)
            else:
                released = np.zeros(len(raw_top1), dtype=bool)

            rel_n       = int(released.sum())
            rel_gt_c    = int((released & (gt_all == c)).sum())
            rel_gt_bg   = int((released & (gt_all == bg_idx)).sum())
            rel_other   = rel_n - rel_gt_c - rel_gt_bg
            rel_gt_pct  = rel_gt_c  / max(rel_n, 1) * 100
            rel_bg_pct  = rel_gt_bg / max(rel_n, 1) * 100
            rel_oth_pct = rel_other / max(rel_n, 1) * 100

            # worst delta among all classes EXCEPT the scanned one
            delta_all = (iou - base_iou) * 100
            delta_all[c] = np.nan
            worst_idx = int(np.nanargmin(delta_all))
            worst_d   = delta_all[worst_idx]
            worst_nm  = class_names[worst_idx]

            mark = "  <-- Δm>0" if dm > 0 else ""
            print(f"  {thd:6.3f}{miou:9.2f}{dm:+7.2f}"
                  f"{iou[c]*100:10.2f}{dcls:+7.2f}"
                  f"{rel_n:10,}{rel_gt_pct:7.1f}%{rel_bg_pct:7.1f}%{rel_oth_pct:8.1f}%"
                  f"  {worst_nm}:{worst_d:+.2f}{mark}")

            records.append((thd, miou, iou, rel_n, rel_gt_c, rel_gt_bg))

        # ---- per-class IoU table at the best-by-mIoU threshold ----
        best = max(records, key=lambda r: r[1])
        best_thd, best_miou, best_iou = best[0], best[1], best[2]

        print(f"\n  Best by mIoU: thd={best_thd:.3f}  mIoU={best_miou:.2f}  "
              f"(Δm={best_miou - base_miou:+.2f})")
        print(f"  Full per-class IoU at best threshold:")
        print(f"    {'class':<14}{'baseline':>10}{'scan':>10}{'Δ':>8}")
        order = np.argsort(np.nan_to_num(base_iou, nan=-1))
        for k in order:
            b = base_iou[k] * 100
            v = best_iou[k] * 100
            d = v - b
            marker = " **" if k == c else ""
            print(f"    {class_names[k]:<14}{b:10.2f}{v:10.2f}{d:+8.2f}{marker}")
        print(f"    {'mIoU':<14}{base_miou:10.2f}{best_miou:10.2f}"
              f"{best_miou-base_miou:+8.2f}")

        print(f"\n  Column guide:")
        print(f"    released    : pixels (raw_top1=={cname}) newly passed gate at this thd")
        print(f"    rel_GT%     : % of released where GT=={cname} (the recoverable ones)")
        print(f"    rel_bg%     : % of released where GT==bg (false positives vs background)")
        print(f"    rel_oth%    : % of released where GT==another foreground class")
        print(f"                  (the main source of neighbour-class damage)")
        print(f"    worst-Δ(cls): biggest IoU drop among all OTHER classes")
        print(f"  Pareto target: rel_GT% high + rel_bg%/rel_oth% low + worst-Δ ~0")
        print("=" * 90)

    print("\nInterpretation:")
    print("  * Pick the Pareto knee: IoU[target] up, worst-other-Δ ≈ 0.")
    print("  * rel_GT% high + rel_bg% low => released pixels are mostly true positive.")
    print("  * If worst-other-Δ is bad from the start, lowering this class threshold")
    print("    introduces too many false positives in neighbours -- skip or use mild thd.")
    print("  * After fixing thresholds, RE-RUN oracle_ceiling.py: gated pixels that")
    print("    return as foreground enter the strict low-margin pool, changing CLIP space.")

if __name__ == "__main__":
    main()
