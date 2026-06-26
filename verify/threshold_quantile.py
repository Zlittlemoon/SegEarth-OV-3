"""
Quantile-based adaptive per-class threshold calibration (Step A — generalised).

Instead of hand-tuning absolute thresholds per class, we compute:

    thd_c = quantile_q( {max_score_i : raw_top1_i == c} )

for every foreground class c. A single hyperparameter q ∈ [0, 1) controls the
fraction of the winner-score distribution that is kept as foreground.

Intuition: SAM3 fusion scores have different dynamic ranges per class (pavement
scores ~0.1, water scores ~0.05 on OEM/LoveDA respectively). A global absolute
threshold is too strict for low-scoring classes and too lax for high-scoring
ones. Quantile calibration normalises for these per-class scale differences
using a single, cross-dataset hyperparameter — without any labels.

Key: the quantile threshold is ALWAYS used as-is (no clipping to global_thd):
  - low-score classes (pavement):  qt < global_thd → gate LOWERED → rescues foreground pixels
  - high-score classes (tree/road): qt > global_thd → gate RAISED  → reduces false positives
  - q=0.00 → thd_c = min(scores) ≈ 0 → no suppression at all ("prob_thd=0" baseline)
  - q=0.05 → class-adaptive gates at 5th percentile of each class's winner scores

Decision rule (identical to predict() and threshold_scan.py):
    pred = raw_top1
    thd  = thd_vec[raw_top1]      # per-class quantile threshold
    if max_score < thd: pred = bg_idx

Usage:
    # scan a range of q values on OEM (80-image quick scan)
    python verify/threshold_quantile.py configs/cfg_openearthmap.py --limit 80

    # scan on LoveDA
    python verify/threshold_quantile.py configs/cfg_loveda.py --limit 80

    # full-val evaluation at a specific q
    python verify/threshold_quantile.py configs/cfg_openearthmap.py --q 0.05

    # compare several q values explicitly
    python verify/threshold_quantile.py configs/cfg_loveda.py --q 0.02 0.05 0.10 0.15

The script:
1. Runs ONE inference pass and caches (raw_top1, max_score, gt).
2. For each q in --q (or a default grid), computes per-class quantile thresholds
   from the SAME inference cache (unsupervised — no labels used).
3. Reports: mIoU, Δ vs global-thd baseline, pixel-flow, and a full per-class
   IoU table so you can compare with hand-tuned numbers.
4. Prints the inferred per-class thresholds so they can be pasted into the
   model config if desired.
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
    ap = argparse.ArgumentParser(
        description="Quantile-based adaptive per-class threshold scan")
    ap.add_argument("config")
    ap.add_argument("--limit", type=int, default=None,
                    help="max images to process (default: all)")
    ap.add_argument("--q", type=float, nargs="+", default=None,
                    help="quantile(s) to evaluate. if omitted, scans a default "
                         "grid [0.00, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]")
    ap.add_argument("--min-pixels", type=int, default=100,
                    help="classes with fewer winner pixels than this are skipped "
                         "(threshold kept at global_thd). default: 100")
    ap.add_argument("--cfg-options", nargs="+", action=DictAction, default=None)
    return ap.parse_args()


def iou_from_cm(cm):
    tp = np.diag(cm).astype(np.float64)
    union = cm.sum(1) + cm.sum(0) - tp
    iou = np.full(len(tp), np.nan)
    ok = union > 0
    iou[ok] = tp[ok] / union[ok]
    return iou


def cm_from_pred(g, p, C):
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

    global_thd = float(cfg.model.get("prob_thd", 0.0))
    bg_idx     = int(cfg.model.get("bg_idx", 0))

    q_vals = args.q if args.q else [0.00, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    q_vals = sorted(set(q_vals))

    print(f"Config       : {args.config}")
    print(f"Classes ({C}) : {class_names}")
    print(f"global prob_thd={global_thd}  bg_idx={bg_idx}")
    print(f"q values     : {q_vals}")
    print(f"limit        : {args.limit or 'all'} images")
    print(f"min-pixels   : {args.min_pixels}\n")

    model = MODELS.build(cfg.model)
    model.processor.model.eval()
    loader = Runner.build_dataloader(cfg.test_dataloader)

    # ---- single inference pass ----
    top1_chunks, max_chunks, gt_chunks = [], [], []
    n_done = 0
    for batch in loader:
        out = model.predict(batch["inputs"], batch["data_samples"])
        for ds in out:
            logits = ds.seg_logits.data.float().cpu()
            gt     = ds.gt_sem_seg.data.squeeze(0).long().cpu()
            valid  = (gt >= 0) & (gt < C)
            flat   = logits.reshape(C, -1)
            vmask  = valid.reshape(-1)
            mx, top1 = flat.max(0)
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
    N = len(gt_all)
    print(f"Valid pixels : {N:,}\n")

    # ---- winner-score statistics per class (unsupervised) ----
    print("Winner-score statistics per class (unsupervised):")
    print(f"  {'class':<14}{'n_winner':>10}{'p1':>8}{'p5':>8}{'p10':>8}"
          f"{'p25':>8}{'median':>8}{'mean':>8}")
    winner_scores = {}   # class_idx -> float32 array of max_sc where raw_top1==c
    for c in range(C):
        if c == bg_idx:
            continue
        mask = raw_top1 == c
        sc_c = max_sc[mask]
        winner_scores[c] = sc_c
        n = len(sc_c)
        if n == 0:
            print(f"  {class_names[c]:<14}{'0':>10}")
            continue
        ps = np.percentile(sc_c, [1, 5, 10, 25, 50]) if n >= 5 else [np.nan]*5
        print(f"  {class_names[c]:<14}{n:10,}{ps[0]:8.4f}{ps[1]:8.4f}{ps[2]:8.4f}"
              f"{ps[3]:8.4f}{ps[4]:8.4f}{sc_c.mean():8.4f}")
    print()

    def predict_with(thd_vec):
        pred = raw_top1.copy()
        pred[max_sc < thd_vec[raw_top1]] = bg_idx
        return pred

    # ---- baseline (global prob_thd) ----
    base_thd  = np.full(C, global_thd, dtype=np.float32)
    base_pred = predict_with(base_thd)
    base_iou  = iou_from_cm(cm_from_pred(gt_all, base_pred, C))
    base_miou = np.nanmean(base_iou) * 100
    print(f"Baseline mIoU (global prob_thd={global_thd}): {base_miou:.2f}%")
    print("=" * 80)

    best_q    = None
    best_miou = -1.0
    best_thd  = None

    for q in q_vals:
        # compute per-class quantile threshold (unsupervised)
        thd_vec = base_thd.copy()
        applied = {}
        for c in range(C):
            if c == bg_idx:
                continue
            sc_c = winner_scores.get(c, np.array([]))
            if len(sc_c) < args.min_pixels:
                continue
            qt = float(np.quantile(sc_c, q))
            # Always use the quantile as the per-class threshold.
            # If qt < global_thd  → gate LOWERED → rescues low-score classes
            #                        (e.g. pavement p5=0.077 < global_thd=0.10)
            # If qt > global_thd  → gate RAISED  → suppresses over-confident classes
            # If qt == global_thd → no change
            direction = "lower" if qt < global_thd else ("raise" if qt > global_thd else "same")
            thd_vec[c] = qt
            applied[c] = ("q", qt, direction)

        pred = predict_with(thd_vec)
        iou  = iou_from_cm(cm_from_pred(gt_all, pred, C))
        miou = np.nanmean(iou) * 100

        # pixel-flow
        changed = pred != base_pred
        rescue  = (base_pred != gt_all) & (pred == gt_all)
        damage  = (base_pred == gt_all) & (pred != gt_all)
        from_bg = (base_pred == bg_idx) & (pred != bg_idx)
        to_bg   = (base_pred != bg_idx) & (pred == bg_idx)
        net     = int(rescue.sum()) - int(damage.sum())

        delta = miou - base_miou
        marker = " ◄ best" if miou > best_miou else ""
        if miou > best_miou:
            best_miou = miou
            best_q    = q
            best_thd  = thd_vec.copy()

        print(f"\nq={q:.2f}  mIoU={miou:.2f}%  Δ={delta:+.2f}{marker}")
        print(f"  pixel-flow: changed={changed.sum()/N*100:.3f}%  "
              f"rescue={rescue.sum()/N*100:.3f}%  damage={damage.sum()/N*100:.3f}%  "
              f"net={net/N*100:+.3f}%")
        print(f"             from_bg={from_bg.sum()/N*100:.3f}%  "
              f"to_bg={to_bg.sum()/N*100:.3f}%")

        # inferred per-class thresholds
        print(f"  inferred thresholds (↓=lower than global, ↑=raise, →=same):")
        for c in range(C):
            if c == bg_idx:
                continue
            entry = applied.get(c, ("skip", None, None))
            src = entry[0]
            if src == "skip":
                tag = f"(skipped, <{args.min_pixels} winners)"
            else:
                _, val, direction = entry
                arrow = "↓" if direction == "lower" else ("↑" if direction == "raise" else "→")
                tag = f"{val:.4f}  {arrow}  (global={global_thd:.4f})"
            print(f"    {class_names[c]:<14}: {tag}")

        # per-class IoU delta
        print(f"  {'class':<14}{'baseline':>10}{'q-thd':>10}{'Δ':>8}")
        order = np.argsort(np.nan_to_num(base_iou, nan=-1))
        for k in order:
            b = base_iou[k] * 100
            v = iou[k] * 100
            d = v - b
            entry = applied.get(k, ("skip",))
            direction = entry[2] if len(entry) > 2 else None
            mark = f" {('↓' if direction=='lower' else '↑') if direction else ''}"
            print(f"  {class_names[k]:<14}{b:10.2f}{v:10.2f}{d:+8.2f}{mark}")
        print(f"  {'mIoU':<14}{base_miou:10.2f}{miou:10.2f}{delta:+8.2f}")
        print("=" * 80)

    # ---- summary ----
    print(f"\n{'='*80}")
    print(f"Best q: {best_q:.2f}  →  mIoU={best_miou:.2f}%  "
          f"Δ={best_miou-base_miou:+.2f}% vs baseline")
    print(f"\nInferred thresholds at q={best_q:.2f}:")
    for c in range(C):
        if c == bg_idx:
            continue
        print(f"  {class_names[c]:<14}: {best_thd[c]:.4f}"
              f"{'  (= global)' if best_thd[c] == global_thd else ''}")
    print(f"\nTo apply in model config, add to cfg.model:")
    items = ", ".join(
        f"'{class_names[c]}': {best_thd[c]:.4f}"
        for c in range(C) if c != bg_idx and best_thd[c] != global_thd
    )
    print(f"  class_prob_thd = {{{items}}}")
    print("\nNote: * marks classes whose threshold was raised by quantile calibration.")
    print("Classes kept at global_thd had too few winners or quantile ≤ global_thd.")


if __name__ == "__main__":
    main()
