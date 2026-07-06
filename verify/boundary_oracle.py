"""
Boundary replacement oracle for SegEarth-OV3.

Answers: "If we keep the same class label correction as label-only flip,
but also replace the CC's pixel mask with the GT mask, how much extra mIoU
do we gain over label-only flip?"

The gap between Oracle 2 (boundary oracle) and Oracle 1 (label-only flip)
is the theoretical upper bound for SAM3 geometry refinement.

Three oracles are computed for each pair:

  Oracle 1 - label-only flip:
    For each rescue CC (GT majority == runnerup), flip all CC pixels from
    winner -> runnerup. CC shape is unchanged (SAM3's original mask).

  Oracle 2 - boundary replacement:
    For each rescue CC, replace its pixels with the GT pixels of the runnerup
    class inside the CC's bounding box (padded by PAD pixels). Specifically:
      - GT pixels inside bbox that are runnerup -> predicted as runnerup (gain)
      - Non-GT-runnerup pixels inside bbox that were winner -> revert to winner
        (i.e. we only change the CC footprint to match GT, not the whole bbox)
    Only CC pixels are modified; pixels outside the CC bbox are untouched.

  Oracle 3 - perfect mask union:
    Like Oracle 2, but we also flip any GT-runnerup pixel inside bbox that
    the original prediction left as winner (i.e. we pick up all GT-runnerup
    pixels in the local region, not just the CC).

Gap = Oracle2_mIoU - Oracle1_mIoU  (how much boundary accuracy is worth)
Gap2 = Oracle3_mIoU - Oracle2_mIoU (how much local GT-runnerup coverage is worth)

Usage (OEM):
    python verify/boundary_oracle.py configs/cfg_openearthmap.py \\
        --pairs "grass->cropland" "building->pavement" "bareland->pavement" "bareland->grass" \\
        --margin-thr 0.10 --limit 80

Usage (LoveDA):
    python verify/boundary_oracle.py configs/cfg_loveda.py \\
        --pairs "agricultural->forest" "agricultural->barren" "agricultural->water" \\
        --margin-thr 0.10 --limit 80

Full val:
    python verify/boundary_oracle.py configs/cfg_openearthmap.py \\
        --pairs "grass->cropland" "building->pavement" "bareland->pavement" "bareland->grass" \\
        --margin-thr 0.10
"""

import argparse
from collections import defaultdict

import numpy as np
import torch
from scipy import ndimage as ndi
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmseg.registry import MODELS

import mmseg.datasets
import segearthov3_segmentor_merge
import custom_datasets

MIN_CC_PIXELS = 16   # ignore CCs smaller than this
BBOX_PAD = 4         # padding around CC bbox when looking for GT mask pixels


def parse_args():
    ap = argparse.ArgumentParser(description="Boundary replacement oracle")
    ap.add_argument("config")
    ap.add_argument("--pairs", nargs="+", metavar="winner->runnerup",
                    help="Directed pairs, e.g. 'grass->cropland'.")
    ap.add_argument("--margin-thr", type=float, default=0.10,
                    help="Top1-top2 logit margin threshold defining low-margin pixels.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-cc", type=int, default=MIN_CC_PIXELS,
                    help="Minimum CC size in pixels.")
    ap.add_argument("--bbox-pad", type=int, default=BBOX_PAD,
                    help="Padding around CC bbox when searching for GT runnerup pixels.")
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


def extract_connected_components(binary_mask):
    label_map, n_labels = ndi.label(binary_mask)
    ccs = []
    for lb in range(1, n_labels + 1):
        ys, xs = np.where(label_map == lb)
        ccs.append((ys, xs))
    return ccs


class PairOracle:
    """Accumulates CM deltas for one (winner, runnerup) directed pair."""

    def __init__(self, w_idx, ru_idx):
        self.w_idx = w_idx
        self.ru_idx = ru_idx
        self.n_cc_total = 0
        self.n_cc_rescue = 0   # majority GT == runnerup
        self.n_cc_correct = 0  # majority GT == winner
        self.n_cc_other = 0

        # CM delta accumulators (relative to baseline cm_base)
        # Each is a list of (gt_pixel_array, from_class, to_class) tuples
        # We accumulate flips and apply at the end.
        self.flips_o1 = []   # Oracle 1: label-only flip of rescue CCs
        self.flips_o2 = []   # Oracle 2: boundary replacement of rescue CCs
        self.flips_o3 = []   # Oracle 3: full local GT-runnerup coverage

        # O3 extra pixel source breakdown
        self.o3_extra_total = 0        # total GT-runnerup pixels outside CC in bbox
        self.o3_extra_low_margin = 0   # subset that are low-margin (same pair)
        self.o3_extra_high_conf = 0    # subset that are high-confidence winner

    def add_cc(self, gt_2d, pred_2d, ys_cc, xs_cc, H, W, ru_idx, w_idx, bbox_pad,
               margin_2d=None, margin_thr=0.10):
        """
        Process one CC. Determines majority GT category and accumulates
        the three oracle flip operations.

        gt_2d: (H,W) int32 ground truth
        pred_2d: (H,W) int32 baseline prediction
        ys_cc, xs_cc: pixel coordinates of the CC
        """
        self.n_cc_total += 1

        cc_gt = gt_2d[ys_cc, xs_cc]
        n_rescue_px = (cc_gt == ru_idx).sum()
        n_correct_px = (cc_gt == w_idx).sum()
        n_px = len(ys_cc)

        # Majority category of this CC
        counts = np.array([n_correct_px, n_rescue_px, n_px - n_correct_px - n_rescue_px])
        majority = int(counts.argmax())  # 0=correct, 1=rescue, 2=other

        if majority == 0:
            self.n_cc_correct += 1
        elif majority == 1:
            self.n_cc_rescue += 1
        else:
            self.n_cc_other += 1

        # Only rescue CCs are flipped in the oracles
        if majority != 1:
            return

        # ----------------------------------------------------------------
        # Oracle 1: label-only flip
        # All CC pixels: winner -> runnerup
        # GT determines whether each flip is a gain or loss.
        # We record (gt_values_of_cc_pixels, from=w_idx, to=ru_idx)
        # ----------------------------------------------------------------
        self.flips_o1.append(cc_gt.copy())

        # ----------------------------------------------------------------
        # Oracle 2: boundary replacement
        # Replace the CC footprint with GT-runnerup pixels inside bbox.
        # Concretely:
        #   Step A: revert all CC pixels back to winner (undo SAM3's CC)
        #   Step B: among CC pixels where GT==runnerup, predict runnerup
        # Net effect: only CC pixels where GT==runnerup get the correct label.
        # Pixels outside the CC are untouched.
        # ----------------------------------------------------------------
        # Step A: revert CC pixels (winner -> winner, no change needed since
        #   baseline already predicts winner for these pixels by definition).
        #   For GT-accounting: we need to know what we change.
        #   Oracle 2 prediction for pixel p in CC:
        #     if GT[p] == runnerup -> predict runnerup (correct)
        #     else                 -> keep winner (no change from baseline)
        # So Oracle 2 flips ONLY the GT==runnerup pixels within the CC.
        rescue_px_mask = cc_gt == ru_idx
        if rescue_px_mask.sum() > 0:
            self.flips_o2.append(cc_gt[rescue_px_mask])  # these are all ru_idx values

        # ----------------------------------------------------------------
        # Oracle 3: full local GT-runnerup coverage
        # In addition to Oracle 2, also flip any winner-predicted pixel in the
        # bbox that has GT==runnerup (picking up GT-runnerup pixels that SAM3
        # didn't include in the CC).
        # ----------------------------------------------------------------
        y0 = max(0, int(ys_cc.min()) - bbox_pad)
        y1 = min(H, int(ys_cc.max()) + 1 + bbox_pad)
        x0 = max(0, int(xs_cc.min()) - bbox_pad)
        x1 = min(W, int(xs_cc.max()) + 1 + bbox_pad)

        bbox_gt = gt_2d[y0:y1, x0:x1]
        bbox_pred = pred_2d[y0:y1, x0:x1]

        # CC mask in bbox coords
        cc_in_bbox = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        cc_in_bbox[ys_cc - y0, xs_cc - x0] = True

        # Pixels outside the CC, in bbox, predicted as winner, GT==runnerup
        extra_mask = (~cc_in_bbox) & (bbox_pred == w_idx) & (bbox_gt == ru_idx)
        extra_gt = bbox_gt[extra_mask]  # all == ru_idx

        # O3 source breakdown: classify extra pixels by confidence
        n_extra = int(extra_mask.sum())
        self.o3_extra_total += n_extra
        if n_extra > 0 and margin_2d is not None:
            bbox_margin = margin_2d[y0:y1, x0:x1]
            extra_margins = bbox_margin[extra_mask]
            self.o3_extra_low_margin += int((extra_margins < margin_thr).sum())
            self.o3_extra_high_conf += int((extra_margins >= margin_thr).sum())

        # Oracle 3 = Oracle 2 + extra bbox pixels
        o3_gt = cc_gt[rescue_px_mask]
        if len(extra_gt) > 0:
            o3_gt = np.concatenate([o3_gt, extra_gt])
        self.flips_o3.append(o3_gt)

    def compute_oracles(self, cm_base, C):
        """Compute mIoU for all three oracles using CM delta method."""
        base_miou = float(np.nanmean(iou_from_cm(cm_base)) * 100)

        def _apply_flips(flip_list, from_cls, to_cls):
            """Apply accumulated flips to cm_base and return new mIoU."""
            if not flip_list:
                return base_miou
            all_gt = np.concatenate(flip_list)
            cm = cm_base.astype(np.float64).copy()
            np.add.at(cm, (all_gt, from_cls), -1)
            np.add.at(cm, (all_gt, to_cls), 1)
            return float(np.nanmean(iou_from_cm(cm)) * 100)

        miou_o1 = _apply_flips(self.flips_o1, self.w_idx, self.ru_idx)
        miou_o2 = _apply_flips(self.flips_o2, self.w_idx, self.ru_idx)
        miou_o3 = _apply_flips(self.flips_o3, self.w_idx, self.ru_idx)

        return {
            "base_miou": base_miou,
            "o1_miou": miou_o1,
            "o2_miou": miou_o2,
            "o3_miou": miou_o3,
            "o1_gain": miou_o1 - base_miou,
            "o2_gain": miou_o2 - base_miou,
            "o3_gain": miou_o3 - base_miou,
            "gap_o2_o1": miou_o2 - miou_o1,
            "gap_o3_o2": miou_o3 - miou_o2,
            "n_cc_total": self.n_cc_total,
            "n_cc_rescue": self.n_cc_rescue,
            "n_cc_correct": self.n_cc_correct,
            "n_cc_other": self.n_cc_other,
            "o3_extra_total": self.o3_extra_total,
            "o3_extra_low_margin": self.o3_extra_low_margin,
            "o3_extra_high_conf": self.o3_extra_high_conf,
        }


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

    prob_thd = float(cfg.model.get("prob_thd", 0.0))
    bg_idx = int(cfg.model.get("bg_idx", 0))

    if not args.pairs:
        raise SystemExit("--pairs required, e.g. --pairs 'grass->cropland'")

    requested_pairs = []
    for p in args.pairs:
        if "->" not in p:
            raise SystemExit(f"bad pair '{p}', expected 'winner->runnerup'")
        wn, run = p.split("->", 1)
        wn, run = wn.strip(), run.strip()
        if wn not in name2idx:
            raise SystemExit(f"unknown class '{wn}'. choices: {class_names}")
        if run not in name2idx:
            raise SystemExit(f"unknown class '{run}'. choices: {class_names}")
        requested_pairs.append((name2idx[wn], name2idx[run]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"margin_thr={args.margin_thr}  min_cc={args.min_cc}  bbox_pad={args.bbox_pad}")
    print(f"Limit: {args.limit or 'all'}")

    seg_model = MODELS.build(cfg.model)
    seg_model.processor.model.eval()
    loader = Runner.build_dataloader(cfg.test_dataloader)

    cm_base = np.zeros((C, C), dtype=np.int64)
    pair_oracles = {pair: PairOracle(pair[0], pair[1]) for pair in requested_pairs}

    n_done = 0
    for batch in loader:
        out = seg_model.predict(batch["inputs"], batch["data_samples"])
        for ds in out:
            logits = ds.seg_logits.data.float().cpu()        # (C, H, W)
            gt = ds.gt_sem_seg.data.squeeze(0).long().cpu()  # (H, W)
            valid = (gt >= 0) & (gt < C)
            H, W = gt.shape

            flat = logits.reshape(C, -1)
            vmask = valid.reshape(-1)

            base_p = flat.argmax(0).clone()
            mx = flat.max(0).values
            thd_vec_t = torch.full((C,), prob_thd)
            gate = thd_vec_t[base_p]
            base_p[mx < gate] = bg_idx

            g_v = gt.reshape(-1)[vmask]
            p_v = base_p[vmask]
            np.add.at(cm_base, (g_v.numpy(), p_v.numpy()), 1)

            flat_v = flat[:, vmask]
            top2_v = flat_v.topk(2, dim=0)
            top1_idx = top2_v.indices[0].numpy()
            top2_idx = top2_v.indices[1].numpy()
            margin_v = (top2_v.values[0] - top2_v.values[1]).numpy()

            p_v_np = p_v.numpy()
            g_v_np = g_v.numpy()

            not_backfilled = (p_v_np == top1_idx) & (p_v_np != bg_idx) & (top2_idx != bg_idx)
            lm_mask = (margin_v < args.margin_thr) & not_backfilled

            # Build 2D arrays for CC extraction and GT lookup
            gt_2d = gt.numpy().astype(np.int32)    # (H, W)

            # Build 2D baseline pred map
            pred_2d = np.full((H, W), bg_idx, dtype=np.int32)
            valid_ys, valid_xs = torch.where(valid)
            valid_ys = valid_ys.numpy()
            valid_xs = valid_xs.numpy()
            pred_2d[valid_ys, valid_xs] = p_v_np

            # Build 2D margin map (top1 - top2 logit gap); large value = high confidence
            margin_2d = np.full((H, W), float("inf"), dtype=np.float32)
            margin_2d[valid_ys, valid_xs] = margin_v

            for (w_idx, ru_idx) in requested_pairs:
                pair_lm = lm_mask & (top1_idx == w_idx) & (top2_idx == ru_idx)
                if not pair_lm.any():
                    continue

                local_indices = np.where(pair_lm)[0]
                pair_ys = valid_ys[local_indices]
                pair_xs = valid_xs[local_indices]

                pair_2d = np.zeros((H, W), dtype=bool)
                pair_2d[pair_ys, pair_xs] = True

                ccs = extract_connected_components(pair_2d)

                oracle = pair_oracles[(w_idx, ru_idx)]
                for ys_cc, xs_cc in ccs:
                    if len(ys_cc) < args.min_cc:
                        continue
                    oracle.add_cc(gt_2d, pred_2d, ys_cc, xs_cc,
                                  H, W, ru_idx, w_idx, args.bbox_pad,
                                  margin_2d=margin_2d, margin_thr=args.margin_thr)

        n_done += len(out)
        if n_done % 20 == 0:
            print(f"  processed {n_done} images ...")
        if args.limit and n_done >= args.limit:
            break

    print(f"\nTotal images: {n_done}")
    base_miou = float(np.nanmean(iou_from_cm(cm_base)) * 100)
    print(f"Baseline mIoU: {base_miou:.2f}%\n")

    # =========================================================================
    # Report
    # =========================================================================
    print("=" * 80)
    print("Boundary Oracle Results")
    print("=" * 80)
    print()
    print("  Oracle 1 (label-only flip): flip all CC pixels winner->runnerup")
    print("  Oracle 2 (boundary replace): only flip GT==runnerup pixels within CC")
    print("    Gap O2-O1 = value of having perfect CC boundary (what SAM3 refine can gain)")
    print("  Oracle 3 (local GT coverage): Oracle 2 + GT-runnerup pixels in bbox outside CC")
    print("    Gap O3-O2 = value of picking up missed GT-runnerup pixels near CC")
    print()

    all_results = {}
    for (w_idx, ru_idx) in requested_pairs:
        oracle = pair_oracles[(w_idx, ru_idx)]
        res = oracle.compute_oracles(cm_base, C)
        all_results[(w_idx, ru_idx)] = res

        w_name = class_names[w_idx]
        ru_name = class_names[ru_idx]
        print(f"Pair: {w_name} -> {ru_name}")
        print(f"  CCs: total={res['n_cc_total']}  "
              f"rescue={res['n_cc_rescue']}  "
              f"correct={res['n_cc_correct']}  "
              f"other={res['n_cc_other']}")
        print()
        print(f"  {'Oracle':<30}  {'mIoU':>8}  {'ΔmIoU':>8}")
        print(f"  {'-'*50}")
        print(f"  {'Baseline':<30}  {res['base_miou']:>8.3f}  {'':>8}")
        print(f"  {'O1: label-only flip':<30}  {res['o1_miou']:>8.3f}  "
              f"{res['o1_gain']:>+8.3f}")
        print(f"  {'O2: boundary replace':<30}  {res['o2_miou']:>8.3f}  "
              f"{res['o2_gain']:>+8.3f}")
        print(f"  {'O3: local GT coverage':<30}  {res['o3_miou']:>8.3f}  "
              f"{res['o3_gain']:>+8.3f}")
        print()
        print(f"  Gap O2-O1 (boundary value):      {res['gap_o2_o1']:>+8.3f} mIoU")
        print(f"  Gap O3-O2 (local coverage value): {res['gap_o3_o2']:>+8.3f} mIoU")
        print()

        # O3 extra pixel source breakdown
        n_tot = res['o3_extra_total']
        n_lm  = res['o3_extra_low_margin']
        n_hc  = res['o3_extra_high_conf']
        if n_tot > 0:
            pct_lm = n_lm / n_tot * 100
            pct_hc = n_hc / n_tot * 100
            print(f"  O3 extra pixel source breakdown (GT-runnerup outside CC in bbox):")
            print(f"    total extra pixels : {n_tot}")
            print(f"    low-margin (< thr) : {n_lm}  ({pct_lm:.1f}%)  <- operatable by reranking")
            print(f"    high-conf  (>= thr): {n_hc}  ({pct_hc:.1f}%)  <- NOT operatable, confident winner")
        else:
            print(f"  O3 extra pixel source breakdown: no extra pixels found in bbox")
        print()

        # Interpretation
        gap = res['gap_o2_o1']
        o1 = res['o1_gain']
        if o1 <= 0:
            verdict = "Label-only flip is not beneficial for this pair; boundary irrelevant."
        elif abs(gap) < 0.05:
            verdict = ("Boundary value is negligible (<0.05 mIoU). "
                       "Problem is semantic label, not boundary shape. "
                       "SAM3 geometry refinement NOT recommended for this pair.")
        elif gap < 0:
            verdict = (f"Boundary replacement HURTS ({gap:+.3f}). "
                       "SAM3's CC mask covers rescue pixels better than GT bbox replacement. "
                       "Geometry refinement NOT recommended.")
        elif gap < 0.2:
            verdict = (f"Boundary value is small ({gap:+.3f} mIoU, "
                       f"{gap/o1*100:.0f}% of label-only gain). "
                       "Label correction dominates; SAM3 geometry refinement has marginal value.")
        elif gap < 0.5:
            verdict = (f"Boundary value is moderate ({gap:+.3f} mIoU, "
                       f"{gap/o1*100:.0f}% of label-only gain). "
                       "SAM3 geometry refinement may be worth exploring.")
        else:
            verdict = (f"Boundary value is significant ({gap:+.3f} mIoU, "
                       f"{gap/o1*100:.0f}% of label-only gain). "
                       "SAM3 geometry refinement is STRONGLY recommended for this pair.")
        print(f"  Verdict: {verdict}")
        print()
        print("-" * 80)
        print()

    # Cross-pair summary
    print("=" * 80)
    print("Cross-pair summary:\n")
    print(f"  {'pair':<30}  {'n_rescue':>9}  {'O1 ΔmIoU':>10}  "
          f"{'O2 ΔmIoU':>10}  {'O3 ΔmIoU':>10}  {'gap O2-O1':>10}  {'gap%':>7}")
    print("  " + "-" * 85)
    for (w_idx, ru_idx) in requested_pairs:
        res = all_results[(w_idx, ru_idx)]
        pair_str = f"{class_names[w_idx]} -> {class_names[ru_idx]}"
        o1 = res['o1_gain']
        gap_pct = res['gap_o2_o1'] / o1 * 100 if abs(o1) > 1e-6 else float("nan")
        gap_pct_str = f"{gap_pct:>6.1f}%" if gap_pct == gap_pct else "   nan%"
        print(f"  {pair_str:<30}  {res['n_cc_rescue']:>9}  "
              f"{res['o1_gain']:>+10.3f}  {res['o2_gain']:>+10.3f}  "
              f"{res['o3_gain']:>+10.3f}  {res['gap_o2_o1']:>+10.3f}  {gap_pct_str}")

    print()
    print("gap O2-O1 = value of perfect boundary (SAM3 geometry refinement upper bound)")
    print("gap% = gap as fraction of label-only gain")
    print()
    print("Interpretation:")
    print("  gap% < 10%  -> boundary negligible, focus on semantic label correction only")
    print("  gap% 10-30% -> boundary has some value, SAM3 refine optional")
    print("  gap% > 30%  -> boundary matters, SAM3 geometry refinement justified")


if __name__ == "__main__":
    main()
