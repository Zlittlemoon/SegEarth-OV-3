"""
Context-neighbor hypothesis diagnostic for SegEarth-OV3.

For each low-margin CC (top1=i, top2=j, margin < thr), this script:
  1. Extracts a ring-shaped neighbor region around the CC
     (dilate by r_outer pixels, subtract dilate by r_inner pixels).
  2. Keeps only high-confidence neighbor pixels (margin > conf_thr, not bg).
  3. Computes the majority class among those neighbor pixels.
  4. Asks: does the neighbor majority class match the GT class of the CC?

Metrics per pair:
  n_cc          : total CC patches evaluated
  rescue_rate   : fraction of CCs where GT == runnerup
  correct_rate  : fraction of CCs where GT == winner
  neighbor_hits_rescue  : among rescue CCs, fraction where neighbor majority == runnerup
  neighbor_hits_correct : among correct CCs, fraction where neighbor majority == winner
  neighbor_hits_top1    : among all CCs, fraction where neighbor majority == top1
  neighbor_hits_top2    : among all CCs, fraction where neighbor majority == top2
  neighbor_agree_gt     : among all CCs with a clear neighbor majority, fraction where
                          neighbor majority == GT (CC-level GT = majority GT of CC pixels)
  abstain_rate  : fraction of CCs where neighbor region has no high-conf pixels

The key question: is neighbor_agree_gt >> random (random = rescue_rate for runnerup)?

Usage (OEM, no pre-specified pairs -- auto-discovers top pairs):
    python verify/context_neighbor_diagnose.py configs/cfg_openearthmap.py \\
        --margin-thr 0.10 --conf-thr 0.30 --r-inner 0 --r-outer 16 \\
        --top-pairs 6 --limit 80

Usage (OEM, specific pairs):
    python verify/context_neighbor_diagnose.py configs/cfg_openearthmap.py \\
        --pairs "grass->cropland" "building->pavement" "bareland->pavement" "bareland->grass" \\
        --margin-thr 0.10 --conf-thr 0.30 --r-inner 0 --r-outer 16

Usage (LoveDA):
    python verify/context_neighbor_diagnose.py configs/cfg_loveda.py \\
        --pairs "agricultural->forest" "agricultural->barren" "agricultural->water" \\
        --margin-thr 0.10 --conf-thr 0.30 --r-inner 0 --r-outer 16
"""

import argparse
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmseg.registry import MODELS

import mmseg.datasets
import segearthov3_segmentor_merge
import custom_datasets

MIN_CC_PIXELS = 16       # ignore CCs smaller than this
MIN_NEIGHBOR_PIXELS = 4  # need at least this many high-conf neighbor pixels


def parse_args():
    ap = argparse.ArgumentParser(description="Context-neighbor hypothesis diagnostic")
    ap.add_argument("config")
    ap.add_argument("--pairs", nargs="+", metavar="winner->runnerup",
                    help="Directed pairs to evaluate, e.g. 'grass->cropland'. "
                         "If omitted, auto-discover top pairs by CC count.")
    ap.add_argument("--top-pairs", type=int, default=6,
                    help="When --pairs is omitted, show top-N pairs by CC count.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--margin-thr", type=float, default=0.10,
                    help="Top1-top2 logit margin threshold defining 'low-margin' pixels.")
    ap.add_argument("--conf-thr", type=float, default=0.30,
                    help="Margin threshold for a neighbor pixel to be considered "
                         "'high-confidence'.")
    ap.add_argument("--r-inner", type=int, default=0,
                    help="Inner dilation radius (pixels to exclude around the CC itself). "
                         "0 means the CC boundary is the inner edge.")
    ap.add_argument("--r-outer", nargs="+", type=int, default=[8, 16, 32],
                    help="Outer dilation radius (ring width). Multiple values are swept.")
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


def dilate_mask(mask, radius):
    """Binary dilation with a square structuring element of size 2*radius+1."""
    if radius <= 0:
        return mask.copy()
    struct = ndi.generate_binary_structure(2, 1)
    return ndi.binary_dilation(mask, structure=struct, iterations=radius)


def get_neighbor_ring(cc_mask_2d, r_inner, r_outer, H, W):
    """
    Returns a boolean mask for the ring region around cc_mask_2d:
      ring = dilate(cc, r_outer) - dilate(cc, r_inner) - cc itself
    """
    outer = dilate_mask(cc_mask_2d, r_outer)
    if r_inner > 0:
        inner = dilate_mask(cc_mask_2d, r_inner)
    else:
        inner = cc_mask_2d
    ring = outer & ~inner
    return ring


class PairStats:
    """Accumulates per-CC neighbor stats for one (winner, runnerup) pair."""

    def __init__(self, w_idx, ru_idx, r_outers):
        self.w_idx = w_idx
        self.ru_idx = ru_idx
        self.r_outers = r_outers
        # Per ring radius: list of (gt_cat, neighbor_majority_class or -1 if abstain)
        # gt_cat: 0=correct(GT==winner), 1=rescue(GT==runnerup), 2=other
        self.records = {r: [] for r in r_outers}
        self.n_cc = 0

    def add(self, gt_cat, neighbor_votes):
        """
        neighbor_votes: dict {r_outer -> int class index, or -1 if no high-conf neighbors}
        """
        self.n_cc += 1
        for r in self.r_outers:
            self.records[r].append((gt_cat, neighbor_votes.get(r, -1)))

    def report(self, class_names, C):
        lines = []
        n = self.n_cc
        if n == 0:
            return [f"  No CCs collected."]

        cats = np.array([rec[0] for rec in self.records[self.r_outers[0]]])
        n_rescue = int((cats == 1).sum())
        n_correct = int((cats == 0).sum())
        n_other = int((cats == 2).sum())
        lines.append(f"  n_cc={n}  rescue={n_rescue}({100*n_rescue/n:.1f}%)  "
                     f"correct={n_correct}({100*n_correct/n:.1f}%)  "
                     f"other={n_other}({100*n_other/n:.1f}%)")
        lines.append("")

        w_name = class_names[self.w_idx]
        ru_name = class_names[self.ru_idx]

        header = (f"  {'r_outer':>8}  {'abstain':>8}  "
                  f"{'nbr==GT':>8}  {'nbr==top2':>10}  {'nbr==top1':>10}  "
                  f"{'rescue:nbr==top2':>17}  {'correct:nbr==top1':>18}  "
                  f"{'ΔmIoU_af':>10}")
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for r in self.r_outers:
            recs = self.records[r]
            gt_cats = np.array([x[0] for x in recs])
            nbr_cls = np.array([x[1] for x in recs])

            abstain = nbr_cls == -1
            n_abstain = abstain.sum()
            abstain_rate = n_abstain / n if n > 0 else float("nan")

            has_nbr = ~abstain
            n_has = has_nbr.sum()

            if n_has == 0:
                lines.append(f"  {r:>8}  {abstain_rate:>8.1%}  "
                              f"{'---':>8}  {'---':>10}  {'---':>10}  "
                              f"{'---':>17}  {'---':>18}  {'---':>10}")
                continue

            # Among CCs with neighbor signal, how often does neighbor match GT?
            nbr_eq_gt = (nbr_cls == self.w_idx) & (gt_cats == 0) | \
                        (nbr_cls == self.ru_idx) & (gt_cats == 1)
            # Simpler: nbr majority class == majority GT class of the CC
            # gt_cat==0 means GT==winner, gt_cat==1 means GT==runnerup
            gt_class = np.where(gt_cats == 0, self.w_idx,
                                np.where(gt_cats == 1, self.ru_idx, -1))
            nbr_agree_gt = ((nbr_cls == gt_class) & has_nbr & (gt_cats != 2)).sum()
            denom_agree = ((has_nbr) & (gt_cats != 2)).sum()
            nbr_agree_gt_rate = nbr_agree_gt / denom_agree if denom_agree > 0 else float("nan")

            nbr_eq_top2 = ((nbr_cls == self.ru_idx) & has_nbr).sum() / n_has
            nbr_eq_top1 = ((nbr_cls == self.w_idx) & has_nbr).sum() / n_has

            # rescue CCs: neighbor == top2 (runnerup) rate
            rescue_mask = (gt_cats == 1) & has_nbr
            n_rescue_has = rescue_mask.sum()
            rescue_nbr_top2 = ((nbr_cls == self.ru_idx) & rescue_mask).sum() / n_rescue_has \
                if n_rescue_has > 0 else float("nan")

            # correct CCs: neighbor == top1 (winner) rate
            correct_mask = (gt_cats == 0) & has_nbr
            n_correct_has = correct_mask.sum()
            correct_nbr_top1 = ((nbr_cls == self.w_idx) & correct_mask).sum() / n_correct_has \
                if n_correct_has > 0 else float("nan")

            def _pct(v):
                return f"{'nan':>8}" if (isinstance(v, float) and np.isnan(v)) \
                    else f"{v:>8.1%}"

            lines.append(
                f"  {r:>8}  {abstain_rate:>8.1%}  "
                f"{nbr_agree_gt_rate:>8.1%}  {nbr_eq_top2:>10.1%}  {nbr_eq_top1:>10.1%}  "
                f"{rescue_nbr_top2:>17.1%}  {correct_nbr_top1:>18.1%}  "
                f"{'(see below)':>10}"
            )

        lines.append("")
        lines.append(f"  Key baseline rates (random guess):")
        lines.append(f"    If neighbor is uninformative, nbr==GT rate ≈ rescue_rate = "
                     f"{n_rescue/(n_rescue+n_correct):.1%} (among rescue+correct CCs)")
        lines.append(f"    If neighbor always picks winner (top1): nbr==GT rate for "
                     f"correct = 100%, rescue = 0%")
        return lines


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

    requested_pairs = []
    if args.pairs:
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

    r_outers = sorted(args.r_outer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"margin_thr={args.margin_thr}  conf_thr={args.conf_thr}  "
          f"r_inner={args.r_inner}  r_outers={r_outers}")
    print(f"Limit: {args.limit or 'all'}")

    seg_model = MODELS.build(cfg.model)
    seg_model.processor.model.eval()
    loader = Runner.build_dataloader(cfg.test_dataloader)

    cm_base = np.zeros((C, C), dtype=np.int64)

    # Auto-discover pairs if not specified
    pair_cc_counts = defaultdict(int)  # (w,ru) -> CC count (for auto-discovery)

    # PairStats accumulators
    pair_stats = {}  # (w,ru) -> PairStats
    for pair in requested_pairs:
        pair_stats[pair] = PairStats(pair[0], pair[1], r_outers)

    n_done = 0
    for batch in loader:
        out = seg_model.predict(batch["inputs"], batch["data_samples"])
        for ds in out:
            logits = ds.seg_logits.data.float().cpu()       # (C, H, W)
            gt = ds.gt_sem_seg.data.squeeze(0).long().cpu() # (H, W)
            valid = (gt >= 0) & (gt < C)
            H, W = gt.shape

            flat = logits.reshape(C, -1)                    # (C, P)
            vmask = valid.reshape(-1)

            base_p = flat.argmax(0).clone()
            mx = flat.max(0).values
            thd_vec_t = torch.full((C,), prob_thd)
            gate = thd_vec_t[base_p]
            base_p[mx < gate] = bg_idx

            g_v = gt.reshape(-1)[vmask]
            p_v = base_p[vmask]
            np.add.at(cm_base, (g_v.numpy(), p_v.numpy()), 1)

            flat_v = flat[:, vmask]                         # (C, V)
            top2_v = flat_v.topk(2, dim=0)
            top1_idx = top2_v.indices[0].numpy()            # (V,)
            top2_idx = top2_v.indices[1].numpy()            # (V,)
            margin_v = (top2_v.values[0] - top2_v.values[1]).numpy()  # (V,)

            p_v_np = p_v.numpy()
            g_v_np = g_v.numpy()

            not_backfilled = (p_v_np == top1_idx) & (p_v_np != bg_idx) & (top2_idx != bg_idx)
            lm_mask = (margin_v < args.margin_thr) & not_backfilled

            # High-confidence mask (for neighbor region)
            hc_mask = (margin_v >= args.conf_thr) & not_backfilled  # (V,) bool

            # Valid pixel 2D coords
            valid_ys, valid_xs = torch.where(valid)
            valid_ys = valid_ys.numpy()
            valid_xs = valid_xs.numpy()

            # Build full 2D maps for neighbor lookup
            # pred_2d[y,x] = predicted class (only for high-conf valid fg pixels)
            hc_class_2d = np.full((H, W), -1, dtype=np.int32)
            hc_positions = np.where(hc_mask)[0]
            if len(hc_positions) > 0:
                hc_ys = valid_ys[hc_positions]
                hc_xs = valid_xs[hc_positions]
                hc_class_2d[hc_ys, hc_xs] = top1_idx[hc_positions]

            # Determine pairs to process for this image
            active_pairs = requested_pairs if requested_pairs else []

            # If auto-discover mode, also count all lm pairs
            if not requested_pairs:
                lm_positions = np.where(lm_mask)[0]
                for vi in lm_positions:
                    w = int(top1_idx[vi])
                    ru = int(top2_idx[vi])
                    if w != bg_idx and ru != bg_idx:
                        pair_cc_counts[(w, ru)] += 1

            for (w_idx, ru_idx) in active_pairs:
                pair_lm = lm_mask & (top1_idx == w_idx) & (top2_idx == ru_idx)
                if not pair_lm.any():
                    continue

                local_indices = np.where(pair_lm)[0]
                pair_ys = valid_ys[local_indices]
                pair_xs = valid_xs[local_indices]
                g_pair = g_v_np[pair_lm]
                cat = np.where(g_pair == w_idx, 0,
                               np.where(g_pair == ru_idx, 1, 2))

                pair_2d = np.zeros((H, W), dtype=bool)
                pair_2d[pair_ys, pair_xs] = True

                ccs = extract_connected_components(pair_2d)

                cc_pixel_set_cache = {}
                for ys_cc, xs_cc in ccs:
                    if len(ys_cc) < MIN_CC_PIXELS:
                        continue

                    # Map CC pixels to local_indices
                    cc_2d_set = set(zip(ys_cc.tolist(), xs_cc.tolist()))
                    cc_local = np.array([
                        li for li, (py, px) in enumerate(zip(pair_ys, pair_xs))
                        if (int(py), int(px)) in cc_2d_set
                    ])
                    if len(cc_local) == 0:
                        continue

                    cc_cats = cat[cc_local]
                    majority_cat = int(np.bincount(cc_cats, minlength=3).argmax())

                    # Build CC mask
                    cc_mask_2d = np.zeros((H, W), dtype=bool)
                    cc_mask_2d[ys_cc, xs_cc] = True

                    # Compute neighbor majority for each r_outer
                    neighbor_votes = {}
                    for r_out in r_outers:
                        ring = get_neighbor_ring(cc_mask_2d, args.r_inner, r_out, H, W)
                        # Get high-conf classes in ring
                        ring_classes = hc_class_2d[ring]
                        ring_classes = ring_classes[ring_classes >= 0]  # filter -1

                        if len(ring_classes) < MIN_NEIGHBOR_PIXELS:
                            neighbor_votes[r_out] = -1  # abstain
                        else:
                            # majority vote
                            counts = np.bincount(ring_classes, minlength=C)
                            neighbor_votes[r_out] = int(counts.argmax())

                    if (w_idx, ru_idx) not in pair_stats:
                        pair_stats[(w_idx, ru_idx)] = PairStats(w_idx, ru_idx, r_outers)
                    pair_stats[(w_idx, ru_idx)].add(majority_cat, neighbor_votes)

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

    if not requested_pairs and pair_cc_counts:
        top_pairs = sorted(pair_cc_counts.items(), key=lambda x: -x[1])[:args.top_pairs]
        print(f"Auto-discovered top-{args.top_pairs} pairs by CC count:")
        for (w, ru), cnt in top_pairs:
            print(f"  {class_names[w]:>20} -> {class_names[ru]:<20}  n_cc_pixels={cnt}")
        print()

    if not pair_stats:
        print("No pairs to report. Use --pairs or run without --pairs for auto-discovery.")
        return

    for (w_idx, ru_idx), stats in pair_stats.items():
        w_name = class_names[w_idx]
        ru_name = class_names[ru_idx]
        print(f"\nPair: {w_name} -> {ru_name}   "
              f"(margin_thr={args.margin_thr}, conf_thr={args.conf_thr})")
        print("-" * 70)
        for line in stats.report(class_names, C):
            print(line)
        print()

    # =========================================================================
    # Cross-pair summary
    # =========================================================================
    print("=" * 80)
    print("Cross-pair summary (r_outer sweep, nbr_agree_gt = key metric):\n")
    header = f"  {'pair':<30}"
    for r in r_outers:
        header += f"  {'r='+str(r)+' nbr_agree_gt':>18}  {'abstain':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for (w_idx, ru_idx), stats in pair_stats.items():
        row = f"  {class_names[w_idx]+' -> '+class_names[ru_idx]:<30}"
        for r in r_outers:
            recs = stats.records[r]
            if not recs:
                row += f"  {'---':>18}  {'---':>8}"
                continue
            gt_cats = np.array([x[0] for x in recs])
            nbr_cls = np.array([x[1] for x in recs])
            abstain = nbr_cls == -1
            has_nbr = ~abstain
            abstain_rate = abstain.sum() / len(recs)

            gt_class = np.where(gt_cats == 0, w_idx,
                                np.where(gt_cats == 1, ru_idx, -1))
            denom = ((has_nbr) & (gt_cats != 2)).sum()
            agree = ((nbr_cls == gt_class) & has_nbr & (gt_cats != 2)).sum()
            rate = agree / denom if denom > 0 else float("nan")
            row += f"  {rate:>18.1%}  {abstain_rate:>8.1%}"
        print(row)

    print("\nNote: nbr_agree_gt is the fraction of CCs where neighbor majority class == GT.")
    print("      Random baseline = rescue_rate (if neighbor always picks winner,")
    print("      nbr_agree_gt = correct_rate; if always picks runnerup = rescue_rate).")
    print("      A value significantly above max(rescue_rate, correct_rate) means the")
    print("      neighbor context is informative beyond simple class-frequency prior.")


if __name__ == "__main__":
    main()
