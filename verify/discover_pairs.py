"""
Automatically discover directed fg-fg low-margin pairs from the calibration split.

For every ordered pair (winner, runnerup) of foreground classes, counts the number
of connected components (CCs) that satisfy:
  - margin < margin_thr
  - prediction == winner (not backfilled to background)
  - top2 == runnerup
  - CC size >= min_cc pixels

Reports pairs sorted by n_cc descending and writes a JSON file listing pairs
above a minimum CC count threshold, along with their rescue rate estimate
(fraction of CC pixels where GT == runnerup).

The output JSON is a list of {"winner": ..., "runnerup": ..., "n_cc": ...,
"rescue_rate": ..., "n_pixel": ...} entries, sorted by n_cc descending.
This list can be passed directly to calibrate_pair_policy.py via --pairs.

Usage (OEM):
    python verify/discover_pairs.py configs/cfg_openearthmap.py \\
        --image-list verify/splits/oem/cal_images.txt \\
        --out verify/splits/oem/discovered_pairs.json

Usage (LoveDA):
    python verify/discover_pairs.py configs/cfg_loveda.py \\
        --image-list verify/splits/loveda/cal_images.txt \\
        --out verify/splits/loveda/discovered_pairs.json
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmseg.registry import MODELS
from scipy import ndimage as ndi

import mmseg.datasets
import segearthov3_segmentor_merge
import custom_datasets

MIN_CC_PIXELS = 16
PATCH_PAD = 8


def parse_args():
    ap = argparse.ArgumentParser(
        description="Discover directed fg-fg low-margin pairs from the cal split"
    )
    ap.add_argument("config")
    ap.add_argument("--image-list", required=True,
                    help="Path to cal_images.txt from split_val.py")
    ap.add_argument("--out", required=True,
                    help="Output JSON path for discovered pairs")
    ap.add_argument("--margin-thr", type=float, default=0.10)
    ap.add_argument("--min-cc", type=int, default=MIN_CC_PIXELS,
                    help="Minimum CC size in pixels to count")
    ap.add_argument("--min-n-cc", type=int, default=200,
                    help="Minimum total CC count for a pair to be reported")
    ap.add_argument("--cfg-options", nargs="+", action=DictAction, default=None)
    return ap.parse_args()


def extract_connected_components(binary_mask):
    label_map, n_labels = ndi.label(binary_mask)
    ccs = []
    for lb in range(1, n_labels + 1):
        ys, xs = np.where(label_map == lb)
        ccs.append((ys, xs))
    return ccs


def load_image_set(path):
    with open(path) as f:
        return set(ln.strip() for ln in f if ln.strip())


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

    prob_thd = float(cfg.model.get("prob_thd", 0.0))
    bg_idx = int(cfg.model.get("bg_idx", 0))

    cal_set = load_image_set(args.image_list)
    print(f"Cal images: {len(cal_set)}")
    print(f"Classes ({C}): {class_names}")
    print(f"margin_thr={args.margin_thr}  min_cc={args.min_cc}  min_n_cc={args.min_n_cc}\n")

    seg_model = MODELS.build(cfg.model)
    seg_model.processor.model.eval()
    loader = Runner.build_dataloader(cfg.test_dataloader)

    # Accumulators keyed by (w_idx, ru_idx)
    # n_cc[pair]         : total CC count
    # n_pixel[pair]      : total low-margin pixel count across all CCs
    # rescue_pixel[pair] : pixels where GT == runnerup (rescue pixels)
    n_cc = defaultdict(int)
    n_pixel = defaultdict(int)
    rescue_pixel = defaultdict(int)

    n_done = 0
    n_cal = 0
    for batch in loader:
        out = seg_model.predict(batch["inputs"], batch["data_samples"])
        for ds in out:
            img_path = ds.img_path if hasattr(ds, "img_path") else ""
            if img_path not in cal_set:
                continue
            n_cal += 1

            logits = ds.seg_logits.data.float().cpu()
            gt = ds.gt_sem_seg.data.squeeze(0).long().cpu()
            valid = (gt >= 0) & (gt < C)
            H, W = gt.shape

            flat = logits.reshape(C, -1)
            vmask = valid.reshape(-1)

            base_p = flat.argmax(0).clone()
            mx = flat.max(0).values
            thd_vec_t = torch.full((C,), prob_thd)
            gate = thd_vec_t[base_p]
            base_p[mx < gate] = bg_idx

            flat_v = flat[:, vmask]
            top2_v = flat_v.topk(2, dim=0)
            top1_idx = top2_v.indices[0]
            top2_idx = top2_v.indices[1]
            margin_v = top2_v.values[0] - top2_v.values[1]

            p_v = base_p[vmask]
            g_v = gt.reshape(-1)[vmask]

            not_backfilled = (p_v == top1_idx) & (p_v != bg_idx) & (top2_idx != bg_idx)
            lm_mask = (margin_v < args.margin_thr) & not_backfilled

            if not lm_mask.any():
                continue

            valid_ys, valid_xs = torch.where(valid)
            valid_ys_np = valid_ys.numpy()
            valid_xs_np = valid_xs.numpy()
            gt_np = gt.numpy().astype(np.int32)

            top1_np = top1_idx.numpy()
            top2_np = top2_idx.numpy()
            lm_np = lm_mask.numpy()
            g_np = g_v.numpy()

            # Collect all (w, ru) pairs present in this image
            lm_indices = np.where(lm_np)[0]
            if len(lm_indices) == 0:
                continue

            w_arr = top1_np[lm_indices]
            ru_arr = top2_np[lm_indices]

            # Find unique directed pairs in this image (skip pairs involving bg)
            unique_pairs = set(zip(w_arr.tolist(), ru_arr.tolist()))
            unique_pairs = {(w, ru) for w, ru in unique_pairs
                            if w != bg_idx and ru != bg_idx and w != ru}

            for (w_idx, ru_idx) in unique_pairs:
                pair_lm = lm_np & (top1_np == w_idx) & (top2_np == ru_idx)
                if not pair_lm.any():
                    continue

                local_indices = np.where(pair_lm)[0]
                pair_ys = valid_ys_np[local_indices]
                pair_xs = valid_xs_np[local_indices]

                pair_2d = np.zeros((H, W), dtype=bool)
                pair_2d[pair_ys, pair_xs] = True
                ccs = extract_connected_components(pair_2d)

                for ys_cc, xs_cc in ccs:
                    if len(ys_cc) < args.min_cc:
                        continue
                    cc_gt = gt_np[ys_cc, xs_cc]
                    n_cc[(w_idx, ru_idx)] += 1
                    n_pixel[(w_idx, ru_idx)] += len(ys_cc)
                    rescue_pixel[(w_idx, ru_idx)] += int((cc_gt == ru_idx).sum())

        n_done += len(out)
        if n_done % 20 == 0:
            print(f"  processed {n_done} images (cal: {n_cal}) ...")

    print(f"\nTotal images processed: {n_done}  |  Cal images used: {n_cal}")

    # Build result table
    all_pairs = sorted(n_cc.keys(), key=lambda p: n_cc[p], reverse=True)

    print("\n" + "=" * 72)
    print(f"{'pair':<35}  {'n_cc':>6}  {'n_pixel':>9}  {'rescue_rate':>12}  {'keep':>5}")
    print("=" * 72)

    results = []
    for (w_idx, ru_idx) in all_pairs:
        wn = class_names[w_idx]
        run = class_names[ru_idx]
        nc = n_cc[(w_idx, ru_idx)]
        np_ = n_pixel[(w_idx, ru_idx)]
        rr = rescue_pixel[(w_idx, ru_idx)] / max(1, np_)
        keep = nc >= args.min_n_cc
        flag = "YES" if keep else "---"
        print(f"  {wn}->{run:<30}  {nc:>6}  {np_:>9}  {rr:>11.1%}  {flag:>5}")
        if keep:
            results.append({
                "winner": wn,
                "runnerup": run,
                "w_idx": int(w_idx),
                "ru_idx": int(ru_idx),
                "n_cc": int(nc),
                "n_pixel": int(np_),
                "rescue_rate": round(float(rr), 4),
            })

    print(f"\nPairs with n_cc >= {args.min_n_cc}: {len(results)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to: {args.out}")

    # Print suggested --pairs argument for calibrate_pair_policy.py
    if results:
        pairs_arg = " ".join(f'"{r["winner"]}->{r["runnerup"]}"' for r in results)
        print(f"\nSuggested --pairs argument for calibrate_pair_policy.py:")
        print(f"  --pairs {pairs_arg}")


if __name__ == "__main__":
    main()
