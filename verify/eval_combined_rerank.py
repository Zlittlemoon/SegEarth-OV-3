"""
Combined directed pair reranking evaluation on the held-out eval split.

Reads the policy.json produced by calibrate_pair_policy.py and applies all
pair policies simultaneously on the evaluation split (eval_images.txt).

For each image, all pairs are processed in a single pass; flips from all pairs
are accumulated into one unified confusion matrix before computing mIoU.
This avoids the error of summing single-pair ΔmIoU values.

Three strategies are reported:
  Baseline          : no flip
  AF-selected       : same pairs as Hybrid, but all use always-flip
  Hybrid            : each pair uses the policy selected during calibration
                      (no-op / af / remoteclip with calibrated delta)

The key question:
  Hybrid > AF-selected  =>  RemoteCLIP gate adds value beyond always-flip
  Hybrid ≈ AF-selected  =>  AF is sufficient; RemoteCLIP not needed in main method

Usage (OEM):
    python verify/eval_combined_rerank.py configs/cfg_openearthmap.py \\
        --image-list verify/splits/oem/eval_images.txt \\
        --policy     verify/splits/oem/policy.json

Usage (LoveDA):
    python verify/eval_combined_rerank.py configs/cfg_loveda.py \\
        --image-list verify/splits/loveda/eval_images.txt \\
        --policy     verify/splits/loveda/policy.json
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmseg.registry import MODELS
from scipy import ndimage as ndi

try:
    import open_clip
    HAS_OPEN_CLIP = True
except ImportError:
    HAS_OPEN_CLIP = False

import mmseg.datasets
import segearthov3_segmentor_merge
import custom_datasets

WEIGHTS_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
REMOTECLIP_WEIGHT = os.path.join(WEIGHTS_ROOT, "RemoteCLIP", "RemoteCLIP-ViT-B-32.pt")

MIN_CC_PIXELS = 16
PATCH_PAD = 8
MIN_PATCH_SIZE = 32


def parse_args():
    ap = argparse.ArgumentParser(description="Combined reranking evaluation on eval split")
    ap.add_argument("config")
    ap.add_argument("--image-list", required=True,
                    help="Path to eval_images.txt from split_val.py")
    ap.add_argument("--policy", required=True,
                    help="Path to policy.json from calibrate_pair_policy.py")
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


def crop_patch(image_np, ys, xs, pad, min_size):
    H, W = image_np.shape[:2]
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(H, int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(W, int(xs.max()) + 1 + pad)
    if (y1 - y0) < min_size or (x1 - x0) < min_size:
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        half = min_size // 2
        y0 = max(0, cy - half); y1 = min(H, cy + half)
        x0 = max(0, cx - half); x1 = min(W, cx + half)
    patch = image_np[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    return Image.fromarray(patch)


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

    with open(args.policy) as f:
        policy_raw = json.load(f)

    # Build active pair list: only pairs with policy != "no-op"
    active_pairs = []
    for pk, pv in policy_raw.items():
        if pv["policy"] != "no-op":
            active_pairs.append(pv)
    print(f"Active pairs from policy: {len(active_pairs)}")
    for pv in active_pairs:
        d_str = f"{pv['delta']:+.2f}" if pv['delta'] is not None else "n/a"
        print(f"  {pv['winner']}->{pv['runnerup']}  policy={pv['policy']}  delta={d_str}")

    margin_thr = policy_raw[list(policy_raw.keys())[0]]["margin_thr"]
    print(f"margin_thr: {margin_thr}")

    needs_rc = any(pv["policy"] == "remoteclip" for pv in active_pairs)
    rc_model = rc_preprocess = text_feats = None

    if needs_rc:
        if not HAS_OPEN_CLIP:
            raise SystemExit("open_clip required: pip install open-clip-torch")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading RemoteCLIP from {REMOTECLIP_WEIGHT} ...")
        rc_model, _, rc_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=REMOTECLIP_WEIGHT, device=device)
        rc_model.eval()
        rc_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        prompts = [f"a satellite image of {cn}" for cn in class_names]
        tokens = rc_tokenizer(prompts).to(device)
        text_feats = F.normalize(rc_model.encode_text(tokens), dim=-1).float()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_set = load_image_set(args.image_list)
    print(f"Eval images: {len(eval_set)}")

    seg_model = MODELS.build(cfg.model)
    seg_model.processor.model.eval()
    loader = Runner.build_dataloader(cfg.test_dataloader)

    # Three CMs: baseline, AF-selected, Hybrid
    cm_base = np.zeros((C, C), dtype=np.int64)
    # Flip accumulators: list of (gt_array, w_idx, ru_idx)
    af_flips_all = []     # AF-selected: all active pairs use AF
    hybrid_flips_all = [] # Hybrid: each pair uses its calibrated policy

    # Per-pair stats
    pair_stats = {pv["winner"] + "->" + pv["runnerup"]: {
        "n_cc": 0, "n_af_flip": 0, "n_hybrid_flip": 0
    } for pv in active_pairs}

    n_done = 0
    n_eval = 0
    for batch in loader:
        out = seg_model.predict(batch["inputs"], batch["data_samples"])
        for ds in out:
            img_path = ds.img_path if hasattr(ds, "img_path") else ""
            if img_path not in eval_set:
                continue
            n_eval += 1

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

            g_v = gt.reshape(-1)[vmask]
            p_v = base_p[vmask]
            np.add.at(cm_base, (g_v.numpy(), p_v.numpy()), 1)

            flat_v = flat[:, vmask]
            top2_v = flat_v.topk(2, dim=0)
            top1_idx = top2_v.indices[0]
            top2_idx = top2_v.indices[1]
            margin_v = top2_v.values[0] - top2_v.values[1]

            not_backfilled = (p_v == top1_idx) & (p_v != bg_idx) & (top2_idx != bg_idx)
            lm_mask = (margin_v < margin_thr) & not_backfilled

            if not lm_mask.any():
                continue

            valid_ys, valid_xs = torch.where(valid)
            valid_ys = valid_ys.numpy()
            valid_xs = valid_xs.numpy()
            gt_np = gt.numpy().astype(np.int32)

            image_np = None
            if needs_rc and os.path.exists(img_path):
                image_np = np.array(Image.open(img_path).convert("RGB"))

            for pv in active_pairs:
                w_idx = pv["w_idx"]
                ru_idx = pv["ru_idx"]
                pol = pv["policy"]
                rc_delta = pv["delta"]
                pkey = pv["winner"] + "->" + pv["runnerup"]

                pair_lm = lm_mask & (top1_idx == w_idx) & (top2_idx == ru_idx)
                if not pair_lm.any():
                    continue

                local_indices = np.where(pair_lm.numpy())[0]
                pair_ys = valid_ys[local_indices]
                pair_xs = valid_xs[local_indices]

                pair_2d = np.zeros((H, W), dtype=bool)
                pair_2d[pair_ys, pair_xs] = True
                ccs = extract_connected_components(pair_2d)

                for ys_cc, xs_cc in ccs:
                    if len(ys_cc) < MIN_CC_PIXELS:
                        continue

                    cc_gt = gt_np[ys_cc, xs_cc]
                    pair_stats[pkey]["n_cc"] += 1

                    # AF-selected: always flip this CC
                    af_flips_all.append((cc_gt.copy(), w_idx, ru_idx))
                    pair_stats[pkey]["n_af_flip"] += 1

                    # Hybrid: apply calibrated policy
                    if pol == "af":
                        hybrid_flips_all.append((cc_gt.copy(), w_idx, ru_idx))
                        pair_stats[pkey]["n_hybrid_flip"] += 1
                    elif pol == "remoteclip" and image_np is not None:
                        patch = crop_patch(image_np, ys_cc, xs_cc, PATCH_PAD, MIN_PATCH_SIZE)
                        if patch is not None:
                            img_t = rc_preprocess(patch).unsqueeze(0).to(device)
                            img_feat = F.normalize(rc_model.encode_image(img_t), dim=-1).float()
                            sim = (img_feat @ text_feats.T).squeeze(0)
                            sim_delta = float(sim[ru_idx].cpu()) - float(sim[w_idx].cpu())
                            if sim_delta > rc_delta:
                                hybrid_flips_all.append((cc_gt.copy(), w_idx, ru_idx))
                                pair_stats[pkey]["n_hybrid_flip"] += 1

        n_done += len(out)
        if n_done % 20 == 0:
            print(f"  processed {n_done} images (eval: {n_eval}) ...")

    print(f"\nTotal images processed: {n_done}  |  Eval images used: {n_eval}")

    # Apply flips to get final CMs
    def _apply_flips(flip_list):
        cm = cm_base.astype(np.float64).copy()
        for (gt_arr, w_idx, ru_idx) in flip_list:
            np.add.at(cm, (gt_arr, w_idx), -1)
            np.add.at(cm, (gt_arr, ru_idx), 1)
        return cm

    cm_af = _apply_flips(af_flips_all)
    cm_hybrid = _apply_flips(hybrid_flips_all)

    base_miou = float(np.nanmean(iou_from_cm(cm_base)) * 100)
    af_miou = float(np.nanmean(iou_from_cm(cm_af)) * 100)
    hybrid_miou = float(np.nanmean(iou_from_cm(cm_hybrid)) * 100)

    iou_base = iou_from_cm(cm_base) * 100
    iou_af = iou_from_cm(cm_af) * 100
    iou_hybrid = iou_from_cm(cm_hybrid) * 100

    print("\n" + "=" * 70)
    print("Combined Reranking Results (eval split)")
    print("=" * 70)
    print(f"\n  {'Method':<20}  {'mIoU':>8}  {'ΔmIoU':>8}")
    print(f"  {'-'*42}")
    print(f"  {'Baseline':<20}  {base_miou:>8.3f}  {'':>8}")
    print(f"  {'AF-selected':<20}  {af_miou:>8.3f}  {af_miou-base_miou:>+8.3f}")
    print(f"  {'Hybrid':<20}  {hybrid_miou:>8.3f}  {hybrid_miou-base_miou:>+8.3f}")
    print(f"\n  Hybrid vs AF-selected:  {hybrid_miou-af_miou:>+.3f} mIoU")

    verdict = ""
    diff = hybrid_miou - af_miou
    if diff > 0.02:
        verdict = "Hybrid > AF-selected: RemoteCLIP gate adds value. Include in main method."
    elif diff < -0.02:
        verdict = "Hybrid < AF-selected: RemoteCLIP gate hurts. Use AF-only as main method."
    else:
        verdict = "Hybrid ≈ AF-selected: RemoteCLIP gate has no clear advantage. AF sufficient."
    print(f"\n  Verdict: {verdict}")

    # Per-class delta table
    print("\n  Per-class IoU delta (Hybrid vs Baseline):")
    print(f"  {'class':<20}  {'base IoU':>9}  {'hybrid IoU':>10}  {'Δ':>7}")
    print(f"  {'-'*52}")
    for i, cn in enumerate(class_names):
        b = iou_base[i]
        h = iou_hybrid[i]
        if np.isnan(b) and np.isnan(h):
            continue
        b_str = f"{b:9.2f}" if not np.isnan(b) else "      nan"
        h_str = f"{h:10.2f}" if not np.isnan(h) else "       nan"
        d_str = f"{h-b:+7.2f}" if not (np.isnan(b) or np.isnan(h)) else "    nan"
        print(f"  {cn:<20}  {b_str}  {h_str}  {d_str}")

    # Per-pair CC / flip counts
    print("\n  Per-pair statistics:")
    print(f"  {'pair':<30}  {'policy':<12}  {'n_cc':>6}  {'af_flips':>9}  {'hybrid_flips':>13}")
    print(f"  {'-'*76}")
    for pv in active_pairs:
        pkey = pv["winner"] + "->" + pv["runnerup"]
        st = pair_stats[pkey]
        d_str = f"{pv['delta']:+.2f}" if pv['delta'] is not None else "  n/a"
        pol_str = f"{pv['policy']}({d_str})" if pv['policy'] == 'remoteclip' else pv['policy']
        print(f"  {pkey:<30}  {pol_str:<12}  {st['n_cc']:>6}  "
              f"{st['n_af_flip']:>9}  {st['n_hybrid_flip']:>13}")

    changed_ratio = len(hybrid_flips_all) / max(1, sum(s["n_cc"] for s in pair_stats.values()))
    print(f"\n  Total CCs flipped (Hybrid): {len(hybrid_flips_all)}"
          f"  ({changed_ratio:.1%} of active CCs)")


if __name__ == "__main__":
    main()
