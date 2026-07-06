"""
Calibration-guided directed pair policy selection.

Runs inference on the calibration split (cal_images.txt produced by split_val.py),
evaluates three policy options for each candidate pair:
  - no-op  : keep baseline prediction
  - af     : always-flip all low-margin CCs (winner -> runnerup)
  - rc_<d> : RemoteCLIP gate at delta d (flip only if sim(top2)-sim(top1) > d)

Selects the best policy per pair using an eps guard:
  if best_rc_gain > max(af_gain, 0) + eps  -> policy = "remoteclip", delta = best_d
  elif af_gain > eps                        -> policy = "af"
  else                                      -> policy = "no-op"

Outputs a policy JSON file consumed by eval_combined_rerank.py.

Usage (OEM):
    python verify/calibrate_pair_policy.py configs/cfg_openearthmap.py \\
        --image-list verify/splits/oem/cal_images.txt \\
        --pairs "grass->cropland" "building->pavement" "bareland->pavement" "bareland->grass" \\
        --out verify/splits/oem/policy.json

Usage (LoveDA):
    python verify/calibrate_pair_policy.py configs/cfg_loveda.py \\
        --image-list verify/splits/loveda/cal_images.txt \\
        --pairs "agricultural->forest" "agricultural->barren" "agricultural->water" \\
        --out verify/splits/loveda/policy.json
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
    ap = argparse.ArgumentParser(description="Calibrate directed pair policy on cal split")
    ap.add_argument("config")
    ap.add_argument("--image-list", required=True,
                    help="Path to cal_images.txt from split_val.py")
    ap.add_argument("--pairs", nargs="+", required=True, metavar="winner->runnerup")
    ap.add_argument("--out", required=True, help="Output policy.json path")
    ap.add_argument("--margin-thr", type=float, default=0.10)
    ap.add_argument("--min-cc", type=int, default=MIN_CC_PIXELS)
    ap.add_argument("--delta-sweep", nargs="+", type=float,
                    default=[-0.10, -0.05, -0.02, 0.00, 0.02, 0.05, 0.10],
                    help="RemoteCLIP gate delta values to sweep")
    ap.add_argument("--eps", type=float, default=0.02,
                    help="Minimum gain above AF to prefer RemoteCLIP; min AF gain to prefer AF over no-op")
    ap.add_argument("--batch-clip", type=int, default=32)
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


def load_image_set(image_list_path):
    with open(image_list_path) as f:
        paths = [ln.strip() for ln in f if ln.strip()]
    return set(paths)


@torch.no_grad()
def main():
    args = parse_args()

    if not HAS_OPEN_CLIP:
        raise SystemExit("open_clip required: pip install open-clip-torch")

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
    for p in args.pairs:
        if "->" not in p:
            raise SystemExit(f"bad pair '{p}'")
        wn, run = p.split("->", 1)
        wn, run = wn.strip(), run.strip()
        requested_pairs.append((name2idx[wn], name2idx[run]))

    cal_set = load_image_set(args.image_list)
    print(f"Cal images: {len(cal_set)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading RemoteCLIP from {REMOTECLIP_WEIGHT} ...")
    rc_model, _, rc_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=REMOTECLIP_WEIGHT, device=device)
    rc_model.eval()
    rc_tokenizer = open_clip.get_tokenizer("ViT-B-32")

    prompts = [f"a satellite image of {cn}" for cn in class_names]
    tokens = rc_tokenizer(prompts).to(device)
    text_feats = F.normalize(rc_model.encode_text(tokens), dim=-1).float().cpu()
    print(f"Text features: {text_feats.shape}")

    seg_model = MODELS.build(cfg.model)
    seg_model.processor.model.eval()
    loader = Runner.build_dataloader(cfg.test_dataloader)

    cm_base = np.zeros((C, C), dtype=np.int64)

    # Per-pair accumulators
    # af_flips[(w,ru)]    : list of gt arrays for AF (all low-margin CC pixels)
    # rc_data[(w,ru)]     : list of (sim_delta, gt_array) per CC
    af_flips = {pair: [] for pair in requested_pairs}
    rc_data = {pair: [] for pair in requested_pairs}  # (sim_delta float, gt np.array)

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

            g_v = gt.reshape(-1)[vmask]
            p_v = base_p[vmask]
            np.add.at(cm_base, (g_v.numpy(), p_v.numpy()), 1)

            flat_v = flat[:, vmask]
            top2_v = flat_v.topk(2, dim=0)
            top1_idx = top2_v.indices[0]
            top2_idx = top2_v.indices[1]
            margin_v = top2_v.values[0] - top2_v.values[1]

            not_backfilled = (p_v == top1_idx) & (p_v != bg_idx) & (top2_idx != bg_idx)
            lm_mask = (margin_v < args.margin_thr) & not_backfilled

            if not lm_mask.any():
                continue

            valid_ys, valid_xs = torch.where(valid)
            valid_ys = valid_ys.numpy()
            valid_xs = valid_xs.numpy()
            gt_np = gt.numpy().astype(np.int32)

            image_np = None
            if os.path.exists(img_path):
                image_np = np.array(Image.open(img_path).convert("RGB"))

            for (w_idx, ru_idx) in requested_pairs:
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
                    if len(ys_cc) < args.min_cc:
                        continue

                    cc_gt = gt_np[ys_cc, xs_cc]
                    # AF: record GT of all pixels in this CC
                    af_flips[(w_idx, ru_idx)].append(cc_gt.copy())

                    # RemoteCLIP: compute sim_delta for this CC
                    if image_np is not None:
                        patch = crop_patch(image_np, ys_cc, xs_cc, PATCH_PAD, MIN_PATCH_SIZE)
                        if patch is not None:
                            img_t = rc_preprocess(patch).unsqueeze(0).to(device)
                            img_feat = F.normalize(rc_model.encode_image(img_t), dim=-1).float()
                            sim = (img_feat @ text_feats.to(device).T).squeeze(0)
                            sim_delta = float(sim[ru_idx].cpu()) - float(sim[w_idx].cpu())
                            rc_data[(w_idx, ru_idx)].append((sim_delta, cc_gt.copy()))

        n_done += len(out)
        if n_done % 20 == 0:
            print(f"  processed {n_done} images (cal: {n_cal}) ...")

    print(f"\nTotal images processed: {n_done}  |  Cal images used: {n_cal}")
    base_miou = float(np.nanmean(iou_from_cm(cm_base)) * 100)
    print(f"Cal set baseline mIoU: {base_miou:.3f}%\n")

    def _apply_gt_flips(flip_gt_list, w_idx, ru_idx):
        if not flip_gt_list:
            return base_miou
        all_gt = np.concatenate(flip_gt_list)
        cm = cm_base.astype(np.float64).copy()
        np.add.at(cm, (all_gt, w_idx), -1)
        np.add.at(cm, (all_gt, ru_idx), 1)
        return float(np.nanmean(iou_from_cm(cm)) * 100)

    policy = {}
    print("=" * 70)
    print("Policy selection per pair")
    print("=" * 70)

    for (w_idx, ru_idx) in requested_pairs:
        wn = class_names[w_idx]
        run = class_names[ru_idx]
        pair_key = f"{wn}->{run}"

        # AF gain
        af_miou = _apply_gt_flips(af_flips[(w_idx, ru_idx)], w_idx, ru_idx)
        af_gain = af_miou - base_miou

        # RemoteCLIP sweep
        rc_entries = rc_data[(w_idx, ru_idx)]
        best_rc_gain = -999.0
        best_delta = None
        rc_sweep_rows = []

        for delta in sorted(args.delta_sweep):
            flipped_gts = [gt_arr for (sd, gt_arr) in rc_entries if sd > delta]
            rc_miou = _apply_gt_flips(flipped_gts, w_idx, ru_idx) if flipped_gts else base_miou
            rc_gain = rc_miou - base_miou
            n_flip = len(flipped_gts)
            n_total = len(rc_entries)
            rc_sweep_rows.append((delta, rc_gain, n_flip, n_total))
            if rc_gain > best_rc_gain:
                best_rc_gain = rc_gain
                best_delta = delta

        # Policy decision with eps guard
        if best_rc_gain > max(af_gain, 0.0) + args.eps:
            chosen = "remoteclip"
            chosen_delta = best_delta
            chosen_gain = best_rc_gain
        elif af_gain > args.eps:
            chosen = "af"
            chosen_delta = None
            chosen_gain = af_gain
        else:
            chosen = "no-op"
            chosen_delta = None
            chosen_gain = 0.0

        policy[pair_key] = {
            "winner": wn,
            "runnerup": run,
            "w_idx": int(w_idx),
            "ru_idx": int(ru_idx),
            "policy": chosen,
            "delta": chosen_delta,
            "cal_gain": float(chosen_gain),
            "af_gain": float(af_gain),
            "best_rc_gain": float(best_rc_gain),
            "best_rc_delta": float(best_delta) if best_delta is not None else None,
            "margin_thr": args.margin_thr,
        }

        print(f"\nPair: {pair_key}")
        print(f"  AF gain:  {af_gain:+.3f} mIoU")
        print(f"  RemoteCLIP sweep:")
        for (d, rg, nf, nt) in rc_sweep_rows:
            marker = " <-- best" if d == best_delta else ""
            print(f"    delta={d:+.2f}  gain={rg:+.3f}  flipped={nf}/{nt}{marker}")
        print(f"  => Policy: {chosen}  (delta={chosen_delta})  gain={chosen_gain:+.3f}"
              f"  (eps={args.eps})")

    print("\n" + "=" * 70)
    print("Policy summary:")
    print(f"  {'pair':<30}  {'policy':<12}  {'delta':>7}  {'cal_gain':>9}")
    for pk, pv in policy.items():
        d_str = f"{pv['delta']:+.2f}" if pv['delta'] is not None else "   n/a"
        print(f"  {pk:<30}  {pv['policy']:<12}  {d_str:>7}  {pv['cal_gain']:>+9.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(policy, f, indent=2)
    print(f"\nPolicy saved to: {args.out}")


if __name__ == "__main__":
    main()
