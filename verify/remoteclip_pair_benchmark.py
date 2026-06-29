"""
Offline RemoteCLIP (vs. CLIP) directed-pair arbiter benchmark for SegEarth-OV3.

For each directed (winner, runnerup) pair in the oracle whitelist, this script:
  1. Runs the same inference pass as oracle_ceiling.py to collect low-margin
     connected-component patches with GT labels.
  2. Feeds each patch to RemoteCLIP and vanilla CLIP (ViT-B-32) and computes
     sim(top1_text) vs sim(top2_text). If sim(top2) - sim(top1) > delta, the
     arbiter flips winner -> runnerup.
  3. Reports per-pair and aggregate metrics:
       flip_precision   : among flipped pixels, % with GT == runnerup
       flip_recall      : among rescue pixels (GT==runnerup), % the arbiter flipped
       abstain_rate     : % of candidates where arbiter held (delta not exceeded)
       expected_ΔmIoU   : full CM simulation of the actual flip decisions
  4. Sweeps abstain threshold delta to produce a precision-recall-like curve.

Model weights:
  CLIP        : weights/clip/ViT-B-32.pt
  RemoteCLIP  : weights/RemoteCLIP/RemoteCLIP-ViT-B-32.pt

The whitelist of directed pairs is supplied via --pairs "winner->runnerup" (one or
more). If omitted, the top pairs from oracle_ceiling Phase 1B output are used
as a default heuristic.

Usage (OEM):
    python verify/remoteclip_pair_benchmark.py configs/cfg_openearthmap.py \\
        --pairs "grass->cropland" "building->pavement" "bareland->pavement" \\
                "bareland->grass" \\
        --limit 80 --margin-thr 0.2 \\
        --delta 0.0 0.05 0.10 0.15 0.20 0.30

Usage (LoveDA):
    python verify/remoteclip_pair_benchmark.py configs/cfg_loveda.py \\
        --pairs "agricultural->forest" "agricultural->barren" \\
        --limit 80 --margin-thr 0.2 \\
        --delta 0.0 0.05 0.10 0.15 0.20 0.30
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmseg.registry import MODELS

try:
    import open_clip
    HAS_OPEN_CLIP = True
except ImportError:
    HAS_OPEN_CLIP = False
    print("WARNING: open_clip not found. Install with: pip install open-clip-torch")

import mmseg.datasets
import segearthov3_segmentor_merge
import custom_datasets

WEIGHTS_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
CLIP_WEIGHT = os.path.join(WEIGHTS_ROOT, "clip", "ViT-B-32.pt")
REMOTECLIP_WEIGHT = os.path.join(WEIGHTS_ROOT, "RemoteCLIP", "RemoteCLIP-ViT-B-32.pt")

# Minimum cc size (pixels) to extract a patch — smaller regions are too noisy
MIN_CC_PIXELS = 16
# Padding around the bounding box (pixels) when extracting patch
PATCH_PAD = 8
# Minimum patch side length (pixels) after padding
MIN_PATCH_SIZE = 32


def parse_args():
    ap = argparse.ArgumentParser(description="RemoteCLIP vs CLIP directed-pair benchmark")
    ap.add_argument("config")
    ap.add_argument("--pairs", nargs="+", metavar="winner->runnerup",
                    help="Directed pairs to evaluate, e.g. 'grass->cropland'. "
                         "Multiple pairs allowed. If omitted, all pairs in the "
                         "top-margin low-margin pool are benchmarked.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--margin-thr", type=float, default=0.2,
                    help="Top1-top2 margin threshold defining low-margin pixels.")
    ap.add_argument("--delta", nargs="+", type=float,
                    default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.30],
                    help="Abstain threshold(s) delta to sweep: arbiter flips only "
                         "when sim(top2)-sim(top1) > delta.")
    ap.add_argument("--models", nargs="+", choices=["clip", "remoteclip", "both"],
                    default=["both"],
                    help="Which vision-language model(s) to benchmark.")
    ap.add_argument("--batch-clip", type=int, default=32,
                    help="Batch size for CLIP/RemoteCLIP inference.")
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


def load_clip_model(weight_path, device):
    """Load ViT-B-32 CLIP model from local weights."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained=weight_path,
        device=device,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenizer


def load_remoteclip_model(weight_path, device):
    """Load RemoteCLIP ViT-B-32 from local weights (same arch as CLIP, RS fine-tuned)."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained=weight_path,
        device=device,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenizer


def encode_texts(model, tokenizer, texts, device):
    """Encode a list of text strings, return L2-normalised features (N, D)."""
    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = F.normalize(feats, dim=-1)
    return feats.float()


def encode_images_batch(model, preprocess, images, device, batch_size=32):
    """
    Encode a list of PIL images in batches.
    Returns L2-normalised image features (N, D).
    """
    all_feats = []
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i: i + batch_size]
        tensors = torch.stack([preprocess(img) for img in batch_imgs]).to(device)
        with torch.no_grad():
            feats = model.encode_image(tensors)
            feats = F.normalize(feats, dim=-1)
        all_feats.append(feats.float().cpu())
    return torch.cat(all_feats, dim=0)


def extract_connected_components(binary_mask):
    """
    Simple 4-connectivity CC labelling on a 2D boolean numpy array.
    Returns label_map (H,W int32) and list of (row_coords, col_coords) per CC.
    """
    from scipy import ndimage as ndi
    label_map, n_labels = ndi.label(binary_mask)
    ccs = []
    for lb in range(1, n_labels + 1):
        ys, xs = np.where(label_map == lb)
        ccs.append((ys, xs))
    return label_map, ccs


def crop_patch(image_np, ys, xs, pad, min_size):
    """
    Crop the bounding box of (ys, xs) from image_np (H, W, 3) with padding.
    Returns PIL Image, or None if patch is degenerate.
    """
    H, W = image_np.shape[:2]
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(H, int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(W, int(xs.max()) + 1 + pad)
    if (y1 - y0) < min_size or (x1 - x0) < min_size:
        # Expand symmetrically to reach min_size
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        half = min_size // 2
        y0 = max(0, cy - half); y1 = min(H, cy + half)
        x0 = max(0, cx - half); x1 = min(W, cx + half)
    patch = image_np[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    return Image.fromarray(patch)


class PairBenchmark:
    """Accumulates per-patch results for one (winner, runnerup) directed pair."""

    def __init__(self, w_name, ru_name, class_names):
        self.w_name = w_name
        self.ru_name = ru_name
        self.class_names = class_names
        # Per-patch: (sim_top2_minus_top1, gt_category)
        # gt_category: 0=correct (GT==winner), 1=rescue (GT==runnerup), 2=other
        self.sim_deltas = []   # float: sim(top2) - sim(top1)
        self.categories = []   # int: 0/1/2

    def add(self, sim_delta, category):
        self.sim_deltas.append(float(sim_delta))
        self.categories.append(int(category))

    def evaluate(self, delta_thresholds, cm_base, gt_flat, pixel_indices, C, w_idx, ru_idx):
        """
        For each abstain threshold delta, compute metrics and simulate ΔmIoU.

        pixel_indices: list of 1D numpy arrays, one per patch, with the global
                       pixel offsets of that patch's low-margin pixels.
                       We use these to simulate the actual flip on the CM.
        """
        sim_deltas = np.array(self.sim_deltas)
        categories = np.array(self.categories)
        n_total = len(sim_deltas)

        results = []
        for delta in delta_thresholds:
            flip_mask = sim_deltas > delta
            n_flip = int(flip_mask.sum())
            n_abstain = n_total - n_flip

            flipped_cats = categories[flip_mask]
            n_flip_rescue = int((flipped_cats == 1).sum())
            n_flip_correct = int((flipped_cats == 0).sum())
            n_rescue_total = int((categories == 1).sum())

            flip_prec = n_flip_rescue / n_flip if n_flip > 0 else float("nan")
            flip_rec = n_flip_rescue / n_rescue_total if n_rescue_total > 0 else float("nan")
            abstain_rate = n_abstain / n_total if n_total > 0 else float("nan")

            # ΔmIoU simulation: collect pixel indices of flipped patches,
            # apply CM delta exactly as oracle_ceiling does
            flip_global_indices = []
            for idx_arr, do_flip in zip(pixel_indices, flip_mask):
                if do_flip:
                    flip_global_indices.append(idx_arr)

            delta_miou = float("nan")
            if len(flip_global_indices) > 0:
                all_flip = np.concatenate(flip_global_indices)
                cm = cm_base.astype(np.float64).copy()
                gt_flipped = gt_flat[all_flip]
                np.add.at(cm, (gt_flipped, w_idx), -1)
                np.add.at(cm, (gt_flipped, ru_idx), 1)
                base_miou = float(np.nanmean(iou_from_cm(cm_base)) * 100)
                new_miou = float(np.nanmean(iou_from_cm(cm)) * 100)
                delta_miou = new_miou - base_miou
            else:
                delta_miou = 0.0

            results.append({
                "delta": delta,
                "n_total": n_total,
                "n_flip": n_flip,
                "n_abstain": n_abstain,
                "n_flip_rescue": n_flip_rescue,
                "n_flip_correct": n_flip_correct,
                "flip_precision": flip_prec,
                "flip_recall": flip_rec,
                "abstain_rate": abstain_rate,
                "delta_miou": delta_miou,
            })
        return results


@torch.no_grad()
def main():
    args = parse_args()

    if not HAS_OPEN_CLIP:
        raise SystemExit("open_clip is required. Install: pip install open-clip-torch")

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

    # Parse directed pairs
    requested_pairs = []  # list of (w_idx, ru_idx)
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

    run_clip = args.models[0] in ("clip", "both") or "both" in args.models
    run_remoteclip = args.models[0] in ("remoteclip", "both") or "both" in args.models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load VLM models
    vlm_models = {}
    if run_clip:
        print(f"Loading CLIP from {CLIP_WEIGHT} ...")
        vlm_models["CLIP"] = load_clip_model(CLIP_WEIGHT, device)
    if run_remoteclip:
        print(f"Loading RemoteCLIP from {REMOTECLIP_WEIGHT} ...")
        vlm_models["RemoteCLIP"] = load_remoteclip_model(REMOTECLIP_WEIGHT, device)

    print(f"Config  : {args.config}")
    print(f"Classes ({C}): {class_names}")
    print(f"margin_thr: {args.margin_thr}  |  delta sweep: {args.delta}")
    print(f"Limit: {args.limit or 'all'}")

    # Pre-encode class name texts for all models
    text_feats = {}  # model_name -> (C, D) tensor on CPU
    for mname, (model, preprocess, tokenizer) in vlm_models.items():
        prompts = [f"a satellite image of {cn}" for cn in class_names]
        feats = encode_texts(model, tokenizer, prompts, device)
        text_feats[mname] = feats.cpu()
        print(f"  {mname} text features: {feats.shape}")

    # Build segmentation model
    seg_model = MODELS.build(cfg.model)
    seg_model.processor.model.eval()
    loader = Runner.build_dataloader(cfg.test_dataloader)

    # Baseline confusion matrix
    cm_base = np.zeros((C, C), dtype=np.int64)

    # Per-pair benchmark accumulators per model
    # benchmarks[model_name][(w,ru)] = PairBenchmark
    benchmarks = {mname: defaultdict(lambda: None) for mname in vlm_models}
    for mname in vlm_models:
        for pair in requested_pairs:
            benchmarks[mname][pair] = PairBenchmark(
                class_names[pair[0]], class_names[pair[1]], class_names
            )

    # pixel_indices_per_pair[(w,ru)] = list of 1D arrays (global pixel offsets per CC)
    # used for per-patch ΔmIoU simulation later (same for all models)
    pixel_indices_per_pair = defaultdict(list)

    gt_all_chunks = []
    pred_all_chunks = []
    pixel_offset = 0

    n_done = 0
    for batch in loader:
        out = seg_model.predict(batch["inputs"], batch["data_samples"])
        for ds_idx, ds in enumerate(out):
            logits = ds.seg_logits.data.float().cpu()          # (C, H, W)
            gt = ds.gt_sem_seg.data.squeeze(0).long().cpu()    # (H, W)
            valid = (gt >= 0) & (gt < C)
            H, W = gt.shape

            flat = logits.reshape(C, -1)                        # (C, P)
            vmask = valid.reshape(-1)                           # (P,)
            g_all = gt.reshape(-1)

            base_p = flat.argmax(0).clone()
            mx = flat.max(0).values
            thd_vec_t = torch.full((C,), prob_thd)
            gate = thd_vec_t[base_p]
            base_p[mx < gate] = bg_idx

            g_v = g_all[vmask]
            p_v = base_p[vmask]
            np.add.at(cm_base, (g_v.numpy(), p_v.numpy()), 1)

            gt_all_chunks.append(g_v.numpy().astype(np.int32))
            pred_all_chunks.append(p_v.numpy().astype(np.int32))

            # top-2 over valid pixels
            flat_v = flat[:, vmask]                             # (C, V)
            top2_v = flat_v.topk(2, dim=0)
            top1_idx = top2_v.indices[0]                        # (V,)
            top2_idx = top2_v.indices[1]                        # (V,)
            margin_v = top2_v.values[0] - top2_v.values[1]     # (V,)

            # strict fg↔fg low-margin candidate
            not_backfilled = (p_v == top1_idx) & (p_v != bg_idx) & (top2_idx != bg_idx)
            lm_mask = (margin_v < args.margin_thr) & not_backfilled  # (V,) bool

            # Load image for patch extraction if any requested pairs have lm pixels
            image_np = None
            if lm_mask.any() and len(requested_pairs) > 0:
                img_path = ds.img_path if hasattr(ds, "img_path") else None
                if img_path is not None and os.path.exists(img_path):
                    img = Image.open(img_path).convert("RGB")
                    image_np = np.array(img)

            # Valid pixel coordinates in the original 2D image (for CC extraction)
            # Build mapping from valid-pixel index -> (row, col)
            valid_ys, valid_xs = torch.where(valid)  # row/col of valid pixels
            valid_ys = valid_ys.numpy()
            valid_xs = valid_xs.numpy()

            for (w_idx, ru_idx) in requested_pairs:
                # pixels belonging to this directed pair
                pair_lm = lm_mask & (top1_idx == w_idx) & (top2_idx == ru_idx)
                if not pair_lm.any():
                    continue
                if image_np is None:
                    continue

                pair_lm_np = pair_lm.numpy()
                local_indices = np.where(pair_lm_np)[0]     # indices into valid pixels

                # GT category per lm pixel of this pair
                g_pair = g_v[pair_lm].numpy()
                cat = np.where(g_pair == w_idx, 0,
                               np.where(g_pair == ru_idx, 1, 2))

                # Get 2D coords of lm pixels for this pair
                pair_ys = valid_ys[local_indices]
                pair_xs = valid_xs[local_indices]

                # Build a 2D binary mask for CC labelling
                pair_2d = np.zeros((H, W), dtype=bool)
                pair_2d[pair_ys, pair_xs] = True

                _, ccs = extract_connected_components(pair_2d)

                # For each CC: extract patch, run VLM, accumulate
                for ys_cc, xs_cc in ccs:
                    if len(ys_cc) < MIN_CC_PIXELS:
                        continue

                    patch_pil = crop_patch(image_np, ys_cc, xs_cc, PATCH_PAD, MIN_PATCH_SIZE)
                    if patch_pil is None:
                        continue

                    # Determine majority category for this CC
                    # Map CC pixels to their position in local_indices
                    cc_pixel_set = set(zip(ys_cc.tolist(), xs_cc.tolist()))
                    cc_local = [
                        li for li, (py, px) in enumerate(zip(pair_ys, pair_xs))
                        if (int(py), int(px)) in cc_pixel_set
                    ]
                    if len(cc_local) == 0:
                        continue

                    cc_cats = cat[cc_local]
                    # majority category (0=correct, 1=rescue, 2=other)
                    majority_cat = int(np.bincount(cc_cats, minlength=3).argmax())

                    # Global pixel offsets for this CC's pixels
                    cc_global = (local_indices[cc_local] + pixel_offset).astype(np.int64)

                    # Store pixel indices for ΔmIoU simulation (model-independent)
                    pixel_indices_per_pair[(w_idx, ru_idx)].append(cc_global)

                    # Run each VLM on this patch
                    for mname, (model, preprocess, tokenizer) in vlm_models.items():
                        tf = text_feats[mname].to(device)           # (C, D)
                        img_tensor = preprocess(patch_pil).unsqueeze(0).to(device)
                        with torch.no_grad():
                            img_feat = model.encode_image(img_tensor)
                            img_feat = F.normalize(img_feat, dim=-1).float()  # (1, D)

                        # Similarity to winner and runnerup text
                        sim = (img_feat @ tf.T).squeeze(0)          # (C,)
                        sim_w = float(sim[w_idx].cpu())
                        sim_ru = float(sim[ru_idx].cpu())
                        sim_delta = sim_ru - sim_w  # positive => model prefers runnerup

                        bench = benchmarks[mname][(w_idx, ru_idx)]
                        if bench is None:
                            benchmarks[mname][(w_idx, ru_idx)] = PairBenchmark(
                                class_names[w_idx], class_names[ru_idx], class_names
                            )
                            bench = benchmarks[mname][(w_idx, ru_idx)]
                        bench.add(sim_delta, majority_cat)

            pixel_offset += int(vmask.sum())

        n_done += len(out)
        if n_done % 20 == 0:
            print(f"  processed {n_done} images ...")
        if args.limit and n_done >= args.limit:
            break

    print(f"\nTotal images: {n_done}")

    gt_flat = np.concatenate(gt_all_chunks) if gt_all_chunks else np.array([], np.int32)
    pred_flat = np.concatenate(pred_all_chunks) if pred_all_chunks else np.array([], np.int32)
    base_miou = float(np.nanmean(iou_from_cm(cm_base)) * 100)
    print(f"Baseline mIoU: {base_miou:.2f}%\n")

    # =========================================================================
    # Report
    # =========================================================================
    delta_thrs = sorted(args.delta)

    for (w_idx, ru_idx) in requested_pairs:
        wn = class_names[w_idx]
        run = class_names[ru_idx]
        pair_key = (w_idx, ru_idx)
        pix_idx_list = pixel_indices_per_pair.get(pair_key, [])

        print("=" * 80)
        print(f"Pair: {wn} -> {run}   (margin_thr={args.margin_thr})")
        print(f"  CC patches collected: {len(pix_idx_list)}")

        for mname in vlm_models:
            bench = benchmarks[mname].get(pair_key)
            if bench is None or len(bench.sim_deltas) == 0:
                print(f"\n  [{mname}] no patches collected for this pair.")
                continue

            n_patches = len(bench.sim_deltas)
            n_rescue = int(sum(c == 1 for c in bench.categories))
            n_correct = int(sum(c == 0 for c in bench.categories))
            n_other = int(sum(c == 2 for c in bench.categories))
            print(f"\n  [{mname}] {n_patches} CC patches: "
                  f"rescue={n_rescue} correct={n_correct} other={n_other}")

            results = bench.evaluate(
                delta_thrs, cm_base, gt_flat, pix_idx_list, C, w_idx, ru_idx
            )

            print(f"  {'delta':>7} {'n_flip':>7} {'abs_rate':>9} "
                  f"{'flip_prec':>10} {'flip_rec':>10} {'ΔmIoU':>8}")
            for r in results:
                def _f(v, w=8, p=2):
                    return f"{'nan':>{w}}" if (isinstance(v, float) and v != v) else f"{v:{w}.{p}f}"
                print(f"  {r['delta']:7.2f} {r['n_flip']:7d} {r['abstain_rate']:9.1%} "
                      f"{_f(r['flip_precision'], 10, 3)} {_f(r['flip_recall'], 10, 3)} "
                      f"{_f(r['delta_miou'], 8)}")

            # best operating point: highest ΔmIoU
            best = max(results, key=lambda r: r["delta_miou"] if r["delta_miou"] == r["delta_miou"] else -99)
            print(f"  => best delta={best['delta']:.2f}: "
                  f"ΔmIoU={best['delta_miou']:+.3f}  "
                  f"prec={best['flip_precision']:.3f}  "
                  f"rec={best['flip_recall']:.3f}  "
                  f"abstain={best['abstain_rate']:.1%}")

    print("=" * 80)

    # Cross-model summary table
    if len(vlm_models) > 1 and len(requested_pairs) > 0:
        print("\nCross-model summary (best delta per pair):")
        print(f"  {'pair':<28}", end="")
        for mname in vlm_models:
            print(f"  {mname+' ΔmIoU':>14}  {mname+' prec':>12}", end="")
        print()

        for (w_idx, ru_idx) in requested_pairs:
            wn = class_names[w_idx]
            run = class_names[ru_idx]
            pair_key = (w_idx, ru_idx)
            pix_idx_list = pixel_indices_per_pair.get(pair_key, [])
            print(f"  {wn+' -> '+run:<28}", end="")

            for mname in vlm_models:
                bench = benchmarks[mname].get(pair_key)
                if bench is None or len(bench.sim_deltas) == 0:
                    print(f"  {'n/a':>14}  {'n/a':>12}", end="")
                    continue
                results = bench.evaluate(
                    delta_thrs, cm_base, gt_flat, pix_idx_list, C, w_idx, ru_idx
                )
                best = max(results, key=lambda r: r["delta_miou"] if r["delta_miou"] == r["delta_miou"] else -99)
                print(f"  {best['delta_miou']:>+14.3f}  {best['flip_precision']:>12.3f}", end="")
            print()

    print("\nDone.")


if __name__ == "__main__":
    main()
