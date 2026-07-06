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
                         "If omitted, all fg-fg pairs with n_cc >= min-n-cc are used.")
    ap.add_argument("--min-n-cc", type=int, default=200,
                    help="Minimum CC count for a pair to be included in auto-discovery "
                         "(ignored when --pairs is specified).")
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
    ap.add_argument("--patch-mode", choices=["raw", "masked_mean", "masked_gray"],
                    default="raw",
                    help="How to construct the patch sent to VLM: "
                         "raw=bbox crop (C0), "
                         "masked_mean=CC pixels kept, outside filled with CC mean (C1), "
                         "masked_gray=CC pixels kept, outside filled with gray 128 (C1 variant).")
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
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        half = min_size // 2
        y0 = max(0, cy - half); y1 = min(H, cy + half)
        x0 = max(0, cx - half); x1 = min(W, cx + half)
    patch = image_np[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    return Image.fromarray(patch), (y0, x0)


def masked_crop_patch(image_np, ys, xs, pad, min_size, fill="mean"):
    """
    Crop the bounding box of the CC, then fill pixels outside the CC mask
    with either the patch mean (fill='mean') or a neutral gray (fill='gray').
    Returns PIL Image, or None if patch is degenerate.
    """
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
    patch = image_np[y0:y1, x0:x1].copy()
    if patch.size == 0:
        return None

    # Build CC mask in patch coordinates
    cc_mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    local_ys = ys - y0
    local_xs = xs - x0
    valid = (local_ys >= 0) & (local_ys < (y1 - y0)) & \
            (local_xs >= 0) & (local_xs < (x1 - x0))
    cc_mask[local_ys[valid], local_xs[valid]] = True

    if fill == "mean":
        fill_val = patch[cc_mask].mean(axis=0).astype(np.uint8) if cc_mask.any() else np.array([128, 128, 128], dtype=np.uint8)
    else:
        fill_val = np.array([128, 128, 128], dtype=np.uint8)

    patch[~cc_mask] = fill_val
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

    # Parse directed pairs (manual override)
    manual_pairs = []
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
            manual_pairs.append((name2idx[wn], name2idx[run]))

    auto_discover = len(manual_pairs) == 0
    if auto_discover:
        print(f"Auto-discover mode: will use all fg-fg pairs with n_cc >= {args.min_n_cc}")
    else:
        print(f"Manual pairs: {[class_names[w]+'->'+class_names[r] for w,r in manual_pairs]}")

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
    print(f"patch_mode: {args.patch_mode}")
    print(f"Limit: {args.limit or 'all'}")

    # Pre-encode class name texts for all models, kept ON GPU (encoded once,
    # reused for every patch batch instead of being re-copied to device each time)
    text_feats = {}  # model_name -> (C, D) tensor on device
    for mname, (model, preprocess, tokenizer) in vlm_models.items():
        prompts = [f"a satellite image of {cn}" for cn in class_names]
        feats = encode_texts(model, tokenizer, prompts, device)
        text_feats[mname] = feats  # stays on device
        print(f"  {mname} text features: {feats.shape}")

    # Build segmentation model
    seg_model = MODELS.build(cfg.model)
    seg_model.processor.model.eval()
    loader = Runner.build_dataloader(cfg.test_dataloader)

    # Baseline confusion matrix
    cm_base = np.zeros((C, C), dtype=np.int64)

    # Per-pair accumulators (lazily created for all encountered fg-fg pairs)
    # benchmarks[model_name][(w,ru)] = PairBenchmark
    benchmarks = {mname: {} for mname in vlm_models}
    # pixel_indices_per_pair[(w,ru)] = list of 1D arrays (global pixel offsets per CC)
    pixel_indices_per_pair = defaultdict(list)
    # n_cc counter for auto-discovery
    n_cc_counter = defaultdict(int)

    gt_all_chunks = []
    pixel_offset = 0

    # ---- Patch batching buffer ----------------------------------------------
    # Instead of forwarding each CC patch through the VLM(s) one-at-a-time
    # (batch=1, GPU idle waiting on CPU crops), we accumulate patches + their
    # metadata and flush them through the model(s) in batches of --batch-clip.
    # This is a pure throughput change: each patch still goes through the same
    # model producing the same sim_delta, only grouped for the forward pass.
    patch_buf = []          # list of PIL patches
    meta_buf = []           # list of (w_idx, ru_idx, majority_cat)
    clip_bs = max(1, int(args.batch_clip))

    def flush_patch_buffer():
        if not patch_buf:
            return
        for mname, (model, preprocess, tokenizer) in vlm_models.items():
            tf = text_feats[mname]  # already on device
            # Encode all buffered patches for this model in mini-batches
            for i in range(0, len(patch_buf), clip_bs):
                imgs = patch_buf[i: i + clip_bs]
                tensors = torch.stack([preprocess(im) for im in imgs]).to(device)
                with torch.no_grad():
                    feats = F.normalize(model.encode_image(tensors), dim=-1).float()
                sims = feats @ tf.T  # (b, C)
                for j, (w_idx, ru_idx, majority_cat) in enumerate(
                        meta_buf[i: i + clip_bs]):
                    sim_delta = float(sims[j, ru_idx] - sims[j, w_idx])
                    if (w_idx, ru_idx) not in benchmarks[mname]:
                        benchmarks[mname][(w_idx, ru_idx)] = PairBenchmark(
                            class_names[w_idx], class_names[ru_idx], class_names)
                    benchmarks[mname][(w_idx, ru_idx)].add(sim_delta, majority_cat)
        patch_buf.clear()
        meta_buf.clear()

    n_done = 0
    for batch in loader:
        out = seg_model.predict(batch["inputs"], batch["data_samples"])
        for ds in out:
            logits = ds.seg_logits.data.float().cpu()
            gt = ds.gt_sem_seg.data.squeeze(0).long().cpu()
            valid = (gt >= 0) & (gt < C)
            H, W = gt.shape

            flat = logits.reshape(C, -1)
            vmask = valid.reshape(-1)
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

            flat_v = flat[:, vmask]
            top2_v = flat_v.topk(2, dim=0)
            top1_idx = top2_v.indices[0]
            top2_idx = top2_v.indices[1]
            margin_v = top2_v.values[0] - top2_v.values[1]

            not_backfilled = (p_v == top1_idx) & (p_v != bg_idx) & (top2_idx != bg_idx)
            lm_mask = (margin_v < args.margin_thr) & not_backfilled

            if not lm_mask.any():
                pixel_offset += int(vmask.sum())
                continue

            image_np = None
            img_path = ds.img_path if hasattr(ds, "img_path") else None
            if img_path is not None and os.path.exists(img_path):
                image_np = np.array(Image.open(img_path).convert("RGB"))

            valid_ys, valid_xs = torch.where(valid)
            valid_ys = valid_ys.numpy()
            valid_xs = valid_xs.numpy()

            top1_np = top1_idx.numpy()
            top2_np = top2_idx.numpy()
            lm_np = lm_mask.numpy()
            g_np = g_v.numpy()

            # Determine which pairs to process this image
            lm_indices = np.where(lm_np)[0]
            w_arr = top1_np[lm_indices]
            ru_arr = top2_np[lm_indices]
            unique_pairs = set(zip(w_arr.tolist(), ru_arr.tolist()))
            unique_pairs = {(w, ru) for w, ru in unique_pairs
                            if w != bg_idx and ru != bg_idx and w != ru}

            if not auto_discover:
                unique_pairs = unique_pairs & set(manual_pairs)

            for (w_idx, ru_idx) in unique_pairs:
                pair_lm = lm_np & (top1_np == w_idx) & (top2_np == ru_idx)
                if not pair_lm.any():
                    continue

                local_indices = np.where(pair_lm)[0]
                pair_ys = valid_ys[local_indices]
                pair_xs = valid_xs[local_indices]
                g_pair = g_np[local_indices]
                cat = np.where(g_pair == w_idx, 0,
                               np.where(g_pair == ru_idx, 1, 2))

                pair_2d = np.zeros((H, W), dtype=bool)
                pair_2d[pair_ys, pair_xs] = True
                _, ccs = extract_connected_components(pair_2d)

                for ys_cc, xs_cc in ccs:
                    if len(ys_cc) < MIN_CC_PIXELS:
                        continue

                    n_cc_counter[(w_idx, ru_idx)] += 1

                    if image_np is None:
                        continue

                    if args.patch_mode == "raw":
                        result = crop_patch(image_np, ys_cc, xs_cc, PATCH_PAD, MIN_PATCH_SIZE)
                        patch_pil = result[0] if result is not None else None
                    elif args.patch_mode == "masked_mean":
                        patch_pil = masked_crop_patch(image_np, ys_cc, xs_cc, PATCH_PAD, MIN_PATCH_SIZE, fill="mean")
                    else:
                        patch_pil = masked_crop_patch(image_np, ys_cc, xs_cc, PATCH_PAD, MIN_PATCH_SIZE, fill="gray")

                    if patch_pil is None:
                        continue

                    cc_pixel_set = set(zip(ys_cc.tolist(), xs_cc.tolist()))
                    cc_local = [
                        li for li, (py, px) in enumerate(zip(pair_ys, pair_xs))
                        if (int(py), int(px)) in cc_pixel_set
                    ]
                    if len(cc_local) == 0:
                        continue

                    cc_cats = cat[cc_local]
                    majority_cat = int(np.bincount(cc_cats, minlength=3).argmax())
                    cc_global = (local_indices[cc_local] + pixel_offset).astype(np.int64)
                    pixel_indices_per_pair[(w_idx, ru_idx)].append(cc_global)

                    # Queue this patch; it will be forwarded in a batch below.
                    patch_buf.append(patch_pil)
                    meta_buf.append((w_idx, ru_idx, majority_cat))
                    if len(patch_buf) >= clip_bs:
                        flush_patch_buffer()

            pixel_offset += int(vmask.sum())

        n_done += len(out)
        if n_done % 20 == 0:
            print(f"  processed {n_done} images ...")
        if args.limit and n_done >= args.limit:
            break

    # Forward any patches still buffered from the last partial batch.
    flush_patch_buffer()

    print(f"\nTotal images: {n_done}")

    gt_flat = np.concatenate(gt_all_chunks) if gt_all_chunks else np.array([], np.int32)
    base_miou = float(np.nanmean(iou_from_cm(cm_base)) * 100)
    print(f"Baseline mIoU: {base_miou:.2f}%\n")

    # Determine final pair list
    if auto_discover:
        active_pairs = sorted(
            [(w, ru) for (w, ru), cnt in n_cc_counter.items() if cnt >= args.min_n_cc],
            key=lambda p: n_cc_counter[p], reverse=True
        )
        print(f"Auto-discovered {len(active_pairs)} pairs (n_cc >= {args.min_n_cc}):")
        for (w, ru) in active_pairs:
            print(f"  {class_names[w]}->{class_names[ru]}  n_cc={n_cc_counter[(w,ru)]}")
        print()
    else:
        active_pairs = manual_pairs

    # =========================================================================
    # Report
    # =========================================================================
    delta_thrs = sorted(args.delta)

    def _f(v, w=8, p=2):
        return f"{'nan':>{w}}" if (isinstance(v, float) and v != v) else f"{v:{w}.{p}f}"

    def _always_flip_miou(pix_idx_list, w_idx, ru_idx):
        if len(pix_idx_list) == 0:
            return 0.0
        all_pix = np.concatenate(pix_idx_list)
        cm = cm_base.astype(np.float64).copy()
        gt_flipped = gt_flat[all_pix]
        np.add.at(cm, (gt_flipped, w_idx), -1)
        np.add.at(cm, (gt_flipped, ru_idx), 1)
        return float(np.nanmean(iou_from_cm(cm)) * 100) - base_miou

    always_flip_dmiou = {}
    for (w_idx, ru_idx) in active_pairs:
        pix_idx_list = pixel_indices_per_pair.get((w_idx, ru_idx), [])
        always_flip_dmiou[(w_idx, ru_idx)] = _always_flip_miou(pix_idx_list, w_idx, ru_idx)

    for (w_idx, ru_idx) in active_pairs:
        wn = class_names[w_idx]
        run = class_names[ru_idx]
        pair_key = (w_idx, ru_idx)
        pix_idx_list = pixel_indices_per_pair.get(pair_key, [])
        af_dmiou = always_flip_dmiou[pair_key]

        print("=" * 80)
        print(f"Pair: {wn} -> {run}   (margin_thr={args.margin_thr}  n_cc={n_cc_counter[pair_key]})")
        print(f"  CC patches collected: {len(pix_idx_list)}")
        print(f"  always-flip ΔmIoU: {af_dmiou:+.3f}")

        for mname in vlm_models:
            bench = benchmarks[mname].get(pair_key)
            if bench is None or len(bench.sim_deltas) == 0:
                print(f"\n  [{mname}] no patches collected.")
                continue

            n_patches = len(bench.sim_deltas)
            n_rescue = int(sum(c == 1 for c in bench.categories))
            n_correct = int(sum(c == 0 for c in bench.categories))
            n_other = int(sum(c == 2 for c in bench.categories))
            sim_arr = np.array(bench.sim_deltas)
            print(f"\n  [{mname}] {n_patches} CC patches: "
                  f"rescue={n_rescue} correct={n_correct} other={n_other}")
            print(f"    sim_delta: mean={sim_arr.mean():+.4f}  std={sim_arr.std():.4f}  "
                  f"P(>0)={float((sim_arr > 0).mean()):.1%}")

            results = bench.evaluate(delta_thrs, cm_base, gt_flat, pix_idx_list, C, w_idx, ru_idx)

            print(f"  {'delta':>7} {'n_flip':>7} {'abs_rate':>9} "
                  f"{'flip_prec':>10} {'flip_rec':>10} {'ΔmIoU':>8} {'vs AF':>8}")
            for r in results:
                vs_af = r["delta_miou"] - af_dmiou if r["delta_miou"] == r["delta_miou"] else float("nan")
                marker = " ★" if (vs_af == vs_af and vs_af > 0) else ""
                print(f"  {r['delta']:7.2f} {r['n_flip']:7d} {r['abstain_rate']:9.1%} "
                      f"{_f(r['flip_precision'], 10, 3)} {_f(r['flip_recall'], 10, 3)} "
                      f"{_f(r['delta_miou'], 8)} {_f(vs_af, 8)}{marker}")

            best = max(results, key=lambda r: r["delta_miou"] if r["delta_miou"] == r["delta_miou"] else -99)
            vs_af_best = best["delta_miou"] - af_dmiou
            verdict = "VLM beats AF ★" if vs_af_best > 0 else "VLM <= AF"
            print(f"  => best delta={best['delta']:.2f}: ΔmIoU={best['delta_miou']:+.3f}  "
                  f"vs AF={vs_af_best:+.3f}  ({verdict})")

    print("=" * 80)

    # Cross-model summary
    if active_pairs:
        print("\nSummary (best delta per pair):")
        header = f"  {'pair':<30}  {'n_cc':>6}  {'AF ΔmIoU':>10}"
        for mname in vlm_models:
            header += f"  {mname+' ΔmIoU':>14}  {'vs AF':>8}"
        print(header)

        for (w_idx, ru_idx) in active_pairs:
            wn = class_names[w_idx]
            run = class_names[ru_idx]
            pair_key = (w_idx, ru_idx)
            pix_idx_list = pixel_indices_per_pair.get(pair_key, [])
            af_dmiou = always_flip_dmiou[pair_key]
            nc = n_cc_counter[pair_key]
            row = f"  {wn+'->'+run:<30}  {nc:>6}  {af_dmiou:>+10.3f}"

            for mname in vlm_models:
                bench = benchmarks[mname].get(pair_key)
                if bench is None or len(bench.sim_deltas) == 0:
                    row += f"  {'n/a':>14}  {'n/a':>8}"
                    continue
                results = bench.evaluate(delta_thrs, cm_base, gt_flat, pix_idx_list, C, w_idx, ru_idx)
                best = max(results, key=lambda r: r["delta_miou"] if r["delta_miou"] == r["delta_miou"] else -99)
                vs_af = best["delta_miou"] - af_dmiou
                row += f"  {best['delta_miou']:>+14.3f}  {vs_af:>+8.3f}"
            print(row)

    # =========================================================================
    # Combined mIoU: all pairs flipped simultaneously on one CM
    # =========================================================================
    if active_pairs:
        print("\n" + "=" * 80)
        print("Combined mIoU (all pairs flipped simultaneously):")
        print(f"  Baseline: {base_miou:.4f}%")

        # AF combined: flip all pairs unconditionally
        cm_af = cm_base.astype(np.float64).copy()
        for (w_idx, ru_idx) in active_pairs:
            pix_idx_list = pixel_indices_per_pair.get((w_idx, ru_idx), [])
            if len(pix_idx_list) == 0:
                continue
            all_pix = np.concatenate(pix_idx_list)
            gt_flipped = gt_flat[all_pix]
            np.add.at(cm_af, (gt_flipped, w_idx), -1)
            np.add.at(cm_af, (gt_flipped, ru_idx), 1)
        af_combined_miou = float(np.nanmean(iou_from_cm(cm_af)) * 100)
        print(f"  AF (all pairs): {af_combined_miou:.4f}%  (Δ={af_combined_miou - base_miou:+.4f})")

        # VLM combined: for each model and each delta, flip only when sim_delta > delta
        for mname in vlm_models:
            for delta in delta_thrs:
                cm_vlm = cm_base.astype(np.float64).copy()
                for (w_idx, ru_idx) in active_pairs:
                    bench = benchmarks[mname].get((w_idx, ru_idx))
                    pix_idx_list = pixel_indices_per_pair.get((w_idx, ru_idx), [])
                    if bench is None or len(bench.sim_deltas) == 0 or len(pix_idx_list) == 0:
                        continue
                    sim_deltas_arr = np.array(bench.sim_deltas)
                    flip_mask = sim_deltas_arr > delta
                    flip_pix = [idx for idx, do_flip in zip(pix_idx_list, flip_mask) if do_flip]
                    if len(flip_pix) == 0:
                        continue
                    all_pix = np.concatenate(flip_pix)
                    gt_flipped = gt_flat[all_pix]
                    np.add.at(cm_vlm, (gt_flipped, w_idx), -1)
                    np.add.at(cm_vlm, (gt_flipped, ru_idx), 1)
                vlm_miou = float(np.nanmean(iou_from_cm(cm_vlm)) * 100)
                print(f"  {mname} delta={delta:.2f}: {vlm_miou:.4f}%  "
                      f"(Δ={vlm_miou - base_miou:+.4f}  vs AF={vlm_miou - af_combined_miou:+.4f})")
        print("=" * 80)

    print("\nDone.")


if __name__ == "__main__":
    main()
