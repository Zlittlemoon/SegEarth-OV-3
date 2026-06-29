"""
Phase-1A — Release-band Raw Multi-Head Signal Separability (collection script).

Goal (label-free question, validated against the Phase-0 oracle target):
    Inside the CAFR v2 release-band, do SegEarth-OV3's RAW multi-head signals
    (semantic-only, instance-raw, fused-before-presence, class-bg gap, ...)
    contain enough information to predict whether a pixel is GT=c / GT=bg /
    GT=other?  If yes, a Multi-Head Distribution Selector is buildable; if not,
    the internal scores are insufficient and an external prior (RemoteCLIP / VLM
    / light calibration) is required.

This script ONLY collects features; it does not modify the main pipeline and it
does not produce a segmentation.  Analysis (AUC / Spearman / distributions) is a
separate step that reads the dumped .npz files.

Sample set — FROZEN to the CAFR v2 failure scene (must match cafr_diagnose.py):
    B_c = { raw_top1_post = c,  λτ ≤ fused_after_presence < τ_global }
  where raw_top1_post / fused_after_presence are the argmax / max over classes of
  the FINAL fused-and-presence-modulated seg_logits — exactly the raw_top1 / max_sc
  that cafr_diagnose.py derives from model.predict().  Raw signals are FEATURES on
  these pixels; they never redefine the sample set.

Per-pixel features dumped:
    semantic_pixel_before_presence              outputs["semantic_seg"].sigmoid()
    instance_raw_pixel_before_confidence_presence
                                                max_Q(mask.sigmoid * logit.sigmoid)
                                                (NO confidence keep, NO presence)
    instance_original_prefinal_pixel            v2 instance path: max over kept
                                                (obj_prob>thr) of mask*obj_prob,
                                                obj_prob = sigmoid(logit)*presence
                                                (contains presence once -> presence^2
                                                 after the final *presence below)
    fused_raw_before_presence                   max(semantic, instance_raw)
    fused_after_presence                        v2 final: max(inst_orig, sem)*presence
                                                (== cafr_diagnose max_sc at channel c)
    presence_score                              outputs["presence_logit_dec"].sigmoid()
                                                (image-class scalar; spatial mean under
                                                 slide.  NOT a standalone pixel predictor)
    bg_score_post                               fused_after_presence at background channel
    class_minus_bg_gap_post                     fused_after_presence(c) - bg_score_post

Labels per pixel:  is_c / is_bg / is_other  (from GT).

Sampling:  per-image-class cap (reservoir within image) + per-class global cap
           (reservoir across images).  Keeps statistics stable without exploding.

Output:  one .npz per (dataset, class):
           rawdiag_<dataset>_releaseband_<class>.npz

First run = SMOKE TEST (small --limit), check:
  1. per-class release-band counts track cafr_diagnose's n_band trend
  2. instance_raw vs instance_original_prefinal really differ (presence^2 split)
  3. output file sizes are controllable
then run full val.

Usage (smoke test):
    python verify/raw_score_diagnose.py configs/cfg_loveda.py \\
        --dump-release-band \\
        --classes water,building,road,forest,agricultural \\
        --limit 10 --per-image-class-cap 2000 --global-class-cap 100000 \\
        --out-dir verify/rawdiag

    python verify/raw_score_diagnose.py configs/cfg_openearthmap.py \\
        --dump-release-band \\
        --classes grass,pavement,building,road,bareland \\
        --limit 10 --per-image-class-cap 2000 --global-class-cap 100000 \\
        --out-dir verify/rawdiag

Usage (full val): drop --limit, raise caps (e.g. --global-class-cap 2000000).
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmseg.registry import MODELS

import mmseg.datasets
import segearthov3_segmentor_merge   # registers SegEarthOV3Segmentation
import custom_datasets

from sam3.model.data_misc import FindStage, interpolate


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Phase-1A release-band raw multi-head signal collection")
    ap.add_argument("config")
    ap.add_argument("--dump-release-band", action="store_true",
                    help="collect the release-band pool (Phase 1A). Required in v1; "
                         "this is the only pool implemented so far.")
    ap.add_argument("--dump-low-margin", action="store_true",
                    help="(Phase 1B, NOT implemented here) collect the top1/top2 "
                         "low-margin pool. Refused in this version on purpose — the "
                         "two pools have different labels and must stay separate.")
    ap.add_argument("--classes", type=str, required=True,
                    help="comma-separated class names to collect (e.g. "
                         "'water,building,road,forest,agricultural').")
    ap.add_argument("--limit", type=int, default=None,
                    help="max images to process (default: all). Use a small value "
                         "(e.g. 10) for the smoke test.")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.2,
                    help="λ for the release band lower edge λτ (default 0.2, "
                         "matches cafr_diagnose.py).")
    ap.add_argument("--per-image-class-cap", type=int, default=3000,
                    help="max pixels sampled per image per class (default 3000).")
    ap.add_argument("--global-class-cap", type=int, default=1_000_000,
                    help="max pixels retained per class across all images via "
                         "reservoir sampling (default 1,000,000).")
    ap.add_argument("--out-dir", type=str, default="verify/rawdiag",
                    help="directory for the dumped .npz files (created if absent).")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for reservoir sampling (default 0).")
    ap.add_argument("--cfg-options", nargs="+", action=DictAction, default=None)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Raw multi-head capture (replicates the v2 batch fusion, capturing every head)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _grounding_heads_crop(model, image):
    """Run one batched grounding forward on a single PIL image/crop and return a
    dict of per-query full-resolution head maps (shape (nq, h, w)) plus the
    per-query presence vector.  Mirrors _inference_single_view_batch's fusion
    exactly for the v2 paths, and adds the raw (de-presence / de-confidence)
    instance path.

    NOTE on instance_raw: max over ALL queries of mask.sigmoid()*logit.sigmoid()
    is taken at the native mask resolution and then interpolated, whereas the v2
    instance_original path interpolates each kept instance first.  This tiny
    order difference is acceptable for a raw diagnostic signal and keeps the
    all-query max affordable.
    """
    w, h = image.size
    n = model.num_queries
    thr = model.processor.confidence_threshold
    device = model.device

    out = {}
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = model.processor.set_image(image)
        text_outputs = model.processor.model.backbone.forward_text(
            model.query_words, device=device)
        state["backbone_out"].update(text_outputs)
        geometric_prompt = model.processor.model._get_dummy_prompt(num_prompts=n)
        find_stage = FindStage(
            img_ids=torch.zeros(n, device=device, dtype=torch.long),
            text_ids=torch.arange(n, device=device, dtype=torch.long),
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=None, input_points_mask=None,
        )
        outputs = model.processor.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=find_stage,
            geometric_prompt=geometric_prompt,
            find_target=None,
        )

        pred_logits = outputs["pred_logits"]                 # (N, Q, 1)
        pred_masks = outputs["pred_masks"]                   # (N, Q, h', w')
        presence = outputs["presence_logit_dec"].sigmoid()   # (N, 1)
        obj_probs = (pred_logits.sigmoid() * presence.unsqueeze(1)).squeeze(-1)  # (N,Q)
        logit_probs = pred_logits.sigmoid().squeeze(-1)      # (N,Q) raw object prob

        # semantic head (presence-free), interpolated + sigmoid -> (N, h, w)
        sem_all = interpolate(
            outputs["semantic_seg"], (h, w),
            mode="bilinear", align_corners=False).sigmoid().squeeze(1)

        semantic_map = torch.zeros((n, h, w), device=device)
        inst_raw_map = torch.zeros((n, h, w), device=device)
        inst_orig_map = torch.zeros((n, h, w), device=device)

        mask_sig = pred_masks.sigmoid()                      # (N, Q, h', w')
        for qi in range(n):
            semantic_map[qi] = sem_all[qi]

            # ---- instance_raw: no keep, no presence ----
            # max over Q of (mask * raw_object_prob) at native res, then interpolate
            raw_inst_q = mask_sig[qi] * logit_probs[qi][:, None, None]   # (Q, h', w')
            raw_inst = raw_inst_q.max(0)[0]                              # (h', w')
            inst_raw_map[qi] = interpolate(
                raw_inst[None, None], (h, w),
                mode="bilinear", align_corners=False).squeeze()

            # ---- instance_original (v2 path): keep>thr, presence-modulated ----
            keep = obj_probs[qi] > thr                                  # (Q,)
            if keep.any():
                masks_k = interpolate(
                    pred_masks[qi][keep].unsqueeze(1), (h, w),
                    mode="bilinear", align_corners=False).sigmoid().squeeze(1)  # (K,h,w)
                scores_k = obj_probs[qi][keep]                          # (K,)
                inst_orig_map[qi] = (masks_k * scores_k[:, None, None]).max(0)[0]

        # ---- fused maps ----
        fused_raw = torch.maximum(semantic_map, inst_raw_map)           # before presence
        if model.use_sem_seg:
            inst_or_sem = torch.maximum(inst_orig_map, semantic_map)
        else:
            inst_or_sem = inst_orig_map
        presence_vec = presence.squeeze(-1)                            # (N,)
        if model.use_presence_score:
            fused_after = inst_or_sem * presence_vec[:, None, None]
        else:
            fused_after = inst_or_sem

        out["semantic"] = semantic_map.float().cpu()
        out["instance_raw"] = inst_raw_map.float().cpu()
        out["instance_orig"] = inst_orig_map.float().cpu()
        out["fused_raw"] = fused_raw.float().cpu()
        out["fused_after"] = fused_after.float().cpu()
        out["presence_vec"] = presence_vec.float().cpu()              # (nq,)
    return out


@torch.no_grad()
def capture_full_maps(model, image, ori_hw):
    """Return per-query head maps at ori_hw (the GT resolution), handling both the
    whole-image path (LoveDA) and the slide-window path (OEM) with the same
    accumulate / count normalisation the segmentor uses for seg_logits.

    Returns dict of (nq, H, W) cpu tensors for the spatial heads, and a (nq,)
    presence vector (spatially averaged over crops, weighted by crop coverage).
    """
    nq = model.num_queries
    W_img, H_img = image.size
    H_o, W_o = ori_hw

    spatial_keys = ["semantic", "instance_raw", "instance_orig", "fused_raw", "fused_after"]

    use_slide = (model.slide_crop > 0 and
                 (model.slide_crop < W_img or model.slide_crop < H_img))

    if not use_slide:
        heads = _grounding_heads_crop(model, image)
        maps = {k: heads[k] for k in spatial_keys}            # (nq, H_img, W_img)
        presence_img = heads["presence_vec"]                  # (nq,)
    else:
        stride = model.slide_stride
        crop = model.slide_crop
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(crop, int):
            crop = (crop, crop)
        h_stride, w_stride = stride
        h_crop, w_crop = crop

        acc = {k: torch.zeros((nq, H_img, W_img)) for k in spatial_keys}
        cnt = torch.zeros((1, H_img, W_img))
        # presence accumulated as a coverage-weighted spatial map then reduced
        pres_acc = torch.zeros((nq, H_img, W_img))

        h_grids = max(H_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(W_img - w_crop + w_stride - 1, 0) // w_stride + 1
        for hi in range(h_grids):
            for wi in range(w_grids):
                y1 = hi * h_stride
                x1 = wi * w_stride
                y2 = min(y1 + h_crop, H_img)
                x2 = min(x1 + w_crop, W_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = image.crop((x1, y1, x2, y2))
                heads = _grounding_heads_crop(model, crop_img)
                for k in spatial_keys:
                    acc[k][:, y1:y2, x1:x2] += heads[k]
                cnt[:, y1:y2, x1:x2] += 1
                pres_acc[:, y1:y2, x1:x2] += heads["presence_vec"][:, None, None]
        assert (cnt == 0).sum() == 0, "Sparse sliding-window coverage."
        maps = {k: acc[k] / cnt for k in spatial_keys}
        # image-class presence = coverage-weighted spatial mean of per-crop presence
        presence_img = (pres_acc / cnt).reshape(nq, -1).mean(1)

    # resize to GT (ori) resolution, matching predict()'s behaviour
    for k in spatial_keys:
        if maps[k].shape[-2:] != (H_o, W_o):
            maps[k] = torch.nn.functional.interpolate(
                maps[k].unsqueeze(0), size=(H_o, W_o),
                mode="bilinear", align_corners=False).squeeze(0)
    return maps, presence_img


# ---------------------------------------------------------------------------
# Per-class reservoir buffer
# ---------------------------------------------------------------------------

class Reservoir:
    """Fixed-capacity reservoir over a set of named float columns + int columns."""

    FLOAT_COLS = [
        "semantic_pixel_before_presence",
        "instance_raw_pixel_before_confidence_presence",
        "instance_original_prefinal_pixel",
        "fused_raw_before_presence",
        "fused_after_presence",
        "presence_score",
        "bg_score_post",
        "class_minus_bg_gap_post",
    ]
    INT_COLS = ["image_id", "class_id", "gt", "is_c", "is_bg", "is_other"]

    def __init__(self, cap, rng):
        self.cap = cap
        self.rng = rng
        self.seen = 0
        self.cols = {c: np.empty(cap, dtype=np.float32) for c in self.FLOAT_COLS}
        for c in self.INT_COLS:
            self.cols[c] = np.empty(cap, dtype=np.int32)
        self.filled = 0

    def add_block(self, block):
        """block: dict col-> 1D np array, all the same length m."""
        m = len(next(iter(block.values())))
        for i in range(m):
            self.seen += 1
            if self.filled < self.cap:
                idx = self.filled
                self.filled += 1
            else:
                j = self.rng.integers(0, self.seen)
                if j >= self.cap:
                    continue
                idx = j
            for c in self.cols:
                self.cols[c][idx] = block[c][i]

    def to_dict(self):
        return {c: self.cols[c][:self.filled] for c in self.cols}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.no_grad()
def main():
    args = parse_args()

    if args.dump_low_margin:
        raise SystemExit(
            "--dump-low-margin is Phase 1B and is intentionally not implemented in "
            "this script version. Run Phase 1A (--dump-release-band) first; the "
            "low-margin pool has different labels and must be dumped/analysed "
            "separately.")
    if not args.dump_release_band:
        raise SystemExit("Pass --dump-release-band (Phase 1A is the only pool here).")

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    init_default_scope(cfg.get("default_scope", "mmseg"))

    with open(cfg.model["classname_path"]) as f:
        class_names = [ln.split(",")[0].strip() for ln in f if ln.strip()]
    C = len(class_names)
    name2idx = {n: i for i, n in enumerate(class_names)}
    global_thd = float(cfg.model.get("prob_thd", 0.0))
    bg_idx = int(cfg.model.get("bg_idx", 0))
    lambda_val = args.lam * global_thd

    target_names = [n.strip() for n in args.classes.split(",") if n.strip()]
    target_ids = []
    for n in target_names:
        if n not in name2idx:
            raise SystemExit(f"unknown class '{n}'. choices: {class_names}")
        if name2idx[n] == bg_idx:
            raise SystemExit(f"'{n}' is background; collect foreground classes only.")
        target_ids.append(name2idx[n])

    dataset_tag = os.path.splitext(os.path.basename(args.config))[0]
    dataset_tag = dataset_tag.replace("cfg_", "")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"{'='*72}")
    print(f"Phase-1A Release-band Raw Multi-Head Collection")
    print(f"Config        : {args.config}")
    print(f"Dataset tag   : {dataset_tag}")
    print(f"Classes ({C})  : {class_names}")
    print(f"Targets       : {[class_names[c] for c in target_ids]}")
    print(f"global_thd τ  : {global_thd}   λτ = {lambda_val:.4f}   bg_idx={bg_idx}")
    print(f"Release-band  : raw_top1_post=c ∧ λτ ≤ fused_after_presence < τ")
    print(f"per-image cap : {args.per_image_class_cap}   global cap : {args.global_class_cap:,}")
    print(f"Out dir       : {args.out_dir}")
    print(f"Limit         : {args.limit or 'all'} images")
    print(f"{'='*72}\n")

    model = MODELS.build(cfg.model)
    model.processor.model.eval()
    loader = Runner.build_dataloader(cfg.test_dataloader)

    # query -> class map (handles synonym words, e.g. 'building,house').
    qidx = model.query_idx.detach().cpu().numpy().astype(np.int64)   # (num_queries,)
    assert int(qidx.max()) + 1 == C, \
        f"class count mismatch: qidx implies {int(qidx.max())+1}, class file has {C}"

    rng = np.random.default_rng(args.seed)
    reservoirs = {c: Reservoir(args.global_class_cap, rng) for c in target_ids}
    # smoke-test diagnostics
    band_count = {c: 0 for c in target_ids}          # total release-band pixels seen
    inst_raw_sum = {c: 0.0 for c in target_ids}
    inst_orig_sum = {c: 0.0 for c in target_ids}

    n_done = 0
    for batch in loader:
        for ds in batch["data_samples"]:
            meta = ds.metainfo
            img_path = meta.get("img_path")
            ori_h, ori_w = meta["ori_shape"][:2]
            image = Image.open(img_path).convert("RGB")

            maps_q, presence_vec = capture_full_maps(model, image, (ori_h, ori_w))

            # ---- query -> class collapse (replicates predict()'s synonym max) ----
            # The fused_after winner over CLASSES (max over each class's synonym
            # words) is exactly cafr_diagnose's raw_top1 / max_sc.  Per target class
            # we also record which synonym word won at each pixel, so every feature
            # head is gathered from the SAME winning word (coherent feature vectors
            # for the semantic-instance consistency analysis).
            faq = maps_q["fused_after"]                              # (nq, H, W)
            Hh, Ww = faq.shape[1], faq.shape[2]
            fa_cls = torch.empty((C, Hh, Ww))
            wsel_for = {}                                            # class -> (H,W) word-row index
            words_for = {}                                          # class -> tensor of query idx
            for cls in range(C):
                words = torch.from_numpy(np.where(qidx == cls)[0])
                words_for[cls] = words
                fa_w = faq[words]                                   # (|w|, H, W)
                if words.numel() == 1:
                    fa_cls[cls] = fa_w[0]
                    if cls in target_ids:
                        wsel_for[cls] = torch.zeros((Hh, Ww), dtype=torch.long)
                else:
                    vals, widx = fa_w.max(0)
                    fa_cls[cls] = vals
                    if cls in target_ids:
                        wsel_for[cls] = widx

            max_sc, raw_top1 = fa_cls.max(0)                         # (H,W),(H,W)
            bg_post = fa_cls[bg_idx]                                 # (H,W)

            gt = ds.gt_sem_seg.data.squeeze(0).long().cpu()          # (H,W)
            valid = (gt >= 0) & (gt < C)

            def _gather_head(head_name, cls):
                """Class-level head map (H,W): value at the winning synonym word."""
                words = words_for[cls]
                hq = maps_q[head_name][words]                       # (|w|, H, W)
                return hq.gather(0, wsel_for[cls][None]).squeeze(0)  # (H,W)

            for c in target_ids:
                band = (raw_top1 == c) & (max_sc >= lambda_val) & \
                       (max_sc < global_thd) & valid
                nb = int(band.sum())
                if nb == 0:
                    continue
                band_count[c] += nb

                sem_c   = _gather_head("semantic", c)[band].numpy()
                iraw_c  = _gather_head("instance_raw", c)[band].numpy()
                iorig_c = _gather_head("instance_orig", c)[band].numpy()
                fraw_c  = _gather_head("fused_raw", c)[band].numpy()
                faft_c  = fa_cls[c][band].numpy()
                bgp_c   = bg_post[band].numpy()
                gt_c    = gt[band].numpy().astype(np.int32)
                # per-pixel presence = the winning synonym word's image-level presence
                pres_map = presence_vec[words_for[c]][wsel_for[c]]   # (H,W)
                pres_pix = pres_map[band].numpy().astype(np.float32)

                inst_raw_sum[c] += float(iraw_c.sum())
                inst_orig_sum[c] += float(iorig_c.sum())

                # per-image cap via random subsample (reservoir-within-image)
                if nb > args.per_image_class_cap:
                    sel = rng.choice(nb, args.per_image_class_cap, replace=False)
                    sem_c, iraw_c, iorig_c = sem_c[sel], iraw_c[sel], iorig_c[sel]
                    fraw_c, faft_c, bgp_c = fraw_c[sel], faft_c[sel], bgp_c[sel]
                    pres_pix = pres_pix[sel]
                    m = args.per_image_class_cap
                else:
                    m = nb

                is_c = (gt_c == c).astype(np.int32)
                is_bg = (gt_c == bg_idx).astype(np.int32)
                is_other = ((gt_c != c) & (gt_c != bg_idx)).astype(np.int32)

                block = {
                    "image_id": np.full(m, n_done, dtype=np.int32),
                    "class_id": np.full(m, c, dtype=np.int32),
                    "gt": gt_c,
                    "is_c": is_c, "is_bg": is_bg, "is_other": is_other,
                    "semantic_pixel_before_presence": sem_c,
                    "instance_raw_pixel_before_confidence_presence": iraw_c,
                    "instance_original_prefinal_pixel": iorig_c,
                    "fused_raw_before_presence": fraw_c,
                    "fused_after_presence": faft_c,
                    "presence_score": pres_pix,
                    "bg_score_post": bgp_c,
                    "class_minus_bg_gap_post": (faft_c - bgp_c).astype(np.float32),
                }
                reservoirs[c].add_block(block)

            n_done += 1
            if n_done % 50 == 0:
                print(f"  processed {n_done} images ...")
            if args.limit and n_done >= args.limit:
                break
        if args.limit and n_done >= args.limit:
            break

    print(f"\nTotal images: {n_done}\n")

    # ---- save + smoke-test report ----
    print(f"{'='*72}")
    print("SMOKE-TEST DIAGNOSTICS")
    print(f"  {'class':<14}{'band_px':>12}{'kept':>10}{'seen':>12}"
          f"{'inst_raw_mu':>12}{'inst_orig_mu':>13}{'differ?':>8}")
    for c in target_ids:
        res = reservoirs[c]
        bc = band_count[c]
        raw_mu = inst_raw_sum[c] / bc if bc > 0 else float("nan")
        orig_mu = inst_orig_sum[c] / bc if bc > 0 else float("nan")
        differ = "yes" if (bc > 0 and abs(raw_mu - orig_mu) > 1e-4) else "NO(!)"
        print(f"  {class_names[c]:<14}{bc:12,}{res.filled:10,}{res.seen:12,}"
              f"{raw_mu:12.4f}{orig_mu:13.4f}{differ:>8}")

        out = res.to_dict()
        fname = os.path.join(
            args.out_dir, f"rawdiag_{dataset_tag}_releaseband_{class_names[c]}.npz")
        np.savez_compressed(
            fname, class_name=class_names[c], dataset=dataset_tag,
            global_thd=global_thd, lambda_val=lambda_val, bg_idx=bg_idx,
            **out)
        sz = os.path.getsize(fname) / 1e6
        print(f"    -> {fname}  ({sz:.1f} MB, {res.filled:,} rows)")

    print(f"\n{'='*72}")
    print("Smoke-test checks:")
    print("  1. band_px per class should track cafr_diagnose.py's n_band trend.")
    print("  2. inst_raw_mu vs inst_orig_mu must DIFFER (presence^2 split working).")
    print("  3. .npz sizes controllable (raise caps for full val once confirmed).")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
