"""
Visualize the low-margin connected components (CCs) of a chosen class for a
single image, using the *exact same* filtering pipeline as
remoteclip_pair_benchmark.py:

  low-margin  : top1_logit - top2_logit < margin_thr
  foreground  : winner (top1) and runnerup (top2) both != bg, not backfilled
  CC          : 4/8-connected components of the surviving pixels for a given
                (winner[, runnerup]) selection, size >= min-cc

You pick a target class by name. Two modes:

  --winner CLASS
      Show all low-margin CCs whose predicted winner == CLASS
      (regardless of runnerup). CCs are grouped/colored by runnerup.

  --winner CLASS --runnerup CLASS2
      Show only the CCs for the directed pair CLASS -> CLASS2.

Outputs a side-by-side panel:
  [ original image | prediction map | CC overlay ]
with each CC outlined + its bbox + a label (id / runnerup / size / GT-rescue%).

Usage (find the image name first, e.g. from a *_images.txt split, or just
pass the basename that appears in ds.img_path):

    python verify/visualize_cc.py configs/cfg_openearthmap.py \\
        --image aachen_1.tif \\
        --winner road --runnerup pavement \\
        --margin-thr 0.1 --out vis_road_pavement.png

    python verify/visualize_cc.py configs/cfg_loveda.py \\
        --image 3514.png --winner agricultural \\
        --margin-thr 0.1 --out vis_agri.png
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
from scipy import ndimage as ndi

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import mmseg.datasets
import segearthov3_segmentor_merge
import custom_datasets

MIN_CC_PIXELS = 16


def parse_args():
    ap = argparse.ArgumentParser(
        description="Visualize low-margin CCs of a class for one image")
    ap.add_argument("config")
    ap.add_argument("--image", required=True,
                    help="Image filename to match against ds.img_path "
                         "(basename match, e.g. 'aachen_1.tif').")
    ap.add_argument("--winner", required=True,
                    help="Target predicted class (top1) whose CCs to show.")
    ap.add_argument("--runnerup", default=None,
                    help="If given, only show CCs for the directed pair "
                         "winner->runnerup. Otherwise show all runnerups.")
    ap.add_argument("--margin-thr", type=float, default=0.1)
    ap.add_argument("--min-cc", type=int, default=MIN_CC_PIXELS)
    ap.add_argument("--connectivity", type=int, choices=[4, 8], default=8,
                    help="CC connectivity (4 or 8). Default 8.")
    ap.add_argument("--out", default="vis_cc.png",
                    help="Output PNG path.")
    ap.add_argument("--cfg-options", nargs="+", action=DictAction, default=None)
    return ap.parse_args()


def cc_structure(connectivity):
    if connectivity == 8:
        return ndi.generate_binary_structure(2, 2)
    return ndi.generate_binary_structure(2, 1)


def color_palette(n):
    """Deterministic distinct-ish colors in [0,1]."""
    rng = np.random.default_rng(12345)
    base = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[:, :3]
    if n <= 20:
        return base[:n]
    extra = rng.random((n - 20, 3))
    return np.concatenate([base, extra], axis=0)


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

    if args.winner not in name2idx:
        raise SystemExit(f"unknown winner class '{args.winner}'. choices: {class_names}")
    w_idx = name2idx[args.winner]
    ru_idx = None
    if args.runnerup is not None:
        if args.runnerup not in name2idx:
            raise SystemExit(f"unknown runnerup class '{args.runnerup}'. choices: {class_names}")
        ru_idx = name2idx[args.runnerup]

    prob_thd = float(cfg.model.get("prob_thd", 0.0))
    bg_idx = int(cfg.model.get("bg_idx", 0))

    struct = cc_structure(args.connectivity)
    target_base = os.path.basename(args.image)

    print(f"Classes ({C}): {class_names}")
    print(f"Target image : {target_base}")
    print(f"winner={args.winner}({w_idx})"
          + (f"  runnerup={args.runnerup}({ru_idx})" if ru_idx is not None else "  runnerup=<any>"))
    print(f"margin_thr={args.margin_thr}  min_cc={args.min_cc}  conn={args.connectivity}\n")

    seg_model = MODELS.build(cfg.model)
    seg_model.processor.model.eval()
    loader = Runner.build_dataloader(cfg.test_dataloader)

    found = None
    for batch in loader:
        out = seg_model.predict(batch["inputs"], batch["data_samples"])
        for ds in out:
            img_path = ds.img_path if hasattr(ds, "img_path") else ""
            if os.path.basename(img_path) != target_base:
                continue
            found = (ds, img_path)
            break
        if found is not None:
            break

    if found is None:
        raise SystemExit(f"image '{target_base}' not found in the dataloader.")

    ds, img_path = found
    print(f"Matched: {img_path}")

    logits = ds.seg_logits.data.float().cpu()
    gt = ds.gt_sem_seg.data.squeeze(0).long().cpu()
    valid = (gt >= 0) & (gt < C)
    H, W = gt.shape

    flat = logits.reshape(C, -1)

    # Prediction map with prob_thd backfill (same as benchmark)
    base_p = flat.argmax(0).clone()
    mx = flat.max(0).values
    thd_vec = torch.full((C,), prob_thd)
    gate = thd_vec[base_p]
    base_p[mx < gate] = bg_idx
    pred_map = base_p.reshape(H, W).numpy()

    # top1/top2 + margin on the full grid
    top2 = flat.topk(2, dim=0)
    top1_idx = top2.indices[0].reshape(H, W).numpy()
    top2_idx = top2.indices[1].reshape(H, W).numpy()
    margin = (top2.values[0] - top2.values[1]).reshape(H, W).numpy()

    valid_np = valid.numpy()
    gt_np = gt.numpy()

    not_backfilled = (pred_map == top1_idx) & (pred_map != bg_idx) & (top2_idx != bg_idx)
    lm = (margin < args.margin_thr) & not_backfilled & valid_np

    # Select target-winner low-margin pixels
    sel = lm & (top1_idx == w_idx)
    if ru_idx is not None:
        sel = sel & (top2_idx == ru_idx)

    if not sel.any():
        print("No low-margin pixels for this selection in this image. "
              "Nothing to visualize (empty overlay will still be saved).")

    label_map, n_labels = ndi.label(sel, structure=struct)

    # Collect kept CCs
    ccs = []  # (id, ys, xs, ru, size, rescue_frac)
    next_id = 0
    for lb in range(1, n_labels + 1):
        ys, xs = np.where(label_map == lb)
        if len(ys) < args.min_cc:
            continue
        # runnerup for this CC (majority, though within-pair it's constant)
        ru_here = int(np.bincount(top2_idx[ys, xs], minlength=C).argmax())
        cc_gt = gt_np[ys, xs]
        rescue_frac = float((cc_gt == ru_here).mean())
        ccs.append((next_id, ys, xs, ru_here, len(ys), rescue_frac))
        next_id += 1

    print(f"Kept CCs (>= {args.min_cc}px): {len(ccs)}  "
          f"(raw components before size filter: {n_labels})")

    # Load original RGB
    if os.path.exists(img_path):
        image_np = np.array(Image.open(img_path).convert("RGB"))
        if image_np.shape[:2] != (H, W):
            image_np = np.array(
                Image.fromarray(image_np).resize((W, H), Image.BILINEAR))
    else:
        image_np = np.zeros((H, W, 3), dtype=np.uint8)

    # Color CCs by runnerup class for legend clarity
    runnerups_present = sorted({c[3] for c in ccs})
    ru2color = {ru: color_palette(len(runnerups_present))[i]
                for i, ru in enumerate(runnerups_present)}

    # Shared class palette for prediction & GT so colors are comparable
    pred_pal = color_palette(C)

    # GT colored with the same palette; invalid pixels -> black
    gt_rgb = np.zeros((H, W, 3), dtype=np.float32)
    gt_clamped = np.clip(gt_np, 0, C - 1)
    gt_rgb[valid_np] = pred_pal[gt_clamped[valid_np]]

    # ---- Plot: original | prediction | GT | CC overlay ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.ravel()

    axes[0].imshow(image_np)
    axes[0].set_title(f"Original\n{target_base}")
    axes[0].axis("off")

    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth (invalid=black)")
    axes[1].axis("off")

    axes[2].imshow(pred_pal[pred_map])
    axes[2].set_title("Prediction (argmax + backfill)")
    axes[2].axis("off")

    axes[3].imshow(image_np)
    title = f"low-margin CCs of '{args.winner}'"
    if ru_idx is not None:
        title += f" -> '{args.runnerup}'"
    axes[3].set_title(title + f"\n(margin<{args.margin_thr}, {len(ccs)} CCs)")
    axes[3].axis("off")

    # Overlay each CC: filled semi-transparent mask + bbox + label
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    for (cid, ys, xs, ru_here, size, rf) in ccs:
        col = ru2color[ru_here]
        overlay[ys, xs, :3] = col
        overlay[ys, xs, 3] = 0.45
    axes[3].imshow(overlay)

    for (cid, ys, xs, ru_here, size, rf) in ccs:
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        col = ru2color[ru_here]
        axes[3].add_patch(Rectangle(
            (x0, y0), x1 - x0 + 1, y1 - y0 + 1,
            fill=False, edgecolor=col, linewidth=1.5))
        axes[3].text(
            x0, max(0, y0 - 2),
            f"#{cid} {class_names[ru_here]} {size}px r{rf:.0%}",
            color="white", fontsize=6,
            bbox=dict(facecolor=col, alpha=0.8, pad=0.5, edgecolor="none"))

    # Legend for runnerup colors (on the CC panel)
    if runnerups_present:
        handles = [Rectangle((0, 0), 1, 1, color=ru2color[ru])
                   for ru in runnerups_present]
        labels = [f"runnerup={class_names[ru]}" for ru in runnerups_present]
        axes[3].legend(handles, labels, loc="lower right", fontsize=7,
                       framealpha=0.7)

    # Class color legend (shared by prediction & GT), placed on GT panel
    cls_handles = [Rectangle((0, 0), 1, 1, color=pred_pal[i]) for i in range(C)]
    cls_labels = [f"{i}:{n}" for i, n in enumerate(class_names)]
    axes[1].legend(cls_handles, cls_labels, loc="lower right", fontsize=6,
                   framealpha=0.7, ncol=1)

    plt.tight_layout()
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nSaved visualization to: {args.out}")

    # Text summary
    if ccs:
        print("\nCC summary (id | runnerup | size | GT-rescue%):")
        for (cid, ys, xs, ru_here, size, rf) in sorted(ccs, key=lambda c: -c[4]):
            print(f"  #{cid:<3} {class_names[ru_here]:<14} {size:>7}px  rescue={rf:.1%}")


if __name__ == "__main__":
    main()
