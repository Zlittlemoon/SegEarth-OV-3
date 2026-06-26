"""
Per-class confusion analysis for the SegEarthOV3 segmentor on a dataset.

Runs the real (three-head fusion) segmentor over the validation set defined in
a config, accumulates a full C x C confusion matrix (rows = ground truth,
cols = prediction), and reports:

  * per-class IoU / recall / precision
  * row-normalized confusion matrix (what each GT class gets predicted as)
  * the top class-vs-class confusion pairs (GT c -> wrongly predicted as c')

This is Step 0 of the class-confusion module: it tells us *whether* class
confusion is the dominant error and *which* class pairs to target, before
building any CLIP-calibration machinery.

Mirrors eval_merge.py's model/dataloader construction, so it uses the same
three-head config you ran the four-dataset benchmark with.

Run:
    python verify/confusion_loveda.py configs/cfg_loveda.py
    python verify/confusion_loveda.py configs/cfg_loveda.py --cfg-options model.sam3_infer_mode=batch
    python verify/confusion_loveda.py configs/cfg_loveda.py --limit 50   # quick look
"""

import argparse
import os

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmseg.registry import MODELS

import mmseg.datasets
import segearthov3_segmentor_merge  # registers SegEarthOV3Segmentation
import custom_datasets             # registers datasets


def parse_args():
    ap = argparse.ArgumentParser(description="Confusion-matrix analysis")
    ap.add_argument("config")
    ap.add_argument("--limit", type=int, default=None,
                    help="only evaluate the first N images (quick look)")
    ap.add_argument("--cfg-options", nargs="+", action=DictAction, default=None)
    ap.add_argument("--out", default=None,
                    help="optional .npy path to save the raw confusion matrix")
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Activate the config's default scope ('mmseg') so DATASETS.build can
    # resolve LoveDADataset etc. The Runner does this automatically, but we
    # call build_dataloader() statically and must set it ourselves.
    init_default_scope(cfg.get("default_scope", "mmseg"))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- class names (from the segmentor's class file) ----
    classname_path = cfg.model["classname_path"]
    with open(classname_path) as f:
        class_names = [ln.split(",")[0].strip() for ln in f if ln.strip()]
    C = len(class_names)
    print(f"Config : {args.config}")
    print(f"Classes ({C}): {class_names}")
    print(f"Infer mode: {cfg.model.get('sam3_infer_mode', 'serial')}\n")

    # ---- build model exactly like eval_merge ----
    model = MODELS.build(cfg.model)
    model.processor.model.eval()

    # ---- build the val dataloader from the config ----
    loader = Runner.build_dataloader(cfg.test_dataloader)

    # confusion[g, p] = #pixels with GT class g predicted as class p
    confusion = torch.zeros((C, C), dtype=torch.long)

    n_done = 0
    for batch in loader:
        data_samples = batch["data_samples"]
        inputs = batch["inputs"]
        out = model.predict(inputs, data_samples)
        for ds in out:
            pred = ds.pred_sem_seg.data.squeeze(0).to(torch.long).cpu()  # (H, W)
            gt = ds.gt_sem_seg.data.squeeze(0).to(torch.long).cpu()      # (H, W)
            valid = (gt >= 0) & (gt < C)   # drops ignore_index (255)
            g = gt[valid]
            p = pred[valid].clamp(0, C - 1)
            idx = g * C + p
            confusion += torch.bincount(idx, minlength=C * C).reshape(C, C)
        n_done += len(out)
        if n_done % 50 == 0:
            print(f"  processed {n_done} images ...")
        if args.limit is not None and n_done >= args.limit:
            break

    print(f"\nTotal images: {n_done}\n")
    cm = confusion.numpy().astype(np.float64)

    # ---- per-class IoU / recall / precision ----
    tp = np.diag(cm)
    gt_tot = cm.sum(1)          # row sums = GT pixels per class
    pred_tot = cm.sum(0)        # col sums = predicted pixels per class
    union = gt_tot + pred_tot - tp
    iou = np.where(union > 0, tp / np.maximum(union, 1), 0.0)
    recall = np.where(gt_tot > 0, tp / np.maximum(gt_tot, 1), 0.0)
    precision = np.where(pred_tot > 0, tp / np.maximum(pred_tot, 1), 0.0)

    print("Per-class metrics (%):")
    print(f"  {'class':<14}{'IoU':>8}{'recall':>8}{'prec':>8}{'GT%':>8}")
    tot = gt_tot.sum()
    order = np.argsort(iou)  # worst first
    for c in order:
        print(f"  {class_names[c]:<14}{iou[c]*100:8.2f}{recall[c]*100:8.2f}"
              f"{precision[c]*100:8.2f}{gt_tot[c]/tot*100:8.2f}")
    print(f"  {'mIoU':<14}{iou.mean()*100:8.2f}")

    # ---- row-normalized confusion (each GT row sums to 100%) ----
    cm_norm = np.where(gt_tot[:, None] > 0, cm / np.maximum(gt_tot[:, None], 1), 0.0)
    print("\nRow-normalized confusion (row=GT, col=pred, %):")
    header = "  " + " " * 14 + "".join(f"{n[:6]:>8}" for n in class_names)
    print(header)
    for g in range(C):
        row = "".join(f"{cm_norm[g, p]*100:8.1f}" for p in range(C))
        print(f"  {class_names[g]:<14}{row}")

    # ---- top off-diagonal confusion pairs ----
    pairs = []
    for g in range(C):
        for p in range(C):
            if g != p and gt_tot[g] > 0:
                pairs.append((cm_norm[g, p], g, p, cm[g, p]))
    pairs.sort(reverse=True)
    print("\nTop class-confusion pairs (GT -> predicted):")
    print(f"  {'GT':<14}{'->':^4}{'pred':<14}{'% of GT':>9}{'pixels':>14}")
    for frac, g, p, npix in pairs[:15]:
        print(f"  {class_names[g]:<14}{'->':^4}{class_names[p]:<14}"
              f"{frac*100:8.1f}%{int(npix):>14,}")

    if args.out:
        np.save(args.out, confusion.numpy())
        print(f"\nSaved raw confusion matrix to {args.out}")


if __name__ == "__main__":
    main()
