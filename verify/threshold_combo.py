"""
Step A (verification): combined class-specific prob_thd validation.

threshold_scan.py scans ONE class at a time. Per-class deltas do NOT add up
linearly: two lowered thresholds can both push false positives into (or pull
them out of) the SAME neighbour (e.g. background), so the combined effect must
be measured directly. This script applies a full class->thd override at once
and reports the real combined mIoU and per-class IoU.

Decision rule (mirrors predict() exactly, same as threshold_scan.py):
    pred = raw_top1
    thd  = override[raw_top1]  if set  else global_prob_thd
    if max_score < thd:  pred = bg_idx

Single inference pass; all configs evaluated in numpy from the cache.

Specify overrides as name=thd pairs. Pass --set multiple times to compare
several configurations against the same baseline in one run.

Run (OEM):
    python verify/threshold_combo.py configs/cfg_openearthmap.py --limit 80 \
        --set pavement=0.0 grass=0.02

Run (LoveDA), comparing two candidate configs at once:
    python verify/threshold_combo.py configs/cfg_loveda.py --limit 80 \
        --set water=0.1 forest=0.4 \
        --set water=0.1 forest=0.3

After picking a config, RE-RUN oracle_ceiling.py with the same overrides to see
how much top1->top2 CLIP headroom remains once these pixels return as foreground.
"""

import argparse

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmseg.registry import MODELS

import mmseg.datasets
import segearthov3_segmentor_merge
import custom_datasets


def parse_args():
    ap = argparse.ArgumentParser(description="Combined class-specific prob_thd check")
    ap.add_argument("config")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--set", nargs="+", action="append", default=None,
                    dest="sets", metavar="name=thd",
                    help="one override config as 'name=thd' pairs. Repeat --set to "
                         "compare multiple configs against the same baseline.")
    ap.add_argument("--cfg-options", nargs="+", action=DictAction, default=None)
    return ap.parse_args()


def iou_from_cm(cm):
    tp = np.diag(cm).astype(np.float64)
    union = cm.sum(1) + cm.sum(0) - tp
    iou = np.full(len(tp), np.nan)
    ok = union > 0
    iou[ok] = tp[ok] / union[ok].astype(np.float64)
    return iou


def cm_from_pred(g, p, C):
    return np.bincount(g * C + p, minlength=C * C).reshape(C, C)


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

    global_thd = float(cfg.model.get("prob_thd", 0.0))
    bg_idx     = int(cfg.model.get("bg_idx", 0))

    if not args.sets:
        raise SystemExit("provide at least one --set name=thd [...]")

    # parse each --set into a {class_idx: thd} override dict
    configs = []
    for raw in args.sets:
        ov = {}
        for item in raw:
            if "=" not in item:
                raise SystemExit(f"bad override '{item}', expected name=thd")
            nm, val = item.split("=", 1)
            if nm not in name2idx:
                raise SystemExit(f"unknown class '{nm}'. choices: {class_names}")
            idx = name2idx[nm]
            if idx == bg_idx:
                raise SystemExit(f"'{nm}' is background; override foreground classes only.")
            ov[idx] = float(val)
        configs.append(ov)

    print(f"Config       : {args.config}")
    print(f"Classes ({C}) : {class_names}")
    print(f"global prob_thd={global_thd}  bg_idx={bg_idx}")
    for i, ov in enumerate(configs):
        desc = ", ".join(f"{class_names[k]}={v}" for k, v in ov.items())
        print(f"config[{i}]    : {{{desc}}}")
        for k, v in ov.items():
            if v > global_thd:
                print(f"  [WARN] {class_names[k]}={v} > global_thd={global_thd}; "
                      "this RAISES the gate, not lowers it.")
    print(f"limit        : {args.limit or 'all'} images\n")

    model = MODELS.build(cfg.model)
    model.processor.model.eval()
    loader = Runner.build_dataloader(cfg.test_dataloader)

    # ---- single inference pass: cache (raw_top1, max_score, gt) ----
    top1_chunks, max_chunks, gt_chunks = [], [], []
    n_done = 0
    for batch in loader:
        out = model.predict(batch["inputs"], batch["data_samples"])
        for ds in out:
            logits = ds.seg_logits.data.float().cpu()
            gt     = ds.gt_sem_seg.data.squeeze(0).long().cpu()
            valid  = (gt >= 0) & (gt < C)
            flat   = logits.reshape(C, -1)
            vmask  = valid.reshape(-1)
            mx, top1 = flat.max(0)
            top1_chunks.append(top1[vmask].numpy().astype(np.int32))
            max_chunks.append(mx[vmask].numpy().astype(np.float32))
            gt_chunks.append(gt.reshape(-1)[vmask].numpy().astype(np.int32))
        n_done += len(out)
        if n_done % 50 == 0:
            print(f"  processed {n_done} images ...")
        if args.limit and n_done >= args.limit:
            break

    print(f"\nTotal images : {n_done}")
    raw_top1 = np.concatenate(top1_chunks).astype(np.int64)
    max_sc   = np.concatenate(max_chunks).astype(np.float32)
    gt_all   = np.concatenate(gt_chunks).astype(np.int64)
    del top1_chunks, max_chunks, gt_chunks
    print(f"Valid pixels : {len(gt_all):,}\n")

    def predict_with(thd_vec):
        pred = raw_top1.copy()
        pred[max_sc < thd_vec[raw_top1]] = bg_idx
        return pred

    # ---- baseline ----
    base_thd  = np.full(C, global_thd, dtype=np.float32)
    base_pred = predict_with(base_thd)
    base_iou  = iou_from_cm(cm_from_pred(gt_all, base_pred, C))
    base_miou = np.nanmean(base_iou) * 100
    N = len(gt_all)
    print(f"Baseline mIoU (global prob_thd={global_thd}): {base_miou:.2f}%")
    print("=" * 78)

    # ---- each override config ----
    for i, ov in enumerate(configs):
        thd_vec = base_thd.copy()
        for k, v in ov.items():
            thd_vec[k] = v
        pred = predict_with(thd_vec)
        iou  = iou_from_cm(cm_from_pred(gt_all, pred, C))
        miou = np.nanmean(iou) * 100

        desc = ", ".join(f"{class_names[k]}={v}" for k, v in ov.items())
        print(f"\nconfig[{i}]: {{{desc}}}")
        print(f"  Combined mIoU: {miou:.2f}%   Δ vs baseline = {miou-base_miou:+.2f}")

        # pixel-flow vs baseline: is the gain recovering true positives or
        # injecting false positives?
        changed = pred != base_pred
        rescue  = (base_pred != gt_all) & (pred == gt_all)
        damage  = (base_pred == gt_all) & (pred != gt_all)
        from_bg = (base_pred == bg_idx) & (pred != bg_idx)
        to_bg   = (base_pred != bg_idx) & (pred == bg_idx)
        net     = int(rescue.sum()) - int(damage.sum())
        print(f"  pixel-flow: changed={changed.sum()/N*100:.3f}%  "
              f"rescue={rescue.sum()/N*100:.3f}%  damage={damage.sum()/N*100:.3f}%  "
              f"net={net/N*100:+.3f}%")
        print(f"              from_bg={from_bg.sum()/N*100:.3f}%  "
              f"to_bg={to_bg.sum()/N*100:.3f}%")

        # per overridden class: how many pixels it released and their GT split
        print(f"  released per overridden class:")
        print(f"    {'class':<14}{'thd':>6}{'released':>12}{'rel_GT%':>9}"
              f"{'rel_bg%':>9}{'rel_oth%':>10}")
        for k, v in ov.items():
            is_k   = raw_top1 == k
            rel    = is_k & (max_sc >= v) & (max_sc < global_thd) if v < global_thd \
                     else np.zeros(N, dtype=bool)
            rn     = int(rel.sum())
            rgt    = int((rel & (gt_all == k)).sum())
            rbg    = int((rel & (gt_all == bg_idx)).sum())
            roth   = rn - rgt - rbg
            print(f"    {class_names[k]:<14}{v:6.3f}{rn:12,}"
                  f"{rgt/max(rn,1)*100:8.1f}%{rbg/max(rn,1)*100:8.1f}%"
                  f"{roth/max(rn,1)*100:9.1f}%")

        # sum of touched-class deltas vs the true combined delta, to expose
        # any non-linear interaction through shared neighbours (e.g. background)
        print(f"\n  {'class':<14}{'baseline':>10}{'combo':>10}{'Δ':>8}")
        order = np.argsort(np.nan_to_num(base_iou, nan=-1))
        for k in order:
            b = base_iou[k] * 100
            v = iou[k] * 100
            d = v - b
            mark = " **" if k in ov else ""
            print(f"  {class_names[k]:<14}{b:10.2f}{v:10.2f}{d:+8.2f}{mark}")
        print(f"  {'mIoU':<14}{base_miou:10.2f}{miou:10.2f}{miou-base_miou:+8.2f}")
        print("=" * 78)

    print("\nNote: ** marks overridden classes. Watch shared neighbours (esp. background):")
    print("  combined Δ may differ from the sum of single-class scans because two")
    print("  lowered thresholds add/remove false positives in the SAME neighbour.")
    print("  Re-run oracle_ceiling.py with these overrides to refresh CLIP headroom.")


if __name__ == "__main__":
    main()
