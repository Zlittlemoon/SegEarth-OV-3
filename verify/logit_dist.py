"""
Per-class logit-scale diagnostic for the SegEarthOV3 three-head segmentor.

Goal: test the hypothesis that inter-class confusion comes from *uncalibrated
logit scales* — some class prompts produce systematically higher logits than
others, so the final argmax is biased regardless of where the true object is.

For every class c it accumulates the fused seg_logit value of channel c at two
groups of pixels:

  * "in"  : pixels whose GT == c          (channel c should win here)
  * "out" : pixels whose GT != c          (channel c should stay low here)

and reports, per class:

  in-mean / in-median  : how strong c fires on its own object
  out-mean / out-p95   : how strong c fires where it should be silent
  separation = in_median - out_median (bigger = better calibrated)

It also reports, at every pixel, which channel produced the max logit and how
that compares to the GT channel's logit (the "margin" the wrong winner had).

If one class (e.g. LoveDA background) shows a high out-mean / out-p95, that is
direct evidence of a scale bias that a *general* cross-class calibration would
fix — and which a dataset-specific threshold tweak would not.

Run:
    python verify/logit_dist.py configs/cfg_loveda.py --limit 80
    python verify/logit_dist.py configs/cfg_openearthmap.py --limit 80
    python verify/logit_dist.py configs/cfg_loveda.py --cfg-options model.sam3_infer_mode=batch
"""

import argparse

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmseg.registry import MODELS

import mmseg.datasets              # registers LoveDADataset / OpenEarthMapDataset etc.
import segearthov3_segmentor_merge  # registers SegEarthOV3Segmentation
import custom_datasets             # registers custom datasets


def parse_args():
    ap = argparse.ArgumentParser(description="Per-class logit-scale diagnostic")
    ap.add_argument("config")
    ap.add_argument("--limit", type=int, default=80,
                    help="evaluate the first N images (default 80, enough for stats)")
    ap.add_argument("--sample", type=int, default=20000,
                    help="max pixels sampled per image per group (keeps memory sane)")
    ap.add_argument("--cfg-options", nargs="+", action=DictAction, default=None)
    ap.add_argument("--out", default=None,
                    help="optional .npz path to dump the raw per-class samples")
    return ap.parse_args()


def _subsample(t, k, generator):
    """Randomly keep at most k elements of 1-D tensor t."""
    if t.numel() <= k:
        return t
    idx = torch.randperm(t.numel(), generator=generator)[:k]
    return t[idx]


@torch.no_grad()
def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    init_default_scope(cfg.get("default_scope", "mmseg"))

    classname_path = cfg.model["classname_path"]
    with open(classname_path) as f:
        class_names = [ln.split(",")[0].strip() for ln in f if ln.strip()]
    C = len(class_names)
    print(f"Config : {args.config}")
    print(f"Classes ({C}): {class_names}")
    print(f"Infer mode: {cfg.model.get('sam3_infer_mode', 'serial')}")
    print(f"Limit: {args.limit} images, sampling <= {args.sample} px/group/image\n")

    model = MODELS.build(cfg.model)
    model.processor.model.eval()
    loader = Runner.build_dataloader(cfg.test_dataloader)

    gen = torch.Generator().manual_seed(0)
    # per-class pools of sampled logit values
    in_pool = [[] for _ in range(C)]
    out_pool = [[] for _ in range(C)]
    # margin that the (wrong) argmax winner had over the GT channel, when wrong
    wrong_margin = []

    n_done = 0
    for batch in loader:
        out = model.predict(batch["inputs"], batch["data_samples"])
        for ds in out:
            logits = ds.seg_logits.data.float().cpu()        # (C, H, W)
            gt = ds.gt_sem_seg.data.squeeze(0).long().cpu()  # (H, W)
            valid = (gt >= 0) & (gt < C)
            if not valid.any():
                continue

            flat = logits.reshape(C, -1)                     # (C, P)
            gtf = gt.reshape(-1)
            vmask = valid.reshape(-1)

            for c in range(C):
                lc = flat[c][vmask]                          # logit of channel c
                gc = gtf[vmask]
                inv = lc[gc == c]
                outv = lc[gc != c]
                if inv.numel():
                    in_pool[c].append(_subsample(inv, args.sample, gen))
                if outv.numel():
                    out_pool[c].append(_subsample(outv, args.sample, gen))

            # winner margin on mis-classified valid pixels
            pred = flat.argmax(0)                            # (P,)
            gv = gtf[vmask]
            pv = pred[vmask]
            wrong = pv != gv
            if wrong.any():
                idx = torch.nonzero(vmask).squeeze(1)[wrong]
                win = flat[pv[wrong], idx]
                tru = flat[gv[wrong], idx]
                wrong_margin.append(_subsample(win - tru, args.sample, gen))

        n_done += len(out)
        if n_done % 20 == 0:
            print(f"  processed {n_done} images ...")
        if args.limit is not None and n_done >= args.limit:
            break

    print(f"\nTotal images: {n_done}\n")

    def cat(pools):
        return [torch.cat(p) if len(p) else torch.zeros(0) for p in pools]

    inv = cat(in_pool)
    outv = cat(out_pool)

    def q(t, p):
        return float(t.quantile(p)) if t.numel() else float("nan")

    # ---- per-class scale table ----
    print("Per-class fused-logit scale (channel c at GT==c vs GT!=c):")
    print(f"  {'class':<14}{'in_med':>8}{'in_mean':>9}"
          f"{'out_med':>9}{'out_p95':>9}{'out_mean':>9}{'sep':>8}")
    seps = []
    for c in range(C):
        i_med, i_mean = q(inv[c], 0.5), float(inv[c].mean()) if inv[c].numel() else float("nan")
        o_med, o_p95 = q(outv[c], 0.5), q(outv[c], 0.95)
        o_mean = float(outv[c].mean()) if outv[c].numel() else float("nan")
        sep = i_med - o_med
        seps.append(sep)
        print(f"  {class_names[c]:<14}{i_med:8.3f}{i_mean:9.3f}"
              f"{o_med:9.3f}{o_p95:9.3f}{o_mean:9.3f}{sep:8.3f}")

    # ---- global "off" scale ranking: which class fires most where it shouldn't ----
    print("\nClasses ranked by out-of-class firing (out_p95, high = scale bias):")
    order = sorted(range(C), key=lambda c: q(outv[c], 0.95), reverse=True)
    for c in order:
        print(f"  {class_names[c]:<14}out_p95={q(outv[c],0.95):7.3f}"
              f"   out_mean={float(outv[c].mean()) if outv[c].numel() else float('nan'):7.3f}"
              f"   sep={seps[c]:7.3f}")

    if wrong_margin:
        wm = torch.cat(wrong_margin)
        print(f"\nWrong-pixel winner margin (winner_logit - gt_logit), "
              f"n={wm.numel():,}:")
        print(f"  median={float(wm.median()):.3f}  "
              f"p90={q(wm,0.9):.3f}  mean={float(wm.mean()):.3f}")
        print("  (small margins => GT was a close runner-up => calibration can flip it)")

    if args.out:
        np.savez(args.out,
                 class_names=np.array(class_names),
                 **{f"in_{c}": inv[c].numpy() for c in range(C)},
                 **{f"out_{c}": outv[c].numpy() for c in range(C)})
        print(f"\nSaved raw samples to {args.out}")


if __name__ == "__main__":
    main()
