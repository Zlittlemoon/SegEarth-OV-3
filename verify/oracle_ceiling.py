"""
Top-2 oracle ceiling diagnostic for SegEarth-OV3.

Answers: "If a perfect binary arbiter could always pick the correct class
between top-1 and top-2 on low-margin pixels, how much mIoU could we gain?"

This is the headroom estimate for any two-class reranker (CLIP, RemoteCLIP,
or any semantic judge). If the ceiling is high (~3+ mIoU) and concentrated
in a few class pairs, it justifies building the CLIP path. If the ceiling is
low (<1 mIoU), score-based or semantic calibration has no room to help and
the problem lies elsewhere.

STRICT candidate: only low-margin pixels whose FINAL prediction still equals the
raw top1 and is not background (p_v == top1_idx, top1 != bg). This isolates the
genuine "CLIP picks between top1 and top2" case. Pixels that prob_thd reverted to
background are excluded here and accounted for separately in the backfill bucket
(they are recoverable by a class-specific threshold, not by a top1->top2 rerank).

What it computes, for every strict low-margin pixel (top1-top2 < margin_thr):
  * GT == top-1  (winner was already correct; flipping to runner-up = DAMAGE)
  * GT == top-2  (runner-up is correct; a perfect arbiter flips = RESCUE)
  * GT == other  (neither; flipping is a lateral error)

Oracle mIoU: re-evaluate with (GT==top-2) pixels flipped to their runner-up.
Also reports the top winner->runner-up pairs by rescuable pixels with the
break-even precision a real arbiter must beat, plus a backfill bucket showing how
many foreground pixels a class-specific threshold could recover for free.

Run:
    python verify/oracle_ceiling.py configs/cfg_openearthmap.py --limit 80 --margin-thr 0.2
    python verify/oracle_ceiling.py configs/cfg_loveda.py     --limit 80 --margin-thr 0.2

    # sweep margin thresholds to see how sensitive the ceiling is
    python verify/oracle_ceiling.py configs/cfg_openearthmap.py \
        --limit 80 --margin-thr 0.05 0.10 0.20 0.30
"""

import argparse
from collections import defaultdict

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
    ap = argparse.ArgumentParser(description="Top-2 oracle ceiling diagnostic")
    ap.add_argument("config")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--margin-thr", nargs="+", type=float, default=[0.2],
                    help="top1-top2 threshold(s) defining 'low-margin' pixels. "
                         "Multiple values are swept in one pass.")
    ap.add_argument("--top-pairs", type=int, default=15,
                    help="how many winner->runnerup pairs to show in the table")
    ap.add_argument("--set", nargs="+", default=None, metavar="name=thd",
                    help="class-specific prob_thd overrides (Step A), e.g. "
                         "--set pavement=0.0 grass=0.02. The baseline prediction "
                         "uses these per-class thresholds so the strict low-margin "
                         "candidate pool reflects the post-threshold state.")
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
    bg_idx   = int(cfg.model.get("bg_idx", 0))
    thresholds = sorted(args.margin_thr)
    
    # per-class prob_thd vector (Step A overrides); default = global prob_thd
    name2idx = {n: i for i, n in enumerate(class_names)}
    thd_vec = np.full(C, prob_thd, dtype=np.float32)
    overrides = {}
    if args.set:
        for item in args.set:
            if "=" not in item:
                raise SystemExit(f"bad override '{item}', expected name=thd")
            nm, val = item.split("=", 1)
            if nm not in name2idx:
                raise SystemExit(f"unknown class '{nm}'. choices: {class_names}")
            idx = name2idx[nm]
            if idx == bg_idx:
                raise SystemExit(f"'{nm}' is background; override foreground only.")
            thd_vec[idx] = float(val)
            overrides[idx] = float(val)
    thd_vec_t = torch.from_numpy(thd_vec)

    print(f"Config  : {args.config}")
    print(f"Classes ({C}): {class_names}")
    print(f"prob_thd={prob_thd}  bg_idx={bg_idx}")
    if overrides:
        od = ", ".join(f"{class_names[k]}={v}" for k, v in overrides.items())
        print(f"class-thd overrides: {{{od}}}")
    print(f"margin_thr sweep: {thresholds}")
    print(f"Limit   : {args.limit or 'all'} images\n")

    model = MODELS.build(cfg.model)
    model.processor.model.eval()
    loader = Runner.build_dataloader(cfg.test_dataloader)

    # Per-threshold accumulators
    cm_base = np.zeros((C, C), dtype=np.int64)          # baseline confusion matrix
    cm_oracle = {t: np.zeros((C, C), dtype=np.int64) for t in thresholds}

    # Per-threshold pixel counters
    n_lm       = {t: 0 for t in thresholds}  # total low-margin valid pixels
    n_correct  = {t: 0 for t in thresholds}  # GT==top1  (flip = damage)
    n_rescue   = {t: 0 for t in thresholds}  # GT==top2  (flip = rescue)
    n_other    = {t: 0 for t in thresholds}  # GT==other (flip = lateral)

    # Per-threshold pair counters: (winner, runnerup) -> {correct, rescue, other}
    pair_counts = {t: defaultdict(lambda: np.zeros(3, dtype=np.int64))
                   for t in thresholds}

    # Backfill bucket (margin-independent, separate from the CLIP rerank oracle):
    # pixels whose raw top1 is a FOREGROUND class but prob_thd reverted them to bg.
    # These are recoverable by a class-specific threshold, NOT by top1->top2 rerank.
    n_backfill        = 0   # raw top1 != bg, but backfilled to bg
    n_backfill_rescue = 0   # ... and GT == raw top1 (threshold could recover directly)
    bf_pair = defaultdict(lambda: np.zeros(2, dtype=np.int64))  # raw_top1 -> {total, rescue}

    n_done = 0
    for batch in loader:
        out = model.predict(batch["inputs"], batch["data_samples"])
        for ds in out:
            logits = ds.seg_logits.data.float().cpu()           # (C, H, W)
            gt     = ds.gt_sem_seg.data.squeeze(0).long().cpu()  # (H, W)
            valid  = (gt >= 0) & (gt < C)
            flat   = logits.reshape(C, -1)                       # (C, P)
            vmask  = valid.reshape(-1)                           # (P,) bool
            g_all  = gt.reshape(-1)                              # (P,)

            # --- baseline prediction (mirrors predict(), per-class thd) ---
            base_p = flat.argmax(0).clone()
            mx     = flat.max(0).values
            gate   = thd_vec_t[base_p]                          # per-pixel threshold
            base_p[mx < gate] = bg_idx

            g_v = g_all[vmask]
            p_v = base_p[vmask]
            np.add.at(cm_base, (g_v.numpy(), p_v.numpy()), 1)

            # --- top-2 values and indices over valid pixels ---
            flat_v = flat[:, vmask]                             # (C, V)
            top2_v = flat_v.topk(2, dim=0)                     # values (2,V), indices (2,V)
            top1_idx = top2_v.indices[0]                        # (V,) = winner class
            top2_idx = top2_v.indices[1]                        # (V,) = runner-up class
            margin_v = top2_v.values[0] - top2_v.values[1]     # (V,)

            # --- backfill bucket (margin-independent) ---
            # raw top1 is foreground but prob_thd reverted the pixel to bg.
            # Recoverable by a class-specific threshold, not by top1->top2 rerank.
            top1val_v = top2_v.values[0]                         # (V,) = max score
            gate_v = thd_vec_t[top1_idx]                          # per-pixel threshold
            backfilled = (top1val_v < gate_v) & (top1_idx != bg_idx) & (p_v == bg_idx)
            if backfilled.any():
                bf_top1 = top1_idx[backfilled]
                bf_gt   = g_v[backfilled]
                bf_resc = bf_gt == bf_top1                       # threshold could recover
                n_backfill        += int(backfilled.sum())
                n_backfill_rescue += int(bf_resc.sum())
                for c1, r in zip(bf_top1.numpy(), bf_resc.numpy()):
                    bf_pair[int(c1)][0] += 1
                    bf_pair[int(c1)][1] += int(r)

            # STRICT candidate for the top1->top2 oracle: only pixels whose final
            # prediction still equals the RAW top1 and is not background. This
            # isolates genuine "CLIP picks between top1 and top2" cases and excludes
            # the backfill-to-bg pixels accounted for separately above.
            not_backfilled = (p_v == top1_idx) & (p_v != bg_idx)

            for t in thresholds:
                lm = (margin_v < t) & not_backfilled            # (V,) strict candidate

                # always start from baseline; oracle only deviates on lm pixels
                full_oracle = p_v.clone()

                if lm.any():
                    g_lm   = g_v[lm]
                    w_lm   = top1_idx[lm]                       # raw top1 == effective winner
                    ru_lm  = top2_idx[lm]                       # runner-up (raw)

                    is_correct = g_lm == w_lm                   # GT == winner
                    is_rescue  = (~is_correct) & (g_lm == ru_lm)  # GT == runner-up, mutually exclusive
                    is_other   = (~is_correct) & (~is_rescue)

                    n_lm[t]      += int(lm.sum())
                    n_correct[t] += int(is_correct.sum())
                    n_rescue[t]  += int(is_rescue.sum())
                    n_other[t]   += int(is_other.sum())

                    # oracle prediction: flip only where GT==runner-up
                    oracle_p = w_lm.clone()
                    oracle_p[is_rescue] = ru_lm[is_rescue]

                    lm_idx = torch.nonzero(lm).squeeze(1)
                    full_oracle[lm_idx] = oracle_p

                    # pair-level counts for rescue/correct/other
                    w_np  = w_lm.numpy()
                    ru_np = ru_lm.numpy()
                    for i, (w, ru) in enumerate(zip(w_np, ru_np)):
                        key = (int(w), int(ru))
                        if is_correct[i]:
                            pair_counts[t][key][0] += 1
                        elif is_rescue[i]:
                            pair_counts[t][key][1] += 1
                        else:
                            pair_counts[t][key][2] += 1

                # always accumulate cm_oracle (even if no lm pixels in this image)
                np.add.at(cm_oracle[t], (g_v.numpy(), full_oracle.numpy()), 1)

        n_done += len(out)
        if n_done % 50 == 0:
            print(f"  processed {n_done} images ...")
        if args.limit and n_done >= args.limit:
            break

    print(f"\nTotal images: {n_done}\n")

    base_iou  = iou_from_cm(cm_base)
    base_miou = np.nanmean(base_iou) * 100

    print(f"Baseline mIoU: {base_miou:.2f}%\n")
    print("=" * 70)

    for t in thresholds:
        oracle_iou  = iou_from_cm(cm_oracle[t])
        oracle_miou = np.nanmean(oracle_iou) * 100
        gain        = oracle_miou - base_miou

        tot_v = max(cm_base.sum(), 1)          # total valid pixels
        lm    = n_lm[t]
        nc    = n_correct[t]
        nr    = n_rescue[t]
        no    = n_other[t]

        print(f"\nmargin_thr = {t}")
        print(f"  Low-margin pixels : {lm:,}  ({lm/tot_v*100:.2f}% of valid)")
        print(f"    GT==winner (correct, flip=damage): {nc:,}  ({nc/max(lm,1)*100:.1f}%)")
        print(f"    GT==runnerup (rescuable)         : {nr:,}  ({nr/max(lm,1)*100:.1f}%)")
        print(f"    GT==other  (lateral)             : {no:,}  ({no/max(lm,1)*100:.1f}%)")
        print(f"  Oracle mIoU : {oracle_miou:.2f}%   ceiling = {gain:+.2f} mIoU pts")

        # per-class delta
        print(f"\n  Per-class IoU (baseline vs oracle):")
        print(f"    {'class':<14}{'base':>8}{'oracle':>8}{'Δ':>8}")
        order = np.argsort(np.nan_to_num(base_iou, nan=-1))
        for c in order:
            b = base_iou[c] * 100
            v = oracle_iou[c] * 100
            d = v - b
            print(f"    {class_names[c]:<14}{b:8.2f}{v:8.2f}  {'+' if d>=0 else ''}{d:.2f}")
        print(f"    {'mIoU':<14}{base_miou:8.2f}{oracle_miou:8.2f}  "
              f"{'+' if gain>=0 else ''}{gain:.2f}")

        # top winner->runnerup pairs, sorted by rescue (upside), with the
        # break-even precision a real arbiter must beat on THIS pair to net
        # positive:  breakeven = correct / (correct + rescue).
        #   low breakeven + high rescue = a great CLIP target (big upside,
        #   easy precision bar). high breakeven = CLIP must be near-perfect.
        pc = pair_counts[t]
        pairs = [(k, v) for k, v in pc.items()]
        pairs.sort(key=lambda x: x[1][1], reverse=True)  # sort by rescue (upside)

        print(f"\n  Top {args.top_pairs} winner->runnerup pairs (by rescuable pixels):")
        print(f"    {'winner':<14}{'->':^4}{'runnerup':<14}"
              f"{'correct':>10}{'rescue':>10}{'other':>8}{'net':>8}{'breakeven':>11}")
        for (w, ru), counts in pairs[:args.top_pairs]:
            c_cnt, r_cnt, o_cnt = int(counts[0]), int(counts[1]), int(counts[2])
            net = r_cnt - c_cnt
            denom = c_cnt + r_cnt
            breakeven = c_cnt / denom if denom > 0 else float("nan")
            wn  = class_names[w]  if w  < C else f"cls{w}"
            run = class_names[ru] if ru < C else f"cls{ru}"
            print(f"    {wn:<14}{'->':^4}{run:<14}"
                  f"{c_cnt:10,}{r_cnt:10,}{o_cnt:8,}{net:+8,}{breakeven*100:10.1f}%")
        print("    breakeven = correct/(correct+rescue): min arbiter precision to")
        print("    net positive on this pair. Low breakeven + high rescue = best target.")

        print("=" * 70)

    # ---- backfill bucket: the class-specific-threshold opportunity ----
    tot_v = max(cm_base.sum(), 1)
    print(f"\nBackfill bucket (prob_thd={prob_thd} reverting foreground top1 -> bg):")
    print(f"  Backfilled foreground pixels : {n_backfill:,}  "
          f"({n_backfill/tot_v*100:.2f}% of valid)")
    print(f"    GT == raw top1 (threshold-recoverable): {n_backfill_rescue:,}  "
          f"({n_backfill_rescue/max(n_backfill,1)*100:.1f}%)")
    print(f"  These are recoverable by a class-specific / lower threshold, NOT by "
          f"top1->top2 rerank.")
    if n_backfill > 0:
        rows = sorted(bf_pair.items(), key=lambda kv: kv[1][1], reverse=True)
        print(f"\n  Top backfilled classes (raw top1 reverted to bg):")
        print(f"    {'class':<14}{'backfilled':>12}{'GT==top1':>12}{'recover%':>10}")
        for c1, (tot, resc) in rows[:args.top_pairs]:
            name = class_names[c1] if c1 < C else f"cls{c1}"
            pct = resc / max(tot, 1) * 100
            print(f"    {name:<14}{int(tot):12,}{int(resc):12,}{pct:9.1f}%")
        print("    high GT==top1 % => a lower threshold for that class recovers it "
              "directly (cheap, no CLIP).")
    print("=" * 70)

    print("\nInterpretation guide:")
    print("  NOTE: the oracle is DOUBLE-OPTIMISTIC (perfect rescue + zero damage).")
    print("  A real arbiter (CLIP) also damages GT==winner pixels, so realistic")
    print("  gain is a fraction of the ceiling. Decide on these, not the ceiling alone:")
    print("    1. per-class oracle Δ : which classes actually have headroom")
    print("       (read this, NOT the aggregate rescuable%, since mIoU is per-class).")
    print("    2. a pair with HIGH rescue AND LOW breakeven precision")
    print("       => CLIP has big upside and an easy precision bar => good target.")
    print("    3. if every pair has high breakeven (>~60%) or rescue is tiny")
    print("       => CLIP must be near-perfect to help => not worth building.")


if __name__ == "__main__":
    main()
