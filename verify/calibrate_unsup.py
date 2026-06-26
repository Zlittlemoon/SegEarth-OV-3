"""
Unsupervised per-class score calibration validation.

Hypothesis (from logit_dist.py): some class channels fire high *outside* their
own region (high out-of-class scale), so the final argmax wrongly favours them.
A training-free fix down-weights only those over-firing channels, using scale
estimates computed without any GT (works for any vocabulary).

Key unsupervised quantities (per image, no labels):
  raw_pred        = argmax (NO prob_thd backfill) -> defines win/non-win regions
  base_pred       = argmax + prob_thd backfill (mirrors predict()) -> eval + bg guard
  nonwinner_tau_c = q-th quantile of channel c over pixels where c did NOT win (raw)
  in_ref_c        = in_q-th quantile of channel c over pixels where c DID win (raw)
  overfire_ratio_c = nonwinner_tau_c / in_ref_c
       > 1 => channel fires higher outside its own region than inside
              = pseudo over-firing (OEM pavement, LoveDA agricultural)
       < 1 => well-calibrated (OEM road: high in_ref, lower outside)
  This is a SELF-ESTIMATED (GT-free) signal, hence "selected/pseudo overfire",
  never claimed as ground-truth overfire.

Robust selector (ambiguous mode), all per image, no GT:
  valid_c        = class c has >= min_win winning pixels (else ratio is unstable)
  ratio_c > high_tau_thr            (relative: leaks high vs own confident response)
  tau_c   > leak_tau_mult * median(tau)   (absolute: leak strength not trivially low)
  Only winners passing ALL THREE, on low-margin pixels, get suppressed.

Modes (--mode):
  normalize   scale_c = tau_c (aggressive control; BREAKS prob_thd, expect drop)
  suppress    global down-weight: score[c] /= (tau_c/median(tau))**alpha clamped.
  ambiguous   the real method: only suppress the current winner on low-margin
              (top1-top2 < margin_thr) pixels whose winner is a selected-overfire
              class. Confident pixels and healthy classes are untouched.

Diagnostics: global AND per-winner-class pixel-flow (touched / changed / rescue /
damage), so degradation can be attributed to specific classes (is pavement the
one being touched? is road wrongly selected? who causes the damage?).
Baseline reproduces predict() exactly; mIoU uses nanmean over present classes.

Run:
    python verify/calibrate_unsup.py configs/cfg_openearthmap.py \
        --mode ambiguous --alphas 0.2 0.5 1.0 --margin-thr 0.2 \
        --high-tau-thr 1.0 --in-q 0.75 --leak-tau-mult 1.0 --limit 80
    python verify/calibrate_unsup.py configs/cfg_loveda.py \
        --mode ambiguous --alphas 0.2 0.5 1.0 --margin-thr 0.2 \
        --high-tau-thr 1.0 --in-q 0.75 --leak-tau-mult 1.0 --limit 80
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
    ap = argparse.ArgumentParser(description="Unsupervised calibration validation")
    ap.add_argument("config")
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N images (full set if omitted)")
    ap.add_argument("--tau-mode", choices=["all", "nonwinner"], default="nonwinner",
                    help="how to estimate the per-class scale tau_c (no GT either way)")
    ap.add_argument("--mode", choices=["suppress", "normalize", "ambiguous"],
                    default="suppress",
                    help="suppress: global per-class scale down (prob_thd-safe). "
                         "normalize: divide by tau_c (aggressive control). "
                         "ambiguous: only suppress the current winner on low-margin "
                         "pixels where the winner is a selected-overfire class.")
    ap.add_argument("--q", type=float, default=0.95,
                    help="quantile level for nonwinner tau_c (the OUT-region tail)")
    ap.add_argument("--in-q", type=float, default=0.75,
                    help="quantile level for the IN-region reference in_ref_c. "
                         "Higher = compare outside leak against a more confident "
                         "in-region response (more conservative selector). q75 default; "
                         "median is recovered with --in-q 0.5.")
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.2, 0.3, 0.5, 1.0],
                    help="suppress/ambiguous strength(s) to sweep")
    ap.add_argument("--clamp", nargs=2, type=float, default=[1.0, 1.5],
                    metavar=("LO", "HI"),
                    help="scale clamp; LO>=1.0 keeps it suppress-only")
    ap.add_argument("--margin-thr", type=float, default=0.2,
                    help="ambiguous mode: only touch pixels with top1-top2 < this")
    ap.add_argument("--high-tau-thr", type=float, default=1.0,
                    help="ambiguous: relative condition. A class is selected-overfire "
                         "if overfire_ratio (nonwinner_tau / in_ref) exceeds this.")
    ap.add_argument("--leak-tau-mult", type=float, default=1.0,
                    help="ambiguous: absolute condition. A class is selected-overfire "
                         "only if its nonwinner_tau also exceeds "
                         "leak_tau_mult * median(tau). Guards against globally-low "
                         "classes (water/background) getting a large ratio for free.")
    ap.add_argument("--min-win-pixels", type=int, default=256,
                    help="min raw-argmax winning pixels for a class to be eligible "
                         "as overfire (else in_ref is unstable -> ratio explodes).")
    ap.add_argument("--min-win-frac", type=float, default=0.0005,
                    help="same guard as a fraction of image pixels; the effective "
                         "threshold is max(min_win_pixels, min_win_frac * P).")
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--cfg-options", nargs="+", action=DictAction, default=None)
    return ap.parse_args()


def iou_from_cm(cm):
    """Per-class IoU; absent classes (union==0) -> NaN so nanmean ignores them."""
    tp = np.diag(cm).astype(np.float64)
    gt = cm.sum(1).astype(np.float64)
    pd = cm.sum(0).astype(np.float64)
    union = gt + pd - tp
    iou = np.full(tp.shape, np.nan, dtype=np.float64)
    valid = union > 0
    iou[valid] = tp[valid] / union[valid]
    return iou


def estimate_tau(flat, tau_mode, q, in_q, min_win, eps):
    """Per-class tau_c, overfire_ratio_c, valid_c. No GT, no prob_thd dependence.

    win/non-win regions use the RAW argmax (NOT prob_thd backfill) so in_ref is
    not distorted by low-confidence winners being reverted to background.

      tau_c    = q-th quantile of channel c OUTSIDE its winning pixels.
      in_ref_c = in_q-th quantile of channel c INSIDE its winning pixels.
      overfire_ratio = tau_c / (in_ref_c + eps).
      valid_c  = winning pixels >= min_win  (else ratio is statistically unstable;
                 such a class is NOT eligible to be flagged overfire).

    Returns taus (C,), overfire_ratio (C,), valid (C,) bool.
    """
    C = flat.shape[0]
    raw_pred = flat.argmax(0)          # raw argmax, unaffected by prob_thd
    taus = torch.empty(C, dtype=flat.dtype)
    ratio = torch.zeros(C, dtype=flat.dtype)
    valid = torch.zeros(C, dtype=torch.bool)
    for c in range(C):
        win_mask = raw_pred == c       # pixels where c wins the raw argmax
        win_vals = flat[c][win_mask]
        non_vals = flat[c][~win_mask]
        if tau_mode == "all":
            taus[c] = flat[c].quantile(q)
        else:
            taus[c] = non_vals.quantile(q) if non_vals.numel() > 0 else flat[c].quantile(q)
        # small-sample guard: too few winners -> in_ref unstable -> not eligible.
        if win_vals.numel() >= min_win:
            in_ref = win_vals.quantile(in_q)
            ratio[c] = taus[c] / (in_ref + eps)
            valid[c] = True
        # else: ratio stays 0, valid stays False
    return taus, ratio, valid


def compute_scale(tau, mode, alpha, clamp_lo, clamp_hi, eps):
    if mode == "normalize":
        return tau + eps
    # suppress-only relative scale: tau_ratio decoupled from overfire detection
    ref = tau.median()
    scale = (tau / (ref + eps)).clamp(min=eps) ** alpha
    return scale.clamp(clamp_lo, clamp_hi)


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
    clamp_lo, clamp_hi = args.clamp

    if args.mode == "normalize":
        settings = ["normalize"]
    else:
        settings = [(args.mode, a) for a in sorted(args.alphas)]

    print(f"Config : {args.config}")
    print(f"Classes ({C}): {class_names}")
    print(f"Infer mode : {cfg.model.get('sam3_infer_mode', 'serial')}")
    print(f"prob_thd={prob_thd}  bg_idx={bg_idx}")
    print(f"tau-mode={args.tau_mode}  q={args.q}  in_q={args.in_q}  mode={args.mode}")
    if args.mode in ("suppress", "ambiguous"):
        print(f"alphas={sorted(args.alphas)}  clamp=[{clamp_lo}, {clamp_hi}]")
    if args.mode == "ambiguous":
        print(f"margin_thr={args.margin_thr}  high_tau_thr={args.high_tau_thr}  "
              f"leak_tau_mult={args.leak_tau_mult}")
        print(f"min_win_pixels={args.min_win_pixels}  min_win_frac={args.min_win_frac}")
    print(f"Limit  : {args.limit or 'all'} images\n")

    model = MODELS.build(cfg.model)
    model.processor.model.eval()
    loader = Runner.build_dataloader(cfg.test_dataloader)

    cm_base = np.zeros((C, C), dtype=np.int64)
    cms = {s: np.zeros((C, C), dtype=np.int64) for s in settings}
    # global pixel-flow diagnostics (over valid pixels)
    diag = {s: dict(changed=0, rescue=0, damage=0, to_bg=0, from_bg=0, touched=0, total=0)
            for s in settings}
    # per-winner-class pixel-flow diagnostics
    diag_cls = {s: dict(touched=np.zeros(C, np.int64), changed=np.zeros(C, np.int64),
                        rescue=np.zeros(C, np.int64), damage=np.zeros(C, np.int64))
                for s in settings}
    tau_acc = np.zeros(C, dtype=np.float64)
    ratio_acc = np.zeros(C, dtype=np.float64)
    valid_acc = np.zeros(C, dtype=np.float64)   # how often each class was eligible
    sel_acc = np.zeros(C, dtype=np.float64)     # how often each class was selected-overfire
    n_images = 0

    def backfill(score):
        """argmax + prob_thd backfill, mirroring predict()."""
        p = score.argmax(0).clone()
        mx = score.max(0).values
        p[mx < prob_thd] = bg_idx
        return p

    def selected_overfire(tau, ratio, valid):
        """The three-condition selector (C,) bool, all GT-free, per image."""
        tau_ref = tau.median()
        return (valid
                & (ratio > args.high_tau_thr)
                & (tau > args.leak_tau_mult * tau_ref))

    def apply_cal(flat, base_pred, true_overfire, scale, mode):
        """Return (score_cal (C,P), touched_count, sel (P,) bool or None)."""
        if mode in ("normalize", "suppress"):
            return flat / scale.unsqueeze(1), None, None
        # ambiguous: only suppress the winner on low-margin pixels whose winner
        # is a selected-overfire class (relative + absolute + min-win conditions).
        score_cal = flat.clone()
        top2 = flat.topk(2, dim=0).values                    # (2, P)
        margin = top2[0] - top2[1]                            # (P,)
        sel = (base_pred != bg_idx) & true_overfire[base_pred] & (margin < args.margin_thr)
        touched = int(sel.sum())
        if sel.any():
            idx = torch.nonzero(sel).squeeze(1)
            win = base_pred[idx]
            score_cal[win, idx] = flat[win, idx] / scale[win]
        return score_cal, touched, sel

    n_done = 0
    for batch in loader:
        out = model.predict(batch["inputs"], batch["data_samples"])
        for ds in out:
            logits = ds.seg_logits.data.float().cpu()           # (C, H, W)
            gt     = ds.gt_sem_seg.data.squeeze(0).long().cpu()  # (H, W)
            valid  = (gt >= 0) & (gt < C)
            g      = gt[valid].numpy()
            flat   = logits.reshape(C, -1)                       # (C, P)
            vmask  = valid.reshape(-1)
            P      = flat.shape[1]
            min_win = max(args.min_win_pixels, int(args.min_win_frac * P))

            # baseline eval prediction (argmax + prob_thd backfill)
            base_pred = backfill(flat)
            p_base_t = base_pred[vmask]
            np.add.at(cm_base, (g, p_base_t.numpy()), 1)
            gt_v = gt[valid]

            # GT-free per-class scale, overfire ratio, eligibility.
            tau, ratio, vld = estimate_tau(flat, args.tau_mode, args.q,
                                           args.in_q, min_win, args.eps)
            true_overfire = selected_overfire(tau, ratio, vld)
            tau_acc += tau.numpy()
            ratio_acc += ratio.numpy()
            valid_acc += vld.numpy().astype(np.float64)
            sel_acc += true_overfire.numpy().astype(np.float64)

            for s in settings:
                if s == "normalize":
                    scale = compute_scale(tau, "normalize", None,
                                          clamp_lo, clamp_hi, args.eps)
                    mode = "normalize"
                else:
                    mode, alpha = s
                    scale = compute_scale(tau, "suppress", alpha,
                                          clamp_lo, clamp_hi, args.eps)
                score_cal, touched, sel = apply_cal(flat, base_pred,
                                                    true_overfire, scale, mode)
                diag[s]["touched"] += touched or 0
                p_cal_t = backfill(score_cal)[vmask]
                np.add.at(cms[s], (g, p_cal_t.numpy()), 1)

                # ---- global pixel-flow ----
                d = diag[s]
                changed = p_cal_t != p_base_t
                base_right = p_base_t == gt_v
                cal_right = p_cal_t == gt_v
                d["total"]   += int(vmask.sum())
                d["changed"] += int(changed.sum())
                d["rescue"]  += int(((~base_right) & cal_right).sum())
                d["damage"]  += int((base_right & (~cal_right)).sum())
                d["to_bg"]   += int(((p_base_t != bg_idx) & (p_cal_t == bg_idx)).sum())
                d["from_bg"] += int(((p_base_t == bg_idx) & (p_cal_t != bg_idx)).sum())

                # ---- per-winner-class pixel-flow ----
                dc = diag_cls[s]
                sel_v = sel[vmask] if sel is not None else None
                for c in range(C):
                    clsm = p_base_t == c
                    if not clsm.any():
                        continue
                    dc["changed"][c] += int((changed & clsm).sum())
                    dc["rescue"][c]  += int(((~base_right) & cal_right & clsm).sum())
                    dc["damage"][c]  += int((base_right & (~cal_right) & clsm).sum())
                    if sel_v is not None:
                        dc["touched"][c] += int((sel_v & clsm).sum())

            n_images += 1

        n_done += len(out)
        if n_done % 50 == 0:
            print(f"  processed {n_done} images ...")
        if args.limit and n_done >= args.limit:
            break

    print(f"\nTotal images: {n_done}\n")

    base_iou = iou_from_cm(cm_base)
    base_miou = np.nanmean(base_iou) * 100

    def label(s):
        return "normalize" if s == "normalize" else f"{s[0]} a={s[1]:.2f}"

    print("mIoU comparison (%):")
    print(f"  {'baseline':<18}: {base_miou:6.2f}")
    best_s, best_miou = None, base_miou
    for s in settings:
        miou = np.nanmean(iou_from_cm(cms[s])) * 100
        dd = miou - base_miou
        print(f"  {label(s):<18}: {miou:6.2f}  ({'+' if dd>=0 else ''}{dd:.2f})")
        if miou > best_miou:
            best_miou, best_s = miou, s

    # ---- global pixel-flow ----
    print("\nGlobal pixel-flow (% of valid pixels):")
    print(f"  {'setting':<18}{'touched':>9}{'changed':>9}{'rescue':>8}{'damage':>8}"
          f"{'net':>7}{'to_bg':>8}{'from_bg':>9}")
    for s in settings:
        d = diag[s]
        tot = max(d["total"], 1)
        net = (d["rescue"] - d["damage"]) / tot * 100
        print(f"  {label(s):<18}{d['touched']/tot*100:8.2f}%"
              f"{d['changed']/tot*100:8.2f}%"
              f"{d['rescue']/tot*100:7.2f}%{d['damage']/tot*100:7.2f}%"
              f"{net:+7.2f}{d['to_bg']/tot*100:7.2f}%{d['from_bg']/tot*100:8.2f}%")
    print("  touched=selected by sel mask; changed=argmax actually flipped")

    # choose a setting to show in detail
    if best_s is not None:
        show_s, tag = best_s, "best"
    elif settings:
        show_s = min(settings,
                     key=lambda s: abs(np.nanmean(iou_from_cm(cms[s])) - base_miou / 100))
        tag = "least-degraded"
    else:
        show_s, tag = None, ""

    if show_s is not None:
        # ---- per-class IoU ----
        show_iou = iou_from_cm(cms[show_s])
        show_miou = np.nanmean(show_iou) * 100
        print(f"\nPer-class IoU: baseline vs {tag} ({label(show_s)}) (%):")
        print(f"  {'class':<14}{'base':>8}{'cal':>8}{'Δ':>8}")
        order = np.argsort(np.nan_to_num(base_iou, nan=-1))
        for c in order:
            b, v = base_iou[c] * 100, show_iou[c] * 100
            dd = v - b
            print(f"  {class_names[c]:<14}{b:8.2f}{v:8.2f}  {'+' if dd>=0 else ''}{dd:.2f}")
        print(f"  {'mIoU':<14}{base_miou:8.2f}{show_miou:8.2f}"
              f"  {'+' if show_miou-base_miou>=0 else ''}{show_miou-base_miou:.2f}")

        # ---- per-winner-class pixel-flow for the shown setting ----
        dc = diag_cls[show_s]
        tot = max(diag[show_s]["total"], 1)
        print(f"\nPer-winner-class pixel-flow for {tag} ({label(show_s)}) "
              f"(% of all valid pixels):")
        print(f"  {'winner class':<14}{'touched':>9}{'changed':>9}"
              f"{'rescue':>8}{'damage':>8}{'net':>7}")
        order = np.argsort(dc["touched"] + dc["changed"])[::-1]
        for c in order:
            t = dc["touched"][c] / tot * 100
            ch = dc["changed"][c] / tot * 100
            r = dc["rescue"][c] / tot * 100
            dm = dc["damage"][c] / tot * 100
            if t == 0 and ch == 0:
                continue
            print(f"  {class_names[c]:<14}{t:8.3f}%{ch:8.3f}%"
                  f"{r:7.3f}%{dm:7.3f}%{(r-dm):+7.3f}")

    # ---- unsupervised overfire diagnostics ----
    mean_tau = tau_acc / max(n_images, 1)
    mean_ratio = ratio_acc / np.maximum(valid_acc, 1)   # avg over images where eligible
    sel_frac = sel_acc / max(n_images, 1)
    valid_frac = valid_acc / max(n_images, 1)
    print(f"\nUnsupervised overfire diagnostics (tau-mode={args.tau_mode}, "
          f"q={args.q}, in_q={args.in_q}):")
    print("  overfire_ratio = nonwinner_tau / in_ref  (>1 fires more outside than inside)")
    print("  selected = ratio>high_tau_thr AND tau>leak_mult*median_tau AND eligible")
    print(f"  {'class':<14}{'nonwin_tau':>12}{'ratio':>9}"
          f"{'valid%':>9}{'sel%':>8}")
    for c in np.argsort(sel_frac)[::-1]:
        mark = "  <-- selected overfire" if sel_frac[c] > 0.5 else ""
        print(f"  {class_names[c]:<14}{mean_tau[c]:12.4f}{mean_ratio[c]:9.3f}"
              f"{valid_frac[c]*100:8.1f}%{sel_frac[c]*100:7.1f}%{mark}")
    print(f"\n  high_tau_thr={args.high_tau_thr}  leak_tau_mult={args.leak_tau_mult}  "
          f"(sel% = fraction of images where the class was selected-overfire)")


if __name__ == "__main__":
    main()