"""
Distribution-Guided Class-Selective Foreground Recovery — v2 Diagnostic Script

Implements the three-stage CAFR framework using only the unsupervised winner-score
cache (no ground-truth labels used in the method itself; GT is used only for
evaluating mIoU after applying the inferred thresholds).

Stage 0  Dataset-Adaptive Background Gating
         Estimates whether the dataset's background gate is primarily
         "semantic-bg dominant", "uncertainty-bg dominant", or "hybrid",
         using label-free proxy statistics.
         Decisive probe: naive-q0 effect (mIoU change when gate turned off).

Stage 1  Class Eligibility Selector  (v2: candidate-topk + mode-aware GMM risk)
         For each foreground class, computes:
           suppressed_mass(c): fraction of winner pixels below global_thd
           score_safety(c):  1 - P(suppressed score < τ/2)  [v1 proxy]
           rel_band_mass(c): P(λτ ≤ score < τ | class)       [v2]
           high_anchor(c):   P(score ≥ τ | class)             [v2]
           gmm_tau(c):       GMM crossing on suppressed scores [v2 risk signal]
           mean_margin(c):   mean(top1-score - top2-score) in suppressed region
           Eligibility(c) = rel_band_mass × high_anchor  [v2 release_band mode]
                          or suppressed_mass × score_safety  [v1 score_safety mode]
           GMM risk (uncertainty-bg mode only):
             gmm_risk = clip((gmm_tau - λτ) / (τ - λτ), 0, 1)
             elig = base × (1 - gmm_risk)
         Candidate generation: top-K by elig score, then expand via ambiguous-gap.

Stage 2  Threshold Estimator (applied only to eligible classes)
         Three estimators compared side-by-side, ALL estimated on SUPPRESSED
         scores only (score<global_thd) and clamped ≤ global_thd:
           λ-rule:   τ_c = λ · τ_global   (λ=0.2 from empirical observation)
           Otsu:     maximise inter-class variance of suppressed-score histogram
           GMM:      2-component Gaussian mixture; threshold at posterior crossing
                     NOTE: low score ≠ FP in our task — GMM used only after
                     eligibility is confirmed, with sanity checks.

Usage:
    # OEM: validate Stage 2 with known-correct eligible classes
    python verify/cafr_diagnose.py configs/cfg_openearthmap.py --limit 80 \
        --hand-tuned pavement=0.02,grass=0.02 --force-eligible pavement,grass

    # OEM: run automatic v1 selector for diagnosis
    python verify/cafr_diagnose.py configs/cfg_openearthmap.py --limit 80 \
        --hand-tuned pavement=0.02,grass=0.02

    # LoveDA: validate Stage 2 with known-correct eligible class
    python verify/cafr_diagnose.py configs/cfg_loveda.py --limit 80 \
        --hand-tuned water=0.10 --force-eligible water

    # LoveDA: v2 release-band + mode-aware GMM risk + candidate-topk
    python verify/cafr_diagnose.py configs/cfg_loveda.py --limit 80 \
        --hand-tuned water=0.10 --elig-mode release_band \
        --candidate-topk 3 --ambiguous-gap 0.05

    # LoveDA: run automatic v1 selector for diagnosis (selects forest, not water)
    python verify/cafr_diagnose.py configs/cfg_loveda.py --limit 80 \
        --hand-tuned water=0.10 --max-eligible 1 --min-suppressed-mass 0.05

Notes:
  - Stage 0 outputs a DIAGNOSTIC SUGGESTION driven by the naive-q0 effect.
  - Stage 1 'score_safety' is a score-based proxy, NOT true release safety.
  - GMM risk is ONLY applied in uncertainty-bg dominant mode; it demotes classes
    whose suppressed distribution is bimodal with the low mode close to the gate,
    indicating their suppressed pixels are ambiguous uncertainty-bg, not fg.
  - Stage 2 Otsu/GMM estimated on SUPPRESSED scores and clamped ≤ global_thd.
  - candidate-topk generates a candidate set; ambiguous-gap expands it to avoid
    hard top-1 decisions when two classes are nearly tied in eligibility.
"""

import argparse

import numpy as np
import torch
from scipy.ndimage import binary_dilation
from scipy.ndimage import label as cc_label
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmseg.registry import MODELS

import mmseg.datasets
import segearthov3_segmentor_merge
import custom_datasets

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="CAFR v1 diagnostic: Stage 0/1/2 unsupervised analysis")
    ap.add_argument("config")
    ap.add_argument("--limit", type=int, default=None,
                    help="max images to process (default: all)")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.2,
                    help="relaxation factor for λ-rule: τ_c = λ·τ_global (default 0.2)")
    ap.add_argument("--elig-pct", type=float, default=50.0,
                    help="eligibility percentile: classes above this percentile of "
                         "the eligibility score are candidate-eligible (default 50). "
                         "Combined (AND) with --min-suppressed-mass and --max-eligible.")
    ap.add_argument("--min-suppressed-mass", type=float, default=0.05,
                    help="absolute floor: a class must have suppressed_mass >= this "
                         "to be eligible, regardless of percentile (default 0.05). "
                         "Prevents selecting classes the gate barely touches.")
    ap.add_argument("--max-eligible", type=int, default=None,
                    help="cap the number of eligible classes to the top-K by "
                         "eligibility score (default: no cap). Use e.g. 2 on LoveDA "
                         "to avoid selecting building/barren/road alongside water.")
    ap.add_argument("--min-pixels", type=int, default=500,
                    help="skip classes with fewer winner pixels (default 500)")
    ap.add_argument("--gmm-max-samples", type=int, default=200000,
                    help="subsample winner scores to at most this many points before "
                         "fitting GMM (default 200000). Otsu always uses the full "
                         "histogram; only GMM is subsampled.")
    ap.add_argument("--elig-mode", choices=["score_safety", "release_band"],
                    default="score_safety",
                    help="eligibility scoring mode (default: score_safety = v1 behaviour). "
                         "'release_band' uses rel_band_mass * high_anchor instead: "
                         "rel_band = P(λτ ≤ score < τ | class), "
                         "high_anchor = P(score ≥ τ | class). "
                         "In release_band mode, GMM risk soft-penalty is automatically "
                         "applied when Stage 0 detects uncertainty-bg dominant policy.")
    ap.add_argument("--candidate-topk", type=int, default=None,
                    help="generate a candidate set of the top-K classes by eligibility "
                         "score, then optionally expand via --ambiguous-gap. This avoids "
                         "hard top-1 selection when two classes are nearly tied. "
                         "Combined with --max-eligible cap after gap expansion. "
                         "Example: --candidate-topk 3 (default: disabled, all eligible "
                         "classes from percentile/min-suppressed-mass filter are used).")
    ap.add_argument("--ambiguous-gap", type=float, default=0.05,
                    help="relative gap threshold for candidate set expansion: include a "
                         "class if its elig score is within gap of the top-1 elig score "
                         "(i.e. elig >= top1_elig - ambiguous_gap). Only active when "
                         "--candidate-topk is set. Default 0.05.")
    ap.add_argument("--force-eligible", type=str, default=None,
                    help="comma-separated class names to FORCE into the eligible set, "
                         "bypassing the Stage 1 selector entirely (e.g. 'water' or "
                         "'pavement,grass'). Use to validate Stage 2 estimators given "
                         "a known-correct class selection, decoupled from selector bugs.")
    ap.add_argument("--hand-tuned", type=str, default=None,
                    help="comma-separated class=threshold pairs for hand-tuned "
                         "reference, e.g. 'pavement=0.02,grass=0.02'")
    ap.add_argument("--spatial-radius", type=int, default=5,
                    help="dilation radius (pixels) for spatial anchor-touch computation "
                         "(default: 5). A band pixel is considered 'touching' an anchor "
                         "if it lies within this many pixels of a high-confidence anchor "
                         "pixel of the same (or competitor) class.")
    ap.add_argument("--spatial-thr", type=float, default=None,
                    help="anchor-coherence threshold for the spatial hard filter "
                         "(semantic-bg dominant mode only). "
                         "anchor_coherence = anc_tch × (1 - island_ratio). "
                         "Classes with anchor_coherence < spatial_thr are removed from "
                         "the eligible set AFTER the primary eligibility selection. "
                         "Not applied in uncertainty-bg or hybrid modes. "
                         "Example: --spatial-thr 0.5 keeps only classes whose band "
                         "pixels are well-connected to own high-confidence anchors. "
                         "(default: None = disabled)")
    ap.add_argument("--cfg-options", nargs="+", action=DictAction, default=None)
    return ap.parse_args()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iou_from_cm(cm):
    tp = np.diag(cm).astype(np.float64)
    union = cm.sum(1) + cm.sum(0) - tp
    iou = np.full(len(tp), np.nan)
    ok = union > 0
    iou[ok] = tp[ok] / union[ok]
    return iou

def cm_from_pred(g, p, C):
    return np.bincount(g * C + p, minlength=C * C).reshape(C, C)

def otsu_threshold(scores, n_bins=256):
    """1-D Otsu threshold on a float array in [0, 1]."""
    hist, edges = np.histogram(scores, bins=n_bins, range=(0.0, 1.0))
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    total = hist.sum()
    if total == 0:
        return np.nan
    best_tau, best_var = 0.0, -1.0
    w0, sum0 = 0.0, 0.0
    total_sum = (hist * bin_centers).sum()
    w1_sum = total_sum
    for i in range(n_bins):
        w0 += hist[i]
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue
        sum0 += hist[i] * bin_centers[i]
        mu0 = sum0 / w0
        mu1 = (w1_sum - sum0) / w1
        var_between = (w0 / total) * (w1 / total) * (mu0 - mu1) ** 2
        if var_between > best_var:
            best_var = var_between
            best_tau = bin_centers[i]
    return best_tau

def _gaussian_pdf(x, mu, var):
    var = max(var, 1e-8)
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)

def gmm_threshold(scores, max_iter=200, tol=1e-6, seed=42, max_samples=200000):
    """
    Fit a 2-component 1-D Gaussian mixture by EM (pure numpy, no sklearn) and
    return the posterior crossing point (where the two component responsibilities
    are equal between the two means).

    IMPORTANT: in our task, low score ≠ FP.  We identify the lower-mean
    component only to locate the boundary, not to label it as noise.
    Returns (tau, mu_low, mu_high, converged).
    """
    n = len(scores)
    if n < 20:
        return np.nan, np.nan, np.nan, False
    if n > max_samples:
        rng = np.random.RandomState(seed)
        scores = rng.choice(scores, max_samples, replace=False)
        n = max_samples
    x = scores.astype(np.float64)

    # deterministic init via score quantiles (avoids RNG; reproducible)
    mu = np.array([np.quantile(x, 0.2), np.quantile(x, 0.8)], dtype=np.float64)
    if mu[0] == mu[1]:
        return float(mu[0]), float(mu[0]), float(mu[1]), False
    var = np.array([x.var() + 1e-6, x.var() + 1e-6], dtype=np.float64)
    pi = np.array([0.5, 0.5], dtype=np.float64)

    converged = False
    prev_ll = -np.inf
    for _ in range(max_iter):
        # E-step
        r0 = pi[0] * _gaussian_pdf(x, mu[0], var[0])
        r1 = pi[1] * _gaussian_pdf(x, mu[1], var[1])
        denom = r0 + r1 + 1e-300
        g0 = r0 / denom
        g1 = r1 / denom
        # M-step
        n0, n1 = g0.sum(), g1.sum()
        if n0 < 1e-6 or n1 < 1e-6:
            break
        mu[0] = (g0 * x).sum() / n0
        mu[1] = (g1 * x).sum() / n1
        var[0] = max((g0 * (x - mu[0]) ** 2).sum() / n0, 1e-8)
        var[1] = max((g1 * (x - mu[1]) ** 2).sum() / n1, 1e-8)
        pi[0] = n0 / n
        pi[1] = n1 / n
        ll = np.log(denom).mean()   # mean (not sum) so tol is sample-size invariant
        if abs(ll - prev_ll) < tol:
            converged = True
            break
        prev_ll = ll

    order = np.argsort(mu)
    lo, hi = order[0], order[1]
    mu_lo, mu_hi = mu[lo], mu[hi]

    # locate posterior crossing between the two means
    cands = np.linspace(mu_lo, mu_hi, 2000)
    rlo = pi[lo] * _gaussian_pdf(cands, mu[lo], var[lo])
    rhi = pi[hi] * _gaussian_pdf(cands, mu[hi], var[hi])
    diff = rlo - rhi
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:
        tau = 0.5 * (mu_lo + mu_hi)
    else:
        i = sign_changes[0]
        d0, d1 = diff[i], diff[i + 1]
        tau = cands[i] - d0 * (cands[i + 1] - cands[i]) / (d1 - d0)
    return float(tau), float(mu_lo), float(mu_hi), converged

def predict_with(raw_top1, max_sc, thd_vec, bg_idx):
    pred = raw_top1.copy()
    pred[max_sc < thd_vec[raw_top1]] = bg_idx
    return pred

def report_miou(tag, pred, gt_all, C, base_iou, base_miou, class_names, bg_idx):
    iou = iou_from_cm(cm_from_pred(gt_all, pred, C))
    miou = np.nanmean(iou) * 100
    fg_iou = np.nanmean([iou[c] for c in range(C) if c != bg_idx]) * 100
    delta = miou - base_miou
    print(f"  [{tag}]  mIoU={miou:.2f}%  fg-mIoU={fg_iou:.2f}%  Δ={delta:+.2f}%")
    worst_drop = min(
        (iou[c] - base_iou[c]) * 100
        for c in range(C) if c != bg_idx and not np.isnan(base_iou[c])
    )
    print(f"          worst-fg-drop={worst_drop:+.2f}%")
    return iou, miou

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    global_thd = float(cfg.model.get("prob_thd", 0.0))
    bg_idx = int(cfg.model.get("bg_idx", 0))

    # parse hand-tuned reference
    hand_tuned_thd = np.full(C, global_thd, dtype=np.float32)
    if args.hand_tuned:
        for pair in args.hand_tuned.split(","):
            name, val = pair.strip().split("=")
            name = name.strip()
            val = float(val.strip())
            found = [i for i, n in enumerate(class_names) if n == name]
            if not found:
                print(f"[WARN] hand-tuned class '{name}' not in class list, ignored")
            else:
                hand_tuned_thd[found[0]] = val

    print(f"\n{'='*72}")
    print(f"CAFR Diagnostic v1")
    print(f"Config       : {args.config}")
    print(f"Classes ({C:2d}) : {class_names}")
    print(f"global_thd   : {global_thd}   bg_idx={bg_idx}")
    print(f"λ-rule λ     : {args.lam}   (τ_c = {args.lam:.2f}·{global_thd} = {args.lam*global_thd:.4f})")
    print(f"elig_mode    : {args.elig_mode}")
    if args.candidate_topk is not None:
        print(f"eligibility  : candidate-topk={args.candidate_topk} "
              f"ambiguous-gap={args.ambiguous_gap} "
              f"supp_mass≥{args.min_suppressed_mass}"
              + (f" AND top-{args.max_eligible}" if args.max_eligible else ""))
    else:
        print(f"eligibility  : p{args.elig_pct:.0f} AND supp_mass≥{args.min_suppressed_mass}"
              + (f" AND top-{args.max_eligible}" if args.max_eligible else ""))
    print(f"limit        : {args.limit or 'all'} images")
    print(f"{'='*72}\n")

    # lambda_val needed both in inference loop (spatial band) and Stage 1
    lambda_val = args.lam * global_thd

    # ---- spatial support accumulators ----
    # Computed per-image in 2D; accumulated as pixel counts across all images.
    # All counts are over band pixels (raw_top1==c, lambda_val <= score < global_thd).
    #   sp_n_band[c]      : total band pixels for class c
    #   sp_anc_touch[c]   : band pixels within spatial_radius of own-class anchor
    #   sp_island_px[c]   : band pixels in CCs that never touch own-class anchor
    #   sp_comp_touch[c]  : band pixels within spatial_radius of any other-fg anchor
    #   sp_t2comp_touch[c]: band pixels within spatial_radius of their runner-up class anchor
    def _make_disk(r):
        y, x = np.ogrid[-r:r+1, -r:r+1]
        return (x**2 + y**2 <= r**2)

    disk_struct     = _make_disk(args.spatial_radius)
    sp_n_band       = np.zeros(C, dtype=np.int64)
    sp_anc_touch    = np.zeros(C, dtype=np.int64)
    sp_island_px    = np.zeros(C, dtype=np.int64)
    sp_comp_touch   = np.zeros(C, dtype=np.int64)
    sp_t2comp_touch = np.zeros(C, dtype=np.int64)

    model = MODELS.build(cfg.model)
    model.processor.model.eval()
    loader = Runner.build_dataloader(cfg.test_dataloader)

    # ---- single inference pass ----
    # Cache: top1 class/score, top2 class/score, bg class score
    # top2_class is used for top2_entropy (class competition in suppressed region)
    # bg_sc is used for band_bg_gap (score_c - score_bg in release band)
    top1_chunks, max_chunks, top2_chunks, top2cls_chunks, bgsc_chunks, gt_chunks = \
        [], [], [], [], [], []
    n_done = 0
    for batch in loader:
        out = model.predict(batch["inputs"], batch["data_samples"])
        for ds in out:
            logits = ds.seg_logits.data.float().cpu()
            gt = ds.gt_sem_seg.data.squeeze(0).long().cpu()
            valid = (gt >= 0) & (gt < C)
            flat = logits.reshape(C, -1)
            vmask = valid.reshape(-1)
            if C >= 2:
                top2_vals, top2_idx = flat.topk(2, dim=0)
                mx   = top2_vals[0]     # top1 score
                top1 = top2_idx[0]      # top1 class
                sc2  = top2_vals[1]     # top2 score
                cls2 = top2_idx[1]      # top2 class
            else:
                mx, top1 = flat.max(0)
                sc2  = torch.zeros_like(mx)
                cls2 = torch.zeros_like(top1)
            bg_sc = flat[bg_idx]        # background class score per pixel

            # ---- spatial support: per-image 2D computation ----
            if C >= 2:
                H2d, W2d = logits.shape[1], logits.shape[2]
                top1_2d   = top2_idx[0].reshape(H2d, W2d).numpy().astype(np.int64)
                maxsc_2d  = top2_vals[0].reshape(H2d, W2d).numpy().astype(np.float32)
                top2cls_2d = top2_idx[1].reshape(H2d, W2d).numpy().astype(np.int64)

                # dilated anchor masks per class (one binary_dilation per class)
                dil_anc = {}
                for k in range(C):
                    anc_k = (top1_2d == k) & (maxsc_2d >= global_thd)
                    dil_anc[k] = binary_dilation(anc_k, structure=disk_struct) \
                                 if anc_k.any() else None

                for c_sp in range(C):
                    if c_sp == bg_idx:
                        continue
                    band_c = ((top1_2d == c_sp) &
                              (maxsc_2d >= lambda_val) &
                              (maxsc_2d < global_thd))
                    nb = int(band_c.sum())
                    if nb == 0:
                        continue
                    sp_n_band[c_sp] += nb

                    # anchor touch + island ratio via connected components
                    labeled_c, _ = cc_label(band_c)
                    if dil_anc[c_sp] is not None:
                        sp_anc_touch[c_sp] += int((band_c & dil_anc[c_sp]).sum())
                        # island: pixels in CCs that have no overlap with dilated anchor
                        touching_labels = set(
                            np.unique(labeled_c[band_c & dil_anc[c_sp]]).tolist()
                        )
                        touching_labels.discard(0)
                        if touching_labels:
                            island_px = int(
                                (band_c & ~np.isin(labeled_c, list(touching_labels))).sum()
                            )
                        else:
                            island_px = nb   # no CC touches anchor → all islands
                    else:
                        island_px = nb       # no anchor at all → all islands
                    sp_island_px[c_sp] += island_px

                    # competitor touch: union of dilated anchors of other fg classes
                    comp_dil = None
                    for k in range(C):
                        if k == bg_idx or k == c_sp or dil_anc[k] is None:
                            continue
                        comp_dil = dil_anc[k] if comp_dil is None \
                                   else (comp_dil | dil_anc[k])
                    if comp_dil is not None:
                        sp_comp_touch[c_sp] += int((band_c & comp_dil).sum())

                    # top2-competitor touch: runner-up class anchor per band pixel
                    for k in np.unique(top2cls_2d[band_c]).tolist():
                        k = int(k)
                        if k == bg_idx or dil_anc.get(k) is None:
                            continue
                        band_t2k = band_c & (top2cls_2d == k)
                        sp_t2comp_touch[c_sp] += int((band_t2k & dil_anc[k]).sum())

            top1_chunks.append(top1[vmask].numpy().astype(np.int32))
            max_chunks.append(mx[vmask].numpy().astype(np.float32))
            top2_chunks.append(sc2[vmask].numpy().astype(np.float32))
            top2cls_chunks.append(cls2[vmask].numpy().astype(np.int32))
            bgsc_chunks.append(bg_sc[vmask].numpy().astype(np.float32))
            gt_chunks.append(gt.reshape(-1)[vmask].numpy().astype(np.int32))
        n_done += len(out)
        if n_done % 50 == 0:
            print(f"  processed {n_done} images ...")
        if args.limit and n_done >= args.limit:
            break

    raw_top1  = np.concatenate(top1_chunks).astype(np.int64)
    max_sc    = np.concatenate(max_chunks).astype(np.float32)
    top2_sc   = np.concatenate(top2_chunks).astype(np.float32)
    top2_cls  = np.concatenate(top2cls_chunks).astype(np.int64)
    bg_sc_all = np.concatenate(bgsc_chunks).astype(np.float32)
    gt_all    = np.concatenate(gt_chunks).astype(np.int64)
    del top1_chunks, max_chunks, top2_chunks, top2cls_chunks, bgsc_chunks, gt_chunks
    N = len(gt_all)
    print(f"Total images : {n_done}   valid pixels : {N:,}\n")

    # ---- baseline ----
    base_thd = np.full(C, global_thd, dtype=np.float32)
    base_pred = predict_with(raw_top1, max_sc, base_thd, bg_idx)
    base_iou = iou_from_cm(cm_from_pred(gt_all, base_pred, C))
    base_miou = np.nanmean(base_iou) * 100
    base_fg_miou = np.nanmean([base_iou[c] for c in range(C) if c != bg_idx]) * 100

    # ---- naive-q0: gate fully off (all foreground classes thd≈0) ----
    # This is the single most informative probe for gate policy: if turning the
    # gate off IMPROVES mIoU, the gate is over-suppressing foreground (semantic-bg);
    # if it COLLAPSES, the gate is a necessary uncertainty fallback (uncertainty-bg).
    naive_thd = np.full(C, 0.0, dtype=np.float32)
    naive_thd[bg_idx] = global_thd
    naive_pred = predict_with(raw_top1, max_sc, naive_thd, bg_idx)
    naive_iou = iou_from_cm(cm_from_pred(gt_all, naive_pred, C))
    naive_miou = np.nanmean(naive_iou) * 100
    naive_delta = naive_miou - base_miou

    # =========================================================================
    # STAGE 0 — Dataset-level background gate policy (DIAGNOSTIC, not a decision)
    # =========================================================================
    print(f"{'='*72}")
    print("STAGE 0 — Background Gate Policy  (diagnostic suggestion only)")
    print(f"{'='*72}")

    # pixels suppressed by global gate (argmax≠bg but score<global_thd)
    gate_mask = (raw_top1 != bg_idx) & (max_sc < global_thd)
    # pixels where argmax itself is background
    argbg_mask = (raw_top1 == bg_idx)

    r_gate = gate_mask.sum() / N
    r_argbg = argbg_mask.sum() / N

    gate_scores = max_sc[gate_mask]
    argbg_scores = max_sc[argbg_mask]
    gate_mean = gate_scores.mean() if len(gate_scores) > 0 else 0.0
    argbg_mean = argbg_scores.mean() if len(argbg_scores) > 0 else 0.0

    if gate_mask.sum() > 0:
        gate_class_counts = np.bincount(raw_top1[gate_mask], minlength=C).astype(np.float64)
        gate_class_prob = gate_class_counts / gate_class_counts.sum()
        gate_entropy = -sum(
            p * np.log(p + 1e-12) for p in gate_class_prob if p > 0
        )
        gate_entropy_norm = gate_entropy / np.log(C)
    else:
        gate_entropy_norm = 0.0

    print(f"  -- load statistics (descriptive only) --")
    print(f"  r_gate          : {r_gate*100:.2f}%  "
          f"(raw_top1≠bg but score<{global_thd}, gated to bg)")
    print(f"  r_argbg         : {r_argbg*100:.2f}%  "
          f"(raw_top1=background directly)")
    print(f"  gate-bg mean sc : {gate_mean:.4f}   argmax-bg mean sc : {argbg_mean:.4f}")
    print(f"  gate entropy(n) : {gate_entropy_norm:.3f}  "
          f"(0=one class dominates gated pixels, 1=uniform → risky to release)")
    print(f"\n  -- decisive probe: effect of turning the gate OFF (naive-q0) --")
    print(f"  baseline mIoU   : {base_miou:.2f}%")
    print(f"  gate-off  mIoU  : {naive_miou:.2f}%   Δ={naive_delta:+.2f}%")

    # Policy SUGGESTION is driven by the naive-q0 effect (the only proxy that
    # directly measures whether the gate helps), NOT by gate load alone.
    if naive_delta > 1.0:
        policy = "semantic-bg dominant (suggestion)"
        policy_desc = ("Turning the gate off IMPROVES mIoU → the global gate is "
                       "over-suppressing reliable foreground. Consider weakening "
                       "or disabling it, while still applying class-selective recovery.")
    elif naive_delta < -1.0:
        policy = "uncertainty-bg dominant (suggestion)"
        policy_desc = ("Turning the gate off COLLAPSES mIoU → the global gate is a "
                       "necessary uncertainty fallback. KEEP the gate; only relax a "
                       "few eligible foreground classes.")
    else:
        policy = "hybrid (suggestion)"
        policy_desc = ("Turning the gate off is roughly neutral overall. Keep the "
                       "gate and apply class-selective relaxation for eligible classes.")

    print(f"\n  → Policy suggestion: [{policy}]")
    print(f"    {policy_desc}")
    print(f"    (high gate entropy {gate_entropy_norm:.2f} would further argue for "
          f"keeping the gate.)\n")

    # =========================================================================
    # STAGE 1 — Class Eligibility Selector
    # =========================================================================
    print(f"{'='*72}")
    print("STAGE 1 — Class Eligibility Selector")
    print(f"{'='*72}")

    winner_scores = {}
    suppressed = {}
    eligibility = {}
    release_band = {}
    deep_noise = {}
    anchor_coherence = {}   # anc_tch × (1 - island_ratio), spatial hard-filter signal

    print(f"  elig_mode    : {args.elig_mode}  (Stage 0 policy: {policy})")
    if args.elig_mode == "release_band" and "uncertainty-bg dominant" in policy:
        print(f"  [GMM risk penalty ACTIVE: elig = base × (1 - gmm_risk) where "
              f"gmm_risk = clip((gmm_tau - λτ) / (τ - λτ), 0, 1)]")
    print(f"  score_safety = 1 - P(score < τ/2 | suppressed)  [v1 proxy]")
    print(f"  rel_band = P(λτ ≤ score < τ | class)   high_anchor = P(score ≥ τ | class)  "
          f"[v2 release-band; λτ={lambda_val:.4f}, τ={global_thd}]")
    print(f"  deep_noise = P(score < λτ | class)  (λ-rule never releases this mass)")
    print(f"  gmm_tau = GMM crossing on suppressed scores (risk signal in uncertainty-bg mode)")
    print(f"  band_* columns computed on RELEASE BAND only: raw_top1=c ∧ λτ ≤ score < τ")
    print(f"  band_mg  = mean(top1-top2) in band    band_mg25 = q25 of same")
    print(f"  band_bg  = mean(top1-bg)   in band    band_bg25 = q25 of same")
    print(f"  t2_bg    = P(top2_class = background | band)  (runner-up is bg → release risky)")
    print(f"  t2_entr  = entropy of top2_class distribution in band")
    print(f"             (high entropy → class confused with many others → risky)")
    print(f"  n_band guard: classes with <100 band pixels show 'nan' for band_* / t2_*")
    print(f"  spatial columns (radius={args.spatial_radius}px, band pixels only):")
    print(f"    anc_tch  = P(band pixel within {args.spatial_radius}px of own-class anchor)")
    print(f"    isl_rt   = P(band pixel in CC not touching own-class anchor)")
    print(f"    cmp_tch  = P(band pixel within {args.spatial_radius}px of ANY other-fg anchor)")
    print(f"    t2c_tch  = P(band pixel within {args.spatial_radius}px of runner-up class anchor)")
    print(f"    anc_coh  = anc_tch x (1-isl_rt)  [spatial hard-filter signal, semantic-bg mode]")
    print(f"    sp_sc    = anc_tch x (1-isl_rt) x (1-cmp_tch)  [diagnostic only; cmp_tch saturates on OEM]")
    print(f"  {'class':<14}{'n_win':>12}{'n_band':>11}{'rel_bnd':>8}{'hi_anch':>8}"
          f"{'gmm_tau':>9}{'band_mg':>9}{'band_mg25':>10}{'band_bg':>9}{'band_bg25':>10}"
          f"{'t2_bg':>8}{'t2_entr':>8}{'elig':>8}"
          f"{'anc_tch':>8}{'isl_rt':>7}{'cmp_tch':>8}{'t2c_tch':>8}{'anc_coh':>8}{'sp_sc':>7}")

    for c in range(C):
        if c == bg_idx:
            continue
        mask_all = raw_top1 == c
        mask_supp = mask_all & (max_sc < global_thd)
        n_all = mask_all.sum()
        n_supp = mask_supp.sum()

        if n_all < args.min_pixels:
            winner_scores[c] = np.array([])
            suppressed[c] = 0.0
            eligibility[c] = 0.0
            release_band[c] = 0.0
            deep_noise[c] = 0.0
            print(f"  {class_names[c]:<14}{n_all:12,}{'(skipped)':>11}")
            continue

        sc_c = max_sc[mask_all]
        winner_scores[c] = sc_c
        supp_mass = n_supp / n_all

        # v2 release-band: mass in [λτ, τ) and deep-noise below λτ
        rel_band_mass = ((sc_c >= lambda_val) & (sc_c < global_thd)).mean()
        deep_noise_frac = (sc_c < lambda_val).mean()
        release_band[c] = rel_band_mass
        deep_noise[c] = deep_noise_frac

        # score_safety (v1 proxy)
        if n_supp > 0:
            sc_supp = sc_c[sc_c < global_thd]
            deeply_suppressed_frac = (sc_supp < global_thd / 2).mean()
            score_safety = 1.0 - deeply_suppressed_frac
        else:
            sc_supp = np.array([])
            score_safety = 1.0

        # GMM tau on suppressed scores — used as uncertainty-bg risk signal in Stage 1
        # and as Stage 2 estimator when class is eligible.
        if len(sc_supp) >= 20:
            tau_gmm_s1, _, _, _ = gmm_threshold(sc_supp, max_samples=args.gmm_max_samples)
            tau_gmm_s1 = float(tau_gmm_s1) if not np.isnan(tau_gmm_s1) else np.nan
        else:
            tau_gmm_s1 = np.nan

        # ---- Release-band class-competition signals (Method A+B) ----
        # Computed ONLY on the release band [λτ, τ) — the pixels λ-rule actually
        # releases — NOT all suppressed pixels (deep noise <λτ would contaminate).
        #   band_margin = top1_score - top2_score   (class confidence: is the winner
        #                 dominant, or is it confused with the runner-up?)
        #   band_bg_gap = top1_score - bg_score      (release safety: how far the
        #                 winner sits above the background class it competes against)
        #   t2_bg_rate  = P(runner-up = background)  (if bg is the frequent runner-up,
        #                 the gate is genuinely competing → releasing is risky)
        #   t2_entropy  = entropy of the runner-up class distribution
        # q25 (not just mean) guards against large-area classes whose mean is pulled
        # by a few extreme regions.
        mask_band = mask_all & (max_sc >= lambda_val) & (max_sc < global_thd)
        n_band = int(mask_band.sum())
        if n_band >= 100:
            band_margin = max_sc[mask_band] - top2_sc[mask_band]
            band_bg_gap = max_sc[mask_band] - bg_sc_all[mask_band]
            band_mg_mean = float(band_margin.mean())
            band_mg_q25 = float(np.percentile(band_margin, 25))
            band_bg_mean = float(band_bg_gap.mean())
            band_bg_q25 = float(np.percentile(band_bg_gap, 25))
            t2_band = top2_cls[mask_band]
            t2_bg_rate = float((t2_band == bg_idx).mean())
            t2_counts = np.bincount(t2_band, minlength=C).astype(np.float64)
            t2_prob = t2_counts / t2_counts.sum()
            t2_entropy = float(-sum(p * np.log(p + 1e-12) for p in t2_prob if p > 0)
                               / np.log(C))
        else:
            band_mg_mean = band_mg_q25 = np.nan
            band_bg_mean = band_bg_q25 = np.nan
            t2_bg_rate = t2_entropy = np.nan

        high_anchor = (sc_c >= global_thd).mean()

        if args.elig_mode == "release_band":
            base_elig = rel_band_mass * high_anchor
            # Mode-aware GMM risk soft penalty: in uncertainty-bg dominant mode,
            # a class whose GMM crossing is close to (or above) λτ has its suppressed
            # distribution dominated by the upper mode (scores near τ), meaning those
            # pixels are likely true uncertainty-bg rather than recoverable foreground.
            # We penalise proportionally: gmm_risk=0 when gmm_tau=λτ (all mass below
            # λτ is recoverable), gmm_risk=1 when gmm_tau=τ (gate is right — don't release).
            # Only applied in uncertainty-bg dominant mode to avoid penalising OEM classes
            # (pavement/grass) which legitimately have gmm_tau >> λτ but are fg.
            if "uncertainty-bg dominant" in policy and not np.isnan(tau_gmm_s1):
                gmm_risk = np.clip(
                    (tau_gmm_s1 - lambda_val) / (global_thd - lambda_val + 1e-6),
                    0.0, 1.0)
                elig = base_elig * (1.0 - gmm_risk)
            else:
                gmm_risk = np.nan
                elig = base_elig
        else:
            # v1 (default): suppressed_mass × score_safety
            gmm_risk = np.nan
            elig = supp_mass * score_safety

        suppressed[c] = supp_mass
        eligibility[c] = elig

        def _f(v, w, p=4):
            return f"{v:{w}.{p}f}" if not (isinstance(v, float) and np.isnan(v)) \
                else f"{'nan':>{w}}"

        # spatial support rates for this class
        if sp_n_band[c] >= 100:
            _anc  = sp_anc_touch[c]    / sp_n_band[c]
            _isl  = sp_island_px[c]    / sp_n_band[c]
            _comp = sp_comp_touch[c]   / sp_n_band[c]
            _t2c  = sp_t2comp_touch[c] / sp_n_band[c]
            _sp   = _anc * (1.0 - _isl) * (1.0 - _comp)
            anchor_coherence[c] = _anc * (1.0 - _isl)   # hard-filter signal
        else:
            _anc = _isl = _comp = _t2c = _sp = float("nan")
            anchor_coherence[c] = float("nan")

        print(f"  {class_names[c]:<14}{n_all:12,}{n_band:11,}"
              f"{rel_band_mass:8.4f}{high_anchor:8.4f}"
              f"{_f(tau_gmm_s1,9)}{_f(band_mg_mean,9)}{_f(band_mg_q25,10)}"
              f"{_f(band_bg_mean,9)}{_f(band_bg_q25,10)}"
              f"{_f(t2_bg_rate,8)}{_f(t2_entropy,8)}{elig:8.4f}"
              f"{_f(_anc,8)}{_f(_isl,7)}{_f(_comp,8)}{_f(_t2c,8)}"
              f"{_f(anchor_coherence[c],8)}{_f(_sp,7)}")

    # ---- eligibility selection ----
    valid_classes = [c for c in range(C)
                     if c != bg_idx and len(winner_scores.get(c, [])) >= args.min_pixels]

    if args.force_eligible:
        # bypass the selector entirely — validate Stage 2 on a known-correct selection
        forced_names = [n.strip() for n in args.force_eligible.split(",") if n.strip()]
        eligible_classes = []
        for name in forced_names:
            found = [i for i, n in enumerate(class_names) if n == name]
            if not found:
                print(f"  [WARN] --force-eligible class '{name}' not in class list, ignored")
            elif found[0] == bg_idx:
                print(f"  [WARN] --force-eligible class '{name}' is background, ignored")
            else:
                eligible_classes.append(found[0])
        eligible_classes = sorted(set(eligible_classes))
        elig_cutoff = float("nan")
        print(f"\n  Selection rule: FORCED via --force-eligible (selector bypassed)")
        print(f"  Eligible classes: {[class_names[c] for c in eligible_classes]}\n")
    else:
        if args.candidate_topk is not None:
            # v2 candidate-topk + ambiguous-gap selection:
            # 1. Take top-K classes by elig score as the initial candidate set.
            # 2. Expand: also include any class within ambiguous_gap of the top-1.
            #    (gap is absolute, not relative — avoids shrinking near zero)
            # 3. Apply min_suppressed_mass floor.
            # 4. Apply max_eligible cap.
            sorted_by_elig = sorted(valid_classes, key=lambda c: eligibility[c], reverse=True)
            topk_candidates = sorted_by_elig[:args.candidate_topk]
            if topk_candidates:
                top1_elig = eligibility[topk_candidates[0]]
                gap_threshold = top1_elig - args.ambiguous_gap
                # expand: any class (from ALL valid) within gap of top-1
                gap_candidates = [c for c in valid_classes
                                  if eligibility[c] >= gap_threshold
                                  and suppressed[c] >= args.min_suppressed_mass]
                # union of topk and gap candidates (both filtered by min_suppressed_mass)
                candidate_set = set(
                    c for c in topk_candidates if suppressed[c] >= args.min_suppressed_mass
                ) | set(gap_candidates)
                eligible_classes = sorted(candidate_set, key=lambda c: eligibility[c], reverse=True)
            else:
                eligible_classes = []
            if args.max_eligible is not None and len(eligible_classes) > args.max_eligible:
                eligible_classes = eligible_classes[:args.max_eligible]
            eligible_classes = sorted(eligible_classes)
            print(f"\n  Selection rule: candidate-topk={args.candidate_topk}  "
                  f"ambiguous-gap={args.ambiguous_gap}  "
                  f"supp_mass≥{args.min_suppressed_mass}"
                  + (f"  AND  top-{args.max_eligible}" if args.max_eligible else ""))
        else:
            # v1: percentile AND absolute floor, then optional top-K
            elig_vals = [eligibility[c] for c in valid_classes]
            elig_cutoff = np.percentile(elig_vals, args.elig_pct) if elig_vals else 0.0
            eligible_classes = [
                c for c in valid_classes
                if eligibility[c] >= elig_cutoff
                and suppressed[c] >= args.min_suppressed_mass
            ]
            if args.max_eligible is not None and len(eligible_classes) > args.max_eligible:
                eligible_classes = sorted(
                    eligible_classes, key=lambda c: eligibility[c], reverse=True
                )[:args.max_eligible]
            eligible_classes = sorted(eligible_classes)
            print(f"\n  Selection rule: eligibility ≥ p{args.elig_pct:.0f}={elig_cutoff:.4f}  "
                  f"AND  suppressed_mass ≥ {args.min_suppressed_mass}"
                  + (f"  AND  top-{args.max_eligible} by score" if args.max_eligible else ""))
        print(f"  Eligible classes: {[class_names[c] for c in eligible_classes]}\n")

    # ---- spatial hard filter (semantic-bg dominant mode only) ----
    # In semantic-bg mode, release-band pixels of recoverable classes should be
    # spatially contiguous with their own high-confidence anchors (spatial extension
    # of confident core).  Classes whose band pixels are isolated islands (high
    # island_ratio) or weakly connected to own anchors (low anc_tch) are likely
    # spurious expansions rather than recoverable foreground.
    # Filter: anchor_coherence = anc_tch × (1 - island_ratio) >= spatial_thr.
    # NOT applied in uncertainty-bg or hybrid modes (LoveDA's water recovery is
    # semantically stable but spatially isolated — spatial filter would wrongly remove it).
    if args.spatial_thr is not None and "semantic-bg dominant" in policy:
        before_sp = [class_names[c] for c in eligible_classes]
        eligible_classes = [
            c for c in eligible_classes
            if not np.isnan(anchor_coherence.get(c, float("nan")))
            and anchor_coherence[c] >= args.spatial_thr
        ]
        after_sp = [class_names[c] for c in eligible_classes]
        removed = [n for n in before_sp if n not in after_sp]
        print(f"  [Spatial hard filter]  anchor_coherence ≥ {args.spatial_thr}  "
              f"(semantic-bg mode, radius={args.spatial_radius}px)")
        if removed:
            print(f"    Removed by spatial filter: {removed}")
        print(f"  Eligible after spatial filter: {after_sp}\n")

    # =========================================================================
    # STAGE 2 — Threshold Estimators (eligible classes only)
    # =========================================================================
    print(f"{'='*72}")
    print("STAGE 2 — Threshold Estimators (eligible classes only)")
    print(f"{'='*72}")

    lam_thd = base_thd.copy()
    otsu_thd = base_thd.copy()
    gmm_thd = base_thd.copy()

    # lambda_val already computed before the inference loop

    print(f"  Otsu/GMM estimated on SUPPRESSED scores only (score<global_thd) and "
          f"clamped ≤ {global_thd}.")
    print(f"  Rationale: recovery must LOWER the gate; estimating on the full winner "
          f"distribution\n  would yield τ>global_thd (raising the gate — the quantile "
          f"failure mode).")
    print(f"  {'class':<14}{'hand-tuned':>12}{'λ-rule':>10}{'Otsu':>10}"
          f"{'GMM':>10}{'n_supp':>10}{'gmm_mu_lo':>11}{'gmm_mu_hi':>11}{'conv':>6}")

    for c in eligible_classes:
        sc_c = winner_scores[c]
        sc_supp = sc_c[sc_c < global_thd]
        n_supp = len(sc_supp)

        # λ-rule
        lam_thd[c] = lambda_val

        # Otsu on suppressed region, clamped ≤ global_thd
        if n_supp >= 20:
            tau_otsu = otsu_threshold(sc_supp)
            otsu_thd[c] = min(tau_otsu, global_thd) if not np.isnan(tau_otsu) else global_thd
            tau_gmm, mu_lo, mu_hi, conv = gmm_threshold(
                sc_supp, max_samples=args.gmm_max_samples)
            gmm_thd[c] = min(tau_gmm, global_thd) if not np.isnan(tau_gmm) else global_thd
        else:
            otsu_thd[c] = lambda_val
            gmm_thd[c] = lambda_val
            mu_lo, mu_hi, conv = np.nan, np.nan, False

        ht = hand_tuned_thd[c]
        print(f"  {class_names[c]:<14}{ht:12.4f}{lam_thd[c]:10.4f}{otsu_thd[c]:10.4f}"
              f"{gmm_thd[c]:10.4f}{n_supp:10,}{mu_lo:11.4f}{mu_hi:11.4f}"
              f"{'Y' if conv else 'N':>6}")

    print()
    non_eligible = [c for c in range(C) if c != bg_idx and c not in eligible_classes
                    and len(winner_scores.get(c, [])) >= args.min_pixels]
    if non_eligible:
        print(f"  Non-eligible (kept at global_thd={global_thd}):")
        for c in non_eligible:
            ht = hand_tuned_thd[c]
            ht_str = f"{ht:.4f}" if ht != global_thd else f"{ht:.4f} (=global)"
            print(f"    {class_names[c]:<14}: hand-tuned={ht_str}")
        print()

    # =========================================================================
    # EVALUATION — mIoU for all methods
    # =========================================================================
    print(f"{'='*72}")
    print("EVALUATION — mIoU comparison")
    print(f"{'='*72}")
    print(f"  Baseline mIoU  : {base_miou:.2f}%  fg-mIoU={base_fg_miou:.2f}%\n")

    # naive quantile q=0.0 (gate fully off) — already computed in Stage 0
    print("  Naive quantile (q=0.00, all-class gate off):")
    report_miou("naive-q0", naive_pred, gt_all, C, base_iou, base_miou,
                class_names, bg_idx)
    print()

    for tag, thd in [("Eligibility + λ-rule", lam_thd),
                     ("Eligibility + Otsu  ", otsu_thd),
                     ("Eligibility + GMM   ", gmm_thd),
                     ("Hand-tuned reference", hand_tuned_thd)]:
        pred = predict_with(raw_top1, max_sc, thd, bg_idx)
        print(f"  {tag}:")
        report_miou(tag.strip(), pred, gt_all, C, base_iou, base_miou,
                    class_names, bg_idx)
        print()

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print(f"{'='*72}")
    print("SUMMARY — Per-class IoU across all methods")
    print(f"{'='*72}")

    methods = {
        "baseline":   predict_with(raw_top1, max_sc, base_thd, bg_idx),
        "naive-q0":   predict_with(raw_top1, max_sc, naive_thd, bg_idx),
        "λ-rule":     predict_with(raw_top1, max_sc, lam_thd, bg_idx),
        "Otsu":       predict_with(raw_top1, max_sc, otsu_thd, bg_idx),
        "GMM":        predict_with(raw_top1, max_sc, gmm_thd, bg_idx),
        "hand-tuned": predict_with(raw_top1, max_sc, hand_tuned_thd, bg_idx),
    }

    ious = {k: iou_from_cm(cm_from_pred(gt_all, v, C)) for k, v in methods.items()}
    mious = {k: np.nanmean(v) * 100 for k, v in ious.items()}

    hdr = f"  {'class':<14}" + "".join(f"{k:>12}" for k in methods)
    print(hdr)
    print("  " + "-" * (14 + 12 * len(methods)))

    for c in range(C):
        row = f"  {class_names[c]:<14}"
        for k, iou_arr in ious.items():
            v = iou_arr[c] * 100
            row += f"{v:12.2f}" if not np.isnan(v) else f"{'nan':>12}"
        print(row)

    print("  " + "-" * (14 + 12 * len(methods)))
    row = f"  {'mIoU':<14}"
    for k in methods:
        row += f"{mious[k]:12.2f}"
    print(row)

    fg_row = f"  {'fg-mIoU':<14}"
    for k, iou_arr in ious.items():
        fg_vals = [iou_arr[c] for c in range(C) if c != bg_idx]
        fg_row += f"{np.nanmean(fg_vals)*100:12.2f}"
    print(fg_row)

    print(f"\n  Eligible classes: {[class_names[c] for c in eligible_classes]}")
    print(f"  λ={args.lam}, τ_global={global_thd}, λ·τ_global={lambda_val:.4f}")
    print(f"\n  Consistency check (does λ ≈ Otsu ≈ hand-tuned?):")
    for c in eligible_classes:
        ht = hand_tuned_thd[c]
        la = lam_thd[c]
        ot = otsu_thd[c]
        gm = gmm_thd[c]
        ht_str = f"{ht:.4f}" if ht != global_thd else "  (=global)"
        print(f"    {class_names[c]:<14}  λ={la:.4f}  Otsu={ot:.4f}  "
              f"GMM={gm:.4f}  hand-tuned={ht_str}")

if __name__ == "__main__":
    main()