"""
Phase-1A analysis — Release-band raw multi-head signal separability.

Reads the .npz files dumped by raw_score_diagnose.py and answers the Phase-1A
question with zero external dependencies (numpy only):

    Inside the CAFR v2 release-band, can the RAW multi-head signals separate
    is_c (true foreground the gate suppressed) from is_bg (correctly gated
    background) and from is_other (lateral foreground confusion)?

Outputs, per dataset:
  1. Per-class AUC table — each signal predicting is_c vs is_bg and is_c vs
     is_other.  Two flavours per pair:
        pooled        : AUC over all sampled pixels of the class
        img-balanced  : mean of per-image AUCs (each image weighted equally;
                        cancels the per-image-cap reweighting in collection)
     The v2 post-fusion signal fused_after_presence is included as the BASELINE
     row — raw signals must beat it for the "fusion lost information" thesis.
     AUC < 0.5 means the signal is inversely related (still informative).
  2. Class-level Spearman — each signal's class-mean vs the class is_c rate
     (image-balanced), across the dataset's collected classes.  Small-n
     (≈5 classes) so treated as a trend, not a test.  elig is NOT available here;
     compare against the Phase-0 oracle GT=c% offline.
  3. Semantic-instance consistency four-cell table — for is_c vs is_bg pixels,
     P(sem hi/lo ∧ inst hi/lo) under two hi/lo definitions:
        absolute : sem ≥ λτ , inst ≥ λτ
        relative : sem ≥ class-q75 , inst ≥ class-q75  (within the release band)
     Tests whether true recoverable foreground (is_c) shows higher
     semantic-instance agreement than gated background (is_bg).

Signals analysed (column name in the .npz):
    semantic     semantic_pixel_before_presence
    inst_raw     instance_raw_pixel_before_confidence_presence
    inst_orig    instance_original_prefinal_pixel
    fused_raw    fused_raw_before_presence
    fused_after  fused_after_presence          (BASELINE: the failed v2 signal)
    bg_post      bg_score_post
    gap_post     class_minus_bg_gap_post
    presence     presence_score                (class/image-level; reported, not a
                                                 standalone pixel predictor)

Usage:
    python verify/raw_score_analyze.py verify/rawdiag_loveda --lambda-tau 0.10
    python verify/raw_score_analyze.py verify/rawdiag_oem    --lambda-tau 0.02

(λτ defaults are read from each .npz; --lambda-tau only overrides if needed.)
"""

import argparse
import glob
import os

import numpy as np


SIGNALS = [
    ("semantic",    "semantic_pixel_before_presence"),
    ("inst_raw",    "instance_raw_pixel_before_confidence_presence"),
    ("inst_orig",   "instance_original_prefinal_pixel"),
    ("fused_raw",   "fused_raw_before_presence"),
    ("fused_after", "fused_after_presence"),    # baseline (failed v2 signal)
    ("bg_post",     "bg_score_post"),
    ("gap_post",    "class_minus_bg_gap_post"),
    ("presence",    "presence_score"),
]
BASELINE_KEY = "fused_after"


# ---------------------------------------------------------------------------
# numpy-only rank utilities
# ---------------------------------------------------------------------------

def _rankdata_avg(x):
    """Average ranks (1..n), ties share the mean rank. Pure numpy."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sx = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0   # 1-based average of positions i..j
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def auc_score(scores, pos_mask):
    """AUC via the Mann-Whitney U rank statistic. pos_mask: bool, True=positive.
    Returns nan if either group is empty."""
    pos_mask = np.asarray(pos_mask, dtype=bool)
    n_pos = int(pos_mask.sum())
    n_neg = int((~pos_mask).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata_avg(scores)
    sum_pos = ranks[pos_mask].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def spearman(x, y):
    """Spearman rank correlation, pure numpy. nan if <3 finite pairs."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx = _rankdata_avg(x[ok])
    ry = _rankdata_avg(y[ok])
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def img_balanced_auc(scores, pos_mask, neg_mask, image_id, min_per_group=10):
    """Mean of per-image AUCs over images that have >= min_per_group of BOTH
    the positive and negative group.  Returns (mean_auc, n_images_used)."""
    scores = np.asarray(scores)
    pos_mask = np.asarray(pos_mask, dtype=bool)
    neg_mask = np.asarray(neg_mask, dtype=bool)
    keep = pos_mask | neg_mask
    if keep.sum() == 0:
        return float("nan"), 0
    sc = scores[keep]
    pm = pos_mask[keep]
    img = image_id[keep]
    aucs = []
    for im in np.unique(img):
        sel = img == im
        p = pm[sel]
        if int(p.sum()) >= min_per_group and int((~p).sum()) >= min_per_group:
            aucs.append(auc_score(sc[sel], p))
    if not aucs:
        return float("nan"), 0
    return float(np.nanmean(aucs)), len(aucs)


# ---------------------------------------------------------------------------
# per-class analysis
# ---------------------------------------------------------------------------

def load_class(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def fmt(v, w=6, p=3):
    if isinstance(v, float) and (v != v):   # nan
        return f"{'nan':>{w}}"
    return f"{v:{w}.{p}f}"


def analyze_class(data, lam_tau):
    """Return a dict of results for one class file."""
    is_c = data["is_c"].astype(bool)
    is_bg = data["is_bg"].astype(bool)
    is_other = data["is_other"].astype(bool)
    image_id = data["image_id"]
    n = len(is_c)

    res = {
        "class_name": str(data["class_name"]),
        "n": n,
        "p_is_c": float(is_c.mean()),
        "p_is_bg": float(is_bg.mean()),
        "p_is_other": float(is_other.mean()),
        "auc": {},          # sig -> dict(c_bg, c_bg_ib, c_oth, c_oth_ib)
        "means": {},        # sig -> {c: mean over is_c, bg: mean over is_bg}
    }

    for sig_name, col in SIGNALS:
        x = data[col].astype(np.float64)
        # c vs bg
        m_cb = is_c | is_bg
        auc_cb = auc_score(x[m_cb], is_c[m_cb]) if m_cb.any() else float("nan")
        auc_cb_ib, _ = img_balanced_auc(x, is_c, is_bg, image_id)
        # c vs other
        m_co = is_c | is_other
        auc_co = auc_score(x[m_co], is_c[m_co]) if m_co.any() else float("nan")
        auc_co_ib, _ = img_balanced_auc(x, is_c, is_other, image_id)
        res["auc"][sig_name] = dict(c_bg=auc_cb, c_bg_ib=auc_cb_ib,
                                    c_oth=auc_co, c_oth_ib=auc_co_ib)
        res["means"][sig_name] = dict(
            c=float(x[is_c].mean()) if is_c.any() else float("nan"),
            bg=float(x[is_bg].mean()) if is_bg.any() else float("nan"),
        )

    # ---- semantic-instance consistency four-cell (is_c vs is_bg) ----
    sem = data["semantic_pixel_before_presence"].astype(np.float64)
    inst = data["instance_raw_pixel_before_confidence_presence"].astype(np.float64)

    def four_cell(mask, sem_hi, inst_hi):
        if mask.sum() == 0:
            return dict(hh=float("nan"), hl=float("nan"),
                        lh=float("nan"), ll=float("nan"))
        sh = sem_hi[mask]
        ih = inst_hi[mask]
        m = mask.sum()
        return dict(
            hh=float((sh & ih).sum()) / m,
            hl=float((sh & ~ih).sum()) / m,
            lh=float((~sh & ih).sum()) / m,
            ll=float((~sh & ~ih).sum()) / m,
        )

    cons = {}
    # absolute thresholds
    sem_hi_abs = sem >= lam_tau
    inst_hi_abs = inst >= lam_tau
    cons["abs_is_c"] = four_cell(is_c, sem_hi_abs, inst_hi_abs)
    cons["abs_is_bg"] = four_cell(is_bg, sem_hi_abs, inst_hi_abs)
    # relative thresholds (class-q75 within the whole release band of this class)
    sem_q75 = np.quantile(sem, 0.75) if n > 0 else 0.0
    inst_q75 = np.quantile(inst, 0.75) if n > 0 else 0.0
    sem_hi_rel = sem >= sem_q75
    inst_hi_rel = inst >= inst_q75
    cons["rel_is_c"] = four_cell(is_c, sem_hi_rel, inst_hi_rel)
    cons["rel_is_bg"] = four_cell(is_bg, sem_hi_rel, inst_hi_rel)
    res["consistency"] = cons
    res["q75"] = dict(sem=float(sem_q75), inst=float(inst_q75))
    return res


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase-1A separability analysis (numpy only)")
    ap.add_argument("npz_dir", help="directory of rawdiag_<ds>_releaseband_<class>.npz")
    ap.add_argument("--lambda-tau", type=float, default=None,
                    help="override λτ for the consistency absolute threshold "
                         "(default: read from each .npz)")
    ap.add_argument("--min-rows", type=int, default=200,
                    help="skip a class with fewer than this many sampled rows")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.npz_dir, "*releaseband_*.npz")))
    if not files:
        raise SystemExit(f"no *releaseband_*.npz files in {args.npz_dir}")

    print(f"{'='*100}")
    print(f"Phase-1A Separability Analysis   dir={args.npz_dir}")
    print(f"  AUC: probability a signal ranks an is_c pixel above the comparison class.")
    print(f"  0.5 = no separation; >0.5 raw signal predicts is_c; <0.5 inverse (still informative).")
    print(f"  BASELINE row = fused_after (the failed v2 post-fusion signal). Raw signals must beat it.")
    print(f"{'='*100}")

    results = []
    for f in files:
        data = load_class(f)
        if len(data["is_c"]) < args.min_rows:
            print(f"\n[skip] {os.path.basename(f)} — only {len(data['is_c'])} rows")
            continue
        lam_tau = args.lambda_tau if args.lambda_tau is not None \
            else float(data.get("lambda_val", 0.0))
        res = analyze_class(data, lam_tau)
        res["_file"] = os.path.basename(f)
        res["_lam_tau"] = lam_tau
        results.append(res)

    # ---- 1. per-class AUC tables ----
    for res in results:
        print(f"\n{'-'*100}")
        print(f"CLASS: {res['class_name']}   n={res['n']:,}   "
              f"is_c={res['p_is_c']*100:.1f}%  is_bg={res['p_is_bg']*100:.1f}%  "
              f"is_other={res['p_is_other']*100:.1f}%   λτ={res['_lam_tau']:.4f}")
        print(f"  (NOTE: is_c/is_bg/is_other here are cap-reweighted marginals, "
              f"not true release-band priors — use Phase-0 oracle GT=c% for priors.)")
        print(f"  {'signal':<12}{'AUC c|bg':>10}{'(img-bal)':>11}"
              f"{'AUC c|oth':>11}{'(img-bal)':>11}   {'mean|c':>8}{'mean|bg':>8}  note")
        for sig_name, _ in SIGNALS:
            a = res["auc"][sig_name]
            mu = res["means"][sig_name]
            note = "BASELINE(v2)" if sig_name == BASELINE_KEY else ""
            if sig_name == "presence":
                note = "class-level"
            print(f"  {sig_name:<12}{fmt(a['c_bg'],10)}{fmt(a['c_bg_ib'],11)}"
                  f"{fmt(a['c_oth'],11)}{fmt(a['c_oth_ib'],11)}   "
                  f"{fmt(mu['c'],8)}{fmt(mu['bg'],8)}  {note}")
        # delta vs baseline (img-balanced c|bg) for the strongest message
        base = res["auc"][BASELINE_KEY]["c_bg_ib"]
        print(f"  → Δ(img-bal AUC c|bg) vs baseline fused_after={fmt(base,5)}:")
        deltas = []
        for sig_name, _ in SIGNALS:
            if sig_name in (BASELINE_KEY, "presence"):
                continue
            v = res["auc"][sig_name]["c_bg_ib"]
            if v == v and base == base:
                deltas.append((sig_name, v - base))
        deltas.sort(key=lambda t: (t[1] if t[1] == t[1] else -9), reverse=True)
        print("     " + "  ".join(f"{s}:{d:+.3f}" for s, d in deltas))

    # ---- 2. class-level Spearman: signal class-mean(on is_c) vs class is_c rate ----
    print(f"\n{'='*100}")
    print("CLASS-LEVEL Spearman  (signal mean over is_c pixels  vs  class is_c rate)")
    print("  small-n trend only (~5 classes). For the real target use Phase-0 oracle GT=c%.")
    isc_rate = np.array([r["p_is_c"] for r in results])
    print(f"  {'signal':<12}{'spearman':>10}")
    for sig_name, _ in SIGNALS:
        means_c = np.array([r["means"][sig_name]["c"] for r in results])
        rho = spearman(means_c, isc_rate)
        print(f"  {sig_name:<12}{fmt(rho,10)}")
    print("  classes:", [r["class_name"] for r in results])
    print("  is_c rate:", [round(r["p_is_c"], 3) for r in results])

    # ---- 3. semantic-instance consistency four-cell ----
    print(f"\n{'='*100}")
    print("SEMANTIC-INSTANCE CONSISTENCY  (four-cell, is_c vs is_bg)")
    print("  hh=P(sem hi ∧ inst hi)  hl=P(sem hi ∧ inst lo)  lh=P(sem lo ∧ inst hi)  ll=P(both lo)")
    print("  Hypothesis: true fg (is_c) shows higher agreement (hh) than gated bg (is_bg).")
    for defn in ("abs", "rel"):
        tag = "absolute (≥λτ)" if defn == "abs" else "relative (≥class-q75)"
        print(f"\n  --- {tag} ---")
        print(f"  {'class':<12}{'grp':>6}{'hh':>8}{'hl':>8}{'lh':>8}{'ll':>8}")
        for res in results:
            for grp in ("is_c", "is_bg"):
                c = res["consistency"][f"{defn}_{grp}"]
                print(f"  {res['class_name']:<12}{grp:>6}"
                      f"{fmt(c['hh'],8)}{fmt(c['hl'],8)}{fmt(c['lh'],8)}{fmt(c['ll'],8)}")

    print(f"\n{'='*100}")
    print("Read-out guide:")
    print("  * A raw signal whose img-bal AUC c|bg is clearly >0.5 AND > baseline fused_after")
    print("    means raw scores carry release-precision info that v2 fusion lost.")
    print("  * Compare c|bg (LoveDA failure mode) and c|oth (OEM failure mode) separately.")
    print("  * If NO raw signal beats baseline on either, internal scores are insufficient")
    print("    -> external prior (RemoteCLIP / VLM / light calibration).")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
