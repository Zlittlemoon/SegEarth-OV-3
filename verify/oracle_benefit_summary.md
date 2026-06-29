# Phase-0 Oracle Benefit Analysis — Cross-Dataset Reference

Gold-standard per-class recovery labels from single-class λ-release experiments
(lower **only** `τ_c` to `λτ`, keep every other class at `τ_global`), measured on
the **full** validation set. Produced by the Oracle Benefit Analysis section of
`verify/cafr_diagnose.py`. GT labels are used here for diagnosis only — they do
**not** enter the method.

- LoveDA: full val = 1669 images, `τ_global=0.5`, `λτ=0.10`, baseline mIoU **47.39%**
- OEM:    full val =  384 images, `τ_global=0.1`, `λτ=0.02`, baseline mIoU **44.20%**

Run commands:

```bash
python verify/cafr_diagnose.py configs/cfg_loveda.py \
    --elig-mode release_band --candidate-topk 3 --ambiguous-gap 0.05
python verify/cafr_diagnose.py configs/cfg_openearthmap.py \
    --min-suppressed-mass 0.03 --spatial-thr 0.5
```

Label rule (three-way, keeps a grey zone):
- **beneficial** — `ΔmIoU>0` ∧ `Δcls>0` ∧ `worst_fg_drop ≥ -1%`
- **harmful** — `ΔmIoU<0` ∨ `Δcls<0` ∨ `worst_fg_drop < -2%`
- **ambiguous** — otherwise

---

## LoveDA (full val, 1669 imgs)

| class | n_rel | rel% | GT=c% | GT=bg% | GT=oth% | ΔmIoU | Δcls | Δbg | worst_fg | label |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| water | 32.7M | 19.6 | **63.2** | 29.9 | 7.0 | **+0.93** | **+6.44** | +0.07 | 0.00 | **beneficial** |
| building | 36.7M | 19.6 | 29.7 | 40.0 | 30.3 | −0.52 | −3.13 | −0.51 | 0.00 | harmful |
| road | 30.1M | 22.8 | 25.0 | 64.1 | 10.9 | −0.74 | −3.64 | −1.55 | 0.00 | harmful |
| forest | 55.7M | 35.0 | 22.9 | 60.7 | 16.4 | −0.49 | −0.86 | −2.60 | 0.00 | harmful |
| agricultural | 240.8M | 33.6 | 21.6 | 61.7 | 16.6 | −2.45 | −4.54 | −12.60 | 0.00 | harmful |
| barren | 78.2M | 30.2 | 11.3 | 55.7 | 33.1 | −1.70 | −8.81 | −3.07 | 0.00 | harmful |

- **beneficial**: water
- **harmful**: building, road, forest, agricultural, barren
- Harmful mechanism: **GT=bg dominant** (releasing real uncertainty-background). Consistent with Stage-0 *uncertainty-bg dominant* (naive-q0 Δ = −11.26%).

## OpenEarthMap (full val, 384 imgs)

| class | n_rel | rel% | GT=c% | GT=bg% | GT=oth% | ΔmIoU | Δcls | Δbg | worst_fg | label |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| pavement | 3.63M | 9.5 | **71.5** | 0.7 | 27.8 | **+0.78** | **+2.70** | +4.35 | 0.00 | **beneficial** |
| grass | 1.75M | 2.4 | **44.0** | 3.6 | 52.4 | **+0.20** | **+0.33** | +1.45 | 0.00 | **beneficial** |
| tree | 0.95M | 1.3 | 26.4 | 2.9 | 70.7 | +0.06 | −0.22 | +0.80 | 0.00 | harmful |
| water | 0.08M | 1.0 | 26.6 | 4.9 | 68.5 | −0.01 | −0.18 | +0.05 | 0.00 | harmful |
| building | 2.02M | 2.4 | 13.0 | 1.4 | 85.6 | +0.09 | −1.23 | +2.05 | 0.00 | harmful |
| cropland | 0.36M | 0.8 | 6.1 | 21.1 | 72.8 | −0.04 | −0.21 | −0.16 | 0.00 | harmful |
| bareland | 2.25M | 11.5 | 5.8 | 0.1 | 94.1 | +0.20 | −0.78 | +2.56 | 0.00 | harmful |
| road | 2.21M | 6.9 | 4.6 | 0.8 | 94.6 | +0.03 | −2.15 | +2.39 | 0.00 | harmful |

- **beneficial**: pavement, grass
- **harmful**: tree, water, building, cropland, bareland, road
- Harmful mechanism: **GT=other dominant** (releasing steals pixels laterally from other foreground classes). Consistent with Stage-0 *semantic-bg dominant* (naive-q0 Δ = +1.07%).

---

## Key cross-dataset findings

### 1. `GT=c%` is a clean, dataset-agnostic separator
Beneficial classes all have `GT=c% ≥ 44`; harmful classes all have `GT=c% ≤ 30`.
Across 13 foreground classes on both datasets the split is at ~35% with **zero
crossover**. `GT=c%` = precision of the released band pixels.

| | beneficial GT=c% | harmful GT=c% (max) |
|---|--:|--:|
| LoveDA | water 63.2 | building 29.7 |
| OEM | pavement 71.5 / grass 44.0 | tree 26.6 / water 26.6 |

### 2. Two harmful mechanisms, one signal
- **LoveDA**: harmful = `GT=bg` dominant → releasing real background.
- **OEM**: harmful = `GT=other` dominant → lateral foreground confusion.

Different failure directions, but `GT=c%` captures both — no need for two separate
per-dataset criteria.

### 3. The current Stage-1 selector is misaligned with ground truth

| dataset | selected (auto) | TP (sel∧benef) | FP (sel∧harm) | FN (miss∧benef) |
|---|---|---|---|---|
| LoveDA | building, road, water, forest, agricultural | water | **building, road, forest, agricultural** | — |
| OEM | pavement, building | pavement | **building** | **grass** |

- LoveDA `elig` is **anti-correlated**: forest gets the highest elig (0.1450) yet is
  harmful; water (the only beneficial) is demoted to 0.1305 by the GMM-risk penalty.
- OEM spatial hard filter keeps **building at the highest `anc_coh`=0.7426** (pavement
  is only 0.6163) — spatial coherence does **not** track benefit.
- OEM `grass` is dropped by the `supp_mass ≥ 0.03` floor (`1−hi_anch = 1−0.9714 =
  0.0286 < 0.03`), the known 0.0044-margin fragility.

### 4. The achievable recovery headroom is small (~1 mIoU per dataset)

| dataset | oracle single-class gain | best automatic CAFR | baseline | net vs baseline |
|---|--:|--:|--:|--:|
| LoveDA | water +0.93 (rest negative) | Otsu 46.59 | 47.39 | **−0.80 (net loss)** |
| OEM | pavement +0.78, grass +0.20 | Otsu 45.04 | 44.20 | +0.84 |

Even a perfect selector caps at ~+0.9 (LoveDA) / ~+1 (OEM). On LoveDA the current
automatic pipeline is a net loss because its 4 false positives outweigh water's gain.

---

## Implication for Phase 1

Phase-0 turns a vague question into a supervised one:

> Can the label-free multi-head signals (semantic-only, instance-only, presence,
> fused-before-presence, class−bg gap) predict the oracle `GT=c%` (equivalently the
> beneficial/harmful label)?

- Supervised binary target: `{water}` vs rest (LoveDA); `{grass, pavement}` vs rest (OEM).
- Continuous regression target: `GT=c%` (13 class points across both datasets).

If a label-free signal correlates with `GT=c%` → a multi-head distribution selector is
buildable. If none correlate → SAM3's internal scores lack the information and an
external semantic prior (RemoteCLIP / region-VLM / light calibration) is required.

Because the headroom is small (~1 mIoU), Phase 1 must also answer **whether the gain
justifies the added complexity** — not only whether separability exists.
