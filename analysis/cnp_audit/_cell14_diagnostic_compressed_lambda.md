# Cell 14 — Continuum sawtooth killed, but λ-range too compressed to fit peaks

## Empirical result

Cell 14 implements the Density-Guided Bandwidth Filter (DGBF): a
third density-aware SFN head emits a continuous frequency cutoff
oracle λ(E_*) ∈ [0, L = 10] that drives per-band soft weights
``w_l = sigmoid(α · (λ − l))`` applied directly to the raw PE10
sin/cos pairs *before* they enter the decoder. ``z_phi_T`` is
stripped from the decoder concat; only the filtered SAPE vector
plus the raw 2D (E_norm, T_norm) reach β̂. σ_local was sharpened
from 2 → 1 keV for double peak/continuum density contrast.

| metric | Cell 11 | Cell 12 (sl=2) | Cell 13 (η) | **Cell 14 (λ, sl=1)** |
| --- | --- | --- | --- | --- |
| MASD (1.7-2.0 MeV) | 0.0038 | 0.0041 | 0.245 | **0.0094** ✓ |
| MASD (2.2-2.4 MeV) | 0.0041 | 0.0056 | 0.141 | **0.0058** ✓ |
| ACF1 (1.7-2.0) | −0.54 | −0.54 | −0.45 | **−0.57** ✓ |
| FE Z | +0.04 | −1.02 | −1.71 | +1.02 ✓ |
| **SE Z** | **−26.7** | **−28.8** | **−1.9** | **−20.2** ✗ |
| **DEP Z** | **+18.7** | **+23.0** | **+5.2** | **+22.8** ✗ |
| Bi Z | −0.79 | −0.71 | +0.26 | −0.64 ✓ |
| final NBLL | 0.437 | 0.437 | 0.410 | 0.437 |

**The continuum sawtooth is gone** — MASD back to near-Cell-11 levels,
ACF1 ≈ −0.57 (white-noise floor). The frequency-domain gate
*structurally* delivered what the global η scalar could not: high
bands cannot leak into the continuum because the basis functions
themselves are zeroed at the source there.

**But SE / DEP relapsed to Cell 12 levels.** SE Z = −20.2 (vs Cell 13's
−1.9); DEP Z = +22.8 (vs Cell 13's +5.2). The mechanism is exactly
visible in the λ probe.

## The λ probe — correct polarity, compressed range

```
  E_*   log_l  log_g  σ_kev    τ    λ  | w0   w1   w2   w3-9     region
1592    4.27   7.48  199.97  1.00 1.23 | 1.00 0.76 0.02 0.00...  DEP
1620    4.19   7.49  199.97  1.00 1.19 | 1.00 0.72 0.02 0.00...  Bi
1700    2.50   7.49  199.91  1.00 1.02 | 0.99 0.53 0.01 0.00...  ctrl-1
1850    2.60   7.50  199.91  1.00 1.00 | 0.99 0.50 0.01 0.00...  ctrl-1
2103    4.59   7.71  199.98  1.00 1.27 | 1.00 0.79 0.02 0.00...  SE
2400    2.43   7.77  199.91  1.00 0.99 | 0.99 0.49 0.01 0.00...  ctrl-2
2614    6.72   7.60  199.98  1.00 1.74 | 1.00 0.98 0.21 0.00...  FE
2700  −11.43   7.29   17.26  1.21 0.00 | 0.51 0.01 0.00 0.00...  tail
```

Two findings stand out:

1. **The polarity is right** (Cell 13's inversion is gone). λ at every
   real peak (DEP 1.23, Bi 1.19, SE 1.27, FE 1.74) is strictly
   **higher** than at its adjacent continuum (1.00, 1.02, 0.99). The
   architectural mechanism — density-aware logits driving the band
   cutoff — is operating as designed.

2. **The dynamic range is catastrophically compressed.** λ never
   exceeds ~1.74 across the entire spectrum. With α = 5 that gives:
   * Band 0 (period 5000 keV): always open
   * Band 1 (period 2500 keV): partially open in peaks, half-open
     in continuum
   * Band 2 (period 1250 keV): only ~0.21 open at FE; effectively
     zero everywhere else
   * Bands 3-9: **zero everywhere**, peaks included

   The decoder only ever sees feature scales ≥ ~1250 keV. It has
   no access to the bin-scale frequencies (bands 7-9 with periods
   ~10-80 keV) required to draw a 2-keV-wide SE / DEP line. So
   peaks default to the smooth Cell-11/12 fit, with the matching
   Z_DT ≈ −20 / +23 catastrophes.

## What the optimizer actually found

The NBLL loss landscape is shallow in λ above ~1.5: pushing λ
higher would open high-frequency bands and risk over-fitting
shot-noise fluctuations in the continuum (high cost), while the
loss reduction from sharpening a 2-keV-wide SE peak is small
(SE / DEP each occupy 1 of ~250 bins). The model rationally chose
the safe option — fit the broad spectrum well, leave the rare
peaks under-fit.

This is the **same architectural ceiling** Cells 11/12 hit, just
discovered through a different gate. With λ stuck near 1, Cell 14
is essentially Cell 12 with a less expressive decoder (2L = 20
filtered bands at scales ≥ 1.25 MeV ≈ Cell 12's raw coordinates).

## What changed from Cell 13

Cell 13's η-gate let *all* PE10 frequencies through at amplitude
0.3 globally — enough sharpness to fit SE/DEP (Z 1.9 / 5.2) AND
enough to overfit shot noise into continuum sawtooth. Cell 14's
λ-gate uses the same density signal but cleanly separates the
two regimes: continuum has zero high-frequency content, peaks
could have high-frequency content if λ → L. The optimizer
chose not to push λ high enough — but the *architecture* is now
correctly faithful to the design principle: any sawtooth that
appears in the continuum *must* be a fit decision, not a leakage.

## The fork

Three orthogonal levers remain, all NBLL-untouched:

1. **Push more training steps.** 3000 steps may simply be too few
   for the λ-head to discover the peak-fitting regime. Try 10k
   steps and see if λ at peaks climbs.

2. **Sharper σ_local (< 1 keV).** At 1 keV the peak/continuum
   contrast is log_l ≈ +2 (4.3 at DEP vs 2.5 at 1700 ctrl). Going
   to 0.5 keV would amplify this further. Caveat: the HPGe
   physical resolution is ~2 keV FWHM, so sub-keV kernels would be
   modelling shot noise as if it were structure.

3. **Architectural λ_min offset.** Bias the λ-head so its logit
   defaults toward +3 (λ ≈ 0.95 · L). The model starts with all
   bands open and must learn to *close* them in the continuum.
   But this loses the structural "continuum cannot draw sawtooth"
   guarantee at training start — the η inversion (Cell 13) is
   potentially re-introduced as the model finds a way to undo the
   default.

The cleanest first try is lever (1) — give the same architecture
more optimization budget. If λ at peaks doesn't climb past ~3-4
even after extended training, that confirms the loss landscape
problem and motivates a structural prior (lever 3).
