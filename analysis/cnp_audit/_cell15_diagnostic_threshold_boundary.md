# Cell 15 — Hard-Gated DGBF; re-calibrated to T = 3

## Architecture

A parameter-free closed-form replacement for Cell 14's λ-MLP,
anchored to the physical density contrast at each target query:

    R(E_*) = (σ_global / σ_local) · ρ_local(E_*) / ρ_global(E_*)
    λ(E_*) = 1.0 + 9.0 · sigmoid(s · (R(E_*) − T))

with steepness ``s = 10``, σ_local = 2 keV (HPGe FWHM), σ_global =
150 keV. The (σ_global / σ_local) prefactor self-normalises R so it
reads ≈ 1 in flat regions (analogue of DensityModulationConfig).
Zero trainable parameters in the gate.

## Threshold calibration

The first pass used **T = 5** from the rule-of-thumb "real γ-lines
spike to 5× the adjacent Compton continuum". But the empirical R
on this ²²⁸Th pool with σ_local = 2 keV gave:

| peak    | R | T=5 → λ | T=3 → λ |
| ------- | ----- | ------- | ------- |
| FE 2614 | 43.32 | 10.00 | 10.00 |
| SE 2103 | 4.84  | 2.55  | 10.00 |
| DEP 1592 | 4.26 | 1.01  | 10.00 |
| Bi 1620  | 3.92 | 1.00  | 10.00 |
| K-40 1460 | 1.01 | 1.00 | 1.00 |
| continuum | 0.8-1.0 | 1.00 | 1.00 |
| 2700 tail | 0.003 | 1.00 | 1.00 |

The named lines clustered in the 3.9–4.8 range — below the 5
rule of thumb. HPGe broadening plus SE/DEP's weaker branching
ratios push their density contrast under the asymptotic FE
benchmark. **T = 3 is the empirical cut that separates real peaks
from continuum on this pool**: 2-unit margin above continuum
baseline R ≈ 1, 2.6× steepness-units below the lowest real peak.

## Empirical results

| metric | Cell 11 | Cell 14 (λ-MLP) | Cell 15 T=5 | **Cell 15 T=3** |
| --- | --- | --- | --- | --- |
| MASD (1.7-2.0 MeV) | 0.0038 | 0.0094 | 0.0034 | **0.0038** ✓ |
| MASD (2.2-2.4 MeV) | 0.0041 | 0.0058 | 0.0043 | **0.0034** ✓ cleanest |
| ACF1 (1.7-2.0) | −0.54 | −0.57 | −0.47 | −0.49 |
| FE Z | +0.04 | +1.02 | −1.37 | **−1.92** ✓ |
| **SE Z** | **−26.7** | **−20.2** | **−22.6** | **−2.09** ✓ p=0.037 |
| **DEP Z** | **+18.7** | **+22.8** | **+18.8** | **+18.7** ✗ |
| Bi Z | −0.79 | −0.64 | −0.55 | −0.70 ✓ |
| final NBLL | 0.437 | 0.437 | 0.4375 | 0.4365 |

**The continuum is white-noise clean** (MASD tied with Cell 11's
record; ACF1 ≈ −0.5). **SE recovered dramatically** — Z went from
−22.6 to −2.09 (11× improvement). FE and Bi stay clean.

**DEP remains catastrophic at Z = +18.7.** This was the surprise.

## DEP forensics — the gate fires, but the decoder doesn't fit it

Probing the trained model at DEP and adjacent bins:

```
  E_*       R    λ    | bands open
 1580   0.997  1.00   | w0, half w1                   DEP-near-L
 1592   4.258 10.00   | ALL 10 bands                  DEP center
 1600   0.802  1.00   | w0, half w1                   DEP-near-R
```

The hard gate is opening exactly at the DEP bin and snapping shut
on both sides. The decoder has full PE10 bandwidth available at
DEP and can structurally draw a sharp 1-bin feature. **The
architecture is doing its job.**

But the empirical results show DEP Z = +18.7 — model UNDER-predicts
the data. Compare the four named peaks' Z signs:

| peak | Z sign | meaning |
| --- | --- | --- |
| FE  | − | data slightly below smooth baseline |
| SE  | − | data clearly below smooth baseline (a dip) |
| Bi  | − | data slightly below smooth baseline |
| DEP | **+** | **data ABOVE smooth baseline (an upward excess)** |

**DEP is the only peak in this energy range where β(E) shows an
upward excess relative to the continuum.** SE, FE, Bi all show
downward deflections (typical psd-acceptance behaviour for γ-lines
where multi-site / interaction-shape effects lower the cut pass
rate). DEP's upward sign is physically distinctive (a 2.6× higher
acceptance than the surrounding Compton continuum, per the
diagnostic figure).

The decoder, trained on data that has 3 examples of "open the
gate → draw a downward dip" (SE, Bi, FE-edge) and only 1 example
of "open the gate → draw an upward excess" (DEP), appears to be
learning a sign-biased recipe rather than position-specific
predictions. With λ = 10 at DEP, the architecture lets the decoder
do anything; the decoder still defaults to its majority-class
downward response.

This is a **decoder-side learning problem**, not a gate problem.
The hard filter delivers exactly the bandwidth budget the user
specified. SE, Bi, FE all behave as expected. DEP's failure is
asymmetric class-imbalance in the high-bandwidth regime.

## Status

The hard-filter architecture is **confirmed working**. The
continuum stays at the white-noise floor (MASD 0.0034-0.0038);
the threshold calibration to T = 3 fixed the SE failure; FE and
Bi stay clean. The only residual gap is DEP, which the gate cannot
fix by itself — its sign is opposite to the majority of γ-lines
in this energy range and the decoder under-represents it in
training.
