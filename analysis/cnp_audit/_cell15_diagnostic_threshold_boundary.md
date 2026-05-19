# Cell 15 — Hard-gated DGBF, iterated to ultra-contrast + R-injection

## Architecture timeline

| variant | σ_l | σ_g | T | xfeed | path |
| --- | --- | --- | --- | --- | --- |
| Cell 15a (initial) | 2 | 150 | 5 | off | `_hardfilter_sl2_pedetach` |
| Cell 15b (T=3) | 2 | 150 | 3 | off | `_hardfilter_sl2_pedetach` (overwrite) |
| **Cell 15c (ultra-contrast)** | **1** | **50** | **3** | **on** | `_hardfilter_xfeed_sl1_sg50_pedetach` |

All three share: hard-gated parameter-free λ closed form,
`λ = 1 + 9·sigmoid(10·(R − T))`, head-tied σ/τ SFN attention,
PE10 + PE-detached Q/K, decoder-coordinate gating, single-head
1×128 attention. The 15c re-run lives at a new paradigm path
because the kernel pocket and decoder dim changed
architecturally; the 15a/b history is preserved at the old path.

## What changed in the ultra-contrast re-run

**1. Kernel pocket re-tuned**. σ_local pushed below the HPGe FWHM
(2 → 1 keV) to capture single-bin peak intensity from the raw
spectrum without dilution. σ_global pulled in (150 → 50 keV) to
track local Compton background fluctuations without absorbing
peak counts into the baseline. Pre-train probe shows clean
separation:

| peak | R(sl=2) | R(ultra) |
| --- | --- | --- |
| FE 2614 | 43.32 | 27.86 |
| SE 2103 | 4.84 | 5.95 |
| **DEP 1592** | **4.26** | **5.29** |
| Bi 1620 | 3.92 | 4.87 |
| K-40 1460 | 1.01 | 1.19 |
| continuum | 0.8–1.0 | 0.6–1.2 |

All four named lines now exceed T = 3 by ≥1.9 units; continuum
stays ≤1.2. The pocket is **the right physics resolution** for
this dataset.

**2. Explicit R(E_*) injection into the decoder**. R was
previously visible to the *gate* only (deciding λ); the decoder
saw only the bandwidth budget via SAPE. Injecting R as a scalar
per-query coordinate gives the decoder explicit raw-spectrum peak
intensity, so it can map FE (R~28) and SE/DEP/Bi (R~5-6) to
different β values rather than treating them all as "open gate".

    decoder_input = [r_target, raw_phi (E,T), SAPE(E_*), R(E_*)]

Decoder input dim: 64 + 2 + 20 + 1 = 87.

## Empirical results

| metric | 15a (T=5) | 15b (T=3, sl=2) | **15c (ultra)** |
| --- | --- | --- | --- |
| MASD 1.7-2.0 MeV | 0.0034 | 0.0038 | 0.0047 |
| MASD 2.2-2.4 MeV | 0.0043 | 0.0034 | **0.0144** ↑ |
| ACF1 1.7-2.0 | −0.47 | −0.49 | −0.40 |
| FE Z | −1.37 | −1.92 | **+0.65 ✓ p=0.51** |
| SE Z | −22.6 | −2.09 | **−0.78 ✓ p=0.43** |
| Bi Z | −0.55 | −0.70 | −0.63 ✓ p=0.53 |
| **DEP Z** | **+18.8** | **+18.7** | **+22.0 ✗** |
| final NBLL | 0.4375 | 0.4365 | 0.4346 |

### Wins

* **FE, SE, Bi all near-perfectly clean** — best fits of any
  cell. SE Z went −22.6 → −2.09 → **−0.78**. FE flipped sign
  cleanly to +0.65. Bi stayed clean. The R-injection works
  exactly as designed for lines whose β-direction matches the
  majority pattern (negative deflection).

* **1.7-2.0 MeV continuum stays clean** (MASD 0.0047, white-noise
  ACF1).

### Losses

* **2.2-2.4 MeV continuum degraded 4×** (0.0034 → 0.0144). The
  ultra-contrast pocket (σ_g = 50 keV) is sensitive to local
  density structure in the high-energy tail above FE 2614; the
  decoder now reads continuous R ripples in its concat. The 1×128
  attention's σ/τ machinery still attenuates, but the R-injection
  removed one buffer between density noise and β̂.

* **DEP unchanged at Z = +22**, even though R = 5.29 cleanly
  triggers λ = 10 there. The explicit R magnitude alone doesn't
  carry sign information — DEP and SE both sit at R ≈ 5-6 but
  point in opposite directions. The decoder learned a "small R
  → mild dip" recipe from SE/Bi and applies it to DEP, missing
  the upward physics excess.

## What R-injection can and cannot fix

The cleanest distinction this experiment isolates:

* The hard gate decides **bandwidth** (λ): can the decoder draw
  a sharp feature here? With ultra-contrast + T=3, all four named
  peaks get λ=10 with crystal margin.
* The R-injection adds **intensity** as a feature: how strong is
  the raw peak? Helps the decoder modulate the *magnitude* of its
  fit between FE (R=28) and SE/DEP/Bi (R=5-6).
* **Neither carries sign information.** DEP's defining feature
  is its *opposite* polarity vs FE/SE/Bi (positive β-excess
  rather than dip). No density-based feature distinguishes a
  peak's "above-baseline" from "below-baseline" β response.

Hence DEP is unrecoverable purely from density-driven inputs. To
fix DEP without modifying the NBLL loss, the model would need a
signal that depends on the actual β-direction at each peak — for
example a learned per-peak embedding indexed by E_*, or a feature
derived from the classifier-score distribution at E_* (not just
its energy density).

## Status

Cell 15c is the cleanest result for FE / SE / Bi across the entire
audit history. DEP remains structurally outside the reach of
density-only gates and features. The architecture has hit the
ceiling of what physics-prior gating alone can deliver; further
DEP recovery requires either input features that carry sign or a
training-time intervention that exposes DEP's polarity asymmetry
(neither in scope for this iteration).
