# Cell 13 — η-head opens the peak channel but learns inverted polarity

## Empirical result

Cell 13 took Cell 12's architecture verbatim and added a third
density-aware SFN head η(log_l, log_g) ∈ [0, 1] that gates a
`z_phi_T` block in the decoder concat:

    decoder_input = [r_target, raw_E_norm, η(E_*) · z_phi_T]

The design intent was a bimodal switch: η → 0 in the continuum
(decoder sees only raw coords → smooth β̂, Cell 11/12 regime),
η → 1 at γ-peaks (decoder sees full PE10 features → sharp β̂,
Cell 6 regime).

| metric | Cell 11 | Cell 12 (σ_l = 2) | **Cell 13 (+pegate)** | verdict |
| --- | --- | --- | --- | --- |
| MASD (1.7-2.0 MeV) | 0.0038 | 0.0041 | **0.2449** | **60× worse** |
| MASD (2.2-2.4 MeV) | 0.0041 | 0.0056 | **0.1409** | **25× worse** |
| FE Z | +0.04 | −1.02 | −1.71 | clean ✓ |
| **SE Z** | **−26.7** | **−28.8** | **−1.86** | **15× better ✓** |
| **DEP Z** | **+18.7** | **+23.0** | **+5.16** | **4.5× better** |
| Bi Z | −0.79 | −0.71 | +0.26 | clean ✓ |
| final NBLL | 0.437 | 0.437 | 0.410 | better ✓ |

Peaks recovered dramatically — SE went from catastrophic |Z|=29 to
near-clean |Z|=1.9; DEP went from |Z|=23 to |Z|=5.2 (still a miss
but architecturally fittable). But the continuum sawtooth came
back near Cell 7-10 levels. The architectural switch is real, but
it failed in a specific, instructive way.

## The η probe — opposite polarity from the design intent

Sampling the three SFN heads across the spectrum at T = T* with
the model in eval mode:

```
   E_*    log_l  log_g    σ[keV]   τ      η      region
   1592    4.61   7.48    199.98   1.00   0.278   DEP
   1620    4.54   7.49    199.98   1.00   0.276   Bi PEAK
   1700    3.00   7.49    199.95   1.00   0.378   ctrl-1
   1850    3.16   7.50    199.95   1.00   0.351   ctrl-1
   2103    4.97   7.71    199.98   1.00   0.273   SE
   2400    3.33   7.77    199.96   1.00   0.329   ctrl-2
   2614    7.05   7.60    199.99   1.00   0.196   FE PEAK
   2700   −2.98   7.29    170.99   1.01   0.956
   2800   −8.67   6.62     51.52   1.12   0.985   sparse tail
```

Two things stand out:

1. **The polarity is inverted.** η ≈ 0.20-0.28 at peaks (where we
   wanted η ≈ 1), η ≈ 0.33-0.38 in the continuum (where we wanted
   η ≈ 0). The MLP did not learn "open the gate at high local
   density"; it learned **the opposite**.

2. **The dynamic range across the main spectrum is tiny.** η spans
   only 0.20-0.38 across 1500-2614 keV — a difference of 0.18 in
   a [0, 1] interval. The η-head is *nearly constant* at ≈ 0.3.
   Only in the sparse extrapolation tail (E > 2700 keV, where the
   pool has almost no events) does η climb toward 1.

## What the optimizer actually found

The NBLL loss, with no smoothness prior, prefers a *moderate
uniform* η over a bimodal {0, 1} switch:

- η ≈ 0.3 everywhere lets the decoder use PE10 features at modest
  amplitude. At peaks, this is enough to draw sharp single-bin
  spikes (SE Z −1.9, DEP Z +5.2 — both recovered).
- The *same* η ≈ 0.3 in the continuum lets the decoder draw
  bin-scale fluctuations into the (otherwise flat) Compton plateau.
  The NBLL doesn't penalize this — every fluctuation that happens
  to align with a shot-noise spike *reduces* loss. The model has
  no incentive to commit to η → 0 in the continuum.

The η-head learned to be an **inverse-density / data-scarcity
detector** rather than a peak detector. It opens the floodgates
only where attention has nothing else to work with (the sparse
extrapolation tail), and stays closed everywhere attention can
do the job. This is a perfectly reasonable solution to a
loss-shaped problem — it just isn't the solution we wanted.

## Why micro-tweaks won't fix this

Two obvious micro-tweaks would be:

- **Steeper sigmoid** (η = sigmoid(α · MLP), α ≈ 10) — forces η
  closer to {0, 1}. But the polarity is wrong; pushing harder on
  a wrong-polarity gate just makes it block PE10 *more strongly*
  at peaks.
- **Sharper σ_local** (2 → 1 keV) — doubles peak/continuum
  contrast in the η-MLP's input. Useful, but only if the MLP
  decides to use that signal correctly; given the polarity it
  already chose, more contrast may simply make the wrong-polarity
  gate sharper.

The fundamental issue is that **any global scalar gate η that
fractionally enables a wideband PE10 block leaks high-frequency
components everywhere it is non-zero**. As long as the highest
Fourier band can pass at any amplitude, the decoder can draw
sawtooth in the continuum. Pinching the gate harder doesn't
remove the high-frequency *content* — it only attenuates it
globally.

## The fork — frequency-domain gating

The fix is to move from amplitude gating (multiply the whole
latent by η) to **frequency-domain gating** (low-pass filter
the PE10 bands themselves, governed by density). Replace the
scalar η with a continuous *cutoff oracle* λ(E_*) ∈ [0, 10] and
apply a per-band soft cutoff weight

    w_l(E_*) = sigmoid(α · (λ(E_*) - l)),    α = 5

to each of the 10 Fourier bands *before* they reach the decoder.
In the continuum λ → 0 and the highest bands are physically
zeroed out at the source; at peaks λ → 10 and the full Fourier
basis is restored locally. This is **Cell 14 — Density-Guided
Bandwidth Filter (DGBF)**.

The contrast in the SFN input is simultaneously doubled by
pushing σ_local from 2 → 1 keV, matching the physical HPGe FWHM
limit.
