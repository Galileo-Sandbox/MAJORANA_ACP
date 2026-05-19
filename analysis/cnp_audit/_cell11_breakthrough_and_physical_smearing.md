# Cell 11 — Architectural breakthrough + physical-smearing diagnostic

## The continuum sawtooth is dead

Cell 11 added `decoder_coordinate_gating: true` on top of Cell 10
(PE-detached Q/K, single-head, head-tied DG-SFN). With the decoder
concat now sourcing raw `(E_norm, T_norm)` instead of `z_phi_T`, the
*only* PE10-bearing path into β̂(E_*) becomes

    PE10 → phi_encoder → z_phi → ContextPointEncoder → h_C
                                                       │
                                                       ↓
                                          W_v → V (PE10 inside)
                                                       │
                                                       ↓
                                            σ,τ-gated attention
                                                       │
                                                       ↓
                                                  r_target
                                                       │
                                                       ↓
                                  decoder([r_target, raw_E_norm])

and that path is throttled by `σ(E_*)` and `τ(E_*)` at the attention
layer. The SFN has *no alternative bypass*.

The empirical confirmation is unambiguous:

| paradigm | MASD (1.7-2.0 MeV) | MASD (2.2-2.4 MeV) | ACF1 (1.7-2.0) |
| --- | --- | --- | --- |
| Cell 6 SFN (physics_anchored) | 0.0796 | 0.0984 | −0.43 |
| Cells 7-10 (flat, SFN-decorative) | 0.29-0.30 | 0.13-0.15 | −0.44 to −0.49 |
| **Cell 11 (fully gated)** | **0.0038** | **0.0041** | **−0.53** |

**MASD dropped 75× vs Cell 10 and 20× vs Cell 6.** ACF1 ≈ −0.5 is
the white-noise floor — what remains is pure MC-Dropout numerical
jitter, not structured sawtooth. The continuum is essentially
perfectly smooth.

## But peaks broke at SE 2103 and DEP 1592

| peak | Z_DT |
| --- | --- |
| Tl-208 FE 2614 | +0.04 ✓ clean |
| Tl-208 SE 2103 | **−26.71** ✗ catastrophic |
| Tl-208 DEP 1592 | **+18.70** ✗ catastrophic |
| Bi-214 1620 | −0.79 ✓ clean |

The model now under-predicts DEP (Z = +18.7) and over-predicts SE
(Z = −26.7). FE and Bi-1620 stayed clean.

## σ/τ probe — the gates ARE doing real work now

```
   E_*    log_l  log_g     σ(E_*) [keV]   τ(E_*)    region
   1450    4.61   7.44       194.56        1.00
   1592    5.30   7.48       198.02        1.00      DEP
   1620    5.17   7.49       197.61        1.00      Bi PEAK
   1850    4.68   7.50       194.92        1.00      ctrl-1
   2103    5.66   7.71       198.66        1.00      SE
   2400    5.06   7.77       196.79        1.00      ctrl-2
   2614    7.26   7.60       199.51        1.00      FE PEAK
   2800   −0.06   6.62        14.37        1.02      tail
```

σ no longer flat-pinned at σ_max — it varies from 14 keV (sparse
2800 keV tail) up to ~200 keV (peaks). The MLP is finally a
learnable function rather than a trivial constant. τ stayed near
1.0 because once σ is doing the work, the temperature gate gets no
additional gradient pressure.

But **σ does not contract at DEP or SE.** σ at DEP = 198 keV; σ at
adjacent continuum (1850) = 194.9 keV. Nearly identical. The SFN
learned to use σ only as a "data-sparsity detector" (small σ when
`log_l → −∞`), not as a "peak resolution detector".

## Diagnosis — the physical-smearing problem

The SFN sees only `[log ρ_local, log ρ_global]` with `σ_local = 10
keV`. A 10 keV Gaussian kernel **convolves the empirical event
distribution against a 10 keV smear**. But HPGe γ-peak FWHM is
~2 keV. The kernel is therefore ~5× wider than the physical
peak width — it dilutes peak densities into the surrounding
Compton continuum:

| location | log_l (σ_local = 10 keV) | physical reality |
| --- | --- | --- |
| DEP 1592 | 5.30 | sharp ~2-keV-FWHM γ-line |
| continuum 1850 | 4.68 | smooth Compton background |
| SE 2103 | 5.66 | sharp ~2-keV-FWHM γ-line |
| continuum 2400 | 5.06 | smooth Compton background |

The peak/continuum contrast in log_l is only ~0.5-0.6 (a factor of
~1.8 in density). The Compton continuum near DEP is *almost as
event-dense as DEP itself* once the kernel is wider than the peak.
The MLP **physically cannot** distinguish "moderate density that is
a sharp γ-line" from "moderate density that is smooth continuum"
from the two log-density features alone.

FE 2614 stayed clean because its log_l = 7.26 — dramatically above
everything else, because FE dominates the spectrum tail with no
Compton background above it to dilute the contrast. Bi 1620 stayed
clean for similar reasons. SE 2103 and DEP 1592 sit on dense
Compton backgrounds and *vanish into them* under a 10-keV kernel.

## Implication

Sharpening `σ_local` to match the physical HPGe FWHM (~2 keV)
should expose the peaks against the continuum:

- 2 keV-σ kernel centred on a 2 keV-FWHM γ-line ≈ √2 × peak height
- 2 keV-σ kernel centred on smooth continuum ≈ same density as before

The expected log_l contrast at peak vs adjacent continuum would
grow from ~0.5 to ~3-4, giving the SFN the resolving signal it
currently lacks. The architecture stays untouched; this is purely a
data-preprocessing correction to match the physical resolution of
the underlying detector.
