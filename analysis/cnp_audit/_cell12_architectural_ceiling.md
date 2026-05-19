# Cell 12 — Sharper kernel activates the SFN, but reveals an architectural ceiling

## Empirical result

Cell 12 took Cell 11's architecture verbatim and only sharpened
the pool-density local kernel from σ_local = 10 keV → 2 keV (to
match the physical ~2 keV HPGe γ-peak FWHM). All else identical.

| metric | Cell 11 | Cell 12 (σ_l = 2) | verdict |
| --- | --- | --- | --- |
| MASD (1.7-2.0 MeV) | 0.0038 | 0.0041 | smooth ✓ |
| MASD (2.2-2.4 MeV) | 0.0041 | 0.0056 | smooth ✓ |
| FE Z | +0.04 | −1.02 | slightly worse |
| **SE Z** | **−26.7** | **−28.8** | **worse** |
| **DEP Z** | **+18.7** | **+23.0** | **worse** |
| Bi Z | −0.79 | −0.71 | clean |

## The kernel-resolution fix worked exactly as predicted

The log_l peak/continuum contrast jumped from ~0.5 (Cell 11) to
~1.4-3.9 (Cell 12):

```
   E_*    log_l(σ_l=2)  contrast vs 1850   σ(E_*)    τ(E_*)
   1592       4.61         +1.45            197.6     1.10    DEP
   1620       4.54         +1.37            197.9     1.11    Bi PEAK
   1850       3.16          0.00            199.4     1.68    ctrl-1
   2103       4.97         +1.80            197.2     1.07    SE
   2614       7.05         +3.89            173.1     1.10    FE PEAK
   2800      −8.67        −11.84              5.0     9.78    sparse tail
```

For the first time across Cells 7-12, **both σ and τ vary
non-trivially with energy**:

- σ trims at FE (173 keV vs ~199 in continuum)
- σ collapses to σ_min in the sparse tail
- τ varies 1.07 (peaks) → 1.68 (continuum) → 9.78 (sparse tail)

The SFN is finally a learnable function of position, not a trivial
constant. The architectural sequence Cells 7→11 successfully
forced the SFN to be load-bearing; the kernel-resolution fix gave
it the input signal it needed.

## But peaks still failed — the architectural ceiling

The σ change at DEP vs adjacent continuum is **0.4 keV** (197.6 vs
199.4). The MLP found a *gentle* slope rather than a sharp step.
And even if σ had dropped all the way to σ_min = 5 keV at DEP, **it
would not have helped**. The Cell 11/12 architecture is smooth-only
by construction:

```
  Q · K^T          ←  raw (E_norm, T_norm)    smooth function of E
  α(E*, E_i)       =  softmax(Q·K^T/√d/τ − Δ²/2σ²)   smooth in E*
  r_target         =  Σ α · V_i               smooth linear combo
  decoder input    =  [r_target, raw_E_norm]  no Fourier features
  β̂(E*)            =  decoder(...)            necessarily smooth
```

Reducing σ tightens *which V's* contribute, but the result is still
a smooth function of E_*. **There is no path in this architecture
for β̂(E) to spike sharply at a single 10 keV bin.** The smoothness
that gave us MASD ≈ 0.004 in the continuum is the *same property*
that prevents β̂ from resolving SE / DEP.

FE 2614 survives because it dominates the spectrum tail with no
Compton above it — a smooth fit captures it. Bi 1620 spans two
bins (broader effective FWHM) — smooth captures it. SE and DEP
are single-bin sharp jumps that any all-smooth model must miss.

## The fork

The design space has a clear ceiling we've now mapped:

| | continuum MASD | DEP Z | SE Z |
| --- | --- | --- | --- |
| Cell 6 (physics, PE10 free in Q/K) | 0.080 | +4.55 | −1.92 |
| **Cell 11/12 (every PE10 bypass closed)** | **0.004** | **+18-23** | **−27-29** |

We cannot reach the upper-left corner (low MASD AND low |Z|) with
either extreme. The way forward is **conditional gating**: route
PE10 through a high-frequency path *only where the data says it
should exist*. Specifically, add a third SFN head η(log_l, log_g)
∈ [0, 1] that controls how much z_phi_T's PE10 features leak into
the decoder concat:

    decoder_input = [r_target, raw_E_norm, η(E_*) · z_phi_T]

In continuum (low log_l contrast) → η → 0 → decoder sees the raw
coords only → smooth β̂ (Cell 11/12 regime).
At peaks (high log_l contrast) → η → 1 → decoder sees full PE10
→ sharp β̂ (Cell 6 regime).

The on/off switch sits under the SAME density-aware mechanism that
already controls σ and τ — it's not a new bypass, it's the SFN
explicitly *authorising* a high-frequency channel where the data
says peaks exist. Cell 13 is this design.
