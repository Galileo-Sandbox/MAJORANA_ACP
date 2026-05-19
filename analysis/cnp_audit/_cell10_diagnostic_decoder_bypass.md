# Cell 10 — PE-Detached single-head DG-SFN: the third Trojan horse

**Outcome.** Cell 10 blinded the attention's `Q` and `K` projections
to PE10, dropped multi-head specialisation (one head, attn1x128), and
kept the head-tied σ + τ gates. The hypothesis was that with `Q·K^T`
now a smooth function of raw energy, the model would have *no
remaining path* to sharp localisation except through the SFN gates,
finally pressuring σ → σ_min and τ → τ_min in peak regions.

**The sawtooth is unchanged. The gates are even more trivial than in
Cell 9:**

| paradigm | MASD (1.7-2.0) | ACF1 (1.7-2.0) | DEP Z | σ/τ probe verdict |
| --- | --- | --- | --- | --- |
| Cell 6 SFN (physics) | **0.080** | −0.43 | +4.55 | (head-wise, learned) |
| Cell 7 PD-SFN (flat) | 0.292 | −0.46 | +3.26 | σ flat ≈ σ_max, τ N/A |
| Cell 8 DG-SFN (flat, head-wise) | 0.296 | −0.48 | +3.73 | 1-NN single-head escape |
| Cell 9 DG-SFN (flat, tied) | 0.290 | −0.49 | +3.89 | σ = σ_max, τ ≈ 1.05 |
| **Cell 10 PE-detach (flat, 1-head, tied)** | **0.296** | **−0.44** | +3.35 | **σ = σ_max, τ = 1.000** |

## σ/τ scalar probe across the spectrum

```
   E_*    log_l  log_g     σ(E_*) [keV]   τ(E_*)    region
   1592    5.30   7.48       199.98        1.00      PEAK
   1620    5.17   7.49       199.98        1.00      PEAK
   1750    4.74   7.49       199.97        1.00      ctrl-1
   1850    4.68   7.50       199.97        1.00      ctrl-1
   1950    4.67   7.56       199.97        1.00      ctrl-1
   2103    5.66   7.71       199.99        1.00      PEAK
   2300    5.15   7.79       199.98        1.00      ctrl-2
   2614    7.26   7.60       199.99        1.00      PEAK
```

σ pinned at σ_max ≈ 200 keV uniformly across the spectrum. **τ now at
*exactly* 1.000**, flatter than Cell 9's τ ≈ 1.05. Both gates are
more degenerate than in any prior cell.

## The third Trojan horse — decoder concat carries PE10

With Q and K blinded to PE10, `Q · K^T` becomes a smooth function of
raw energy → attention weights become near-uniform across context
events → `r_target` is essentially a *global average of V*, nearly
constant across target queries E_*.

But the decoder takes `[r_target, z_phi_T]`, and **`z_phi_T` still
carries the full PE10 Fourier features**. The decoder can therefore
produce arbitrarily sharp β̂(E_*) by using `z_phi_T` directly — the
attention output `r_target` is no longer needed for localisation.

The model has migrated the bin-grid-fitting pathway through the third
high-frequency channel:

| | Cells 7-9 | Cell 10 |
| --- | --- | --- |
| Q·K^T source | PE10 in z_phi | raw `(E_norm, T_norm)` |
| Q·K^T frequency content | sharp (bin-scale) | smooth |
| Attention used for localisation? | yes | **no, bypassed** |
| Decoder input z_phi_T | PE10 latent | **PE10 latent (unchanged)** |
| β̂(E) high-freq source | attention | **decoder concat** |
| σ/τ gates | pinned, decorative | **even more pinned** |

τ pinning at *exactly* 1.0 (vs Cell 9's 1.05) is the smoking gun:
in Cell 9 the gates still got *some* gradient because the attention
output mattered. In Cell 10 the attention is bypassed entirely, so
the σ/τ MLPs receive essentially zero gradient signal — they sit
at random-init close to the sigmoid floor.

## The remaining lever

The repo already has the config knob to close this last channel:
**`decoder_coordinate_gating`** (added in commit 2796b0a as part of
the cross-attention suite). When True, the decoder's concat input
becomes the raw 2D `(E_norm, T_norm)` vector instead of `z_phi_T`.
With that flag on alongside Cell 10's pe_detach_qk, the *only* PE10-
bearing path from input to β̂ is:

    PE10 features → h_C (via ContextPointEncoder) → V → r_target → decoder

and that path is throttled at the attention layer by σ/τ. The SFN
gates would have no remaining alternative — they would become the
sole load-bearing wall for sharp localisation.

That is the next experiment.
