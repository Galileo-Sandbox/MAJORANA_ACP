# Cell 9 — Tied-Head DG-SFN: the "trivial gate" pathology

**Outcome.** Cell 9 tied σ and τ across all attention heads (one scalar
per target query, broadcast across H = 8) to remove the 1-NN
single-head escape route observed in Cell 8. **The continuum sawtooth
did not move** — and the underlying mechanism flipped from
*head-specialisation collapse* to a *deeper trivial-gate collapse*:

| paradigm | MASD (1.7-2.0) | ACF1 (1.7-2.0) | DEP Z |
| --- | --- | --- | --- |
| Cell 6 SFN (physics, head-wise σ) | **0.080** | −0.43 | +4.55 |
| Cell 7 PD-SFN (flat, head-wise σ)         | 0.292 | −0.46 | +3.26 |
| Cell 8 DG-SFN (flat, head-wise σ+τ)       | 0.296 | −0.48 | +3.73 |
| **Cell 9 DG-SFN (flat, head-tied σ+τ)** | **0.290** | **−0.49** | +3.89 |

The expected gradient pressure to grow σ and τ jointly in
low-density regions never materialised. Why became clear by probing
the trained scalar gates directly.

## σ/τ scalar probe across the spectrum

With head-tying, the σ-MLP and τ-MLP each emit a single scalar per
target query:

```
   E_*    log_l  log_g     σ(E_*) [keV]   τ(E_*)    region
   1592    5.30   7.48       199.97        1.05      PEAK
   1620    5.17   7.49       199.97        1.05      PEAK
   1750    4.74   7.49       199.96        1.05      ctrl-1
   1850    4.68   7.50       199.96        1.05      ctrl-1
   1950    4.67   7.56       199.96        1.05      ctrl-1
   2103    5.66   7.71       199.98        1.04      PEAK
   2300    5.15   7.79       199.97        1.04      ctrl-2
   2500    4.03   7.74       199.95        1.04
   2614    7.26   7.60       199.99        1.04      PEAK
   2800  −0.06   6.62       197.19        1.13      tail
```

**σ has collapsed to σ_max = 200 keV uniformly across the spectrum** —
peaks AND continuum. The peak-vs-continuum log_local contrast (~5.7 vs
~4.7) was available to the MLP but is being ignored. **τ has
collapsed to τ_min ≈ 1.05 uniformly.** Both gates are pinned to
extremes that make them effectively no-ops:

- σ → σ_max ⇒ Δ²/(2σ²) ≪ |Q·K^T|/√d for most context-event distances,
  so the spatial penalty does not drive attention.
- τ → τ_min = 1.0 ⇒ no temperature softening; the softmax is dominated
  by the raw Q·K^T magnitudes.

## Why the gates went trivial — the Q·K^T "Trojan horse"

The cross-attention computes Q from `z_phi_T` and K from `z_phi_C`,
both of which are PE10-encoded latents. With L = 10 Fourier bands, the
highest band period is ≈ 9.8 keV (≈ bin-grid scale). The PE10 features
inside Q and K can therefore produce **sharp, bin-scale variations
in `Q·K^T` without involving the SFN at all**. The model gets a
"free" high-frequency attention pathway that the spatial penalty
cannot suppress unless σ is small enough — but σ shrinking *also*
costs NBLL at the bins between events.

The result: training discovers that the *trivial-gate* configuration
(σ = σ_max, τ = τ_min, attention driven entirely by PE10-Fourier
Q·K^T spikes) achieves the same loss as a *position-aware-gate*
configuration. Both fit individual events at the bin scale; only the
mechanism differs. With both equilibria available, training falls into
the trivial one — it has no gradient pressure to discover the
position-aware one.

Cell 8's "1-NN single-head" pathology and Cell 9's "trivial gate"
pathology are both expressions of the same underlying loophole: **the
PE10 features inside Q/K give the model a load-free path to fit the
bin grid**. The SFN gates are decorative, not load-bearing.

## Implication for Cell 10

Tying heads removed one escape; the model just walked to a different
escape. Architectural constraint must go *deeper*: blind Q and K to
PE10 entirely, while keeping PE10 in V and the decoder. Then
`Q·K^T` becomes a smooth function of raw energy (a low-pass-filtered
inner product over `(E_norm, T_norm)`), and the *only* mechanism left
for sharp localisation flows through the σ/τ gates — they become
the load-bearing wall, not a decoration. Single attention head
suffices, since multi-head only enabled the 1-NN-specialist escape
that Cells 7/8/9 each exploited differently.
