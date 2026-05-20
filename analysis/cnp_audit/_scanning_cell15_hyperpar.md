# Cell 15 — hyperparameter sweep log

## Protocol

Cell 15 ultra-contrast (`flat_stratified_varN640-1024_pe10_attn1x128_gated_gab_dgsfn_tied_hardfilter_xfeed_sl1_sg50_pedetach`)
is locked as the SOTA architecture. This document tracks
hyperparameter sweeps **on top of** that locked design — one knob
at a time, one entry per trained variant.

Workflow per version:

1. **Propose** the next `cell15_vN` — pick one parameter to change,
   document only that delta relative to the base (cell15_ultra).
2. **Create** the YAML at the auto-derived paradigm path under
   `configs/cut_acceptance/simple_cnn_small/hybrid_scale/...`.
3. **Train** end-to-end (`python -m majorana_acp.cut_acceptance.cli
   <yaml>` on GPU, ~2 min) and run inference
   (`python -m scripts.diagnostics.cnp_test_inference <yaml>`).
4. **Record** the metrics block below — exactly six rows of numbers:

   | line | metric |
   |---|---|
   | 1   | `MASD` (continuum sawtooth — mean of 1.7-2.0 + 2.2-2.4 MeV regions) |
   | 2   | `overall`: pooled `z` + `cov 1σ/2σ/3σ` (§8.4.6 overall) |
   | 3   | `FE 2614`: pooled `z` + `cov 1σ/2σ/3σ` |
   | 4   | `SE 2103`: pooled `z` + `cov 1σ/2σ/3σ` |
   | 5   | `DEP 1592`: pooled `z` + `cov 1σ/2σ/3σ` |
   | 6   | `Bi 1620`: pooled `z` + `cov 1σ/2σ/3σ` |

   All coverage values use the §8.4.6 pooled-binomial formula
   (`cov_kσ = Φ(k − z) − Φ(−k − z)`). Two decimals.

5. **Review** the whole document and decide what to sweep next.

## Sweep matrix (priorities)

Highest-leverage axes for the locked Cell 15 design:

| priority | knob | base | candidate range | rationale |
|---|---|---|---|---|
| 1 | `training.n_steps` | 3000 | 1000, 5000, 10000 | longer training; DEP gap may close |
| 1 | `encoder.dropout` | 0.10 | 0.05, 0.20, 0.30 | primary σ_CNP knob (MC-Dropout) |
| 2 | `hard_filter_contrast_threshold` | 3.0 | **2.0–4.0 only** | physics prior caps the search; ≥5 already known to break SE/DEP |
| 2 | `n_trial_events_min/max` | 640/1024 | 320/512 or 1024/2048 | trial size; context budget |
| 3 | `positional_encoding.num_bands` | 10 | 8, 12, 14 | Fourier bandwidth ceiling |
| 3 | `aggregator.num_heads × attention_dim` | 1×128 | 2×128, 4×128 | multi-head specialisation |

## Baseline — cell15_ultra (locked)

Paradigm: `hybrid_scale/flat_stratified_varN640-1024_pe10_attn1x128_gated_gab_dgsfn_tied_hardfilter_xfeed_sl1_sg50_pedetach`

Config: `configs/cut_acceptance/simple_cnn_small/hybrid_scale/flat_stratified_varN640-1024_pe10_attn1x128_gated_gab_dgsfn_tied_hardfilter_xfeed_sl1_sg50_pedetach/bin10/inclusive.yaml`

| metric | value |
|---|---|
| `MASD` | **0.0096** (1.7-2.0: 0.0047 / 2.2-2.4: 0.0144) |
| `overall`   | z = −0.018  ·  cov = 0.68/0.95/1.00 |
| `FE 2614`   | z = −0.729  ·  cov = 0.57/0.90/0.99 |
| `SE 2103`   | z = −0.179  ·  cov = 0.68/0.95/1.00 |
| `DEP 1592`  | z = +2.675  ·  cov = 0.05/0.25/0.63 |
| `Bi 1620`   | z = +0.959  ·  cov = 0.49/0.85/0.98 |

DEP is the only peak still significantly miscalibrated. Everything
else lands inside ±1σ of nominal.

---

<!-- versions appended below as we sweep -->
