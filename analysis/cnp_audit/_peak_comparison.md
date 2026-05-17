# Localized peak-region comparison

Source: `analysis/cnp_audit/simple_cnn_small/<paradigm>/bin10/signal/test_set_audit.json` for each paradigm.

Half-window = ±5 keV around each γ-peak. Reduced χ² target ≈ 1.0; p_DT > 0.5 = indistinguishable from sharp truth; p_DT < 0.05 = significant local miss. D_C measures match to the conditioning set, D_T measures generalization to held-out events.

Loaded cells: true_cnp, w10_fixed48, w10_varN_large, physics_w10_varN, hyper_zoom_w5

| peak | true_cnp | w10_fixed48 | w10_varN_large | physics_w10_varN | hyper_zoom_w5 |
|---|---|---|---|---|---|
|  | χ²_DC  χ²_DT   p_DC    p_DT | χ²_DC  χ²_DT   p_DC    p_DT | χ²_DC  χ²_DT   p_DC    p_DT | χ²_DC  χ²_DT   p_DC    p_DT | χ²_DC  χ²_DT   p_DC    p_DT |
| FE 2614 |   0.55   0.11  4.3e-08 0.380 |   0.85   0.69  3.3e-13 0.019 |   0.34   0.01  4.2e-04 0.849 |   0.01   2.39  0.519 2.4e-06 |   1.21   1.78  4.0e-18 1.6e-04 |
| SE 2103 |   0.42   0.22  8.2e-10 3.3e-06 |   0.35   0.27  6.5e-06 2.8e-05 |   0.44   0.21  2.1e-10 4.5e-06 |   1.34   0.00  2.3e-21 0.846 |   0.13   0.54  9.3e-04 1.6e-12 |
| DEP 1592 |   0.43   0.02  3.9e-11 0.674 |   0.45   0.01  5.8e-12 0.801 |   0.46   0.01  3.4e-14 0.803 |   0.11   1.40  3.9e-04 8.4e-05 |   0.83   0.48  2.1e-43 0.001 |
| Bi 1620 |   7.58   2.06  0.204 0.817 |   7.61   1.61  0.200 0.794 |   7.59   1.80  0.202 0.803 |   6.56   4.33  0.246 0.892 |   8.32   1.25  0.177 0.623 |

## Paradigm legend

- `true_cnp` — `flat_stratified`, fixed N=48. The original True-CNP baseline (bin-uniform context, no focus window).
- `w10_fixed48` — `mixed_density`, 10 keV window, 70% local, fixed N=48.
- `w10_varN_large` — `mixed_density`, 10 keV window, 70% local, per-step variable N ∈ [32, 1024], n_context_max=1023.
- `physics_w10_varN` — `physics_anchored` on {Tl-208 DEP/SE/FE, Bi-214 1620}, 10 keV window, 80% local, variable N ∈ [32, 1024].
- `hyper_zoom_w5` — `mixed_density`, 5 keV window (HPGe FWHM scale), 85% local, variable N ∈ [32, 1024].

## Per-peak ranking (held-out D_T)

Ranked by *p_DT* (high = mean indistinguishable from data). Note: p_DT measures only the *mean* offset over the window; reduced χ²_DT near 1.0 is the stricter local goodness-of-fit. A cell with high p_DT and χ²_DT ≫ 1 (e.g. Bi 1620 for physics_w10_varN, χ²=4.33) is oscillating around the truth rather than tracking it.

**FE 2614**
  1. `w10_varN_large` — p_DT=0.849, χ²_DT=  0.01  ·  ✓ clean
  2. `true_cnp` — p_DT=0.380, χ²_DT=  0.11  ·  ✗ local miss
  3. `w10_fixed48` — p_DT=0.019, χ²_DT=  0.69  ·  ✗ local miss
  4. `hyper_zoom_w5` — p_DT=1.6e-04, χ²_DT=  1.78  ·  ✗ local miss
  5. `physics_w10_varN` — p_DT=2.4e-06, χ²_DT=  2.39  ·  ✗ local miss

**SE 2103**
  1. `physics_w10_varN` — p_DT=0.846, χ²_DT=  0.00  ·  ✓ clean
  2. `w10_fixed48` — p_DT=2.8e-05, χ²_DT=  0.27  ·  ✗ local miss
  3. `w10_varN_large` — p_DT=4.5e-06, χ²_DT=  0.21  ·  ✗ local miss
  4. `true_cnp` — p_DT=3.3e-06, χ²_DT=  0.22  ·  ✗ local miss
  5. `hyper_zoom_w5` — p_DT=1.6e-12, χ²_DT=  0.54  ·  ✗ local miss

**DEP 1592**
  1. `w10_varN_large` — p_DT=0.803, χ²_DT=  0.01  ·  ✓ clean
  2. `w10_fixed48` — p_DT=0.801, χ²_DT=  0.01  ·  ✓ clean
  3. `true_cnp` — p_DT=0.674, χ²_DT=  0.02  ·  ✓ clean
  4. `hyper_zoom_w5` — p_DT=0.001, χ²_DT=  0.48  ·  ✗ local miss
  5. `physics_w10_varN` — p_DT=8.4e-05, χ²_DT=  1.40  ·  ✗ local miss

**Bi 1620**
  1. `physics_w10_varN` — p_DT=0.892, χ²_DT=  4.33  ·  △ mean OK, χ² high (oscillation)
  2. `true_cnp` — p_DT=0.817, χ²_DT=  2.06  ·  △ mean OK, χ² high (oscillation)
  3. `w10_varN_large` — p_DT=0.803, χ²_DT=  1.80  ·  ✓ clean
  4. `w10_fixed48` — p_DT=0.794, χ²_DT=  1.61  ·  ✓ clean
  5. `hyper_zoom_w5` — p_DT=0.623, χ²_DT=  1.25  ·  ✓ clean

## Global sanity check

These are spectrum-wide metrics from the same audit JSONs. They tell you whether a paradigm earned its better peak fits honestly (tighter r, calibrated coverage) or by inflating σ_CNP.

| paradigm | Pearson r | mean offset | combined cov 1σ | 2σ | 3σ |
|---|---|---|---|---|---|
| true_cnp | +0.195 | -0.0098 | 0.661 | 0.989 | 0.994 |
| w10_fixed48 | +0.192 | -0.0141 | 0.684 | 0.989 | 0.994 |
| w10_varN_large | +0.202 | -0.0081 | 0.649 | 0.989 | 0.994 |
| physics_w10_varN | +0.208 | +0.0422 | 0.414 | 0.839 | 0.983 |
| hyper_zoom_w5 | +0.193 | -0.0434 | 0.810 | 1.000 | 1.000 |

Target combined coverage: 1σ ≈ 0.683, 2σ ≈ 0.954, 3σ ≈ 0.997. A 1σ value much above 0.683 means uncertainty bands are too wide; much below means the model is overconfident.
