# Localized peak-region comparison

Source: `analysis/cnp_audit/simple_cnn_small/<paradigm>/bin10/signal/test_set_audit.json` for each paradigm.

Half-window = ±5 keV around each γ-peak. Reduced χ² target ≈ 1.0; p_DT > 0.5 = indistinguishable from sharp truth; p_DT < 0.05 = significant local miss. D_C measures match to the conditioning set, D_T measures generalization to held-out events.

Loaded cells: true_cnp, w10_fixed48, w10_varN_large, physics_w10_varN, hyper_zoom_w5

| peak | true_cnp | w10_fixed48 | w10_varN_large | physics_w10_varN | hyper_zoom_w5 |
|---|---|---|---|---|---|
|  | χ²_DC  χ²_DT   p_DC    p_DT | χ²_DC  χ²_DT   p_DC    p_DT | χ²_DC  χ²_DT   p_DC    p_DT | χ²_DC  χ²_DT   p_DC    p_DT | χ²_DC  χ²_DT   p_DC    p_DT |
| FE 2614 |   0.57   0.13  5.4e-06 0.412 |   0.81   0.59  8.1e-12 0.035 |   0.34   0.01  5.3e-04 0.824 |   0.01   2.36  0.501 2.9e-06 |   1.14   1.58  1.5e-20 1.1e-04 |
| SE 2103 |   0.40   0.23  2.7e-09 2.5e-06 |   0.35   0.27  6.6e-07 4.2e-06 |   0.45   0.19  5.2e-10 1.8e-05 |   1.41   0.00  4.7e-24 0.650 |   0.12   0.57  3.1e-05 1.1e-21 |
| DEP 1592 |   0.42   0.02  4.2e-11 0.642 |   0.50   0.00  9.9e-15 0.904 |   0.48   0.00  2.6e-20 0.900 |   0.10   1.51  0.003 2.3e-04 |   0.80   0.43  1.8e-55 5.6e-04 |
| Bi 1620 |   7.60   1.85  0.202 0.807 |   7.67   1.59  0.198 0.779 |   7.58   1.84  0.203 0.810 |   6.55   4.57  0.248 0.880 |   8.27   1.20  0.178 0.629 |

## Paradigm legend

- `true_cnp` — `flat_stratified`, fixed N=48. The original True-CNP baseline (bin-uniform context, no focus window).
- `w10_fixed48` — `mixed_density`, 10 keV window, 70% local, fixed N=48.
- `w10_varN_large` — `mixed_density`, 10 keV window, 70% local, per-step variable N ∈ [32, 1024], n_context_max=1023.
- `physics_w10_varN` — `physics_anchored` on {Tl-208 DEP/SE/FE, Bi-214 1620}, 10 keV window, 80% local, variable N ∈ [32, 1024].
- `hyper_zoom_w5` — `mixed_density`, 5 keV window (HPGe FWHM scale), 85% local, variable N ∈ [32, 1024].

## Per-peak ranking (held-out D_T)

Ranked by *p_DT* (high = mean indistinguishable from data). Note: p_DT measures only the *mean* offset over the window; reduced χ²_DT near 1.0 is the stricter local goodness-of-fit. A cell with high p_DT and χ²_DT ≫ 1 (e.g. Bi 1620 for physics_w10_varN, χ²=4.33) is oscillating around the truth rather than tracking it.

**FE 2614**
  1. `w10_varN_large` — p_DT=0.824, χ²_DT=  0.01  ·  ✓ clean
  2. `true_cnp` — p_DT=0.412, χ²_DT=  0.13  ·  ✗ local miss
  3. `w10_fixed48` — p_DT=0.035, χ²_DT=  0.59  ·  ✗ local miss
  4. `hyper_zoom_w5` — p_DT=1.1e-04, χ²_DT=  1.58  ·  ✗ local miss
  5. `physics_w10_varN` — p_DT=2.9e-06, χ²_DT=  2.36  ·  ✗ local miss

**SE 2103**
  1. `physics_w10_varN` — p_DT=0.650, χ²_DT=  0.00  ·  ✓ clean
  2. `w10_varN_large` — p_DT=1.8e-05, χ²_DT=  0.19  ·  ✗ local miss
  3. `w10_fixed48` — p_DT=4.2e-06, χ²_DT=  0.27  ·  ✗ local miss
  4. `true_cnp` — p_DT=2.5e-06, χ²_DT=  0.23  ·  ✗ local miss
  5. `hyper_zoom_w5` — p_DT=1.1e-21, χ²_DT=  0.57  ·  ✗ local miss

**DEP 1592**
  1. `w10_fixed48` — p_DT=0.904, χ²_DT=  0.00  ·  ✓ clean
  2. `w10_varN_large` — p_DT=0.900, χ²_DT=  0.00  ·  ✓ clean
  3. `true_cnp` — p_DT=0.642, χ²_DT=  0.02  ·  ✓ clean
  4. `hyper_zoom_w5` — p_DT=5.6e-04, χ²_DT=  0.43  ·  ✗ local miss
  5. `physics_w10_varN` — p_DT=2.3e-04, χ²_DT=  1.51  ·  ✗ local miss

**Bi 1620**
  1. `physics_w10_varN` — p_DT=0.880, χ²_DT=  4.57  ·  △ mean OK, χ² high (oscillation)
  2. `w10_varN_large` — p_DT=0.810, χ²_DT=  1.84  ·  ✓ clean
  3. `true_cnp` — p_DT=0.807, χ²_DT=  1.85  ·  ✓ clean
  4. `w10_fixed48` — p_DT=0.779, χ²_DT=  1.59  ·  ✓ clean
  5. `hyper_zoom_w5` — p_DT=0.629, χ²_DT=  1.20  ·  ✓ clean

## Global sanity check

These are spectrum-wide metrics from the same audit JSONs. They tell you whether a paradigm earned its better peak fits honestly (tighter r, calibrated coverage) or by inflating σ_CNP.

| paradigm | Pearson r | mean offset | combined cov 1σ | 2σ | 3σ |
|---|---|---|---|---|---|
| true_cnp | +0.193 | -0.0099 | 0.655 | 0.989 | 0.994 |
| w10_fixed48 | +0.225 | -0.0142 | 0.695 | 0.989 | 1.000 |
| w10_varN_large | +0.186 | -0.0079 | 0.655 | 0.989 | 0.994 |
| physics_w10_varN | +0.206 | +0.0425 | 0.431 | 0.828 | 0.983 |
| hyper_zoom_w5 | +0.197 | -0.0432 | 0.816 | 1.000 | 1.000 |

Target combined coverage: 1σ ≈ 0.683, 2σ ≈ 0.954, 3σ ≈ 0.997. A 1σ value much above 0.683 means uncertainty bands are too wide; much below means the model is overconfident.
