# Localized peak-region comparison

Source: `analysis/cnp_audit/simple_cnn_small/<paradigm>/bin10/signal/test_set_audit.json` for each paradigm.

Half-window = ±5 keV around each γ-peak. Reduced χ² target ≈ 1.0; p_DT > 0.5 = indistinguishable from sharp truth; p_DT < 0.05 = significant local miss. D_C measures match to the conditioning set, D_T measures generalization to held-out events.

Loaded cells: true_cnp, w10_fixed48, w10_varN_large, physics_w10_varN, hyper_zoom_w5

Z is **signed**: ``Z > 0`` means the empirical rate is *above* the model (CNP under-predicts); ``Z < 0`` means *below* (CNP over-predicts). p columns are two-tailed (use |Z|). The verdict column applies to D_T (held-out generalization) only.

### FE 2614

| paradigm | χ²_DC | Z_DC | p_DC | χ²_DT | Z_DT | p_DT | verdict (D_T) |
|---|---|---|---|---|---|---|---|
| true_cnp |   0.57 |  -4.55 | 5.4e-06 |   0.13 |  -0.82 | 0.412 | ~ marginal |
| w10_fixed48 |   0.81 |  -6.84 | 8.1e-12 |   0.59 |  -2.11 | 0.035 | ✗ local miss |
| w10_varN_large |   0.34 |  -3.47 | 5.3e-04 |   0.01 |  +0.22 | 0.824 | ✓ clean |
| physics_w10_varN |   0.01 |  -0.67 | 0.501 |   2.36 |  +4.68 | 2.9e-06 | ✗ local miss |
| hyper_zoom_w5 |   1.14 |  -9.29 | 1.5e-20 |   1.58 |  -3.88 | 1.1e-04 | ✗ local miss |

### SE 2103

| paradigm | χ²_DC | Z_DC | p_DC | χ²_DT | Z_DT | p_DT | verdict (D_T) |
|---|---|---|---|---|---|---|---|
| true_cnp |   0.40 |  +5.95 | 2.7e-09 |   0.23 |  -4.71 | 2.5e-06 | ✗ local miss |
| w10_fixed48 |   0.35 |  +4.97 | 6.6e-07 |   0.27 |  -4.60 | 4.2e-06 | ✗ local miss |
| w10_varN_large |   0.45 |  +6.21 | 5.2e-10 |   0.19 |  -4.29 | 1.8e-05 | ✗ local miss |
| physics_w10_varN |   1.41 | +10.12 | 4.7e-24 |   0.00 |  +0.45 | 0.650 | ✓ clean |
| hyper_zoom_w5 |   0.12 |  +4.16 | 3.1e-05 |   0.57 |  -9.56 | 1.1e-21 | ✗ local miss |

### DEP 1592

| paradigm | χ²_DC | Z_DC | p_DC | χ²_DT | Z_DT | p_DT | verdict (D_T) |
|---|---|---|---|---|---|---|---|
| true_cnp |   0.42 |  -6.60 | 4.2e-11 |   0.02 |  +0.46 | 0.642 | ✓ clean |
| w10_fixed48 |   0.50 |  -7.74 | 9.9e-15 |   0.00 |  -0.12 | 0.904 | ✓ clean |
| w10_varN_large |   0.48 |  -9.23 | 2.6e-20 |   0.00 |  +0.13 | 0.900 | ✓ clean |
| physics_w10_varN |   0.10 |  -2.98 | 0.003 |   1.51 |  +3.68 | 2.3e-04 | ✗ local miss |
| hyper_zoom_w5 |   0.80 | -15.69 | 1.8e-55 |   0.43 |  -3.45 | 5.6e-04 | ✗ local miss |

### Bi 1620

| paradigm | χ²_DC | Z_DC | p_DC | χ²_DT | Z_DT | p_DT | verdict (D_T) |
|---|---|---|---|---|---|---|---|
| true_cnp |   7.60 |  -1.28 | 0.202 |   1.85 |  -0.24 | 0.807 | ✓ clean |
| w10_fixed48 |   7.67 |  -1.29 | 0.198 |   1.59 |  -0.28 | 0.779 | ✓ clean |
| w10_varN_large |   7.58 |  -1.27 | 0.203 |   1.84 |  -0.24 | 0.810 | ✓ clean |
| physics_w10_varN |   6.55 |  -1.16 | 0.248 |   4.57 |  +0.15 | 0.880 | △ mean OK, χ² high |
| hyper_zoom_w5 |   8.27 |  -1.35 | 0.178 |   1.20 |  -0.48 | 0.629 | ✓ clean |


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
  2. `true_cnp` — p_DT=0.412, χ²_DT=  0.13  ·  ~ marginal
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
  1. `physics_w10_varN` — p_DT=0.880, χ²_DT=  4.57  ·  △ mean OK, χ² high
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
