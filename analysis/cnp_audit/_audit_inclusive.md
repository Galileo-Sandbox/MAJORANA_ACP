# Inclusive cell audit  ·  bin10/inclusive across paradigms

Source: `analysis/cnp_audit/simple_cnn_small/<paradigm>/bin10/inclusive/test_set_audit.json` (regenerated via `python -m scripts.diagnostics.cnp_test_inference <cfg>` or the full sweep). The inclusive cell uses ``target_class="all"`` — every test event contributes to D_T, no label filter — so this is the spectrum-wide pass-rate view the experiment ultimately reports.

Loaded cells: true_cnp, w10_fixed48, w10_varN_large, physics_w10_varN, hyper_zoom_w5

## Coverage  (target: Gaussian 0.683 / 0.954 / 0.997)

**combined-σ** uses √(σ_CNP² + σ_emp²) (model uncertainty + Wilson binomial scatter); **cnp-only** uses just σ_CNP and reveals whether the model is honest about its own epistemic uncertainty. The 1σ value is the key calibration knob — much above 0.683 = bands too wide; much below = overconfident.

| paradigm | combined 1σ | combined 2σ | combined 3σ | cnp-only 1σ | cnp-only 2σ | cnp-only 3σ | verdict |
|---|---|---|---|---|---|---|---|
| true_cnp | 0.576 | 0.893 | 0.946 | 0.117 | 0.259 | 0.390 | ↓ overconfident (σ too tight) |
| w10_fixed48 | 0.566 | 0.859 | 0.917 | 0.137 | 0.239 | 0.302 | ↓ overconfident (σ too tight) |
| w10_varN_large | 0.566 | 0.868 | 0.927 | 0.098 | 0.215 | 0.273 | ↓ overconfident (σ too tight) |
| physics_w10_varN | 0.546 | 0.815 | 0.917 | 0.102 | 0.176 | 0.254 | ↓ overconfident (σ too tight) |
| hyper_zoom_w5 | 0.566 | 0.863 | 0.932 | 0.068 | 0.156 | 0.254 | ↓ overconfident (σ too tight) |

## Localized peak-region goodness-of-fit  (±5 keV window, held-out D_T)

Reduced χ²_DT target ≈ 1.0; p_DT > 0.5 = mean indistinguishable from sharp truth; p_DT < 0.05 = significant local miss; |Z_DT| ≫ 3 = many-σ disagreement. Z_DT is **signed**: ``Z > 0`` means the empirical rate is *above* the model (CNP under-predicts); ``Z < 0`` means *below* (CNP over-predicts).

### FE 2614

| paradigm | χ²_DT | Z_DT | p_DT | verdict |
|---|---|---|---|---|
| true_cnp |   0.01 |  +0.11 | 0.911 | ✓ clean |
| w10_fixed48 |   0.14 |  +0.40 | 0.692 | ✓ clean |
| w10_varN_large |   2.11 |  -1.61 | 0.108 | ✗ local miss |
| physics_w10_varN |   2.40 |  -2.04 | 0.042 | ✗ local miss |
| hyper_zoom_w5 |   4.85 |  -2.52 | 0.012 | ✗ local miss |

### SE 2103

| paradigm | χ²_DT | Z_DT | p_DT | verdict |
|---|---|---|---|---|
| true_cnp |  37.22 | -15.48 | 4.9e-54 | ✗ local miss |
| w10_fixed48 |  31.34 | -15.38 | 2.2e-53 | ✗ local miss |
| w10_varN_large |  41.25 | -19.60 | 1.7e-85 | ✗ local miss |
| physics_w10_varN |  17.44 | -13.91 | 5.9e-44 | ✗ local miss |
| hyper_zoom_w5 |  31.14 | -22.33 | 1.7e-110 | ✗ local miss |

### DEP 1592

| paradigm | χ²_DT | Z_DT | p_DT | verdict |
|---|---|---|---|---|
| true_cnp |  24.77 | +18.21 | 4.3e-74 | ✗ local miss |
| w10_fixed48 |  23.82 | +19.57 | 2.9e-85 | ✗ local miss |
| w10_varN_large |  21.44 | +16.26 | 1.9e-59 | ✗ local miss |
| physics_w10_varN |  31.02 | +25.03 | 2.8e-138 | ✗ local miss |
| hyper_zoom_w5 |  28.19 | +21.06 | 1.8e-98 | ✗ local miss |

### Bi 1620

| paradigm | χ²_DT | Z_DT | p_DT | verdict |
|---|---|---|---|---|
| true_cnp |   0.43 |  -0.54 | 0.591 | ✓ clean |
| w10_fixed48 |   0.51 |  -0.65 | 0.518 | ✓ clean |
| w10_varN_large |   0.64 |  -0.79 | 0.431 | ✗ local miss |
| physics_w10_varN |   0.24 |  -0.04 | 0.966 | ✓ clean |
| hyper_zoom_w5 |   0.31 |  -0.26 | 0.792 | ✓ clean |

## Per-peak ranking (held-out D_T)

Ranked by *p_DT* (high = mean indistinguishable from data). High p_DT with χ²_DT ≫ 1 means oscillation rather than tracking.

**FE 2614**
  1. `true_cnp` — p_DT=0.911, Z_DT= +0.11, χ²_DT=  0.01  ·  ✓ clean
  2. `w10_fixed48` — p_DT=0.692, Z_DT= +0.40, χ²_DT=  0.14  ·  ✓ clean
  3. `w10_varN_large` — p_DT=0.108, Z_DT= -1.61, χ²_DT=  2.11  ·  ✗ local miss
  4. `physics_w10_varN` — p_DT=0.042, Z_DT= -2.04, χ²_DT=  2.40  ·  ✗ local miss
  5. `hyper_zoom_w5` — p_DT=0.012, Z_DT= -2.52, χ²_DT=  4.85  ·  ✗ local miss

**SE 2103**
  1. `physics_w10_varN` — p_DT=5.9e-44, Z_DT=-13.91, χ²_DT= 17.44  ·  ✗ local miss
  2. `w10_fixed48` — p_DT=2.2e-53, Z_DT=-15.38, χ²_DT= 31.34  ·  ✗ local miss
  3. `true_cnp` — p_DT=4.9e-54, Z_DT=-15.48, χ²_DT= 37.22  ·  ✗ local miss
  4. `w10_varN_large` — p_DT=1.7e-85, Z_DT=-19.60, χ²_DT= 41.25  ·  ✗ local miss
  5. `hyper_zoom_w5` — p_DT=1.7e-110, Z_DT=-22.33, χ²_DT= 31.14  ·  ✗ local miss

**DEP 1592**
  1. `w10_varN_large` — p_DT=1.9e-59, Z_DT=+16.26, χ²_DT= 21.44  ·  ✗ local miss
  2. `true_cnp` — p_DT=4.3e-74, Z_DT=+18.21, χ²_DT= 24.77  ·  ✗ local miss
  3. `w10_fixed48` — p_DT=2.9e-85, Z_DT=+19.57, χ²_DT= 23.82  ·  ✗ local miss
  4. `hyper_zoom_w5` — p_DT=1.8e-98, Z_DT=+21.06, χ²_DT= 28.19  ·  ✗ local miss
  5. `physics_w10_varN` — p_DT=2.8e-138, Z_DT=+25.03, χ²_DT= 31.02  ·  ✗ local miss

**Bi 1620**
  1. `physics_w10_varN` — p_DT=0.966, Z_DT= -0.04, χ²_DT=  0.24  ·  ✓ clean
  2. `hyper_zoom_w5` — p_DT=0.792, Z_DT= -0.26, χ²_DT=  0.31  ·  ✓ clean
  3. `true_cnp` — p_DT=0.591, Z_DT= -0.54, χ²_DT=  0.43  ·  ✓ clean
  4. `w10_fixed48` — p_DT=0.518, Z_DT= -0.65, χ²_DT=  0.51  ·  ✓ clean
  5. `w10_varN_large` — p_DT=0.431, Z_DT= -0.79, χ²_DT=  0.64  ·  ✗ local miss

## Global sanity check

Spectrum-wide metrics. Pearson r close to +1 = CNP β(E) tracks D_T; mean offset close to 0 = no systematic bias.

| paradigm | N target | N context | Pearson r | mean offset | combined 1σ |
|---|---|---|---|---|---|
| true_cnp | 5659 | 1415 | +0.482 | +0.0030 | 0.576 |
| w10_fixed48 | 5659 | 1415 | +0.424 | +0.0024 | 0.566 |
| w10_varN_large | 5659 | 1415 | +0.463 | -0.0188 | 0.566 |
| physics_w10_varN | 5659 | 1415 | +0.349 | +0.0390 | 0.546 |
| hyper_zoom_w5 | 5659 | 1415 | +0.444 | +0.0155 | 0.566 |
