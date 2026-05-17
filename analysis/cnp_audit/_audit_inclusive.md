# Inclusive cell audit  ·  bin10/inclusive across paradigms

Source: `analysis/cnp_audit/simple_cnn_small/<paradigm>/bin10/inclusive/test_set_audit.json` (regenerated via `python -m scripts.diagnostics.cnp_test_inference <cfg>` or the full sweep). The inclusive cell uses ``target_class="all"`` — every test event contributes to D_T, no label filter — so this is the spectrum-wide pass-rate view the experiment ultimately reports.

Loaded cells: true_cnp, w10_fixed48, w10_varN_large, physics_w10_varN, hyper_zoom_w5

## Coverage  (target: Gaussian 0.683 / 0.954 / 0.997)

**combined-σ** uses √(σ_CNP² + σ_emp²) (model uncertainty + Wilson binomial scatter); **cnp-only** uses just σ_CNP and reveals whether the model is honest about its own epistemic uncertainty. The 1σ value is the key calibration knob — much above 0.683 = bands too wide; much below = overconfident.

| paradigm | combined 1σ | combined 2σ | combined 3σ | cnp-only 1σ | cnp-only 2σ | cnp-only 3σ | verdict |
|---|---|---|---|---|---|---|---|
| true_cnp | 0.585 | 0.888 | 0.946 | 0.122 | 0.239 | 0.371 | ↓ overconfident (σ too tight) |
| w10_fixed48 | 0.571 | 0.859 | 0.927 | 0.151 | 0.234 | 0.302 | ↓ overconfident (σ too tight) |
| w10_varN_large | 0.566 | 0.868 | 0.927 | 0.112 | 0.224 | 0.307 | ↓ overconfident (σ too tight) |
| physics_w10_varN | 0.556 | 0.824 | 0.917 | 0.137 | 0.239 | 0.351 | ↓ overconfident (σ too tight) |
| hyper_zoom_w5 | 0.580 | 0.868 | 0.932 | 0.127 | 0.215 | 0.332 | ↓ overconfident (σ too tight) |

## Localized peak-region goodness-of-fit  (±5 keV window)

Each cell shows χ²_DT / Z_DT / p_DT (held-out target set, the stricter of the two sides). Reduced χ² target ≈ 1.0; p_DT > 0.5 = mean indistinguishable from sharp truth; p_DT < 0.05 = significant local miss; |Z_DT| ≫ 3 = many-σ disagreement at the peak window.

| peak | true_cnp | w10_fixed48 | w10_varN_large | physics_w10_varN | hyper_zoom_w5 |
|---|---|---|---|---|---|
|  | χ²_DT    Z_DT     p_DT | χ²_DT    Z_DT     p_DT | χ²_DT    Z_DT     p_DT | χ²_DT    Z_DT     p_DT | χ²_DT    Z_DT     p_DT |
| FE 2614 |   0.01   +0.09  0.932 |   0.43   +0.72  0.474 |   2.14   -1.67  0.094 |   2.15   -1.85  0.064 |   5.86   -2.75  0.006 |
| SE 2103 |  41.07  -21.90  2.7e-106 |  33.59  -16.41  1.7e-60 |  40.93  -17.40  8.5e-68 |  16.36   -9.69  3.4e-22 |  29.31  -15.02  5.1e-51 |
| DEP 1592 |  24.40  +16.81  2.2e-63 |  22.87  +17.46  2.9e-68 |  21.80  +15.99  1.5e-57 |  27.06  +13.20  9.4e-40 |  26.31  +14.52  9.5e-48 |
| Bi 1620 |   0.40   -0.50  0.614 |   0.50   -0.63  0.526 |   0.63   -0.81  0.416 |   0.23   -0.09  0.928 |   0.31   -0.25  0.800 |

## Per-peak ranking (held-out D_T)

Ranked by *p_DT* (high = mean indistinguishable from data). High p_DT with χ²_DT ≫ 1 means oscillation rather than tracking.

**FE 2614**
  1. `true_cnp` — p_DT=0.932, Z_DT= +0.09, χ²_DT=  0.01  ·  ✓ clean
  2. `w10_fixed48` — p_DT=0.474, Z_DT= +0.72, χ²_DT=  0.43  ·  ✗ local miss
  3. `w10_varN_large` — p_DT=0.094, Z_DT= -1.67, χ²_DT=  2.14  ·  ✗ local miss
  4. `physics_w10_varN` — p_DT=0.064, Z_DT= -1.85, χ²_DT=  2.15  ·  ✗ local miss
  5. `hyper_zoom_w5` — p_DT=0.006, Z_DT= -2.75, χ²_DT=  5.86  ·  ✗ local miss

**SE 2103**
  1. `physics_w10_varN` — p_DT=3.4e-22, Z_DT= -9.69, χ²_DT= 16.36  ·  ✗ local miss
  2. `hyper_zoom_w5` — p_DT=5.1e-51, Z_DT=-15.02, χ²_DT= 29.31  ·  ✗ local miss
  3. `w10_fixed48` — p_DT=1.7e-60, Z_DT=-16.41, χ²_DT= 33.59  ·  ✗ local miss
  4. `w10_varN_large` — p_DT=8.5e-68, Z_DT=-17.40, χ²_DT= 40.93  ·  ✗ local miss
  5. `true_cnp` — p_DT=2.7e-106, Z_DT=-21.90, χ²_DT= 41.07  ·  ✗ local miss

**DEP 1592**
  1. `physics_w10_varN` — p_DT=9.4e-40, Z_DT=+13.20, χ²_DT= 27.06  ·  ✗ local miss
  2. `hyper_zoom_w5` — p_DT=9.5e-48, Z_DT=+14.52, χ²_DT= 26.31  ·  ✗ local miss
  3. `w10_varN_large` — p_DT=1.5e-57, Z_DT=+15.99, χ²_DT= 21.80  ·  ✗ local miss
  4. `true_cnp` — p_DT=2.2e-63, Z_DT=+16.81, χ²_DT= 24.40  ·  ✗ local miss
  5. `w10_fixed48` — p_DT=2.9e-68, Z_DT=+17.46, χ²_DT= 22.87  ·  ✗ local miss

**Bi 1620**
  1. `physics_w10_varN` — p_DT=0.928, Z_DT= -0.09, χ²_DT=  0.23  ·  ✓ clean
  2. `hyper_zoom_w5` — p_DT=0.800, Z_DT= -0.25, χ²_DT=  0.31  ·  ✓ clean
  3. `true_cnp` — p_DT=0.614, Z_DT= -0.50, χ²_DT=  0.40  ·  ✓ clean
  4. `w10_fixed48` — p_DT=0.526, Z_DT= -0.63, χ²_DT=  0.50  ·  ✓ clean
  5. `w10_varN_large` — p_DT=0.416, Z_DT= -0.81, χ²_DT=  0.63  ·  ✗ local miss

## Global sanity check

Spectrum-wide metrics. Pearson r close to +1 = CNP β(E) tracks D_T; mean offset close to 0 = no systematic bias.

| paradigm | N target | N context | Pearson r | mean offset | combined 1σ |
|---|---|---|---|---|---|
| true_cnp | 5659 | 1415 | +0.473 | +0.0028 | 0.585 |
| w10_fixed48 | 5659 | 1415 | +0.430 | +0.0027 | 0.571 |
| w10_varN_large | 5659 | 1415 | +0.458 | -0.0199 | 0.566 |
| physics_w10_varN | 5659 | 1415 | +0.353 | +0.0365 | 0.556 |
| hyper_zoom_w5 | 5659 | 1415 | +0.444 | +0.0137 | 0.580 |
