# Inclusive cell audit  ·  bin10/inclusive across paradigms

Source: `analysis/cnp_audit/simple_cnn_small/<paradigm>/bin10/inclusive/test_set_audit.json` (regenerated via `python -m scripts.diagnostics.cnp_test_inference <cfg>` or the full sweep). The inclusive cell uses ``target_class="all"`` — every test event contributes to D_T, no label filter — so this is the spectrum-wide pass-rate view the experiment ultimately reports.

Loaded cells: true_cnp, w10_fixed48, w10_varN_large, physics_w10_varN, hyper_zoom_w5, w10_varN_pe10, w10_varN_pe10_attn, physics_pe10_attn_gated, physics_pe9_attn_gated, physics_pure_attn4x64, physics_pe10_attn_gab, physics_pe10_attn_gab_debinned, physics_pe10_attn_gab_dense, physics_peOff_attn_gab_dense, physics_pe10_attn_gab_bpbn, physics_pe10_attn_gab_sfn, flat_pe10_attn_gab_pdsfn, flat_pe10_attn_gab_dgsfn, flat_pe10_attn_gab_dgsfn_tied, flat_pe10_attn1_pedetach_dgsfn_tied, flat_pe10_attn1_gated_pedetach_dgsfn_tied

## Coverage  (target: Gaussian 0.683 / 0.954 / 0.997)

**combined-σ** uses √(σ_CNP² + σ_emp²) (model uncertainty + Wilson binomial scatter); **cnp-only** uses just σ_CNP and reveals whether the model is honest about its own epistemic uncertainty. The 1σ value is the key calibration knob — much above 0.683 = bands too wide; much below = overconfident.

| paradigm | combined 1σ | combined 2σ | combined 3σ | cnp-only 1σ | cnp-only 2σ | cnp-only 3σ | verdict |
|---|---|---|---|---|---|---|---|
| true_cnp | 0.590 | 0.883 | 0.946 | 0.117 | 0.263 | 0.385 | ↓ overconfident (σ too tight) |
| w10_fixed48 | 0.571 | 0.859 | 0.922 | 0.141 | 0.244 | 0.307 | ↓ overconfident (σ too tight) |
| w10_varN_large | 0.571 | 0.868 | 0.927 | 0.107 | 0.215 | 0.278 | ↓ overconfident (σ too tight) |
| physics_w10_varN | 0.541 | 0.820 | 0.912 | 0.107 | 0.185 | 0.249 | ↓ overconfident (σ too tight) |
| hyper_zoom_w5 | 0.566 | 0.868 | 0.932 | 0.059 | 0.156 | 0.215 | ↓ overconfident (σ too tight) |
| w10_varN_pe10 | 0.561 | 0.849 | 0.941 | 0.146 | 0.288 | 0.395 | ↓ overconfident (σ too tight) |
| w10_varN_pe10_attn | 0.571 | 0.859 | 0.941 | 0.151 | 0.293 | 0.415 | ↓ overconfident (σ too tight) |
| physics_pe10_attn_gated | 0.580 | 0.854 | 0.937 | 0.141 | 0.283 | 0.405 | ↓ overconfident (σ too tight) |
| physics_pe9_attn_gated | 0.590 | 0.873 | 0.951 | 0.117 | 0.239 | 0.366 | ↓ overconfident (σ too tight) |
| physics_pure_attn4x64 | 0.576 | 0.815 | 0.912 | 0.117 | 0.195 | 0.268 | ↓ overconfident (σ too tight) |
| physics_pe10_attn_gab | 0.595 | 0.863 | 0.961 | 0.215 | 0.395 | 0.566 | ↓ overconfident (σ too tight) |
| physics_pe10_attn_gab_debinned | 0.580 | 0.868 | 0.961 | 0.171 | 0.366 | 0.537 | ↓ overconfident (σ too tight) |
| physics_pe10_attn_gab_dense | 0.600 | 0.868 | 0.961 | 0.195 | 0.400 | 0.580 | ↓ overconfident (σ too tight) |
| physics_peOff_attn_gab_dense | 0.541 | 0.863 | 0.941 | 0.132 | 0.224 | 0.322 | ↓ overconfident (σ too tight) |
| physics_pe10_attn_gab_bpbn | 0.580 | 0.868 | 0.966 | 0.200 | 0.410 | 0.566 | ↓ overconfident (σ too tight) |
| physics_pe10_attn_gab_sfn | 0.600 | 0.888 | 0.971 | 0.244 | 0.468 | 0.624 | ↓ overconfident (σ too tight) |
| flat_pe10_attn_gab_pdsfn | 0.502 | 0.849 | 0.966 | 0.205 | 0.420 | 0.644 | ↓ overconfident (σ too tight) |
| flat_pe10_attn_gab_dgsfn | 0.517 | 0.839 | 0.961 | 0.249 | 0.454 | 0.634 | ↓ overconfident (σ too tight) |
| flat_pe10_attn_gab_dgsfn_tied | 0.551 | 0.844 | 0.966 | 0.273 | 0.463 | 0.639 | ↓ overconfident (σ too tight) |
| flat_pe10_attn1_pedetach_dgsfn_tied | 0.571 | 0.815 | 0.971 | 0.288 | 0.517 | 0.659 | ↓ overconfident (σ too tight) |
| flat_pe10_attn1_gated_pedetach_dgsfn_tied | 0.595 | 0.888 | 0.951 | 0.063 | 0.171 | 0.254 | ↓ overconfident (σ too tight) |

## Localized peak-region goodness-of-fit  (±5 keV window, held-out D_T)

Reduced χ²_DT target ≈ 1.0; p_DT > 0.5 = mean indistinguishable from sharp truth; p_DT < 0.05 = significant local miss; |Z_DT| ≫ 3 = many-σ disagreement. Z_DT is **signed**: ``Z > 0`` means the empirical rate is *above* the model (CNP under-predicts); ``Z < 0`` means *below* (CNP over-predicts).

### FE 2614

| paradigm | χ²_DT | Z_DT | p_DT | verdict |
|---|---|---|---|---|
| true_cnp |   0.00 |  -0.03 | 0.980 | ✓ clean |
| w10_fixed48 |   0.30 |  +0.59 | 0.555 | ✓ clean |
| w10_varN_large |   1.84 |  -1.49 | 0.137 | ~ marginal |
| physics_w10_varN |   3.06 |  -2.72 | 0.007 | ✗ local miss |
| hyper_zoom_w5 |   7.32 |  -3.33 | 8.6e-04 | ✗ local miss |
| w10_varN_pe10 |   3.58 |  -2.09 | 0.037 | ✗ local miss |
| w10_varN_pe10_attn |   1.00 |  -1.08 | 0.280 | ~ marginal |
| physics_pe10_attn_gated |   0.36 |  -0.75 | 0.451 | ~ marginal |
| physics_pe9_attn_gated |   0.69 |  -1.09 | 0.275 | ~ marginal |
| physics_pure_attn4x64 |   3.00 |  -2.31 | 0.021 | ✗ local miss |
| physics_pe10_attn_gab |   0.01 |  +0.15 | 0.880 | ✓ clean |
| physics_pe10_attn_gab_debinned |   0.15 |  -0.53 | 0.593 | ✓ clean |
| physics_pe10_attn_gab_dense |   0.01 |  -0.13 | 0.896 | ✓ clean |
| physics_peOff_attn_gab_dense |   4.53 |  -3.06 | 0.002 | ✗ local miss |
| physics_pe10_attn_gab_bpbn |   0.98 |  +1.26 | 0.207 | ~ marginal |
| physics_pe10_attn_gab_sfn |   0.89 |  +1.37 | 0.170 | ~ marginal |
| flat_pe10_attn_gab_pdsfn |   0.00 |  -0.03 | 0.973 | ✓ clean |
| flat_pe10_attn_gab_dgsfn |   0.01 |  -0.11 | 0.909 | ✓ clean |
| flat_pe10_attn_gab_dgsfn_tied |   0.11 |  +0.36 | 0.716 | ✓ clean |
| flat_pe10_attn1_pedetach_dgsfn_tied |   0.13 |  -0.37 | 0.711 | ✓ clean |
| flat_pe10_attn1_gated_pedetach_dgsfn_tied |   0.00 |  +0.04 | 0.972 | ✓ clean |

### SE 2103

| paradigm | χ²_DT | Z_DT | p_DT | verdict |
|---|---|---|---|---|
| true_cnp |  38.81 | -18.04 | 8.9e-73 | ✗ local miss |
| w10_fixed48 |  33.91 | -16.54 | 1.9e-61 | ✗ local miss |
| w10_varN_large |  43.45 | -20.82 | 2.7e-96 | ✗ local miss |
| physics_w10_varN |  17.35 | -13.65 | 2.0e-42 | ✗ local miss |
| hyper_zoom_w5 |  28.99 | -18.21 | 4.5e-74 | ✗ local miss |
| w10_varN_pe10 |   0.30 |  -0.67 | 0.504 | ✓ clean |
| w10_varN_pe10_attn |   0.94 |  -1.23 | 0.219 | ~ marginal |
| physics_pe10_attn_gated |   2.87 |  -5.19 | 2.1e-07 | ✗ local miss |
| physics_pe9_attn_gated |   1.21 |  -2.28 | 0.023 | ✗ local miss |
| physics_pure_attn4x64 |  16.62 | -13.75 | 4.8e-43 | ✗ local miss |
| physics_pe10_attn_gab |   0.28 |  -1.10 | 0.272 | ~ marginal |
| physics_pe10_attn_gab_debinned |   0.35 |  -1.30 | 0.192 | ~ marginal |
| physics_pe10_attn_gab_dense |   0.52 |  -1.65 | 0.100 | ~ marginal |
| physics_peOff_attn_gab_dense |  36.09 | -15.20 | 3.4e-52 | ✗ local miss |
| physics_pe10_attn_gab_bpbn |   0.37 |  -1.49 | 0.135 | ~ marginal |
| physics_pe10_attn_gab_sfn |   0.44 |  -1.92 | 0.055 | ~ marginal |
| flat_pe10_attn_gab_pdsfn |   0.99 |  -1.19 | 0.232 | ~ marginal |
| flat_pe10_attn_gab_dgsfn |   0.31 |  -0.69 | 0.490 | ~ marginal |
| flat_pe10_attn_gab_dgsfn_tied |   0.23 |  -0.57 | 0.566 | ✓ clean |
| flat_pe10_attn1_pedetach_dgsfn_tied |   0.07 |  -0.36 | 0.717 | ✓ clean |
| flat_pe10_attn1_gated_pedetach_dgsfn_tied |  43.69 | -26.71 | 4.0e-157 | ✗ local miss |

### DEP 1592

| paradigm | χ²_DT | Z_DT | p_DT | verdict |
|---|---|---|---|---|
| true_cnp |  24.26 | +14.39 | 6.0e-47 | ✗ local miss |
| w10_fixed48 |  23.13 | +16.17 | 8.6e-59 | ✗ local miss |
| w10_varN_large |  22.60 | +18.31 | 7.3e-75 | ✗ local miss |
| physics_w10_varN |  30.64 | +19.50 | 1.0e-84 | ✗ local miss |
| hyper_zoom_w5 |  29.21 | +26.99 | 1.9e-160 | ✗ local miss |
| w10_varN_pe10 |  17.34 |  +9.68 | 3.5e-22 | ✗ local miss |
| w10_varN_pe10_attn |  19.80 | +11.94 | 7.6e-33 | ✗ local miss |
| physics_pe10_attn_gated |  52.81 | +16.23 | 3.2e-59 | ✗ local miss |
| physics_pe9_attn_gated |  18.49 | +12.69 | 6.8e-37 | ✗ local miss |
| physics_pure_attn4x64 |  25.80 | +19.02 | 1.1e-80 | ✗ local miss |
| physics_pe10_attn_gab |  13.41 |  +6.97 | 3.1e-12 | ✗ local miss |
| physics_pe10_attn_gab_debinned |  14.14 |  +5.87 | 4.3e-09 | ✗ local miss |
| physics_pe10_attn_gab_dense |  16.97 |  +8.12 | 4.8e-16 | ✗ local miss |
| physics_peOff_attn_gab_dense |  16.55 | +13.25 | 4.6e-40 | ✗ local miss |
| physics_pe10_attn_gab_bpbn |  12.48 |  +5.57 | 2.5e-08 | ✗ local miss |
| physics_pe10_attn_gab_sfn |   7.72 |  +4.55 | 5.4e-06 | ✗ local miss |
| flat_pe10_attn_gab_pdsfn |   6.08 |  +3.26 | 0.001 | ✗ local miss |
| flat_pe10_attn_gab_dgsfn |   7.70 |  +3.73 | 1.9e-04 | ✗ local miss |
| flat_pe10_attn_gab_dgsfn_tied |   6.15 |  +3.89 | 1.0e-04 | ✗ local miss |
| flat_pe10_attn1_pedetach_dgsfn_tied |   6.74 |  +3.35 | 8.1e-04 | ✗ local miss |
| flat_pe10_attn1_gated_pedetach_dgsfn_tied |  22.63 | +18.70 | 5.0e-78 | ✗ local miss |

### Bi 1620

| paradigm | χ²_DT | Z_DT | p_DT | verdict |
|---|---|---|---|---|
| true_cnp |   0.44 |  -0.51 | 0.609 | ✓ clean |
| w10_fixed48 |   0.58 |  -0.70 | 0.486 | ~ marginal |
| w10_varN_large |   0.69 |  -0.84 | 0.402 | ~ marginal |
| physics_w10_varN |   0.23 |  -0.04 | 0.969 | ✓ clean |
| hyper_zoom_w5 |   0.28 |  -0.23 | 0.821 | ✓ clean |
| w10_varN_pe10 |   2.14 |  +0.93 | 0.350 | ✗ local miss |
| w10_varN_pe10_attn |   1.05 |  +0.88 | 0.379 | ~ marginal |
| physics_pe10_attn_gated |   0.21 |  -0.48 | 0.631 | ✓ clean |
| physics_pe9_attn_gated |   0.49 |  -0.55 | 0.580 | ✓ clean |
| physics_pure_attn4x64 |   0.36 |  -0.44 | 0.661 | ✓ clean |
| physics_pe10_attn_gab |   1.06 |  -1.15 | 0.251 | ~ marginal |
| physics_pe10_attn_gab_debinned |   3.77 |  -2.29 | 0.022 | ✗ local miss |
| physics_pe10_attn_gab_dense |   1.08 |  -1.00 | 0.317 | ~ marginal |
| physics_peOff_attn_gab_dense |   1.33 |  -1.37 | 0.169 | ~ marginal |
| physics_pe10_attn_gab_bpbn |   1.13 |  -1.17 | 0.240 | ~ marginal |
| physics_pe10_attn_gab_sfn |   1.33 |  -0.94 | 0.346 | ~ marginal |
| flat_pe10_attn_gab_pdsfn |   0.52 |  -0.15 | 0.879 | ✓ clean |
| flat_pe10_attn_gab_dgsfn |   1.32 |  +0.34 | 0.732 | ✓ clean |
| flat_pe10_attn_gab_dgsfn_tied |   1.05 |  -0.09 | 0.930 | ✓ clean |
| flat_pe10_attn1_pedetach_dgsfn_tied |   0.37 |  +0.21 | 0.836 | ✓ clean |
| flat_pe10_attn1_gated_pedetach_dgsfn_tied |   0.62 |  -0.79 | 0.432 | ~ marginal |

## Per-peak ranking (held-out D_T)

Ranked by *p_DT* (high = mean indistinguishable from data). High p_DT with χ²_DT ≫ 1 means oscillation rather than tracking.

**FE 2614**
  1. `true_cnp` — p_DT=0.980, Z_DT= -0.03, χ²_DT=  0.00  ·  ✓ clean
  2. `flat_pe10_attn_gab_pdsfn` — p_DT=0.973, Z_DT= -0.03, χ²_DT=  0.00  ·  ✓ clean
  3. `flat_pe10_attn1_gated_pedetach_dgsfn_tied` — p_DT=0.972, Z_DT= +0.04, χ²_DT=  0.00  ·  ✓ clean
  4. `flat_pe10_attn_gab_dgsfn` — p_DT=0.909, Z_DT= -0.11, χ²_DT=  0.01  ·  ✓ clean
  5. `physics_pe10_attn_gab_dense` — p_DT=0.896, Z_DT= -0.13, χ²_DT=  0.01  ·  ✓ clean
  6. `physics_pe10_attn_gab` — p_DT=0.880, Z_DT= +0.15, χ²_DT=  0.01  ·  ✓ clean
  7. `flat_pe10_attn_gab_dgsfn_tied` — p_DT=0.716, Z_DT= +0.36, χ²_DT=  0.11  ·  ✓ clean
  8. `flat_pe10_attn1_pedetach_dgsfn_tied` — p_DT=0.711, Z_DT= -0.37, χ²_DT=  0.13  ·  ✓ clean
  9. `physics_pe10_attn_gab_debinned` — p_DT=0.593, Z_DT= -0.53, χ²_DT=  0.15  ·  ✓ clean
  10. `w10_fixed48` — p_DT=0.555, Z_DT= +0.59, χ²_DT=  0.30  ·  ✓ clean
  11. `physics_pe10_attn_gated` — p_DT=0.451, Z_DT= -0.75, χ²_DT=  0.36  ·  ~ marginal
  12. `w10_varN_pe10_attn` — p_DT=0.280, Z_DT= -1.08, χ²_DT=  1.00  ·  ~ marginal
  13. `physics_pe9_attn_gated` — p_DT=0.275, Z_DT= -1.09, χ²_DT=  0.69  ·  ~ marginal
  14. `physics_pe10_attn_gab_bpbn` — p_DT=0.207, Z_DT= +1.26, χ²_DT=  0.98  ·  ~ marginal
  15. `physics_pe10_attn_gab_sfn` — p_DT=0.170, Z_DT= +1.37, χ²_DT=  0.89  ·  ~ marginal
  16. `w10_varN_large` — p_DT=0.137, Z_DT= -1.49, χ²_DT=  1.84  ·  ~ marginal
  17. `w10_varN_pe10` — p_DT=0.037, Z_DT= -2.09, χ²_DT=  3.58  ·  ✗ local miss
  18. `physics_pure_attn4x64` — p_DT=0.021, Z_DT= -2.31, χ²_DT=  3.00  ·  ✗ local miss
  19. `physics_w10_varN` — p_DT=0.007, Z_DT= -2.72, χ²_DT=  3.06  ·  ✗ local miss
  20. `physics_peOff_attn_gab_dense` — p_DT=0.002, Z_DT= -3.06, χ²_DT=  4.53  ·  ✗ local miss
  21. `hyper_zoom_w5` — p_DT=8.6e-04, Z_DT= -3.33, χ²_DT=  7.32  ·  ✗ local miss

**SE 2103**
  1. `flat_pe10_attn1_pedetach_dgsfn_tied` — p_DT=0.717, Z_DT= -0.36, χ²_DT=  0.07  ·  ✓ clean
  2. `flat_pe10_attn_gab_dgsfn_tied` — p_DT=0.566, Z_DT= -0.57, χ²_DT=  0.23  ·  ✓ clean
  3. `w10_varN_pe10` — p_DT=0.504, Z_DT= -0.67, χ²_DT=  0.30  ·  ✓ clean
  4. `flat_pe10_attn_gab_dgsfn` — p_DT=0.490, Z_DT= -0.69, χ²_DT=  0.31  ·  ~ marginal
  5. `physics_pe10_attn_gab` — p_DT=0.272, Z_DT= -1.10, χ²_DT=  0.28  ·  ~ marginal
  6. `flat_pe10_attn_gab_pdsfn` — p_DT=0.232, Z_DT= -1.19, χ²_DT=  0.99  ·  ~ marginal
  7. `w10_varN_pe10_attn` — p_DT=0.219, Z_DT= -1.23, χ²_DT=  0.94  ·  ~ marginal
  8. `physics_pe10_attn_gab_debinned` — p_DT=0.192, Z_DT= -1.30, χ²_DT=  0.35  ·  ~ marginal
  9. `physics_pe10_attn_gab_bpbn` — p_DT=0.135, Z_DT= -1.49, χ²_DT=  0.37  ·  ~ marginal
  10. `physics_pe10_attn_gab_dense` — p_DT=0.100, Z_DT= -1.65, χ²_DT=  0.52  ·  ~ marginal
  11. `physics_pe10_attn_gab_sfn` — p_DT=0.055, Z_DT= -1.92, χ²_DT=  0.44  ·  ~ marginal
  12. `physics_pe9_attn_gated` — p_DT=0.023, Z_DT= -2.28, χ²_DT=  1.21  ·  ✗ local miss
  13. `physics_pe10_attn_gated` — p_DT=2.1e-07, Z_DT= -5.19, χ²_DT=  2.87  ·  ✗ local miss
  14. `physics_w10_varN` — p_DT=2.0e-42, Z_DT=-13.65, χ²_DT= 17.35  ·  ✗ local miss
  15. `physics_pure_attn4x64` — p_DT=4.8e-43, Z_DT=-13.75, χ²_DT= 16.62  ·  ✗ local miss
  16. `physics_peOff_attn_gab_dense` — p_DT=3.4e-52, Z_DT=-15.20, χ²_DT= 36.09  ·  ✗ local miss
  17. `w10_fixed48` — p_DT=1.9e-61, Z_DT=-16.54, χ²_DT= 33.91  ·  ✗ local miss
  18. `true_cnp` — p_DT=8.9e-73, Z_DT=-18.04, χ²_DT= 38.81  ·  ✗ local miss
  19. `hyper_zoom_w5` — p_DT=4.5e-74, Z_DT=-18.21, χ²_DT= 28.99  ·  ✗ local miss
  20. `w10_varN_large` — p_DT=2.7e-96, Z_DT=-20.82, χ²_DT= 43.45  ·  ✗ local miss
  21. `flat_pe10_attn1_gated_pedetach_dgsfn_tied` — p_DT=4.0e-157, Z_DT=-26.71, χ²_DT= 43.69  ·  ✗ local miss

**DEP 1592**
  1. `flat_pe10_attn_gab_pdsfn` — p_DT=0.001, Z_DT= +3.26, χ²_DT=  6.08  ·  ✗ local miss
  2. `flat_pe10_attn1_pedetach_dgsfn_tied` — p_DT=8.1e-04, Z_DT= +3.35, χ²_DT=  6.74  ·  ✗ local miss
  3. `flat_pe10_attn_gab_dgsfn` — p_DT=1.9e-04, Z_DT= +3.73, χ²_DT=  7.70  ·  ✗ local miss
  4. `flat_pe10_attn_gab_dgsfn_tied` — p_DT=1.0e-04, Z_DT= +3.89, χ²_DT=  6.15  ·  ✗ local miss
  5. `physics_pe10_attn_gab_sfn` — p_DT=5.4e-06, Z_DT= +4.55, χ²_DT=  7.72  ·  ✗ local miss
  6. `physics_pe10_attn_gab_bpbn` — p_DT=2.5e-08, Z_DT= +5.57, χ²_DT= 12.48  ·  ✗ local miss
  7. `physics_pe10_attn_gab_debinned` — p_DT=4.3e-09, Z_DT= +5.87, χ²_DT= 14.14  ·  ✗ local miss
  8. `physics_pe10_attn_gab` — p_DT=3.1e-12, Z_DT= +6.97, χ²_DT= 13.41  ·  ✗ local miss
  9. `physics_pe10_attn_gab_dense` — p_DT=4.8e-16, Z_DT= +8.12, χ²_DT= 16.97  ·  ✗ local miss
  10. `w10_varN_pe10` — p_DT=3.5e-22, Z_DT= +9.68, χ²_DT= 17.34  ·  ✗ local miss
  11. `w10_varN_pe10_attn` — p_DT=7.6e-33, Z_DT=+11.94, χ²_DT= 19.80  ·  ✗ local miss
  12. `physics_pe9_attn_gated` — p_DT=6.8e-37, Z_DT=+12.69, χ²_DT= 18.49  ·  ✗ local miss
  13. `physics_peOff_attn_gab_dense` — p_DT=4.6e-40, Z_DT=+13.25, χ²_DT= 16.55  ·  ✗ local miss
  14. `true_cnp` — p_DT=6.0e-47, Z_DT=+14.39, χ²_DT= 24.26  ·  ✗ local miss
  15. `w10_fixed48` — p_DT=8.6e-59, Z_DT=+16.17, χ²_DT= 23.13  ·  ✗ local miss
  16. `physics_pe10_attn_gated` — p_DT=3.2e-59, Z_DT=+16.23, χ²_DT= 52.81  ·  ✗ local miss
  17. `w10_varN_large` — p_DT=7.3e-75, Z_DT=+18.31, χ²_DT= 22.60  ·  ✗ local miss
  18. `flat_pe10_attn1_gated_pedetach_dgsfn_tied` — p_DT=5.0e-78, Z_DT=+18.70, χ²_DT= 22.63  ·  ✗ local miss
  19. `physics_pure_attn4x64` — p_DT=1.1e-80, Z_DT=+19.02, χ²_DT= 25.80  ·  ✗ local miss
  20. `physics_w10_varN` — p_DT=1.0e-84, Z_DT=+19.50, χ²_DT= 30.64  ·  ✗ local miss
  21. `hyper_zoom_w5` — p_DT=1.9e-160, Z_DT=+26.99, χ²_DT= 29.21  ·  ✗ local miss

**Bi 1620**
  1. `physics_w10_varN` — p_DT=0.969, Z_DT= -0.04, χ²_DT=  0.23  ·  ✓ clean
  2. `flat_pe10_attn_gab_dgsfn_tied` — p_DT=0.930, Z_DT= -0.09, χ²_DT=  1.05  ·  ✓ clean
  3. `flat_pe10_attn_gab_pdsfn` — p_DT=0.879, Z_DT= -0.15, χ²_DT=  0.52  ·  ✓ clean
  4. `flat_pe10_attn1_pedetach_dgsfn_tied` — p_DT=0.836, Z_DT= +0.21, χ²_DT=  0.37  ·  ✓ clean
  5. `hyper_zoom_w5` — p_DT=0.821, Z_DT= -0.23, χ²_DT=  0.28  ·  ✓ clean
  6. `flat_pe10_attn_gab_dgsfn` — p_DT=0.732, Z_DT= +0.34, χ²_DT=  1.32  ·  ✓ clean
  7. `physics_pure_attn4x64` — p_DT=0.661, Z_DT= -0.44, χ²_DT=  0.36  ·  ✓ clean
  8. `physics_pe10_attn_gated` — p_DT=0.631, Z_DT= -0.48, χ²_DT=  0.21  ·  ✓ clean
  9. `true_cnp` — p_DT=0.609, Z_DT= -0.51, χ²_DT=  0.44  ·  ✓ clean
  10. `physics_pe9_attn_gated` — p_DT=0.580, Z_DT= -0.55, χ²_DT=  0.49  ·  ✓ clean
  11. `w10_fixed48` — p_DT=0.486, Z_DT= -0.70, χ²_DT=  0.58  ·  ~ marginal
  12. `flat_pe10_attn1_gated_pedetach_dgsfn_tied` — p_DT=0.432, Z_DT= -0.79, χ²_DT=  0.62  ·  ~ marginal
  13. `w10_varN_large` — p_DT=0.402, Z_DT= -0.84, χ²_DT=  0.69  ·  ~ marginal
  14. `w10_varN_pe10_attn` — p_DT=0.379, Z_DT= +0.88, χ²_DT=  1.05  ·  ~ marginal
  15. `w10_varN_pe10` — p_DT=0.350, Z_DT= +0.93, χ²_DT=  2.14  ·  ✗ local miss
  16. `physics_pe10_attn_gab_sfn` — p_DT=0.346, Z_DT= -0.94, χ²_DT=  1.33  ·  ~ marginal
  17. `physics_pe10_attn_gab_dense` — p_DT=0.317, Z_DT= -1.00, χ²_DT=  1.08  ·  ~ marginal
  18. `physics_pe10_attn_gab` — p_DT=0.251, Z_DT= -1.15, χ²_DT=  1.06  ·  ~ marginal
  19. `physics_pe10_attn_gab_bpbn` — p_DT=0.240, Z_DT= -1.17, χ²_DT=  1.13  ·  ~ marginal
  20. `physics_peOff_attn_gab_dense` — p_DT=0.169, Z_DT= -1.37, χ²_DT=  1.33  ·  ~ marginal
  21. `physics_pe10_attn_gab_debinned` — p_DT=0.022, Z_DT= -2.29, χ²_DT=  3.77  ·  ✗ local miss

## Sawtooth diagnostic suite  (control regions, no γ-peaks)

Three complementary roughness metrics computed on the dense predicted β̂(E) curve inside two control windows. **MASD** = amplitude axis (mean |2nd difference|, scales with the size of the wiggles); **ED** = frequency axis (local extrema per keV); **ACF1** = pattern axis (Pearson correlation of successive first-differences). A smooth slope gives MASD ≈ 0, ED ≈ 0, ACF1 ≥ 0; random noise lands ACF1 ≈ −0.5; a deterministic up-down-up-down sawtooth drives ACF1 → −1. The two complementary axes (amplitude vs frequency vs pattern) distinguish structured oscillation from random wiggle even when individual metrics agree.

### Control region 1.7–2.0 MeV

| paradigm | MASD | ED (keV⁻¹) | ACF1 |
|---|---|---|---|
| true_cnp | 0.0057 | 0.225 | -0.584 |
| w10_fixed48 | 0.0055 | 0.215 | -0.490 |
| w10_varN_large | 0.0053 | 0.198 | -0.483 |
| physics_w10_varN | 0.0044 | 0.209 | -0.468 |
| hyper_zoom_w5 | 0.0042 | 0.222 | -0.633 |
| w10_varN_pe10 | 0.0648 | 0.192 | -0.422 |
| w10_varN_pe10_attn | 0.0803 | 0.205 | -0.434 |
| physics_pe10_attn_gated | 0.0532 | 0.168 | -0.295 |
| physics_pe9_attn_gated | 0.0188 | 0.145 | -0.148 |
| physics_pure_attn4x64 | 0.0049 | 0.188 | -0.545 |
| physics_pe10_attn_gab | 0.1204 | 0.232 | -0.554 |
| physics_pe10_attn_gab_debinned | 0.1431 | 0.202 | -0.580 |
| physics_pe10_attn_gab_dense | 0.1299 | 0.222 | -0.547 |
| physics_peOff_attn_gab_dense | 0.0043 | 0.182 | -0.470 |
| physics_pe10_attn_gab_bpbn | 0.0931 | 0.212 | -0.554 |
| physics_pe10_attn_gab_sfn | 0.0796 | 0.185 | -0.431 |
| flat_pe10_attn_gab_pdsfn | 0.2922 | 0.219 | -0.461 |
| flat_pe10_attn_gab_dgsfn | 0.2958 | 0.205 | -0.484 |
| flat_pe10_attn_gab_dgsfn_tied | 0.2898 | 0.212 | -0.488 |
| flat_pe10_attn1_pedetach_dgsfn_tied | 0.2957 | 0.212 | -0.440 |
| flat_pe10_attn1_gated_pedetach_dgsfn_tied | 0.0038 | 0.229 | -0.529 |

### Control region 2.2–2.4 MeV

| paradigm | MASD | ED (keV⁻¹) | ACF1 |
|---|---|---|---|
| true_cnp | 0.0057 | 0.193 | -0.395 |
| w10_fixed48 | 0.0071 | 0.188 | -0.427 |
| w10_varN_large | 0.0057 | 0.188 | -0.540 |
| physics_w10_varN | 0.0044 | 0.188 | -0.395 |
| hyper_zoom_w5 | 0.0041 | 0.183 | -0.416 |
| w10_varN_pe10 | 0.0540 | 0.193 | -0.411 |
| w10_varN_pe10_attn | 0.0612 | 0.193 | -0.421 |
| physics_pe10_attn_gated | 0.0799 | 0.193 | -0.479 |
| physics_pe9_attn_gated | 0.0269 | 0.137 | -0.114 |
| physics_pure_attn4x64 | 0.0052 | 0.193 | -0.478 |
| physics_pe10_attn_gab | 0.1288 | 0.208 | -0.502 |
| physics_pe10_attn_gab_debinned | 0.1221 | 0.213 | -0.607 |
| physics_pe10_attn_gab_dense | 0.1347 | 0.208 | -0.481 |
| physics_peOff_attn_gab_dense | 0.0045 | 0.193 | -0.303 |
| physics_pe10_attn_gab_bpbn | 0.1224 | 0.203 | -0.463 |
| physics_pe10_attn_gab_sfn | 0.0984 | 0.188 | -0.467 |
| flat_pe10_attn_gab_pdsfn | 0.1484 | 0.213 | -0.462 |
| flat_pe10_attn_gab_dgsfn | 0.1335 | 0.193 | -0.458 |
| flat_pe10_attn_gab_dgsfn_tied | 0.1380 | 0.193 | -0.436 |
| flat_pe10_attn1_pedetach_dgsfn_tied | 0.1404 | 0.213 | -0.400 |
| flat_pe10_attn1_gated_pedetach_dgsfn_tied | 0.0041 | 0.172 | -0.496 |

## Global sanity check

Spectrum-wide metrics. Pearson r close to +1 = CNP β(E) tracks D_T; mean offset close to 0 = no systematic bias.

| paradigm | N target | N context | Pearson r | mean offset | combined 1σ |
|---|---|---|---|---|---|
| true_cnp | 5659 | 1415 | +0.474 | +0.0029 | 0.590 |
| w10_fixed48 | 5659 | 1415 | +0.424 | +0.0029 | 0.571 |
| w10_varN_large | 5659 | 1415 | +0.466 | -0.0189 | 0.571 |
| physics_w10_varN | 5659 | 1415 | +0.351 | +0.0390 | 0.541 |
| hyper_zoom_w5 | 5659 | 1415 | +0.448 | +0.0155 | 0.566 |
| w10_varN_pe10 | 5659 | 1415 | +0.415 | +0.0079 | 0.561 |
| w10_varN_pe10_attn | 5659 | 1415 | +0.403 | +0.0122 | 0.571 |
| physics_pe10_attn_gated | 5659 | 1415 | +0.365 | -0.0040 | 0.580 |
| physics_pe9_attn_gated | 5659 | 1415 | +0.470 | -0.0117 | 0.590 |
| physics_pure_attn4x64 | 5659 | 1415 | +0.337 | +0.0189 | 0.576 |
| physics_pe10_attn_gab | 5659 | 1415 | +0.463 | +0.0158 | 0.595 |
| physics_pe10_attn_gab_debinned | 5659 | 1415 | +0.453 | -0.0122 | 0.580 |
| physics_pe10_attn_gab_dense | 5659 | 1415 | +0.460 | +0.0175 | 0.600 |
| physics_peOff_attn_gab_dense | 5659 | 1415 | +0.433 | -0.0311 | 0.541 |
| physics_pe10_attn_gab_bpbn | 5659 | 1415 | +0.459 | +0.0306 | 0.580 |
| physics_pe10_attn_gab_sfn | 5659 | 1415 | +0.448 | +0.0276 | 0.600 |
| flat_pe10_attn_gab_pdsfn | 5659 | 1415 | +0.334 | +0.0273 | 0.502 |
| flat_pe10_attn_gab_dgsfn | 5659 | 1415 | +0.314 | +0.0302 | 0.517 |
| flat_pe10_attn_gab_dgsfn_tied | 5659 | 1415 | +0.368 | +0.0267 | 0.551 |
| flat_pe10_attn1_pedetach_dgsfn_tied | 5659 | 1415 | +0.336 | +0.0113 | 0.571 |
| flat_pe10_attn1_gated_pedetach_dgsfn_tied | 5659 | 1415 | +0.464 | +0.0054 | 0.595 |
