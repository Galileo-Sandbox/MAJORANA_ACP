# Inclusive cell audit  ·  bin10/inclusive across paradigms

Source: `analysis/cnp_audit/simple_cnn_small/<paradigm>/bin10/inclusive/test_set_audit.json` (regenerated via `python -m scripts.diagnostics.cnp_test_inference <cfg>` or the full sweep). The inclusive cell uses ``target_class="all"`` — every test event contributes to D_T, no label filter — so this is the spectrum-wide pass-rate view the experiment ultimately reports.

**Missing cells (rendered as `—`):** w10_fixed48, w10_varN_large, physics_w10_varN, hyper_zoom_w5

Loaded cells: true_cnp

## Coverage  (target: Gaussian 0.683 / 0.954 / 0.997)

**combined-σ** uses √(σ_CNP² + σ_emp²) (model uncertainty + Wilson binomial scatter); **cnp-only** uses just σ_CNP and reveals whether the model is honest about its own epistemic uncertainty. The 1σ value is the key calibration knob — much above 0.683 = bands too wide; much below = overconfident.

| paradigm | combined 1σ | combined 2σ | combined 3σ | cnp-only 1σ | cnp-only 2σ | cnp-only 3σ | verdict |
|---|---|---|---|---|---|---|---|
| true_cnp | 0.585 | 0.888 | 0.946 | 0.122 | 0.239 | 0.371 | ↓ overconfident (σ too tight) |
| w10_fixed48 | — | — | — | — | — | — | — |
| w10_varN_large | — | — | — | — | — | — | — |
| physics_w10_varN | — | — | — | — | — | — | — |
| hyper_zoom_w5 | — | — | — | — | — | — | — |

## Localized peak-region goodness-of-fit  (±5 keV window)

Each cell shows χ²_DT / Z_DT / p_DT (held-out target set, the stricter of the two sides). Reduced χ² target ≈ 1.0; p_DT > 0.5 = mean indistinguishable from sharp truth; p_DT < 0.05 = significant local miss; |Z_DT| ≫ 3 = many-σ disagreement at the peak window.

| peak | true_cnp | w10_fixed48 | w10_varN_large | physics_w10_varN | hyper_zoom_w5 |
|---|---|---|---|---|---|
|  | χ²_DT    Z_DT     p_DT | χ²_DT    Z_DT     p_DT | χ²_DT    Z_DT     p_DT | χ²_DT    Z_DT     p_DT | χ²_DT    Z_DT     p_DT |
| FE 2614 |   0.01   +0.09  0.932 | —       —       — | —       —       — | —       —       — | —       —       — |
| SE 2103 |  41.07  -21.90  2.7e-106 | —       —       — | —       —       — | —       —       — | —       —       — |
| DEP 1592 |  24.40  +16.81  2.2e-63 | —       —       — | —       —       — | —       —       — | —       —       — |
| Bi 1620 |   0.40   -0.50  0.614 | —       —       — | —       —       — | —       —       — | —       —       — |

## Per-peak ranking (held-out D_T)

Ranked by *p_DT* (high = mean indistinguishable from data). High p_DT with χ²_DT ≫ 1 means oscillation rather than tracking.

**FE 2614**
  1. `true_cnp` — p_DT=0.932, Z_DT= +0.09, χ²_DT=  0.01  ·  ✓ clean

**SE 2103**
  1. `true_cnp` — p_DT=2.7e-106, Z_DT=-21.90, χ²_DT= 41.07  ·  ✗ local miss

**DEP 1592**
  1. `true_cnp` — p_DT=2.2e-63, Z_DT=+16.81, χ²_DT= 24.40  ·  ✗ local miss

**Bi 1620**
  1. `true_cnp` — p_DT=0.614, Z_DT= -0.50, χ²_DT=  0.40  ·  ✓ clean

## Global sanity check

Spectrum-wide metrics. Pearson r close to +1 = CNP β(E) tracks D_T; mean offset close to 0 = no systematic bias.

| paradigm | N target | N context | Pearson r | mean offset | combined 1σ |
|---|---|---|---|---|---|
| true_cnp | 5659 | 1415 | +0.473 | +0.0028 | 0.585 |
| w10_fixed48 | — | — | — | — | — |
| w10_varN_large | — | — | — | — | — |
| physics_w10_varN | — | — | — | — | — |
| hyper_zoom_w5 | — | — | — | — | — |
