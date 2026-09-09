# Reused final local-shape evidence

Status: exported from the 60 saved final neural prediction cells; no inference or training was rerun. This is a prospectively specified follow-up analysis on historically exposed data.

## Five-keV bin errors

Predictions were first evaluated at each reference event energy and then averaged within bins. Values below are percentage points and use training seeds as the top-level replication unit.

| Region | Attentive CNP + PE MAE | Density-guided CNP MAE | Attentive CNP + PE RMSE | Density-guided CNP RMSE | Valid/excluded bins |
|---|---:|---:|---:|---:|---:|
| Full 500-3000 keV | 8.414 ± 0.499 | 4.408 ± 0.058 | 10.765 ± 0.445 | 5.827 ± 0.071 | 442/58 |
| Tl-208 DEP 1592 keV | 5.906 ± 1.148 | 3.050 ± 0.419 | 7.091 ± 1.224 | 3.642 ± 0.350 | 6/0 |
| Bi-212 1620.74-keV feature | 8.188 ± 0.664 | 3.390 ± 0.899 | 9.786 ± 0.832 | 4.059 ± 1.162 | 6/0 |
| Continuum 1700-2000 keV | 7.424 ± 0.629 | 4.796 ± 0.498 | 9.236 ± 0.757 | 5.541 ± 0.376 | 60/0 |
| Tl-208 SE 2103 keV | 5.235 ± 1.109 | 3.127 ± 0.210 | 6.634 ± 1.315 | 4.135 ± 0.576 | 6/0 |
| Continuum 2200-2400 keV | 7.342 ± 1.271 | 3.368 ± 0.586 | 8.616 ± 1.265 | 4.052 ± 0.739 | 40/0 |
| Tl-208 FE 2614 keV | 4.900 ± 1.426 | 4.353 ± 0.205 | 5.925 ± 1.855 | 5.428 ± 0.538 | 5/1 |
| Sparse tail 2700-3000 keV | 23.189 ± 1.261 | 11.932 ± 1.153 | 26.014 ± 2.121 | 15.366 ± 1.244 | 13/47 |

## Peak/sideband contrast error

| Feature | Attentive CNP + PE absolute error (pp) | Density-guided CNP absolute error (pp) | Center/sideband events |
|---|---:|---:|---:|
| Tl-208 DEP 1592 keV | 3.579 ± 1.199 | 2.790 ± 1.219 | 828/464 |
| Bi-212 1620.74-keV feature | 5.283 ± 1.876 | 3.064 ± 1.626 | 790/541 |
| Tl-208 SE 2103 keV | 10.266 ± 1.053 | 3.955 ± 1.348 | 1356/657 |
| Tl-208 FE 2614 keV | 5.702 ± 0.507 | 3.596 ± 1.239 | 8522/202 |

## Limits

- The 114,400-event reference is finite and noisy, not an exact acceptance function.
- The exported curve band is the standard deviation across three training-seed means after averaging ten contexts. It is not a calibrated confidence or posterior interval.
- Contexts overlap and share a target. Full cell-level values and context dispersion are retained in the CSV files.
- The original 50-pass roughness statistic remains unsuitable for a smoothness claim because MC-estimator noise was material.
- These two methods are the only matched neural pair already complete. Missing matched CNP and Attentive CNP baselines must not be filled with confounded historical checkpoints.
