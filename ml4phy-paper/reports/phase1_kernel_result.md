# Phase 1 Gaussian-kernel controls

Status: complete. The context-only estimator and the separately labeled
pooled-data control use the Nadaraya--Watson probability estimate. The same
formula is a common-bandwidth KDE ratio only when the passing-event density is
multiplied by the context pass fraction; it is not counted as a second method.

| Method | Peak MAE, n=500 | Continuum MAE, n=500 | Peak MAE, n=2000 | Continuum MAE, n=2000 |
|---|---:|---:|---:|---:|
| Gaussian kernel regression (context only) | 9.612 ± 1.289 | 6.467 ± 2.018 | 7.787 ± 0.963 | 5.138 ± 1.141 |
| Gaussian kernel regression (pooled data + context) | 7.768 ± 0.367 | 4.056 ± 0.248 | 7.828 ± 0.383 | 4.163 ± 0.289 |

Values are mean ± SD across ten overlapping context draws in percentage
points. Bandwidth was selected independently for every development context by
fixed five-fold context CV with seed 20260908 over 2, 5, 10, 20, 50, and 100
keV, breaking exact score ties toward the smaller bandwidth. Final outcomes
were not used for selection. The original n=2,000 context-only bandwidth
choices were reproduced exactly.

Attempt 1 stopped before producing an output archive because the context-only
Gaussian contribution could legitimately produce an all-zero denominator at a
sparse query for a 2-keV bandwidth. Attempt 2 retained exact zero context
contributions, required the pooled base and combined denominators to remain
nonzero, and completed. This numerical handling leaves the defined pooled
Nadaraya--Watson ratio unchanged. The empty attempt-1 server directory is
preserved.

The context-only method uses no acceptance pretraining data. The pooled-data
control uses all 18,866 nominal acceptance-input events plus the current
context, with no minimum-bin filter in the kernel calculation. It therefore
has a different and much larger information budget and must not be presented
as a context-only comparator. The server-only archive retains all event-level
predictions. The final reference is finite and noisy, and context draws are
overlapping rather than independent datasets.
