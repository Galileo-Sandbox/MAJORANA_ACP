# Cache-Only Reanalysis

Status tags: **historical**, **fixed context**, and **interpolated where applicable**. No model inference or training was run.

## Protocol

All five caches share 18,866 training, 2,000 context, and 5,074 target energy-score pairs. Predictions at target event energies are linear interpolations of the saved 800-point dense grid. The fixed threshold is `0.540643572807312` and was historically selected by Youden-J on the same 7,074-event evaluation file; these numbers are development evidence, not independent final-test evidence.

The context-only Gaussian Nadaraya-Watson baseline selected `100 keV` from [2.0, 5.0, 10.0, 20.0, 50.0, 100.0] by five-fold context-only Brier score (seed 20260908), then refit all 2,000 context events. Target outcomes were not used for bandwidth selection.

## Full-range and equal-region Brier score

| Model | Full-range Brier | Equal-region mean Brier | Full-range log loss |
|---|---:|---:|---:|
| Cell 17 | 0.223674 | 0.214915 | 0.638460 |
| Cell 15 v5 | 0.223999 | 0.216366 | 0.639074 |
| Base 3 | 0.227601 | 0.220077 | 0.646891 |
| Kernel (context-only, h=100 keV) | 0.228093 | 0.228148 | 0.647889 |
| True CNP | 0.230041 | 0.236645 | 0.651945 |
| Base 1 | 0.230646 | 0.236991 | 0.653535 |

The full-range ranking is descriptive only. Regional errors, event support, and continuum roughness must be considered jointly; a single favorable peak bin or legacy p-value is not treated as evidence of shape recovery.

## Support and exclusions

| Region | Status | Context events | Target events | Expected 10-keV bins | Retained bins | Omitted bins |
|---|---|---:|---:|---:|---:|---:|
| full | supported | 2000 | 5074 | 250 | 212 | 38 |
| DEP | supported | 15 | 58 | 3 | 3 | 0 |
| Bi-214 | supported | 32 | 57 | 2 | 2 | 0 |
| continuum_1700_2000 | supported | 161 | 392 | 30 | 30 | 0 |
| SE | supported | 49 | 84 | 3 | 3 | 0 |
| continuum_2200_2400 | supported | 172 | 405 | 20 | 20 | 0 |
| FE | supported | 143 | 405 | 3 | 2 | 1 |
| sparse_tail | inconclusive | 1 | 2 | 30 | 0 | 30 |

Regions with fewer than 20 target events are marked inconclusive and excluded from the equal-region summary. The cached bin predictions are evaluated at bin centers, while event-level scores use interpolation. Neither reconstructs unsaved sub-grid structure. The caches contain no event IDs, class labels, independent threshold calibration, per-MC draws, or checkpoint hashes.

## Artifacts

- `tables/cache_reanalysis_event_metrics.csv`: event-level Brier score and log loss by region.
- `tables/cache_reanalysis_bin_metrics.csv`: bin-center MAE/RMSE with explicit valid-bin counts.
- `tables/cache_reanalysis_support.csv`: event and retained-bin support, including the sparse tail.
- `tables/cache_reanalysis_kernel_bandwidth_cv.csv`: context-only bandwidth selection scores.
- `tables/cache_reanalysis_metrics.json`: machine-readable aggregate results and roughness diagnostics.
- `figures/cache_full_range_historical.png` and `figures/cache_regions_historical.png`: figures with historical status in their labels.
