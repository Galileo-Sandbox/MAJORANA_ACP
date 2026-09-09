# Phase 3 data-efficiency confirmation

Status: complete. The dense Bernoulli-GP campaign completed 120/120 cells, the neural campaign completed 36/36 training jobs and 360/360 evaluations, and no scientific run failed. Dense GP used 1073.98 seconds (17.90 minutes) under its four-hour cap. Neural training plus evaluation used 2334.56 seconds (38.91 minutes) under the separate two-hour cap.

## Nested acceptance-training budget result

Values below are peak/continuum MAE in percentage points, averaged hierarchically over three initialization seeds and the same ten overlapping 500-event contexts. The 2k, 5k, and 10k rows use nested prefixes of the original outcome-blind ordering; 18.9k is the exact 18,866-event full pool.

| Budget | CNP | Attentive CNP | Attentive CNP + PE | Density-guided CNP (ours) |
|---|---:|---:|---:|---:|
| 2k (2,000) | 8.11/4.47 | 8.04/4.83 | 13.77/16.65 | 9.40/12.89 |
| 5k (5,000) | 7.54/5.21 | 7.73/5.94 | 11.39/11.96 | 5.29/3.96 |
| 10k (10,000) | 8.14/4.91 | 8.15/5.21 | 7.06/9.12 | 5.84/4.01 |
| 18.9k (18,866) | 7.99/4.59 | 8.25/5.53 | 6.00/7.38 | 3.56/4.07 |

The 2k degradation remains visible and no method is uniformly best across every region and budget. Sparse-tail results are retained in the CSV tables; the small training subsets have no sampling-eligible sparse-tail events, so peak-region findings must not be generalized to the tail.

The original-ordering curve is not monotonic: Density-guided CNP changes from 5.29/3.96 pp at 5k to 5.84/4.01 pp at 10k before reaching 3.56/4.07 pp at 18.9k. This supports direct measured-budget comparisons, not interpolation to an exact sample threshold.

## Five-thousand-event subset robustness

For Density-guided CNP, peak/continuum MAE across the original, seed-20260910, and seed-20260911 orderings is:

- Original: 5.29/3.96 pp.
- Seed 20260910: 5.11/4.56 pp.
- Seed 20260911: 5.32/3.95 pp.

All three 5k peak errors are below the best full-budget non-density-guided neural peak error (6.00 pp), while the continuum errors span 3.95--4.56 pp. This supports robust competitive local peak reconstruction at the acceptance-model stage, not superiority in every region. The 5k Density-guided CNP sparse-tail errors span 18.96--21.12 pp, compared with 2.99 pp for the pooled-data kernel control.

These are three overlapping random subsets of one finite parent pool, not independent datasets. The tables separate initialization-seed SD, mean within-seed context SD, and between-subset SD. They do not identify an exact minimum sample requirement or justify interpolation between budgets.

## Dense GP and method boundary

Development-only selection chose ConstantKernel × Matern(ν=1.5): mean global development Brier was 0.231406, versus 0.231708 for RBF. At 500 context events the selected context-only GP has 9.97 pp peak MAE and 6.74 pp continuum MAE. It uses no acceptance pretraining, whereas the neural models are conditional on acceptance-model pretraining; this is not an equal-total-information comparison.

Three fits reported a length-scale-at-upper-bound convergence warning (one development and two final cells); no bound was changed after seeing results. Scheduler recovery also left two bitwise-identical redundant completed outputs and two interrupted attempts with no scientific result. The attempt audit excludes all four from the prespecified 120-cell matrix and preserves their provenance.

The compact method table includes the full-pool neural models, context-only and pooled-data kernel controls, and selected GP. Brier remains secondary. No dropout interval is presented as calibrated, and no MC smoothness claim is made.

## Claim boundary

The result tests acceptance-model training-data efficiency conditional on the classifier pretrained with 18,866 selected events from 377,330 candidates. It does not support end-to-end training on 5k events. All targets and contexts were historically exposed; this is a prospectively specified follow-up analysis, not an untouched test. The 1,620.74-keV structure is named by energy because its isotope identity remains unresolved in the audited sources.

The pre-training frozen support table is retained byte-for-byte for execution provenance, but one of its display labels selected an isotope name before the identity discrepancy was resolved. `phase3_training_subset_region_support_v2.csv` supersedes that label only; event counts, windows, subset identities, and the frozen input hashes are unchanged.
