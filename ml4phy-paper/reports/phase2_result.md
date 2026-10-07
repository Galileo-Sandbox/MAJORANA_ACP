# Phase 2 acceptance-training-budget result

Status: complete. All 24 approved 3,000-step training jobs and all 240 new
evaluation cells completed without scientific failure. The analysis combines
those cells with 120 compatible Phase 1 cells at the original 18,866-event
budget. Every cell uses a 500-event context, the same ten overlapping context
draws, 50 MC passes with fixed dropout seed 10100, and the fixed 114,400-event
target. The 240-cell evaluation campaign took 856.10 seconds of wall time and
used at most 6,726.31 MiB of GPU memory.

## Main result

Values are mean ± SD across three training-seed means, in percentage points;
each seed mean averages the same ten context draws.

| Nominal training events | Method | Peak MAE | Continuum MAE | Sparse-tail MAE |
|---:|---|---:|---:|---:|
| 2,000 | CNP | 8.11 ± 0.33 | 4.47 ± 0.56 | 7.44 ± 0.36 |
| 2,000 | Attentive CNP | 8.04 ± 0.12 | 4.83 ± 0.86 | 10.44 ± 0.56 |
| 2,000 | Attentive CNP + PE | 13.77 ± 2.62 | 16.65 ± 0.40 | 30.88 ± 6.75 |
| 2,000 | Density-guided CNP (ours) | 9.40 ± 1.59 | 12.89 ± 1.05 | 10.80 ± 2.41 |
| 5,000 | CNP | 7.54 ± 0.09 | 5.21 ± 0.49 | 8.83 ± 1.48 |
| 5,000 | Attentive CNP | 7.73 ± 0.82 | 5.94 ± 0.68 | 11.91 ± 3.44 |
| 5,000 | Attentive CNP + PE | 11.39 ± 0.15 | 11.96 ± 1.06 | 30.05 ± 4.38 |
| 5,000 | Density-guided CNP (ours) | 5.29 ± 0.31 | 3.96 ± 0.37 | 19.39 ± 4.00 |
| 18,866 | CNP | 7.99 ± 0.35 | 4.59 ± 0.02 | 8.44 ± 1.06 |
| 18,866 | Attentive CNP | 8.25 ± 0.32 | 5.53 ± 0.70 | 7.81 ± 1.03 |
| 18,866 | Attentive CNP + PE | 6.00 ± 1.08 | 7.38 ± 0.86 | 23.25 ± 1.21 |
| 18,866 | Density-guided CNP (ours) | 3.56 ± 0.33 | 4.07 ± 0.11 | 14.70 ± 1.96 |

The prespecified peak metric equally averages DEP, the 1,620.74-keV Bi-212
feature, SE, and FE regional 5-keV-bin MAEs. The continuum metric equally
averages the 1,700--2,000 and 2,200--2,400-keV regions.

- Among the four neural architectures at 2,000 nominal events, the lowest peak MAE is Attentive CNP (8.04 pp); the lowest continuum MAE is CNP (4.47 pp).
- Among the four neural architectures at 5,000 nominal events, the lowest peak MAE is Density-guided CNP (ours) (5.29 pp); the lowest continuum MAE is Density-guided CNP (ours) (3.96 pp).
- Among the four neural architectures at 18,866 nominal events, the lowest peak MAE is Density-guided CNP (ours) (3.56 pp); the lowest continuum MAE is Density-guided CNP (ours) (4.07 pp).

The full-precision tables retain regional RMSE, support, excluded-bin counts,
peak/sideband contrasts, Brier checks, and all unfavorable outcomes. The
simpler CNP variants are more robust at the 2,000-event budget: the
density-guided model degrades to 9.40 pp peak MAE and 12.89 pp continuum MAE.
The density-guided advantage appears at 5,000 and improves further at 18,866,
but one frozen nested pool ordering does not identify a precise transition
budget or establish robustness to alternative training-pool draws.

The sparse tail remains a separate diagnostic: only 13 target bins meet the
four-event rule and 47 are excluded. The 2,000- and 5,000-event training
subsets contain no sample-eligible sparse-tail event, so no broad all-region
superiority claim is supported.

## Data-efficiency boundary

The nominal acceptance-training budgets retain 1,895, 4,984, and 18,836
sample-eligible events, respectively. The smaller pools are outcome-blind
nested prefixes shared by all methods. For Density-guided CNP, the density
pool shrinks with the nominal pool and does not reconstruct the full pool.
Training uses a fixed 3,000-step schedule with repeated sampling, so this is a
fixed-compute acceptance-stage study rather than a per-method optimum study.

The classifier remains fixed and was trained on 18,866 events selected from
377,330 candidates. Therefore this study can support only acceptance-model
training-data efficiency conditional on the pretrained classifier; it cannot
support a lower total end-to-end training-data claim. The extra 500 context
events and the evaluation, calibration, and development costs remain
separately disclosed in the data-budget ledger.

## Statistical and provenance boundary

The ten contexts overlap and all cells share the same finite, noisy target;
they are not independent datasets. Training-seed SD and within-seed context
variation are kept separate in the summary table. Brier is a secondary check,
and the earlier dropout-coverage diagnostic remains uncalibrated and unsuitable
as a positive headline result. This is a prospectively specified follow-up on
historically exposed data, not an untouched test.

Dense GP was not run in Phase 2. Its separately proposed budget extension still
requires explicit approval; sparse GP and new uncertainty methods remain out
of scope.
