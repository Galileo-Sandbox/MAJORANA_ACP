# Phase 3 protocol and resource gate

Status: frozen after explicit author approval; no Phase 3 fit or neural
training was started by this step. Source commit: `f41fe02b55414458a9d3f98f6a7856669501324a`.

## Reuse and required work

The original Phase 2 5k subset, its 12 neural checkpoints, and 120 matching
evaluation cells are reusable as the first of three 5k subset realizations.
The original 2k and 18.9k results remain unchanged. Phase 3 adds exactly 36
training jobs and 360 neural evaluation cells: two new 5k orderings plus the
10k prefix of the original ordering.

The dense-GP pilot is reusable for timing but not as a campaign cell: its
explicit optimizer seed 31000 differs from the frozen campaign seed rule,
which gives 31400 for the n=2,000, context-seed-100 cell. The required matrix
therefore remains 80 development fits and 40 final fits. Its existing
scaling-aware estimate is 8,662.74 seconds (2.41 hours), below the separately
approved four-hour hard limit.

## Frozen neural subsets

New orderings retain the Phase 2 outcome-blind SHA-256 priority algorithm but
encode the approved numeric ordering seed in a distinct namespace. The 10k
subset reuses the original Phase 2 row ordering byte-for-byte; the original 2k
and 5k prefixes were verified and were not regenerated.

| Subset | Nominal events | Sampling-eligible | Excluded | Kept 10-keV bins |
|---|---:|---:|---:|---:|
| original_n10000 | 10,000 | 9,980 | 20 | 209 |
| seed20260910_n5000 | 5,000 | 4,981 | 19 | 206 |
| seed20260911_n5000 | 5,000 | 4,980 | 20 | 203 |

All three new subsets have zero identity overlap with the fixed 114,400-event
final target and the 20,000-event final context reservoir. Each density buffer
must use its exact matching nominal subset. No target region is removed when a
training subset lacks support.

The three 5k subsets have a union of 11,358 unique events and a
three-way intersection of 344. Pairwise overlaps and exact
hashes are exported in the accompanying tables and manifest. These are subset
draws from one finite parent pool, not independent datasets.

## Neural resource gate

Phase 2 measurements imply approximately 82.9
minutes before allowance. Charging a 25% margin gives
103.6 minutes, below the independent two-hour neural
cap. It charges each new 5k realization at the measured Phase 2 5k cost and
the 10k slice at 1.5 times that cost, consistent with the observed fixed-step
training and density-inference scaling. This is not a runtime guarantee. The
campaign runner must stop at 7,200 seconds,
preserve partial results, and never shrink the matrix.

The fixed classifier still used 18,866 selected events from 377,330 candidates.
This campaign therefore tests acceptance-model training-data efficiency only,
conditional on that classifier; it is not an end-to-end low-data experiment.
