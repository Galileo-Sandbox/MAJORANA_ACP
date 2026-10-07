# Phase 2 acceptance-model training result

Status: all 24 approved jobs completed. The summed job wall time was 1791.91 seconds (29.87 minutes), below the pre-execution 35.33-minute projection.

| Budget | Method | Seed | Final loss | Wall time (s) |
|---:|---|---:|---:|---:|
| 2,000 | CNP | 0 | 0.447830 | 58.01 |
| 2,000 | CNP | 1 | 0.442674 | 59.04 |
| 2,000 | CNP | 2 | 0.491604 | 58.13 |
| 2,000 | Attentive CNP | 0 | 0.447466 | 59.10 |
| 2,000 | Attentive CNP | 1 | 0.443047 | 60.07 |
| 2,000 | Attentive CNP | 2 | 0.491074 | 61.29 |
| 2,000 | Attentive CNP + PE | 0 | 0.223511 | 76.00 |
| 2,000 | Attentive CNP + PE | 1 | 0.225213 | 74.06 |
| 2,000 | Attentive CNP + PE | 2 | 0.229243 | 75.02 |
| 2,000 | Density-guided CNP (ours) | 0 | 0.413811 | 81.17 |
| 2,000 | Density-guided CNP (ours) | 1 | 0.402306 | 80.65 |
| 2,000 | Density-guided CNP (ours) | 2 | 0.455560 | 80.65 |
| 5,000 | CNP | 0 | 0.452379 | 70.46 |
| 5,000 | CNP | 1 | 0.427950 | 68.04 |
| 5,000 | CNP | 2 | 0.488293 | 68.57 |
| 5,000 | Attentive CNP | 0 | 0.451640 | 71.37 |
| 5,000 | Attentive CNP | 1 | 0.428594 | 70.13 |
| 5,000 | Attentive CNP | 2 | 0.489244 | 70.72 |
| 5,000 | Attentive CNP + PE | 0 | 0.333717 | 86.81 |
| 5,000 | Attentive CNP + PE | 1 | 0.325419 | 87.73 |
| 5,000 | Attentive CNP + PE | 2 | 0.370719 | 84.07 |
| 5,000 | Density-guided CNP (ours) | 0 | 0.447071 | 97.26 |
| 5,000 | Density-guided CNP (ours) | 1 | 0.426461 | 96.61 |
| 5,000 | Density-guided CNP (ours) | 2 | 0.484299 | 96.96 |

The 2,000-event nominal pool retained 1,895 sampling-eligible events in 159 bins. The 5,000-event nominal pool retained 4,984 events in 205 bins. All jobs used the fixed 3,000-step schedule, sampled with replacement, and produced complete checkpoint histories through step 2,999. No training log contained a warning or non-finite value.

Every resolved configuration points to its matching server-only subset HDF5. For Density-guided CNP, this also forces the reconstructed density buffer to contain exactly 2,000 or 5,000 nominal events; no smaller-budget run references the original full pool. Checkpoints and event-level subset files remain server-only.

One initial Density-guided CNP budget-2,000 seed-0 attempt stopped before optimization because the default execution sandbox hid the CUDA device. Its directory and record are preserved. The uniquely named attempt 2 completed normally and is the only version admitted to evaluation.

This is fixed-compute training-budget evidence on historically exposed data. Final performance requires the prespecified n=500 evaluation and must retain unfavorable outcomes.
