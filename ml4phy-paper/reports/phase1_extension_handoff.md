# Phase 1 extension handoff

Status: Phase 1 is complete for the four neural architectures and both Gaussian-kernel information budgets. The dense Gaussian-process campaign was stopped by its frozen two-hour approval gate. Phase 2 has not started.

All results are prospectively specified follow-up analyses on historically exposed data, not an untouched test. The fixed classifier, threshold, target identities, original endpoints, and commit `48a59d0b2d93a6c76d2001522abcb98ac74432fb` remain preserved.

## Reused evidence

- The fixed classifier and threshold, all original target roles, 60 compatible neural cells at context size 2,000, and saved event predictions for the original local-shape export were reused.
- The archived Gaussian-kernel estimator and original development bandwidth choices were reproduced exactly before extending the context-size matrix.
- No classifier retraining, new dataset, optional gate control, or Phase 2 training was performed.

## New execution

- Six matched baseline models were trained: CNP and Attentive CNP, three 3,000-step seeds each. All six completed without warnings.
- The 480-cell neural matrix is complete: 60 archived cells, one reused pilot cell, and 419 new evaluations. Twenty-five cells whose provenance recorded a transient dirty worktree were rerun cleanly; arrays agreed to absolute tolerance 1e-12.
- Eighty Gaussian-kernel cells were evaluated across context-only and pooled-data budgets. The pooled control adds all 18,866 acceptance-input events to each context.
- One full 2,000-context dense Bernoulli-GP timing pilot completed. The scaling-aware campaign estimate was 2.41 hours, exceeding the frozen two-hour gate, so no complete GP comparison was launched.
- The nested MC diagnostic extended Attentive CNP + PE to 800 passes when its trigger fired. Material stream disagreement and grid dependence remained, so a smoother-curve claim is rejected.

## Completed method comparison

At context size 500:

| Method | Peak MAE (pp) | Continuum MAE (pp) | Acceptance pretraining |
|---|---:|---:|---:|
| CNP | 7.993 | 4.585 | 18,866 |
| Attentive CNP | 8.245 | 5.533 | 18,866 |
| Attentive CNP + PE | 5.996 | 7.380 | 18,866 |
| Density-guided CNP (ours) | 3.557 | 4.072 | 18,866 |
| Gaussian kernel regression (context only) | 9.612 | 6.467 | 0 |
| Gaussian kernel regression (pooled data + context) | 7.768 | 4.056 | 18,866 |

At context size 2,000:

| Method | Peak MAE (pp) | Continuum MAE (pp) | Acceptance pretraining |
|---|---:|---:|---:|
| CNP | 7.993 | 4.592 | 18,866 |
| Attentive CNP | 8.257 | 5.530 | 18,866 |
| Attentive CNP + PE | 6.057 | 7.383 | 18,866 |
| Density-guided CNP (ours) | 3.480 | 4.082 | 18,866 |
| Gaussian kernel regression (context only) | 7.787 | 5.138 | 0 |
| Gaussian kernel regression (pooled data + context) | 7.828 | 4.163 | 18,866 |

Neural uncertainty columns separate the SD across three training-seed means from the mean within-seed SD across ten overlapping contexts. Kernel rows have context SD only. These contexts share targets and overlap; they are sensitivity replicates, not independent datasets.

## Supported claims and boundaries

- Among completed methods, Density-guided CNP has the lowest prespecified peak-region MAE at both headline context sizes. At n=500 it is 3.557 percentage points, compared with 5.996--9.612 for the other completed estimators.
- Its continuum MAE is 4.072 percentage points at n=500. The pooled-data kernel control is numerically similar at 4.056 but consumes the additional 18,866-event pool; this is not evidence that either method wins under equal total information.
- Neural errors change little from 250 to 2,000 context events. The result supports retained performance with a small context conditional on disclosed pretraining, not a strong context-scaling improvement and not total-data efficiency.
- The context-only kernel improves with more context but remains worse on the peak endpoint. The pooled kernel's strong sparse-tail result is retained and reported rather than suppressed.
- Brier remains secondary evidence. The local-error and fixed contrast analyses support localized reconstruction; the MC diagnostic does not support a smoother-curve claim.
- Dropout intervals cover only 0--3.8% of eligible empirical bins in the reported diagnostic and are not calibrated confidence intervals.

## Unresolved items

- No final Gaussian-process row exists because the agreed cost gate fired. A separately labeled sparse variational Bernoulli-GP protocol requires approval and resource testing.
- Common uncertainty coverage across dropout, kernel bootstrap, GP posterior, and finite-reference uncertainty remains undefined and was not invented after inspection.
- Acceptance-training-size efficiency remains untested. The exact Phase 2 slice and measured projection are in `phase2_approval_request.md`.
- Recovered seed-0 provenance limitations remain as recorded in the data-budget report.
