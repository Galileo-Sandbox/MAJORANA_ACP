# Phase 2 approval request: acceptance-training-size slice

Status: awaiting explicit approval. No Phase 2 subset was materialized and no Phase 2 model was trained.

## Proposed frozen slice

- Keep the classifier, threshold, final targets, 3,000-step schedule, optimizer, and context size 500 fixed.
- Use acceptance-training budgets 2,000, 5,000, and 18,866. Reuse the completed 18,866-budget models when all protocol fields match.
- Before training, create one identity-hash-based, outcome-blind ordering of the 18,866-event input pool. Use nested 2,000- and 5,000-event prefixes for every architecture and seed.
- Recompute effective retained counts after the minimum-four-events-per-bin sampler filter. Do not remove target regions when a smaller pool lacks support.
- For Density-guided CNP, construct the density buffer only from the matching prefix and verify its identity hash; do not reconstruct the full pool.
- Train four architectures with three initialization seeds at each of the two smaller budgets: 24 new jobs. Evaluate the same ten existing context draws at n=500 for 240 full-target cells.
- Treat the result as a fixed-compute data-budget study, not an exhaustive per-method optimum. Preserve unfavorable outcomes.

## Measured cost basis

| Architecture | Mean measured 3,000-step training time (s) |
|---|---:|
| CNP | 69.257 |
| Attentive CNP | 72.310 |
| Attentive CNP + PE | 85.911 |
| Density-guided CNP (ours) | 125.824 |

The fixed-step training projection is **2119.82 seconds (35.33 minutes)** for 24 jobs. It assumes training time is approximately pool-size independent because the step and sampled trial-size schedules stay fixed; the actual times will still be recorded.

The completed Phase 1 neural campaign used 2,169.45 wall-seconds for 419 new cells. Scaling that measured campaign rate to 240 Phase 2 cells projects **1242.64 seconds (20.71 minutes)** for evaluation. Training plus evaluation is **3362.46 seconds (56.04 minutes)** sequentially before aggregation, or **4203.08 seconds (70.05 minutes)** with a 25% operational allowance. The slow-path measured GPU allocation was 6,726 MiB.

## Approval decision requested

Approve or reject the 24-job fixed slice above. Approval would authorize subset freezing, a short input-validation dry run, 24 sequential training jobs, and the matching 240-cell evaluation only. It would not authorize a full training-size by context-size factorial grid, classifier retraining, a gate control, or a GP campaign.
