# Phase 2 protocol freeze

Status: frozen after explicit approval on 2026-09-09; no training was started by this freeze step.

The two laptop-only planning documents were not present in the server checkout at commit `9210ede4e9978706019c2fbb64e8cd05dc05afd2`. The approved user instruction, Phase 1 handoff, existing frozen protocol, and preserved repository plans therefore govern this execution.

## Outcome-blind nested training pools

The ordering is determined only from the four event identity fields (`run_number`, `detector`, `id`, and `tp0`). Each identity receives SHA-256 priority under the fixed namespace `ml4ps-phase2-training-order-v1`; labels, classifier scores, energies, and final outcomes do not enter the ordering. The 2,000-event pool is the exact prefix of the 5,000-event pool. The 18,866-event budget reuses the completed Phase 1 models.

| Nominal budget | Sampling-eligible events | Excluded events | Kept 10-keV bins |
|---:|---:|---:|---:|
| 2,000 | 1,895 | 105 | 159 |
| 5,000 | 4,984 | 16 | 205 |

Minimum-four-event filtering affects only which input events can be sampled during training. It does not change any final target identity or evaluation-region definition. Training remains with replacement for 3,000 steps, so this is a fixed-compute budget study.

For Density-guided CNP, the configured `train_predictions_path` will point to the matching subset HDF5. Its density pool is therefore exactly the nominal prefix (2,000 or 5,000 identities), including events excluded by the minimum-bin sampler. The runner must verify this path and identity hash before training, and evaluation must rebuild from the same resolved configuration.

The server-only subset HDF5 files and row ordering remain under `ml4phy-paper/local/phase2/protocol-v1/`. Portable ledgers disclose counts, region support, overlaps, and hashes without exporting identities or event-level predictions.
