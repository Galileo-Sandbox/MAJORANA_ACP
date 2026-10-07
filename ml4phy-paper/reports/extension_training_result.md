# Matched baseline training result

Status: all six prospectively justified jobs completed successfully on 2026-09-09.

| Method | Seed | Final training loss | Wall time (s) | Logged warnings |
|---|---:|---:|---:|---:|
| CNP | 0 | 0.434477 | 70.11 | 0 |
| CNP | 1 | 0.420394 | 68.88 | 0 |
| CNP | 2 | 0.483689 | 68.79 | 0 |
| Attentive CNP | 0 | 0.432943 | 72.00 | 0 |
| Attentive CNP | 1 | 0.421398 | 73.47 | 0 |
| Attentive CNP | 2 | 0.483504 | 71.46 | 0 |

The six jobs took 424.70 seconds of summed wall time. Each used the
same 18,866-event input pool, the same 18,836 sampling-eligible events, 3,000
steps, batch size 16, variable 640--1,024-event trials, context sizes 128--512,
four training loss samples, Adam at 1e-3, dropout 0.2, and final-checkpoint
selection. Sampling remained with replacement. No run failed and no warning or
non-finite loss was recorded.

CNP has 116,482 parameters and uses mean pooling without positional encoding.
Attentive CNP has 149,250 parameters and uses deterministic cross-attention
without positional encoding; it is not a latent ANP. For comparison, the
matched Attentive CNP + PE model has 151,682 parameters and Density-guided CNP
(ours) has 130,693 parameters.

Checkpoints and event-level training pools remain in ignored server-only run
directories. `configs/extension_models_v1.json` records their paths and hashes
alongside the six reusable matched core checkpoints. This training result does
not by itself support a performance claim; evaluation follows the frozen
nested-context protocol. Phase 2 remains unapproved and was not started.
