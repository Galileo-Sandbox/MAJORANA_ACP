# Mechanism controls and measurement interpretation

Status: approved on 2026-09-10. This directory implements the bounded lab-server
campaign authorized by the author on that date.

## Scientific purpose

The campaign tests whether energy-dependent density modulation improves the
joint reconstruction of sharp efficiency features and off-feature behavior
beyond learned energy-independent frequency and attention settings. It also
translates existing reference-band agreement into regional efficiency and
expected passing-event-count discrepancies. This is a prospectively specified
post-review extension on historically exposed data, not an untouched test.
Null or unfavorable ablations are valid outcomes.

## Frozen existing-artifact stage

Recover the 600 neural, 10 context-only kernel, 10 pooled-kernel, and 10 selected
Bernoulli-GP cells from the presentation/coverage manifests. Reproduce global
C1/C2/C3, 442 supported and 58 excluded bins, 101 target events in excluded
bins, and the frozen 114,400-event target. Export shared reference bins and a
compressed per-cell bin table from saved event-level means only. No inference,
MC sampling, interpolation, fitting, or tuning is permitted.

For every frozen region and cell, export observed and predicted passing counts,
signed efficiency differences, relative passing-count differences where
defined, event-weighted bin MAE, and reference-band C1/C2/C3. Keep supported-bin
and all-observed-event domains separate. Retain the exact historical narrow
cores, broader peak windows, two continuum windows, Overall, and sparse tail.
Use physical display labels while retaining historical machine IDs. Audit the
nominal, sampling-eligible, density-buffer, and regional training-pool counts.

## Frozen mechanism slice

Use only the original 5,000-event subset (4,984 sampling-eligible events and a
5,000-event density buffer), three initialization seeds, the ten frozen
500-event contexts, 3,000 training steps, and the fixed classifier/threshold.

| Mode | Decoder frequency gate | Attention bandwidth/temperature | Direct density input |
|---|---|---|---|
| `full_density` | original density-adaptive | original density-adaptive | real R |
| `global_gate` | learned global cutoff | original density-adaptive | real R |
| `global_attention` | original density-adaptive | learned global settings | real R |
| `global_both` | learned global cutoff | learned global settings | real R |
| `density_free_global` | learned global cutoff | learned global settings | constant zero |

Reuse the three `full_density` checkpoints and 30 matching evaluation cells.
Train the other four modes for seeds 0--2: 12 new jobs. Evaluate each on context
seeds 100--109: 120 new cells. The resulting mechanism inventory is 150 cells.

The global gate is
`lambda_global = 1 + 9 sigmoid(phi)`, initialized with
`phi = log(2/7)` so `lambda_global = 3`, followed by the unchanged
`w_l = sigmoid(5 (lambda_global-l))` for bands 0--9. The original trainable
`kappa_raw` tensor is explicitly repurposed as `phi`. Global attention feeds
`Z0=(0,0)` to the existing `pool_sfn_net` and `pool_tau_net`, retaining their
bounded learned outputs and all raw-query/key and Gaussian-penalty operations.
Only `density_free_global` replaces the direct decoder R feature by zero.

## Correctness and resource gates

Before training: establish deterministic and fixed-RNG stochastic full-mode
parity with a recovered 5k checkpoint; test mode-aware save/load, gradients,
optimizer participation, global-control invariance, density-free invariance,
shared initialization, task-schedule identity, hashes, data separation, and
the expected inventory. Stop before training if full parity fails.

One successful training job is the pilot and counts toward 12. Forecast the
remaining training plus 120 evaluations with at least 25% margin. The scientific
clock has a two-hour hard limit from pilot start, including retries. Use at most
four concurrent jobs after measuring GPU memory; do not change batch size,
steps, or scientific settings to meet the limit. One retry per technical failure
is permitted. Existing-artifact export has a separate 45-minute limit.

Scientific evaluation uses threshold `0.540643572807312`, initialization seeds
0--2, context seeds 100--109, context size 500, 114,400 targets, 50 dropout
passes, fixed dropout seed 10100, and the validated batching/transformation.

## Prespecified comparisons

- `full_density` versus `global_gate`: density-adaptive decoder gating.
- `full_density` versus `global_attention`: density-adaptive attention.
- The first four modes: two-by-two modulation-rule comparison with direct R.
- `global_both` versus `density_free_global`: direct R under global modulation.
- `full_density` versus `density_free_global`: complete density package versus
  learned global flexibility.

Primary endpoints are Peaks and two-window Continuum C2 jointly. Report all
individual peak cores, Overall, sparse tail, and C1/C3. Pair by initialization
and context, average contexts within seed, and keep context and seed variation
separate. Do not use naive 30-cell significance tests or invent uncertainty.

Export deterministic mechanism maps on the fixed 1-keV 500--3000-keV grid.
These maps describe learned controls and are not predictive smoothness evidence.

## Scope exclusions and claim limits

No new dataset, classifier retraining, threshold selection, training budget or
ordering, baseline retraining, GP/KDE fitting, sparse GP, uncertainty method,
high-MC study, hyperparameter sweep, or broader factorial campaign is approved.
Do not tune away the 2k failure, replace Figure 1, or overwrite existing exports.
Do not claim calibrated dropout intervals, proven overfitting, MC-corrected
smoothness, an exact minimum data requirement, universal superiority, signal
efficiency, cross-section bias, a new physics-search region, or cross-experiment
transfer. The fixed classifier itself used 18,866 training events.

Keep checkpoints and event-level outputs server-only. Track less than 5 MiB
where possible and never exceed 20 MiB. Commit the frozen protocol and tested
implementation before training. At completion, validate, commit only intended
lightweight artifacts, fetch before push, and verify the live remote head.
