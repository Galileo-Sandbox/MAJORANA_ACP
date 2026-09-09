# Pre-Training Gap Decision

Decision date: 2026-09-08  
Status: **training not yet authorized; inference-first plan approved**

This decision follows the server artifact recovery and cache-only reanalysis. No new CNP training or fresh CNP inference preceded it.

## Decision table

| Desired paper statement or question | Evidence recovered | Provenance quality | Remaining ambiguity | Cheapest valid resolution | Decision |
|---|---|---|---|---|---|
| Existing density-guided variants can improve historical predictive error over conventional CNPs | Five shared-array caches; Cell 17 has full-range Brier 0.223674 and Cell 15 v5 0.223999 versus True CNP 0.230041 on one 5,074-event target | Medium for array equality; low for independent evaluation because threshold and model selection used the same historical evaluation source | One context draw, interpolated 800-point curves, no IDs, no v9 cache | Retain as explicitly historical development evidence | **Reuse (E0 complete)** |
| The talk's v9 result is reproducible | v9 checkpoint/config/training pool and 141,474-row full-test classifier export recovered; counts match 2,000 + 139,474 | Medium; source linkage is strongly suggested by exact counts, not proven by an original manifest | Original context IDs, seed, curve, MC draws, and source commit absent | Re-run v9 with explicit IDs, fixed threshold, and recorded randomness | **Inference (E1)** |
| v9, v5, or Cell 17 should be the paper candidate | All three checkpoints and exact architecture metadata recovered | Medium; each is seed 0 and historically selected on inspected data | They have not been compared under a common provenance-safe split and threshold | Paired development inference with identical contexts and queries | **Inference (E1); no retraining** |
| The selected package improves on mean CNP, attention-only CNP, and ungated PE | True CNP and Base 1/2/3 weights recovered | Low as a controlled ablation: sampler, trial range, context range, dropout, attention dimensions, and PE differ | Existing names containing `matched` do not establish the proposed controlled ladder | First evaluate recovered models as secondary evidence; if the central claim survives E1, train only missing M0/M1/M2 controls and missing candidate seeds | **Conditional training (E2)** |
| The decoder frequency gate itself causes the gain | Nearby historical cells exist, but none is an otherwise identical constant/full-band mask control | Low | Density attention/bias, direct contrast, and decoder gating change together | A matched E3 control only if the manuscript makes a narrow gate-attribution claim | **Defer E3** |
| Direct density-contrast injection is not the sole cause | No otherwise matched no-injection checkpoint was found | Low | Removing the input changes architecture and requires retraining | Train E4 only if a narrow frequency-gating attribution remains necessary after E2 | **Defer E4** |
| The candidate improves on kernel regression | A context-only kernel can be fit from saved context outcomes; 100 keV wins five-fold context-only CV in the historical cache | Medium for the cache, not final protocol | Final development/final context roles are not yet frozen | Refit/tune only on frozen development/context information; evaluate on paired target IDs | **Inference-only baseline (E5)** |
| Results are robust to context selection | Full event identities and all priority weights are present | No direct evidence yet | Historical caches contain one draw only | Ten paired context draws, seeds 100-109, fixed targets; keep dropout seed separate | **Inference (E6)** |
| Results are robust to context size | Same as above | No direct evidence yet | Not required for the core claim unless context sensitivity appears | Nested 500/1,000/2,000 contexts only if E6 is unstable | **Conditional inference (E7)** |
| The fixed cut is independently selected | Labels and identities are available; legacy threshold 0.540643572807312 is known to use the full 7,074-event evaluation export | Historical threshold is invalid for an independent paper evaluation | A new calibration partition and frozen threshold are needed | Freeze a calibration subset before model evaluation and pass the value explicitly | **Implement before inference** |
| The evaluation is untouched | The 7,074-event subset and full-test data were previously inspected, including the talk | High evidence that “untouched” would be inaccurate | No new MAJORANA dataset is required or justified | Use a prospectively frozen protocol and disclose historical exposure | **Do not claim untouched test performance** |

## Frozen role proposal to implement

The evaluator will use composite identities `(run_number, detector, id, tp0)` and persist only hashes/counts in portable manifests.

1. Remove the 7,074 historical-development identities from the 141,474-row full-test export, leaving 134,400 rows.
2. Deterministically split the 7,074 historical-development rows into 2,000 threshold-calibration rows, a 3,000-row development context reservoir, and a fixed 2,074-row development target. The calibration selection is label-stratified; all later splits are identity-stable and outcome-blind.
3. Select Youden-J on the 2,000 calibration rows only and freeze the resulting numeric threshold and identity-manifest hash before model inference.
4. Deterministically split the other 134,400 rows into a 20,000-row final context reservoir and a fixed 114,400-row target. These data remain historically exposed, so results will be called a prospectively frozen repeated-split evaluation, not an untouched test.
5. Use paired context-draw seeds 100-109 for all models. Keep context-selection, MC-dropout, and training seeds separate. Use 50 MC passes after a smaller timing pilot and record the per-pass context cap.

The initial paper evaluation range remains 500-3,000 keV. Regions, binning, sparse-support rules, Brier score, log loss, bin MAE/RMSE, peak/sideband contrasts, and fixed-grid roughness follow `EXPERIMENT_PLAN.md`. A region with fewer than 20 target events is reported as inconclusive and excluded from the equal-region aggregate.

## Authorized execution sequence

1. Reconstruct the compatible RESUM_FLEX source overlay at commit `edba6a294581fde6f905b330d477f2c1b42d6adb` under ignored `ml4phy-paper/local/`; do not alter the current sibling checkout or environment.
2. Implement and test the fixed-role manifest builder, evaluator, strict cache/protocol validation, unique-output runner, and compact exporter under `ml4phy-paper/`.
3. Run one v9 development inference timing pilot with reduced MC passes. Measure wall time and GPU memory; do not extrapolate cost before this measurement.
4. Run paired E1 development inference for v9, v5, and Cell 17. Freeze the candidate using joint peak/continuum predictive error and stability; prefer fixed v9 if learned Cell 17 has no consistent benefit.
5. Evaluate recovered True CNP and Base 1/2/3 under the same split as secondary, explicitly confounded historical comparisons. Run E5 and E6 without training.
6. If E1 supports the scoped density-guided claim, run a single M2 seed-0 training pilot first because ungated PE is the closest scientific control. Compare it with the recovered candidate seed-0 checkpoint. Stop or narrow the claim if the comparison is unfavorable.
7. Only after a favorable pilot, complete the genuinely missing controlled ladder: M0/M1/M2 seeds 0-2 and selected M3 seeds not already recoverable. The present maximum is 11 new 3,000-step jobs if v9 seed 0 is reusable. Recompute this count if another recovered checkpoint proves compatible.
8. Keep E3, E4, E7, and E8 deferred unless the preceding evidence makes their specific claim necessary.

## Stop conditions

- Do not train if the compatible source overlay cannot reproduce a recovered checkpoint load and deterministic inference.
- Do not train if event roles, fixed threshold, and output provenance are not validated.
- Stop after the M2 seed-0 pilot if the candidate does not improve the joint localized/continuum criteria; report the unfavorable comparison and narrow the paper.
- Do not launch the full 11-job ceiling as one batch. Add seeds/models only after the measured pilot and interim comparison justify them.
- Do not perform a kappa sweep, classifier retraining, unrelated datasets, synthetic benchmarks, or a broad hyperparameter search.

## Current evidence summary

The cache-only result favors Cell 17 by a small global margin over v5 on one historical draw, but it cannot select the final candidate. The server recovery makes v9 directly evaluable and removes any rationale for retraining it from scratch. The smallest next experiment is therefore one provenance-correct v9 inference pilot, not model training.
