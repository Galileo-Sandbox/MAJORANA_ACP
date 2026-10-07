# Development candidate decision

Status: candidate frozen from the development protocol. No new model training contributed to this decision.

## Decision

Cell 17 is the frozen density-guided candidate for the next conditional experiment. It had the lowest mean global and equal-supported-region event-level Brier score under every one of the ten paired context draws while the dropout seed was held fixed at 10100.

| Model | Global Brier, mean | Equal-region Brier, mean | Training seeds | Context draws |
|---|---:|---:|---:|---:|
| Cell 15 v9 | 0.225955238 | 0.234857416 | 1 | 10 |
| Cell 15 v5 | 0.225810479 | 0.234649581 | 1 | 10 |
| Cell 17 | 0.225620633 | 0.234185498 | 1 | 10 |

Relative to Cell 15 v5, Cell 17's mean paired Brier difference was -0.000189846 globally and -0.000464083 for the equal-region summary; negative favors Cell 17. Both comparisons favored Cell 17 in 10/10 paired context draws. The result satisfies the prespecified preference rule because the learnable Cell 17 variant shows a consistent development benefit over the simpler fixed-floor alternatives.

## Scope and limitations

- The comparison uses one recovered training seed per model. Context-draw dispersion is reported separately and cannot substitute for training-seed replication.
- The ten context draws overlap, share the same 2,074-event development target, and use one fixed dropout seed. They are paired sensitivity checks, not ten independent datasets.
- DEP has only 17 development-target events and the sparse tail has none. Those regions remain inconclusive and were not used to make a favorable exception.
- Regional results are mixed: candidate selection uses the two frozen aggregate Brier criteria, while the peak-contrast and continuum-roughness tables preserve local trade-offs.
- A second campaign that varied context and dropout seeds together exists server-side but is excluded from the context-only dispersion reported here.

## Next gate

Run a fixed-context dropout-estimator sensitivity check and a 0.5-keV grid convergence check for Cell 17. Evaluate recovered CNP controls and the context-only kernel baseline under the same development protocol. If Cell 17 remains competitive, the gap decision authorizes one controlled M2 training pilot at training seed 0; it does not authorize a full training grid.
