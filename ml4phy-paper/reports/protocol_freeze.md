# Frozen Paper Evaluation Protocol

Freeze date: 2026-09-08  
Protocol manifest: `manifests/frozen_protocol_v1.json`

The protocol was frozen after artifact recovery and the pre-training gap decision, and before any paper-specific CNP inference or training.

## Data roles

| Role | Count | Identity SHA-256 | Use |
|---|---:|---|---|
| Threshold calibration | 2,000 | `9e6985c6138726fdbc2c05035c5daa3779826842e735985c7441e61a23cb1021` | Select the fixed classifier threshold only |
| Development context reservoir | 3,000 | `4a96d126e6dc38a738d5746267a655f936a4d58aeab4fa6a394b9b7769d93fbf` | Paired candidate/model-development contexts |
| Development target | 2,074 | `fbbaefc5ab880c8837ce8582c21bac0f5286dafc67775726edbbe3ea8bb3c6f3` | Candidate selection and development diagnostics |
| Final context reservoir | 20,000 | `2ea72e8682c10a6d09e7dfc0981525394c6f834d981aaca6f076e39aa0594347` | Paired final contexts |
| Fixed final target | 114,400 | `03853bd1c4a3f596b4e0614152b3a1f5306f4ea1a0485a4cb384c70b17faec73` | Prospectively frozen repeated-split evaluation |

Identities are the composite `(run_number, detector, id, tp0)`. The portable manifest contains only counts and order-independent identity hashes. HDF5 row indices remain in ignored `local/` storage.

The 7,074 historical-development identities are an exact subset of the 141,474-row full-test export. They were removed before creating the final context reservoir and target. All roles are disjoint within their evaluation stage.

## Threshold

The fixed threshold is `0.540643572807312`, selected by maximizing Youden-J on the 2,000 calibration events only. The calibration ROC AUC is 0.948692. The equality with the legacy threshold is a numerical coincidence supported by the recovered scores; unlike the legacy computation, no development or final target outcome enters the new threshold selection.

## Randomness and evaluation

- Split seed: 20260908.
- Primary context count: 2,000.
- Paired context seeds: 100-109.
- Dropout seed: `10000 + context_seed`, recorded separately.
- Initial MC passes after the timing pilot: 50.
- Per-pass context count must equal the requested context draw; the evaluator rejects a lower cap to prevent mixing dropout and context-subsampling variation.
- Event-level Brier score is primary. Log loss, 5-keV bin MAE/RMSE, peak/sideband contrast error, and 1-keV-grid continuum roughness are secondary.
- Regions with fewer than 20 target events are marked inconclusive. Bins with fewer than four target events are excluded and counted.

## Exposure limitation

The full-test data were inspected historically, including for the talk. The final result therefore must be described as a prospectively frozen repeated-split evaluation, not an untouched test. Model development, controlled-training decisions, and kernel bandwidth selection use the development roles before final evaluation.
