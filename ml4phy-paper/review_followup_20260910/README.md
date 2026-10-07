# Mechanism-control follow-up

This additive directory contains the prespecified 2026-09-10 mechanism-control
campaign. It leaves the historical model package and earlier paper exports
unchanged.

## Workflow

1. `PLAN.md` records the authorized scientific scope and stopping rules.
2. `configs/protocol_v1.json` freezes modes, seeds, inputs, expected cells, and
   resource limits.
3. `audit_and_freeze.py` validates historical sources, subset identities, and
   full-model parity before training.
4. `control_models.py` implements the local mode-aware controls.
5. `train_control.py` and `run_training_campaign.py` execute 12 training jobs.
6. `evaluate_control.py` and `run_evaluation_campaign.py` execute the 120 new
   evaluation cells.
7. `export_existing.py`, `aggregate_mechanism.py`, and
   `validate_outputs.py` create and independently verify portable results.

## Outputs

- `reports/` contains the pretraining audit, measurement interpretation,
  mechanism result, corrections, and final handoff.
- `tables/` contains run registries, compressed per-bin predictions, regional
  diagnostics, paired contrasts, and learned mechanism maps.
- `figures/` contains mechanism and passing-count diagnostics.
- `manifests/provenance_v1.json` records hashes, versions, failures, corrected
  evaluations, runtime, and completed inventories.
- `tests/` covers control invariants, checkpoint restoration, aggregation, and
  saved-artifact reconstruction.

Checkpoint and event-level prediction directories are ignored. The compact
tables retain source hashes. The campaign supports a decoder-gating mechanism
result on one original 5k subset; it does not establish universal superiority,
calibrated uncertainty, or end-to-end 5k training.
