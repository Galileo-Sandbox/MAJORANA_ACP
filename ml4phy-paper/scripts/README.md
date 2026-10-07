# Paper scripts

This directory contains the frozen paper workflow. Scripts are grouped by
their role in the evidence chain.

## Protocol and audit

- `build_protocol.py`, `build_extension_protocol.py`,
  `freeze_phase2_protocol.py`, and `freeze_phase3_protocol.py` create immutable
  role and subset records.
- `audit_data_budget.py` reconstructs unique counts, overlaps, preprocessing,
  sampling exposure, and classifier provenance.
- `run_training_pilot.py` and `summarize_training_pilot.py` enforce resource
  gates before a campaign.

## Scientific execution

- `run_extension_neural_campaign.py` and `evaluate_extension_neural.py` cover
  the context-size neural study.
- `run_extension_kernel.py` evaluates the context-only and pooled-data kernel
  estimators.
- `run_phase2_*` and `run_phase3_*` execute the frozen training-budget slices.
- `run_dense_gp_cell.py` and `run_dense_gp_campaign.py` execute the event-level
  Bernoulli-GP protocol.
- `run_mc_smoothness.py` performs the separately bounded MC-noise diagnostic.

## Aggregation and export

- `aggregate_*` scripts summarize completed campaigns without selecting new
  settings.
- `export_paper_presentation.py` produces paper-facing curves, regional
  agreement diagnostics, and continuum residual exports from saved arrays.
- `export_paper_coverage.py` produces regional reference-band coverage.
- Plot and handoff scripts write the tracked files in `../tables/`,
  `../figures/`, `../reports/`, and `../manifests/`.

Many execution scripts can start training or inference. Read the corresponding
protocol manifest and use `--help` before running them. The portable artifact
verifier in `../artifact/` never launches scientific computation.
