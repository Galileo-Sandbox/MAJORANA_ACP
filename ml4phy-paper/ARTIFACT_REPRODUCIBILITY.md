# Artifact reproducibility guide

This guide accompanies
[Density-Guided Conditional Neural Processes for Detector Efficiency Estimation](https://arxiv.org/abs/2609.13593).

This guide separates claims that can be checked from a Git clone from work that
requires large external data or historical third-party source. It avoids
calling a result reproducible when a required input is absent.

## Reproduction levels

### Level 1: verify the published portable evidence

This level requires only the repository. It verifies release anchors, every
portable output recorded by the presentation, coverage, and mechanism-control
manifests, the expected experiment inventories, and the independently
recomputed mechanism summaries.

```bash
git switch main
uv sync --frozen --dev
.venv/bin/python ml4phy-paper/artifact/verify_artifact.py
PYTHONPATH=ml4phy-paper/scripts:ml4phy-paper/review_followup_20260910:. \
  .venv/bin/python -m pytest \
  ml4phy-paper/tests \
  ml4phy-paper/review_followup_20260910/tests \
  ml4phy-paper/artifact/tests -q
PYTHONPATH=ml4phy-paper/scripts:ml4phy-paper/review_followup_20260910:. \
  .venv/bin/python ml4phy-paper/review_followup_20260910/validate_outputs.py
```

The first command is standard-library-only. The other commands use the locked
project environment. `validate_outputs.py` recomputes the mechanism-region
scores from the committed compressed per-cell bin predictions rather than
trusting the summary CSVs.

### Level 2: regenerate portable exports from saved predictions

This optional author-side audit requires the frozen role archives, training
subsets, and event-level predictions at the logical paths recorded in the
manifests. Checkpoints are not required to regenerate exports from saved
predictions. After restoring the local files, run:

```bash
.venv/bin/python ml4phy-paper/artifact/verify_artifact.py --server
PYTHONPATH=ml4phy-paper/scripts:ml4phy-paper/review_followup_20260910:. \
  .venv/bin/python ml4phy-paper/review_followup_20260910/export_existing.py \
  --repo .
PYTHONPATH=ml4phy-paper/scripts:ml4phy-paper/review_followup_20260910:. \
  .venv/bin/python ml4phy-paper/review_followup_20260910/aggregate_mechanism.py
```

These exporters consume saved predictions. They do not train models or create
new Monte Carlo predictions. The source registries identify interrupted or
invalid duplicate attempts so that only the prespecified successful cells are
used.

These local files are not committed because the public artifact already
contains the derived bin-level evidence needed to check the paper's numerical
claims. Their hashes remain available for author-side provenance verification.
Publishing checkpoints or event-level predictions is not a requirement of this
artifact.

### Level 3: rerun training and inference

This level starts from the public waveform data rather than historical
checkpoints. It requires a CUDA-capable environment for the recorded GPU
campaigns, deterministic reconstruction of the frozen roles and subsets, and
the exact historical RESUM_FLEX snapshot. Restore that snapshot from a
lawfully obtained archive with:

```bash
.venv/bin/python ml4phy-paper/artifact/restore_resum_flex.py \
  --archive /path/to/resum-flex-edba6a.tar
```

The command checks SHA-256 before extraction, rejects unsafe archive members,
verifies the four execution-critical source files, and refuses to overwrite an
existing destination. The expected archive hash and source hashes are in
`manifests/resum_flex_compatibility.json`.

The historical snapshot is not redistributed here because the recovered copy
does not contain a license. Its recorded revision is
`edba6a294581fde6f905b330d477f2c1b42d6adb`. A release manager must provide a
licensed source location before claiming that a fresh third party can perform
the full training rerun without author assistance.

Once inputs are restored, the executed stages and their entry points are:

| Stage | Entry point | Frozen record |
| --- | --- | --- |
| Role and threshold construction | `scripts/build_protocol.py` | `manifests/frozen_protocol_v1.json` |
| Context-size neural campaign | `scripts/run_extension_neural_campaign.py` | `manifests/phase1_neural_result.json` |
| Kernel campaign | `scripts/run_extension_kernel.py` | `manifests/phase1_kernel_result.json` |
| Phase 2 training/evaluation | `scripts/run_phase2_training_campaign.py`, `scripts/run_phase2_evaluation_campaign.py` | `manifests/phase2_result.json` |
| Phase 3 training/evaluation | `scripts/run_phase3_training_campaign.py`, `scripts/run_phase3_evaluation_campaign.py` | `manifests/phase3_result.json` |
| Dense Bernoulli GP | `scripts/run_dense_gp_campaign.py` | `manifests/phase3_result.json` |
| Mechanism controls | `review_followup_20260910/run_training_campaign.py`, `run_evaluation_campaign.py` | `review_followup_20260910/manifests/provenance_v1.json` |

All paths in the table are relative to `ml4phy-paper/`. Use `--help` on an
entry point before execution. The campaign records and protocol JSON files,
rather than this overview, are authoritative for seeds, job inventories,
resource gates, and retry decisions. Reproduction must not replace missing
inputs, retune methods, or select favorable cells.

## Public raw data

The waveform source is the Majorana Demonstrator AI/ML data release:

- DOI: <https://doi.org/10.5281/zenodo.8257027>
- 16 `MJD_Train_*.hdf5` files form the training source.
- 6 `MJD_Test_*.hdf5` files form the evaluation source.

The exact paper analysis does not blindly sum all row counts. The classifier
uses 18,866 unique events selected from 377,330 energy-eligible training
events. The fixed final reference contains 114,400 events. Exact role counts,
overlaps, preprocessing, labels, repeated exposure, and identity hashes are in
`reports/data_budget_ledger.md` and `manifests/data_budget_ledger.json`.

The raw-data DOI alone is insufficient for a matched rerun: the recorded
classifier workflow, deterministic role and subset construction, and the
RESUM_FLEX snapshot are also required. Historical checkpoints are not required
when rerunning the complete pipeline from scratch.

## Environment

`pyproject.toml` and `uv.lock` define the current project environment. The
executed server environment is recorded in `manifests/server_environment.json`:
Python 3.12.13, PyTorch 2.11.0+cu129, CUDA 12.9, NumPy 1.26.4, and an RTX 5090.
The current lock may resolve newer packages than the historical run for
libraries whose lower bounds were used originally; the execution manifests
record the observed versions and numerical validations. Do not describe a
successful portable verification as a bitwise training reproduction.

## Result map

- Paper-facing curves and diagnostics:
  `reports/paper_presentation_export.md` and
  `manifests/paper_presentation_export.json`.
- Regional reference-band coverage:
  `reports/paper_coverage_export.md` and
  `manifests/paper_coverage_export.json`.
- Training-budget evidence: `reports/phase3_result.md`.
- Mechanism interpretation:
  `review_followup_20260910/reports/mechanism_result.md`.
- Final mechanism handoff:
  `review_followup_20260910/reports/final_handoff.md`.

## Known limits that must remain visible

- The efficiency-model result is conditional on the fixed classifier; it is
  not end-to-end training on 5,000 events.
- The 2k failure, 10k nonmonotonicity, and sparse-tail weakness are retained.
- Final targets were historically exposed.
- The reference set is finite and noisy. Reference-band coverage is not
  calibrated confidence coverage.
- Pointwise dropout standard deviations do not determine bin-mean covariance.
- The mechanism controls use one 5k subset.
- No 20k efficiency-model result exists; the full pool contains exactly 18,866
  events.
- The 1620-keV structure is named by energy in new outputs because historical
  isotope labels conflict.

## Remaining repository-owner actions

The code and lightweight evidence are reviewable and internally verified. Two
repository policy decisions remain outside the artifact build:

1. Confirm redistribution terms or a public source for the historical
   RESUM_FLEX revision.
2. Select a repository license before inviting code reuse.

The raw Majorana data remain available from their public DOI. Checkpoints and
event-level predictions intentionally remain server-side and do not need to be
published for the portable numerical verification or a from-scratch rerun.
