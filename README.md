# Majorana-ACP

Majorana-ACP studies the energy-dependent efficiency of a fixed pulse-shape
discrimination classifier on Majorana Demonstrator calibration data. The
repository contains the waveform-classifier pipeline, conditional efficiency
models, frozen ML4PS 2026 protocols, and portable paper evidence.

The paper estimand is

```text
efficiency(E) = P(classifier score >= fixed threshold | energy = E).
```

Historical code and schema fields may call the same quantity `acceptance`.

## Overall workflow

1. Load the public waveform data and apply the recorded preprocessing.
2. Train and evaluate the waveform classifier on the frozen split.
3. Calibrate one classifier threshold on its dedicated calibration role.
4. Train an efficiency model from energies and fixed-classifier outcomes.
5. Condition the efficiency model on a small context sample.
6. Evaluate saved predictions on the frozen finite reference and export paper
   tables, figures, and provenance.

Raw data, checkpoints, and event-level predictions are not committed. The
repository tracks code, configurations, compact outputs, identity hashes, and
instructions for verifying or restoring external inputs.

## Repository map

| Path | Contents |
| --- | --- |
| [`majorana_acp/`](majorana_acp/) | Installable data, model, training, evaluation, and efficiency-estimation package. |
| [`configs/`](configs/) | Classifier and efficiency-model configurations. |
| [`scripts/`](scripts/) | General diagnostics and compact-cache builders. |
| [`notebooks/`](notebooks/) | Interactive visualization entry points. |
| [`cache/`](cache/) | Small tracked inference products used by notebooks. |
| [`tests/`](tests/) | Package unit, integration, and parity tests. |
| [`experiments/`](experiments/) | Preserved historical ablation configurations. |
| [`ml4phy-paper/`](ml4phy-paper/) | ML4PS 2026 protocols, reports, tables, figures, and reproducibility tools. |

Each directory README describes its local files, inputs, outputs, and intended
use. The paper artifact has its own deeper index rather than duplicating all
experiment details here.

## Data and environment

The waveform source is the public Majorana Demonstrator AI/ML data release:

- Paper: <https://arxiv.org/abs/2308.10856>
- Dataset: <https://doi.org/10.5281/zenodo.8257027>

Create the locked project environment with:

```bash
uv sync --frozen --dev
```

Run the repository tests with the pinned historical RESUM_FLEX source on
`PYTHONPATH` when exercising the conditional-model pipeline:

```bash
PYTHONPATH=ml4phy-paper/local/resum-flex-edba6a:. uv run pytest
```

The historical dependency and large-input policy are documented in
[`ml4phy-paper/ARTIFACT_REPRODUCIBILITY.md`](ml4phy-paper/ARTIFACT_REPRODUCIBILITY.md).

## Paper artifact quick check

The portable paper evidence can be checked without raw data, checkpoints, or
event-level predictions:

```bash
uv run python ml4phy-paper/artifact/verify_artifact.py
```

Start with [`ml4phy-paper/README.md`](ml4phy-paper/README.md) for results,
scientific limitations, and the complete artifact navigation.

## Citation and license status

Paper-artifact citation metadata are provided in
[`ml4phy-paper/CITATION.cff`](ml4phy-paper/CITATION.cff). Cite the public data
release separately when using its waveforms or labels.

This repository does not currently declare a project license. No reuse license
should be inferred until the repository owner adds one.
