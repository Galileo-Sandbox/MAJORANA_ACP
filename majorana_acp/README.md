# Python package

`majorana_acp` contains the reusable implementation shared by the classifier
and efficiency-estimation workflows.

## Modules

| Path | Responsibility |
| --- | --- |
| `data/` | HDF5 dataset loading, preprocessing, and deterministic split helpers. |
| `models/` | Waveform classifiers and the attentive conditional model components. |
| `training/` | Classifier configuration, losses, optimization, and checkpoint writing. |
| `eval/` | Classifier evaluation and prediction export. |
| `cut_acceptance/` | Historical package name for efficiency-model configuration, sampling, training, and inference. |
| `analysis/` | Reusable numerical metrics. |
| `cli/` | Classifier training and evaluation command-line entry points. |

The paper-specific runners live under `ml4phy-paper/scripts/` so that frozen
paper protocols remain separate from the general package API.

## Data flow

The classifier consumes preprocessed waveforms and emits logits or scores. The
efficiency pipeline consumes event energies and binary outcomes formed by the
fixed score threshold; it does not retrain the classifier during paper
experiments. Conditional models receive a context set and produce efficiency
predictions at target energies.

## Extension points

- Register a new waveform model in `models/registry.py` and import it from
  `models/__init__.py`.
- Add efficiency-model configuration fields in `cut_acceptance/config.py`.
- Keep model construction in `cut_acceptance/pipeline.py` compatible with the
  pinned training interface.
- Add matching tests under `tests/` for new code paths.

The ML4PS mechanism controls are additive wrappers under
`ml4phy-paper/review_followup_20260910/`; the historical package implementation
is preserved for provenance.
