# majorana-acp

Machine-learning research on the public Majorana Demonstrator AI/ML data
release ([arXiv:2308.10856](https://arxiv.org/abs/2308.10856),
[Zenodo](https://doi.org/10.5281/zenodo.8257027)). The current task is
a binary classifier that predicts the `psd_label_low_avse` cut from a
raw Germanium-detector waveform and emits a continuous score in
`[0, 1]` so the cut threshold can be tuned at inference time.

This is **not** the NPML challenge — train and test are kept strictly
separated (the released `MJD_Train_*.hdf5` and `MJD_Test_*.hdf5` files,
respectively).

## Status

- ✅ Data loader (`majorana_acp/data`)
- ✅ Training config schema (`majorana_acp/training/config.py`)
- ✅ Model registry + concrete models: `simple_cnn`, `mlp`
  (`majorana_acp/models`)
- ✅ Trainer with per-epoch test monitoring (`majorana_acp/training`)
- ✅ Evaluation pipeline (`majorana_acp/eval`)
- ✅ CLIs: `majorana_acp.cli.train`, `majorana_acp.cli.evaluate`
- ✅ Run discovery + comparison plots in
  `notebooks/data_visualization.ipynb`

113 unit tests, ~3s end-to-end on CPU.

## Setup

Python 3.12 via [uv](https://docs.astral.sh/uv/). The project pins
`torch` to PyTorch's `cu129` wheels (the default `cu130` build fails on
this driver — see `pyproject.toml` for the override).

```bash
uv sync                                          # creates .venv and installs deps
.venv/bin/python -c "import majorana_acp"        # smoke test the editable install
```

The Zenodo dataset must already be present at
`/home/klz/Data/MAJORANA/` (16 train + 6 test + 3 NPML HDF5 files).
The path is configurable via `data.data_dir` in any experiment YAML.

## Project layout

```
majorana_acp/
  data/         HDF5 Dataset, file-list resolver
  models/       registry + simple_cnn + mlp
  training/     config schema, losses, trainer
  eval/         load checkpoint, run inference, save metrics
  cli/          train.py and evaluate.py entry points
configs/
  full_data_configs/   full-run experiment YAMLs (50 epoch baselines)
  small_data_configs/  same models but on a fixed 1% slice (subset_portion=0.01)
  smoke_tests/         fast iteration configs (one file, one epoch)
notebooks/      data_visualization.ipynb (exploration + run comparison)
tests/          pytest suite — synthetic HDF5 fixtures, no real data needed
runs/           per-run output dirs, mirroring the configs/ subfolders
                (gitignored). e.g. configs/full_data_configs/simple_cnn.yaml
                writes to runs/full_data_configs/simple_cnn_baseline/
```

A single training run writes the following to `runs/<exp-name>/`:

| File | Purpose |
|---|---|
| `metadata.json` | Full config + runtime info (git SHA, host, versions, start/end times, completed epochs, final metrics) |
| `training_history.json` | Per-epoch `train_loss`, `test_loss`, `test_roc_auc` |
| `epoch_NNN.pt` | Model + optimizer checkpoint (one per epoch) |
| `train.log` | Log file scoped to this run |
| `eval/predictions.h5` | (After running `evaluate`) raw per-event scores, logits, labels, energy, tp0, detector, run_number, id |
| `eval/metrics.json` | (After running `evaluate`) scalar metrics: counts, ROC-AUC, accuracy at 0.5 |
| `eval/eval.log` | Eval-side log |

## Train a model

Pick or write a config under `configs/`, then:

```bash
.venv/bin/python -m majorana_acp.cli.train configs/full_data_configs/simple_cnn.yaml
```

The reference configs are:

| Config | Purpose |
|---|---|
| `configs/full_data_configs/simple_cnn.yaml` | 1D CNN baseline, all 16 train files |
| `configs/full_data_configs/simple_cnn_derivative.yaml` | SimpleCNN with the derivative as a 2nd input channel |
| `configs/full_data_configs/mlp.yaml` | MLP baseline (1M params) |
| `configs/full_data_configs/mlp_v2.yaml` | Smaller MLP (122k params) |
| `configs/full_data_configs/mlp_derivative.yaml` | MLP with derivative channel |
| `configs/full_data_configs/resnet.yaml` / `resnet_single.yaml` | ResNet-1D, 2-channel and 1-channel |
| `configs/full_data_configs/inception.yaml` / `inception_single.yaml` | InceptionTime, 2-channel and 1-channel |
| `configs/small_data_configs/simple_cnn_small.yaml` / `resnet_single_small.yaml` | Same models on a fixed 1% slice (subset_portion=0.01, train_portion=1.0) for data-scaling studies |
| `configs/smoke_tests/quick_smoke.yaml` | One-file, one-epoch smoke test for SimpleCNN (~12 s on the 5090) |
| `configs/smoke_tests/quick_smoke_mlp.yaml` | Same smoke shape but with `model.name: mlp` |

A `train_portion` of `0.1` means each epoch draws a random 10 % of the
training events (sampled without replacement, reshuffled each epoch).
Set it to `1.0` to use the full ~1.04 M-event train set every epoch.

## Evaluate a trained run

The evaluator accepts either a `.pt` file or a run directory (it picks
the latest `epoch_*.pt` for you):

```bash
# Easiest — point at the run directory
.venv/bin/python -m majorana_acp.cli.evaluate runs/simple_cnn_baseline

# Or pin to a specific checkpoint
.venv/bin/python -m majorana_acp.cli.evaluate runs/simple_cnn_baseline/epoch_005.pt

# Override the output directory (default is <run-dir>/eval)
.venv/bin/python -m majorana_acp.cli.evaluate runs/simple_cnn_baseline --out /tmp/foo
```

Output goes to `runs/<exp-name>/eval/` by default.

## Inspect the results

`notebooks/data_visualization.ipynb` does double duty:

- Sections 1–7: data exploration (HDF5 schema, energy spectrum,
  per-PSD-cut survival, sample raw / preprocessed waveforms, …).
- Section 8: training results. The discovery cell at the top of
  section 8 walks `runs/` recursively and exposes a `GROUPS` list
  (e.g., `["full_data_configs"]` or `None` for all) so you can scope
  the comparison to a subset of subfolders. All later subsections
  (loss curves, ROC, energy-stratified rates, score / logit
  distributions, leaderboard table) operate on the filtered set.

Launch with:

```bash
.venv/bin/jupyter lab
```

The kernel uses the project's `.venv`. If your Jupyter doesn't see
`majorana_acp`, you may need to register the kernel once:

```bash
.venv/bin/python -m ipykernel install --user --name majorana-acp \
    --display-name "Python (majorana-acp)"
```

## Add a new model

The trainer is model-agnostic: it builds whichever class is registered
under `model.name` in your YAML, passing through `model.params` as
keyword arguments. Adding a new architecture is three steps.

### 1. Write the model file

Create `majorana_acp/models/<your_model>.py`:

```python
from __future__ import annotations

import torch
from torch import nn

from majorana_acp.models.registry import register_model


@register_model("my_resnet")  # this name is what configs reference
class MyResNet(nn.Module):
    """Input  : (B, L) float32 — preprocessed waveform.
    Output : (B,) raw logits (NO sigmoid; trainer uses BCEWithLogitsLoss)."""

    def __init__(self, channels: int = 32, depth: int = 4) -> None:
        super().__init__()
        # ... your architecture ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)        # (B, L) -> (B, 1, L)
        # ... compute features ...
        return logits.squeeze(-1)     # shape (B,)
```

Two conventions are non-negotiable:

- **Output a single raw logit per event** (no sigmoid in the model).
  Training uses `BCEWithLogitsLoss` for numerical stability;
  inference applies `torch.sigmoid` to get the [0, 1] score.
- **Accept `(B, L)`** waveforms; `(B, 1, L)` is also accepted as a
  convenience.

### 2. Register the import

Add one line to `majorana_acp/models/__init__.py` so the decorator
fires on package import:

```python
from majorana_acp.models import my_resnet  # noqa: F401
```

### 3. Reference it in a YAML config

Copy `configs/full_data_configs/simple_cnn.yaml`, change only the `model` block:

```yaml
model:
  name: my_resnet            # registry key from step 1
  params:
    channels: 64
    depth: 6
```

That's the entire change required to swap models — the trainer, eval
module, dataset, and metadata schema are all unchanged.

### Tests for the new model

Add a tiny test mirroring `tests/test_models.py`:

```python
def test_my_resnet_is_registered():
    assert "my_resnet" in list_models()

def test_my_resnet_forward_shape():
    model = build_model("my_resnet", channels=8, depth=2)
    out = model(torch.randn(4, 3800))
    assert out.shape == (4,)
```

Run the suite:

```bash
.venv/bin/python -m pytest -q
```

## Compare runs

After multiple `train` + `evaluate` invocations, re-run the cells in
section 8 of the notebook. The discovery cell scans every directory in
`runs/`; subsequent cells stack loss curves, ROC curves, and the
leaderboard table without any further configuration.

## Coding standards

See `CLAUDE.md` for the full picture. Highlights:

- pydantic models for config validation.
- pytest for everything; tests should not depend on the real
  `/home/klz/Data/MAJORANA/` files (use `tests/conftest.py` fixtures).
- ruff for lint and format. CI-equivalent locally:

  ```bash
  .venv/bin/python -m ruff check majorana_acp/ tests/
  .venv/bin/python -m ruff format --check majorana_acp/ tests/
  ```
