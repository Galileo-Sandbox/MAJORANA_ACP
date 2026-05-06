## Project Overview

Independent machine-learning research using the **Majorana Demonstrator AI/ML Data Release** (publicly released subset of DS6 ²²⁸Th calibration data; 3.2M HPGe detector waveforms with energy + PSD analysis labels; arXiv:2308.10856, Zenodo DOI 10.5281/zenodo.8257027).

This project is **not** the NPML challenge. We are pursuing our own research direction; see "Current Task" below for the first concrete piece of work.

## Environment

- **Python**: 3.12, managed by **uv** (`pyproject.toml` + `uv.lock`).
- **Always use the project virtualenv**: every Python command must use `/home/yuema137/MAJORANA_ACP/.venv/bin/python` (or activate `.venv/bin/activate` first). Never use the system `python` or `python3` — they are Python 3.8 and will fail on f-strings and other modern syntax.
- **Hardware**: single NVIDIA RTX 5090 (32 GB VRAM) on this server. The GPU is **shared with other users**, so check `nvidia-smi` before long runs and prefer `CUDA_VISIBLE_DEVICES=0` + memory caps when appropriate.

## Data

- **Location**: `/home/klz/Data/MAJORANA/` (the Zenodo partial release: 16 train + 6 test + 3 NPML files, ~45 GB total).
- **Treat this path as read-only**. Do not move, rename, or write into it.
- **Loading approach**: direct `h5py` + a `torch.utils.data.Dataset` (option a). Switch to a more elaborate pipeline (memmap, WebDataset, Parquet, etc.) only if profiling shows I/O is a real bottleneck.
- **HDF5 caveat**: `h5py` file handles do not survive `fork()`. When using `DataLoader(num_workers > 0)`, open the HDF5 file lazily inside the worker (e.g., on first `__getitem__`), not in `__init__`.
- **Dataset fields per event** (see arXiv:2308.10856 Table I):
  `raw_waveform` (3800-float array), `energy_label` (float, keV), `psd_label_low_avse` / `psd_label_high_avse` / `psd_label_dcr` / `psd_label_lq` (binary), `tp0` (int, rising-edge index), `detector` (int), `run_number` (int), `id` (int).
- **NPML split** has only `raw_waveform`, `detector`, `run_number`, `tp0`, `id` (no labels) — usable for unlabeled / self-supervised work only.

## Project Structure (planned)

```
majorana_acp/
  data/         # HDF5 loaders, Dataset, transforms
  models/       # nn.Module classes
  training/     # train loop, optimizer, checkpointing
  eval/         # metrics, energy histograms, diagnostics
  cli/          # entry points (train, evaluate, predict)
configs/        # pydantic-yaml config files
tests/          # pytest unit tests, mirrors package layout
notebooks/      # exploratory only — no source-of-truth logic here
data/           # gitignored; symlinks or local cache only
```

## Current Task: Binary Classifier for `psd_label_low_avse`

Train an ML classifier that predicts `psd_label_low_avse` from the raw waveform and outputs a continuous score in `[0, 1]`, so the score can be thresholded at inference time as a tunable cut. Other PSD labels are expected to follow the same pattern later.

### Data and split policy
- **Training** uses `MJD_Train_*.hdf5` files. File indices are configurable per experiment (a list, or the `"all"` shortcut).
- **Evaluation** uses `MJD_Test_*.hdf5` files only. Train and test files are never mixed.
- No further holdout from train files in v1 — there is no formal validation set during training. Revisit if overfitting becomes a concern.

### Two-stage data subsetting

When the run uses less than the full training pool, two independent knobs control the trimming, applied in this order:

1. **`data.subset_portion` (default `1.0`)** — *across all epochs* the Dataset only exposes a fraction of events. Picked once at construction using `data.subset_seed`, so the same fraction reused per run reproducibly. Useful for data-scaling studies and controlled comparisons across models.
2. **`data.train_portion` (default `1.0`)** — *per epoch*, the trainer's `WeightedRandomSampler` draws this fraction of the (already-subsetted) Dataset. Reshuffled each epoch.

Composition: `events_per_epoch = N_after_energy_filter × subset_portion × train_portion`. Existing configs run with `subset_portion=1.0`; older runs are equivalent.

### Waveform preprocessing (applied inside the Dataset)
1. Subtract baseline = mean of the **first 500 samples**.
2. Divide by the max of the baseline-subtracted waveform (per-event peak normalization).
3. Cast to **float32** before returning. HDF5 is float64; using that on the GPU doubles memory and halves throughput.

Optional alignment (DatasetConfig flag):
- `align_t90` (default `False`) — crop a fixed-length window around the **first sample at or above 0.9 of the normalized peak** ("90% rising-edge sample"). Window is `[t90 - t90_pre, t90 + t90_post)`, zero-padded if it extends beyond the waveform. Defaults give a 2200-sample window (`t90_pre=200`, `t90_post=2000`). Use this for models without translation invariance (e.g. MLP) so the rising-edge / decay-tail boundary lands at the same input index across events. CNNs don't need it.

### Dataset interface
Each item is a `dict` containing at least: `waveform` (float32 tensor), `label` (target PSD label, scalar), and the auxiliary fields `energy`, `tp0`, `detector`, `run_number`, `id`. The training loop consumes only `waveform` + `label`; the eval module consumes the rest for stratified analysis (e.g., acceptance vs. energy).

### Model output convention
- Models output a **single raw logit** per event — no sigmoid inside the model.
- Training loss is `BCEWithLogitsLoss` (numerically stable).
- At inference / evaluation, `sigmoid(logit)` produces the `[0, 1]` score that downstream code thresholds.

### Class imbalance handling
`psd_label_low_avse` has a pass rate ≳95%, so naive BCE collapses to "always pass". The `loss` config block exposes three strategies, all configurable and combinable:
- **Weighted BCE** via `pos_weight` (auto-computed from training data, or a fixed float).
- **Balanced sampling** via `torch.utils.data.WeightedRandomSampler`.
- **Focal loss**.

## Tooling

- **Configuration**: `pydantic` + `pydantic-yaml`. Every config object is a `pydantic.BaseModel`; YAML files load into and validate against these models.
- **Testing**: `pytest`. Each module under `majorana_acp/` should have a corresponding test file. Aim to test pure logic without requiring real HDF5 access where possible (use small synthetic fixtures).
- **Lint + format**: `ruff` (both `ruff check` and `ruff format`).
- **Type checking**: not enforced for now; revisit if the project grows.
- **Experiment tracking**: TBD. Decide before the first real training run.

## Coding Standards

- **Logic First**: Before every modification, review the current structure of the whole project and consider whether the structure itself is appropriate, rather than just bolting on the desired feature. Keep the code clean and elegant.
- **Slow is Smooth, Smooth is Fast**: Never be greedy when adding a feature or refactoring. Fix the bug first, then improve structure. Focus on the current problem at each step; do not over-optimize.
- **Clear docstrings and comments**: write correct types for inputs and outputs. `pydantic` validation and informative error messages are strongly encouraged for every function and class.
- **Avoid deep coupling between modules**: each module should be testable in isolation, pluggable, and decoupled.
- **Always think about what test we can add for each single module**: pytest is powerful — use it.
- **Be humble and curious**: if you are unsure about something — feature details, data format, intent — do not guess. Ask the user explicitly.
- **Be strict with the user and double-check**: what the user says is not always correct. If a statement seems wrong or an idea seems impractical, ask for clarification and state the objection clearly.
- **Never directly continue right after conversation compression**: stop after compression. The user will re-supply the context, docs, and code to read. Do not start blindly.
