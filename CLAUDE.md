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

### Per-epoch test loader uses the same sampler as training

When `data.sampler_strategies` is non-empty (or the legacy `loss.balanced_sampler=True` is set), the trainer's per-epoch test DataLoader gets a `WeightedRandomSampler` built from the **same** strategies as the train sampler — full coverage (`num_samples = len(test_ds)`), no `train_portion` factor. This makes the train_loss / test_loss curves directly comparable: both are computed on identically-distributed (class+energy balanced) batches, eliminating the train/test loss gap that arose from train batches being balanced while test batches were natural-distribution. The post-training `cli/evaluate` keeps natural iteration so the final metrics still reflect real-world performance on the unbalanced test set.

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

## Cut-Acceptance · True CNP (Per-Event Coordinates) Plan (FINAL)

### Reframe

The CNP for cut acceptance is **1D regression with binomial noise**: feature `x = (E, T)`, outcome `y ∈ {0,1}`, target `β(x) = P(y=1 | x)`. A faithful CNP for this problem must encode each context event's *own* coordinates `(E_i, T)` (Garnelo et al. 2018). The current `BinnedSampler` + `DESIGN_ONLY` setup broadcasts a single θ across all events in a trial — every event in the trial looks coordinate-identical to the encoder, which is why we needed bins to make sense of locality. The architecture, as used, was a degenerate special case.

The bin-isolated trial definition (`binned_sampler.py:209-215`) is a downstream consequence of this shortcut. Patching the inference with a window (Option B from the previous draft) makes the picture continuous, but it hides the real fix: **let the model see each event's coordinate, then the bins are just a visualization grid**.

### The breakthrough (no upstream library changes needed)

RESUM_FLEX's `StandardBatch` schema already supports per-event coordinates via the `EVENT_ONLY` mode: `theta=None`, `phi=[B, N, D_φ]`. We put `(E_i_normalized, T_normalized)` into `phi` (per event) and the existing `UniversalEncoder` ingests each event's coordinate individually:

```
phi_i = (E_i_norm, T_norm)        per event                    [B, N, 2]
phi_encoder(phi_i) → z_phi_i                                   [B, N, Z]
ContextPointEncoder([z_θ_null, z_phi_i, X_i]) → r_i            [B, N, agg]
mean(r_i over events) → r_trial                                [B, agg]
CnpDecoder(r_trial, z_phi_target) → (μ*, log σ*)               [B, N_t]
```

The `z_θ_null` slot is just an unused learnable token (the CNP learns to ignore it). The existing `train_cnp` loop already handles variable-N context via `cnp_config.n_context_min/max` and the per-batch `split_context_target` — so variable-N comes for free.

**Net effect: a textbook 1D-regression CNP, implemented by switching `InputMode.DESIGN_ONLY` → `InputMode.EVENT_ONLY` and writing one new sampler. No edits to `core/surrogate_cnp.py`, `core/networks.py`, or `core/training.py`.**

### Final decided values

| Knob                            | Value             | Rationale                                                         |
|---------------------------------|-------------------|-------------------------------------------------------------------|
| `InputMode`                     | `EVENT_ONLY`      | `phi=[B,N,2]` carries per-event `(E_i, T)`; `theta=None`.         |
| `dim_phi`                       | 2                 | `(E_i_norm, T_norm)`.                                             |
| `dim_theta`                     | `None`            | No broadcasted trial-level coordinate; the shortcut is gone.      |
| `n_events_per_trial_total`      | 48                | Headroom for ≥16 target points after `n_context_max` is removed.  |
| `cnp.n_context_min`             | 2                 | Variable-N: model learns σ_CNP scaling with data density.         |
| `cnp.n_context_max`             | 32                | Per-step `n_context ~ Uniform[2, 32]` in `train_cnp`.             |
| **Training sampling**           | **Bin-stratified**| Flatten energy distribution so the model sees rare-energy regions as often as dense ones; mandatory for high-E peak physics. |
| Inference D_C cap (per MC pass) | 256               | Big enough for a stable `r`; small enough for fast 50× MC.        |
| Dense grid for β(E) curve       | 800 points        | Already in place.                                                 |

### Concrete changes

1. **New file: `majorana_acp/cut_acceptance/event_sampler.py`**

   `class EventSampler` (~150 LOC). RESUM_FLEX `PseudoDataGenerator` duck-type. At construction, precompute a `_BinIndex` (same logic as `BinnedSampler`) — bins are used **only** as a sampling-stratification grid, never seen by the CNP.

   Per call to `generate(n_trials, n_events, seed)`:
   - For each trial `k`: sample `T_k` via boundary-mix (unchanged).
   - **Bin-stratified event draw**: for each of the `n_events` event slots, (a) pick a kept bin uniformly at random over `kept_bin_indices`, (b) pick one event uniformly at random from that bin's pool. Result: a trial's events are uniformly spread over the *bin grid*, not the natural energy density.
   - Per event: features `phi_i = (E_i_norm, T_k_norm)` (using the event's real energy, NOT the bin center); label `X_i = 1[score_i ≥ T_k]`.
   - Emit `StandardBatch(mode=EVENT_ONLY, theta=None, phi=[B, N, 2], labels=[B, N])`.
   - Stratification bin width: re-use `cfg.energy_bin_width` (the same 5/10/20 keV the user picked at the YAML level). The CNP never sees bin boundaries; this knob just controls how aggressively we flatten the training distribution.

2. **`majorana_acp/cut_acceptance/config.py`**
   - Add: `n_events_per_trial_total: int = 48` (positive, gt-equal `cnp.n_context_max + 1`).
   - Keep `energy_bin_width` — now serves a dual role explicitly documented:
     - Training: stratification bin width for `EventSampler`.
     - Inference / plotting: D_T binning grid for blue Wilson points + coverage metric.
   - Keep `min_events_per_bin` — used to drop sparse bins from the *stratification* pool (bins below the threshold contribute no events to training). Default unchanged.
   - Drop `n_per_trial` from the public surface (subsumed by `n_events_per_trial_total`).

3. **`majorana_acp/cut_acceptance/pipeline.py`**
   - Build CNP with `dim_theta=None, dim_phi=2`.
   - Replace `BinnedSampler` with `EventSampler`.
   - `training_pool.npz` simplifies to `{bin_centers, bin_event_counts, n_events_total}` — kept for inference's D_T binning, no longer load-bearing for training.
   - `run_summary.json`: drop `n_bins_used`; add `n_context_min/max`, `n_events_per_trial_total`, `sampling_strategy="bin_stratified"`.

4. **9 config YAMLs** under `configs/cut_acceptance/simple_cnn_small/bin{5,10,20}/{signal,background,inclusive}.yaml`
   - Add `n_events_per_trial_total: 48`.
   - Set `cnp.n_context_min: 2`, `cnp.n_context_max: 32`.
   - Remove `n_per_trial` from each YAML (or leave it as a no-op deprecated field).
   - `energy_bin_width` keeps its existing 5/10/20 value per cell.

5. **`scripts/diagnostics/cnp_test_inference.py` — simplification, not extension**

   The just-written windowed-context code is now **obsolete** and will be replaced with a clean global-aggregation path:
   - Each MC pass: sample up to **256 events** uniformly from D_C without replacement → one `StandardBatch(mode=EVENT_ONLY, phi=[1, n_ctx, 2], labels=...)` → one CNP forward to produce `r_trial`.
   - Build the target batch from the union of (bin centers, dense grid) with `phi=[1, N_query, 2]` rows `(E*_i_norm, T*_norm)`. Decode once per MC pass.
   - MC Dropout: `n_mc=50` passes, each with dropout active AND a fresh D_C subsample.
   - **Remove**: `_select_context_indices`, `context_window_kev`, `nearest_fallback_k`, `--context-window` CLI flag, and the matching arg in `run_all_test_inference.py`.
   - Keep: dense grid, `InferenceResult.dense_*` fields, plot layout, notebook `_plot_one`.

6. **Retrain all 9 cells**
   - `for cfg in configs/cut_acceptance/**/*.yaml: python -m majorana_acp.cut_acceptance.cli "$cfg"`
   - Estimate: similar to current (~5–10 min/cell on CPU).

### Tests

- `tests/cut_acceptance/test_event_sampler.py` (new): verify (i) emitted batches are valid `EVENT_ONLY` with `phi.shape == (B, N, 2)`, (ii) features are normalized into `[0, 1]`, (iii) labels are binary, (iv) different trials get different T values, (v) sampling is reproducible under fixed seed.
- `tests/cut_acceptance/test_pipeline.py`: update the smoke test for the new sampler. Drop the bin-related assertions; add an assertion that the saved checkpoint's metadata records `mode=EVENT_ONLY, dim_phi=2`.
- `tests/cut_acceptance/test_binned_sampler.py`: keep as a regression test on the old class until we delete it. The class itself stays in-repo for now (other diagnostics may still call it) — we can remove it in a follow-up.

### Validation

After retraining + new inference:
- β(E) curve is **continuous by construction** (one global `r` informs every query). NaN gaps physically cannot occur unless |D_C| = 0.
- σ_CNP at the data-sparse end of the spectrum should be **wider** than at the data-rich end — the model's calibrated epistemic uncertainty, not an artifact of empty bins.
- Pearson `r` vs D_T should be ≥ current bin-isolated value; combined coverage at 1σ should approach 0.683.

### Phased execution (so the user can intervene between phases)

1. **Phase 1 — Sampler + tests.** Implement `EventSampler`, write tests, run them. No pipeline changes yet. ~30 min.
2. **Phase 2 — Pipeline wiring.** Update `pipeline.py`, `config.py`, and one YAML (`simple_cnn_small/bin10/signal.yaml`) to use the new sampler. Retrain that ONE cell end-to-end. Verify the saved CNP loads back and the existing inference (after Phase 3) gives a sensible β(E) curve.
3. **Phase 3 — Inference rewrite.** Replace the windowed-context code in `cnp_test_inference.py` with the global-aggregation path. Verify on the Phase-2 checkpoint.
4. **Phase 4 — Roll out + retrain remaining 8 cells.** Update remaining YAMLs, retrain, regenerate `analysis/cnp_audit/` figures.

### Status of code already on this branch (uncommitted)

- `scripts/diagnostics/cnp_test_inference.py`: windowed-context + dense-grid version. **Will be partially overwritten in Phase 3.** The dense-grid plumbing + `InferenceResult` extension stay; the `_select_context_indices` / nearest-K / window kwarg go away.
- `notebooks/data_visualization.ipynb`: `_plot_one` already reads `res.dense_*`. **Keep.**
- `scripts/diagnostics/run_all_test_inference.py`: the `--context-window` flag becomes dead. **Will revert in Phase 3.**
- No commits yet; nothing destructive.

### Statistical note: why bin-stratified training is consistent with natural-density inference

Training under bin-stratified sampling, inference on natural-density D_C, looks like a train/test distribution mismatch at first glance. It isn't:

- The CNP learns the **conditional** rate `β(E, T) = P(y=1 | E, T)`, not the marginal energy distribution `p(E)`. Bin-stratification reweights the **input distribution** the model sees during training, but `β` is the same function regardless.
- At inference we condition on a specific D_C set; the model returns `β(E*, T*)` at any query point. The natural-density D_C just means the encoder aggregates a representation `r` reflecting the actual deployment context.
- The win: under natural-density training, sparse-energy regions (esp. high-E above the Tl-208 FE peak) contribute almost nothing to the loss → the model is under-fit there. Bin-stratification forces equal-weighted attention across the spectrum.

This is the same principle as the rare-class oversampling we already do for `psd_label_low_avse` — reweight the inputs to fix what the loss actually sees, without touching the target function.

