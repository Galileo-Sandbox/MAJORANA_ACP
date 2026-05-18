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

## Proposed Phase: Attentive CNP Aggregation Layers

### Why now

The PE10 experiment rescued the SE 2103 over-smoothing miss (Z_DT
−19.6 → −0.7) but introduced visible high-frequency oscillation /
ringing across the rest of the inclusive spectrum. This is a textbook
failure mode of the `mean` aggregator: once the encoder can represent
high-frequency modes (thanks to the Fourier features), the global
arithmetic mean over context events forces those modes to leak
uniformly across all target queries — the model has no mechanism to
say "use the high-frequency component near SE but not in the smooth
Compton continuum". We need a **target-conditioned, local-aware
aggregator**. The standard fix from Kim et al. 2019 (Attentive Neural
Processes) is to replace the mean with multi-head cross-attention from
target queries onto context keys.

### Upstream dependency audit — RESUM_FLEX (read-only verdict)

Audited `/home/yuema137/RESUM_FLEX/core/` to determine whether
pluggable aggregation requires touching the upstream repo.

| Component                              | Location                                                   | Aggregator-relevant? |
| -------------------------------------- | ---------------------------------------------------------- | -------------------- |
| `UniversalEncoder`                     | `core/networks.py:87`                                      | No — produces `(z_θ, z_φ)`; aggregation-agnostic. |
| `ContextPointEncoder`                  | `core/surrogate_cnp.py:46`                                 | No — MLP from `(z_θ, z_φ, X)` to per-event `r_i`; runs before aggregation. |
| `ConditionalNeuralProcess.aggregate()` | `core/surrogate_cnp.py:160` (`r_per_event.mean(dim=1)`)    | **Yes — mean is hard-coded.** |
| `ConditionalNeuralProcess.forward()`   | `core/surrogate_cnp.py:177`                                | Yes — calls `aggregate()` then broadcasts `r_trial` into the decoder. |
| `CnpDecoder.forward()`                 | `core/surrogate_cnp.py:106` (`r_trial.unsqueeze(1).expand`) | Yes — assumes `r_trial: [B, agg]` and broadcasts to `[B, N_t, agg]`. |
| `build_cnp()`                          | `core/surrogate_cnp.py:254`                                | Yes — fixed wiring; emits the mean-based CNP only. |
| `core.training.train_cnp()`            | `core/training.py:58`                                      | No — calls `cnp(ctx, tgt) → CnpOutput`; aggregator-agnostic as long as our class returns the same dataclass. |

**Verdict: zero upstream changes required.** The mean is hard-coded
*inside* `ConditionalNeuralProcess`, not behind an abstract base
class, so we cannot swap it via subclass override of `aggregate()`
alone (cross-attention needs target queries, which `aggregate()`
doesn't see). The cleanest cut is to write a **parallel local CNP
class** in `majorana_acp/models/attentive_cnp.py` that:

* reuses upstream `UniversalEncoder` (drop-in),
* reuses upstream `ContextPointEncoder` to produce per-event values
  `h_C = r_i` (the V tensor),
* implements a local `CrossAttentionAggregator(nn.Module)` that
  emits a target-conditioned `r(φ_T) ∈ [B, N_T, d_v]`,
* reuses upstream `CnpDecoder` **as a stateless MLP** (call
  `decoder.net` directly to bypass the `r_trial.unsqueeze(1)` line
  that assumes mean aggregation),
* returns the upstream `CnpOutput(mu_logit, log_sigma)` dataclass so
  `core.training.train_cnp` and `core.surrogate_cnp.cnp_loss` work
  unchanged.

Aggregator selection lives in our local `pipeline.py::build_cnp(...)`
which dispatches between upstream `build_cnp(...)` (mean) and the
local `build_attentive_cnp(...)` based on `cfg.aggregator.type`. The
upstream factory and the entire upstream tree stay byte-identical;
every existing checkpoint loads through the same `build_cnp(...)`
call it was saved with.

### Config schema extension

Add a nested `AggregatorConfig` to `CutAcceptanceConfig`:

```yaml
aggregator:
  type: "mean"          # Literal["mean", "cross_attention"]; MUST default to "mean"
  num_heads: 8          # H ∈ ℤ≥1; ignored when type == "mean"
  attention_dim: 64     # d_k = d_v per-head total; ignored when type == "mean"
```

Validators:

* `type` must be in `{"mean", "cross_attention"}` (pydantic `Literal`).
* `num_heads ≥ 1`, `attention_dim ≥ 1`.
* `attention_dim % num_heads == 0` (so per-head dim is integral).
* When `type == "mean"`, `num_heads` / `attention_dim` are kept in the
  config object (no override) but are unused — they're no-ops, not
  errors, so flipping the flag back and forth doesn't require also
  re-tuning them.

Default factory: `AggregatorConfig(type="mean", num_heads=8,
attention_dim=64)`. Every existing YAML parses unchanged with PE-style
backward compatibility (`Field(default_factory=AggregatorConfig)`).

### Mathematical contract & tensor topology

Notation: batch size `B`, context size `N_C`, target queries `N_T`,
encoder latent dim `Z = encoder.latent_dim`, aggregator-output dim
`d_v` (equals `Z` by default so the decoder input dim is invariant
across paths — the key trick that keeps the decoder shape-stable).

**Path A — `type: "mean"` (legacy, upstream `ConditionalNeuralProcess`, UNCHANGED)**

```
z_φ_C   = phi_encoder(φ_C)                            [B, N_C, Z]
z_φ_T   = phi_encoder(φ_T)                            [B, N_T, Z]
h_C     = ContextPointEncoder(z_θ_pe, z_φ_C, X_C)     [B, N_C, agg]    # agg = Z
r_trial = mean(h_C, dim=1)                            [B, agg]
r_bcast = r_trial.unsqueeze(1).expand(-1, N_T, -1)    [B, N_T, agg]
X_dec   = concat([r_bcast, z_φ_T], dim=-1)            [B, N_T, agg + Z]
(μ, logσ) = CnpDecoder.net(X_dec)                     [B, N_T, 2]
```

**Path B — `type: "cross_attention"` (local `AttentiveCNP`)**

```
z_φ_C   = phi_encoder(φ_C)                            [B, N_C, Z]
z_φ_T   = phi_encoder(φ_T)                            [B, N_T, Z]
h_C     = ContextPointEncoder(z_θ_pe, z_φ_C, X_C)     [B, N_C, agg]    # agg = Z

# Multi-head projections — keys/queries derived from the same latent
# space (z_φ) the upstream encoder already trains end-to-end; values
# from h_C so the attention can route the per-event context
# representation that already absorbed the label X_C.
Q = z_φ_T · W_Q                                       [B, N_T, attention_dim]
K = z_φ_C · W_K                                       [B, N_C, attention_dim]
V = h_C   · W_V                                       [B, N_C, attention_dim]

# Reshape for H heads, each of width d = attention_dim / num_heads.
Q' = reshape(Q, [B, N_T, H, d]).transpose(1, 2)       [B, H, N_T, d]
K' = reshape(K, [B, N_C, H, d]).transpose(1, 2)       [B, H, N_C, d]
V' = reshape(V, [B, N_C, H, d]).transpose(1, 2)       [B, H, N_C, d]

# Scaled dot-product attention per head.
α  = softmax(Q' @ K'.transpose(-2, -1) / √d, dim=-1)  [B, H, N_T, N_C]
O' = α @ V'                                           [B, H, N_T, d]

# Merge heads + final linear projection W_O back to d_v = agg = Z so
# the decoder input dim equals the mean path's (no decoder changes).
O = O'.transpose(1, 2).reshape(B, N_T, attention_dim) [B, N_T, attention_dim]
r(φ_T) = O · W_O                                      [B, N_T, d_v=Z]

X_dec   = concat([r(φ_T), z_φ_T], dim=-1)             [B, N_T, d_v + Z]
(μ, logσ) = CnpDecoder.net(X_dec)                     [B, N_T, 2]
```

**Why this design**

* `Q` and `K` derived from `z_φ` (post-encoder), not raw `φ` — keeps
  attention operating in the same latent space the rest of the model
  uses; consistent with upstream's "all downstream sees the latent"
  philosophy and lets the PE features flow through naturally.
* `V` derived from `h_C` (not `z_φ_C`) so the value carries the
  per-event representation that already absorbed the binary label
  `X_C` — same source of truth the mean aggregator uses.
* `d_v = Z = agg_dim` by default → decoder input dim is **invariant**
  across paths. Same `CnpDecoder` instance works for both. (The
  factory still uses the upstream `CnpDecoder`; we only bypass its
  `forward` to skip the unsqueeze-expand.)
* Multi-head: each head can specialize on a different scale of the PE
  basis (low-frequency global context vs high-frequency peak detail)
  — exactly the gating we want.

### Backward-compatibility guardrails

1. **Default = legacy.** `AggregatorConfig(type="mean", ...)` is the
   default factory. Every existing YAML and trained checkpoint runs
   through upstream `build_cnp(...)` and upstream
   `ConditionalNeuralProcess` — byte-identical to today.
2. **`build_local_cnp(cfg)` dispatch lives in `pipeline.py`.** When
   `cfg.aggregator.type == "mean"`, we call upstream `build_cnp(...)`
   verbatim. Otherwise we call `majorana_acp.models.attentive_cnp.build_attentive_cnp(...)`.
   The upstream factory signature is unchanged; pre-PE-feature
   checkpoints continue to load via the same `core.build_cnp` import.
3. **`_load_cnp` in `cnp_test_inference.py`** branches on
   `cfg.aggregator.type` symmetrically. Pre-aggregator checkpoints
   default to `mean` (the missing config field defaults in pydantic).
4. **Parity unit tests (mandatory):**
   * `test_aggregator_disabled_is_bitwise_identical.py`: build two
     CNPs from the same encoder seed — one via upstream `build_cnp`,
     one via our `build_local_cnp(cfg with type="mean")`. Assert
     `state_dict()` equality (or, more precisely: same architecture
     class is returned — `isinstance(cnp, ConditionalNeuralProcess)`
     from upstream — and `forward(ctx, tgt)` outputs are equal
     tensor-for-tensor under identical RNG).
   * `test_load_legacy_checkpoint.py`: load any existing trained
     `cnp.ckpt` (we'll point it at the `true_cnp/bin10/signal` ckpt
     that ships in the registry) and assert `_load_cnp(cfg)` succeeds
     with no architecture mismatch.
5. **Paradigm path suffix.** Append `_attn<H>x<d>` only when
   `type == "cross_attention"` (e.g.
   `..._varN32-1024_pe10_attn8x64`). Mean stays unsuffixed so
   existing paths never collide.
6. **Checkpoint metadata.** Save
   `aggregator_type / aggregator_num_heads / aggregator_attention_dim`
   in the checkpoint's metadata dict so each artifact records the
   exact aggregator it was trained under.

### Phased execution plan (NOT YET AUTHORIZED — for sign-off)

| # | Phase                                                                                                                | Files                                                                                                                                                          |
| - | -------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | Config schema + `AggregatorConfig` model + validators                                                                | `majorana_acp/cut_acceptance/config.py`, `tests/cut_acceptance/test_config.py`                                                                                 |
| 2 | New module `majorana_acp/models/__init__.py` + `attentive_cnp.py` (CrossAttentionAggregator + AttentiveCNP class)    | `majorana_acp/models/attentive_cnp.py` (new), `tests/models/test_attentive_cnp.py` (new)                                                                       |
| 3 | Pipeline dispatch — `build_local_cnp(cfg)` selecting mean vs attention; paradigm-suffix tag; checkpoint metadata     | `majorana_acp/cut_acceptance/pipeline.py`, `tests/cut_acceptance/test_pipeline.py`                                                                             |
| 4 | Inference dispatch — `_load_cnp` branches on aggregator type                                                         | `scripts/diagnostics/cnp_test_inference.py`                                                                                                                    |
| 5 | Parity regression — mean path bitwise-equal to legacy                                                                | `tests/cut_acceptance/test_aggregator_parity.py` (new)                                                                                                         |
| 6 | Smoke YAML at `configs/.../hybrid_scale/mixed_density_f0_70_w10_varN32-1024_pe10_attn8x64/bin10/inclusive.yaml`       | new YAML inheriting w10_varN_pe10 + `aggregator: {type: cross_attention, num_heads: 8, attention_dim: 64}`                                                     |
| 7 | Train + full-context infer + rebuild `_audit_inclusive.md` to include `w10_varN_pe10_attn`                            | `scripts/diagnostics/build_inclusive_audit.py` (add paradigm row), regenerate report                                                                           |

### Settlements (LOCKED — user sign-off)

1. **Dimension sizing: `d_v = Z`.** Locked. Decoder input shape stays
   `[B, N_T, 2Z]`; upstream `CnpDecoder.net` is reused unchanged.
2. **`Q / K` source: `z_φ` (post-phi-encoder).** Locked. With PE
   enabled, the 21-dim Fourier feature flows through `phi_encoder`
   first, then attention operates in the resulting latent.
3. **Dropout: reuse `encoder_config.dropout`.** Locked. Applied to
   attention weights `α` AND to the output projection `W_O`. No new
   YAML knob.
4. **Single cross-attention layer.** Locked. Stacking is a follow-up
   only if a single layer wins decisively.
5. **Smoke target: `bin10/inclusive`.** Locked. Direct A/B against
   `w10_varN_pe10/bin10/inclusive` — the PE10 cell whose sawtooth
   motivated this proposal.

### What this will NOT touch

* `/home/yuema137/RESUM_FLEX/` (read-only — confirmed by audit).
* Existing `true_cnp` / `w10_*` paradigm tags.
* Existing trained checkpoints — they continue to load and run via
  the upstream `build_cnp` path with no aggregator key in their YAML
  (pydantic defaults `aggregator.type="mean"`).
* The notebook (per the standing rule).

## Cut-Acceptance · 1D Fourier Positional Encoding for Energy (PROPOSED)

### Why

The inclusive-cell audit (`analysis/cnp_audit/_audit_inclusive.md`)
shows every paradigm catastrophically over-smoothing the sharp
acceptance features at Tl-208 SE (Z_DT ≈ −15σ) and Tl-208 DEP
(Z_DT ≈ +18σ): the CNP's MLP-encoder + mean-aggregation pipeline
acts as a strong low-pass filter on the 1D energy coordinate, and
the inclusive function's peak structure is ~5× sharper than the
signal function's. Hybrid sampling (focus windows, physics anchors,
variable N) shifts the bias but cannot remove it — the limiting
factor is the **representational bandwidth** of the network's
treatment of `E`, not the sampling.

The standard cure from NeRF / Tancik et al. is a 1D Fourier feature
expansion: lift the scalar `E` into a high-dimensional sinusoidal
basis before any MLP touches it. This is a *purely mathematical*
inductive bias — no physical priors, no hand-crafted peaks — that
gives the network access to high-frequency modes of `E` it cannot
otherwise represent from a 1D input.

### Formula

Per-event normalization (same `E_min`, `E_max` as the energy range
in the existing config):
```
E_norm = (E - E_min) / (E_max - E_min)        # E_norm ∈ [0, 1]
```

Multi-scale sinusoidal expansion with `L = num_bands` bands:
```
γ(E_norm) = [
    sin(2⁰ π E_norm), cos(2⁰ π E_norm),
    sin(2¹ π E_norm), cos(2¹ π E_norm),
    ...,
    sin(2^(L-1) π E_norm), cos(2^(L-1) π E_norm),
]                                              # 2L-dim vector
```

The highest band (`2^(L-1) π`) has period `2 / 2^(L-1)` in the
normalized coordinate; at `L=10` over a 2500-keV range that's
`2500 / 256 ≈ 9.8 keV` — deliberately matched to the bin10 grid so
the network can resolve bin-scale features.

### Architectural placement

Our CNP runs in `InputMode.EVENT_ONLY` with per-event
`phi_i = (E_i_norm, T_norm)` ∈ ℝ². With PE enabled, replace `E_i_norm`
with `γ(E_i_norm)` and keep `T_norm` as a single scalar appended at
the end:
```
phi_i = [γ(E_i_norm), T_norm]                  # (2L + 1)-dim
```
Therefore `dim_phi = 2L + 1` when PE is enabled and `dim_phi = 2`
when disabled. The transform is applied **at phi-construction time**
(inside `EventSampler` for training and inside
`scripts/diagnostics/cnp_test_inference.py` for evaluation) — this
avoids modifying RESUM_FLEX upstream and keeps the CNP module
oblivious to whether its input went through PE. Both the context
phi and the target query phi are encoded identically.

### Config schema

```yaml
positional_encoding:
  enabled: false          # MUST default false (backward compat)
  num_bands: 10           # L
  min_energy_kev: 500.0   # = energy_range[0] in practice
  max_energy_kev: 3000.0  # = energy_range[1] in practice
```

Defaults: `enabled=False`. Validators: `num_bands ≥ 1`,
`max_energy_kev > min_energy_kev`. The `min_energy_kev` /
`max_energy_kev` fields shadow `energy_range` so PE has its own
explicit normalization window (we don't tie them by validator; if
the user wants a different PE window than the training energy range,
they can set it).

### Backward compatibility contract

* `positional_encoding: PositionalEncodingConfig = default(enabled=False)`
  added to `CutAcceptanceConfig` via `Field(default_factory=…)`. All
  existing YAMLs parse unchanged.
* When `enabled=False`, the PE module is a no-op pass-through —
  phi keeps shape `(B, N, 2)` and `dim_phi = 2`. Training,
  inference, and reports are bit-for-bit identical.
* Trained checkpoints (all PE-disabled) load and run unchanged
  because the CNP architecture is rebuilt from the same YAML.
* New PE cells get their own paradigm-tag suffix (`_pe<L>`) so the
  canonical results path stays unique and registry stays clean.

### Implementation plan

| # | Phase                                          | Files                                                                                                                                |
| - | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 1 | Config schema + PositionalEncodingConfig model | `majorana_acp/cut_acceptance/config.py`, `tests/cut_acceptance/test_config.py`                                                       |
| 2 | PE module (pure numpy `encode_phi`)            | `majorana_acp/cut_acceptance/positional_encoding.py`, `tests/cut_acceptance/test_positional_encoding.py`                             |
| 3 | Sampler + pipeline hookup (`dim_phi` dynamic)  | `majorana_acp/cut_acceptance/event_sampler.py`, `majorana_acp/cut_acceptance/pipeline.py`, `tests/cut_acceptance/test_event_sampler.py` |
| 4 | Inference hookup (encode context + query phi)  | `scripts/diagnostics/cnp_test_inference.py`                                                                                          |
| 5 | Paradigm-suffix serializer (`_pe10` tag)       | `majorana_acp/cut_acceptance/pipeline.py::paradigm_path_suffix`                                                                      |
| 6 | Smoke YAML at canonical path                   | `configs/cut_acceptance/simple_cnn_small/hybrid_scale/mixed_density_f0_70_w10_varN32-1024_pe10/bin10/inclusive.yaml`                  |
| 7 | Train + infer + report                         | (sweep + `_audit_inclusive.md` rebuild; add `w10_varN_pe10` to `PARADIGMS`)                                                         |

### Open decisions (flagging before execution)

1. **PE on T (threshold) too?** Spec says energy only. The CNP
   currently learns `β(E, T)` jointly via the encoder; PE on T
   could similarly help if the model is also smoothing in T-space,
   but no evidence of that yet. Default: leave T as a scalar.
2. **Path-suffix encoding of `num_bands`**: include `_pe<L>` in the
   paradigm path (e.g. `..._pe10`). With L=10 default we get one
   string; if the user runs PE at L=6 and L=14 we get different
   paths — good for registry hygiene.
3. **Test fixture for parity**: we want a pytest assertion that
   "PE disabled produces the same `phi` array as the pre-PE
   sampler". Easiest implementation: parameterize an `encode_phi`
   call with `enabled=False` and assert `np.array_equal(out, input)`.
4. **Should existing inclusive YAMLs auto-gain PE?** No — the
   point of the PE cell is to A/B against the existing
   `w10_varN_large` golden. Keep PE opt-in.

## Cut-Acceptance · Localized Peak Evaluation + High-Capacity Hybrid Sweep (PROPOSED)

### Why now

The first hybrid smoke (`mixed_density_f0_70_w50`) showed a 22 % NLL
drop vs True-CNP baseline, but a 50 keV focus window smooths over
HPGe peak structure (FWHM ~few keV). To get the CNP to actually
*resolve* peak-region acceptance transitions we need (a) much
narrower focus windows that match real detector physics (10 / 5 keV),
(b) much larger N per trial so the encoder's `r` carries enough
information to support that resolution, and (c) a quantitative
evaluation rooted in the peak regions themselves — global Pearson r
is dominated by the flat majority of the spectrum and can't tell us
whether sharp features are captured.

### Phase 1 — Localized peak-region metrics

Add a `PeakRegionMetrics` dataclass and a computation function to
`scripts/diagnostics/cnp_test_inference.py`. For each peak `E₀` in
the existing `PEAK_MARKERS` list, define the evaluation window
`I_peak = [E₀ − 5, E₀ + 5]` keV and compute, **separately for D_C
and D_T**:

* `chi2_<X>` — reduced χ²:
  `(1/N_bins) Σ (y_i − β_i)² / (σ_stat_i² + σ_CNP_i²)`
* `z_<X>` — local Z-score:
  `(ȳ_I − β̄_I) / √(Var(y)_I + ⟨σ_CNP²⟩_I)`
* `p_<X>` — two-tailed p-value from |Z| via `scipy.stats.norm.sf`.

D_C support requires also binning the context set with Wilson
intervals (today only D_T is binned for the blue points). Cheap
addition — reuse `_empirical_with_wilson`.

`InferenceResult.peak_metrics: list[PeakRegionMetrics]` carries the
arrays. `write_report` formats a per-peak table and writes it into
`test_set_audit.txt` + `test_set_audit.json`. The audit PNG is
unchanged in this phase.

**Statistical edge case**: with 10-keV-wide peak windows on a 10-keV
bin grid, `n_bins_in_window ∈ {1, 2}`. χ² and Z over 1 datapoint
are weak signals; treat them as point estimates rather than rigorous
hypothesis tests until we run wider-window or finer-binning cells.

### Phase 2 — Configuration matrix (4 new YAMLs)

All at `bin10/signal` to enable direct baseline comparison. YAMLs
live at the canonical auto-derived paths (not the user-supplied
nickname filenames) so the registry, results tree, and inference
sweep stay in sync. Mapping:

| nickname           | canonical YAML path                                                                            |
| ------------------ | ---------------------------------------------------------------------------------------------- |
| `w10_fixed48`      | `hybrid_scale/mixed_density_f0_70_w10/bin10/signal.yaml`                                       |
| `w10_varN_large`   | `hybrid_scale/mixed_density_f0_70_w10_varN32-1024/bin10/signal.yaml`                           |
| `physics_w10_varN` | `hybrid_scale/physics_anchored_f0_80_w10_varN32-1024/bin10/signal.yaml`                        |
| `hyper_zoom_w5`    | `hybrid_scale/mixed_density_f0_85_w5_varN32-1024/bin10/signal.yaml`                            |

Knobs per cell (only deltas from `true_cnp/bin10/signal.yaml` shown):

* `w10_fixed48`: `sampling_pattern: mixed_density`, `zoom_window_width_kev: 10`, `local_event_fraction: 0.70`, `trial_size_strategy: fixed`, `training.n_events_per_trial: 48`.
* `w10_varN_large`: same `mixed_density / w=10 / f=0.70` as above plus `trial_size_strategy: variable_uniform`, `n_trial_events_min: 32`, `n_trial_events_max: 1024`.
* `physics_w10_varN`: `sampling_pattern: physics_anchored`, `w=10`, `local_event_fraction: 0.80`, `physics_peaks_kev: [1592, 1620, 2103, 2614]`, `varN 32-1024`.
* `hyper_zoom_w5`: `sampling_pattern: mixed_density`, `w=5`, `local_event_fraction: 0.85`, `varN 32-1024`.

### Compute / memory plan for the large-N cells

`n_events=1024 × batch_size=16 = 16,384` events per step. Per-event
encoder activations are `[B, N, Z=64]` ≈ 4 MB; gradients double
that. Single forward + backward sits comfortably in CPU RAM and
GPU VRAM (32 GB free). Per-step cost grows roughly linearly in N:

| N    | est. step time on CPU | total for 3000 steps |
| ---- | --------------------- | -------------------- |
| 48   | ~7 ms                  | ~25 s   (current)    |
| 1024 | ~150 ms                | ~7–8 min             |

For three N=1024 cells + one fixed-N cell that's ~25 min of CPU. If
the per-step time blows past 200 ms during smoke training, I'll
switch to GPU (one-line addition: `cnp.to("cuda")` + move batches).
Otherwise CPU is fine and simpler.

### Phase 3 — Sweep + comparison table

1. Train all 4 cells with `python -m majorana_acp.cut_acceptance.cli <cfg>`.
2. Run `python -m scripts.diagnostics.run_all_test_inference` to
   regenerate audit PNGs + per-cell `test_set_audit.json` (now
   carrying peak metrics).
3. Build a comparison report at `analysis/cnp_audit/_peak_comparison.md`
   that loads each cell's JSON, ranks by peak χ² and p-value, and
   prints a table:

   ```
   peak     | true_cnp                | w10_fixed48              | w10_varN_large            | physics_w10_varN          | hyper_zoom_w5
            | χ²_DC χ²_DT  p_DC  p_DT | χ²_DC χ²_DT  p_DC  p_DT | ...
   FE 2614  |  1.23  1.85  0.20  0.04 |  ...
   SE 2103  |  ...
   DEP 1592 |  ...
   1620     |  ...
   ```

Targets to look for: χ²_DT closer to 1.0 (cleanly piercing data),
p_DT > 0.5 (statistically indistinguishable from sharp truth).

### Open questions

1. **Naming**: I'll use canonical paths for the YAMLs. If you want
   the user-supplied nicknames literally (e.g.
   `configs/.../hybrid_scale/w10_fixed48.yaml` as a top-level YAML
   without the `binN/{class}.yaml` subdirectory), say so — that
   would require an explicit `out_dir` / `name` override per YAML
   and a small change to the diagnostic sweep's path expectation.
2. **Peak window**: 10 keV-wide (`half_w = 5.0`) gives 1–2 bins per
   peak at bin10. If you want a "wider statistical aperture" without
   changing the focus physics (e.g. half_w=20 for evaluation only,
   while training still uses w=5 or w=10), I'll add it as a kwarg
   default.
3. **Compute mode**: CPU vs GPU. I'll start CPU; pivot if step time
   exceeds the estimate.

### Commit plan

| # | Commit                                                              |
| - | ------------------------------------------------------------------- |
| 1 | `Peak-region χ² + Z metrics for D_C and D_T`                        |
| 2 | `4 hybrid-scale configs: w10/fixed48, w10/varN, physics, hyper_w5`  |
| 3 | `Trained registry entries for the new 4 cells`  (4 run_summary.json) |
| 4 | `Peak-comparison report under analysis/cnp_audit/`                  |

---

## Cut-Acceptance · Hybrid-Scale Sampling Plan (FINAL)

### Settlements
1. **Variable-N**: per-step wrapper in `pipeline.py`; RESUM_FLEX untouched.
2. **Path derivation**: auto when YAML leaves `out_dir`/`name` blank; YAML-explicit values override.
3. **Path serialization**: underscore for decimals → `mixed_density_f0_70_w50` (not `f0p70`).
4. **`random_clusters` failure**: raise `ValueError` after 50 rejection attempts. No silent fallback.
5. **Inference**: untouched by `variable_uniform`. D_C remains the global-aggregation set built from the test split.

(Earlier "PROPOSED" copy below; settlements above supersede where they conflict.)

### Why now

The True-CNP framework is in place but every cell uses one sampling strategy: `flat_stratified` (bin-uniform). Real spectra have sharp acceptance transitions at γ peaks (Tl-208 FE 2614 keV, ⁴⁰K 1460 keV) that a uniformly-stratified context smooths over. We need a *matrix* of sampling strategies so the same CNP architecture can be probed under different spatial-information regimes, with clean version control so experiments don't clobber each other.

### Design intent — two orthogonal axes

| Axis | Knob | Values |
|---|---|---|
| **A. Trial size** | `trial_size_strategy` | `fixed` \| `variable_uniform` |
| **B. Spatial pattern** | `sampling_pattern` | `flat_stratified` \| `mixed_density` \| `random_clusters` \| `physics_anchored` |

The two are deliberately decoupled — every (A, B) combination is a valid experiment.

### Critical RESUM_FLEX compatibility constraint (read this before agreeing to the spec)

`core/training.train_cnp` reads `training_config.n_events_per_trial` **once** outside the step loop, and passes that fixed value to `generator.generate(n_events=…)` every step. The downstream encoder also requires uniform `N` per batch (`StandardBatch.labels` is `[B, N]`, no padding mask).

Consequence: **per-trial variable `N` within a single batch is not implementable without modifying RESUM_FLEX**. Two ways to honor the `variable_uniform` spec:

* **Per-step variable N (recommended)**: each training step picks one `N ~ Uniform[N_min, N_max]` and all trials in that step share it. Across the full training, the model still sees the full range. We get there by writing a thin training loop wrapper in `majorana_acp/cut_acceptance/pipeline.py` that mimics `train_cnp` but resamples `n_events` per step. RESUM_FLEX stays untouched.
* **True per-trial variable N**: would need a padded-N + mask extension to the CNP encoder. Bigger refactor, deferred.

I'll proceed with per-step variable N. If you want true per-trial variable N, that's a separate plan (RESUM_FLEX padding-mask extension).

### Phase 1 — Config schema + sampler refactor + tests

**Config (`majorana_acp/cut_acceptance/config.py`)** — add the new knobs as two sub-blocks. Defaults reproduce current behavior exactly so the 9 existing YAMLs need no edits:

```python
trial_size_strategy: Literal["fixed", "variable_uniform"] = "fixed"
n_events_per_trial_total: int = 48                 # used by "fixed"
n_trial_events_min: int = 16                       # used by "variable_uniform"
n_trial_events_max: int = 64                       # used by "variable_uniform"

sampling_pattern: Literal[
    "flat_stratified", "mixed_density",
    "random_clusters", "physics_anchored",
] = "flat_stratified"
zoom_window_width_kev: float = 50.0                # used by mixed_density / clusters / anchored
local_event_fraction: float = 0.70                 # used by mixed_density / anchored
n_clusters: int = 2                                # used by random_clusters
physics_peaks_kev: list[float] = [1460.0, 2614.0]  # used by physics_anchored
```

Validators: enforce `n_trial_events_max ≥ n_trial_events_min ≥ 4`, `0 < local_event_fraction < 1`, `n_clusters ≥ 1`, `zoom_window_width_kev > 0`. Filter `physics_peaks_kev` to those inside `energy_range` at config-load time (warn if any dropped).

**Sampler (`majorana_acp/cut_acceptance/event_sampler.py`)** — keep the same `generate(n_trials, n_events, seed)` signature; the caller continues to control `n_events`. The sampler now reads the pattern config and dispatches per trial:

```python
class EventSampler:
    def __init__(self, …, *,
                 sampling_pattern, zoom_window_width_kev, local_event_fraction,
                 n_clusters, physics_peaks_kev):
        …

    def generate(self, n_trials, n_events, seed) -> StandardBatch:
        rng = np.random.default_rng(seed)
        t_k = self._sample_t(rng, n_trials)
        event_indices = np.empty((n_trials, n_events), dtype=np.int64)
        for k in range(n_trials):
            event_indices[k] = self._draw_one_trial(rng, n_events)
        # ... assemble phi + labels from event_indices ...
```

`_draw_one_trial(rng, n)` is the dispatch. Each pattern is a private helper:

**Pattern 1 — `_draw_flat_stratified(rng, n)`**: current behavior, factored out (no logic change). For `n` slots, pick `n` kept-bin indices uniformly with replacement, then pick one event uniformly from each picked bin.

**Pattern 2 — `_draw_mixed_density(rng, n)`**:
1. `focus_idx = rng.integers(0, n_kept)`; `E_focus = self.bin_centers[focus_idx]`.
2. `half_w = self.zoom_window_width_kev / 2`.
3. `local_bins = np.flatnonzero(np.abs(self.bin_centers - E_focus) <= half_w + 0.5 * self.energy_bin_width)`. (Half-bin pad so partial-overlap bins are included.)
4. `n_local = round(n * self.local_event_fraction)`; `n_global = n - n_local`.
5. Draw `n_local` events bin-stratified within `local_bins`; `n_global` events flat-stratified.
6. Concatenate + shuffle so the encoder sees a permutation-invariant set.

**Pattern 3 — `_draw_random_clusters(rng, n)`**:
1. Pick `self.n_clusters` non-overlapping focus energies via rejection sampling: draw a candidate from `kept_bin_indices`, reject if its `±zoom_window_width_kev/2` window overlaps any previously-accepted window. Cap rejections at 50; if can't fit `n_clusters`, raise (config error).
2. Build the union of in-window bins across all clusters.
3. Split `n` event slots evenly across clusters (`n // n_clusters`, distribute remainder round-robin).
4. Within each cluster, draw events bin-stratified from its own bin set.
5. Concatenate + shuffle.

**Pattern 4 — `_draw_physics_anchored(rng, n)`**: identical to `_draw_mixed_density` except `E_focus = rng.choice(self.physics_peaks_kev)` (already filtered to in-range at config time). Window/bin selection / split / draw all reuse the same code path.

**Tests** (`tests/cut_acceptance/test_event_sampler.py`) — add a fixture-parameterized matrix:
* For each (size_strategy, pattern) pair: emit a large batch, verify shape, mode, phi range, label dtype, reproducibility.
* `mixed_density`: with `local_event_fraction=0.7` and a focus near 1500 keV, ≥60% of phi[:,:,0] events lie within `±half_w` of focus. (Loose bound — gives RNG slack.)
* `random_clusters` with `n_clusters=2`, `W=50`: every emitted event falls inside one of the two picked windows; zero events lie outside both.
* `physics_anchored`: every trial's focus is within `±half_w` of some peak in `physics_peaks_kev`.
* `variable_uniform`: not a sampler-level concern (handled in the training loop). Skipped here; covered by a pipeline test in Phase 2.

### Phase 2 — Pipeline wiring + dynamic paths + smoke

**Auto-derived `out_dir` and `name`** — when the user leaves `out_dir` / `name` blank in the YAML, the pipeline derives them from a canonical pattern serializer:

```python
def paradigm_path_suffix(cfg) -> str:
    """Returns e.g. 'true_cnp' or 'hybrid_scale/mixed_density_f0p70_w50'."""
    if cfg.sampling_pattern == "flat_stratified" and cfg.trial_size_strategy == "fixed":
        return "true_cnp"
    # Otherwise sit under hybrid_scale/<pattern>_<serialized_params>/
    bits = [cfg.sampling_pattern]
    if cfg.sampling_pattern in ("mixed_density", "physics_anchored"):
        bits.append(f"f{cfg.local_event_fraction:.2f}".replace(".", "p"))
        bits.append(f"w{int(cfg.zoom_window_width_kev)}")
    elif cfg.sampling_pattern == "random_clusters":
        bits.append(f"n{cfg.n_clusters}")
        bits.append(f"w{int(cfg.zoom_window_width_kev)}")
    if cfg.trial_size_strategy == "variable_uniform":
        bits.append(f"varN{cfg.n_trial_events_min}-{cfg.n_trial_events_max}")
    return f"hybrid_scale/{'_'.join(bits)}"
```

Examples produced:
* baseline → `true_cnp/`
* `mixed_density f=0.70 w=50` → `hybrid_scale/mixed_density_f0p70_w50/`
* `random_clusters n=2 w=50 varN 16-64` → `hybrid_scale/random_clusters_n2_w50_varN16-64/`

`CutAcceptanceConfig` makes `out_dir` and `name` **optional**. The pipeline resolves them like:
```python
paradigm = paradigm_path_suffix(cfg)
out_dir = cfg.out_dir or Path(f"results/cut_acceptance/{model}/{paradigm}/bin{bw}/{cls}")
name = cfg.name or f"{model}_bin{bw}_{cls}__{paradigm.replace('/', '__')}"
```
Existing YAMLs that have explicit `out_dir` / `name` keep working unchanged (the True-CNP YAMLs do).

**Training loop wrapper** — when `trial_size_strategy == "variable_uniform"`, `pipeline.py` uses a local wrapper `train_cnp_variable_n_per_step` that mimics `train_cnp` but draws `n_events` fresh from `Uniform[n_trial_events_min, n_trial_events_max]` each step and clamps `n_ctx_max ≤ n_events - 1` before calling `split_context_target`. When `fixed`, the stock `train_cnp` is used (no behavior change for the 9 existing cells).

**Smoke cell**: write a new YAML at `configs/cut_acceptance/simple_cnn_small/hybrid_scale/mixed_density_f0p70_w50/bin10/signal.yaml` that sets:
```yaml
sampling_pattern: mixed_density
zoom_window_width_kev: 50.0
local_event_fraction: 0.70
```
Train it end-to-end. Verify:
* `out_dir` lands at `results/cut_acceptance/simple_cnn_small/hybrid_scale/mixed_density_f0p70_w50/bin10/signal/`.
* `cnp.ckpt` is reachable from the diagnostic script.
* `run_summary.json` carries `upstream_classifier_sha256` + the resolved paradigm string.

### Phase 3 — Evaluation sweep + audit

* Add a YAML at `hybrid_scale/random_clusters_n2_w50/bin10/signal.yaml` as a second variant.
* Sweep both with `run_all_test_inference.py` (still uses rglob → picks up the deeper tree automatically).
* Visually verify on the audit PNG:
  * `mixed_density` red curve shows sharper local features near the focus regions vs the `true_cnp` baseline — should "see" peaks better.
  * `random_clusters` shows wider `σ_CNP` band in the out-of-cluster regions (no D_C → honest uncertainty).
* Coverage plot (combined-σ pulls vs N(0,1)) should remain near calibration target.

### Files touched (sketch)

| File | Phase | Net change |
|---|---|---|
| `majorana_acp/cut_acceptance/config.py` | 1 | New fields + validators; `out_dir`/`name` become optional |
| `majorana_acp/cut_acceptance/event_sampler.py` | 1 | Pattern dispatch + 4 private draw helpers |
| `tests/cut_acceptance/test_event_sampler.py` | 1 | Parameterized matrix, ~6 new tests |
| `majorana_acp/cut_acceptance/pipeline.py` | 2 | `paradigm_path_suffix` resolver, optional `out_dir/name` derivation, `train_cnp_variable_n_per_step` wrapper |
| `configs/cut_acceptance/simple_cnn_small/hybrid_scale/mixed_density_f0p70_w50/bin10/signal.yaml` | 2 | New (smoke) |
| `configs/cut_acceptance/simple_cnn_small/hybrid_scale/random_clusters_n2_w50/bin10/signal.yaml` | 3 | New (sweep variant) |
| `tests/cut_acceptance/test_pipeline.py` | 2 | A `variable_uniform` smoke pass + asserts on derived path |

Estimated effort: ~45 min code + 6 min two-cell training + 5 min audit.

### Open questions

1. **Per-step vs per-trial variable N**: confirm per-step (within-batch N is uniform) is acceptable, or do you want me to plan the RESUM_FLEX padding-mask extension as a separate workstream?
2. **`out_dir` / `name` derivation**: confirm "auto-derive when blank; keep YAML-explicit value when provided". Or do you want strict auto-derive (ignore YAML's `out_dir` and `name` entirely)?
3. **Pattern serialization in path**: `mixed_density_f0p70_w50` (`f0p70` = `0.70` with the dot replaced) — readable? Or prefer `f0_70`, `f70pct`, or numeric `f70_w50`?
4. **`random_clusters` failure mode**: if rejection sampling can't fit `n_clusters` non-overlapping windows of the requested width (e.g., `n_clusters=8`, `W=500` keV in a 2500-keV range), raise a config error vs. silently fall back to overlapping clusters. I lean **raise** — clearer.
5. **Validation set lineage**: the `variable_uniform` setting affects training only; inference still uses the same global-aggregation D_C subsampling. Confirm this is the intended decoupling.

### Commit plan

| # | Commit | Scope |
|---|---|---|
| 1 | `Decoupled trial-size and sampling-pattern config knobs` | config.py + validators + test_config additions |
| 2 | `Hybrid sampling patterns in EventSampler` | event_sampler.py dispatch + 4 helpers + test_event_sampler matrix |
| 3 | `Auto-derive out_dir + name from paradigm; variable-N wrapper` | pipeline.py (path resolver + variable-N loop) |
| 4 | `Smoke YAML: hybrid_scale/mixed_density_f0p70_w50/bin10/signal` | one new YAML + trained run_summary.json |
| 5 | `Sweep variant: hybrid_scale/random_clusters_n2_w50/bin10/signal` | second new YAML + trained run_summary.json |
| 6 | (notebook update — uncommitted per standing rule) | §8.4 paradigm-suffix display |

---

## Cut-Acceptance · Version-Control & Lineage Refactor Plan (PROPOSED)

### Why now
The True-CNP rollout is shipped and exercised across 9 cells. Before any
hybrid-scale evolution, close four naming / lineage gaps that would
otherwise compound into a multi-paradigm mess:

| # | Gap | Fix in this refactor |
|---|---|---|
| 1 | Paradigm invisible in config names / paths | Sibling-tree under `<paradigm>/` |
| 2 | No explicit link from cut-acceptance to its upstream classifier | Mandatory `upstream_classifier_config` + recorded SHA256 |
| 3 | Trained artifacts live in gitignored `results/`; no fingerprint in git | Track `run_summary.json` via a `.gitignore` exception |
| 4 | Notebook hard-codes paradigm | New `CURRENT_PARADIGM` toggle |

### Phase 1 — Configuration migration (sibling tree)

* New layout:
  `configs/cut_acceptance/<model>/<paradigm>/bin{5,10,20}/{signal,background,inclusive}.yaml`
* Operation:
  ```
  mkdir -p configs/cut_acceptance/simple_cnn_small/true_cnp
  git mv configs/cut_acceptance/simple_cnn_small/bin5   configs/cut_acceptance/simple_cnn_small/true_cnp/
  git mv configs/cut_acceptance/simple_cnn_small/bin10  configs/cut_acceptance/simple_cnn_small/true_cnp/
  git mv configs/cut_acceptance/simple_cnn_small/bin20  configs/cut_acceptance/simple_cnn_small/true_cnp/
  ```
* Inside each of the 9 migrated YAMLs, edit:
  * `name:` append `__true_cnp` suffix (e.g. `simple_cnn_small_bin10_signal__true_cnp`).
  * `out_dir:` mirror new tree → `results/cut_acceptance/simple_cnn_small/true_cnp/bin{N}/{class}`.
* `run_all_test_inference.py` already uses `rglob` over `configs/cut_acceptance/`, so it will pick up the new tree automatically — but double-check the `OUT_ROOT / rel.parent / rel.stem` calculation lands at the expected `analysis/cnp_audit/simple_cnn_small/true_cnp/...`.

### Phase 2 — Upstream classifier lineage

* `CutAcceptanceConfig` gains:
  ```python
  upstream_classifier_config: Path = Field(
      ...,
      description=(
          "Path (relative to repo root) to the YAML that trained the "
          "classifier whose predictions.h5 outputs we score against. "
          "Mandatory — the SHA256 of this file is recorded in "
          "run_summary.json so each cut-acceptance run is bound to "
          "an exact upstream classifier state."
      ),
  )
  ```
* No existence validation at config-load time (so tests don't need to stub the upstream YAML on disk just to parse). Existence is asserted at pipeline-execution time, when we open it to hash.
* All 9 YAMLs add:
  `upstream_classifier_config: configs/small_data_configs/simple_cnn_small.yaml`
* `pipeline.py` additions inside `run_pipeline`:
  * Compute SHA256 of `cfg.upstream_classifier_config`.
  * Store both the path string and the hash in `run_summary.json` as
    `upstream_classifier_config` and `upstream_classifier_sha256`.
* `tests/cut_acceptance/test_pipeline.py`: smoke test now writes a stub upstream YAML to `tmp_path` and passes its path through `_fast_cfg`.
* `tests/cut_acceptance/test_config.py`: add a case asserting loading fails when `upstream_classifier_config` is missing.

### Phase 3 — Lightweight checkpoint fingerprinting

Goal: `run_summary.json` becomes the version-controlled "what trained at this path, when, against which upstream" record. We don't try to commit the `cnp.ckpt` itself.

* `.gitignore` exception (idiomatic git form):
  ```
  results/*
  !results/**/
  !results/**/run_summary.json
  ```
  (The current `results/` rule is replaced by these three lines. Smoke-test with `git check-ignore -v results/cut_acceptance/.../cnp.ckpt` (ignored) and `…/run_summary.json` (tracked).)
* Extend `PipelineSummary` to include `upstream_classifier_config`, `upstream_classifier_sha256`, and (cheap) the training-data-subset descriptor — anything that uniquely identifies the run.
* Update the test_pipeline assertions to verify these fields are present in the saved JSON.

### Phase 4 — Notebook paradigm toggle

Cell 39:

* Add `CURRENT_PARADIGM = "true_cnp"` next to `CURRENT_MODEL`.
* Path construction in `get_inference`:
  ```python
  cfg_path = (
      CFG_ROOT / model / CURRENT_PARADIGM / f"bin{bin_width}" / f"{target_class}.yaml"
  )
  ```
* Include `CURRENT_PARADIGM` in the `_inference_cache` key so swapping the toggle doesn't return a stale result.
* Update the suptitle strings in §8.4.{2,3,4} cells to show the active paradigm.

### Verification

Per-phase: run `pytest tests/cut_acceptance/ -q`. End-to-end:

1. Retrain one cell (`true_cnp/bin10/signal`) and confirm:
   * `out_dir` lands at `results/cut_acceptance/simple_cnn_small/true_cnp/bin10/signal/`.
   * `run_summary.json` contains the new lineage fields.
   * `git status` shows that single `run_summary.json` as a trackable new file (and `cnp.ckpt` as still ignored).
2. Full retrain + sweep (~5 min training + ~1 min inference): all 9 cells produce True-CNP outputs at the new paths.
3. Open notebook, set `CURRENT_PARADIGM = "true_cnp"`, re-run §8.4 — figures match the regenerated PNGs.

### Commit plan

One commit per phase, then a verification-output commit:

1. `Sibling-tree layout: configs/.../<paradigm>/bin*/...`
2. `Bind cut-acceptance to upstream classifier via SHA256`
3. `Track run_summary.json as the lightweight experiment registry`
4. `Notebook: CURRENT_PARADIGM toggle for paradigm-aware paths`
5. `Refresh 9 run_summary.json after True-CNP retraining`  *(adds the now-tracked JSONs)*

### Files touched

| File | Phase |
|---|---|
| `configs/cut_acceptance/simple_cnn_small/bin{5,10,20}/*.yaml` (moved + edited) | 1, 2 |
| `majorana_acp/cut_acceptance/config.py` | 2 |
| `majorana_acp/cut_acceptance/pipeline.py` | 2, 3 |
| `tests/cut_acceptance/test_config.py` | 2 |
| `tests/cut_acceptance/test_pipeline.py` | 2, 3 |
| `.gitignore` | 3 |
| `notebooks/data_visualization.ipynb` | 4 |

### Estimated effort

~25 min code + 6 min retraining + sweep + 5 min verification. No upstream library changes.

### Open questions

1. **Paradigm name**: `true_cnp` per your spec. Confirm the suffix `__true_cnp` (double underscore) is what you want in `name:`; or single underscore `_true_cnp`?
2. **Old `results/cut_acceptance/simple_cnn_small/bin{N}/`** directories will be left as orphans after migration (gitignored, harmless). Do you want me to `rm -rf` them at the end, or leave them?
3. **Backfill or retrain?** Migration moves config paths; the existing checkpoints sit at the *old* `results/.../bin{N}/` paths and could be `mv`'d to the new paths. But their `run_summary.json` won't have the new lineage fields. Cleanest is retrain. ~5 min cost.

---

## Cut-Acceptance · True CNP (Per-Event Coordinates) Plan (IMPLEMENTED 2026-05-13)

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

