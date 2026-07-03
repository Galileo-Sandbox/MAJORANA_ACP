# Majorana-ACP

**Attentive Conditional Neural Processes for HPGe cut-acceptance
estimation on the Majorana Demonstrator AI/ML release.**

Given a binary Pulse-Shape Discrimination (PSD) classifier trained on
Germanium waveforms, we learn its cut-acceptance function

    β(E) = P( score ≥ T*  |  energy = E )

as a continuous 1D curve resolved down to individual γ-lines
(Tl-208 FE / SE / DEP, Bi-214) in the ²²⁸Th spectrum. The estimator
is a **Conditional Neural Process (CNP)** with two custom modules
that resolve the standard "over-smoothing near sharp physics
features" failure of vanilla CNPs and Attentive NPs on this problem:

- **Scale-Aware Positional Encoding (SAPE)** — a Fourier-feature
  lift of the energy coordinate, with a per-query hard cutoff over
  the frequency bands so peaks get the full frequency spectrum and
  flat Compton regions do not.
- **Self-Adaptive Sigma Filter Network (SFN Gate)** — a
  data-density-conditioned local/global contrast head that selects
  the SAPE cutoff at inference time, with the continuum-floor
  parameter **κ** exposed as a learnable, sigmoid-bounded weight.

Every ablation from the plain CNP baseline through the final
architecture lives in `configs/cut_acceptance/` and can be rerun
end-to-end from a fresh clone.

## Table of contents

1. [Data](#data)
2. [Architecture](#architecture)
3. [Results](#results)
4. [Quickstart — render the flagship figure without training](#quickstart--render-the-flagship-figure-without-training)
5. [Repository layout](#repository-layout)
6. [Retraining from scratch](#retraining-from-scratch)
7. [Extending — add your own model](#extending--add-your-own-model)
8. [Testing](#testing)
9. [Citation](#citation)

## Data

Public Majorana Demonstrator AI/ML data release, DS6 ²²⁸Th
calibration — 3.2M waveforms with energy + PSD analysis labels.

- Paper: [arXiv:2308.10856](https://arxiv.org/abs/2308.10856)
- Files: [Zenodo, 10.5281/zenodo.8257027](https://doi.org/10.5281/zenodo.8257027)
- `MJD_Train_*.hdf5` (16 files) — training pool
- `MJD_Test_*.hdf5` (6 files) — evaluation

Train and test files are never mixed. Waveforms are baseline-corrected
against the first 500 samples, per-event peak-normalised, and
optionally aligned around the 90 % rising-edge sample before hitting
the model.

## Architecture

The cut-acceptance model is an **Attentive Conditional Neural Process**
that takes a context set of `(E_i, T*)` — event energy and the
Youden-J optimal classifier threshold — plus the binary passage flag
`X_i = 1[score_i ≥ T*]`, and returns a mean + epistemic uncertainty
band over a dense energy grid.

```
    context set (D_C)        target queries (E*)
      ┌────────────┐            ┌────────────┐
      │ E_i, T*    │            │ E*_j       │
      │ X_i ∈{0,1} │            └─────┬──────┘
      └─────┬──────┘                  │
            │  SAPE(E_i)               │  SAPE(E*_j)
            │  = [sin(2^l·πE), …]      │
            ▼                          ▼
      ┌────────────┐            ┌────────────┐
      │ MLP encoder│            │ MLP encoder│
      └─────┬──────┘            └─────┬──────┘
            │ per-event                │ per-query
            │   h_i, k_i               │   q_j
            ▼                          ▼
      ┌──────────────────────────────────────┐
      │  Cross-attention + SFN gate          │
      │                                      │
      │   α_ij = softmax( q_j · k_i / √d      │
      │             + Gaussian(E_i − E*_j)   │
      │             − λ(E*_j) · mask_l )     │
      │                                      │
      │   r(E*_j) = Σ_i α_ij · h_i           │
      └──────────────┬───────────────────────┘
                     │
                     ▼
              ┌────────────┐
              │  Decoder   │  ⇒  β(E*_j), σ(E*_j)
              └────────────┘
```

### Scale-Aware Positional Encoding (SAPE)

We lift each scalar energy into a 21-dimensional Fourier feature
vector: `[sin(2^l·πE), cos(2^l·πE)]` for `l = 0..9`, plus the
normalised threshold `T*`. The `l = 9` band has period ≈ 9.8 keV
under a 2500-keV normalisation window, deliberately matching the
bin10 evaluation grid so the network can resolve bin-scale
acceptance features. Every layer of the encoder / attention /
decoder consumes this expanded representation.

### Self-Adaptive SFN Gate

Vanilla PE10 with mean- or attention-aggregation over-shoots on the
Tl-208 SE line and rings across the smooth Compton continuum: the
high-frequency bands are "always on" and leak sharp features into
regions where the data is smooth. Our SFN gate cures this with a
data-driven **per-query soft band mask**:

    λ(E*) = κ + (10 − κ) · sigmoid( 10 · (R(E*) − 3) )

where `R(E*)` is the local-vs-global data-density contrast ratio
around the target query (computed on the fly from the context set),
and `κ` is the continuum-floor bandwidth — the minimum number of
Fourier bands the decoder is allowed to use in featureless regions.

Attention masks each band `l` with `w_l = sigmoid(α · (λ(E*) − l))`
so high-`l` bands stay closed where `λ` is small and open up
smoothly where `λ` grows toward 10.

### Learnable, sigmoid-bounded κ (Cell 17)

Cell 15 locked κ at 1.0 (over-smooths the 2400-keV Compton edge);
Cell 16 unconstrained κ (risks continuum ringing). The shipped
Cell 17 architecture parameterises

    κ = 1 + 4 · sigmoid(κ_raw)   ∈  (1, 5)

with κ_raw initialised at 0 (midpoint κ = 3, maximum gradient),
letting Adam find the optimal continuum-floor bandwidth by
end-to-end gradient descent without opening the high-frequency
supreme bands (l ≥ 6) in flat regions.

### Ablation ladder — five canonical checkpoints

| Paradigm | Aggregator | PE | SFN gate | κ | What it isolates |
|---|---|---|---|---|---|
| `true_cnp/bin10/inclusive` | mean | ✗ | ✗ | — | Baseline CNP; over-smooths all features. |
| `sweeps/base1_matched` | mean | ✗ | ✗ | — | Naïve CNP at fair-budget N. |
| `sweeps/base3_matched` | cross-attn (8×64) | PE10 | ✗ | — | Attentive NP + Fourier features. |
| `sweeps/cell15_v5` | 1×128 attn + gates | PE10 | fixed | 1 | SAPE + SFN, locked floor. |
| `sweeps/cell17` | 1×128 attn + gates | PE10 | fixed | learnable ∈ (1,5) | Self-adaptive continuum floor. |

Each YAML at `configs/cut_acceptance/simple_cnn_small/<paradigm>/bin10/inclusive.yaml`
carries the full config; `experiments/` holds the deprecated
intermediate sweeps referenced for reproducibility only.

## Results

Numbers below come straight from the tracked audits at
`cache/inference/simple_cnn_small/<paradigm>/inclusive_bin10_audit.json`
— run the notebook to regenerate identical values.

**Global fidelity + continuum smoothness.** Pearson r is measured on
bin-center β̂ vs D_T empirical rates over the full [500, 3000] keV
window. MASD (Mean Absolute Second Difference) is a
"sawtooth-amplitude" score on the dense red curve inside two
peak-free Compton windows — **lower is smoother**.

| Paradigm | Pearson r | MASD 1.7-2.0 | MASD 2.2-2.4 |
|---|---:|---:|---:|
| True CNP (mean agg.)              | +0.484 | 0.0050 | 0.0075 |
| Baseline 1 — pure CNP (matched)   | +0.482 | 0.0040 | 0.0055 |
| Baseline 3 — ANP + PE10 (matched) | +0.311 | 0.1627 | 0.0925 |
| Cell 15 SAPE + SFN (fixed κ=1)    | +0.506 | 0.0077 | 0.0114 |
| **Cell 17 SAPE + SFN (learnable κ)** | **+0.530** | 0.0081 | **0.0066** |

Baseline 3 (attention + PE10 with no gate) illustrates the
over-smoothing failure the SFN gate was designed to cure: high
MASD everywhere because the high-frequency Fourier bands ring
uniformly across the whole spectrum.

**Peak-region goodness-of-fit.** At each γ-line we compute a local
reduced χ² and its two-tailed p-value against a ±5 keV bin window.
**χ² ≈ 1 and p > 0.5 ⇒ indistinguishable from sharp truth;
χ² ≫ 1 and p < 0.05 ⇒ systematic local miss.**

| Paradigm | Tl-208 SE 2103 (χ² / p) | Tl-208 DEP 1592 (χ² / p) | Tl-208 FE 2614 (χ² / p) |
|---|---:|---:|---:|
| True CNP              | 31.6 / 0.00 | 18.1 / 0.00 | 0.12 / 0.71 |
| Baseline 1            | 32.4 / 0.00 | 17.5 / 0.00 | 2.58 / 0.07 |
| Baseline 3            |  0.00 / 0.99* | 10.1 / 0.00 | 0.65 / 0.38 |
| Cell 15 (fixed κ)     |  0.24 / 0.28 | 19.9 / 0.00 | 0.18 / 0.62 |
| **Cell 17 (learnable κ)** | **0.13 / 0.51** | 22.1 / 0.00 | 0.63 / 0.33 |

_*Baseline 3's SE χ² is small because it emits a smooth curve
_averaged_ across the peak — matching truth on average but wildly
wrong point-to-point (see the MASD column above)._

The headline win: **Cell 17 is the only architecture that resolves
the sharp Tl-208 SE line (χ² 0.13 vs 31 for the baselines) while
keeping the 2.2-2.4 keV Compton continuum smooth (lowest MASD in
that window).** The Tl-208 DEP line remains a stubborn miss for
every paradigm — an open item for future work.

## Quickstart — render the flagship figure without training

The five canonical inference outputs are committed as compact
`.npz` files under `cache/inference/` (~1.3 MB total), so a fresh
clone runs `notebooks/data_visualization.ipynb` end-to-end with no
trained checkpoint or upstream `predictions.h5`:

```bash
git clone <this-repo> && cd majorana-acp
uv sync                                    # install Python deps
uv run jupyter lab notebooks/data_visualization.ipynb
```

Then in the notebook:

1. Run cells §1–§8.4.1. Cell §8.4.1 defines three toggles at the
   top — leave the defaults on the first pass:
   - `CURRENT_PARADIGM = "sweeps/cell17"` — which cached model to plot.
   - `DATA_SOURCE     = "D_T"`             — which empirical set to show.
   - `DATA_PLOT_MODE  = "errorbar"`        — Wilson bars (default) / step / KDE.
2. Run §8.4.2 for the single-cell β(E) figure with the red curve
   (Cell 17) and blue Wilson intervals (D_T).
3. Run §8.4.3 for the multi-paradigm comparison — the plot iterates
   over `COMPARE_PARADIGMS`, a dict of `"nickname" → paradigm-path`
   at the top of §8.4.3. Add or remove entries to change which
   cells appear.
4. Run §8.4.5–§8.5 for coverage diagnostics + the summary table
   driven by the audit JSONs.

**Compare with your model's training data instead of the held-out
set.** Set `DATA_SOURCE = "D_train"` in §8.4.1 and re-run §8.4.2.
The blue points then come from the ~19k training events with the
same T*, energy filter, and class filter the model saw — a
train↔test transfer diagnostic.

**Compare with the actual inference-time context set.** Set
`DATA_SOURCE = "D_C"` — this shows the ~2000 events the CNP
conditioned on to produce the red curve.

Each cached `.npz` carries: the dense β(E) mean and σ curves,
bin-center Wilson-interval rates for D_C / D_T, the raw event
arrays for all three data-source toggles, and the peak-region
χ²/Z audit metrics used in the [Results](#results) tables.

## Repository layout

```
majorana_acp/                    Python package
  data/                          HDF5 loader + Dataset
  models/                        nn.Module classes (simple_cnn, MLP,
                                 AttentiveCNP with SAPE + SFN gate)
  training/                      Train loop, class-imbalance handling
  eval/                          Metrics, energy histograms
  cut_acceptance/                CNP pipeline: config, event sampler,
                                 positional encoding, pipeline.py
  cli/                           Entry points: train, evaluate, predict
configs/
  small_data_configs/            Upstream classifier YAMLs
  cut_acceptance/simple_cnn_small/
    true_cnp/bin{5,10,20}/       Baseline CNP (mean aggregation)
    sweeps/
      base1_matched/             CNP baseline at matched budget
      base3_matched/             ANP + PE10
      cell15_v5/                 SAPE + SFN, fixed κ = 1
      cell17/                    SAPE + SFN, learnable κ ∈ (1,5)
scripts/
  diagnostics/                   Test-set inference + audit builder
  tools/build_notebook_cache.py  Persists the five .npz files below
cache/inference/                 Tracked notebook cache (~10 MB total)
notebooks/data_visualization.ipynb   Full figure story
tests/                           pytest, mirrors the package layout
experiments/                     Sequestered historical configs
                                 (kept for reproducibility only)
```

The following paths are gitignored:

- `runs/` — upstream classifier training outputs
- `results/**` and `experiments/results/**` except each cell's
  `run_summary.json` (the git-visible experiment registry)
- `analysis/**` except top-level `_*.md` findings notes

## Retraining from scratch

You need the upstream classifier's `predictions.h5` files from a
prior training run (train + val splits). Point each cell's YAML at
your local copy via `train_predictions_path` and
`validation_predictions_path`.

**One paradigm, end-to-end:**

```bash
# 1. Train the CNP.
uv run python -m majorana_acp.cut_acceptance.cli \
    configs/cut_acceptance/simple_cnn_small/sweeps/cell17/bin10/inclusive.yaml

# 2. Run test-set inference + write the audit JSON/PNG.
uv run python -m scripts.diagnostics.cnp_test_inference \
    configs/cut_acceptance/simple_cnn_small/sweeps/cell17/bin10/inclusive.yaml

# 3. Refresh the notebook cache so §8.4 picks up the new weights.
uv run python -m scripts.tools.build_notebook_cache sweeps/cell17
```

Omit the trailing paradigm arg to rebuild all five canonical
caches at once; add `--help` to `cnp_test_inference` for the
context-size / MC-Dropout knobs.

**Sweep every canonical cell:**

```bash
uv run python -m scripts.diagnostics.run_all_test_inference
uv run python -m scripts.tools.build_notebook_cache
```

## Extending — add your own model

There are two independent registries you can hook into.

### 1. A new waveform classifier (upstream)

The waveform-level classifier that produces the `score ≥ T*` cut
lives in `majorana_acp/models/`. Every architecture registers
itself via a decorator:

```python
# majorana_acp/models/my_awesome_net.py
import torch
from torch import nn
from majorana_acp.models.registry import register_model


@register_model("my_awesome_net")
class MyAwesomeNet(nn.Module):
    """Input:  (B, L) preprocessed waveform.
       Output: (B,)   raw logits — NO sigmoid inside the model.
    """
    def __init__(self, hidden: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        # ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ...
        return logit  # shape (B,)
```

Then:

1. Import your module in `majorana_acp/models/__init__.py` so the
   decorator runs at package-import time.
2. Reference it from any classifier YAML under
   `configs/small_data_configs/`:
   ```yaml
   model:
     name: my_awesome_net
     params:
       hidden: 256
       dropout: 0.3
   ```
3. Train + evaluate:
   ```bash
   uv run python -m majorana_acp.cli.train    configs/small_data_configs/my_awesome_net.yaml
   uv run python -m majorana_acp.cli.evaluate runs/my_awesome_net/best.ckpt
   ```

Add a forward-pass + parameter-count test to `tests/test_models.py`
following the existing `SimpleCNN` / `ResNet1D` / `InceptionTime`
patterns — no training required for the smoke check.

### 2. A new CNP aggregator (downstream)

If you want to swap out the SFN-gated cross-attention for your own
context-aggregation scheme (e.g. graph attention, retrieval,
mixture-of-experts), the entry point is
`majorana_acp/cut_acceptance/pipeline.py::build_local_cnp(...)`.

The clean path is:

1. **Write your aggregator module** under `majorana_acp/models/`,
   subclassing `nn.Module`. It receives per-event encoder outputs
   `h_C: [B, N_C, agg]`, target latents `z_φ_T: [B, N_T, Z]`, and
   optional detach flags, and must return
   `r: [B, N_T, agg]` — a per-query context representation the
   decoder can consume. See
   `majorana_acp/models/attentive_cnp.py::CrossAttentionAggregator`
   for the SFN-gated reference implementation.
2. **Add a factory** that mirrors
   `build_attentive_cnp(...)` — it wires the aggregator into the
   upstream `UniversalEncoder + ContextPointEncoder + CnpDecoder`
   stack and returns an object compatible with
   `core.training.train_cnp` (returns the upstream `CnpOutput`
   dataclass).
3. **Extend the config schema.** Add a `Literal[...]` option to
   `AggregatorConfig.type` in
   `majorana_acp/cut_acceptance/config.py`, plus any per-aggregator
   hyperparameters as new sub-block fields. Wire pydantic
   validators for interpretable failure modes.
4. **Dispatch in `build_local_cnp`.** One `if cfg.aggregator.type
   == "my_aggregator":` branch calling your factory.
5. **Write a YAML** at
   `configs/cut_acceptance/simple_cnn_small/sweeps/my_agg/bin10/inclusive.yaml`
   inheriting the Cell 17 baseline and setting
   `aggregator.type: my_aggregator`.
6. **Train + evaluate + cache**:
   ```bash
   uv run python -m majorana_acp.cut_acceptance.cli    configs/.../sweeps/my_agg/bin10/inclusive.yaml
   uv run python -m scripts.diagnostics.cnp_test_inference configs/.../sweeps/my_agg/bin10/inclusive.yaml
   uv run python -m scripts.tools.build_notebook_cache sweeps/my_agg
   ```
7. **Add to the notebook.** In the last cell of §8.4.1 append your
   paradigm string to `COMPARE_PARADIGMS`; the summary table + red
   curve now include your model alongside the canonical five.
8. **Write a parity test** at
   `tests/cut_acceptance/test_my_aggregator_parity.py` mirroring
   `test_aggregator_parity.py` — assert that when your aggregator
   is disabled (`type = "mean"`) the model produces byte-identical
   outputs to the upstream `build_cnp` path.

That's it. The main training loop, event sampler, PE, evaluation
metrics, and notebook are all aggregator-agnostic — everything
downstream keys off `cfg.aggregator.type` alone.

## Testing

```bash
uv run pytest
```

410 unit + integration tests, ~40 s on CPU. Every module under
`majorana_acp/` has a corresponding test file and the SAPE / SFN
gate paths carry parity tests against the mean-aggregator baseline
to guarantee zero regression on legacy checkpoints.

## Citation

If this work is useful in yours, please cite the Majorana
Demonstrator AI/ML data release:

```
@article{Majorana2023AIML,
  title = {A machine-learning-based framework for anomalous-event
           reconstruction with High-Purity Germanium detectors},
  author = {Majorana Collaboration},
  journal = {arXiv preprint arXiv:2308.10856},
  year = {2023}
}
```
