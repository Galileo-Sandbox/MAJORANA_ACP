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
3. [Quickstart — reproduce the flagship figure](#quickstart--reproduce-the-flagship-figure)
4. [Repository layout](#repository-layout)
5. [Retraining from scratch](#retraining-from-scratch)
6. [Testing](#testing)
7. [Citation](#citation)

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

## Quickstart — reproduce the flagship figure

**Zero-training path.** The five canonical inference outputs are
committed as compact `.npz` files under `cache/inference/`, so a
fresh clone renders `notebooks/data_visualization.ipynb` §8.4
without any trained checkpoint or upstream `predictions.h5`:

```bash
git clone <this-repo>
cd majorana-acp
uv sync                                    # install Python deps
uv run jupyter lab notebooks/data_visualization.ipynb
# In §8.4.1: leave CURRENT_PARADIGM = "sweeps/cell17"
# Run cells §8.4.2, §8.4.3, §8.5 top-to-bottom.
```

Each cached `.npz` is < 2 MB and carries: the dense β(E) mean and
σ curves, bin-center Wilson-interval empirical rates for D_C /
D_T, the raw event arrays for the three data-source toggles
(D_T / D_C / D_train), and the peak-region χ²/Z audit metrics.

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

Train one paradigm:

```bash
uv run python -m majorana_acp.cut_acceptance.cli \
    configs/cut_acceptance/simple_cnn_small/sweeps/cell17/bin10/inclusive.yaml
```

Run test-set inference + audit:

```bash
uv run python -m scripts.diagnostics.cnp_test_inference \
    configs/cut_acceptance/simple_cnn_small/sweeps/cell17/bin10/inclusive.yaml
```

Rebuild the notebook cache after retraining:

```bash
uv run python -m scripts.tools.build_notebook_cache
```

Sweep every canonical cell:

```bash
uv run python -m scripts.diagnostics.run_all_test_inference
```

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
