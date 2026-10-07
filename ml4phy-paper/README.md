# ML4PS 2026 paper artifact

This directory is the reproducibility package for
[Density-Guided Conditional Neural Processes for Detector Efficiency Estimation](https://arxiv.org/abs/2609.13593)
by Yue Ma and Aobo Li.

The paper asks whether limited calibration data can support an efficiency curve
that recovers narrow physical changes without following every finite-sample
fluctuation. It compares CNP, deterministic Attentive CNP, Attentive CNP with
Fourier positional encoding, the density-guided CNP, Gaussian kernel
regression, and a Bernoulli Gaussian process. It also tests training budgets and
controlled replacements of the density-dependent mechanisms.

This artifact contains the exact lightweight evidence used for the paper:
frozen protocols, configurations, executed scripts, tables, figures, reports,
tests, and source/output hashes. Raw waveforms, checkpoints, role identities,
and event-level predictions remain external.

## Paper-to-artifact map

| Paper item | Claim or definition | Primary artifact |
| --- | --- | --- |
| Figure 1 | Efficiency curves for four neural architectures at 5k training and 500 context events | [`figures/paper_figure1_efficiency_curves.png`](figures/paper_figure1_efficiency_curves.png), [`tables/paper_mean_grid_curves.csv`](tables/paper_mean_grid_curves.csv) |
| Table 1 | Overall, four peak cores, and continuum C2 agreement for neural and classical methods | [`tables/paper_coverage_summary.csv`](tables/paper_coverage_summary.csv), [`reports/paper_coverage_export.md`](reports/paper_coverage_export.md) |
| Table 2 | C1/C2/C3 across 2k, 5k, and 10k efficiency-training budgets | [`tables/paper_coverage_budget_comparison.csv`](tables/paper_coverage_budget_comparison.csv), [`tables/paper_coverage_cross_budget_pairs.csv`](tables/paper_coverage_cross_budget_pairs.csv) |
| Table 3 | Density-mechanism controls on the original 5k subset | [`review_followup_20260910/tables/mechanism_summary.csv`](review_followup_20260910/tables/mechanism_summary.csv), [`review_followup_20260910/reports/mechanism_result.md`](review_followup_20260910/reports/mechanism_result.md) |
| Table 4 | Density-guided architecture dimensions | [`configs/cell17_controlled_seed0.yaml`](configs/cell17_controlled_seed0.yaml), [`review_followup_20260910/tables/mechanism_checkpoint_inventory.csv`](review_followup_20260910/tables/mechanism_checkpoint_inventory.csv) |
| Table 5 | Fixed density, attention, and frequency-gate constants | [`review_followup_20260910/configs/protocol_v1.json`](review_followup_20260910/configs/protocol_v1.json), [`review_followup_20260910/control_models.py`](review_followup_20260910/control_models.py) |
| Table 6 | Unique event budgets and role allocation | [`reports/data_budget_ledger.md`](reports/data_budget_ledger.md), [`tables/data_budget_ledger.csv`](tables/data_budget_ledger.csv), [`tables/data_identity_overlap.csv`](tables/data_identity_overlap.csv) |
| Table 7 | C1/C2/C3 threshold sensitivity | [`tables/paper_coverage_budget_comparison.csv`](tables/paper_coverage_budget_comparison.csv), [`review_followup_20260910/tables/mechanism_full_region_sensitivity.csv`](review_followup_20260910/tables/mechanism_full_region_sensitivity.csv) |
| Supplement B.2 | Kernel bandwidth and Bernoulli-GP selection | [`tables/phase1_kernel_bandwidth_selection.csv`](tables/phase1_kernel_bandwidth_selection.csv), [`tables/phase3_gp_development_selection.csv`](tables/phase3_gp_development_selection.csv) |
| Supplement B.3 | Supported bins and Wilson reference bands | [`tables/paper_coverage_region_bins.csv`](tables/paper_coverage_region_bins.csv), [`tables/paper_reference_bin_predictions.csv`](tables/paper_reference_bin_predictions.csv) |
| Supplement B.5 | Count-scale, continuous-error, and sparse-tail interpretation | [`review_followup_20260910/reports/existing_measurement_interpretation.md`](review_followup_20260910/reports/existing_measurement_interpretation.md), [`review_followup_20260910/tables/mechanism_regional_cells.csv.gz`](review_followup_20260910/tables/mechanism_regional_cells.csv.gz) |

## What was executed

### Frozen measurement protocol

- Classifier source: 377,330 training-split events in 500--3000 keV.
- Fixed classifier training subset: 18,866 unique events.
- Fixed threshold: `0.540643572807312`, selected from 2,000 calibration
  events.
- Final context reservoir: 20,000 events, with ten frozen 500-event draws.
- Final reference: 114,400 events, disjoint from training and context.
- Evaluation bins: 5 keV over 500--3000 keV, requiring at least four reference
  events.

The full role construction and identity hashes are in
[`manifests/frozen_protocol_v1.json`](manifests/frozen_protocol_v1.json) and
[`manifests/extension_protocol_v1.json`](manifests/extension_protocol_v1.json).

### Architecture comparison

Four neural architectures use matched 3,000-step training and the same context
draws:

1. **CNP**: mean pooling, no positional encoding.
2. **Attentive CNP**: deterministic attention, no positional encoding.
3. **Attentive CNP + PE**: attention with ten Fourier frequency levels.
4. **Density-guided CNP**: adaptive decoder frequency gating,
   density-conditioned attention, and direct density contrast.

The attentive comparator is not presented as a complete latent ANP. Kernel and
Bernoulli-GP methods use context only and therefore do not have neural-model
pretraining information. The pooled-kernel control uses the full 18,866-event
training pool and is labeled as a stronger-data comparison.

### Training-budget campaign

The original 2k, 5k, and 10k budgets are nested prefixes of one frozen
outcome-blind ordering. Two additional outcome-blind orderings repeat the 5k
comparison. Each neural setting uses three initialization seeds and ten shared
contexts. Phase 2 and Phase 3 contain 600 saved neural evaluation cells in
total.

The primary result is conditional efficiency-model data efficiency: at 5k,
the density-guided model exceeds 10k CNP and Attentive CNP on both peak and
continuum C2. It does not exceed 10k PE at peaks, 2k fails, and 5k-to-10k
performance is nonmonotonic.

### Classical baselines

The kernel estimator is the Gaussian Nadaraya-Watson estimator, equivalent to
a common-bandwidth KDE ratio only when the passing fraction is included. Its
bandwidth candidates are 2, 5, 10, 20, 50, and 100 keV and are selected on
development data.

The GP is an event-level Bernoulli probability model with a logistic likelihood
and Laplace approximation. Development Brier selects Matérn-3/2 over RBF. The
completed campaign contains 80 development fits and 40 final fits. Recorded
convergence warnings remain in the result manifest.

### Mechanism campaign

The one-subset 5k mechanism slice contains 12 new training jobs and 120 new
evaluations, plus 30 reused full-model cells. The controls replace the decoder
gate, attention density inputs, or the direct density input while keeping the
rest of the architecture and training protocol fixed.

The clearest evidence supports density-adaptive decoder gating. Learned global
attention performs nearly identically to adaptive attention on the primary
summaries, so the paper does not claim a necessary independent benefit from
that pathway. An initial checkpoint-loader error affected 90 evaluations;
those outputs were excluded, the loader received a regression test, and every
affected cell was rerun.

## Metrics and interpretation

For supported reference bin `b`, predictions are averaged at the actual event
energies before comparison with the measured passing fraction. `C_k` is the
fraction of bins within `k` times the half-width of the `z = 1` Wilson reference
interval.

The paper reports:

- Overall C1/C2/C3 over 442 supported bins.
- Four individual two-bin peak cores and their equal-feature mean.
- Two continuum windows and their equal-window mean.
- Per-bin pulls, residual histograms, bias, centered RMS, and pull RMS.
- Event-weighted absolute discrepancy and predicted-minus-observed passing
  counts.
- Separate subset, initialization, and context variation.

These quantities compare predictions with a finite calibration reference. They
are not calibrated confidence coverage, known physical bias, or independent
replication across the overlapping contexts and subsets. The artifact does not
invent bin-mean model uncertainty from pointwise dropout standard deviations.

## Artifact directory

| Path | Contents |
| --- | --- |
| [`configs/`](configs/) | Resolved neural, GP, MC-diagnostic, and budget-campaign configurations. |
| [`manifests/`](manifests/) | Frozen protocols, environments, source/output hashes, campaign inventories, and release anchors. |
| [`scripts/`](scripts/) | Protocol builders, training/evaluation runners, aggregators, exporters, and plotters. |
| [`tables/`](tables/) | Full-precision portable metrics and curve summaries. |
| [`figures/`](figures/) | Paper-facing and diagnostic visualizations. |
| [`reports/`](reports/) | Audit decisions, protocol freezes, results, warnings, and claim boundaries. |
| [`tests/`](tests/) | Presentation and coverage-export tests. |
| [`review_followup_20260910/`](review_followup_20260910/) | Additive mechanism implementation, tests, results, and compressed per-bin predictions. |
| [`artifact/`](artifact/) | Portable integrity verification and external-dependency restoration tools. |

Initial plans and approval gates remain tracked because they distinguish
prospective choices from later observations. Result manifests and final reports
record what was actually executed.

## Verification levels

### Portable result verification

A Git clone is sufficient to verify tracked hashes, inventories, and numerical
summaries:

```bash
uv sync --frozen --dev
uv run python ml4phy-paper/artifact/verify_artifact.py
PYTHONPATH=ml4phy-paper/scripts:ml4phy-paper/review_followup_20260910:. \
  uv run pytest \
  ml4phy-paper/tests \
  ml4phy-paper/review_followup_20260910/tests \
  ml4phy-paper/artifact/tests -q
PYTHONPATH=ml4phy-paper/scripts:ml4phy-paper/review_followup_20260910:. \
  uv run python ml4phy-paper/review_followup_20260910/validate_outputs.py
```

The independent validator recomputes 150 mechanism cells across 500 bins from
the committed compressed bin table. It also confirms reproduction of the prior
630-cell global coverage export.

### Saved-prediction regeneration

Regenerating all portable tables requires the external event-level predictions
at the logical paths and hashes recorded by the presentation and mechanism
manifests. Checkpoints are not needed for this level.

### Training rerun

Training requires the public Majorana waveform data, the frozen local role and
subset archives, and the exact historical RESUM_FLEX source. Checkpoints do not
need to be distributed because the training campaigns rebuild them. See
[`ARTIFACT_REPRODUCIBILITY.md`](ARTIFACT_REPRODUCIBILITY.md) for commands and
the external-dependency boundary.

## Scientific limitations

- The classifier uses 18,866 events. A 5k efficiency-training result is not an
  end-to-end 5k result.
- Final targets were historically inspected; the follow-up is not an untouched
  test.
- The 2k failure and 10k nonmonotonicity are part of the result.
- The sparse tail contains 150 reference events, four passes, and no
  sampler-eligible events in the small training subsets. The method is not best
  there.
- Context and subset variation are descriptive because draws overlap.
- Dropout spreads are not calibrated efficiency intervals.
- Finite-pass curve roughness mixes learned variation and numerical MC noise;
  no smoothness guarantee is claimed.
- The mechanism controls cover one 5k subset and one calibration mixture.
- No 20k efficiency-training run exists. The historical full pool is exactly
  18,866 events.

## Recommended reading order

1. [`reports/data_budget_ledger.md`](reports/data_budget_ledger.md)
2. [`reports/phase3_result.md`](reports/phase3_result.md)
3. [`reports/paper_coverage_export.md`](reports/paper_coverage_export.md)
4. [`review_followup_20260910/reports/mechanism_result.md`](review_followup_20260910/reports/mechanism_result.md)
5. [`ARTIFACT_REPRODUCIBILITY.md`](ARTIFACT_REPRODUCIBILITY.md)
6. [`reports/artifact_release_v1.md`](reports/artifact_release_v1.md)

All newly authored paper text, code, comments, and figure labels are in English.
Historical filenames and schema keys remain unchanged for compatibility and
hash stability.
