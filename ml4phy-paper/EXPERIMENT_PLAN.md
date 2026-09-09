# Evidence-First Experimental Plan for ML4PS 2026

Status: planning and local artifact audit only. No new model training or inference has been performed for this plan.

Execution location: all training and expensive inference must run on the lab server, not the author's laptop. Local artifact absence is a recovery lead for the server audit, not a reason to recreate the training environment on the laptop. Follow [SERVER_WORKFLOW.md](SERVER_WORKFLOW.md) for the Git handoff and result-return contract. The current package is a plan and inventory utility; the paper-specific evaluator and experiment runner are not implemented yet.

## 1. Scope and intended claim

Develop a four-page Research-track paper on improving MAJORANA cut-acceptance estimation using energy-spectrum structure to balance localized acceptance changes and smooth continuum behavior. MAJORANA is the sole required real dataset. Additional datasets, synthetic benchmarks, cross-domain generalization, new classifiers, and broad architecture searches are outside the core plan.

Working scientific question:

> Can density-guided frequency control improve energy-dependent inclusive cut-acceptance estimation in the MAJORANA calibration data, relative to conventional CNP variants and kernel regression, without introducing excessive continuum oscillations?

The primary estimand is provisionally the inclusive acceptance

\[
\beta(E;T_*)=P(s\ge T_*\mid E),
\]

for a frozen waveform classifier, a fixed threshold, and a specified event population. Inclusive acceptance must not be described as pure signal efficiency. Signal/background-conditioned acceptance is secondary and requires reliable label provenance. The physical hypothesis is that energy-spectrum structure helps identify locations requiring flexible acceptance estimates; density does not determine the sign or magnitude of an acceptance feature by itself.

Keep the title and conclusions application-specific. Do not claim a universal solution to local function reconstruction or calibrated uncertainty without supporting experiments.

## 2. Preservation and execution rules

1. Read existing artifacts before deciding to reproduce or replace them.
2. Never edit, delete, overwrite, or relocate existing source, YAMLs, notebooks, caches, summaries, or checkpoints.
3. Write all new work under `ml4phy-paper/`, with a unique directory for each run. Refuse execution if its intended output files already exist.
4. Treat historical results as development evidence. Do not select a model using a final evaluation set and then describe that same set as untouched.
5. Do not launch the old notebook, cache builder, or batch inference commands indiscriminately. Some routes run inference or overwrite historical cache paths.
6. Implement any needed evaluation changes as new adapters in this directory. Existing modules may be imported after checking import side effects; use `PYTHONDONTWRITEBYTECODE=1` and route temporary files and tool caches here. Do not modify original code to make the new protocol work.
7. Record executed commands, source and dependency revisions, data and checkpoint hashes, resolved configurations, seeds, event identities, output paths, and resource usage.
8. A saved configuration or summary is not proof that a runnable model is locally available. Recover artifacts before considering retraining.

## 3. Verified local evidence

Audit reference: repository commit `c0664d1572812b4750fec0a117a19593c6b30be1`, September 8, 2026. See `inventory.json` for exact paths and SHA-256 hashes.

### 3.1 Artifact availability

| Artifact | Observed local state | What it permits |
|---|---|---|
| Cut-acceptance YAMLs | 25 under `configs/cut_acceptance/`; 49 archived YAMLs under `experiments/configs/` | Inspect actual settings and historical experiment coverage |
| Run summaries | 58 JSON files | Recover event counts, threshold, lineage fields, and historical output locations |
| Canonical caches | Five NPZ files and five associated audit JSONs | Analyze one saved prediction per model without training |
| Checkpoints | No `.ckpt` or `.pt` files found within this checkout | New inference cannot currently be reproduced here |
| Event files | No `.h5` or `.hdf5` files found within this checkout; `runs/` absent | Original classifier outputs and raw-event provenance must be recovered |
| Notebook | 61 cells, including stored outputs and kernel-regression helpers | Recover plotting logic and historical presentation settings |
| Training dependencies | Imports from external `core` and `schemas`; no RESUM_FLEX dependency declared in `pyproject.toml` | Exact upstream package/revision must be established before execution |

The classifier YAML points to `/home/klz/Data/MAJORANA`, and the lock configuration selects a CUDA 12.9 PyTorch index. These indicate a different execution environment from this Mac. Do not assume a local `uv sync` is sufficient. No environment installation was attempted. The adjacent path `/Users/yuema137/RESUM_FLEX` is absent; this is not an exhaustive search for the dependency elsewhere.

### 3.2 What the five caches contain

Canonical paradigms: `true_cnp`, `sweeps/base1_matched`, `sweeps/base3_matched`, `sweeps/cell15_v5`, and `sweeps/cell17`.

Each cache contains:

- 18,866 training energies and classifier scores;
- 2,000 context energies and scores;
- 5,074 target energies and scores;
- 212 retained evaluation bin centers and predictions;
- 800 dense energy queries with predictive mean and standard deviation;
- threshold, context-size, and summary scalars.

The stored NPY members for training/context/target energies and scores, bin centers, and dense query grids have identical hashes across all five caches. This establishes shared saved arrays, not event-level independence or complete model provenance. No event identifiers, class labels, per-MC-draw predictions, or model weights are present in these caches.

They support matched evaluation of the saved curves on the saved events, recomputation of descriptive errors, and a context-only kernel-regression baseline. They do not support a new CNP context realization, a new threshold, recovery of sub-grid features, or validation that the saved YAML exactly matches the unavailable checkpoint.

### 3.3 Historical metrics, not final paper results

| Saved model | Pearson r | Existing SE residual statistic | Existing DEP residual statistic |
|---|---:|---:|---:|
| True CNP | 0.4842 | 31.609 | 18.119 |
| Base 1 | 0.4822 | 32.453 | 17.525 |
| Base 3 | 0.3111 | 0.000166 | 10.127 |
| Cell 15 v5 | 0.5060 | 0.245 | 19.888 |
| Cell 17 | 0.5297 | 0.130 | 22.057 |

The residual columns reproduce the audit fields named `chi2_DT`; they are not endorsed here as calibrated chi-square tests. They show why one favorable SE number is insufficient: Base 3 also has a tiny SE statistic while its saved continuum roughness is much larger. Cell 17 does not improve every peak in these caches. Do not rank models by p-value.

### 3.4 The talk and caches represent different evidence

The talk `/Users/yuema137/Papers/MJD-Paper/resum_majorana_legend_meeting.pdf`, dated August 19, 2026, shows `Cell 15 v9` on slides 15-16, with 2,000 context events, 139,474 target events, and 5 keV display bins. The archived v9 YAML has fixed `hard_filter_lambda_min: 4.0`. Its summary records 18,866 training and 7,074 validation events, but the displayed full-test predictions and v9 checkpoint are not present here.

Cell 17 instead learns a frequency floor bounded between 1 and 5. Do not combine the talk's v9 curve, Cell 17's cached numbers, and one shared method description. The notebook's full-test override is a plausible route to the larger talk target set, but this linkage remains unverified until the original predictions, execution settings, and model hashes are recovered.

## 4. Concrete issues to resolve before additional training

### 4.1 Existing baselines are not controlled single-factor ablations

Read YAML values, not variant names or comments:

| Setting | Base 1 | Base 2 (archived) | Base 3 | v9 / Cell 17 |
|---|---|---|---|---|
| Sampling | Default flat-stratified | Physics-anchored | Mixed-density | Flat-stratified |
| Trial event range | 512-2048 | 512-2048 | 512-2048 | 640-1024 |
| Training context range | 128-2048 | 128-2048 | 128-2048 | 128-512 |
| Encoder dropout | 0.1 | 0.1 | 0.1 | 0.2 |
| Attention heads / dimension | Mean aggregation | 4 / 64 | 4 / 64 | 1 / 128 |
| Fourier features | Off | Off | 10 bands | 10 bands |

All listed configurations use 3,000 training steps and batch size 16, but equal steps do not imply equal sampled-event exposure. Some comments describe different head counts or context limits than the active values. The README architecture labels are therefore not sufficient evidence of checkpoint architecture.

First search the archived configurations and recovered model metadata for genuinely controlled comparisons. If none exist, train only the missing controlled variants, as specified in Section 7.

### 4.2 Threshold selection uses the evaluation file

`pipeline.py::run_pipeline` computes Youden-J threshold from `validation_predictions_path`. `cnp_test_inference.py::_filter_test_events` independently recomputes it from all labels in that same file before splitting context and target.

Consequences: changing the evaluation file can change the estimand; historical target labels influenced the cut selection. This does not by itself prove training leakage, but it prevents describing that evaluation as an untouched test at an independently selected cut.

Paper evaluation must choose the threshold on designated development/calibration events, freeze it, and pass it explicitly to the new evaluator. Merely redirecting the old CLI output directory does not fix threshold selection. Cached scores can be re-thresholded, but cached CNP predictions cannot: new predictions at a new threshold require model inference.

### 4.3 Cache retrieval can disagree with notebook controls

Notebook cell index 39 contains `get_inference`, which returns a persisted result when available before applying the live full-test override. The stored cache is not validated against requested context count, seed, threshold, or override file. Event loading has different override behavior. This is a possible route to incompatible curve and data combinations; it is not proof that the talk used such a combination.

Use a paper-specific loader that rejects mismatched metadata. Do not use notebook toggles as evidence that inference was rerun. A bootstrap or rebinning of saved targets also does not create new model predictions.

### 4.4 Metric definitions need a paper-specific implementation

`_peak_chi2_and_z` computes mean squared standardized residuals with empirical and model variance in the denominator. Its separate Z denominator uses cross-bin empirical variance plus mean model variance, not a complete sampling variance of the estimated mean. For a single bin, that cross-bin term is zero. The resulting normal-tail p-values must not be reused as calibrated evidence of agreement.

Most named peaks have one valid bin in the cached +/-5 keV windows. This cannot establish feature width or full shape recovery. Sparse/empty bins and bins omitted by training-derived selection must be reported, not silently excluded from the paper's evaluation domain.

The dense grid has 800 points over the configured energy range. Interpolation is acceptable for a preliminary cache audit but cannot recover unsaved narrow structure. Final local metrics should query the model at target energies or a sufficiently resolved deterministic grid.

### 4.5 Density and uncertainty semantics

The final density path uses the full filtered training energy pool, reconstructed from `train_predictions_path`, not only the inference context. That pool is a non-persistent model buffer and must be recovered with the checkpoint. The decoder receives raw coordinates, gated Fourier features, and an explicit density-contrast feature. Frequency gating is a sigmoid weighting even where the option is named `hard_filter`.

The current inference routine can combine MC dropout with context subsampling when the available context exceeds its per-pass cap. Record `n_context_per_pass` and inspect the active default rather than assuming every curve uses all supplied context. Separate dropout variation, repeated context sampling, and repeated training in the paper. Model-only dispersion should not be labeled a calibrated confidence interval for an empirical bin rate.

## 5. Phase A: recover and analyze existing evidence first

### A0. Completed local discovery

- Inventory paths, source hashes, summaries, cache schemas, and array identity.
- Inspect core model, sampler/pipeline interfaces, threshold derivation, diagnostic definitions, and notebook cache/override behavior.
- Compare the talk's model and evaluation labels with archived configurations.

These are inspection results. Historical numerical metrics were read, not independently reproduced from checkpoints.

### A1. Recover the smallest useful artifact bundle

Priority order:

1. v9 checkpoint and its exact resolved configuration; original full-test curve/export used on talk slides 15-16.
2. v5, Cell 17, Base 1, Base 2, and Base 3 checkpoints with original configuration snapshots.
3. Small-training and full-evaluation `predictions.h5`, frozen waveform-classifier checkpoint, classifier configuration, and event-ID/split manifests.
4. The `training_pool.npz` files required by the old evaluator, plus original density-pool membership.
5. Exact RESUM_FLEX revision and execution environment; training logs and runtime/device records if available.
6. Close archived controls: `cell15_matched`, `cell16`, and the variants without direct contrast injection or with learned versus explicit frequency gating.

Inspect recorded paths and author-specified storage locations before any retraining. A run-summary checkpoint path is a recovery lead, not a valid local input. Do not scan unrelated private storage or retrieve external data without a known relevant location. Put newly recovered artifacts in unique subdirectories here or reference their existing read-only locations.

Deliverable: `reports/artifact_recovery.md` with recovered/missing status, path, hash, source machine/revision, and implications. If a needed bundle cannot be found, record precisely which experiments it prevents; continue all cache-based work.

### A2. Cache-only reanalysis, zero new training

Use the five shared-array caches as historical development evidence:

- Recompute per-region errors on the shared target sample, with explicit interpolation and sparse-bin flags.
- Compare saved dense curves and roughness on exactly the same grid.
- Produce DEP, Bi, SE, and FE close-ups, plus both continuum windows and the high-energy tail. Do not select only favorable regions.
- Recover the training-density contrast from cached training energies, reproducing kernel normalization and epsilon; verify against implementation before interpreting.
- Implement the notebook's Nadaraya-Watson estimator as a documented baseline, fitting bandwidth on development/context information only. A kernel estimate using target outcomes is a reference visualization, not an eligible predictor.
- Quantify event support, retained/omitted bins, and the distinction between bin-center predictions and event-weighted bin averages.

Deliverables: `reports/cache_reanalysis.md`, numeric tables, and figures tagged `historical`, `fixed_context`, and `interpolated_where_applicable`. Do not claim fresh context robustness, final held-out performance, or uncertainty calibration from these artifacts.

### A3. Reconcile the talk and select a development candidate

Recreate v9, v5, and Cell 17 with the same event selection, fixed threshold, context, inference cap, MC settings, and query grid if weights are recovered. Use development data to choose a candidate based on joint peak/continuum performance and stability. Prefer the simpler fixed-floor model if added learnability brings no consistent benefit. If v9 cannot be recovered, record that the talk's central curve is not yet reproducible; do not replace it silently with Cell 17.

### A4. Produce the gap decision

Before training, produce `reports/gap_decision.md` with one row per desired claim: evidence found, provenance quality, remaining ambiguity, cheapest resolution, and train/re-evaluate/reuse/defer decision. Estimate wall time only after an actual inference or training pilot on the selected machine. Do not invent GPU-hour estimates from the README.

## 6. Frozen evaluation protocol

Finalize this protocol after artifact recovery and before inspecting final-test results. Values below are proposed defaults, not statements about completed experiments.

### 6.1 Data roles and identity

Keep the waveform classifier fixed. Audit how its checkpoint was selected; do not equate file names with independence. Document classifier-training events, CNP-training events, the density pool, threshold-development events, model-development events, context reservoir, and final target set.

Use file plus original event index as stable identity where possible. Cached energy-score pairs are not sufficient proof of identity. Require final target outcomes to be excluded from training, threshold selection, model selection, bandwidth tuning, density construction, and context. The training energy pool may also train the CNP, provided this is disclosed.

Choose the final holdout from events not used in historical model selection when available. If the full test data were already examined, do not relabel a random subset as previously untouched. State this limitation and use a prospectively frozen repeated-split protocol, or identify an unused portion of MAJORANA. New physical datasets remain unnecessary.

Freeze the threshold on a separate development/calibration set with adequate class labels. Do not automatically preserve the historical `T*=0.540643572807312` unless its independent calibration can be established. Since the CNP is trained across thresholds, a new threshold may require only inference, subject to checking the trained range and loss implementation.

### 6.2 Context and randomness

Primary context count: 2,000 if supported by the available reservoir. Start with 10 paired context draws using proposed seeds 100-109; increase to 20 only if results are materially unstable or inconclusive. Keep target membership fixed across these draws. Use identical context IDs for all compared models.

Separate context-selection seeds from dropout seeds and training seeds. The original `split_test_data` changes both context and target with its seed, so a new adapter is required for this design. Use 50 MC dropout passes initially to match historical practice; check convergence on development data before increasing. Explicitly report and match per-pass context budgets, including any computational cap. Do not treat 50 MC passes as 50 independent training runs.

For core newly trained models, use three training seeds (0, 1, 2) unless comparable existing seeds are recoverable. Show paired differences for each seed and context draw. Context draws share a target set and may overlap; do not count all combinations as independent datasets or use their number to exaggerate confidence.

### 6.3 Regions and reference construction

Full display range: 500-3000 keV. Audit the historically excluded bins explicitly.

Proposed localized reporting windows: +/-15 keV around the existing nominal energies 1592, 1620, 2103, and 2614 keV. Split the DEP and Bi windows at their midpoint (1606 keV) to avoid double counting. Confirm labels and energy calibration from the data documentation before finalizing boundaries. Use 5 keV bins for primary diagnostic plots and 10 keV bins as a sensitivity display; never infer detector resolution from the chosen bin width.

Use the existing 1700-2000 and 2200-2400 keV continuum windows for historical comparability, subject to a physics check for other structures. Treat 2200-2400 as a varying continuum/edge region, not a constant acceptance plateau. Report 2700-3000 keV separately as a sparse-tail diagnostic. Show counts and unsupported bins; avoid smoothness claims based on visually empty regions.

### 6.4 Metrics and interpretation

Primary predictive metric: event-level Brier score, globally and within each predefined region,

\[
\mathrm{BS}_R=\frac{1}{N_R}\sum_{i\in R}(\hat\beta(E_i;T_*)-\mathbb{1}[s_i\ge T_*])^2.
\]

This evaluates predictions without pretending empirical bin rates are noiseless truth. Report an equal-region summary as well as the event-weighted global score so populous continuum regions cannot hide a missed narrow feature. If a region has inadequate support, mark it inconclusive rather than changing its boundaries after seeing results.

Secondary metrics:

- Bin-level MAE/RMSE using the model averaged at the actual target event energies in each bin and the empirical passage fraction on those same events. Report bin count thresholds and all exclusions.
- Peak/sideband contrasts with empirical uncertainty where support allows. This tests direction and amplitude beyond one bin. Width/location claims require enough resolved measurements and are not mandatory.
- Mean absolute second difference on one fixed grid, paired with continuum prediction error. MASD depends on grid spacing; report spacing and never compare across grids without normalization or regeneration. For final plots, pilot 1 keV spacing and check 0.5 keV convergence on development data only.
- Predictive log loss as a secondary diagnostic with explicitly specified probability clipping; Pearson correlation only as a descriptive supplement.
- Runtime, memory, context cap, and offline training cost if recoverable. Report kernel fitting cost and the full density-pool cost rather than counting context alone.

Report training-seed and context-draw dispersion separately. If confidence intervals are added, respect event grouping and the shared/overlapping evaluation design; use paired resampling for model differences. Empirical-rate uncertainty and model uncertainty are different quantities. Do not reuse the historical Z-derived p-values, interpret non-rejection as equivalence, or claim calibrated coverage without a dedicated validated calculation.

## 7. Conditional experiment queue

The table specifies priority and decision rules. It is not a request to execute every row.

| ID | Question | Reuse first | Run only if | Deliverable / completion condition |
|---|---|---|---|---|
| E0 | What do existing predictions establish? | All five caches and audit files | Analysis only; no weights needed | Shared-sample regional table and full-range/peak plots with caveats |
| E1 | Which historical final variant is appropriate? | v9, v5, Cell 17 recovered weights | Saved outputs do not share protocol | Paired development comparison; candidate and reason frozen |
| E2 | Does the candidate improve over CNP, attention, and ungated PE? | Existing models with verified matching metadata | No adequately controlled archived models exist | Controlled ladder below, three training seeds only for missing models |
| E3 | Is the decoder's frequency gate useful beyond other density paths? | Closest archived controls | E2 cannot isolate that component | Candidate versus constant/full-band decoder mask, retaining attention-density and direct-R paths |
| E4 | Is direct contrast injection responsible for the benefit? | Archived no-injection variant with otherwise equal settings | Needed to support an explicit frequency-gating claim | Candidate versus no direct R input; architecture rebuilt and retrained if input dimension changes |
| E5 | Does the model help relative to kernel regression? | Cached context plus notebook formula | No held-out tuned baseline exists | Context-only kernel baseline and, where meaningful, a development-selected train+context baseline with disclosed information budgets |
| E6 | Does performance depend on a fortunate context? | Saved repeated inference if found | Only one context is available | 10 paired draws; extend to 20 only if justified |
| E7 | Does local smoothing survive changes in available context? | Any comparable historical evaluations | E6 or a sample-efficiency claim makes it necessary | Context sizes 500, 1000, 2000 with nested paired samples; inference only |
| E8 | Is the conclusion tied to a single cut? | Existing threshold evaluations | A threshold-robustness claim is desired or E1 reveals sensitivity | Two additional development-chosen cuts, frozen before final evaluation; no classifier retraining by default |

### 7.1 Controlled ladder specification

Proposed base settings are the candidate's training setup: flat-stratified sampling, bin-stratified density sampling, trial range 640-1024, training contexts 128-512, encoder latent size 64 and hidden sizes [128,128], dropout 0.2, decoder hidden sizes [128,128,128], 3,000 steps, batch size 16, learning rate 0.001, and common loss/mixup/threshold sampling. Verify effective sampler behavior in the recovered upstream implementation. Use shared event/threshold sampling seeds where feasible and record realized event exposure.

Models: M0 mean CNP without PE; M1 attentive CNP without PE; M2 attentive CNP with 10 Fourier bands; M3 selected density-guided candidate. Use the same attention dimensions (1 head, dimension 128) for attention variants. Keep all other nonessential settings fixed. Report parameter counts; do not claim exact parameter matching when input dimensions or modules differ.

The M2-to-M3 comparison tests the complete density-guided package because multiple density paths change. E3 is needed for a narrow decoder-frequency-gate attribution; E4 tests direct contrast injection. A full-band control must actually use unit band weights, not merely set a finite cutoff whose sigmoid weights remain below one. Evaluation-only gate removal is a sensitivity probe, not a trained ablation.

Do not train all kappa values 1-5 again. Recover the archived floor sweep and choose among fixed v9 and learnable Cell 17 on development evidence. Training a new controlled model is justified only by a missing comparison, missing weights after recovery attempts, or invalid provenance.

### 7.2 Kernel-regression specification

Use Gaussian Nadaraya-Watson regression of binary passage flags, not a KDE of energy alone. Candidate bandwidths: 2, 5, 10, 20, 50, and 100 keV, selected by development predictive score. Use stable normalized weights and flag effective support in sparse regions. Do not pick 5 keV simply because it produces a visually noisy baseline.

Report context-only inference explicitly as a different information regime from a pretrained CNP. A train+context baseline can use additional historical labeled events when that pooled population matches the estimand; select any weighting on development data. If classifier train/test score distributions differ, pooling is not automatically valid. Document the difference and avoid presenting the comparison as information-matched without justification.

### 7.3 Resource limits and stopping decisions

First execute one inference pilot, then one 3,000-step training pilot only if E2-E4 require training. Measure wall time and peak memory. Estimate remaining cost as measured per-job cost times the unresolved model/seed count. There is no supported fixed hour estimate yet.

The full ladder requires at most 12 training jobs for four models and three seeds if nothing is reusable; E3 and E4 add at most six jobs if both are scientifically necessary. These are conditional upper counts, not approved launch batches. Reuse every compatible existing run. Stop adding experiments once the scoped application claim is supported, limitations are characterized, and the primary comparison is stable. An unfavorable result calls for a narrower claim or explicit limitation, not an unlimited hyperparameter search.

## 8. Paper outputs and completion criteria

Proposed manuscript allocation: about 0.75 page for the physical task and motivation, 1 page for method, 1.75 pages for setup/results, and 0.5 page for limitations/conclusion. References are separate. The final four-page layout determines exact allocation.

- Figure 1: density profile -> contrast -> frequency weights -> acceptance estimate, with actual model paths and declared data sources.
- Figure 2: common held-out comparison with full energy range and compact DEP/SE/continuum close-ups; retain FE/Bi and sparse-tail diagnostics in the analysis record.
- Table 1: core model errors, continuum roughness, and repeated-run variation; no unsupported p-value verdicts.

The empirical package is ready when:

- [ ] Inclusive population, classifier identity, threshold, and energy range are explicit.
- [ ] Talk versus cache provenance is reconciled or the unavailable result is excluded transparently.
- [ ] Final evaluation roles and historical model-selection exposure are documented.
- [ ] At least the baseline, selected method, and scientifically necessary controls share a valid protocol.
- [ ] Comparisons include localized accuracy and continuum accuracy/roughness, with event support.
- [ ] Repeated contexts and core training variation are measured or their absence limits the claim explicitly.
- [ ] Kernel bandwidth is selected independently of final target outcomes.
- [ ] Failed regions are reported; no generalization or uncertainty claim exceeds the evidence.
- [ ] Final figures are generated from identified numeric artifacts with commands and hashes.
- [ ] Original repository files are unchanged; all paper work remains in this directory.

## 9. Planned directory layout and handoff

Only the initial planning files and inventory exist at this stage. Other paths below are proposed future outputs.

```text
ml4phy-paper/
  README.md
  EXPERIMENT_PLAN.md
  audit_inventory.py
  inventory.json
  reports/             # Recovery, cache reanalysis, and gap decision
  manifests/           # Event identities, frozen protocol, dependency revisions
  configs/             # New resolved configurations with unique output paths
  scripts/             # New loaders, evaluators, metrics, and plotting adapters
  runs/                # New runs only; never reuse a historical output path
  tables/
  figures/
```

Immediate next action: perform A1 and A2, then write the gap decision in A4. No new training should precede that decision. Missing remote artifacts are a recovery question, not evidence that the corresponding scientific experiment was never performed.
