# Extension Gap Decision

Date: 2026-09-09 (America/Los_Angeles)  
Status: prospective decision recorded before extension training or inference

## Decision

The archived final result at commit
`48a59d0b2d93a6c76d2001522abcb98ac74432fb` remains frozen. The extension
will reuse compatible saved predictions and checkpoints and will not rerun the
classifier. Six matched neural training jobs are justified: three seeds of CNP
and three seeds of Attentive CNP. No other training campaign is authorized at
this stage.

The Phase 1 evaluation matrix contains 480 neural cells: four architectures,
three training seeds, ten context draws, and four nested context sizes. Sixty
existing 2,000-context cells for Attentive CNP + PE and Density-guided CNP are
compatible and reusable. Subject to the timing pilot, 420 neural cells remain
to be evaluated after the six missing checkpoints are trained.

The Phase 2 acceptance-training-size study, classifier retraining, optional
full-band gate control, broad hyperparameter searches, new datasets, and an
equal-information or sparse-GP campaign are outside this decision. Phase 2
requires a separate approval after Phase 1 results and a measured cost estimate
are reported.

## Reusable artifacts

- The frozen threshold is `0.540643572807312`, calibrated on 2,000 events.
- The classifier checkpoint and its saved train/test score exports are reused.
  The classifier was trained for 50 epochs on a fixed 5% subset: 18,866 unique
  events selected from 377,330 energy-eligible training events. The acceptance
  input pool is exactly the same set of 18,866 identities.
- Attentive CNP + PE and Density-guided CNP have matched 3,000-step checkpoints
  for seeds 0, 1, and 2. Their ten saved final context predictions per seed use
  50 MC passes, dropout seed 10100, the original 2,000-event contexts, and the
  fixed 114,400-event target set. These 60 cells remain valid endpoints.
- The saved final predictions are sufficient to export 5-keV support, local
  MAE/RMSE, peak/sideband contrasts, and acceptance curves without inference.
- The archived Gaussian kernel regression is the Nadaraya-Watson estimator,
  equivalently a common-bandwidth KDE ratio only when the passing-event density
  is multiplied by the context pass fraction. Re-running the archived
  aggregation with identical inputs reproduced `final_kernel_summary.csv`
  byte-for-byte (SHA-256
  `5f85bf6e245e6aa77ef1d78930f8fb95b7e30682a30f64e2bcda4d6dbbde9ef7`).
  The original development-selected bandwidths remain frozen.

## Incompatible historical checkpoints

Historical pooling and attention-only checkpoints exist, but they do not form
matched final-comparison rows. The small historical CNP used fixed 48-event
trials, context sizes 2--32, and dropout 0.1. The larger mean-pooling and
attention-only controls used 512--2,048-event trials, context sizes 128--2,048,
and dropout 0.1. These differences are confounded with architecture, so those
checkpoints will remain historical diagnostics only.

The new controlled configurations match the accepted 3,000-step sampling,
hidden widths, dropout 0.2, optimizer, and checkpoint rule wherever applicable:

- CNP: mean pooling, no positional encoding, 116,482 parameters.
- Attentive CNP: deterministic cross-attention, no positional encoding,
  149,250 parameters. It is not described as a latent ANP.
- Attentive CNP + PE: existing matched model, 151,682 parameters.
- Density-guided CNP (ours): existing matched model, 130,693 parameters.

Each training run saves the final 3,000-step checkpoint. There is no
validation-based checkpoint selection. The validation path is used only for a
legacy post-training threshold summary and does not contribute gradients.

## Data and protocol findings affecting interpretation

The exact data-budget ledger is exported separately. Important consequences
are:

- EventSampler filtering makes 18,836 of the 18,866 acceptance-pool events
  eligible for repeated model-training draws. The density-guided model still
  constructs its density pool from all 18,866 events.
- Classifier training and the full 141,474-event test export are identity
  disjoint. The classifier monitoring subset contains 7,074 fixed test events
  and exactly partitions into threshold calibration (2,000), development
  context reservoir (3,000), and development target (2,074).
- The final 20,000-event context reservoir and 114,400-event target are
  disjoint from training. The ten original 2,000-event final contexts have a
  union of 13,060 identities; they are overlapping draws, not ten independent
  datasets.
- The unique union of classifier-training and all scored test events is
  160,340. The union of the complete energy-eligible training denominator and
  the scored test set is 518,804. Training exposures are much larger because
  both classifier and acceptance training sample with replacement.
- The data were historically inspected. All extension results are
  prospectively specified follow-up analyses on historically exposed data,
  not an untouched test.
- Repository labels for the feature near 1,620 keV conflict. The extension
  uses the verified reader-facing label `Bi-212 1620.74 keV`, based on the IAEA
  Th-228 decay-chain reference, while retaining historical labels only where
  needed for provenance.

The originally frozen protocol at commit `1760df3` (2026-09-08 20:37:55 PDT)
specified a context-dependent dropout seed. Before candidate selection, commit
`7671cc7` (2026-09-08 20:54:30 PDT) changed execution to fixed seed 10100 so
context variation would not be mixed with dropout-stream variation. The
archived result consistently used 10100. The earlier joint context/dropout
campaign remains excluded and the original manifest is not overwritten.

## Required pilots and stopping rules

Before the Phase 1 neural matrix, measure an inference pilot including the
114,400-event target prediction, then report the projected cell count and wall
time. Compatible cells must be reused rather than recomputed.

Before the Gaussian-process campaign, run one full 2,000-context development
fit and target-probability timing pilot using the installed scikit-learn
version. The estimator must remain event-level Bernoulli classification with
posterior-integrated probabilities. If the measured projection exceeds two
hours on the available allocation, stop and report the cost for approval. Do
not silently replace it with binned Gaussian regression, reduced contexts,
omitted regions, or coarse interpolation.

The kernel and GP covariance-family choices for every new context size must be
made on development data only. The final targets cannot be used to select a
bandwidth, kernel family, seed, region, or presentation context. The Phase 1
full-range and local-zoom presentations are fixed at context sizes 500 and
2,000.

For the neural smoothness diagnostic, dropout-disabled predictions and nested
50/200-pass estimates with two independent streams are separate estimators.
Extension to 800 passes is conditional on material residual MC noise and cost.
No smoothness claim will be made unless numerical noise is controlled and the
continuum error is reported beside roughness.

