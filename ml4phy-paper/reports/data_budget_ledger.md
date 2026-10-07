# Data-budget and executed-method ledger

Audit date: 2026-09-09 (America/Los_Angeles)
Status: complete before any extension training. The two uncommitted laptop plans named in the extension prompt were absent from this checkout.

## Unique-event accounting

The classifier's 5% budget is **18,866 unique events selected from 377,330 eligible raw train-split events**, not a percentage of the acceptance export. The denominator is the concatenation of all 16 configured train files after applying `500 <= energy < 3000 keV`; `round(0.05 * 377330) = 18866`. Reconstructing NumPy's seed-0 selection from the raw files reproduces the 18,866-row classifier train prediction export identity-for-identity, including order, energy, and labels.

The acceptance model reads those same 18,866 score evaluations. Its `target_class: all` and energy filter remove none, but the sampler retains only bins with at least four events: **18,836 unique events are sampling-eligible and 30 are excluded across 21 sparse nonempty bins**. Sampling is with replacement. The density-guided model separately reconstructs a nonpersistent energy-only density buffer from all **18,866** input events before the minimum-bin sampler filter. Therefore the density-pool/acceptance-input overlap is 18,866/18,866, while the sampling-effective/density overlap is 18,836/18,866.

The classifier train split is identity-disjoint from every evaluation role. The 7,074-event classifier monitoring subset is exactly partitioned into 2,000 threshold-calibration events, a 3,000-event development context reservoir, and a 2,074-event development target. The remainder of the full 141,474-event test export is exactly partitioned into the 20,000-event final context reservoir and 114,400-event final target.

The ten executed 2,000-event development contexts have a unique union of **3,000** events within their 3,000-event reservoir. The ten final contexts have a unique union of **13,060** within their 20,000-event reservoir. Context overlaps are sensitivity replications, not independent additional data.

The union of unique waveform events that were either used for classifier training or score-evaluated in the full follow-up data is **160,340 = 18,866 train + 141,474 disjoint test**. If the full filtered classifier-training denominator is disclosed as the source population considered for the fixed subset, its union with the full test export is **518,804 = 377,330 + 141,474**. Acceptance pretraining, density estimation, calibration, conditioning, and evaluation reuse these events; their row counts must not be summed as if disjoint.

The machine-readable ledger and complete pairwise intersections are in `tables/data_budget_ledger.csv` and `tables/data_identity_overlap.csv`. No event identity is committed; only counts and order-independent SHA-256 hashes are exported.

## Classifier execution

- Configuration: fixed subset seed 0, `subset_portion=0.05`, 50 epochs, batch size 256, `train_portion=1.0`.
- Preprocessing: subtract the mean of the first 500 waveform samples, divide by the positive maximum, align to the first 90%-rise sample, crop 200 samples before and 2,000 after with zero padding, then cast to float32.
- Label: raw `psd_label_low_avse`. The classifier loss is unweighted `BCEWithLogitsLoss`; `pos_weight: auto` is inactive because `loss.type` is `bce`.
- Sampling: class-balanced and 10-keV energy-balanced weights are multiplied; `WeightedRandomSampler` draws 18,866 samples with replacement per epoch. This gives 943,300 repeated train draws over 50 epochs. The exact realized unique coverage was not recorded and cannot be reconstructed from the saved checkpoints because model/dropout RNG consumption shared the PyTorch generator.
- Monitoring: a separately reconstructed 7,074-event fixed 5% test subset is sampled with replacement under the same weights each epoch, for 353,700 monitoring draws. It is not used for gradient updates.
- Checkpoint rule: every epoch is saved. Directory-based evaluation selects the lexicographically latest checkpoint, epoch 50. There is no best-validation checkpoint selection; epoch 49 has the highest recorded monitoring ROC AUC, while epoch 50 was executed and retained as the frozen classifier.

## Acceptance-model execution

The executed output is a logistic-normal Bernoulli-probability parameterization. The decoder emits unconstrained `mu_logit` and `log_sigma`; `sigma = softplus(log_sigma)`. For each target event and each of four MC samples, training draws standard-normal noise and forms `beta = sigmoid(mu_logit + sigma * epsilon)`, clipped to `[1e-6, 1 - 1e-6]`. The loss is Bernoulli negative log likelihood with one final unweighted mean over MC samples, batch trials, and target events. At deterministic evaluation, the pinned implementation uses `sigmoid(mu_logit)`; the paper's 50-pass estimator instead averages stochastic network evaluations with dropout active.

`mixup_alpha=0.01` is present in the validated configuration but **no executed training code reads it**, so mixup is inactive. Each training seed uses Adam at 1e-3, gradient-norm clipping at 1.0, 3,000 steps, batch size 16, variable trial size 640-1,024, and context size 128-512. The exact repeated exposure implied by the training-loop RNG is:

| Training seed | Pool-event draws | Context uses | Target-label uses | Four-sample NLL terms |
|---:|---:|---:|---:|---:|
| 0 | 40,056,128 | 15,301,808 | 24,754,320 | 99,017,280 |
| 1 | 40,041,120 | 15,344,784 | 24,696,336 | 98,785,344 |
| 2 | 39,875,408 | 15,458,896 | 24,416,512 | 97,666,048 |

Only the final 3,000-step acceptance checkpoint is written; `eval_every=0`, so the 7,074-event validation path does not select checkpoints or affect gradients. The legacy pipeline computes a Youden threshold from that file after training for its summary, but the paper evaluator ignores it and uses the separately frozen calibration threshold.

## Decoder gate

For Density-guided CNP, the executed contrast gate is `lambda(R) = kappa + (10 - kappa) * sigmoid(10 * (R - 3))`. Thus the contrast threshold is 3 and the sigmoid argument slope is exactly 10 per unit density contrast. The derivative `d lambda / d R` at `R=3` depends on the learned continuum floor:

| Training seed | Saved raw kappa | Executed kappa | Transition derivative at R=3 |
|---:|---:|---:|---:|
| 0 | 0.420302033 | 3.414222086 | 16.464444784 |
| 1 | 0.379859835 | 3.375357185 | 16.561607036 |
| 2 | 0.401080459 | 3.395788898 | 16.510527756 |

The subsequent per-Fourier-band decoder weight is `sigmoid(5 * (lambda - band_index))`; its logit slope is 5 per band-index unit and its maximum weight derivative with respect to `lambda` is 1.25.

## Threshold, randomness, and provenance limitations

The threshold is **0.540643572807312**, selected by Youden-J on only the frozen 2,000-event calibration role. No development or final target outcome entered threshold selection.

The original protocol manifest committed at 2026-09-08 20:37:55 PDT specified `dropout_seed = 10000 + context_seed`. Before candidate selection, the executed design changed to a fixed dropout seed of **10100** so the ten context draws isolate context variation. The fixed-seed result was committed at 2026-09-08 20:54:30 PDT; the earlier context-and-dropout-varying campaign remains server-side and was excluded. The original frozen manifest is preserved unchanged, and this report records the dated prospective execution amendment rather than silently rewriting it.

Recovered seed-0 Cell 17 has a readable final checkpoint, configuration, loss history, and colocated training pool, but lacks an exact MAJORANA source commit, clean/dirty worktree state, runtime, peak memory, and realized event-draw log. Seeds 1 and 2 have paper-runner provenance. This asymmetry must remain disclosed.

## 1620-keV feature identity

The repository's historical `Bi-214 1620` label is inconsistent with its stated thorium-228 calibration spectrum. The IAEA recommended gamma-ray data for thorium-228 with daughters identify a **1,620.74-keV gamma from bismuth-212**. New artifacts therefore use `Bi-212 1620.74 keV`; archived labels remain untouched. Source: https://www-nds.iaea.org/publications/tecdocs/sti-pub-1287_Vol2.pdf.

## Interpretation boundary

Classifier pretraining uses waveforms and physical PSD labels. Acceptance-model pretraining reuses classifier scores and energies from the same 18,866 identities; it does not add new measured events. Context outcomes add conditional information at inference. Target events are evaluation data, not training data, but their classifier scoring and outcome/reference cost are disclosed. Consequently, a context-size experiment supports conditional context efficiency only, not total-data efficiency.
