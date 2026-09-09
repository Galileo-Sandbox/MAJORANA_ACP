# Lab Server Artifact Recovery

Audit date: 2026-09-08  
Repository branch: `ml4phy-paper`  
Audited source commit: `b26457214ca486a4b356723b456510baf6c4cf8e`

## Executive finding

The lab-server checkout contains the artifact bundle that was absent from the laptop: 62 CNP checkpoints, 62 training-pool archives, 62 run summaries, 652 classifier `.pt` files, 15 classifier prediction HDF5 files, the raw MAJORANA files, and the editable RESUM_FLEX dependency. The priority v9, v5, Cell 17, True CNP, Base 1, Base 2, Base 3, `cell15_matched`, and Cell 16 checkpoints are all present and are readable.

No new training is needed to recover those models. The immediate work is protocol-correct inference and provenance reconstruction. The original full-test v9 curve/export, its exact context IDs, per-MC predictions, and CNP source-commit record were not found.

## Inventory scope

The paper inventory utility scanned 1,412 non-paper files in this checkout. It found:

| Suffix | Count | Interpretation |
|---|---:|---|
| `.ckpt` | 62 | CNP checkpoints |
| `.pt` | 652 | Classifier epoch checkpoints across several historical runs |
| `.h5` | 15 | Classifier prediction exports |
| `.npz` | 67 | 62 training pools plus five canonical inference caches |

The full local inventory remains ignored at `ml4phy-paper/local/server-inventory-003.json`. Two earlier inventory attempts are retained in the same ignored directory as incomplete failure records. They exposed Python 3.8 and non-finite-JSON compatibility issues, which were fixed in `audit_inventory.py` without changing any research artifact.

## Priority CNP artifacts

All listed checkpoints contain a 3,000-step loss history and checkpoint metadata. Parameter counts are derived from the saved state dictionaries. All nine inspected priority models use the byte-identical training-pool archive with SHA-256 `c3a7e7a147dcc625d750b676f5e5a980765573dddb3916cf21d18083e71fc227` and report 18,866 training events.

| Logical model | Checkpoint SHA-256 | Parameters | Checkpoint-confirmed architecture | Reuse decision |
|---|---|---:|---|---|
| True CNP | `fa39ff6270a4588ee73a0ab178a424b31ee0ecafa65d572c57bfb9c8433a` | 116,482 | Mean aggregation, no PE, encoder dropout 0.1 | Reuse for inference; historical training protocol only |
| Base 1 | `77b0032227a8abcbfc546f3c4e1b2673e92878e35dd0636624ed97804527f4` | 116,482 | Mean aggregation, no PE, flat-stratified | Reuse for secondary historical comparison |
| Base 2 | `9b3370e0f4671152823b95349ac8205a74677e9df2dfe2e0bef54408b3530fd6` | 132,866 | 4x64 cross-attention, no PE, physics-anchored | Reuse for secondary historical comparison |
| Base 3 | `f692b427514e4dc551588b89939be95f6bb825ce9a6e2a671ff0d9dfe2ac5353` | 135,298 | 4x64 cross-attention, PE10, mixed-density | Reuse for secondary historical comparison |
| `cell15_matched` | `fea9f50af5ecc1238198fa8953df1ef031315a6a19e611b22963d700850df575` | 130,692 | 1x128 attention, PE10, fixed density gate, direct contrast input | Reuse only with its recorded training budget |
| Cell 15 v5 | `f32ba045d449d5002c694156b525ffb7052532ac929b4ebc77726fd2976f3d25` | 130,692 | 1x128 attention, PE10, dropout 0.2, fixed floor 1 configuration | Reuse for E1 candidate comparison |
| Cell 15 v9 | `1d20c0563b842c3a49ed78abbf74d204e93719e094859aa631d7d8627914f35a` | 130,692 | v5 package with fixed `lambda_min=4`, `lambda_max=10` | Highest-priority recovered candidate; reuse for E1 |
| Cell 16 | `92798cf9506649b64b9d183dd6e5439b1fbe9ed45e77deff7ac819fd8d57eb13` | 130,693 | Unconstrained learnable floor; saved raw value 1.191162 | Development evidence only |
| Cell 17 | `e21af9988a220aa4b203e54d63f59f37778a41570cd283591f27b7a29af38867` | 130,693 | Learnable floor constrained to [1,5]; saved raw value 0.420302 | Reuse for E1 candidate comparison |

Checkpoint metadata confirms the active aggregator, PE, density-path, dropout, and device settings. It does not record `n_trial_events_min/max`, `n_context_min/max`, source commit, training seed beyond the colocated YAML, runtime, or realized sampled-event exposure. The YAMLs and file timestamps are strong recovery leads but cannot fill those missing checkpoint fields cryptographically.

## Classifier predictions and identities

The frozen classifier is `simple_cnn_small`, epoch 50. Its checkpoint SHA-256 is `301d0c1bd015cdcb47fc4839ca81c462a1073dadb46f2262e1d0db5ec7c3170d`. Its metadata records CUDA execution on an NVIDIA GeForce RTX 5090 and source commit `3210ebc7b44bc6ccd6b893ba03fd9dc6ba2dfa38`, with a dirty worktree. The classifier configuration hash recorded by every inspected CNP summary is `ad8f147aabe1fcd13f79932eaaa2950a0054253120f10b175692420c0236429f`.

| Logical role | Rows | SHA-256 | Identity result |
|---|---:|---|---|
| CNP training predictions | 18,866 | `ea2eb5594acdd181ebc18926fd8deabc7b2f6910b117dec863d87c3b70057159` | 18,866 unique composite identities |
| Historical development/evaluation predictions | 7,074 | `d318b49a99dd3d1ead3b1aa9c8d3e97d78aaf3cc0ab97c040bbc2235436ba5cd` | 7,074 unique composite identities |
| Full-test predictions | 141,474 | `47f55e0ba08102462b5348bc3ba7e575dd2c653cc15adadc54ef6393d2e505fb` | 141,474 unique composite identities |

Identity was checked with `(run_number, detector, id, tp0)`. The training predictions are disjoint from both evaluation exports. The 7,074-event evaluation export is an exact identity subset of the full-test export, leaving 134,400 full-test events outside that historical subset.

The full-test row count exactly explains the talk's 2,000 context plus 139,474 target events. This makes the recovered HDF5 file the likely source, but it does not prove which 2,000 identities or which random seed produced the displayed curve. No original full-test prediction curve or event split was found, so the talk result must be recreated rather than silently equated with a cache.

The five canonical caches still contain only 2,000 context and 5,074 target events from the 7,074-event export. They share identical energy-score arrays and support cache-only development analysis. They do not contain event identities, labels, per-MC draws, model hashes, or a provenance-safe threshold.

## Raw data and storage

The configured MAJORANA storage exists and contains 25 raw HDF5 files (16 train, six test, and three NPML files), approximately 46 GB total. These files were not copied, modified, or added to Git. The classifier prediction exports are sufficient for the planned cut-acceptance work unless waveform-level classifier provenance must be revisited.

## Runtime environment

| Component | Recovered state |
|---|---|
| Repository Python | 3.12.13 |
| PyTorch | 2.11.0+cu129 |
| NumPy / h5py / scikit-learn | 1.26.4 / 3.16.0 / 1.8.0 |
| Pydantic / PyYAML / Matplotlib | 2.13.3 / 6.0.3 / 3.10.8 |
| GPU | NVIDIA GeForce RTX 5090, 32,607 MiB total, driver 575.64 |
| Scheduler | No Slurm, PBS, or LSF client found on `PATH`; direct local execution is indicated |
| RESUM_FLEX install | Editable package version 0.0.1 |

The editable RESUM_FLEX checkout is now at `4cd63cf151a422c7bf41e0dcaba73607cd97595c`. That September revision rejects three fields in the historical MAJORANA YAMLs (`output_activation`, `mixup_alpha`, and `n_mc_samples`), so it is not directly compatible with the recovered experiments.

The RESUM_FLEX reflog shows that `edba6a294581fde6f905b330d477f2c1b42d6adb` was the checked-out revision from May 18 until the September 8 update. This spans all priority CNP artifact timestamps on May 20 and is the strongest available upstream-revision provenance. It should be used as a source overlay in an ignored paper-local directory, leaving the current dependency checkout untouched.

## Missing evidence

- Original v9 full-test curve/export, context identity list, inference seed, and per-MC predictions.
- Exact MAJORANA source commit and clean/dirty state at each CNP training invocation.
- CNP training runtime, peak memory, verbose logs, and realized event exposure.
- Independent training seeds for the priority CNP variants.
- A threshold selected on data disjoint from the corresponding evaluation target.
- A controlled M0/M1/M2/M3 ladder sharing sampling, exposure, dropout, attention dimensions, and training seeds.
- A full-band decoder-mask control and an otherwise matched no-direct-contrast control.

These gaps do not justify retraining recovered models. They justify a frozen paper evaluator, paired inference first, and only then the smallest missing controlled training jobs described in `gap_decision.md`.
