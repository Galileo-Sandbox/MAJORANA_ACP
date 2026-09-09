# Lab Server Codex Kickoff

We are preparing a four-page ML4PS 2026 workshop paper using the MAJORANA_ACP repository. Work on the `ml4phy-paper` branch. The manuscript is maintained separately on my laptop; your responsibility is the experimental evidence and reproducible result exports in this repository.

Read any applicable AGENTS.md instructions, then read these files before acting:

1. `ml4phy-paper/README.md`
2. `ml4phy-paper/EXPERIMENT_PLAN.md`
3. `ml4phy-paper/SERVER_WORKFLOW.md`
4. `ml4phy-paper/inventory.json` as a historical laptop inventory, not a description of this server's available artifacts.

Our agreed scope is application-specific: improve energy-dependent inclusive cut-acceptance estimation in MAJORANA using energy-spectrum density information to balance sharp local features and smooth continuum behavior. MAJORANA is our only required real dataset. Do not add unrelated datasets, synthetic benchmarks, or broad hyperparameter searches. All documentation, code, comments, reports, and figure labels you write must be in English. We may converse in Chinese or English.

Preserve all pre-existing research source files, configurations, notebooks, caches, results, environments, and checkpoints. Put all new code, configurations, reports, and outputs under `ml4phy-paper/`. Never overwrite a previous run. Inspect the worktree before Git operations and preserve unrelated changes. Do not run destructive cleanup or force-push.

First audit this server and mine existing evidence before launching any training:

- Locate existing classifier predictions, CNP checkpoints, resolved configurations, full-test exports, split/event identities, training logs, and the working RESUM_FLEX dependency/environment. The laptop had 58 run summaries and five prediction caches but no model weights or HDF5 files; the server may already have everything needed.
- Prioritize the talk's `Cell 15 v9` model, then v5, Cell 17, and the CNP/attention/Fourier baselines. The talk used v9 with fixed kappa=4, 2,000 context events, and 139,474 target events. The five laptop caches instead contain 2,000 context and 5,074 target events. Recover the actual provenance rather than merging these results.
- Read actual configuration values and checkpoint metadata. Existing `matched` names do not establish matched sampling, context exposure, dropout, or architecture.
- Complete Phase A of the plan: artifact recovery, cache-only reanalysis, and a concrete `reports/gap_decision.md` explaining which evidence is reusable, which requires inference, and which genuinely requires training.

There are known evaluation issues to address in new paper-specific adapters, without editing the original implementation:

- The legacy evaluator selects the Youden-J threshold from the evaluation file before splitting context and target. Freeze a threshold on designated calibration/development data and record its provenance.
- The notebook can return persisted predictions before applying requested live settings. Validate cache identity and all protocol metadata.
- Historical peak p-values are not suitable as calibrated significance evidence; single-bin agreement does not establish a recovered local shape.
- The density gate uses a fixed training energy pool, and the decoder also receives the density contrast directly. Disclose these information paths and isolate the claimed mechanism where necessary.
- Repeated context draws, MC dropout passes, and independent training seeds are different sources of variation. Keep them separate.

After writing the gap decision, implement and validate the missing evaluator, matched configurations, runner, and result exporter. Reuse the working server environment after recording its versions. Inspect the actual GPU/scheduler setup; do not assume a particular scheduler or overwrite the environment. Start with an inference pilot. Run new training only for gaps that cannot be resolved with recovered artifacts or inference; follow the conditional queue and stopping rules in the plan. You may proceed with necessary in-scope experiments after establishing these prerequisites. Ask me only for missing artifact locations, access, or scientific choices that cannot be resolved from the repository and agreed scope.

Use a fixed classifier, documented event identities and data roles, paired context samples, a development-selected kernel-regression baseline, and joint peak/continuum evaluation. Report failed regions and unfavorable comparisons. Do not describe previously inspected target data as an untouched final test set.

Keep raw data, weights, large logs, and machine-specific paths on the server in the ignored locations described by SERVER_WORKFLOW.md. Return compact metrics, selected figures, portable configurations, provenance manifests, and necessary compact prediction exports through Git. Record source commit, configuration and artifact hashes, counts, threshold, seeds, inference caps, MC settings, runtime, and completion status for each run. Do not include credentials or raw event-level data in commits.

Commit and push reviewed lightweight progress/results to `origin/ml4phy-paper` at meaningful milestones so I can pull them on my laptop. Review staged content before each commit; never force-push. Start by reporting the server artifact inventory, what can be reused immediately, and the smallest remaining experiment set. Do not start by retraining everything.
