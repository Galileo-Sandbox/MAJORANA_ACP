# Lab Server Kickoff: Phase 3 Approval and Execution

Use this message only when the author intends to approve the scope below.

---

Please continue in the MAJORANA_ACP repository on branch `ml4phy-paper`, building on completed Phase 2 commit `04e3086bc9d857f419a665b039aa665a7805fcc0`.

I approve the following bounded Phase 3 campaign, subject to the resource gates below:

1. Complete the existing dense Bernoulli-GP campaign with a four-hour hard wall-clock limit for the remaining campaign. Preserve its existing statistical protocol. Stop and report partial completion at the limit; do not automatically switch to sparse GP.
2. Repeat the 5,000-event acceptance-training experiment using two additional outcome-blind training-pool orderings, seeds 20260910 and 20260911. Use all four neural architectures, initialization seeds 0–2, and the same ten 500-event contexts: 24 new training jobs and 240 evaluations.
3. Add a 10,000-event prefix of the original Phase 2 training-pool ordering, shared by all four architectures and three initialization seeds: 12 new training jobs and 120 evaluations.

Read `ml4phy-paper/PHASE3_DATA_EFFICIENCY_PLAN.md` for the detailed protocol. If it is missing because the laptop files have not yet been pushed, request that document before training; do not reconstruct missing settings from this summary.

Before new training, audit reusable artifacts, freeze subset hashes and configurations, verify nested prefixes and train/target separation, and estimate resource use. The neural campaign is capped at two hours, separate from the GP cap. If the updated projection exceeds that cap, report and request revised approval before launching. At any runtime cap, retain partial results and report rather than shrinking the agreed matrix.

Keep the classifier, threshold, 3,000-step schedule, final target set, context draws, and Phase 2 evaluation settings fixed. Each model's density buffer must use only its matching training subset. Preserve all previous results, plans, and unrelated local files. Keep all training and heavy inference on this server.

Our scientific question is whether fewer acceptance-training events produce competitive local reconstruction, conditional on the fixed pretrained classifier. Do not assume success. Show the 2,000-event failure and sparse-tail limitations. Report training-subset, initialization, and context variation separately. Do not claim calibrated dropout intervals, MC-curve smoothness, exact minimum sample requirements, or end-to-end training on only 5,000 events.

Use compact budget labels 2k, 5k, 10k, and 18.9k (full pool), with exact counts in the ledger. Do not expand or relabel the 18,866-event pool as 20,000. No sparse GP, classifier retraining, new uncertainty method, or broader factorial campaign is authorized.

Produce the reports, tables, figures, and provenance specified in the plan. All authored text and code must be English. Keep checkpoints and event-level predictions on the server. Validate and commit/push lightweight deliverables to `ml4phy-paper`; verify the remote head and report completed/remaining cells, runtime, and limitations. Do not overwrite unrelated changes or force-push.
