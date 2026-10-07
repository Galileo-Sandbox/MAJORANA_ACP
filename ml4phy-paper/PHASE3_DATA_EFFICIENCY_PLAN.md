# Phase 3: Data-Efficiency Confirmation

Date: 2026-09-09.
Evidence baseline: `04e3086bc9d857f419a665b039aa665a7805fcc0`.
Status: proposed bounded server campaign; execution requires author approval.

## Purpose and precedence

This additive plan defines the proposed next campaign after completed Phases 0–2. It does not modify their protocols, outputs, or historical status. Preserve CLASSICAL_BASELINES_PLAN.md and DATA_EFFICIENCY_PLAN.md, including untracked laptop copies.

Test whether fewer acceptance-training events suffice for useful local reconstruction, conditional on the fixed classifier. Do not assume that the method works best at every budget or that its relative advantage grows as data decrease.

At context size 500, ours trained on 5,000 events has mean peak/continuum MAE of 5.3/4.0 pp, compared with 8.0/4.6, 8.2/5.5, and 6.0/7.4 pp for CNP, Attentive CNP, and Attentive CNP + PE trained on 18,866 events. This is an observed cross-budget mean comparison, not a minimum-sample-complexity estimate or an established fourfold saving at a prespecified accuracy threshold.

The 2,000-event degradation and sparse-tail failures remain visible. The classifier uses 18,866 selected events from 377,330 candidates; acceptance training reuses those identities. Never describe this as end-to-end training on only 500 or 5,000 events.

## Budget presentation

Use 2k, 5k, 10k, and 18.9k (full pool) as compact figure labels, retaining exact nominal and sampling-eligible counts in tables and provenance. The existing full pool contains 18,866 unique events. Do not relabel it 20,000, duplicate events to reach 20,000, or import additional training events for cosmetic rounding.

## A. Complete the existing dense GP comparison

Follow the existing dense Bernoulli-GP protocol and development-only kernel selection. Amend only the campaign budget: propose a four-hour hard wall-clock limit for the remaining campaign. Record the timer boundary before starting. Save incremental results; at the deadline stop, retain partial work, and report completeness without presenting a partial campaign as complete. Do not switch to sparse GP automatically.

The prior 2.41-hour scaling-aware estimate is not a guarantee. Existing pilot evidence can be reused only with matching protocol and provenance. GP is context-only and does not establish equal-total-information superiority over classical methods. A pooled-data GP remains outside scope.

## B. Repeat the central 5,000-event result across training subsets

- Freeze two additional outcome-blind identity orderings, using ordering seeds 20260910 and 20260911. Record the exact ordering algorithm and hashes before training; audit compatibility with the original ordering implementation rather than assuming its seed convention.
- Use each ordering's 5,000-event prefix, shared by all four neural architectures and initialization seeds 0, 1, and 2.
- This adds 24 training jobs and 240 evaluation cells. The original subset supplies a third ordering and is reused only where compatible.
- These are independent random subset selections from one finite parent pool, not independent datasets; subsets can overlap. Report overlap counts.
- Restrict the density buffer to each matching subset. Record nominal, density-buffer, and sampling-eligible counts and regional support.
- Keep the frozen classifier, threshold, 3,000-step training schedule, 500-event contexts, ten context draws, 50-pass estimator, and 114,400 targets unchanged. Preserve all other Phase 2 settings.

## C. Add the missing intermediate training budget

Use the 10,000-event prefix of the original Phase 2 ordering. Do not regenerate or reorder the original 2,000- or 5,000-event prefixes. Verify nesting with hashes before training.

Train all four architectures with initialization seeds 0, 1, and 2; evaluate the same ten 500-event contexts. This adds 12 training jobs and 120 cells. Restrict the density buffer and audit effective support as in B. Do not run a full subset-by-budget-by-context factorial grid.

## Cost and approval gate

The proposed neural scope is exactly 36 new training jobs and 360 new evaluation cells. Scaling Phase 2's measured 29.9-minute training and 14.3-minute evaluation times gives approximately 66 minutes, or 83 minutes with a 25% allowance. This is provisional: budget-dependent density inference costs and server conditions can change it.

Before launching the neural campaign, prepare configurations, reuse checks, and a measured or conservative resource projection. Propose a two-hour neural campaign limit, separate from GP's four-hour limit. If projected neural cost exceeds two hours, request revised authorization before launching the campaign. Stop at the agreed runtime limit, preserve partial outputs, and report. Do not reduce seeds or drop methods to conceal an incomplete matrix.

Suggested order: GP completion, subset replication, then 10,000-event slice. Each module remains separately resumable. Merely receiving this plan does not authorize training; explicit author approval of its bounded scope is required.

## Analysis and reporting

1. Preserve existing peak and continuum regional definitions, binning, target support rules, and MAE aggregation. Report sparse-tail diagnostics separately without hiding them.
2. Plot error against acceptance-training budget, fixing and labeling context size 500. Include all measured budgets and unfavorable results. Show seed dispersion honestly, not as a confidence interval.
3. At 5,000 events, show each training-subset result and method differences within each shared subset. Separate initialization variation, context variation, and between-subset variation. Three subsets do not justify precise population-level uncertainty claims.
4. Compare each 5,000-event ours subset against the fixed full-budget neural baselines. Label the shared baseline and reference; do not count repeated comparisons as independent baseline replications.
5. Report whether 10,000-event baselines catch up. Do not infer monotonic learning curves or interpolate a precise transition budget from a few points.
6. Reuse existing peak/sideband contrast exports and region-wise diagnostics to examine physical structure recovery. Separate context-induced variation from MC estimator noise. Do not claim smoothness or calibrated uncertainty from the existing dropout diagnostics.
7. Do not select a success threshold after seeing results. If no independently justified physical tolerance exists, report direct error-budget comparisons rather than a minimum-data-to-threshold claim.
8. Record runtime and peak memory: practical usefulness includes computational cost as well as event budget. Existing runtime observations are not automatically controlled benchmarks.

## Deliverables

Add new versioned configurations and manifests plus:

- reports/phase3_protocol.md
- reports/phase3_result.md
- tables/phase3_training_subset_summary.csv
- tables/phase3_training_budget_summary.csv
- tables/phase3_cross_budget_comparison.csv
- figures/phase3_training_budget_efficiency.png
- figures/phase3_subset_robustness.png
- manifests/phase3_result.json

Use existing GP output conventions, linking its completion or partial-status report from phase3_result.md. Keep checkpoints, event-level predictions, and raw identities on the server. Commit only lightweight reproducible code, reports, aggregates, figures, and hash-bearing provenance. All authored artifacts must be English. Commit/push on ml4phy-paper after validation when authorized, and verify the actual remote branch head.

No classifier retraining, 20,000-event expansion, sparse GP, new uncertainty method, gate ablation, additional dataset, or broader grid is included.
