# Paper tables

This directory contains lightweight machine-readable evidence exported from
the frozen campaigns. CSV and JSON values retain analysis precision; rounded
numbers in reports and figures are presentation views.

## Table families

- `data_*` and `context_draw_ledger.csv`: event budgets, overlaps, and context
  identities expressed as counts and hashes rather than raw identifiers.
- `phase1_*`: context-size, kernel, regional error, contrast, and curve results.
- `phase2_*`: original 2k/5k training-budget runs and summaries.
- `phase3_*`: additional 5k subsets, the 10k prefix, dense GP, and cross-budget
  comparisons.
- `paper_reference_*`, `paper_regional_*`, and `paper_continuum_*`: the
  presentation export based on saved actual-event predictions.
- `paper_coverage_*`: per-cell regional reference-band coverage and its
  hierarchical summaries.
- `mc_smoothness_*`: bounded numerical-noise and roughness diagnostics.

Rows should be interpreted with their generating manifest and report. Contexts
overlap, so cell rows are not independent experimental replicates. Large
event-level arrays remain on the server; these tables contain aggregates and
source hashes only.
