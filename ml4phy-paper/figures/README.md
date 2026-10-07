# Paper figures

This directory contains compact visual exports from the paper campaigns. The
figures are review artifacts; CSV tables retain the underlying numerical
precision.

## Figure groups

- `extension_*` and `phase1_*`: workflow, context-size, kernel, and local-curve
  diagnostics.
- `phase2_*` and `phase3_*`: training-budget, subset-robustness, GP, local
  feature, and sparse-tail results.
- `paper_figure1_manuscript.png`: Figure 1 rendered directly from page 2 of the
  arXiv manuscript at 300 dpi. This is the image shown on the repository landing
  page.
- `paper_figure1_efficiency_curves.png`: the earlier presentation-export view,
  retained for provenance and compatibility with its recorded output hash.
- `paper_figure2_reference_pulls.png`: the presentation-focused reference-pull
  diagnostic.
- `mc_smoothness_diagnostic.png`: bounded numerical diagnostic; it does not
  establish calibrated uncertainty or noise-corrected physical smoothness.
- `cache_*`, `development_*`, and `final_*`: preserved historical and protocol
  development views.

Mechanism-control figures are kept under
`../review_followup_20260910/figures/` so that the additive campaign remains
self-contained. Do not infer a selection rule from figure appearance; use the
frozen reports and manifests.
