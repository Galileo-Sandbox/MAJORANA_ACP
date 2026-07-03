# experiments/

Historical ablation configurations preserved exclusively for
reproducibility of the intermediate sweeps that led to the shipped
architecture in `configs/cut_acceptance/simple_cnn_small/sweeps/cell17/`.

Nothing in this directory is required for the flagship notebook or
the canonical training pipeline. It exists so that:

- the ten `cell15_v{1..10}` variants that motivated the fixed-then-
  learnable-κ SFN gate can be rerun end-to-end,
- the 25 `hybrid_scale/*` context-sampling experiments that
  motivated the shift from mean/attention aggregation to the SAPE +
  SFN gate remain inspectable,
- and the `cell16` (unbounded κ) and `cell15_matched` /
  `base2_matched` counter-baselines stay archived beside their
  matched-budget siblings.

Each subdirectory mirrors the top-level `configs/` and (where a
trained artifact still exists on disk) `results/` layout, so the
same tooling — `python -m majorana_acp.cut_acceptance.cli <yaml>`
and `python -m scripts.diagnostics.cnp_test_inference <yaml>` —
works uniformly on both. Only each cell's `run_summary.json` is
git-tracked; large binaries stay ignored.
