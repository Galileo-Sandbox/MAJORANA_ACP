# Phase 1 neural inference timing pilot

Status: passed on 2026-09-09 before the complete Phase 1 neural matrix.

The pilot evaluated Density-guided CNP seed 0 with the frozen final context
seed 100, nested context size 500, fixed dropout seed 10100, 50 MC passes, a
1-keV grid, and all 114,400 target events. It is a valid prespecified Phase 1
cell and will be reused.

- Inference time: 10.3226 seconds.
- Total recorded evaluator time: 12.3685 seconds.
- Peak allocated GPU memory: 6,726.36 MiB on `cuda:0`.
- Pilot Brier score: 0.2238490813. This timing endpoint is not used for model
  or context selection.
- Context identity SHA-256:
  `62ee47d9f2640d82138ec9048b7a5137f4a1b10fd2fe9cff74b03f0e64087476`.

The Phase 1 matrix has 480 neural cells. Sixty compatible 2,000-context cells
are reused, leaving 420 new evaluations including this pilot. Multiplying the
representative slow-path total evaluator time by all 420 cells gives a
conservative projection of 5,194.79 seconds (86.58 minutes). After reusing the
pilot, the same calculation projects 5,182.42 seconds (86.37 minutes) for the
remaining 419 cells. This overestimates the expected compute because the three
non-density-guided architectures have historically been substantially faster;
it does not include aggregation time.

The measured cost and memory fit the current allocation. The complete neural
matrix is therefore authorized by the frozen gap decision. Runs remain
sequential, use unique output directories, and fail loudly. This decision does
not authorize Phase 2 or change the separate Gaussian-process two-hour gate.

