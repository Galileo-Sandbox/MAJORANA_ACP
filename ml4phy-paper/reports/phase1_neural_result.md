# Phase 1 neural context-size result

Status: complete. The matrix contains four architectures, three training seeds,
ten overlapping context draws, and four nested context sizes (480 cells). Sixty
compatible 2,000-context cells and one timing-pilot cell were reused; 419 cells
were newly evaluated. The campaign recorded no failed scientific runs.

## Main local-shape result

Values are mean ± SD across the three training-seed means, in percentage
points. Each seed mean averages the same ten context draws.

| Method | Peak MAE, n=500 | Continuum MAE, n=500 | Peak MAE, n=2000 | Continuum MAE, n=2000 |
|---|---:|---:|---:|---:|
| CNP | 7.993 ± 0.349 | 4.585 ± 0.022 | 7.993 ± 0.358 | 4.592 ± 0.014 |
| Attentive CNP | 8.245 ± 0.321 | 5.533 ± 0.699 | 8.257 ± 0.310 | 5.530 ± 0.690 |
| Attentive CNP + PE | 5.996 ± 1.078 | 7.380 ± 0.865 | 6.057 ± 1.069 | 7.383 ± 0.931 |
| Density-guided CNP (ours) | 3.557 ± 0.332 | 4.072 ± 0.109 | 3.480 ± 0.319 | 4.082 ± 0.109 |

Peak MAE is the equal average of the prespecified DEP, 1,620.74-keV Bi-212,
SE, and FE regional 5-keV-bin MAEs. Continuum MAE equally averages the
prespecified 1,700--2,000 and 2,200--2,400-keV regions. Full regional values,
RMSE, support, excluded bins, contrast errors, sparse-tail results, and Brier
checks remain in the CSV files at full precision.

## Variation and uncertainty boundary

The context-variation table reports within-training-seed SD and range across
the ten overlapping contexts. Training-seed variation is reported separately
as the SD of three context-averaged seed estimates. These are sensitivity
diagnostics, not independent-dataset standard errors, because contexts overlap
and every cell shares the same 114,400-event target.

The saved 50-pass files contain means and pointwise standard deviations, not
the stochastic function draws required to aggregate uncertainty over a bin.
No binned coverage or interval-width result is therefore invented from these
files. Model posterior, repeated-context, and finite-reference uncertainty are
not treated as interchangeable.

## Provenance remediation

Twenty-five original M0 cells recorded a dirty worktree while resume and
temporary-directory orchestration code was being edited. No numerical model
code changed, but those cells are excluded from this aggregation. They were
re-evaluated from a clean commit in unique `-cleanrerun` directories. Every
event-prediction and curve array agreed with its preserved original to absolute
tolerance 1e-12. The original files and the comparison record remain on the
server; this manifest records the comparison-record hash.

## Interpretation

The reference acceptance is a finite noisy estimate, not exact ground truth.
This is a prospectively specified follow-up on historically exposed data. The
study measures context efficiency conditional on each disclosed pretrained
acceptance model and classifier; it is not evidence of lower total end-to-end
data use. Brier remains a secondary check rather than the main evidence for
local-shape reconstruction. Phase 2 was not started.
