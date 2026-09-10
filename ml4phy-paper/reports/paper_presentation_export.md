# Existing-prediction export for the talk-led paper

Status: complete. This export recovered existing saved predictions only; it ran no
training, checkpoint inference, MC prediction, GP fitting/prediction, bandwidth
selection, or data selection. New text uses *efficiency* for the probability of
passing the fixed classifier cut; legacy artifact field names retain *acceptance*
for compatibility.

## Recovered matrix

- Neural: 600/600 cells (original-order 2k, 5k, and 10k plus two additional 5k
  orderings; four architectures, three initialization seeds, ten 500-event
  contexts).
- Headline neural comparison: 120/120 original-5k cells.
- Classical headline: 10/10 context-only kernel cells and 10/10 selected
  Bernoulli-GP Matérn-3/2 cells at context size 500.
- Stronger-data control: 10/10 pooled-data kernel cells, labeled separately.
- Every cell uses exactly the same frozen 114,400 target rows and the matching
  context subset. Artifact and manifest hashes were checked before aggregation.

## Table 1: regional Gaussian agreement proxy

Each entry is the mean of per-cell `G1/G2/G3`, with each cell computed before
aggregation. Neural cells use the historical combined proxy; classical combined
proxy values are unavailable because no model SD was saved. The companion column
is the equal-window continuum pull RMS on the common finite-reference scale.

| Method | Overall | FE | SE | DEP | feature (1620 keV) | Continuum pull RMS |
|---|---:|---:|---:|---:|---:|---:|
| CNP | 0.42/0.76/0.94 | 0.64/0.93/0.99 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 1.78 |
| Attentive CNP | 0.48/0.77/0.93 | 0.56/0.88/0.98 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 2.05 |
| Attentive CNP + PE | 0.65/0.94/1.00 | 0.65/0.94/1.00 | 0.68/0.95/1.00 | 0.67/0.95/1.00 | 0.66/0.94/1.00 | 3.88 |
| Density-guided CNP | 0.65/0.94/0.99 | 0.37/0.68/0.89 | 0.62/0.92/0.99 | 0.60/0.91/0.99 | 0.17/0.51/0.84 | 1.26 |
| Kernel | NA | NA | NA | NA | NA | 2.10 |
| Bernoulli GP | NA | NA | NA | NA | NA | 2.22 |

The executable notebook formula was followed: `G_k = Phi(k-z) - Phi(-k-z)`
with `z=(f-p)/sqrt(s_emp^2+u_proxy^2)`. The notebook docstring instead describes
a transform with `s_emp` in the Gaussian-CDF denominator; that prose is
inconsistent with the executable implementation. Here `u_proxy` is the mean
pointwise neural prediction SD and is **not** the SD of a pooled region mean.
Wider proxies can improve G without improving prediction. G is a transform of
one pooled residual, not repeated-sampling coverage or a calibrated posterior
probability. Regions use only frozen targets: Overall is inclusive 500--3000 keV;
FE, SE, DEP, and the 1620-keV feature use inclusive +/-5-keV event windows.

For a fair all-method view, the separate reference-only panel uses `s_emp` alone:

| Method | Overall | FE | SE | DEP | feature (1620 keV) |
|---|---:|---:|---:|---:|---:|
| CNP | 0.00/0.00/0.00 | 0.23/0.44/0.59 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0.00/0.00/0.00 |
| Attentive CNP | 0.00/0.00/0.00 | 0.00/0.00/0.03 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0.00/0.00/0.00 |
| Attentive CNP + PE | 0.02/0.07/0.16 | 0.02/0.05/0.10 | 0.50/0.82/0.95 | 0.27/0.58/0.83 | 0.29/0.60/0.83 |
| Density-guided CNP | 0.14/0.27/0.37 | 0.03/0.09/0.16 | 0.36/0.69/0.91 | 0.42/0.75/0.93 | 0.00/0.01/0.08 |
| Kernel | 0.00/0.02/0.07 | 0.04/0.09/0.14 | 0.00/0.00/0.00 | 0.00/0.00/0.01 | 0.00/0.00/0.00 |
| Bernoulli GP | 0.05/0.09/0.12 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0.04/0.08/0.10 |

The off-feature companion diagnostics make the interpretation testable:

| Method | Continuum pull RMS | Roughness 1700--2000 | Roughness 2200--2400 | Overall combined-proxy full width (pp) |
|---|---:|---:|---:|---:|
| CNP | 1.78 | 0.003297 | 0.004243 | 5.77 |
| Attentive CNP | 2.05 | 0.003294 | 0.004677 | 6.27 |
| Attentive CNP + PE | 3.88 | 0.2448 | 0.1158 | 20.06 |
| Density-guided CNP | 1.26 | 0.006967 | 0.01538 | 5.66 |
| Kernel | 2.10 | 2.772e-06 | 6.267e-06 | NA |
| Bernoulli GP | 2.22 | 0.0008477 | 0.001141 | NA |

The PE model's strong combined-proxy G values coincide with a much wider proxy,
the largest continuum pull RMS, and substantially larger saved-grid roughness.
Thus those G values do not establish better reconstruction. In this matrix the
density-guided model has the lowest continuum pull RMS while retaining much
lower roughness than PE, which supports a useful balance interpretation. CNP
and Attentive CNP are comparatively smooth but their near-zero pooled G values
at SE, DEP, and the 1620-keV feature show that broad-trend behavior alone misses
localized changes. This is a descriptive comparison on one historically
exposed target, not proof of overfitting or calibrated uncertainty.

## What the saved predictions support

The exports let Table 1 complement, rather than duplicate, Figure 1. The
regional pooled scores quantify broad and localized agreement, while Figure 2
shows signed deviations bin by bin. The continuum residual table reports bias,
centered RMS, and pull RMS, so alternating over- and underprediction cannot be
hidden by regional pooling. Its combined row gives equal weight to the
1700--2000 and 2200--2400-keV windows: mean-square quantities are averaged
between windows before taking a square root, while signed means are averaged
between window means.

The density-guided model can be described as balancing local reconstruction and
off-feature discrepancy only where the reported regional and continuum values
show that pattern. These diagnostics do not establish universal superiority.
The 2k failure, 10k nonmonotonicity, and sparse-tail limitation from Phase 3
remain part of the result. In particular, small training subsets contain no
sample-eligible sparse-tail event, and the 5k density-guided sparse-tail error
remains unfavorable.

The retained density-guided budget result is explicit:

| Budget | Exact nominal events | Sampling-eligible events | Peak MAE (pp) | Continuum MAE (pp) |
|---|---:|---:|---:|---:|
| 2k | 2,000 | 1,895 | 9.40 | 12.89 |
| 5k, original ordering | 5,000 | 4,984 | 5.29 | 3.96 |
| 10k | 10,000 | 9,980 | 5.84 | 4.01 |
| 18.9k full pool | 18,866 | 18,836 | 3.56 | 4.07 |

The 10k point is nonmonotonic relative to 5k, so these data do not identify an
exact minimum sample requirement. The 5k density-guided sparse-tail MAE is
19.39 pp, compared with 2.99 pp for the stronger-data pooled-kernel control.
The full pool is exactly 18,866 events and is never relabeled 20k. All neural
claims remain conditional on the classifier pretrained with 18,866 selected
events from 377,330 candidates; they are not end-to-end 5k claims.

Grid roughness is the saved-protocol mean absolute second difference at the
original 1-keV spacing, reported per cell and again for each mean curve. It
combines learned variation and MC noise. Therefore the defensible phrase is
*excessive variation* or *off-feature discrepancy*, not overfitting. The prior
MC-precision study does not match this comparison: it used full-pool PE and
density-guided models at seed 0/context 100 with 2,000 context events, whereas
the headline is original 5k with 500 context events. Its conclusions are not
generalized to these cells.

## Figure 2 uncertainty boundary

All energy-resolved pulls use actual-target-event bin means and the Wilson z=1
half-width as a common reference scale. `C_k` is the fraction of supported bins
with `abs(z_ref)<=k`; it is distinct from Table 1's pooled-residual `G_k`.
Neither is coverage of unknown true efficiency. The Wilson half-width is a
descriptive finite-reference scale, not an exact standard error.

No valid combined bin-mean model uncertainty can be reconstructed. Neural NPZ
files retain only pointwise event SDs; without per-pass bin means or within-pass
covariances, the SD of a bin mean is unidentified. GP and kernel files retain no
predictive SD. Combined pulls, combined `C_k`, and comparable interval widths
are therefore marked unavailable rather than fabricated.

A definitive overfitting attribution remains blocked by missing matched
training/reference and numerical-precision outputs. A separate bounded proposal,
not executed here, would freeze the original-5k seed-0/context-100 checkpoints
for all four architectures, save per-pass bin means on both the matching
training pool and frozen target under two prespecified independent nested
50/200-pass streams, and report generalization-gap and stream-stability
diagnostics in the same fixed windows. It would require explicit approval and a
measured inference cost gate; context fit versus target fit alone would still
not be treated as proof of overfitting.

## Portable panels

- `paper_figure1_efficiency_curves.png`: the prespecified illustrative fit
  (original 5k, initialization seed 0, context seed 100), shown both over the
  full 500--3000-keV range and the 1500--3000-keV presentation view. It includes
  the target reference, context points, four neural methods, context-only kernel,
  pooled-data kernel control, and Bernoulli GP.
- `paper_figure2_reference_pulls.png`: energy-resolved reference-only pulls for
  the mean saved curve in both fixed continuum windows. Mean curves are labeled
  as averages, not deployable single fits.
- CSVs contain the illustrative grids, all 5k-subset mean grids, reference and
  context counts, per-cell regional/continuum/pull diagnostics, hierarchical
  summaries, fixed pull histograms, and compact illustrative/mean bin
  predictions. Complete per-cell bin arrays remain in the server-only NPZ files.

## Validation and interpretation limits

Direct event-level bin means were reconciled against the saved 5-keV-bin arrays
and archived MAE/RMSE summaries. The maximum absolute metric discrepancy was
`2.039e-09` in probability units (required tolerance `2e-14` for float64 neural/GP
artifacts and `2e-8` only for float32 kernel archive comparisons).

The frozen target was historically inspected, including for the talk. This is a
prospectively specified follow-up analysis on historically exposed data, not an
untouched test. The 1620-keV structure remains named by energy because its
physical isotope identity is unresolved in audited sources. Context draws
overlap and share one finite target, so context, initialization, and training-
subset dispersions are descriptive and are not treated as independent-dataset
uncertainty. Hierarchical summary CSVs report within-seed context SD,
initialization SD across seed means, and—on dedicated all-three-5k rows—training-
subset SD across subset means.
