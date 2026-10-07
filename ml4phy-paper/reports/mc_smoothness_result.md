# MC-noise and smoothness diagnostic

Status: complete for the two seed-0 neural models on final context seed 100 at
n=2,000. This fixed diagnostic does not select a favorable context or training
seed.

| Method | Final passes | Region | Two-stream RMSE (pp) | Empirical-bin coverage | Mean interval width (pp) |
|---|---:|---|---:|---:|---:|
| Attentive CNP + PE | 800 | continuum_1700_2000 | 0.337 | 0.033 | 1.130 |
| Attentive CNP + PE | 800 | continuum_2200_2400 | 0.284 | 0.000 | 0.794 |
| Density-guided CNP (ours) | 200 | continuum_1700_2000 | 0.197 | 0.033 | 0.313 |
| Density-guided CNP (ours) | 200 | continuum_2200_2400 | 0.197 | 0.038 | 0.274 |

The 50-pass estimate is the exact prefix of each 200-pass stream. An 800-pass
extension was run only where the frozen numerical-noise trigger fired and the
measured projection was below 30 minutes. Every pass evaluated all 114,400
actual target energies and the complete 0.5-keV grid in one forward call. No
energy chunking was used. Ordinary elementwise dropout was retained; masks
were not forced to be identical across energies, while each query retained its
intended marginal estimator. The 1-keV diagnostic is a subset of the same
0.5-keV function draws, and curvature is divided by squared grid spacing.

Dropout-disabled predictions are reported as a separate deterministic network
estimator, not as the infinite-MC limit. Continuum MAE is listed beside every
roughness value in the CSV. The interval is the central 68% distribution of
bin means formed by averaging each stochastic function draw over actual events
in the bin before taking quantiles. Coverage checks the finite empirical bin
acceptance and is only a consistency diagnostic, not calibrated confidence
coverage. Width is reported beside coverage; higher coverage alone is not
interpreted as better.

## Claim decision

Attentive CNP + PE still has independent-stream RMSE of 0.284--0.337
percentage points after 800 passes. For Density-guided CNP, 200-pass
independent-stream RMSE is about 0.197 percentage points, but its normalized
curvature remains strongly grid-dependent: about 0.010--0.012 on the 0.5-keV
grid versus 0.0029--0.0039 on the 1-keV grid. Its dropout-disabled normalized
curvature is only about 0.0016--0.0018. Numerical estimator noise therefore
remains material, so the paper must omit a claim that our MC-averaged curve is
smoother. Local-shape error and contrast results remain the appropriate
evidence.

The observed 68% dropout-bin coverage is only 0--3.8% while mean widths are
about 0.27--1.13 percentage points. These narrow bands strongly under-cover
the finite empirical reference and are not calibrated uncertainty intervals.

NumPy emitted `All-NaN slice encountered` while forming quantiles for empty
full-range bins. Coverage calculations subsequently restrict to bins with at
least four events, so no reported coverage or width used an empty bin. The
warning is retained in provenance rather than hidden.
