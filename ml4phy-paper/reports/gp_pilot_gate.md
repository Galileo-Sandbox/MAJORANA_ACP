# Dense Bernoulli-GP pilot and cost gate

Status: pilot completed; full GP campaign stopped for approval on 2026-09-09.

The required pilot fit scikit-learn 1.8.0 `GaussianProcessClassifier` to the
frozen 2,000-event development context (seed 100) using a logistic Bernoulli
likelihood, Laplace approximation, double precision, and
`ConstantKernel * RBF`. Energy was normalized as
`(energy_kev - 500) / 2500`. The initial physical length scale was 50 keV,
with the frozen 1--1,000-keV bounds. The covariance amplitude started at one
with bounds 0.01--100. Bounded L-BFGS-B used one initial fit and two restarts,
seed 31000, at most 200 optimizer iterations, and at most 100 Laplace
iterations. Predictions came from posterior-integrated `predict_proba`, not
hard labels or sigmoid of only the posterior mean.

## Measured result

- Fit: 228.6473 seconds.
- Prediction for 2,074 development target events: 8.1740 seconds.
- Prediction for 114,400 final-target energies: 27.6692 seconds.
- Total: 264.4929 seconds.
- Optimized kernel: `0.573**2 * RBF(length_scale=0.0469)`, corresponding to
  amplitude 0.3280205 and physical length scale 117.3084 keV.
- Development global Brier: 0.2302478; equal-supported-region Brier:
  0.2339544. These pilot values do not select a covariance family.
- All three optimizer calls converged in 5--7 iterations. No numerical warning
  was recorded.

The first attempt stopped before fitting because the SHA-priority row order was
not monotonically increasing for h5py indexing. Attempt 2 sorted row indices
only for data access; it preserved the exact frozen identity set. The empty
first-attempt directory remains on the server.

## Campaign projection and decision

The frozen campaign requires 80 development fits (two covariance families,
ten contexts, four sizes) and 40 final fits after family selection. Charging
the 2,000-event measured rate to every fit gives a deliberately crude upper
projection of 29,198.36 seconds (8.11 hours when each development fit includes
development prediction and each final fit includes final prediction).

A scaling-aware estimate is more favorable: cubic scaling for dense fitting
and linear scaling for target prediction from the 2,000-event pilot. It
projects 5,531.47 seconds for development family selection and 3,131.27 seconds
for final fitting/prediction, or **8,662.74 seconds (2.41 hours)** total. This
still exceeds the frozen two-hour stopping threshold.

Therefore the GP campaign was not launched. No context size was reduced, no
region was dropped, and the estimator was not replaced by Gaussian regression,
binning, or interpolation. Continuing requires approval of an amended,
separately labeled protocol, such as a resource-tested sparse variational
Bernoulli GP. A pooled-data GP remains outside scope.

