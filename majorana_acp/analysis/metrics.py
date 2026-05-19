"""Sawtooth-diagnostic metric suite for CNP-predicted β(E) curves.

The CNP inference path produces a dense smooth-looking continuous curve
``β̂(E)`` sampled on an evenly-spaced energy grid. Visual inspection of
the inclusive-cell audits revealed high-frequency sawtooth oscillations
that were impossible to summarise with a single scalar (Pearson r,
χ²_DT, etc.) — those compare β̂ to data, not β̂'s own roughness.

This module provides a three-axis decomposition of curve roughness that
lets us audit each cell for sawtooth-style artifacts retroactively:

* **MASD — Mean Absolute Second Difference.** Pure amplitude axis:
  the average physical magnitude of the curve's local curvature.
  Scales with the size of the wiggles, completely independent of how
  frequently they occur.

* **ED — Extrema Density.** Pure frequency axis: count of local maxima
  + minima per keV. Captures how often the curve changes direction,
  ignoring how big the swings are.

* **Lag-1 ACF of First Difference — Pattern axis.** Pearson correlation
  between successive first differences. Smooth monotone slopes → ≥ 0;
  pure random noise → ≈ −0.5; deterministic up-down-up-down sawtooth →
  −1. Distinguishes "structured oscillation" from "random wiggle" even
  when MASD and ED agree.

All three are evaluated inside narrow Compton-continuum windows
deliberately chosen to avoid known γ-peaks, so we're measuring the
model's behaviour on flat physics where the truth is "smooth slope".

The function is pure NumPy with no dependencies on the CNP pipeline so
it can be reused outside the cut-acceptance audit (other 1D-regression
diagnostics) without dragging in the rest of ``majorana_acp``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["analyze_sawtooth_suite"]


def analyze_sawtooth_suite(
    energies: np.ndarray,
    pass_rates: np.ndarray,
    region_range: tuple[float, float],
) -> dict[str, float]:
    """Compute the three sawtooth-diagnostic metrics over an energy window.

    Parameters
    ----------
    energies
        1D array of energy values (keV) parallel to ``pass_rates``.
        Need not be sorted — but ``region_range`` selects values by
        inclusive interval ``[lo, hi]``.
    pass_rates
        1D array of CNP-predicted continuous pass rates ``β̂(E)`` at
        the energies above. Same shape as ``energies``.
    region_range
        ``(lo, hi)`` energy bounds (keV) of the evaluation window.
        Only points with ``lo ≤ E ≤ hi`` contribute.

    Returns
    -------
    dict
        Three scalars under the keys:

        * ``"masd"`` — Mean Absolute Second Difference
          ``⟨ |β̂_{i+1} − 2β̂_i + β̂_{i−1}| ⟩`` over the masked region.
          Units: dimensionless (same as ``β̂``).

        * ``"extrema_density"`` — Extrema per keV. Number of indices
          ``i`` with a sign change in the first difference, divided
          by the width (keV) of the masked window. Units: keV⁻¹.

        * ``"lag1_acf"`` — Pearson correlation between successive
          first-difference samples ``Δβ̂_i = β̂_{i+1} − β̂_i`` and the
          lag-1 shifted ``Δβ̂_{i+1}``. Dimensionless, ``∈ [-1, 1]``.

    Edge cases:

    * Fewer than 4 samples in the window → all three metrics return
      ``0.0`` (insufficient data to compute the second-order stat).
    * Constant-valued ``β̂`` (zero first-difference variance) → ACF
      returns ``0.0`` instead of NaN. MASD and ED still reflect the
      true zero-roughness signal.
    """
    energies = np.asarray(energies, dtype=np.float64)
    pass_rates = np.asarray(pass_rates, dtype=np.float64)
    if energies.shape != pass_rates.shape:
        raise ValueError(
            f"energies and pass_rates must have the same shape, "
            f"got {energies.shape} vs {pass_rates.shape}"
        )
    if energies.ndim != 1:
        raise ValueError(f"energies must be 1D, got shape {energies.shape}")

    lo, hi = float(region_range[0]), float(region_range[1])
    if not hi > lo:
        raise ValueError(f"region_range must satisfy hi > lo, got {region_range}")

    mask = (energies >= lo) & (energies <= hi)
    x = energies[mask]
    y = pass_rates[mask]

    # Need at least 4 samples for a well-defined extrema density
    # (3 first differences, 2 second differences). Treat sparser
    # windows as un-diagnosable and return zeros.
    if y.size < 4:
        return {"masd": 0.0, "extrema_density": 0.0, "lag1_acf": 0.0}

    # 1. MASD — mean of |second difference|.
    masd = float(np.mean(np.abs(np.diff(y, n=2))))

    # 2. Extrema density — count of sign changes in first difference,
    # divided by the actual span (keV) of the masked window.
    dy = np.diff(y)
    is_extremum = (dy[:-1] * dy[1:]) < 0
    span = float(x[-1] - x[0])
    extrema_density = float(np.sum(is_extremum)) / span if span > 0 else 0.0

    # 3. Lag-1 ACF of first difference. NaN occurs when ``dy`` is
    # constant (zero variance) — return 0.0 in that case so callers
    # don't have to special-case it.
    if dy.size > 1:
        corr = np.corrcoef(dy[:-1], dy[1:])
        c01 = corr[0, 1]
        lag1_acf = float(c01) if np.isfinite(c01) else 0.0
    else:
        lag1_acf = 0.0

    return {"masd": masd, "extrema_density": extrema_density, "lag1_acf": lag1_acf}
