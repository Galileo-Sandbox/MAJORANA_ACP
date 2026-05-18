"""Tests for the localized peak-region goodness-of-fit metrics."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.diagnostics.cnp_test_inference import (
    PeakRegionMetrics,
    compute_peak_metrics,
)


def _make_grid(bin_width: float = 10.0, e_lo: float = 500.0, e_hi: float = 3000.0):
    centers = np.arange(e_lo + 0.5 * bin_width, e_hi, bin_width)
    return centers


def test_perfect_match_gives_chi2_near_zero_and_p_near_one() -> None:
    """If β == empirical rate exactly, χ² ≈ 0 and Z = 0 → p = 1."""
    centers = _make_grid()
    beta = np.full(centers.size, 0.90)
    beta_std = np.full(centers.size, 0.02)
    rate = beta.copy()
    sigma_emp = np.full(centers.size, 0.03)

    # Use a single peak inside the grid.
    out = compute_peak_metrics(
        centers,
        beta,
        beta_std,
        rate_DC=rate,
        sigma_emp_DC=sigma_emp,
        rate_DT=rate,
        sigma_emp_DT=sigma_emp,
        peak_markers=[("test_peak", 1500.0)],
    )
    pm = out[0]
    assert pm.peak_name == "test_peak"
    assert pm.peak_energy_kev == 1500.0
    assert pm.chi2_DT == pytest.approx(0.0, abs=1e-9)
    assert pm.chi2_DC == pytest.approx(0.0, abs=1e-9)
    assert pm.z_DT == pytest.approx(0.0, abs=1e-9)
    assert pm.p_DT == pytest.approx(1.0, abs=1e-6)


def test_one_sigma_offset_gives_chi2_about_one() -> None:
    """Empirical 1σ above β with σ_stat dominating → χ² ≈ 1."""
    centers = _make_grid()
    beta = np.full(centers.size, 0.50)
    beta_std = np.full(centers.size, 0.001)  # negligible model σ
    sigma_emp = np.full(centers.size, 0.05)  # dominant binomial σ
    rate = beta + sigma_emp  # exactly +1σ

    out = compute_peak_metrics(
        centers,
        beta,
        beta_std,
        rate_DC=rate,
        sigma_emp_DC=sigma_emp,
        rate_DT=rate,
        sigma_emp_DT=sigma_emp,
        peak_markers=[("test_peak", 1500.0)],
    )
    pm = out[0]
    # Each bin contributes (sigma_emp)² / (σ_stat² + σ_CNP²) ≈ 1.
    assert pm.chi2_DT == pytest.approx(1.0, rel=1e-3)


def test_nan_bins_are_excluded() -> None:
    """NaN empirical-rate bins drop out of n_valid and the averages."""
    centers = _make_grid()
    beta = np.full(centers.size, 0.5)
    beta_std = np.full(centers.size, 0.01)
    rate = np.full(centers.size, 0.5)
    sigma_emp = np.full(centers.size, 0.05)
    # Drop **every** bin in the peak window so n_valid → 0.
    peak_e = 1500.0
    in_window = np.flatnonzero(np.abs(centers - peak_e) <= 5.0)
    rate[in_window] = np.nan

    out = compute_peak_metrics(
        centers,
        beta,
        beta_std,
        rate_DC=rate,
        sigma_emp_DC=sigma_emp,
        rate_DT=rate,
        sigma_emp_DT=sigma_emp,
        peak_markers=[("test_peak", peak_e)],
    )
    pm = out[0]
    assert pm.n_bins_in_window == in_window.size
    assert pm.n_valid_DT == 0
    assert np.isnan(pm.chi2_DT) and np.isnan(pm.z_DT) and np.isnan(pm.p_DT)


def test_empty_window_yields_nan() -> None:
    """Peak whose window contains no bin centers → all NaN."""
    centers = _make_grid()
    beta = np.full(centers.size, 0.5)
    beta_std = np.full(centers.size, 0.01)
    rate = np.full(centers.size, 0.5)
    sigma_emp = np.full(centers.size, 0.05)

    # Pick a peak energy below the grid lo so no bin overlaps.
    out = compute_peak_metrics(
        centers,
        beta,
        beta_std,
        rate_DC=rate,
        sigma_emp_DC=sigma_emp,
        rate_DT=rate,
        sigma_emp_DT=sigma_emp,
        peak_markers=[("below_grid", 100.0)],
    )
    pm = out[0]
    assert pm.n_bins_in_window == 0
    assert pm.n_valid_DT == 0
    assert np.isnan(pm.chi2_DT)
    assert np.isnan(pm.p_DT)


def test_default_peak_list_runs_against_real_grid() -> None:
    """All four default PEAK_MARKERS produce metrics objects."""
    centers = _make_grid(bin_width=10.0)
    beta = np.full(centers.size, 0.5)
    beta_std = np.full(centers.size, 0.01)
    rate = np.full(centers.size, 0.5)
    sigma_emp = np.full(centers.size, 0.05)
    out = compute_peak_metrics(
        centers,
        beta,
        beta_std,
        rate_DC=rate,
        sigma_emp_DC=sigma_emp,
        rate_DT=rate,
        sigma_emp_DT=sigma_emp,
    )
    assert len(out) == 4
    names = [pm.peak_name for pm in out]
    assert names == ["Tl-208 FE", "Tl-208 SE", "Tl-208 DEP", "Bi-214 1620"]
    # All four energies sit inside [500, 3000]; with ±5-keV windows and
    # 10-keV bins each peak captures 1 bin (peak between centers) or
    # 2 bins (peak exactly on a bin edge — Bi-214 1620 sits between
    # the 1615 and 1625 centers, distance 5 to both, both ≤ half-w).
    for pm in out:
        assert pm.n_bins_in_window in (1, 2)
        assert pm.n_valid_DT == pm.n_bins_in_window


def test_dataclass_is_frozen() -> None:
    pm = PeakRegionMetrics(
        peak_name="x",
        peak_energy_kev=1.0,
        half_window_kev=5.0,
        n_bins_in_window=1,
        chi2_DC=0.5,
        z_DC=0.0,
        p_DC=1.0,
        n_valid_DC=1,
        chi2_DT=0.5,
        z_DT=0.0,
        p_DT=1.0,
        n_valid_DT=1,
    )
    with pytest.raises(Exception):  # FrozenInstanceError, but be liberal
        pm.peak_name = "y"
