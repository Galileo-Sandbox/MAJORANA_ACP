"""Tests for the sawtooth-diagnostic metric suite.

The three metrics are designed to be complementary — MASD = amplitude,
ED = frequency, ACF1 = pattern. The tests below exercise each axis in
isolation against curves whose roughness behaviour is known by
construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from majorana_acp.analysis.metrics import analyze_sawtooth_suite

# -- Construction helpers --------------------------------------------------- #


def _energy_grid(lo: float = 1700.0, hi: float = 2000.0, n: int = 301) -> np.ndarray:
    """Evenly-spaced E grid covering one of the audit's control regions."""
    return np.linspace(lo, hi, n)


# -- Smooth curves should produce near-zero roughness ---------------------- #


def test_constant_curve_has_zero_metrics() -> None:
    """A flat β̂ ≡ const has zero second difference and no extrema.
    The ACF reduces to 0/0 → safely returns 0.0 by the NaN guard."""
    x = _energy_grid()
    y = np.full_like(x, 0.85)
    m = analyze_sawtooth_suite(x, y, (1700.0, 2000.0))
    assert m["masd"] == pytest.approx(0.0, abs=1e-12)
    assert m["extrema_density"] == pytest.approx(0.0, abs=1e-12)
    assert m["lag1_acf"] == pytest.approx(0.0, abs=1e-12)


def test_linear_slope_has_zero_masd_and_extrema() -> None:
    """A clean linear slope has zero second derivative and zero
    extrema — MASD and ED must both vanish."""
    x = _energy_grid()
    y = 0.5 + 1e-4 * (x - x[0])
    m = analyze_sawtooth_suite(x, y, (1700.0, 2000.0))
    assert m["masd"] < 1e-10
    assert m["extrema_density"] == pytest.approx(0.0, abs=1e-12)


# -- Deterministic sawtooth should hit ACF1 ≈ -1 --------------------------- #


def test_hard_sawtooth_drives_acf1_near_minus_one() -> None:
    """A strict ``up-down-up-down`` pattern in β̂ is the canonical
    sawtooth signature. Its first-difference vector has alternating
    signs of equal magnitude, so the lag-1 Pearson correlation of
    that vector is exactly −1 (in the noiseless limit)."""
    n = 101
    x = np.linspace(1700.0, 2000.0, n)
    y = 0.85 + 0.01 * (-1.0) ** np.arange(n)
    m = analyze_sawtooth_suite(x, y, (1700.0, 2000.0))
    # First differences alternate: +d, -d, +d, ... → ACF1 = -1.
    assert m["lag1_acf"] == pytest.approx(-1.0, abs=1e-9)
    # Every interior point is an extremum: ED ≈ (n-2) / (hi-lo).
    expected_ed = (n - 2) / (2000.0 - 1700.0)
    assert m["extrema_density"] == pytest.approx(expected_ed, rel=0.01)
    # MASD picks up the |second difference| = 4·amplitude → 0.04.
    assert m["masd"] == pytest.approx(0.04, abs=1e-9)


# -- Pure noise should land ACF1 around −0.5 ------------------------------- #


def test_gaussian_noise_acf1_around_minus_half() -> None:
    """White Gaussian noise has ACF1(Δy) ≈ −0.5 because successive
    differences ``ε_{i+1} − ε_i`` and ``ε_{i+2} − ε_{i+1}`` share the
    middle ``−ε_{i+1}`` term, producing the classic −1/2 textbook
    correlation. Loose tolerance — finite-sample noise is its own
    stochastic process here."""
    rng = np.random.default_rng(0)
    x = _energy_grid(n=2001)
    y = 0.85 + 0.01 * rng.standard_normal(x.size)
    m = analyze_sawtooth_suite(x, y, (1700.0, 2000.0))
    assert -0.55 < m["lag1_acf"] < -0.45, f"white-noise ACF1 = {m['lag1_acf']} (expected near -0.5)"


# -- Amplitude / frequency decoupling -------------------------------------- #


def test_masd_scales_with_amplitude_independent_of_frequency() -> None:
    """Double the sawtooth amplitude — MASD doubles, ED stays put."""
    x = _energy_grid(n=101)
    y_small = 0.85 + 0.005 * (-1.0) ** np.arange(x.size)
    y_large = 0.85 + 0.010 * (-1.0) ** np.arange(x.size)
    m_s = analyze_sawtooth_suite(x, y_small, (1700.0, 2000.0))
    m_l = analyze_sawtooth_suite(x, y_large, (1700.0, 2000.0))
    assert m_l["masd"] == pytest.approx(2 * m_s["masd"], rel=1e-9)
    assert m_l["extrema_density"] == pytest.approx(m_s["extrema_density"], rel=1e-9)


def test_extrema_density_scales_with_frequency_independent_of_amplitude() -> None:
    """A slower sawtooth (period 4 grid points instead of 2) has half
    the extrema density at the same amplitude — but the per-pair
    second-difference magnitude is unchanged. The mean of those
    second differences over a longer span (with zeros in between)
    drops, so MASD shrinks too, but the two axes are clearly
    decoupled in direction."""
    n = 201
    x = np.linspace(1700.0, 2000.0, n)
    # Fast sawtooth: period 2 samples (alternates every grid point).
    y_fast = 0.85 + 0.01 * (-1.0) ** np.arange(n)
    # Slow sawtooth: period 4 samples (alternates every other grid point).
    pattern = np.array([1, 1, -1, -1])
    y_slow = 0.85 + 0.01 * pattern[np.arange(n) % 4]
    m_fast = analyze_sawtooth_suite(x, y_fast, (1700.0, 2000.0))
    m_slow = analyze_sawtooth_suite(x, y_slow, (1700.0, 2000.0))
    # Faster sawtooth → strictly more extrema.
    assert m_fast["extrema_density"] > m_slow["extrema_density"]
    # Fast sawtooth ACF1 = -1; slow has a different (less-negative) ACF1.
    assert m_fast["lag1_acf"] == pytest.approx(-1.0, abs=1e-9)
    assert m_slow["lag1_acf"] > -0.99  # clearly not the pure −1 limit


# -- Region selection ------------------------------------------------------ #


def test_region_mask_isolates_window() -> None:
    """Metrics computed in two different windows of the same global
    curve must reflect only the masked-window behaviour."""
    x = np.linspace(500.0, 3000.0, 5001)
    # Quiet [500, 1700], noisy [1700, 2000], quiet [2000, ∞).
    y = np.full_like(x, 0.5)
    noisy_mask = (x >= 1700.0) & (x <= 2000.0)
    rng = np.random.default_rng(0)
    y[noisy_mask] += 0.05 * rng.standard_normal(int(noisy_mask.sum()))

    m_noisy = analyze_sawtooth_suite(x, y, (1700.0, 2000.0))
    m_quiet = analyze_sawtooth_suite(x, y, (500.0, 1500.0))
    assert m_noisy["masd"] > 10 * m_quiet["masd"]
    assert m_noisy["extrema_density"] > 10 * m_quiet["extrema_density"]


# -- Edge cases ------------------------------------------------------------ #


def test_too_few_samples_returns_zeros() -> None:
    """The contract: < 4 samples in the window → all three metrics
    are 0.0. Lets callers iterate over multiple windows without
    special-casing very narrow ones."""
    x = np.array([1900.0, 1910.0, 1920.0])
    y = np.array([0.5, 0.6, 0.4])
    m = analyze_sawtooth_suite(x, y, (1700.0, 2000.0))
    assert m == {"masd": 0.0, "extrema_density": 0.0, "lag1_acf": 0.0}


def test_empty_window_returns_zeros() -> None:
    """Region entirely outside the input grid → empty mask → zeros."""
    x = _energy_grid()
    y = np.full_like(x, 0.5)
    m = analyze_sawtooth_suite(x, y, (5000.0, 6000.0))
    assert m == {"masd": 0.0, "extrema_density": 0.0, "lag1_acf": 0.0}


def test_shape_mismatch_raises() -> None:
    x = _energy_grid()
    y = np.zeros(x.size + 1)
    with pytest.raises(ValueError, match="same shape"):
        analyze_sawtooth_suite(x, y, (1700.0, 2000.0))


def test_invalid_region_raises() -> None:
    x = _energy_grid()
    y = np.full_like(x, 0.5)
    with pytest.raises(ValueError, match="hi > lo"):
        analyze_sawtooth_suite(x, y, (2000.0, 1700.0))
