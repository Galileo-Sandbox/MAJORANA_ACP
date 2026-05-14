"""Tests for the Wilson score interval helper in cut_acceptance.pipeline."""

from __future__ import annotations

import numpy as np

from majorana_acp.cut_acceptance.pipeline import wilson_interval


def test_wilson_handles_k_zero() -> None:
    """k=0 → lo=0, hi=z²/(n+z²); does NOT collapse to (0, 0) like the naive σ."""
    lo, hi = wilson_interval(np.array([0]), np.array([10]), z=1.0)
    assert lo[0] == 0.0
    assert hi[0] > 0.0
    assert hi[0] < 0.5  # 1 / (10+1) ≈ 0.091


def test_wilson_handles_k_equal_n() -> None:
    """k=n → hi≈1, lo=n/(n+z²); symmetric to the k=0 case."""
    lo, hi = wilson_interval(np.array([10]), np.array([10]), z=1.0)
    assert hi[0] >= 1.0 - 1e-9  # may not be exactly 1.0 due to FP
    assert lo[0] < 1.0
    assert lo[0] > 0.5


def test_wilson_recovers_naive_at_midrange_large_n() -> None:
    """For p≈0.5 and large n, Wilson reduces to the naive √(p(1-p)/n)."""
    k = np.array([500])
    n = np.array([1000])
    lo, hi = wilson_interval(k, n, z=1.0)
    p = 0.5
    naive_half = np.sqrt(p * (1 - p) / 1000)
    wilson_half = 0.5 * (hi[0] - lo[0])
    # Should agree to within ~5% for large n.
    assert abs(wilson_half - naive_half) / naive_half < 0.05


def test_wilson_returns_nan_for_zero_n() -> None:
    lo, hi = wilson_interval(np.array([0]), np.array([0]), z=1.0)
    assert np.isnan(lo[0])
    assert np.isnan(hi[0])


def test_wilson_is_asymmetric_near_extremes() -> None:
    """At p near 0, hi-rate ≫ rate-lo (the interval is one-sided)."""
    lo, hi = wilson_interval(np.array([1]), np.array([20]), z=1.0)
    rate = 1.0 / 20
    upper_half = hi[0] - rate
    lower_half = rate - lo[0]
    # The naive σ would be symmetric ≈ 0.049; Wilson is much more lopsided.
    assert upper_half > 1.5 * lower_half


def test_wilson_vectorises_over_arrays() -> None:
    k = np.array([0, 5, 10, 1])
    n = np.array([10, 10, 10, 20])
    lo, hi = wilson_interval(k, n, z=1.0)
    assert lo.shape == k.shape
    assert hi.shape == k.shape
    # Monotone: more k → both bounds rise.
    assert lo[0] < lo[1] < lo[2]
    assert hi[0] < hi[1] < hi[2]
