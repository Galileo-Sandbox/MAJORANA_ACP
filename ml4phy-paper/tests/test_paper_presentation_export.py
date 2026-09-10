"""Boundary tests for the existing-prediction presentation export."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/export_paper_presentation.py"
SPEC = importlib.util.spec_from_file_location("paper_presentation_export", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_wilson_bounds_empty_and_extreme_bins():
    lower, upper = MODULE.wilson_interval(
        np.array([0, 0, 10]), np.array([0, 10, 10]), z_value=1.0
    )
    assert np.isnan(lower[0]) and np.isnan(upper[0])
    assert lower[1] == pytest.approx(0.0)
    assert 0.0 < upper[1] < 1.0
    assert 0.0 < lower[2] < 1.0
    assert upper[2] == pytest.approx(1.0)


def test_bin_events_flags_empty_and_unsupported_bins():
    result = MODULE.bin_events(
        np.array([500.1, 500.2, 505.1, 505.2, 505.3, 505.4]),
        np.array([0, 1, 0, 1, 1, 1]),
        np.array([0.1, 0.8, 0.2, 0.7, 0.9, 0.6]),
    )
    assert result["counts"][0] == 2
    assert result["counts"][1] == 4
    assert result["counts"][2] == 0
    assert result["fraction"][0] == pytest.approx(0.5)
    assert np.isnan(result["fraction"][2])
    assert result["prediction_mean"][1] == pytest.approx(0.6)


def test_missing_uncertainty_stays_unavailable():
    rows = MODULE.regional_rows(
        np.array([500.0, 1000.0, 1592.0, 1620.0, 2103.0, 2614.0]),
        np.array([0, 1, 1, 0, 1, 0]),
        np.full(6, 0.5),
        None,
    )
    assert len(rows) == 5
    assert all(row["combined_proxy_G1"] is None for row in rows)
    assert all("unavailable" in row["combined_proxy_status"] for row in rows)


def test_exact_paired_support_and_equal_window_weighting():
    counts = np.full(MODULE.BIN_CENTERS.size, 10, dtype=np.int64)
    fraction = np.full(MODULE.BIN_CENTERS.size, 0.5)
    lower, upper = MODULE.wilson_interval(counts // 2, counts)
    reference = {
        "counts": counts,
        "fraction": fraction,
        "wilson_half_width": 0.5 * (upper - lower),
    }
    prediction = fraction.copy()
    first = MODULE.region_mask(MODULE.BIN_CENTERS, 1700.0, 2000.0)
    second = MODULE.region_mask(MODULE.BIN_CENTERS, 2200.0, 2400.0)
    prediction[first] += 0.01
    prediction[second] += 0.03
    rows = {row["window"]: row for row in MODULE.continuum_diagnostics(reference, prediction)}
    assert rows["continuum_1700_2000"]["supported_bin_count"] == 60
    assert rows["continuum_2200_2400"]["supported_bin_count"] == 40
    assert rows["equal_window_mean"]["continuum_bias_pp"] == pytest.approx(2.0)
    assert rows["equal_window_mean"]["continuum_RMSE_pp"] == pytest.approx(
        np.sqrt((1.0**2 + 3.0**2) / 2.0)
    )


def test_gaussian_transform_matches_zero_residual_reference_values():
    assert MODULE.gaussian_agreement(0.0, 1) == pytest.approx(0.682689492137, rel=1e-10)
    assert MODULE.gaussian_agreement(0.0, 2) == pytest.approx(0.954499736104, rel=1e-10)
    assert MODULE.gaussian_agreement(0.0, 3) == pytest.approx(0.997300203937, rel=1e-10)
