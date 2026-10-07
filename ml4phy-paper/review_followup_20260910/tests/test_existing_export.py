"""Tests for frozen regional definitions and measurement interpretation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "export_existing.py"
SPEC = importlib.util.spec_from_file_location("review_existing_export", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_frozen_core_endpoint_conventions():
    energy = np.array([1586.999, 1587.0, 1597.0, 1597.001])
    selected = MODULE.mask(energy, MODULE.REGIONS["DEP_core"])
    assert selected.tolist() == [False, True, True, False]
    broad = MODULE.mask(np.array([1577.0, 1605.999, 1606.0]), MODULE.REGIONS["DEP_broad"])
    assert broad.tolist() == [True, True, False]


def test_count_and_efficiency_interpretation_identity():
    observed = np.array([0, 1, 1, 0])
    prediction = np.array([0.2, 0.7, 0.8, 0.4])
    difference = prediction.sum() - observed.sum()
    assert difference == pytest.approx(0.1)
    assert 100.0 * difference / observed.size == pytest.approx(2.5)
    assert 100.0 * difference / observed.sum() == pytest.approx(5.0)


def test_empty_relative_count_denominator_is_undefined():
    observed_count = 0
    relative = None if observed_count == 0 else 1.0 / observed_count
    assert relative is None


def test_physical_labels_are_explicit_and_machine_ids_stable():
    assert MODULE.REGIONS["DEP_core"]["physical_label"] == "Thallium-208 double escape"
    assert MODULE.REGIONS["feature_1620_core"]["physical_label"] == "Bismuth-212 full energy"
    assert MODULE.REGIONS["SE_core"]["physical_label"] == "Thallium-208 single escape"
    assert MODULE.REGIONS["FE_core"]["physical_label"] == "Thallium-208 full energy"
