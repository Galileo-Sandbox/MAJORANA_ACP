"""Tests for the saved-prediction regional coverage export."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/export_paper_coverage.py"
SPEC = importlib.util.spec_from_file_location("paper_coverage_export", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def reference(counts, passes):
    counts = np.asarray(counts, dtype=np.int64)
    passes = np.asarray(passes, dtype=np.int64)
    lower, upper = MODULE.wilson_interval(passes, counts, z_value=1.0)
    return {
        "counts": counts,
        "fraction": np.divide(passes, counts, out=np.full(counts.size, np.nan), where=counts > 0),
        "wilson_half_width": 0.5 * (upper - lower),
    }


def test_boundary_bin_membership_is_center_based_and_fixed():
    masks = MODULE.region_masks()
    assert MODULE.BIN_CENTERS[masks["FE_2614_pm5"]].tolist() == [2612.5, 2617.5]
    assert MODULE.BIN_CENTERS[masks["SE_2103_pm5"]].tolist() == [2102.5, 2107.5]
    assert MODULE.BIN_CENTERS[masks["DEP_1592_pm5"]].tolist() == [1587.5, 1592.5]
    assert MODULE.BIN_CENTERS[masks["feature_1620_pm5"]].tolist() == [1617.5, 1622.5]


def test_support_exclusion_and_integer_hit_consistency():
    ref = reference([3, 4, 10], [1, 2, 5])
    result = MODULE.coverage_metrics(ref, np.array([0.0, 0.5, 0.5]), np.ones(3, dtype=bool))
    assert result["supported_bin_count"] == 2
    assert result["excluded_bin_count"] == 1
    assert result["excluded_target_count"] == 3
    assert result["hit_count_C1"] == result["hit_count_C2"] == result["hit_count_C3"] == 2
    assert result["C1_percent"] == pytest.approx(100.0)


def test_smooth_bias_and_alternating_residuals_have_fixed_coverage_rule():
    ref = reference([100] * 4, [50] * 4)
    half = ref["wilson_half_width"]
    smooth = ref["fraction"] + 1.5 * half
    alternating = ref["fraction"] + 1.5 * half * np.array([1, -1, 1, -1])
    smooth_result = MODULE.coverage_metrics(ref, smooth, np.ones(4, dtype=bool))
    alternating_result = MODULE.coverage_metrics(ref, alternating, np.ones(4, dtype=bool))
    assert smooth_result["C1_percent"] == alternating_result["C1_percent"] == 0.0
    assert smooth_result["C2_percent"] == alternating_result["C2_percent"] == 100.0
    assert np.mean(smooth - ref["fraction"]) != pytest.approx(
        np.mean(alternating - ref["fraction"])
    )


def test_missing_model_sd_is_irrelevant_to_reference_coverage():
    assert "model_sd" not in inspect.signature(MODULE.coverage_metrics).parameters
    ref = reference([10, 10], [0, 10])
    prediction = np.array([0.0, 1.0])
    result = MODULE.coverage_metrics(ref, prediction, np.ones(2, dtype=bool))
    assert result["C1_percent"] == 100.0


def test_empty_region_is_explicitly_undefined():
    ref = reference([10, 10], [5, 5])
    result = MODULE.coverage_metrics(ref, np.array([0.5, 0.5]), np.zeros(2, dtype=bool))
    assert result["supported_bin_count"] == 0
    assert result["C1_percent"] is None


def test_portable_outputs_are_internally_consistent():
    paper_root = Path(__file__).parents[1]
    manifest = json.loads((paper_root / "manifests/paper_coverage_export.json").read_text())
    repo = paper_root.parent
    for relative_path, metadata in manifest["outputs"].items():
        payload = (repo / relative_path).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == metadata["sha256"]

    with (paper_root / "tables/paper_coverage_cells.csv").open(newline="") as stream:
        cells = list(csv.DictReader(stream))
    assert len(cells) == 6300
    by_run = {}
    for row in cells:
        by_run.setdefault(row["run_id"], {})[row["region"]] = row
        if row["region"] not in MODULE.COMPOSITES:
            supported = int(row["supported_bin_count"])
            for k in MODULE.KS:
                expected = 100.0 * int(row[f"hit_count_C{k}"]) / supported
                assert float(row[f"C{k}_percent"]) == pytest.approx(expected)
    assert len(by_run) == 630
    for regions in by_run.values():
        for composite, components in MODULE.COMPOSITES.items():
            for k in MODULE.KS:
                expected = np.mean([float(regions[name][f"C{k}_percent"]) for name in components])
                assert float(regions[composite][f"C{k}_percent"]) == pytest.approx(expected)

    with (paper_root / "tables/paper_coverage_cross_budget_pairs.csv").open(newline="") as stream:
        pairs = list(csv.DictReader(stream))
    assert sum(row["row_type"] == "paired_cell" for row in pairs) == 2160
    assert sum(row["row_type"] == "paired_summary" for row in pairs) == 72
