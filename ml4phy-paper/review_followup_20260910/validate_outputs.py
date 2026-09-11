#!/usr/bin/env python3
"""Independently validate portable mechanism summaries from exported bin rows."""

from __future__ import annotations

import csv
import gzip
import json
import lzma
from collections import defaultdict
from pathlib import Path

import numpy as np
from export_existing import REGIONS, mask
from export_paper_coverage import BIN_CENTERS

HERE = Path(__file__).resolve().parent


def main() -> None:
    reference_rows = list(csv.DictReader((HERE / "tables/shared_reference_bins.csv").open()))
    counts = np.array([int(row["event_count"]) for row in reference_rows])
    fraction = np.array([float(row["empirical_efficiency"]) for row in reference_rows])
    scale = np.array([float(row["wilson_half_width_z1"]) for row in reference_rows])
    support = counts >= 4
    if (counts.size, int(support.sum()), int((~support).sum()), int(counts[~support].sum())) != (
        500,
        442,
        58,
        101,
    ):
        raise ValueError("Frozen reference-bin inventory changed")

    predictions = np.full((150, 500), np.nan)
    with lzma.open(HERE / "tables/mechanism_cell_bins.csv.xz", "rt") as stream:
        for row in csv.DictReader(stream):
            predictions[int(row["cell_index"]), int(row["bin_index"])] = float(
                row["mean_predicted_probability"]
            )
    if np.isnan(predictions[:, counts > 0]).any():
        raise ValueError("Missing prediction in a nonempty reference bin")

    with gzip.open(HERE / "tables/mechanism_regional_cells.csv.gz", "rt") as stream:
        regional = list(csv.DictReader(stream))
    reported = {(int(row["cell_index"]), row["region_id"]): row for row in regional}
    maximum = 0.0
    monotonic_rows = 0
    for cell in range(150):
        component_scores = defaultdict(dict)
        for region_id, definition in REGIONS.items():
            selected = mask(BIN_CENTERS, definition) & support
            pull = np.abs((predictions[cell, selected] - fraction[selected]) / scale[selected])
            values = [100.0 * float(np.mean(pull <= k)) for k in (1, 2, 3)]
            row = reported[(cell, region_id)]
            for k, value in enumerate(values, 1):
                maximum = max(maximum, abs(value - float(row[f"C{k}_percent"])))
                component_scores[region_id][k] = value
            if not values[0] <= values[1] <= values[2]:
                raise ValueError(f"Nonmonotonic Ck for cell {cell}, {region_id}")
            monotonic_rows += 1
        composites = {
            "peaks_equal_four_cores": ("FE_core", "SE_core", "DEP_core", "feature_1620_core"),
            "continuum_equal_two_windows": (
                "continuum_1700_2000",
                "continuum_2200_2400",
            ),
            "exploratory_continuum_tail_equal_three": (
                "continuum_1700_2000",
                "continuum_2200_2400",
                "sparse_tail_2700_3000",
            ),
        }
        for composite, components in composites.items():
            row = reported[(cell, composite)]
            for k in (1, 2, 3):
                value = float(np.mean([component_scores[name][k] for name in components]))
                maximum = max(maximum, abs(value - float(row[f"C{k}_percent"])))
    if maximum > 1e-12:
        raise ValueError(f"Portable-bin Ck mismatch: {maximum}")

    registry = list(csv.DictReader((HERE / "tables/mechanism_run_inventory.csv").open()))
    if len(registry) != 150 or len({row["run_id"] for row in registry}) != 150:
        raise ValueError("Run registry is incomplete or duplicated")
    expected = {
        (mode, seed, context)
        for mode in (
            "full_density",
            "global_gate",
            "global_attention",
            "global_both",
            "density_free_global",
        )
        for seed in range(3)
        for context in range(100, 110)
    }
    actual = {
        (row["mode"], int(row["training_seed"]), int(row["context_seed"])) for row in registry
    }
    if actual != expected:
        raise ValueError("Seed/context pairing is incomplete")

    existing = json.loads((HERE / "reports/existing_export_record.json").read_text())
    if (
        existing["recovered_cells"] != 630
        or existing["global_Ck_max_abs_difference_percentage_points"] > 1e-12
    ):
        raise ValueError("Prior 630-cell/global-Ck validation is missing")
    record = {
        "status": "passed",
        "independent_input": "compressed portable per-cell bin predictions",
        "mechanism_cells": 150,
        "regions_per_cell_including_composites": len(REGIONS) + 3,
        "validated_noncomposite_Ck_rows": monotonic_rows,
        "maximum_Ck_absolute_difference_percentage_points": maximum,
        "C1_le_C2_le_C3": True,
        "paired_seed_context_inventory": True,
        "reference_bins": 500,
        "supported_bins": 442,
        "excluded_bins": 58,
        "excluded_target_events": 101,
        "prior_global_Ck_cells_reproduced": 630,
        "prior_global_Ck_max_absolute_difference_percentage_points": existing[
            "global_Ck_max_abs_difference_percentage_points"
        ],
    }
    (HERE / "reports/validation_record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
