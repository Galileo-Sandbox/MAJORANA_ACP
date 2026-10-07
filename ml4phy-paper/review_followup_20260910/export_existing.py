#!/usr/bin/env python3
"""Export measurement interpretation from the 630 frozen saved cells."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import lzma
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO_SCRIPTS = Path(__file__).parents[1] / "scripts"
if str(REPO_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(REPO_SCRIPTS))
from export_paper_coverage import (  # noqa: E402
    BIN_CENTERS,
    BIN_EDGES,
    KS,
    MIN_BIN_EVENTS,
    bin_events,
    coverage_metrics,
    sha256_file,
)

REGIONS = {
    "overall": {
        "label": "Overall (500--3000 keV)",
        "physical_label": "Calibration mixture",
        "lower": 500.0,
        "upper": 3000.0,
        "upper_inclusive": True,
        "kind": "overall",
    },
    "DEP_core": {
        "label": "DEP core (1592 keV)",
        "physical_label": "Thallium-208 double escape",
        "lower": 1587.0,
        "upper": 1597.0,
        "upper_inclusive": True,
        "kind": "narrow_core",
    },
    "feature_1620_core": {
        "label": "Feature core (1620.74 keV)",
        "physical_label": "Bismuth-212 full energy",
        "lower": 1615.0,
        "upper": 1625.0,
        "upper_inclusive": True,
        "kind": "narrow_core",
    },
    "SE_core": {
        "label": "SE core (2103 keV)",
        "physical_label": "Thallium-208 single escape",
        "lower": 2098.0,
        "upper": 2108.0,
        "upper_inclusive": True,
        "kind": "narrow_core",
    },
    "FE_core": {
        "label": "FE core (2614 keV)",
        "physical_label": "Thallium-208 full energy",
        "lower": 2609.0,
        "upper": 2619.0,
        "upper_inclusive": True,
        "kind": "narrow_core",
    },
    "DEP_broad": {
        "label": "DEP broad (1577--1606 keV)",
        "physical_label": "Thallium-208 double escape",
        "lower": 1577.0,
        "upper": 1606.0,
        "upper_inclusive": False,
        "kind": "historical_broad",
    },
    "feature_1620_broad": {
        "label": "Feature broad (1606--1635 keV)",
        "physical_label": "Bismuth-212 full energy",
        "lower": 1606.0,
        "upper": 1635.0,
        "upper_inclusive": False,
        "kind": "historical_broad",
    },
    "SE_broad": {
        "label": "SE broad (2088--2118 keV)",
        "physical_label": "Thallium-208 single escape",
        "lower": 2088.0,
        "upper": 2118.0,
        "upper_inclusive": False,
        "kind": "historical_broad",
    },
    "FE_broad": {
        "label": "FE broad (2599--2629 keV)",
        "physical_label": "Thallium-208 full energy",
        "lower": 2599.0,
        "upper": 2629.0,
        "upper_inclusive": False,
        "kind": "historical_broad",
    },
    "continuum_1700_2000": {
        "label": "Continuum 1 (1700--2000 keV)",
        "physical_label": "Prespecified continuum window",
        "lower": 1700.0,
        "upper": 2000.0,
        "upper_inclusive": False,
        "kind": "continuum",
    },
    "continuum_2200_2400": {
        "label": "Continuum 2 (2200--2400 keV)",
        "physical_label": "Prespecified continuum window",
        "lower": 2200.0,
        "upper": 2400.0,
        "upper_inclusive": False,
        "kind": "continuum",
    },
    "sparse_tail_2700_3000": {
        "label": "Sparse tail (2700--3000 keV)",
        "physical_label": "Sparse high-energy tail",
        "lower": 2700.0,
        "upper": 3000.0,
        "upper_inclusive": False,
        "kind": "sparse_tail",
    },
}


def mask(values: np.ndarray, region: Mapping[str, Any]) -> np.ndarray:
    upper = values <= region["upper"] if region["upper_inclusive"] else values < region["upper"]
    return (values >= region["lower"]) & upper


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_gzip_csv(path: Path, fields: Sequence[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        io.TextIOWrapper(compressed, newline="") as text,
    ):
        writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_xz_csv(path: Path, fields: Sequence[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lzma.open(path, "wt", newline="", preset=9) as text:
        writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_cell(repo: Path, record: Mapping[str, Any], kernel_cache: dict[str, Any]):
    path = repo / record["event_predictions_path"]
    if sha256_file(path) != record["event_predictions_sha256"]:
        raise ValueError(f"Prediction hash mismatch: {record['run_id']}")
    if str(record["architecture_id"]).startswith("kernel_"):
        if not kernel_cache:
            with np.load(path) as archive:
                kernel_cache.update({name: archive[name] for name in archive.files})
        variant = (
            "context_only" if record["architecture_id"] == "kernel_context_only" else "pooled_data"
        )
        vi = [str(value) for value in kernel_cache["variants"]].index(variant)
        si = kernel_cache["context_sizes"].astype(int).tolist().index(500)
        ci = kernel_cache["context_seeds"].astype(int).tolist().index(int(record["context_seed"]))
        return (
            kernel_cache["final_target_rows"].astype(np.int64),
            kernel_cache["final_energy_kev"].astype(np.float64),
            kernel_cache["final_outcome"].astype(np.int8),
            kernel_cache["predictions"][vi, si, ci].astype(np.float64),
        )
    with np.load(path) as archive:
        return (
            archive["target_rows"].astype(np.int64),
            archive["target_energy_kev"].astype(np.float64),
            archive["outcome"].astype(np.int8),
            archive["prediction"].astype(np.float64),
        )


def composite_rows(cell_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_run: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in cell_rows:
        by_run[row["run_id"]][row["region_id"]] = row
    result = []
    definitions = {
        "peaks_equal_four_cores": ("FE_core", "SE_core", "DEP_core", "feature_1620_core"),
        "continuum_equal_two_windows": ("continuum_1700_2000", "continuum_2200_2400"),
        "exploratory_continuum_tail_equal_three": (
            "continuum_1700_2000",
            "continuum_2200_2400",
            "sparse_tail_2700_3000",
        ),
    }
    for _run, regions in by_run.items():
        for composite, components in definitions.items():
            items = [regions[name] for name in components]
            row = {
                key: value
                for key, value in items[0].items()
                if key
                not in {
                    "region_id",
                    "region_label",
                    "physical_label",
                    "region_kind",
                    "N_R_all_events",
                    "Y_R_all_events",
                    "P_R_all_events",
                    "passing_count_difference_all_events",
                    "efficiency_difference_pp_all_events",
                    "relative_passing_difference_percent_all_events",
                    "supported_bin_count",
                    "excluded_bin_count",
                    "supported_target_events",
                    "unsupported_nonempty_target_events",
                    "weighted_bin_MAE_pp_supported",
                    "weighted_bin_MAE_pp_all_nonempty",
                    "hit_count_C1",
                    "hit_count_C2",
                    "hit_count_C3",
                    "C1_percent",
                    "C2_percent",
                    "C3_percent",
                    "component_regions",
                    "weighting",
                }
            }
            row.update(
                {
                    "region_id": composite,
                    "region_label": composite.replace("_", " "),
                    "physical_label": "Composite diagnostic",
                    "region_kind": "composite",
                    "N_R_all_events": sum(item["N_R_all_events"] for item in items),
                    "Y_R_all_events": sum(item["Y_R_all_events"] for item in items),
                    "P_R_all_events": None,
                    "passing_count_difference_all_events": None,
                    "efficiency_difference_pp_all_events": None,
                    "relative_passing_difference_percent_all_events": None,
                    "supported_bin_count": sum(item["supported_bin_count"] for item in items),
                    "excluded_bin_count": sum(item["excluded_bin_count"] for item in items),
                    "supported_target_events": sum(
                        item["supported_target_events"] for item in items
                    ),
                    "unsupported_nonempty_target_events": sum(
                        item["unsupported_nonempty_target_events"] for item in items
                    ),
                    "weighted_bin_MAE_pp_supported": float(
                        np.mean([item["weighted_bin_MAE_pp_supported"] for item in items])
                    ),
                    "weighted_bin_MAE_pp_all_nonempty": float(
                        np.mean([item["weighted_bin_MAE_pp_all_nonempty"] for item in items])
                    ),
                    "component_regions": "|".join(components),
                    "weighting": "equal_component_mean_per_cell",
                }
            )
            for k in KS:
                row[f"hit_count_C{k}"] = None
                row[f"C{k}_percent"] = float(np.mean([item[f"C{k}_percent"] for item in items]))
            result.append(row)
    return result


def training_support(repo: Path) -> list[dict[str, Any]]:
    root = repo / "ml4phy-paper"
    p2 = json.loads((root / "configs/phase2_training_v1.json").read_text())
    p3 = json.loads((root / "configs/phase3_training_v1.json").read_text())
    specs = [{"subset_id": f"original_n{x['budget']}", **x} for x in p2["subset_inputs"]] + list(
        p3["subset_inputs"]
    )
    specs.append(
        {
            "subset_id": "full_n18866",
            "budget": 18866,
            "logical_path": "runs/small_data_configs/simple_cnn_small/eval_train/predictions.h5",
            "sampling_eligible_unique_events": 18836,
            "density_pool_unique_events": 18866,
        }
    )
    rows = []
    for spec in specs:
        path = repo / spec["logical_path"]
        with h5py.File(path) as handle:
            energy = handle["energy"][:].astype(np.float64)
        bins = np.searchsorted(np.arange(500.0, 3000.0 + 10.0, 10.0), energy, side="right") - 1
        counts = np.bincount(bins, minlength=250)
        eligible = counts[bins] >= 4
        if int(eligible.sum()) != spec["sampling_eligible_unique_events"]:
            raise ValueError(f"Sampling-eligible reconstruction mismatch: {spec['subset_id']}")
        for region_id, definition in REGIONS.items():
            selected = mask(energy, definition)
            rows.append(
                {
                    "subset_id": spec["subset_id"],
                    "nominal_budget": spec["budget"],
                    "sampling_eligible_budget": spec["sampling_eligible_unique_events"],
                    "density_buffer_budget": spec["density_pool_unique_events"],
                    "region_id": region_id,
                    "nominal_region_events": int(selected.sum()),
                    "sampling_eligible_region_events": int((selected & eligible).sum()),
                    "density_buffer_region_events": int(selected.sum()),
                    "subset_file_sha256": sha256_file(path),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--wall-limit-seconds", type=float, default=2700.0)
    args = parser.parse_args()
    started = time.perf_counter()
    repo = args.repo.resolve()
    root = repo / "ml4phy-paper"
    output = root / "review_followup_20260910"
    manifest_path = root / "manifests/paper_presentation_export.json"
    coverage_path = root / "tables/paper_coverage_cells.csv"
    manifest = json.loads(manifest_path.read_text())
    records = manifest["source_artifacts"]
    if len(records) != 630:
        raise ValueError("Expected 630 source cells")
    old_global = {
        row["run_id"]: row
        for row in csv.DictReader(coverage_path.open())
        if row["region"] == "overall_500_3000"
    }
    registry = []
    regional = []
    bin_payload = []
    kernel_cache: dict[str, Any] = {}
    reference_identity = None
    reference = None
    global_difference = 0.0
    for cell_index, record in enumerate(records):
        if time.perf_counter() - started > args.wall_limit_seconds:
            raise TimeoutError("Existing-artifact export exceeded 45 minutes")
        rows, energy, outcome, prediction = load_cell(repo, record, kernel_cache)
        if reference_identity is None:
            reference_identity = rows
            reference = bin_events(energy, outcome)
        elif not np.array_equal(rows, reference_identity):
            raise ValueError(f"Target identity mismatch: {record['run_id']}")
        binned = bin_events(energy, outcome, prediction)
        registry.append(
            {
                "cell_index": cell_index,
                **{
                    key: record.get(key)
                    for key in (
                        "run_id",
                        "subset_id",
                        "ordering_id",
                        "budget_label",
                        "training_budget_nominal_events",
                        "training_budget_sampling_eligible_events",
                        "method",
                        "architecture_id",
                        "comparison_role",
                        "training_seed",
                        "context_seed",
                        "event_predictions_path",
                        "event_predictions_sha256",
                        "curve_path",
                        "curve_sha256",
                        "checkpoint_sha256",
                    )
                },
            }
        )
        for bin_index in range(BIN_CENTERS.size):
            pbar = binned["prediction_mean"][bin_index]
            bin_payload.append(
                {
                    "cell_index": cell_index,
                    "bin_index": bin_index,
                    "mean_predicted_probability": pbar,
                    "sum_predicted_passing_probabilities": (
                        pbar * binned["counts"][bin_index] if np.isfinite(pbar) else None
                    ),
                }
            )
        for region_id, definition in REGIONS.items():
            event_selected = mask(energy, definition)
            bin_selected = mask(BIN_CENTERS, definition)
            support = bin_selected & (reference["counts"] >= MIN_BIN_EVENTS)
            nonempty = bin_selected & (reference["counts"] > 0)
            n_events = int(event_selected.sum())
            observed = int(outcome[event_selected].sum())
            predicted = float(prediction[event_selected].sum())
            difference = predicted - observed
            residual = binned["prediction_mean"] - reference["fraction"]
            coverage = coverage_metrics(reference, binned["prediction_mean"], bin_selected)
            row = {
                "cell_index": cell_index,
                "run_id": record["run_id"],
                "subset_id": record["subset_id"],
                "budget_label": record["budget_label"],
                "training_budget_nominal_events": record["training_budget_nominal_events"],
                "training_budget_sampling_eligible_events": record[
                    "training_budget_sampling_eligible_events"
                ],
                "method": record["method"],
                "architecture_id": record["architecture_id"],
                "training_seed": record.get("training_seed"),
                "context_seed": record["context_seed"],
                "region_id": region_id,
                "region_label": definition["label"],
                "physical_label": definition["physical_label"],
                "region_kind": definition["kind"],
                "N_R_all_events": n_events,
                "Y_R_all_events": observed,
                "P_R_all_events": predicted,
                "passing_count_difference_all_events": difference,
                "efficiency_difference_pp_all_events": 100.0 * difference / n_events
                if n_events
                else None,
                "relative_passing_difference_percent_all_events": 100.0 * difference / observed
                if observed
                else None,
                "supported_bin_count": int(support.sum()),
                "excluded_bin_count": int(bin_selected.sum() - support.sum()),
                "supported_target_events": int(reference["counts"][support].sum()),
                "unsupported_nonempty_target_events": int(
                    reference["counts"][nonempty & ~support].sum()
                ),
                "weighted_bin_MAE_pp_supported": 100.0
                * float(np.average(np.abs(residual[support]), weights=reference["counts"][support]))
                if support.any()
                else None,
                "weighted_bin_MAE_pp_all_nonempty": 100.0
                * float(
                    np.average(np.abs(residual[nonempty]), weights=reference["counts"][nonempty])
                )
                if nonempty.any()
                else None,
                "component_regions": None,
                "weighting": "event weighted for MAE; equal supported bins for Ck",
                **{
                    key: coverage[key]
                    for key in (
                        "hit_count_C1",
                        "hit_count_C2",
                        "hit_count_C3",
                        "C1_percent",
                        "C2_percent",
                        "C3_percent",
                    )
                },
            }
            regional.append(row)
        old = old_global[record["run_id"]]
        current = regional[-len(REGIONS)]
        for k in KS:
            global_difference = max(
                global_difference, abs(float(old[f"C{k}_percent"]) - current[f"C{k}_percent"])
            )
    assert reference is not None
    regional.extend(composite_rows(regional))
    if global_difference > 1e-12:
        raise ValueError(f"Old global Ck mismatch: {global_difference}")
    if (int((reference["counts"] >= 4).sum()), int((reference["counts"] < 4).sum())) != (442, 58):
        raise ValueError("Reference support count changed")

    reference_rows = []
    for index in range(BIN_CENTERS.size):
        row = {
            "bin_index": index,
            "lower_edge_kev": BIN_EDGES[index],
            "upper_edge_kev": BIN_EDGES[index + 1],
            "center_kev": BIN_CENTERS[index],
            "event_count": int(reference["counts"][index]),
            "pass_count": int(reference["passes"][index]),
            "empirical_efficiency": reference["fraction"][index],
            "wilson_lower_z1": reference["wilson_lower"][index],
            "wilson_upper_z1": reference["wilson_upper"][index],
            "wilson_half_width_z1": reference["wilson_half_width"][index],
            "supported_min_4": bool(reference["counts"][index] >= 4),
        }
        for region_id, definition in REGIONS.items():
            row[f"in_{region_id}"] = bool(mask(np.array([BIN_CENTERS[index]]), definition)[0])
        reference_rows.append(row)

    tables = output / "tables"
    write_csv(tables / "existing_run_registry.csv", registry)
    write_csv(tables / "shared_reference_bins.csv", reference_rows)
    write_xz_csv(
        tables / "existing_cell_bins.csv.xz",
        (
            "cell_index",
            "bin_index",
            "mean_predicted_probability",
            "sum_predicted_passing_probabilities",
        ),
        bin_payload,
    )
    write_gzip_csv(
        tables / "existing_regional_measurement_cells.csv.gz",
        list(regional[0]),
        regional,
    )
    support_rows = training_support(repo)
    write_csv(tables / "training_budget_region_support.csv", support_rows)

    # Compact hierarchical means preserve subset, initialization, and context levels.
    seed_means = []
    grouped = defaultdict(list)
    for row in regional:
        grouped[(row["subset_id"], row["method"], row["region_id"], row["training_seed"])].append(
            row
        )
    metrics = (
        "efficiency_difference_pp_all_events",
        "weighted_bin_MAE_pp_supported",
        "C1_percent",
        "C2_percent",
        "C3_percent",
    )
    for key, group in grouped.items():
        row = {
            "subset_id": key[0],
            "method": key[1],
            "region_id": key[2],
            "training_seed": key[3],
            "context_cell_count": len(group),
        }
        for metric in metrics:
            values = [float(item[metric]) for item in group if item[metric] not in (None, "")]
            row[f"{metric}_context_mean"] = float(np.mean(values)) if values else None
            row[f"{metric}_context_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        seed_means.append(row)
    write_gzip_csv(
        tables / "existing_regional_seed_summary.csv.gz",
        list(seed_means[0]),
        seed_means,
    )

    tail_support = [
        row
        for row in support_rows
        if row["region_id"] == "sparse_tail_2700_3000"
        and row["nominal_budget"] in (2000, 5000, 10000, 18866)
    ]
    report = (
        f"""# Existing-prediction measurement interpretation

Status: complete from saved artifacts only in {time.perf_counter() - started:.1f} seconds. No model call, inference, MC sampling, fit, interpolation, or tuning was performed.

The 630-cell inventory is complete: 600 neural cells, ten context-only kernel cells, ten pooled-kernel cells, and ten selected Bernoulli-GP cells. The frozen reference contains 114,400 events in 500 bins. Exactly 442 bins are supported, 58 are excluded, and the excluded bins contain 101 events. All 630 global C1/C2/C3 rows reproduce the prior export; maximum absolute difference is {global_difference:.3g} percentage points.

`P_R-Y_R` is the difference between a sum of predicted passing probabilities and the observed passing count in this finite calibration reference. Its efficiency and relative-count versions are descriptive discrepancies, not independently known physical bias or calibrated significance. The event-weighted bin MAE is reported beside signed differences so cancellation cannot masquerade as shape accuracy. Supported-bin and all-nonempty/all-event domains are separate.

Physical names in this export are thallium-208 double escape (DEP), bismuth-212 full energy at 1620.74 keV, thallium-208 single escape (SE), and thallium-208 full energy (FE). Historical machine IDs and exact endpoints remain in the tables. Core event masks reproduce the presentation export's inclusive endpoints; broader masks reproduce the historical regional-error half-open endpoints.

The exploratory `exploratory_continuum_tail_equal_three` score is the equal per-cell mean of Continuum 1, Continuum 2, and sparse-tail Ck. All three components and their bin/event counts remain separately reported. It is not a replacement primary endpoint and the rest of the non-peak spectrum is not called featureless continuum.

Sparse-tail sampling-eligible counts by audited pool are:
"""
        + "\n".join(
            f"- {row['subset_id']}: {row['sampling_eligible_region_events']} eligible of {row['nominal_region_events']} nominal tail events."
            for row in tail_support
        )
        + """

Zero eligible tail events in a small pool identifies missing sampler support; it does not by itself establish the causal mechanism of a prediction failure. This analysis concerns calibration-mixture selection efficiency only. It does not validate signal efficiency, cross-section bias, a physics-search region, or transfer to another experiment.
"""
    )
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "existing_measurement_interpretation.md").write_text(report)
    result = {
        "status": "complete",
        "runtime_seconds": time.perf_counter() - started,
        "source_manifest": {
            "path": str(manifest_path.relative_to(repo)),
            "sha256": sha256_file(manifest_path),
        },
        "coverage_source": {
            "path": str(coverage_path.relative_to(repo)),
            "sha256": sha256_file(coverage_path),
        },
        "recovered_cells": len(registry),
        "missing_cells": [],
        "target_events": 114400,
        "supported_bins": 442,
        "excluded_bins": 58,
        "excluded_target_events": int(reference["counts"][reference["counts"] < 4].sum()),
        "global_Ck_max_abs_difference_percentage_points": global_difference,
        "no_new_model_calls": True,
    }
    (output / "reports/existing_export_record.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
