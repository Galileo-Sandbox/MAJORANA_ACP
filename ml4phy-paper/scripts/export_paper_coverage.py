#!/usr/bin/env python3
"""Export regional reference-band coverage from existing saved predictions.

This is an offline, read-only analysis of the frozen prediction artifacts listed
by the completed presentation-export manifest. It never imports model code and
never performs prediction, fitting, tuning, or data selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

MIN_BIN_EVENTS = 4
BIN_EDGES = np.arange(500.0, 3000.0 + 5.0, 5.0)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])

REGIONS = {
    "overall_500_3000": (500.0, 3000.0, "half_open"),
    "figure1_1500_3000": (1500.0, 3000.0, "half_open"),
    "FE_2614_pm5": (2609.0, 2619.0, "center_inclusive"),
    "SE_2103_pm5": (2098.0, 2108.0, "center_inclusive"),
    "DEP_1592_pm5": (1587.0, 1597.0, "center_inclusive"),
    "feature_1620_pm5": (1615.0, 1625.0, "center_inclusive"),
    "continuum_1700_2000": (1700.0, 2000.0, "half_open"),
    "continuum_2200_2400": (2200.0, 2400.0, "half_open"),
}
COMPOSITES = {
    "peak_equal_feature": (
        "FE_2614_pm5",
        "SE_2103_pm5",
        "DEP_1592_pm5",
        "feature_1620_pm5",
    ),
    "continuum_equal_window": (
        "continuum_1700_2000",
        "continuum_2200_2400",
    ),
}
KS = (1, 2, 3)
NEURAL_SUBSETS = (
    "original_n2000",
    "original_n5000",
    "original_n10000",
    "seed20260910_n5000",
    "seed20260911_n5000",
)
FIVE_K_SUBSETS = (
    "original_n5000",
    "seed20260910_n5000",
    "seed20260911_n5000",
)
METHOD_ORDER = (
    "CNP",
    "Attentive CNP",
    "Attentive CNP + PE",
    "Density-guided CNP",
    "Kernel",
    "Bernoulli GP",
    "Pooled-data kernel",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        return json.load(stream)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def integer_identity_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def wilson_interval(successes: np.ndarray, counts: np.ndarray, z_value: float = 1.0):
    """Vectorized Wilson interval; empty bins remain undefined."""
    successes = np.asarray(successes, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    lower = np.full(counts.shape, np.nan, dtype=np.float64)
    upper = np.full(counts.shape, np.nan, dtype=np.float64)
    valid = counts > 0
    if np.any(valid):
        n = counts[valid]
        p = successes[valid] / n
        z2 = z_value * z_value
        denominator = 1.0 + z2 / n
        center = (p + z2 / (2.0 * n)) / denominator
        half = z_value * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denominator
        lower[valid] = np.maximum(0.0, center - half)
        upper[valid] = np.minimum(1.0, center + half)
    return lower, upper


def bin_events(
    energy: np.ndarray, outcome: np.ndarray, prediction: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    index = np.searchsorted(BIN_EDGES, energy, side="right") - 1
    valid = (index >= 0) & (index < BIN_CENTERS.size)
    counts = np.bincount(index[valid], minlength=BIN_CENTERS.size).astype(np.int64)
    passes = np.bincount(index[valid], weights=outcome[valid], minlength=BIN_CENTERS.size).astype(
        np.int64
    )
    fraction = np.divide(passes, counts, out=np.full(BIN_CENTERS.size, np.nan), where=counts > 0)
    lower, upper = wilson_interval(passes, counts, z_value=1.0)
    result = {
        "counts": counts,
        "passes": passes,
        "fraction": fraction,
        "wilson_lower": lower,
        "wilson_upper": upper,
        "wilson_half_width": 0.5 * (upper - lower),
    }
    if prediction is not None:
        total = np.bincount(index[valid], weights=prediction[valid], minlength=BIN_CENTERS.size)
        result["prediction_mean"] = np.divide(
            total, counts, out=np.full(BIN_CENTERS.size, np.nan), where=counts > 0
        )
    return result


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def region_masks(centers: np.ndarray = BIN_CENTERS) -> dict[str, np.ndarray]:
    masks = {}
    for name, (lower, upper, selection) in REGIONS.items():
        if selection == "center_inclusive":
            masks[name] = (centers >= lower) & (centers <= upper)
        else:
            masks[name] = (centers >= lower) & (centers < upper)
    return masks


def coverage_metrics(
    reference: Mapping[str, np.ndarray],
    prediction_mean: np.ndarray,
    selected: np.ndarray,
) -> dict[str, Any]:
    """Compute reference-band coverage; model uncertainty is not an input."""
    counts = np.asarray(reference["counts"])
    supported = np.asarray(selected, dtype=bool) & (counts >= MIN_BIN_EVENTS)
    excluded = np.asarray(selected, dtype=bool) & ~supported
    n_supported = int(supported.sum())
    if n_supported == 0:
        return {
            "supported_bin_count": 0,
            "excluded_bin_count": int(excluded.sum()),
            "included_target_count": 0,
            "excluded_target_count": int(counts[excluded].sum()),
            **{f"hit_count_C{k}": None for k in KS},
            **{f"C{k}_percent": None for k in KS},
        }
    half_width = np.asarray(reference["wilson_half_width"])[supported]
    if np.any(~np.isfinite(half_width)) or np.any(half_width <= 0):
        raise ValueError("Supported reference bins must have positive Wilson half-widths")
    pull = (
        np.asarray(prediction_mean)[supported] - np.asarray(reference["fraction"])[supported]
    ) / half_width
    result: dict[str, Any] = {
        "supported_bin_count": n_supported,
        "excluded_bin_count": int(excluded.sum()),
        "included_target_count": int(counts[supported].sum()),
        "excluded_target_count": int(counts[excluded].sum()),
    }
    previous = -1
    for k in KS:
        hits = int(np.count_nonzero(np.abs(pull) <= k))
        if hits < previous:
            raise AssertionError("Coverage hit counts are not monotone")
        previous = hits
        result[f"hit_count_C{k}"] = hits
        result[f"C{k}_percent"] = 100.0 * hits / n_supported
    return result


def prediction_bins_from_record(
    repo: Path,
    record: Mapping[str, Any],
    kernel_cache: dict[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """Load one saved cell and reconcile actual-event and archived bin means."""
    event_path = repo / str(record["event_predictions_path"])
    curve_path = repo / str(record["curve_path"])
    if not event_path.is_file() or not curve_path.is_file():
        raise FileNotFoundError(f"Missing saved artifact for {record['run_id']}")
    event_hash = sha256_file(event_path)
    curve_hash = sha256_file(curve_path)
    if event_hash != record["event_predictions_sha256"]:
        raise ValueError(f"Event prediction hash mismatch: {record['run_id']}")
    if curve_hash != record["curve_sha256"]:
        raise ValueError(f"Curve hash mismatch: {record['run_id']}")

    if str(record["architecture_id"]).startswith("kernel_"):
        if not kernel_cache:
            archive = np.load(event_path)
            kernel_cache.update({key: archive[key] for key in archive.files})
            archive.close()
        variant = (
            "context_only" if record["architecture_id"] == "kernel_context_only" else "pooled_data"
        )
        variant_index = [str(x) for x in kernel_cache["variants"]].index(variant)
        size_index = kernel_cache["context_sizes"].astype(int).tolist().index(500)
        context_index = (
            kernel_cache["context_seeds"].astype(int).tolist().index(int(record["context_seed"]))
        )
        rows = kernel_cache["final_target_rows"].astype(np.int64)
        energy = kernel_cache["final_energy_kev"].astype(np.float64)
        outcome = kernel_cache["final_outcome"].astype(np.int8)
        prediction = kernel_cache["predictions"][variant_index, size_index, context_index].astype(
            np.float64
        )
        binned = bin_events(energy, outcome, prediction)
        audit = {
            "route": "kernel consolidated event predictions; no separate archived bin array",
            "maximum_bin_mean_difference": None,
        }
        return (
            {"rows": rows, "energy": energy, "outcome": outcome, "context_rows": None},
            binned["prediction_mean"],
            audit,
        )

    with np.load(event_path) as events:
        rows = events["target_rows"].astype(np.int64)
        energy = events["target_energy_kev"].astype(np.float64)
        outcome = events["outcome"].astype(np.int8)
        prediction = events["prediction"].astype(np.float64)
        context_rows = events["context_rows"].astype(np.int64)
    binned = bin_events(energy, outcome, prediction)
    with np.load(curve_path) as curve:
        centers = curve["bin_centers_kev"].astype(np.float64)
        archived_counts = curve["bin_counts"].astype(np.int64)
        archived_fraction = curve["empirical_rate"].astype(np.float64)
        archived_prediction = curve["bin_prediction_mean"].astype(np.float64)
    if not np.array_equal(centers, BIN_CENTERS):
        raise ValueError(f"Archived bin-center mismatch: {record['run_id']}")
    direct_reference = bin_events(energy, outcome)
    if not np.array_equal(archived_counts, direct_reference["counts"]):
        raise ValueError(f"Archived bin-count mismatch: {record['run_id']}")
    if not np.allclose(
        archived_fraction, direct_reference["fraction"], atol=0, rtol=0, equal_nan=True
    ):
        raise ValueError(f"Archived empirical-fraction mismatch: {record['run_id']}")
    difference = float(np.nanmax(np.abs(archived_prediction - binned["prediction_mean"])))
    if difference > 2e-14:
        raise ValueError(f"Actual-event bin-mean mismatch: {record['run_id']}: {difference}")
    audit = {
        "route": "event prediction means reconciled against saved curve bin means",
        "maximum_bin_mean_difference": difference,
    }
    return (
        {
            "rows": rows,
            "energy": energy,
            "outcome": outcome,
            "context_rows": context_rows,
        },
        binned["prediction_mean"],
        audit,
    )


def source_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "subset_id": record["subset_id"],
        "ordering_id": record["ordering_id"],
        "budget_label": record["budget_label"],
        "training_budget_nominal_events": record["training_budget_nominal_events"],
        "training_budget_sampling_eligible_events": record[
            "training_budget_sampling_eligible_events"
        ],
        "method": record["method"],
        "architecture_id": record["architecture_id"],
        "comparison_role": record["comparison_role"],
        "training_seed": record.get("training_seed"),
        "context_seed": record["context_seed"],
        "context_size": 500,
        "run_id": record["run_id"],
        "source_event_predictions_sha256": record["event_predictions_sha256"],
        "source_curve_sha256": record["curve_sha256"],
    }


def add_composites(rows: list[dict[str, Any]]) -> None:
    by_cell: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_cell[str(row["run_id"])][str(row["region"])] = row
    additions = []
    for _run_id, regions in by_cell.items():
        for composite, components in COMPOSITES.items():
            selected = [regions[name] for name in components]
            row = {
                key: value
                for key, value in selected[0].items()
                if key
                not in {
                    "region",
                    "region_weighting",
                    "supported_bin_count",
                    "excluded_bin_count",
                    "included_target_count",
                    "excluded_target_count",
                    "hit_count_C1",
                    "hit_count_C2",
                    "hit_count_C3",
                    "C1_percent",
                    "C2_percent",
                    "C3_percent",
                    "component_regions",
                    "component_supported_bin_counts",
                    "component_hit_counts",
                }
            }
            row.update(
                {
                    "region": composite,
                    "region_weighting": "equal_component_mean_per_cell",
                    "supported_bin_count": sum(
                        int(item["supported_bin_count"]) for item in selected
                    ),
                    "excluded_bin_count": sum(int(item["excluded_bin_count"]) for item in selected),
                    "included_target_count": sum(
                        int(item["included_target_count"]) for item in selected
                    ),
                    "excluded_target_count": sum(
                        int(item["excluded_target_count"]) for item in selected
                    ),
                    "component_regions": "|".join(components),
                    "component_supported_bin_counts": "|".join(
                        str(item["supported_bin_count"]) for item in selected
                    ),
                    "component_hit_counts": "|".join(
                        ",".join(str(item[f"hit_count_C{k}"]) for k in KS) for item in selected
                    ),
                }
            )
            for k in KS:
                row[f"hit_count_C{k}"] = None
                row[f"C{k}_percent"] = float(
                    np.mean([float(item[f"C{k}_percent"]) for item in selected])
                )
            additions.append(row)
    rows.extend(additions)


def group_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["subset_id"],
        row["ordering_id"],
        row["budget_label"],
        row["training_budget_nominal_events"],
        row["training_budget_sampling_eligible_events"],
        row["method"],
        row["architecture_id"],
        row["comparison_role"],
        row["region"],
    )


def summarize_cells(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[group_identity(row)].append(row)
    summaries: list[dict[str, Any]] = []
    for _key, group in grouped.items():
        first = group[0]
        seed_groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for item in group:
            if item["training_seed"] not in (None, ""):
                seed_groups[int(item["training_seed"])].append(item)
        summary = {
            "summary_level": "training_subset",
            **{
                name: first[name]
                for name in (
                    "subset_id",
                    "ordering_id",
                    "budget_label",
                    "training_budget_nominal_events",
                    "training_budget_sampling_eligible_events",
                    "method",
                    "architecture_id",
                    "comparison_role",
                    "region",
                )
            },
            "cell_count": len(group),
            "initialization_seed_count": len(seed_groups),
            "context_seed_count_per_initialization": 10 if seed_groups else len(group),
            "aggregation": (
                "contexts averaged within initialization seed; initialization seeds equally averaged"
                if seed_groups
                else "equal mean across ten contexts"
            ),
            "source_event_prediction_hash_set_sha256": sha256_text(
                "\n".join(sorted(str(item["source_event_predictions_sha256"]) for item in group))
            ),
            "source_curve_hash_set_sha256": sha256_text(
                "\n".join(sorted(str(item["source_curve_sha256"]) for item in group))
            ),
        }
        for k in KS:
            metric = f"C{k}_percent"
            if seed_groups:
                seed_means = np.array(
                    [
                        np.mean([float(item[metric]) for item in seed_groups[seed]])
                        for seed in sorted(seed_groups)
                    ]
                )
                context_sds = np.array(
                    [
                        np.std([float(item[metric]) for item in seed_groups[seed]], ddof=1)
                        for seed in sorted(seed_groups)
                    ]
                )
                summary[f"{metric}_mean"] = float(seed_means.mean())
                summary[f"{metric}_mean_context_sd_within_seed"] = float(context_sds.mean())
                summary[f"{metric}_initialization_sd_across_seed_means"] = float(
                    seed_means.std(ddof=1)
                )
            else:
                values = np.array([float(item[metric]) for item in group])
                summary[f"{metric}_mean"] = float(values.mean())
                summary[f"{metric}_mean_context_sd_within_seed"] = float(values.std(ddof=1))
                summary[f"{metric}_initialization_sd_across_seed_means"] = None
            summary[f"{metric}_training_subset_sd_across_subset_means"] = None
            summary[f"{metric}_training_subset_min"] = None
            summary[f"{metric}_training_subset_max"] = None
            summary[f"{metric}_training_subset_range"] = None
        summaries.append(summary)

    neural = [row for row in summaries if row["subset_id"] in FIVE_K_SUBSETS]
    by_method_region: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in neural:
        by_method_region[(row["method"], row["architecture_id"], row["region"])].append(row)
    for (_method, _architecture, _region), group in by_method_region.items():
        if len(group) != 3:
            raise ValueError("Expected three independent 5k subset summaries")
        first = group[0]
        summary = {
            "summary_level": "all_three_5k_training_subsets",
            "subset_id": "all_three_5k_subsets_equal_weight",
            "ordering_id": "original|seed20260910|seed20260911",
            "budget_label": "5k",
            "training_budget_nominal_events": 5000,
            "training_budget_sampling_eligible_events": "4984|4981|4980",
            "method": first["method"],
            "architecture_id": first["architecture_id"],
            "comparison_role": first["comparison_role"],
            "region": first["region"],
            "cell_count": 90,
            "initialization_seed_count": 3,
            "context_seed_count_per_initialization": 10,
            "training_subset_count": 3,
            "aggregation": "equal mean of three subset means; each subset uses seed-then-context hierarchy",
            "source_event_prediction_hash_set_sha256": sha256_text(
                "\n".join(
                    sorted(str(item["source_event_prediction_hash_set_sha256"]) for item in group)
                )
            ),
            "source_curve_hash_set_sha256": sha256_text(
                "\n".join(sorted(str(item["source_curve_hash_set_sha256"]) for item in group))
            ),
        }
        for k in KS:
            metric = f"C{k}_percent"
            values = np.array([float(item[f"{metric}_mean"]) for item in group])
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_mean_context_sd_within_seed"] = float(
                np.mean([float(item[f"{metric}_mean_context_sd_within_seed"]) for item in group])
            )
            summary[f"{metric}_initialization_sd_across_seed_means"] = float(
                np.mean(
                    [float(item[f"{metric}_initialization_sd_across_seed_means"]) for item in group]
                )
            )
            summary[f"{metric}_training_subset_sd_across_subset_means"] = float(values.std(ddof=1))
            summary[f"{metric}_training_subset_min"] = float(values.min())
            summary[f"{metric}_training_subset_max"] = float(values.max())
            summary[f"{metric}_training_subset_range"] = float(values.max() - values.min())
        summaries.append(summary)
    return summaries


def build_budget_rows(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    allowed_regions = {
        "overall_500_3000",
        "FE_2614_pm5",
        "SE_2103_pm5",
        "DEP_1592_pm5",
        "feature_1620_pm5",
        "peak_equal_feature",
        "continuum_equal_window",
    }
    result = []
    for row in summaries:
        if row["summary_level"] != "training_subset" or row["subset_id"] not in NEURAL_SUBSETS:
            continue
        if row["region"] not in allowed_regions:
            continue
        for k in KS:
            result.append(
                {
                    "subset_id": row["subset_id"],
                    "ordering_id": row["ordering_id"],
                    "budget_label": row["budget_label"],
                    "training_budget_nominal_events": row["training_budget_nominal_events"],
                    "training_budget_sampling_eligible_events": row[
                        "training_budget_sampling_eligible_events"
                    ],
                    "method": row["method"],
                    "architecture_id": row["architecture_id"],
                    "region": row["region"],
                    "threshold_k": k,
                    "coverage_percent_mean": row[f"C{k}_percent_mean"],
                    "mean_context_sd_within_seed_pp": row[
                        f"C{k}_percent_mean_context_sd_within_seed"
                    ],
                    "initialization_sd_across_seed_means_pp": row[
                        f"C{k}_percent_initialization_sd_across_seed_means"
                    ],
                    "cell_count": row["cell_count"],
                }
            )
    return result


def build_pair_rows(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (
            row["subset_id"],
            row["method"],
            row["training_seed"],
            row["context_seed"],
            row["region"],
        ): row
        for row in cells
    }
    result: list[dict[str, Any]] = []
    for subset in FIVE_K_SUBSETS:
        for comparator in METHOD_ORDER[:4]:
            for seed in (0, 1, 2):
                for context in range(100, 110):
                    for region in ("peak_equal_feature", "continuum_equal_window"):
                        ours = lookup[(subset, "Density-guided CNP", seed, context, region)]
                        other = lookup[("original_n10000", comparator, seed, context, region)]
                        for k in KS:
                            ours_value = float(ours[f"C{k}_percent"])
                            other_value = float(other[f"C{k}_percent"])
                            difference = ours_value - other_value
                            result.append(
                                {
                                    "row_type": "paired_cell",
                                    "five_k_subset_id": subset,
                                    "ten_k_subset_id": "original_n10000",
                                    "ten_k_comparator": comparator,
                                    "training_seed": seed,
                                    "context_seed": context,
                                    "region": region,
                                    "threshold_k": k,
                                    "five_k_ours_coverage_percent": ours_value,
                                    "ten_k_comparator_coverage_percent": other_value,
                                    "difference_percentage_points": difference,
                                    "comparison": "tie"
                                    if difference == 0
                                    else ("win" if difference > 0 else "loss"),
                                    "five_k_run_id": ours["run_id"],
                                    "ten_k_run_id": other["run_id"],
                                    "five_k_source_sha256": ours["source_event_predictions_sha256"],
                                    "ten_k_source_sha256": other["source_event_predictions_sha256"],
                                }
                            )
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in result:
        grouped[
            (row["five_k_subset_id"], row["ten_k_comparator"], row["region"], row["threshold_k"])
        ].append(row)
    additions = []
    for (subset, comparator, region, k), group in grouped.items():
        differences = np.array([row["difference_percentage_points"] for row in group])
        additions.append(
            {
                "row_type": "paired_summary",
                "five_k_subset_id": subset,
                "ten_k_subset_id": "original_n10000",
                "ten_k_comparator": comparator,
                "training_seed": None,
                "context_seed": None,
                "region": region,
                "threshold_k": k,
                "paired_cell_count": len(group),
                "mean_difference_percentage_points": float(differences.mean()),
                "difference_sd_percentage_points": float(differences.std(ddof=1)),
                "win_count": sum(row["comparison"] == "win" for row in group),
                "tie_count": sum(row["comparison"] == "tie" for row in group),
                "loss_count": sum(row["comparison"] == "loss" for row in group),
                "pairing_note": "descriptive pairing by initialization and context seed; does not remove subset uncertainty",
            }
        )
    result.extend(additions)
    return result


def report_text(
    summary: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    checks: Mapping[str, Any],
    runtime: float,
) -> str:
    lookup = {
        (r["subset_id"], r["method"], r["region"]): r
        for r in summary
        if r["summary_level"] == "training_subset"
    }
    display_regions = (
        ("overall_500_3000", "Overall"),
        ("FE_2614_pm5", "FE"),
        ("SE_2103_pm5", "SE"),
        ("DEP_1592_pm5", "DEP"),
        ("feature_1620_pm5", "1620"),
        ("continuum_equal_window", "Continuum"),
    )
    headline_methods = METHOD_ORDER[:6]
    lines = [
        "# Paper regional reference-band coverage export",
        "",
        "## Status",
        "",
        f"This saved-artifact-only export completed in {runtime:.1f} seconds. It recovered 600 neural cells, 10 context-only kernel cells, 10 selected Matérn-3/2 Bernoulli-GP cells, and 10 pooled-kernel control cells. No training, fitting, prediction, MC sampling, or tuning was run.",
        "",
        "Reference-band coverage is the percentage of supported 5-keV bins for which the saved actual-event bin mean lies within `k` times the frozen z=1 Wilson half-width of the empirical target fraction. It is descriptive agreement with a finite empirical reference, not calibrated model-interval coverage, coverage of unknown truth, or the slide's pooled Gaussian transform G.",
        "",
        "## Main-paper Table 1 candidate: original 5k and context-only comparators",
        "",
        "C2 reference-band coverage (%); whole percentages are display rounding. Continuum is the equal per-cell mean of [1700,2000) and [2200,2400), not an event-pooled score.",
        "",
        "| Method | " + " | ".join(label for _, label in display_regions) + " |",
        "|---|" + "---:|" * len(display_regions),
    ]
    for method in headline_methods:
        subset = "original_n5000" if method in METHOD_ORDER[:4] else "context_only"
        values = [
            lookup[(subset, method, region)]["C2_percent_mean"] for region, _ in display_regions
        ]
        lines.append("| " + method + " | " + " | ".join(f"{value:.0f}" for value in values) + " |")
    lines += ["", "Sensitivity (C1/C2/C3, %):", ""]
    for method in headline_methods:
        subset = "original_n5000" if method in METHOD_ORDER[:4] else "context_only"
        chunks = []
        for region, label in display_regions:
            row = lookup[(subset, method, region)]
            chunks.append(
                f"{label} {row['C1_percent_mean']:.1f}/{row['C2_percent_mean']:.1f}/{row['C3_percent_mean']:.1f}"
            )
        lines.append(f"- {method}: " + "; ".join(chunks) + ".")
    lines += [
        "",
        "The pooled-data kernel is excluded from this matched headline because it uses the full 18,866-event efficiency-training pool plus context; it remains in the CSV as a stronger-data control.",
        "",
        "## Main-paper Table 2 candidate: original training ordering",
        "",
        "C2 reference-band coverage (%). Peaks is the equal per-cell mean of FE, SE, DEP, and the 1620-keV feature; Continuum is the equal per-cell mean of the two fixed windows.",
        "",
        "| Method | 2k Overall | 2k Peaks | 2k Continuum | 5k Overall | 5k Peaks | 5k Continuum | 10k Overall | 10k Peaks | 10k Continuum |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    budget_subsets = (
        ("original_n2000", "2k"),
        ("original_n5000", "5k"),
        ("original_n10000", "10k"),
    )
    for method in METHOD_ORDER[:4]:
        values = []
        for subset, _label in budget_subsets:
            for region in ("overall_500_3000", "peak_equal_feature", "continuum_equal_window"):
                values.append(float(lookup[(subset, method, region)]["C2_percent_mean"]))
        lines.append("| " + method + " | " + " | ".join(f"{value:.0f}" for value in values) + " |")

    ours_budget = {
        label: {
            region: float(lookup[(subset, "Density-guided CNP", region)]["C2_percent_mean"])
            for region in ("overall_500_3000", "peak_equal_feature", "continuum_equal_window")
        }
        for subset, label in budget_subsets
    }
    lines += [
        "",
        "## Budget evidence and fixed comparison",
        "",
        "For Density-guided CNP, original-ordering C2 coverage (Overall / equal-feature Peaks / equal-window Continuum) is "
        + "; ".join(
            f"{label}: {v['overall_500_3000']:.1f}/{v['peak_equal_feature']:.1f}/{v['continuum_equal_window']:.1f}%"
            for label, v in ours_budget.items()
        )
        + ".",
    ]
    if any(ours_budget["10k"][region] < ours_budget["5k"][region] for region in ours_budget["5k"]):
        decreases = [
            region
            for region in ours_budget["5k"]
            if ours_budget["10k"][region] < ours_budget["5k"][region]
        ]
        lines.append(
            "The 10k result is nonmonotonic relative to 5k for: " + ", ".join(decreases) + "."
        )
    else:
        lines.append(
            "C2 coverage is nondecreasing from 5k to 10k in all three displayed summaries; other metrics still show the previously documented 10k nonmonotonicity."
        )
    lines.append(
        "The 2k row quantifies the low-budget failure directly; it must remain visible rather than being omitted from the learning curve."
    )
    lines += ["", "C2 change from original 5k to original 10k (percentage points):", ""]
    for method in METHOD_ORDER[:4]:
        deltas = []
        for region, label in (
            ("overall_500_3000", "Overall"),
            ("peak_equal_feature", "Peaks"),
            ("continuum_equal_window", "Continuum"),
        ):
            delta = float(lookup[("original_n10000", method, region)]["C2_percent_mean"]) - float(
                lookup[("original_n5000", method, region)]["C2_percent_mean"]
            )
            deltas.append(f"{label} {delta:+.1f}")
        lines.append(f"- {method}: " + "; ".join(deltas) + ".")

    pair_summary = [r for r in pairs if r["row_type"] == "paired_summary"]
    exceptions = []
    for subset in FIVE_K_SUBSETS:
        for comparator in METHOD_ORDER[:4]:
            for k in KS:
                vals = {
                    (r["region"]): r["mean_difference_percentage_points"]
                    for r in pair_summary
                    if r["five_k_subset_id"] == subset
                    and r["ten_k_comparator"] == comparator
                    and r["threshold_k"] == k
                }
                if vals["peak_equal_feature"] < 0 or vals["continuum_equal_window"] < 0:
                    exceptions.append(
                        f"{subset} vs {comparator} at k={k}: Peaks {vals['peak_equal_feature']:+.2f} pp, Continuum {vals['continuum_equal_window']:+.2f} pp"
                    )
    lines += [
        "",
        "Exact exceptions to the statement that 5k ours matches or improves both peak and continuum coverage relative to every original-10k neural comparator:",
        "",
    ]
    if exceptions:
        lines.extend(f"- {item}." for item in exceptions)
    else:
        lines.append("- None for the frozen k=1,2,3 comparisons.")

    feature_bins = {
        name: [float(BIN_CENTERS[i]) for i in np.flatnonzero(region_masks()[name])]
        for name in ("FE_2614_pm5", "SE_2103_pm5", "DEP_1592_pm5", "feature_1620_pm5")
    }
    lines += [
        "",
        "## Region construction and checks",
        "",
        "Bins are selected by center. The narrow memberships are "
        + "; ".join(f"{name}: {centers}" for name, centers in feature_bins.items())
        + ". These two-bin feature scores are discrete (0%, 50%, or 100% per cell at each k). They are not identical to the previous inclusive +/-5-keV pooled-event G regions because boundary event membership differs.",
        "",
        f"Overall contains {checks['overall_supported_bins']} supported and {checks['overall_excluded_bins']} excluded bins; excluded bins contain {checks['overall_excluded_target_events']} target events. All methods use the identical support and frozen 114,400 targets.",
        "",
        "Aggregation first averages the ten contexts within each initialization seed, then equally averages the three initialization seeds. The three 5k training-subset summaries are kept separate; their combined row equally averages subset means and reports subset range and descriptive SD. Overlapping contexts are not treated as independent datasets.",
        "",
        "All 630 source hashes were checked. Direct event-level bin means were reconciled with saved neural/GP `curve.npz` means to a maximum absolute difference of "
        + f"{checks['maximum_bin_reconciliation_difference']:.3g}. The consolidated kernel archive has no second stored bin-mean route; its predictions and frozen identities were verified directly.",
        "",
        "Exact context membership hashes agree across all neural and GP cells for seeds 100--109. The consolidated kernel archive records context seeds but not context-row identities, so its context association remains inherited from the frozen kernel protocol rather than independently recoverable from that output file.",
        "",
        f"The newly computed global C1/C2/C3 values reproduce all {checks['global_rows_reproduced']} available rows in `paper_reference_pull_cells.csv` within {checks['global_reproduction_max_abs_difference']:.3g} in fraction units.",
        "",
        "## Interpretation limits",
        "",
        "Global coverage can reward continuum agreement while hiding missed narrow features, so the global, equal-feature, and equal-window columns must be read together. Higher reference-band coverage can also reflect broad finite-reference bands; it is not a calibrated 68/95/99.7% statement.",
        "",
        "The existing 5k result supports an efficiency-model training-budget comparison conditional on a fixed classifier trained with 18,866 events and 500 conditioning events. It does not establish end-to-end training on 5,000 events. There is no 20k efficiency-training run; the archived full pool is exactly 18,866 events.",
        "",
        "Sparse-tail limitations from the previous export remain: at original 5k, Density-guided CNP sparse-tail MAE was 19.39 percentage points versus 2.99 for the stronger-data pooled-kernel control. This export does not redefine the tail or claim calibrated uncertainty.",
        "",
        "No missing cells or scientific failures were replaced by new computation. The nonlinear per-cell fractions here cannot be reconstructed from the compact mean curves alone; they were calculated from the saved per-cell event predictions.",
        "",
        "## Portable files",
        "",
        "- `paper_coverage_cells.csv`: all per-cell individual and composite regional scores.",
        "- `paper_coverage_summary.csv`: hierarchical setting summaries and all-three-5k subset dispersion.",
        "- `paper_coverage_region_bins.csv`: exact bin membership, target counts, fractions, and support.",
        "- `paper_coverage_budget_comparison.csv`: all thresholds for the fixed neural budget comparison.",
        "- `paper_coverage_cross_budget_pairs.csv`: descriptive paired 5k-ours versus original-10k comparisons, including ties.",
        "- `paper_coverage_export.json`: definitions, source/output hashes, checks, and provenance.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("ml4phy-paper"))
    parser.add_argument("--wall-clock-limit-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    started = time.perf_counter()
    repo = args.repo.resolve()
    paper_root = repo / "ml4phy-paper"
    output_root = args.output_root if args.output_root.is_absolute() else repo / args.output_root

    def check_time() -> None:
        if time.perf_counter() - started > args.wall_clock_limit_seconds:
            raise TimeoutError("Saved-artifact export exceeded its wall-clock limit")

    prior_manifest_path = paper_root / "manifests/paper_presentation_export.json"
    prior_pull_path = paper_root / "tables/paper_reference_pull_cells.csv"
    prior_manifest = read_json(prior_manifest_path)
    records = prior_manifest["source_artifacts"]
    if len(records) != 630:
        raise ValueError(f"Expected 630 frozen source records, found {len(records)}")
    if len({record["run_id"] for record in records}) != 630:
        raise ValueError("Source run IDs are not unique")

    kernel_cache: dict[str, Any] = {}
    context_identity_by_seed: dict[int, str] = {}
    reference_identity: dict[str, np.ndarray] | None = None
    reference: dict[str, np.ndarray] | None = None
    cell_rows: list[dict[str, Any]] = []
    maximum_reconciliation = 0.0
    masks = region_masks()
    recovered = defaultdict(int)
    missing: list[str] = []
    for index, record in enumerate(records):
        check_time()
        try:
            identity, prediction_bins, audit = prediction_bins_from_record(
                repo, record, kernel_cache
            )
        except FileNotFoundError:
            missing.append(str(record["run_id"]))
            continue
        if reference_identity is None:
            reference_identity = identity
            reference = bin_events(identity["energy"], identity["outcome"])
        else:
            if not np.array_equal(identity["rows"], reference_identity["rows"]):
                raise ValueError(f"Frozen target-row mismatch: {record['run_id']}")
            if not np.array_equal(identity["energy"], reference_identity["energy"]):
                raise ValueError(f"Frozen target-energy mismatch: {record['run_id']}")
            if not np.array_equal(identity["outcome"], reference_identity["outcome"]):
                raise ValueError(f"Frozen target-outcome mismatch: {record['run_id']}")
        if audit["maximum_bin_mean_difference"] is not None:
            maximum_reconciliation = max(
                maximum_reconciliation, audit["maximum_bin_mean_difference"]
            )
        context_rows = identity["context_rows"]
        if context_rows is not None:
            context_seed = int(record["context_seed"])
            context_hash = integer_identity_sha256(np.sort(context_rows))
            if (
                context_seed in context_identity_by_seed
                and context_identity_by_seed[context_seed] != context_hash
            ):
                raise ValueError(f"Paired context membership mismatch for seed {context_seed}")
            context_identity_by_seed[context_seed] = context_hash
        assert reference is not None
        for region, selected in masks.items():
            cell_rows.append(
                {
                    **source_fields(record),
                    "region": region,
                    "region_weighting": "equal_supported_bins",
                    "component_regions": None,
                    "component_supported_bin_counts": None,
                    "component_hit_counts": None,
                    **coverage_metrics(reference, prediction_bins, selected),
                }
            )
        architecture = str(record["architecture_id"])
        recovered[
            "neural"
            if architecture in ("m0", "m1", "m2", "ours")
            else ("gp" if architecture == "bernoulli_gp_matern32" else "kernel")
        ] += 1
        if (index + 1) % 100 == 0:
            print(f"verified {index + 1}/630 saved cells", flush=True)
    if missing:
        raise FileNotFoundError(f"Missing saved cells (no recomputation authorized): {missing}")
    if dict(recovered) != {"neural": 600, "kernel": 20, "gp": 10}:
        raise ValueError(f"Unexpected recovered matrix: {dict(recovered)}")
    assert reference is not None and reference_identity is not None
    if int(reference["counts"].sum()) != 114400:
        raise ValueError("Frozen target count is not 114,400")
    add_composites(cell_rows)

    # Required per-cell global reproduction against the completed export.
    prior_global = {
        row["run_id"]: row for row in read_csv(prior_pull_path) if row["scope"] == "full_500_3000"
    }
    new_global = {row["run_id"]: row for row in cell_rows if row["region"] == "overall_500_3000"}
    if set(prior_global) != set(new_global):
        raise ValueError("Global reproduction run-ID support differs")
    global_differences = []
    for run_id, previous in prior_global.items():
        for k in KS:
            global_differences.append(
                abs(
                    float(previous[f"reference_C{k}"])
                    - float(new_global[run_id][f"C{k}_percent"]) / 100.0
                )
            )
    max_global_difference = max(global_differences)
    if max_global_difference > 2e-15:
        raise ValueError(f"Global coverage reproduction failed: {max_global_difference}")

    overall = coverage_metrics(reference, reference["fraction"], masks["overall_500_3000"])
    if overall["supported_bin_count"] != 442 or overall["excluded_bin_count"] != 58:
        raise ValueError(f"Unexpected global support: {overall}")
    for row in cell_rows:
        if row["region"] not in COMPOSITES:
            supported = int(row["supported_bin_count"])
            for k in KS:
                hits = int(row[f"hit_count_C{k}"])
                coverage = float(row[f"C{k}_percent"])
                if not 0 <= hits <= supported or not 0 <= coverage <= 100:
                    raise ValueError("Coverage bounds or hit consistency failed")
                if abs(coverage - 100.0 * hits / supported) > 1e-13:
                    raise ValueError("Integer hit-count consistency failed")
            if not row["C1_percent"] <= row["C2_percent"] <= row["C3_percent"]:
                raise ValueError("Coverage monotonicity failed")
        elif not row["C1_percent"] <= row["C2_percent"] <= row["C3_percent"]:
            raise ValueError("Composite coverage monotonicity failed")

    setting_counts = defaultdict(int)
    for row in cell_rows:
        if row["region"] == "overall_500_3000":
            setting_counts[(row["subset_id"], row["method"])] += 1
    for (subset, method), count in setting_counts.items():
        expected = 30 if subset in NEURAL_SUBSETS else 10
        if count != expected:
            raise ValueError(f"Unexpected setting cell count {subset}/{method}: {count}")

    region_bin_rows = []
    for region, selected in masks.items():
        for bin_index in np.flatnonzero(selected):
            region_bin_rows.append(
                {
                    "row_type": "individual_region_bin",
                    "region": region,
                    "selection_rule": REGIONS[region][2],
                    "bin_index": int(bin_index),
                    "energy_lower_kev": float(BIN_EDGES[bin_index]),
                    "energy_upper_kev": float(BIN_EDGES[bin_index + 1]),
                    "energy_center_kev": float(BIN_CENTERS[bin_index]),
                    "target_event_count": int(reference["counts"][bin_index]),
                    "target_pass_count": int(reference["passes"][bin_index]),
                    "target_fraction": reference["fraction"][bin_index],
                    "wilson_lower_z1": reference["wilson_lower"][bin_index],
                    "wilson_upper_z1": reference["wilson_upper"][bin_index],
                    "wilson_half_width_z1": reference["wilson_half_width"][bin_index],
                    "supported_min_4_events": bool(
                        reference["counts"][bin_index] >= MIN_BIN_EVENTS
                    ),
                }
            )
    for composite, components in COMPOSITES.items():
        for component in components:
            region_bin_rows.append(
                {
                    "row_type": "composite_component",
                    "region": composite,
                    "component_region": component,
                    "selection_rule": "equal_component_mean_per_cell",
                }
            )

    summaries = summarize_cells(cell_rows)
    budget_rows = build_budget_rows(summaries)
    pair_rows = build_pair_rows(cell_rows)
    checks = {
        "source_record_count": len(records),
        "unique_run_id_count": len({r["run_id"] for r in records}),
        "recovered_counts": dict(recovered),
        "missing_run_ids": missing,
        "frozen_target_count": int(reference["counts"].sum()),
        "frozen_target_identity_sha256": prior_manifest["fixed_protocol"]["target_identity_sha256"],
        "overall_supported_bins": overall["supported_bin_count"],
        "overall_excluded_bins": overall["excluded_bin_count"],
        "overall_excluded_target_events": overall["excluded_target_count"],
        "global_rows_reproduced": len(prior_global),
        "global_reproduction_max_abs_difference": max_global_difference,
        "maximum_bin_reconciliation_difference": maximum_reconciliation,
        "setting_cell_counts": {
            f"{key[0]}::{key[1]}": value for key, value in setting_counts.items()
        },
        "paired_context_seed_identity_hashes": {
            str(seed): value for seed, value in sorted(context_identity_by_seed.items())
        },
        "paired_context_membership_across_neural_and_gp_cells": (
            "passed" if set(context_identity_by_seed) == set(range(100, 110)) else "failed"
        ),
        "coverage_monotonicity_bounds_and_integer_hits": "passed",
        "equal_feature_and_equal_window_composites": "passed",
        "identical_target_identity_and_support": "passed",
    }

    report_path = output_root / "reports/paper_coverage_export.md"
    outputs = {
        "cells": output_root / "tables/paper_coverage_cells.csv",
        "summary": output_root / "tables/paper_coverage_summary.csv",
        "region_bins": output_root / "tables/paper_coverage_region_bins.csv",
        "budget_comparison": output_root / "tables/paper_coverage_budget_comparison.csv",
        "cross_budget_pairs": output_root / "tables/paper_coverage_cross_budget_pairs.csv",
    }
    for name, path in outputs.items():
        rows = {
            "cells": cell_rows,
            "summary": summaries,
            "region_bins": region_bin_rows,
            "budget_comparison": budget_rows,
            "cross_budget_pairs": pair_rows,
        }[name]
        write_csv(path, rows)
    runtime_before_report = time.perf_counter() - started
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text(summaries, pair_rows, checks, runtime_before_report))

    script_path = Path(__file__).resolve()
    output_hashes = {
        str(path.relative_to(repo)): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in [*outputs.values(), report_path]
    }
    manifest_path = output_root / "manifests/paper_coverage_export.json"
    manifest = {
        "schema_version": 1,
        "analysis": "paper_regional_reference_band_coverage",
        "status": "complete",
        "created_at": utc_now(),
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "script": str(script_path.relative_to(repo)),
        "script_sha256": sha256_file(script_path),
        "no_new_training_fitting_prediction_or_selection": True,
        "wall_clock_limit_seconds": args.wall_clock_limit_seconds,
        "definition": {
            "name": "reference-band coverage",
            "bin_width_kev": 5.0,
            "energy_range_kev": [500.0, 3000.0],
            "minimum_target_events_per_bin": 4,
            "reference_interval": "Wilson z=1; one fixed half-width multiplied by k",
            "pull": "(saved actual-event bin mean - empirical fraction) / Wilson z=1 half-width",
            "coverage_percent": "100 * supported-bin fraction with absolute pull <= k",
            "thresholds_k": list(KS),
            "not_model_interval_coverage": True,
            "not_unknown_truth_coverage": True,
            "not_slide_pooled_G": True,
            "model_sd_used": False,
        },
        "regions": {
            name: {"lower_kev": spec[0], "upper_kev": spec[1], "selection": spec[2]}
            for name, spec in REGIONS.items()
        },
        "composites": {
            name: {"components": list(components), "weighting": "equal component mean per cell"}
            for name, components in COMPOSITES.items()
        },
        "aggregation": {
            "neural": "contexts within initialization seed, then equal initialization seeds",
            "classical": "equal contexts",
            "all_three_5k": "equal subset means; subset range and descriptive sample SD",
            "independence_claim": "none; overlapping contexts are descriptive repeated conditions",
        },
        "source_presentation_manifest": {
            "path": str(prior_manifest_path.relative_to(repo)),
            "sha256": sha256_file(prior_manifest_path),
            "source_artifact_count": len(records),
            "source_artifact_record_set_sha256": sha256_text(
                json.dumps(records, sort_keys=True, separators=(",", ":"))
            ),
        },
        "source_global_pull_table": {
            "path": str(prior_pull_path.relative_to(repo)),
            "sha256": sha256_file(prior_pull_path),
        },
        "checks": checks,
        "outputs": output_hashes,
        "runtime_seconds": time.perf_counter() - started,
        "historical_data_exposure": "Prospectively specified follow-up analysis on historically exposed frozen final targets.",
        "limitations": [
            "Reference-band coverage is a descriptive finite-reference agreement statistic.",
            "Narrow feature regions contain only two 5-keV bins each.",
            "Training-subset, initialization, and context dispersions are descriptive, not independent-dataset uncertainty.",
            "The classifier training budget remains 18,866 events; this is not end-to-end 5k training.",
            "The full efficiency-training pool is exactly 18,866 events, not 20k.",
            "Sparse-tail limitations remain and are not represented by the peak composite.",
            "The consolidated kernel output does not embed context-row identities; only its frozen protocol association and context seeds are recoverable.",
        ],
    }
    write_json(manifest_path, manifest)
    check_time()
    print(
        json.dumps(
            {
                "status": "complete",
                "runtime_seconds": manifest["runtime_seconds"],
                "checks": checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
