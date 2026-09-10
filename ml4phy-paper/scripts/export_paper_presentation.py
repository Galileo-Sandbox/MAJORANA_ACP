#!/usr/bin/env python3
"""Export talk-led paper diagnostics from frozen saved predictions only.

This script is intentionally an offline aggregator. It loads existing NPZ and
JSON artifacts, verifies their hashes and fixed-target contract, and never
imports model code or performs prediction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, MutableMapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))
from aggregate_phase2 import run_directory as phase2_run_directory  # noqa: E402
from aggregate_phase3 import phase3_run_dir as phase3_run_directory  # noqa: E402

ARCHITECTURES = ("m0", "m1", "m2", "ours")
METHOD_NAMES = {
    "m0": "CNP",
    "m1": "Attentive CNP",
    "m2": "Attentive CNP + PE",
    "ours": "Density-guided CNP",
}
METHOD_COLORS = {
    "CNP": "#4c78a8",
    "Attentive CNP": "#f58518",
    "Attentive CNP + PE": "#54a24b",
    "Density-guided CNP": "#e45756",
    "Kernel": "#9467bd",
    "Pooled-data kernel": "#8c564b",
    "Bernoulli GP": "#17becf",
}
TRAINING_SEEDS = (0, 1, 2)
CONTEXT_SEEDS = tuple(range(100, 110))
MIN_BIN_EVENTS = 4
BIN_EDGES = np.arange(500.0, 3000.0 + 5.0, 5.0)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
CONTINUUM_WINDOWS = {
    "continuum_1700_2000": (1700.0, 2000.0),
    "continuum_2200_2400": (2200.0, 2400.0),
}
REGIONAL_AGREEMENT = (
    ("Overall", 500.0, 3000.0, True),
    ("FE", 2609.0, 2619.0, True),
    ("SE", 2098.0, 2108.0, True),
    ("DEP", 1587.0, 1597.0, True),
    ("feature", 1615.0, 1625.0, True),
)
EVALUATION_REGIONS = {
    "full": (500.0, 3000.0, True),
    "DEP": (1577.0, 1606.0, False),
    "feature_1620kev": (1606.0, 1635.0, False),
    "continuum_1700_2000": (1700.0, 2000.0, False),
    "SE": (2088.0, 2118.0, False),
    "continuum_2200_2400": (2200.0, 2400.0, False),
    "FE": (2599.0, 2629.0, False),
    "sparse_tail": (2700.0, 3000.0, False),
}
PULL_SCOPES = {
    "full_500_3000": (500.0, 3000.0),
    "figure1_1500_3000": (1500.0, 3000.0),
}
HISTOGRAM_EDGES = np.arange(-10.0, 10.0 + 0.5, 0.5)
SUBSET_SPECS = (
    ("original_n2000", "original", 2000, "2k", 1895, "phase2"),
    ("original_n5000", "original", 5000, "5k", 4984, "phase2"),
    ("original_n10000", "original", 10000, "10k", 9980, "phase3"),
    ("seed20260910_n5000", "seed20260910", 5000, "5k", 4981, "phase3"),
    ("seed20260911_n5000", "seed20260911", 5000, "5k", 4980, "phase3"),
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        return json.load(stream)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def gaussian_agreement(z_value: float, k: int) -> float:
    """Return the slide executable's descriptive Gaussian transform."""
    return normal_cdf(k - z_value) - normal_cdf(-k - z_value)


def wilson_interval(successes: np.ndarray, counts: np.ndarray, z_value: float = 1.0):
    """Vectorized Wilson score interval; empty bins remain NaN."""
    successes = np.asarray(successes, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    lower = np.full(counts.shape, np.nan, dtype=np.float64)
    upper = np.full(counts.shape, np.nan, dtype=np.float64)
    valid = counts > 0
    if not np.any(valid):
        return lower, upper
    n = counts[valid]
    p = successes[valid] / n
    z2 = z_value * z_value
    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    half = (
        z_value
        * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
        / denominator
    )
    lower[valid] = np.maximum(0.0, center - half)
    upper[valid] = np.minimum(1.0, center + half)
    return lower, upper


def bin_events(
    energy: np.ndarray, outcome: np.ndarray, prediction: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    index = np.searchsorted(BIN_EDGES, energy, side="right") - 1
    valid_event = (index >= 0) & (index < BIN_CENTERS.size)
    counts = np.bincount(index[valid_event], minlength=BIN_CENTERS.size).astype(np.int64)
    passes = np.bincount(
        index[valid_event], weights=outcome[valid_event], minlength=BIN_CENTERS.size
    ).astype(np.int64)
    fraction = np.divide(
        passes,
        counts,
        out=np.full(BIN_CENTERS.size, np.nan, dtype=np.float64),
        where=counts > 0,
    )
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
        predicted_sum = np.bincount(
            index[valid_event], weights=prediction[valid_event], minlength=BIN_CENTERS.size
        )
        result["prediction_mean"] = np.divide(
            predicted_sum,
            counts,
            out=np.full(BIN_CENTERS.size, np.nan, dtype=np.float64),
            where=counts > 0,
        )
    return result


def region_mask(values: np.ndarray, lower: float, upper: float, inclusive_upper: bool = False):
    return (values >= lower) & (values <= upper if inclusive_upper else values < upper)


def descriptive_stats(values: Sequence[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": None, "sd": None, "min": None, "max": None}
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if array.size > 1 else None,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def neural_run_dir(
    root: Path,
    source: str,
    subset_id: str,
    budget: int,
    architecture: str,
    training_seed: int,
    context_seed: int,
) -> Path:
    if source == "phase2":
        return phase2_run_directory(
            root.parent, budget, architecture, training_seed, context_seed
        )
    return phase3_run_directory(
        root, subset_id, architecture, training_seed, context_seed
    )


def grouping_fields(meta: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "subset_id": meta["subset_id"],
        "ordering_id": meta["ordering_id"],
        "budget_label": meta["budget_label"],
        "training_budget_nominal_events": meta["training_budget_nominal_events"],
        "training_budget_sampling_eligible_events": meta[
            "training_budget_sampling_eligible_events"
        ],
        "method": meta["method"],
        "architecture_id": meta["architecture_id"],
        "comparison_role": meta["comparison_role"],
    }


def group_key(meta: Mapping[str, Any]) -> tuple[Any, ...]:
    fields = grouping_fields(meta)
    return tuple(fields[key] for key in fields)


def continuum_diagnostics(
    reference: Mapping[str, np.ndarray], prediction_mean: np.ndarray
) -> list[dict[str, Any]]:
    window_rows = []
    internal = []
    for name, (lower, upper) in CONTINUUM_WINDOWS.items():
        supported = (
            region_mask(BIN_CENTERS, lower, upper)
            & (reference["counts"] >= MIN_BIN_EVENTS)
        )
        d_value = prediction_mean[supported] - reference["fraction"][supported]
        scale = reference["wilson_half_width"][supported]
        if not np.all(scale > 0):
            raise ValueError("Non-positive Wilson scale in supported continuum bin")
        pull = d_value / scale
        bias = float(d_value.mean())
        pull_mean = float(pull.mean())
        internal.append((d_value, pull, bias, pull_mean))
        window_rows.append(
            {
                "window": name,
                "window_weighting": "equal bins within this window",
                "supported_bin_count": int(supported.sum()),
                "target_events_in_supported_bins": int(reference["counts"][supported].sum()),
                "continuum_RMSE_pp": 100.0 * float(np.sqrt(np.mean(d_value**2))),
                "continuum_bias_pp": 100.0 * bias,
                "continuum_centered_RMS_pp": 100.0
                * float(np.sqrt(np.mean((d_value - bias) ** 2))),
                "continuum_pull_RMS": float(np.sqrt(np.mean(pull**2))),
                "continuum_pull_mean": pull_mean,
                "continuum_pull_centered_RMS": float(
                    np.sqrt(np.mean((pull - pull_mean) ** 2))
                ),
            }
        )
    mean_square = float(np.mean([np.mean(item[0] ** 2) for item in internal]))
    centered_mean_square = float(
        np.mean([np.mean((item[0] - item[2]) ** 2) for item in internal])
    )
    pull_mean_square = float(np.mean([np.mean(item[1] ** 2) for item in internal]))
    pull_centered_mean_square = float(
        np.mean([np.mean((item[1] - item[3]) ** 2) for item in internal])
    )
    window_rows.append(
        {
            "window": "equal_window_mean",
            "window_weighting": (
                "equal mean of the two window-specific mean-square quantities before "
                "square root; signed bias/pull mean are equal means of window means"
            ),
            "supported_bin_count": int(sum(len(item[0]) for item in internal)),
            "target_events_in_supported_bins": int(
                sum(
                    reference["counts"][
                        region_mask(BIN_CENTERS, low, high)
                        & (reference["counts"] >= MIN_BIN_EVENTS)
                    ].sum()
                    for low, high in CONTINUUM_WINDOWS.values()
                )
            ),
            "continuum_RMSE_pp": 100.0 * math.sqrt(mean_square),
            "continuum_bias_pp": 100.0 * float(np.mean([item[2] for item in internal])),
            "continuum_centered_RMS_pp": 100.0 * math.sqrt(centered_mean_square),
            "continuum_pull_RMS": math.sqrt(pull_mean_square),
            "continuum_pull_mean": float(np.mean([item[3] for item in internal])),
            "continuum_pull_centered_RMS": math.sqrt(pull_centered_mean_square),
        }
    )
    return window_rows


def roughness_rows(grid_energy: np.ndarray, grid_prediction: np.ndarray) -> list[dict[str, Any]]:
    spacing = float(np.diff(grid_energy).mean())
    if not np.allclose(np.diff(grid_energy), spacing, rtol=0.0, atol=1e-12):
        raise ValueError("Grid is not uniformly spaced")
    rows = []
    for name, (lower, upper) in CONTINUUM_WINDOWS.items():
        mask = region_mask(grid_energy, lower, upper)
        rows.append(
            {
                "window": name,
                "grid_spacing_kev": spacing,
                "mean_absolute_second_difference": float(
                    np.mean(np.abs(np.diff(grid_prediction[mask], n=2)))
                ),
            }
        )
    return rows


def regional_rows(
    energy: np.ndarray,
    outcome: np.ndarray,
    prediction: np.ndarray,
    prediction_std: np.ndarray | None,
) -> list[dict[str, Any]]:
    rows = []
    for name, lower, upper, inclusive_upper in REGIONAL_AGREEMENT:
        mask = region_mask(energy, lower, upper, inclusive_upper=inclusive_upper)
        count = int(mask.sum())
        if count == 0:
            raise ValueError(f"Empty regional agreement region: {name}")
        fraction = float(outcome[mask].mean())
        predicted = float(prediction[mask].mean())
        empirical_se = math.sqrt(max(fraction * (1.0 - fraction), 1e-12) / count)
        z_reference = (fraction - predicted) / max(empirical_se, 1e-12)
        row: dict[str, Any] = {
            "region": name,
            "energy_lower_kev": lower,
            "energy_upper_kev": upper,
            "inclusive_bounds": True,
            "event_count": count,
            "empirical_fraction": fraction,
            "mean_prediction": predicted,
            "empirical_standard_error": empirical_se,
            "empirical_full_width_pp": 200.0 * empirical_se,
            "reference_only_z": z_reference,
            "reference_only_G1": gaussian_agreement(z_reference, 1),
            "reference_only_G2": gaussian_agreement(z_reference, 2),
            "reference_only_G3": gaussian_agreement(z_reference, 3),
        }
        if prediction_std is None:
            row.update(
                {
                    "pointwise_sd_proxy": None,
                    "pointwise_sd_proxy_full_width_pp": None,
                    "combined_proxy_full_width_pp": None,
                    "combined_proxy_z": None,
                    "combined_proxy_G1": None,
                    "combined_proxy_G2": None,
                    "combined_proxy_G3": None,
                    "combined_proxy_status": "unavailable: saved model SD absent",
                }
            )
        else:
            proxy = float(prediction_std[mask].mean())
            combined = math.sqrt(empirical_se**2 + proxy**2)
            z_combined = (fraction - predicted) / max(combined, 1e-12)
            row.update(
                {
                    "pointwise_sd_proxy": proxy,
                    "pointwise_sd_proxy_full_width_pp": 200.0 * proxy,
                    "combined_proxy_full_width_pp": 200.0 * combined,
                    "combined_proxy_z": z_combined,
                    "combined_proxy_G1": gaussian_agreement(z_combined, 1),
                    "combined_proxy_G2": gaussian_agreement(z_combined, 2),
                    "combined_proxy_G3": gaussian_agreement(z_combined, 3),
                    "combined_proxy_status": (
                        "descriptive slide proxy: mean pointwise SD, not region-mean SD"
                    ),
                }
            )
        rows.append(row)
    return rows


def pull_diagnostics(
    reference: Mapping[str, np.ndarray], prediction_mean: np.ndarray
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows = []
    pulls_by_scope = {}
    for scope, (lower, upper) in PULL_SCOPES.items():
        supported = (
            region_mask(BIN_CENTERS, lower, upper)
            & (reference["counts"] >= MIN_BIN_EVENTS)
        )
        pull = (
            reference["fraction"][supported] - prediction_mean[supported]
        ) / reference["wilson_half_width"][supported]
        pulls_by_scope[scope] = pull
        rows.append(
            {
                "scope": scope,
                "supported_bin_count": int(supported.sum()),
                "excluded_bin_count": int(
                    region_mask(BIN_CENTERS, lower, upper).sum() - supported.sum()
                ),
                "target_events_in_supported_bins": int(reference["counts"][supported].sum()),
                "reference_pull_mean": float(pull.mean()),
                "reference_pull_sd": float(pull.std(ddof=1)),
                "reference_pull_RMS": float(np.sqrt(np.mean(pull**2))),
                "reference_C1": float(np.mean(np.abs(pull) <= 1.0)),
                "reference_C2": float(np.mean(np.abs(pull) <= 2.0)),
                "reference_C3": float(np.mean(np.abs(pull) <= 3.0)),
                "mean_reference_full_width_pp": 200.0
                * float(reference["wilson_half_width"][supported].mean()),
                "combined_pull_status": (
                    "unavailable: saved per-pass bin means/covariances are absent"
                ),
            }
        )
    return rows, pulls_by_scope


def archived_bin_discrepancy(
    summary: Mapping[str, Any], reference: Mapping[str, np.ndarray], prediction_mean: np.ndarray
) -> float:
    metric_key = "bin_5kev"
    if metric_key not in summary["metrics"]:
        return 0.0
    archived = {
        ("feature_1620kev" if row["region"] == "Bi-214" else row["region"]): row
        for row in summary["metrics"][metric_key]
    }
    maximum = 0.0
    for region, (lower, upper, inclusive_upper) in EVALUATION_REGIONS.items():
        mask = (
            region_mask(BIN_CENTERS, lower, upper, inclusive_upper=inclusive_upper)
            & (reference["counts"] >= MIN_BIN_EVENTS)
        )
        residual = prediction_mean[mask] - reference["fraction"][mask]
        observed_mae = float(np.mean(np.abs(residual)))
        observed_rmse = float(np.sqrt(np.mean(residual**2)))
        maximum = max(
            maximum,
            abs(observed_mae - float(archived[region]["mae"])),
            abs(observed_rmse - float(archived[region]["rmse"])),
        )
    return maximum


def hierarchical_summary(
    rows: Sequence[Mapping[str, Any]], metric_names: Sequence[str], extra_keys: Sequence[str]
) -> list[dict[str, Any]]:
    grouped: MutableMapping[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    group_names = (
        "subset_id",
        "ordering_id",
        "budget_label",
        "training_budget_nominal_events",
        "training_budget_sampling_eligible_events",
        "method",
        "architecture_id",
        "comparison_role",
    ) + tuple(extra_keys)
    for row in rows:
        grouped[tuple(row.get(name) for name in group_names)].append(row)
    output = []
    for key, group in grouped.items():
        base = dict(zip(group_names, key, strict=True))
        seeds = sorted({row.get("training_seed") for row in group if row.get("training_seed") is not None})
        base["cell_count"] = len(group)
        base["training_seed_count"] = len(seeds)
        base["context_count"] = len({row.get("context_seed") for row in group})
        for source_field in (
            "source_event_predictions_sha256",
            "source_curve_sha256",
        ):
            hashes = sorted(
                {str(row[source_field]) for row in group if row.get(source_field)}
            )
            base[source_field + "_count"] = len(hashes)
            base[source_field + "_set_sha256"] = (
                hashlib.sha256(("\n".join(hashes) + "\n").encode()).hexdigest()
                if hashes
                else None
            )
        for metric in metric_names:
            values = [float(row[metric]) for row in group if row.get(metric) not in (None, "")]
            all_stats = descriptive_stats(values)
            base[metric + "_cell_mean"] = all_stats["mean"]
            base[metric + "_cell_sd"] = all_stats["sd"]
            if seeds:
                seed_means = []
                context_sds = []
                for seed in seeds:
                    seed_values = [
                        float(row[metric])
                        for row in group
                        if row.get("training_seed") == seed and row.get(metric) not in (None, "")
                    ]
                    if seed_values:
                        seed_means.append(float(np.mean(seed_values)))
                        if len(seed_values) > 1:
                            context_sds.append(float(np.std(seed_values, ddof=1)))
                base[metric + "_mean_of_seed_means"] = (
                    float(np.mean(seed_means)) if seed_means else None
                )
                base[metric + "_initialization_sd_across_seed_means"] = (
                    float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else None
                )
                base[metric + "_mean_context_sd_within_seed"] = (
                    float(np.mean(context_sds)) if context_sds else None
                )
            else:
                base[metric + "_mean_of_seed_means"] = None
                base[metric + "_initialization_sd_across_seed_means"] = None
                base[metric + "_mean_context_sd_within_seed"] = all_stats["sd"]
        output.append(base)
    return output


def add_five_k_subset_variation(
    summary_rows: list[dict[str, Any]],
    metric_names: Sequence[str],
    extra_keys: Sequence[str],
) -> None:
    """Append rows that isolate variation across the three 5k orderings."""
    candidates = [
        row
        for row in summary_rows
        if row["training_budget_nominal_events"] == 5000
        and row["comparison_role"] == "matched_neural"
        and row["subset_id"]
        in ("original_n5000", "seed20260910_n5000", "seed20260911_n5000")
    ]
    grouped: MutableMapping[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    names = ("method", "architecture_id") + tuple(extra_keys)
    for row in candidates:
        grouped[tuple(row.get(name) for name in names)].append(row)
    additions = []
    for key, group in grouped.items():
        if len(group) != 3:
            raise ValueError("Expected three 5k subset summaries")
        row: dict[str, Any] = {
            "subset_id": "all_three_5k_subsets",
            "ordering_id": "original+seed20260910+seed20260911",
            "budget_label": "5k",
            "training_budget_nominal_events": 5000,
            "training_budget_sampling_eligible_events": None,
            "comparison_role": "between_training_subset_summary",
            "cell_count": int(sum(int(item["cell_count"]) for item in group)),
            "training_seed_count": 3,
            "context_count": 10,
            **dict(zip(names, key, strict=True)),
        }
        for source_field in (
            "source_event_predictions_sha256",
            "source_curve_sha256",
        ):
            hashes = sorted(
                str(item[source_field + "_set_sha256"])
                for item in group
                if item.get(source_field + "_set_sha256")
            )
            row[source_field + "_count"] = int(
                sum(int(item[source_field + "_count"]) for item in group)
            )
            row[source_field + "_set_sha256"] = hashlib.sha256(
                ("\n".join(hashes) + "\n").encode()
            ).hexdigest()
        for metric in metric_names:
            subset_means = np.asarray(
                [item[metric + "_mean_of_seed_means"] for item in group],
                dtype=np.float64,
            )
            row[metric + "_cell_mean"] = float(subset_means.mean())
            row[metric + "_cell_sd"] = None
            row[metric + "_mean_of_seed_means"] = float(subset_means.mean())
            row[metric + "_initialization_sd_across_seed_means"] = float(
                np.mean(
                    [
                        item[metric + "_initialization_sd_across_seed_means"]
                        for item in group
                    ]
                )
            )
            row[metric + "_mean_context_sd_within_seed"] = float(
                np.mean([item[metric + "_mean_context_sd_within_seed"] for item in group])
            )
            row[metric + "_training_subset_sd_across_subset_means"] = float(
                subset_means.std(ddof=1)
            )
        additions.append(row)
    summary_rows.extend(additions)


def add_source_fields(row: dict[str, Any], meta: Mapping[str, Any]) -> dict[str, Any]:
    row.update(grouping_fields(meta))
    row.update(
        {
            "training_seed": meta.get("training_seed"),
            "context_seed": meta["context_seed"],
            "context_size": 500,
            "run_id": meta["run_id"],
            "source_event_predictions_sha256": meta["event_sha256"],
            "source_curve_sha256": meta["curve_sha256"],
            "headline_original_5k": bool(meta["headline_original_5k"]),
        }
    )
    return row


def portable_series_name(row: Mapping[str, Any], prefix: str) -> str:
    """Return a stable wide-CSV column name for one saved-curve group."""
    return "{}__{}__{}".format(prefix, row["subset_id"], row["architecture_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("ml4phy-paper"),
        help="Root containing tables/, figures/, reports/, and manifests/ outputs.",
    )
    args = parser.parse_args()
    start = time.perf_counter()
    repo = args.repo.resolve()
    paper_root = repo / "ml4phy-paper"
    output_root = args.output_root if args.output_root.is_absolute() else repo / args.output_root

    source_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    phase2_manifest_path = paper_root / "manifests/phase2_result.json"
    phase3_manifest_path = paper_root / "manifests/phase3_result.json"
    kernel_manifest_path = paper_root / "manifests/phase1_kernel_result.json"
    frozen_manifest_path = paper_root / "manifests/frozen_protocol_v1.json"
    extension_manifest_path = paper_root / "manifests/extension_protocol_v1.json"
    mc_manifest_path = paper_root / "manifests/mc_smoothness_result.json"
    phase2_manifest = read_json(phase2_manifest_path)
    phase3_manifest = read_json(phase3_manifest_path)
    kernel_manifest = read_json(kernel_manifest_path)
    frozen_manifest = read_json(frozen_manifest_path)
    extension_manifest = read_json(extension_manifest_path)
    mc_manifest = read_json(mc_manifest_path)
    phase2_expected = {item["run_id"]: item for item in phase2_manifest["inputs"]}
    phase3_expected = {item["run_id"]: item for item in phase3_manifest["neural_inputs"]}
    gp_expected = {item["run_id"]: item for item in phase3_manifest["dense_gp_inputs"]}

    role_path = paper_root / "local/protocol/frozen_roles_v1.npz"
    extension_role_path = paper_root / "local/protocol/extension_context_roles_v1.npz"
    if sha256_file(role_path) != frozen_manifest["local_role_archive"]["sha256"]:
        raise ValueError("Frozen role archive hash mismatch")
    if sha256_file(extension_role_path) != extension_manifest["context_protocol"]["local_archive"]["sha256"]:
        raise ValueError("Extension context archive hash mismatch")
    h5_path = repo / frozen_manifest["inputs"]["full_test"]["logical_path"]
    if sha256_file(h5_path) != frozen_manifest["inputs"]["full_test"]["sha256"]:
        raise ValueError("Frozen full-test HDF5 hash mismatch")
    with np.load(role_path) as roles:
        frozen_target_rows = roles["final_target_rows"].astype(np.int64)
        threshold = float(roles["fixed_threshold"])
    if threshold != float(frozen_manifest["threshold"]["value"]):
        raise ValueError("Threshold mismatch")
    with h5py.File(h5_path, "r") as handle:
        target_energy = handle["energy"][frozen_target_rows].astype(np.float64)
        target_outcome = (
            handle["score"][frozen_target_rows].astype(np.float64) >= threshold
        ).astype(np.int8)
    reference = bin_events(target_energy, target_outcome)
    if int(reference["counts"].sum()) != 114400:
        raise ValueError("Unexpected frozen target count")

    context_bin_rows: list[dict[str, Any]] = []
    for index in range(BIN_CENTERS.size):
        context_bin_rows.append(
            {
                "sample": "frozen_final_target",
                "context_seed": None,
                "context_size": None,
                "bin_index": index,
                "energy_lower_kev": BIN_EDGES[index],
                "energy_upper_kev": BIN_EDGES[index + 1],
                "energy_center_kev": BIN_CENTERS[index],
                "event_count": int(reference["counts"][index]),
                "pass_count": int(reference["passes"][index]),
                "empirical_fraction": reference["fraction"][index],
                "wilson_z": 1.0,
                "wilson_lower": reference["wilson_lower"][index],
                "wilson_upper": reference["wilson_upper"][index],
                "supported_min_4_events": bool(reference["counts"][index] >= MIN_BIN_EVENTS),
            }
        )
    context_rows_by_seed: dict[int, np.ndarray] = {}
    with np.load(extension_role_path) as extension_roles, h5py.File(h5_path, "r") as handle:
        for context_seed in CONTEXT_SEEDS:
            rows = extension_roles[
                f"final_context_s{context_seed}_n500_rows"
            ].astype(np.int64)
            context_rows_by_seed[context_seed] = rows
            # h5py requires monotonic fancy indices; event order is irrelevant
            # for these context-bin counts, while the unsorted frozen rows are
            # retained separately for exact membership checks.
            sorted_rows = np.sort(rows)
            energy = handle["energy"][sorted_rows].astype(np.float64)
            outcome = (
                handle["score"][sorted_rows].astype(np.float64) >= threshold
            ).astype(np.int8)
            binned = bin_events(energy, outcome)
            for index in range(BIN_CENTERS.size):
                context_bin_rows.append(
                    {
                        "sample": "context",
                        "context_seed": context_seed,
                        "context_size": 500,
                        "bin_index": index,
                        "energy_lower_kev": BIN_EDGES[index],
                        "energy_upper_kev": BIN_EDGES[index + 1],
                        "energy_center_kev": BIN_CENTERS[index],
                        "event_count": int(binned["counts"][index]),
                        "pass_count": int(binned["passes"][index]),
                        "empirical_fraction": binned["fraction"][index],
                        "wilson_z": 1.0,
                        "wilson_lower": binned["wilson_lower"][index],
                        "wilson_upper": binned["wilson_upper"][index],
                        "supported_min_4_events": bool(
                            binned["counts"][index] >= MIN_BIN_EVENTS
                        ),
                    }
                )

    regional: list[dict[str, Any]] = []
    continuum: list[dict[str, Any]] = []
    roughness: list[dict[str, Any]] = []
    pull_cells: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    illustrative_bins: list[dict[str, Any]] = []
    illustrative_grids: list[dict[str, Any]] = []
    group_curve_sum: dict[tuple[Any, ...], np.ndarray] = {}
    group_bin_sum: dict[tuple[Any, ...], np.ndarray] = {}
    group_count: MutableMapping[tuple[Any, ...], int] = defaultdict(int)
    group_meta: dict[tuple[Any, ...], dict[str, Any]] = {}
    hist_values: MutableMapping[tuple[tuple[Any, ...], str], list[np.ndarray]] = defaultdict(list)
    grid_reference: np.ndarray | None = None
    target_energy_reference: np.ndarray | None = None
    target_outcome_reference: np.ndarray | None = None
    maximum_bin_reconciliation_difference = 0.0
    processed_counts: MutableMapping[str, int] = defaultdict(int)

    def process_cell(
        meta: dict[str, Any],
        event_path: Path,
        curve_path: Path,
        summary: Mapping[str, Any] | None,
        prediction_std_available: bool,
    ) -> None:
        nonlocal grid_reference, target_energy_reference, target_outcome_reference
        nonlocal maximum_bin_reconciliation_difference
        with np.load(event_path) as events:
            rows = events["target_rows"].astype(np.int64)
            energy = events["target_energy_kev"].astype(np.float64)
            outcome = events["outcome"].astype(np.int8)
            prediction = events["prediction"].astype(np.float64)
            prediction_std = (
                events["prediction_std"].astype(np.float64)
                if prediction_std_available
                else None
            )
            saved_context_rows = events["context_rows"].astype(np.int64)
        with np.load(curve_path) as curve:
            grid_energy = curve["energy_kev"].astype(np.float64)
            grid_prediction = curve["prediction"].astype(np.float64)
            archived_centers = curve["bin_centers_kev"].astype(np.float64)
            archived_counts = curve["bin_counts"].astype(np.int64)
            archived_fraction = curve["empirical_rate"].astype(np.float64)
            archived_prediction = curve["bin_prediction_mean"].astype(np.float64)
        if not np.array_equal(rows, frozen_target_rows):
            raise ValueError("Frozen target rows mismatch: {}".format(meta["run_id"]))
        if not np.array_equal(
            saved_context_rows, np.sort(context_rows_by_seed[meta["context_seed"]])
        ):
            raise ValueError("Context rows mismatch: {}".format(meta["run_id"]))
        if target_energy_reference is None:
            target_energy_reference = energy
            target_outcome_reference = outcome
        elif not np.array_equal(energy, target_energy_reference) or not np.array_equal(
            outcome, target_outcome_reference
        ):
            raise ValueError("Target values mismatch: {}".format(meta["run_id"]))
        if not np.array_equal(energy, target_energy) or not np.array_equal(outcome, target_outcome):
            raise ValueError("Saved target values disagree with source HDF5")
        if not np.array_equal(archived_centers, BIN_CENTERS):
            raise ValueError("Archived bin centers mismatch")
        if not np.array_equal(archived_counts, reference["counts"]):
            raise ValueError("Archived bin counts mismatch")
        if not np.allclose(archived_fraction, reference["fraction"], equal_nan=True, atol=0, rtol=0):
            raise ValueError("Archived empirical fractions mismatch")
        binned = bin_events(energy, outcome, prediction)
        if not np.allclose(
            archived_prediction,
            binned["prediction_mean"],
            equal_nan=True,
            atol=2e-8 if meta["estimator_type"] == "kernel" else 2e-14,
            rtol=0,
        ):
            raise ValueError("Archived bin prediction mismatch: {}".format(meta["run_id"]))
        if grid_reference is None:
            grid_reference = grid_energy
        elif not np.array_equal(grid_reference, grid_energy):
            raise ValueError("Saved grid mismatch: {}".format(meta["run_id"]))
        if summary is not None:
            maximum_bin_reconciliation_difference = max(
                maximum_bin_reconciliation_difference,
                archived_bin_discrepancy(summary, reference, binned["prediction_mean"]),
            )

        for row in regional_rows(energy, outcome, prediction, prediction_std):
            regional.append(add_source_fields(row, meta))
        for row in continuum_diagnostics(reference, binned["prediction_mean"]):
            continuum.append(add_source_fields(row, meta))
        for row in roughness_rows(grid_energy, grid_prediction):
            row["curve_kind"] = "individual_cell"
            roughness.append(add_source_fields(row, meta))
        cell_pull_rows, pulls = pull_diagnostics(reference, binned["prediction_mean"])
        for row in cell_pull_rows:
            pull_cells.append(add_source_fields(row, meta))
        key = group_key(meta)
        group_meta[key] = grouping_fields(meta)
        group_curve_sum[key] = group_curve_sum.get(key, np.zeros_like(grid_prediction)) + grid_prediction
        group_bin_sum[key] = group_bin_sum.get(
            key, np.zeros_like(binned["prediction_mean"])
        ) + binned["prediction_mean"]
        group_count[key] += 1
        for scope, values in pulls.items():
            hist_values[(key, scope)].append(values)

        if meta["illustrative"]:
            for index in range(BIN_CENTERS.size):
                illustrative_bins.append(
                    {
                        **grouping_fields(meta),
                        "curve_kind": "illustrative_cell",
                        "training_seed": meta.get("training_seed"),
                        "context_seed": meta["context_seed"],
                        "bin_index": index,
                        "energy_lower_kev": BIN_EDGES[index],
                        "energy_upper_kev": BIN_EDGES[index + 1],
                        "energy_center_kev": BIN_CENTERS[index],
                        "target_event_count": int(reference["counts"][index]),
                        "target_pass_count": int(reference["passes"][index]),
                        "target_fraction": reference["fraction"][index],
                        "wilson_lower_z1": reference["wilson_lower"][index],
                        "wilson_upper_z1": reference["wilson_upper"][index],
                        "supported_min_4_events": bool(
                            reference["counts"][index] >= MIN_BIN_EVENTS
                        ),
                        "prediction_mean_at_actual_target_energies": binned[
                            "prediction_mean"
                        ][index],
                    }
                )
            for energy_value, prediction_value in zip(
                grid_energy, grid_prediction, strict=True
            ):
                illustrative_grids.append(
                    {
                        **grouping_fields(meta),
                        "curve_kind": "illustrative_cell",
                        "training_seed": meta.get("training_seed"),
                        "context_seed": meta["context_seed"],
                        "energy_kev": energy_value,
                        "prediction": prediction_value,
                        "grid_spacing_kev": float(np.diff(grid_energy).mean()),
                    }
                )
        source_records.append(
            {
                **grouping_fields(meta),
                "training_seed": meta.get("training_seed"),
                "context_seed": meta["context_seed"],
                "run_id": meta["run_id"],
                "event_predictions_path": str(event_path.relative_to(repo)),
                "event_predictions_sha256": meta["event_sha256"],
                "curve_path": str(curve_path.relative_to(repo)),
                "curve_sha256": meta["curve_sha256"],
                "checkpoint_sha256": meta.get("checkpoint_sha256"),
            }
        )
        processed_counts[meta["estimator_type"]] += 1

    for subset_id, ordering_id, budget, budget_label, eligible, source in SUBSET_SPECS:
        for architecture in ARCHITECTURES:
            for training_seed in TRAINING_SEEDS:
                for context_seed in CONTEXT_SEEDS:
                    directory = neural_run_dir(
                        paper_root,
                        source,
                        subset_id,
                        budget,
                        architecture,
                        training_seed,
                        context_seed,
                    )
                    summary_path = directory / "summary.json"
                    event_path = directory / "event_predictions.npz"
                    curve_path = directory / "curve.npz"
                    if not summary_path.is_file() or not event_path.is_file() or not curve_path.is_file():
                        raise FileNotFoundError(f"Missing neural cell artifact: {directory}")
                    summary = read_json(summary_path)
                    expected = (
                        phase2_expected[summary["run_id"]]
                        if source == "phase2"
                        else phase3_expected[summary["run_id"]]
                    )
                    if sha256_file(summary_path) != expected["summary_sha256"]:
                        raise ValueError(f"Neural summary hash mismatch: {summary_path}")
                    event_hash = sha256_file(event_path)
                    curve_hash = sha256_file(curve_path)
                    if source == "phase2":
                        expected_event = expected["event_predictions_sha256"]
                        expected_curve = expected["curve_sha256"]
                    else:
                        expected_event = expected["server_only_output_hashes"]["event_predictions"]
                        expected_curve = expected["server_only_output_hashes"]["curve"]
                    if event_hash != expected_event or curve_hash != expected_curve:
                        raise ValueError(f"Neural prediction hash mismatch: {directory}")
                    meta = {
                        "subset_id": subset_id,
                        "ordering_id": ordering_id,
                        "budget_label": budget_label,
                        "training_budget_nominal_events": budget,
                        "training_budget_sampling_eligible_events": eligible,
                        "method": METHOD_NAMES[architecture],
                        "architecture_id": architecture,
                        "comparison_role": "matched_neural",
                        "training_seed": training_seed,
                        "context_seed": context_seed,
                        "run_id": summary["run_id"],
                        "event_sha256": event_hash,
                        "curve_sha256": curve_hash,
                        "checkpoint_sha256": summary["model"]["checkpoint_sha256"],
                        "headline_original_5k": subset_id == "original_n5000",
                        "illustrative": (
                            subset_id == "original_n5000"
                            and training_seed == 0
                            and context_seed == 100
                        ),
                        "estimator_type": "neural",
                    }
                    process_cell(meta, event_path, curve_path, summary, True)

    kernel_path = paper_root / "runs/kernel/20260909-phase1-kernel-v1-attempt2/predictions.npz"
    if sha256_file(kernel_path) != kernel_manifest["server_only_predictions"]["sha256"]:
        raise ValueError("Kernel archive hash mismatch")
    kernel_bins_path = paper_root / "tables/phase1_kernel_bin_error_cells.csv"
    with kernel_bins_path.open(newline="") as stream:
        kernel_archived_bins = list(csv.DictReader(stream))
    with np.load(kernel_path) as kernel:
        variants = [str(value) for value in kernel["variants"]]
        sizes = kernel["context_sizes"].astype(int).tolist()
        seeds = kernel["context_seeds"].astype(int).tolist()
        size_index = sizes.index(500)
        if not np.array_equal(kernel["final_target_rows"].astype(np.int64), frozen_target_rows):
            raise ValueError("Kernel target rows mismatch")
        for variant in ("context_only", "pooled_data"):
            variant_index = variants.index(variant)
            for context_seed in CONTEXT_SEEDS:
                context_index = seeds.index(context_seed)
                prediction = kernel["predictions"][variant_index, size_index, context_index].astype(
                    np.float64
                )
                grid_prediction = kernel["curves"][variant_index, size_index, context_index].astype(
                    np.float64
                )
                run_id = f"kernel-{variant}-ctx-s{context_seed}-n500"
                # Use a temporary in-memory-compatible NPZ view through direct processing below.
                binned = bin_events(target_energy, target_outcome, prediction)
                archived_rows = {
                    ("feature_1620kev" if row["region"] == "Bi-214" else row["region"]): row
                    for row in kernel_archived_bins
                    if row["variant"] == variant
                    and int(row["context_size"]) == 500
                    and int(row["context_seed"]) == context_seed
                }
                for region, (lower, upper, inclusive_upper) in EVALUATION_REGIONS.items():
                    mask = (
                        region_mask(
                            BIN_CENTERS, lower, upper, inclusive_upper=inclusive_upper
                        )
                        & (reference["counts"] >= MIN_BIN_EVENTS)
                    )
                    residual = (
                        binned["prediction_mean"][mask] - reference["fraction"][mask]
                    )
                    observed_mae_pp = 100.0 * float(np.mean(np.abs(residual)))
                    observed_rmse_pp = 100.0 * float(np.sqrt(np.mean(residual**2)))
                    maximum_bin_reconciliation_difference = max(
                        maximum_bin_reconciliation_difference,
                        abs(observed_mae_pp - float(archived_rows[region]["mae_percentage_points"]))
                        / 100.0,
                        abs(observed_rmse_pp - float(archived_rows[region]["rmse_percentage_points"]))
                        / 100.0,
                    )
                method = "Kernel" if variant == "context_only" else "Pooled-data kernel"
                meta = {
                    "subset_id": "context_only" if variant == "context_only" else "full_n18866_plus_context",
                    "ordering_id": "not_applicable",
                    "budget_label": "context-only" if variant == "context_only" else "18.9k+500",
                    "training_budget_nominal_events": 0 if variant == "context_only" else 18866,
                    "training_budget_sampling_eligible_events": (
                        0 if variant == "context_only" else 18866
                    ),
                    "method": method,
                    "architecture_id": f"kernel_{variant}",
                    "comparison_role": (
                        "context_only_classical"
                        if variant == "context_only"
                        else "stronger_data_control"
                    ),
                    "training_seed": None,
                    "context_seed": context_seed,
                    "run_id": run_id,
                    "event_sha256": kernel_manifest["server_only_predictions"]["sha256"],
                    "curve_sha256": kernel_manifest["server_only_predictions"]["sha256"],
                    "checkpoint_sha256": None,
                    "headline_original_5k": variant == "context_only",
                    "illustrative": context_seed == 100,
                    "estimator_type": "kernel",
                }
                if not np.array_equal(kernel["final_energy_kev"].astype(np.float64), target_energy):
                    raise ValueError("Kernel target energy mismatch")
                if not np.array_equal(kernel["final_outcome"].astype(np.int8), target_outcome):
                    raise ValueError("Kernel target outcome mismatch")
                # Inline the generic calculations because the kernel archive contains all cells.
                for row in regional_rows(target_energy, target_outcome, prediction, None):
                    regional.append(add_source_fields(row, meta))
                for row in continuum_diagnostics(reference, binned["prediction_mean"]):
                    continuum.append(add_source_fields(row, meta))
                for row in roughness_rows(kernel["grid_energy_kev"], grid_prediction):
                    row["curve_kind"] = "individual_cell"
                    roughness.append(add_source_fields(row, meta))
                cell_pull_rows, pulls = pull_diagnostics(reference, binned["prediction_mean"])
                for row in cell_pull_rows:
                    pull_cells.append(add_source_fields(row, meta))
                key = group_key(meta)
                group_meta[key] = grouping_fields(meta)
                group_curve_sum[key] = group_curve_sum.get(
                    key, np.zeros_like(grid_prediction)
                ) + grid_prediction
                group_bin_sum[key] = group_bin_sum.get(
                    key, np.zeros_like(binned["prediction_mean"])
                ) + binned["prediction_mean"]
                group_count[key] += 1
                for scope, values in pulls.items():
                    hist_values[(key, scope)].append(values)
                if meta["illustrative"]:
                    for index in range(BIN_CENTERS.size):
                        illustrative_bins.append(
                            {
                                **grouping_fields(meta),
                                "curve_kind": "illustrative_cell",
                                "training_seed": None,
                                "context_seed": context_seed,
                                "bin_index": index,
                                "energy_lower_kev": BIN_EDGES[index],
                                "energy_upper_kev": BIN_EDGES[index + 1],
                                "energy_center_kev": BIN_CENTERS[index],
                                "target_event_count": int(reference["counts"][index]),
                                "target_pass_count": int(reference["passes"][index]),
                                "target_fraction": reference["fraction"][index],
                                "wilson_lower_z1": reference["wilson_lower"][index],
                                "wilson_upper_z1": reference["wilson_upper"][index],
                                "supported_min_4_events": bool(
                                    reference["counts"][index] >= MIN_BIN_EVENTS
                                ),
                                "prediction_mean_at_actual_target_energies": binned[
                                    "prediction_mean"
                                ][index],
                            }
                        )
                    for energy_value, prediction_value in zip(
                        kernel["grid_energy_kev"], grid_prediction, strict=True
                    ):
                        illustrative_grids.append(
                            {
                                **grouping_fields(meta),
                                "curve_kind": "illustrative_cell",
                                "training_seed": None,
                                "context_seed": context_seed,
                                "energy_kev": float(energy_value),
                                "prediction": float(prediction_value),
                                "grid_spacing_kev": 1.0,
                            }
                        )
                source_records.append(
                    {
                        **grouping_fields(meta),
                        "training_seed": None,
                        "context_seed": context_seed,
                        "run_id": run_id,
                        "event_predictions_path": str(kernel_path.relative_to(repo)),
                        "event_predictions_sha256": meta["event_sha256"],
                        "curve_path": str(kernel_path.relative_to(repo)),
                        "curve_sha256": meta["curve_sha256"],
                        "checkpoint_sha256": None,
                    }
                )
                processed_counts["kernel"] += 1

    gp_campaign = read_json(
        paper_root / "runs/campaigns/20260909-phase3-dense-gp-v1/campaign_record.json"
    )
    selected_gp = []
    for item in gp_campaign["completed"]:
        directory = paper_root / "runs/gp/phase3" / item["run_id"]
        summary = read_json(directory / "summary.json")
        if (
            summary["phase"] == "final"
            and summary["estimator"]["family_id"] == "matern32"
            and summary["context"]["size"] == 500
        ):
            selected_gp.append((directory, summary, item))
    if len(selected_gp) != 10:
        raise ValueError("Expected ten selected final Matérn GP cells")
    for directory, summary, _campaign_item in selected_gp:
        summary_path = directory / "summary.json"
        event_path = directory / "predictions.npz"
        curve_path = directory / "curve.npz"
        expected = gp_expected[summary["run_id"]]
        if sha256_file(summary_path) != expected["summary_sha256"]:
            raise ValueError("GP summary hash mismatch")
        event_hash = sha256_file(event_path)
        curve_hash = sha256_file(curve_path)
        if (
            event_hash != expected["server_only_output_hashes"]["predictions"]
            or curve_hash != expected["server_only_output_hashes"]["curve"]
        ):
            raise ValueError("GP saved output hash mismatch")
        context_seed = int(summary["context"]["seed"])
        meta = {
            "subset_id": "context_only",
            "ordering_id": "not_applicable",
            "budget_label": "context-only",
            "training_budget_nominal_events": 0,
            "training_budget_sampling_eligible_events": 0,
            "method": "Bernoulli GP",
            "architecture_id": "bernoulli_gp_matern32",
            "comparison_role": "context_only_classical",
            "training_seed": None,
            "context_seed": context_seed,
            "run_id": summary["run_id"],
            "event_sha256": event_hash,
            "curve_sha256": curve_hash,
            "checkpoint_sha256": None,
            "headline_original_5k": True,
            "illustrative": context_seed == 100,
            "estimator_type": "gp",
        }
        process_cell(meta, event_path, curve_path, summary, False)

    if processed_counts != {"neural": 600, "kernel": 20, "gp": 10}:
        raise ValueError(f"Unexpected recovered matrix: {dict(processed_counts)}")

    mean_bin_rows: list[dict[str, Any]] = []
    mean_grid_rows: list[dict[str, Any]] = []
    for key, count in group_count.items():
        meta = group_meta[key]
        mean_bin = group_bin_sum[key] / count
        mean_curve = group_curve_sum[key] / count
        for row in roughness_rows(grid_reference, mean_curve):
            row.update(meta)
            row.update(
                {
                    "curve_kind": "mean_curve_across_saved_cells",
                    "training_seed": None,
                    "context_seed": None,
                    "context_size": 500,
                    "run_id": None,
                    "source_event_predictions_sha256": None,
                    "source_curve_sha256": None,
                    "headline_original_5k": meta["subset_id"] in (
                        "original_n5000",
                        "context_only",
                    ),
                    "averaged_cell_count": count,
                }
            )
            roughness.append(row)
        for index in range(BIN_CENTERS.size):
            value = mean_bin[index]
            supported = bool(reference["counts"][index] >= MIN_BIN_EVENTS)
            half_width = reference["wilson_half_width"][index]
            mean_bin_rows.append(
                {
                    **meta,
                    "curve_kind": "mean_curve_across_saved_cells",
                    "averaged_cell_count": count,
                    "bin_index": index,
                    "energy_lower_kev": BIN_EDGES[index],
                    "energy_upper_kev": BIN_EDGES[index + 1],
                    "energy_center_kev": BIN_CENTERS[index],
                    "target_event_count": int(reference["counts"][index]),
                    "target_pass_count": int(reference["passes"][index]),
                    "target_fraction": reference["fraction"][index],
                    "wilson_lower_z1": reference["wilson_lower"][index],
                    "wilson_upper_z1": reference["wilson_upper"][index],
                    "supported_min_4_events": supported,
                    "prediction_mean_at_actual_target_energies": value,
                    "reference_pull": (
                        (reference["fraction"][index] - value) / half_width
                        if supported
                        else None
                    ),
                }
            )
        if meta["training_budget_nominal_events"] == 5000 or meta[
            "comparison_role"
        ] in ("context_only_classical", "stronger_data_control"):
            for energy_value, prediction_value in zip(
                grid_reference, mean_curve, strict=True
            ):
                mean_grid_rows.append(
                    {
                        **meta,
                        "curve_kind": "mean_curve_across_saved_cells",
                        "averaged_cell_count": count,
                        "energy_kev": energy_value,
                        "prediction": prediction_value,
                        "grid_spacing_kev": float(np.diff(grid_reference).mean()),
                    }
                )

    histogram_rows: list[dict[str, Any]] = []
    for (key, scope), arrays in hist_values.items():
        meta = group_meta[key]
        values = np.concatenate(arrays)
        underflow = int(np.sum(values < HISTOGRAM_EDGES[0]))
        overflow = int(np.sum(values > HISTOGRAM_EDGES[-1]))
        clipped_values = values[(values >= HISTOGRAM_EDGES[0]) & (values <= HISTOGRAM_EDGES[-1])]
        counts, _ = np.histogram(clipped_values, bins=HISTOGRAM_EDGES)
        histogram_rows.append(
            {
                **meta,
                "scope": scope,
                "histogram_bin": "underflow",
                "pull_lower": None,
                "pull_upper": HISTOGRAM_EDGES[0],
                "count": underflow,
                "total_cell_bin_observations": int(values.size),
                "cell_count": len(arrays),
            }
        )
        for index, count in enumerate(counts):
            histogram_rows.append(
                {
                    **meta,
                    "scope": scope,
                    "histogram_bin": index,
                    "pull_lower": HISTOGRAM_EDGES[index],
                    "pull_upper": HISTOGRAM_EDGES[index + 1],
                    "count": int(count),
                    "total_cell_bin_observations": int(values.size),
                    "cell_count": len(arrays),
                }
            )
        histogram_rows.append(
            {
                **meta,
                "scope": scope,
                "histogram_bin": "overflow",
                "pull_lower": HISTOGRAM_EDGES[-1],
                "pull_upper": None,
                "count": overflow,
                "total_cell_bin_observations": int(values.size),
                "cell_count": len(arrays),
            }
        )

    regional_summary = hierarchical_summary(
        regional,
        (
            "empirical_fraction",
            "mean_prediction",
            "empirical_standard_error",
            "reference_only_z",
            "reference_only_G1",
            "reference_only_G2",
            "reference_only_G3",
            "pointwise_sd_proxy",
            "combined_proxy_full_width_pp",
            "combined_proxy_z",
            "combined_proxy_G1",
            "combined_proxy_G2",
            "combined_proxy_G3",
        ),
        ("region", "event_count"),
    )
    add_five_k_subset_variation(
        regional_summary,
        (
            "empirical_fraction",
            "mean_prediction",
            "empirical_standard_error",
            "reference_only_z",
            "reference_only_G1",
            "reference_only_G2",
            "reference_only_G3",
            "pointwise_sd_proxy",
            "combined_proxy_full_width_pp",
            "combined_proxy_z",
            "combined_proxy_G1",
            "combined_proxy_G2",
            "combined_proxy_G3",
        ),
        ("region", "event_count"),
    )
    continuum_summary = hierarchical_summary(
        continuum,
        (
            "continuum_RMSE_pp",
            "continuum_bias_pp",
            "continuum_centered_RMS_pp",
            "continuum_pull_RMS",
            "continuum_pull_mean",
            "continuum_pull_centered_RMS",
        ),
        ("window", "window_weighting", "supported_bin_count"),
    )
    add_five_k_subset_variation(
        continuum_summary,
        (
            "continuum_RMSE_pp",
            "continuum_bias_pp",
            "continuum_centered_RMS_pp",
            "continuum_pull_RMS",
            "continuum_pull_mean",
            "continuum_pull_centered_RMS",
        ),
        ("window", "window_weighting", "supported_bin_count"),
    )
    individual_roughness = [row for row in roughness if row["curve_kind"] == "individual_cell"]
    roughness_summary = hierarchical_summary(
        individual_roughness,
        ("mean_absolute_second_difference",),
        ("window", "grid_spacing_kev"),
    )
    add_five_k_subset_variation(
        roughness_summary,
        ("mean_absolute_second_difference",),
        ("window", "grid_spacing_kev"),
    )

    table_paths = {
        "regional_cells": output_root / "tables/paper_regional_gaussian_agreement.csv",
        "regional_summary": output_root
        / "tables/paper_regional_gaussian_agreement_summary.csv",
        "continuum_cells": output_root / "tables/paper_continuum_diagnostics.csv",
        "continuum_summary": output_root / "tables/paper_continuum_diagnostics_summary.csv",
        "roughness_cells": output_root / "tables/paper_curve_roughness.csv",
        "roughness_summary": output_root / "tables/paper_curve_roughness_summary.csv",
        "pull_cells": output_root / "tables/paper_reference_pull_cells.csv",
        "pull_histograms": output_root / "tables/paper_reference_pull_histograms.csv",
        "bin_predictions": output_root / "tables/paper_reference_bin_predictions.csv",
        "reference_context": output_root / "tables/paper_reference_context_bins.csv",
        "illustrative_curves": output_root / "tables/paper_illustrative_grid_curves.csv",
        "mean_curves": output_root / "tables/paper_mean_grid_curves.csv",
    }
    write_csv(table_paths["regional_cells"], regional)
    write_csv(table_paths["regional_summary"], regional_summary)
    write_csv(table_paths["continuum_cells"], continuum)
    write_csv(table_paths["continuum_summary"], continuum_summary)
    write_csv(table_paths["roughness_cells"], roughness)
    write_csv(table_paths["roughness_summary"], roughness_summary)
    write_csv(table_paths["pull_cells"], pull_cells)
    write_csv(table_paths["pull_histograms"], histogram_rows)
    # Curve-shaped exports use a wide schema so verbose provenance is not
    # repeated for every energy. Cell-level files above retain source hashes.
    bin_series: MutableMapping[str, np.ndarray] = {}
    for prefix, source_rows in (
        ("illustrative", illustrative_bins),
        (
            "mean",
            [
                row
                for row in mean_bin_rows
                if row["training_budget_nominal_events"] == 5000
                or row["comparison_role"]
                in ("context_only_classical", "stronger_data_control")
            ],
        ),
    ):
        for row in source_rows:
            name = portable_series_name(row, prefix)
            if name not in bin_series:
                bin_series[name] = np.full(BIN_CENTERS.size, np.nan)
            bin_series[name][int(row["bin_index"])] = row[
                "prediction_mean_at_actual_target_energies"
            ]
    wide_bin_rows: list[dict[str, Any]] = []
    for index in range(BIN_CENTERS.size):
        row: dict[str, Any] = {
            "bin_index": index,
            "energy_lower_kev": BIN_EDGES[index],
            "energy_upper_kev": BIN_EDGES[index + 1],
            "energy_center_kev": BIN_CENTERS[index],
            "target_event_count": int(reference["counts"][index]),
            "target_pass_count": int(reference["passes"][index]),
            "target_fraction": reference["fraction"][index],
            "wilson_lower_z1": reference["wilson_lower"][index],
            "wilson_upper_z1": reference["wilson_upper"][index],
            "supported_min_4_events": bool(reference["counts"][index] >= MIN_BIN_EVENTS),
        }
        row.update({name: values[index] for name, values in bin_series.items()})
        wide_bin_rows.append(row)
    write_csv(table_paths["bin_predictions"], wide_bin_rows)
    write_csv(table_paths["reference_context"], context_bin_rows)
    for prefix, source_rows, destination in (
        ("illustrative", illustrative_grids, table_paths["illustrative_curves"]),
        ("mean", mean_grid_rows, table_paths["mean_curves"]),
    ):
        series: MutableMapping[str, list[float]] = defaultdict(list)
        for row in source_rows:
            series[portable_series_name(row, prefix)].append(float(row["prediction"]))
        if any(len(values) != len(grid_reference) for values in series.values()):
            raise ValueError("Incomplete portable grid series")
        wide_grid_rows = []
        for index, energy_value in enumerate(grid_reference):
            row = {"energy_kev": energy_value, "grid_spacing_kev": 1.0}
            row.update({name: values[index] for name, values in series.items()})
            wide_grid_rows.append(row)
        write_csv(destination, wide_grid_rows)

    figure1_path = output_root / "figures/paper_figure1_efficiency_curves.png"
    figure2_path = output_root / "figures/paper_figure2_reference_pulls.png"
    for path in (figure1_path, figure2_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    illustrative_by_method: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method in METHOD_COLORS:
        rows = [row for row in illustrative_grids if row["method"] == method]
        if rows:
            illustrative_by_method[method] = (
                np.asarray([row["energy_kev"] for row in rows]),
                np.asarray([row["prediction"] for row in rows]),
            )
    context100 = [
        row
        for row in context_bin_rows
        if row["sample"] == "context" and row["context_seed"] == 100
    ]
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.4), constrained_layout=True)
    for axis, lower, upper, title in (
        (axes[0], 500.0, 3000.0, "Full frozen range"),
        (axes[1], 1500.0, 3000.0, "Presentation view"),
    ):
        ref_mask = region_mask(BIN_CENTERS, lower, upper)
        supported = ref_mask & (reference["counts"] >= MIN_BIN_EVENTS)
        axis.errorbar(
            BIN_CENTERS[supported],
            reference["fraction"][supported],
            yerr=reference["wilson_half_width"][supported],
            fmt=".",
            color="black",
            alpha=0.32,
            markersize=2,
            linewidth=0.5,
            label="Frozen target reference (Wilson z=1)",
        )
        c_energy = np.asarray([row["energy_center_kev"] for row in context100])
        c_fraction = np.asarray([row["empirical_fraction"] for row in context100], dtype=float)
        c_counts = np.asarray([row["event_count"] for row in context100])
        c_mask = region_mask(c_energy, lower, upper) & (c_counts > 0)
        axis.scatter(
            c_energy[c_mask],
            c_fraction[c_mask],
            marker="x",
            s=11,
            color="#666666",
            alpha=0.7,
            label="Context seed 100 (nonempty 5-keV bins)",
        )
        for method, (energy, prediction) in illustrative_by_method.items():
            mask = region_mask(energy, lower, upper, inclusive_upper=True)
            axis.plot(
                energy[mask],
                prediction[mask],
                color=METHOD_COLORS[method],
                linewidth=1.05,
                label=method,
            )
        axis.set_xlim(lower, upper)
        axis.set_ylim(-0.03, 1.03)
        axis.set_ylabel("Passing efficiency")
        axis.set_title(title)
        axis.grid(alpha=0.15)
    axes[1].set_xlabel("Energy (keV)")
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    axes[0].legend(unique.values(), unique.keys(), ncol=3, frameon=False, fontsize=7)
    fig.savefig(figure1_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 6.8), constrained_layout=True)
    for axis, (window, (lower, upper)) in zip(
        axes, CONTINUUM_WINDOWS.items(), strict=True
    ):
        for method in (
            "CNP",
            "Attentive CNP",
            "Attentive CNP + PE",
            "Density-guided CNP",
            "Kernel",
            "Bernoulli GP",
        ):
            subset = "original_n5000" if method in METHOD_NAMES.values() else "context_only"
            rows = [
                row
                for row in mean_bin_rows
                if row["method"] == method and row["subset_id"] == subset
            ]
            energy = np.asarray([row["energy_center_kev"] for row in rows])
            pull = np.asarray(
                [np.nan if row["reference_pull"] is None else row["reference_pull"] for row in rows]
            )
            mask = region_mask(energy, lower, upper) & np.isfinite(pull)
            axis.plot(
                energy[mask], pull[mask], marker=".", markersize=2, linewidth=0.8,
                color=METHOD_COLORS[method], label=method,
            )
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.axhspan(-1.0, 1.0, color="black", alpha=0.05)
        axis.axhline(-2.0, color="gray", linewidth=0.5, linestyle="--")
        axis.axhline(2.0, color="gray", linewidth=0.5, linestyle="--")
        axis.set_xlim(lower, upper)
        axis.set_ylabel("Reference-only pull")
        axis.set_title(window.replace("continuum_", "").replace("_", "–") + " keV")
        axis.grid(alpha=0.15)
    axes[1].set_xlabel("5-keV-bin center (keV)")
    axes[0].legend(ncol=3, frameon=False, fontsize=8)
    fig.savefig(figure2_path, dpi=180)
    plt.close(fig)

    def summary_lookup(method: str, region: str, subset: str = "original_n5000"):
        matches = [
            row
            for row in regional_summary
            if row["method"] == method
            and row["region"] == region
            and row["subset_id"] == subset
        ]
        if len(matches) != 1:
            raise ValueError(f"Regional summary lookup failed: {method}, {region}, {subset}")
        return matches[0]

    def continuum_lookup(method: str, subset: str = "original_n5000"):
        matches = [
            row
            for row in continuum_summary
            if row["method"] == method
            and row["window"] == "equal_window_mean"
            and row["subset_id"] == subset
        ]
        if len(matches) != 1:
            raise ValueError("Continuum lookup failed")
        return matches[0]

    headline_methods = (
        ("CNP", "original_n5000"),
        ("Attentive CNP", "original_n5000"),
        ("Attentive CNP + PE", "original_n5000"),
        ("Density-guided CNP", "original_n5000"),
        ("Kernel", "context_only"),
        ("Bernoulli GP", "context_only"),
    )
    table_lines = []
    reference_lines = []
    balance_lines = []
    for method, subset in headline_methods:
        values = []
        ref_values = []
        for region, _, _, _ in REGIONAL_AGREEMENT:
            item = summary_lookup(method, region, subset)
            if item["combined_proxy_G1_cell_mean"] is None:
                values.append("NA")
            else:
                values.append(
                    "{:.2f}/{:.2f}/{:.2f}".format(
                        item["combined_proxy_G1_cell_mean"],
                        item["combined_proxy_G2_cell_mean"],
                        item["combined_proxy_G3_cell_mean"],
                    )
                )
            ref_values.append(
                "{:.2f}/{:.2f}/{:.2f}".format(
                    item["reference_only_G1_cell_mean"],
                    item["reference_only_G2_cell_mean"],
                    item["reference_only_G3_cell_mean"],
                )
            )
        cont = continuum_lookup(method, subset)
        table_lines.append(
            "| {} | {} | {} | {} | {} | {} | {:.2f} |".format(
                method, *values, cont["continuum_pull_RMS_cell_mean"]
            )
        )
        reference_lines.append("| {} | {} | {} | {} | {} | {} |".format(method, *ref_values))
        rough = {
            row["window"]: row["mean_absolute_second_difference"]
            for row in roughness
            if row["curve_kind"] == "mean_curve_across_saved_cells"
            and row["method"] == method
            and row["subset_id"] == subset
        }
        overall = summary_lookup(method, "Overall", subset)
        proxy_width = overall["combined_proxy_full_width_pp_cell_mean"]
        balance_lines.append(
            "| {} | {:.2f} | {:.4g} | {:.4g} | {} |".format(
                method,
                cont["continuum_pull_RMS_cell_mean"],
                rough["continuum_1700_2000"],
                rough["continuum_2200_2400"],
                f"{proxy_width:.2f}" if proxy_width is not None else "NA",
            )
        )

    mc_match = {
        "matches_headline_models": False,
        "reason": (
            "The archived MC study used full-pool models, context size 2,000, training seed 0, "
            "and context seed 100; the headline uses original 5k models and context size 500."
        ),
        "archived_context_size": mc_manifest["context_size"],
        "archived_context_seed": mc_manifest["context_seed"],
        "claim_decision": mc_manifest["claim_decision"],
    }
    report_path = output_root / "reports/paper_presentation_export.md"
    report = """# Existing-prediction export for the talk-led paper

Status: complete. This export recovered existing saved predictions only; it ran no
training, checkpoint inference, MC prediction, GP fitting/prediction, bandwidth
selection, or data selection. New text uses *efficiency* for the probability of
passing the fixed classifier cut; legacy artifact field names retain *acceptance*
for compatibility.

## Recovered matrix

- Neural: 600/600 cells (original-order 2k, 5k, and 10k plus two additional 5k
  orderings; four architectures, three initialization seeds, ten 500-event
  contexts).
- Headline neural comparison: 120/120 original-5k cells.
- Classical headline: 10/10 context-only kernel cells and 10/10 selected
  Bernoulli-GP Matérn-3/2 cells at context size 500.
- Stronger-data control: 10/10 pooled-data kernel cells, labeled separately.
- Every cell uses exactly the same frozen 114,400 target rows and the matching
  context subset. Artifact and manifest hashes were checked before aggregation.

## Table 1: regional Gaussian agreement proxy

Each entry is the mean of per-cell `G1/G2/G3`, with each cell computed before
aggregation. Neural cells use the historical combined proxy; classical combined
proxy values are unavailable because no model SD was saved. The companion column
is the equal-window continuum pull RMS on the common finite-reference scale.

| Method | Overall | FE | SE | DEP | feature (1620 keV) | Continuum pull RMS |
|---|---:|---:|---:|---:|---:|---:|
{}

The executable notebook formula was followed: `G_k = Phi(k-z) - Phi(-k-z)`
with `z=(f-p)/sqrt(s_emp^2+u_proxy^2)`. The notebook docstring instead describes
a transform with `s_emp` in the Gaussian-CDF denominator; that prose is
inconsistent with the executable implementation. Here `u_proxy` is the mean
pointwise neural prediction SD and is **not** the SD of a pooled region mean.
Wider proxies can improve G without improving prediction. G is a transform of
one pooled residual, not repeated-sampling coverage or a calibrated posterior
probability. Regions use only frozen targets: Overall is inclusive 500--3000 keV;
FE, SE, DEP, and the 1620-keV feature use inclusive +/-5-keV event windows.

For a fair all-method view, the separate reference-only panel uses `s_emp` alone:

| Method | Overall | FE | SE | DEP | feature (1620 keV) |
|---|---:|---:|---:|---:|---:|
{}

The off-feature companion diagnostics make the interpretation testable:

| Method | Continuum pull RMS | Roughness 1700--2000 | Roughness 2200--2400 | Overall combined-proxy full width (pp) |
|---|---:|---:|---:|---:|
{}

The PE model's strong combined-proxy G values coincide with a much wider proxy,
the largest continuum pull RMS, and substantially larger saved-grid roughness.
Thus those G values do not establish better reconstruction. In this matrix the
density-guided model has the lowest continuum pull RMS while retaining much
lower roughness than PE, which supports a useful balance interpretation. CNP
and Attentive CNP are comparatively smooth but their near-zero pooled G values
at SE, DEP, and the 1620-keV feature show that broad-trend behavior alone misses
localized changes. This is a descriptive comparison on one historically
exposed target, not proof of overfitting or calibrated uncertainty.

## What the saved predictions support

The exports let Table 1 complement, rather than duplicate, Figure 1. The
regional pooled scores quantify broad and localized agreement, while Figure 2
shows signed deviations bin by bin. The continuum residual table reports bias,
centered RMS, and pull RMS, so alternating over- and underprediction cannot be
hidden by regional pooling. Its combined row gives equal weight to the
1700--2000 and 2200--2400-keV windows: mean-square quantities are averaged
between windows before taking a square root, while signed means are averaged
between window means.

The density-guided model can be described as balancing local reconstruction and
off-feature discrepancy only where the reported regional and continuum values
show that pattern. These diagnostics do not establish universal superiority.
The 2k failure, 10k nonmonotonicity, and sparse-tail limitation from Phase 3
remain part of the result. In particular, small training subsets contain no
sample-eligible sparse-tail event, and the 5k density-guided sparse-tail error
remains unfavorable.

The retained density-guided budget result is explicit:

| Budget | Exact nominal events | Sampling-eligible events | Peak MAE (pp) | Continuum MAE (pp) |
|---|---:|---:|---:|---:|
| 2k | 2,000 | 1,895 | 9.40 | 12.89 |
| 5k, original ordering | 5,000 | 4,984 | 5.29 | 3.96 |
| 10k | 10,000 | 9,980 | 5.84 | 4.01 |
| 18.9k full pool | 18,866 | 18,836 | 3.56 | 4.07 |

The 10k point is nonmonotonic relative to 5k, so these data do not identify an
exact minimum sample requirement. The 5k density-guided sparse-tail MAE is
19.39 pp, compared with 2.99 pp for the stronger-data pooled-kernel control.
The full pool is exactly 18,866 events and is never relabeled 20k. All neural
claims remain conditional on the classifier pretrained with 18,866 selected
events from 377,330 candidates; they are not end-to-end 5k claims.

Grid roughness is the saved-protocol mean absolute second difference at the
original 1-keV spacing, reported per cell and again for each mean curve. It
combines learned variation and MC noise. Therefore the defensible phrase is
*excessive variation* or *off-feature discrepancy*, not overfitting. The prior
MC-precision study does not match this comparison: it used full-pool PE and
density-guided models at seed 0/context 100 with 2,000 context events, whereas
the headline is original 5k with 500 context events. Its conclusions are not
generalized to these cells.

## Figure 2 uncertainty boundary

All energy-resolved pulls use actual-target-event bin means and the Wilson z=1
half-width as a common reference scale. `C_k` is the fraction of supported bins
with `abs(z_ref)<=k`; it is distinct from Table 1's pooled-residual `G_k`.
Neither is coverage of unknown true efficiency. The Wilson half-width is a
descriptive finite-reference scale, not an exact standard error.

No valid combined bin-mean model uncertainty can be reconstructed. Neural NPZ
files retain only pointwise event SDs; without per-pass bin means or within-pass
covariances, the SD of a bin mean is unidentified. GP and kernel files retain no
predictive SD. Combined pulls, combined `C_k`, and comparable interval widths
are therefore marked unavailable rather than fabricated.

A definitive overfitting attribution remains blocked by missing matched
training/reference and numerical-precision outputs. A separate bounded proposal,
not executed here, would freeze the original-5k seed-0/context-100 checkpoints
for all four architectures, save per-pass bin means on both the matching
training pool and frozen target under two prespecified independent nested
50/200-pass streams, and report generalization-gap and stream-stability
diagnostics in the same fixed windows. It would require explicit approval and a
measured inference cost gate; context fit versus target fit alone would still
not be treated as proof of overfitting.

## Portable panels

- `paper_figure1_efficiency_curves.png`: the prespecified illustrative fit
  (original 5k, initialization seed 0, context seed 100), shown both over the
  full 500--3000-keV range and the 1500--3000-keV presentation view. It includes
  the target reference, context points, four neural methods, context-only kernel,
  pooled-data kernel control, and Bernoulli GP.
- `paper_figure2_reference_pulls.png`: energy-resolved reference-only pulls for
  the mean saved curve in both fixed continuum windows. Mean curves are labeled
  as averages, not deployable single fits.
- CSVs contain the illustrative grids, all 5k-subset mean grids, reference and
  context counts, per-cell regional/continuum/pull diagnostics, hierarchical
  summaries, fixed pull histograms, and compact illustrative/mean bin
  predictions. Complete per-cell bin arrays remain in the server-only NPZ files.

## Validation and interpretation limits

Direct event-level bin means were reconciled against the saved 5-keV-bin arrays
and archived MAE/RMSE summaries. The maximum absolute metric discrepancy was
`{:.3e}` in probability units (required tolerance `2e-14` for float64 neural/GP
artifacts and `2e-8` only for float32 kernel archive comparisons).

The frozen target was historically inspected, including for the talk. This is a
prospectively specified follow-up analysis on historically exposed data, not an
untouched test. The 1620-keV structure remains named by energy because its
physical isotope identity is unresolved in audited sources. Context draws
overlap and share one finite target, so context, initialization, and training-
subset dispersions are descriptive and are not treated as independent-dataset
uncertainty. Hierarchical summary CSVs report within-seed context SD,
initialization SD across seed means, and—on dedicated all-three-5k rows—training-
subset SD across subset means.
""".format(
        "\n".join(table_lines),
        "\n".join(reference_lines),
        "\n".join(balance_lines),
        maximum_bin_reconciliation_difference,
    )
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x") as stream:
        stream.write(report)

    output_files = list(table_paths.values()) + [figure1_path, figure2_path, report_path]
    manifest_path = output_root / "manifests/paper_presentation_export.json"
    manifest = {
        "schema_version": 1,
        "analysis": "talk-led paper export from existing saved predictions",
        "status": "completed",
        "created_at": utc_now(),
        "source_head": source_head,
        "script": "ml4phy-paper/scripts/export_paper_presentation.py",
        "script_sha256": sha256_file(Path(__file__)),
        "no_new_prediction_or_training": True,
        "terminology": {
            "new_text": "efficiency",
            "legacy_acceptance_fields": "same fixed-cut passing probability; preserved for compatibility",
        },
        "matrix": {
            "neural_recovered": 600,
            "neural_expected": 600,
            "headline_original_5k_neural": 120,
            "context_only_kernel": 10,
            "pooled_data_kernel_control": 10,
            "selected_matern32_gp": 10,
            "missing_cells": [],
        },
        "fixed_protocol": {
            "target_events": 114400,
            "target_identity_sha256": frozen_manifest["roles"]["final_target"][
                "identity_sha256"
            ],
            "target_rows_match_all_cells": True,
            "context_size": 500,
            "context_seeds": list(CONTEXT_SEEDS),
            "threshold": threshold,
            "bin_width_kev": 5.0,
            "minimum_bin_events": MIN_BIN_EVENTS,
            "grid_spacing_kev": float(np.diff(grid_reference).mean()),
        },
        "definitions": {
            "table1": (
                "Per cell: f=mean(outcome), p=mean(prediction), "
                "u_proxy=mean(pointwise prediction_std), "
                "s_emp=sqrt(max(f*(1-f),1e-12)/N), "
                "z=(f-p)/sqrt(s_emp^2+u_proxy^2), G_k=Phi(k-z)-Phi(-k-z)."
            ),
            "table1_reference_only": "Same transform with z=(f-p)/s_emp for all methods.",
            "figure2_reference_pull": (
                "z_ref=(f_bin-p_bin)/Wilson_z1_half_width; supported bins have at least four targets."
            ),
            "continuum_weighting": (
                "Equal bins within each fixed window; equal-window mean of window-specific "
                "mean squares before square root; signed means average window means."
            ),
            "roughness": "Mean absolute second difference on each saved 1-keV grid window.",
            "histogram": "Width 0.5 over [-10,10], with explicit underflow and overflow.",
            "wide_curve_column_names": (
                "<illustrative|mean>__<subset_id>__<architecture_id>; illustrative is fixed "
                "to initialization seed 0/context seed 100, and mean averages the declared group."
            ),
            "hierarchical_variation": (
                "Within-seed context SD, initialization SD across seed means, and dedicated "
                "training-subset SD across the three 5k subset means are reported separately."
            ),
        },
        "uncertainty_availability": {
            "table1_neural_proxy": "available as mean pointwise SD; descriptive only",
            "table1_kernel_gp_proxy": "unavailable: no saved model SD",
            "figure2_combined_bin_mean": (
                "unavailable for every method: per-pass bin means/covariances absent; "
                "pointwise SD is insufficient"
            ),
        },
        "notebook_audit": {
            "path": "notebooks/data_visualization.ipynb",
            "section": "8.4.6 _pooled_coverage",
            "executed": False,
            "executable_formula_followed": True,
            "docstring_formula_consistent": False,
            "historical_context_target_concatenation_removed": True,
            "historical_grid_interpolation_removed": True,
            "historical_total_N_141474_not_transplanted": True,
        },
        "mc_precision_compatibility": mc_match,
        "reconciliation": {
            "maximum_absolute_mae_or_rmse_difference_probability_units": maximum_bin_reconciliation_difference,
            "float64_tolerance": 2e-14,
            "kernel_float32_tolerance": 2e-8,
        },
        "source_manifests": {
            str(path.relative_to(repo)): sha256_file(path)
            for path in (
                phase2_manifest_path,
                phase3_manifest_path,
                kernel_manifest_path,
                frozen_manifest_path,
                extension_manifest_path,
                mc_manifest_path,
            )
        },
        "source_artifacts": source_records,
        "server_only_policy": (
            "Raw event predictions, complete per-cell bin arrays, checkpoints, and row identities remain on server."
        ),
        "outputs": {
            str(path.relative_to(repo) if path.is_relative_to(repo) else path): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in output_files
        },
        "runtime_seconds": time.perf_counter() - start,
        "historical_data_exposure": (
            "Prospectively specified follow-up analysis on historically exposed data; not an untouched test."
        ),
    }
    write_json(manifest_path, manifest)
    print(json.dumps({
        "status": "completed",
        "matrix": manifest["matrix"],
        "maximum_reconciliation_difference": maximum_bin_reconciliation_difference,
        "runtime_seconds": manifest["runtime_seconds"],
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
