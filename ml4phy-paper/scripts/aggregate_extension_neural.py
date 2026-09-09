#!/usr/bin/env python3
"""Aggregate the frozen 480-cell Phase 1 neural context-size matrix."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ARCHITECTURES = ("m0", "m1", "m2", "ours")
METHOD_NAMES = {
    "m0": "CNP",
    "m1": "Attentive CNP",
    "m2": "Attentive CNP + PE",
    "ours": "Density-guided CNP (ours)",
}
TRAINING_SEEDS = (0, 1, 2)
CONTEXT_SEEDS = tuple(range(100, 110))
CONTEXT_SIZES = (250, 500, 1000, 2000)
PEAK_REGIONS = ("DEP", "Bi-214", "SE", "FE")
CONTINUUM_REGIONS = ("continuum_1700_2000", "continuum_2200_2400")
REGION_NAMES = {
    "full": "Full 500-3000 keV",
    "DEP": "Tl-208 DEP 1592 keV",
    "Bi-214": "Bi-212 1620.74-keV feature",
    "continuum_1700_2000": "Continuum 1700-2000 keV",
    "SE": "Tl-208 SE 2103 keV",
    "continuum_2200_2400": "Continuum 2200-2400 keV",
    "FE": "Tl-208 FE 2614 keV",
    "sparse_tail": "Sparse tail 2700-3000 keV",
}
PEAK_NAMES = {
    "DEP": "Tl-208 DEP 1592 keV",
    "Bi-214": "Bi-212 1620.74-keV feature",
    "SE": "Tl-208 SE 2103 keV",
    "FE": "Tl-208 FE 2614 keV",
}
COLORS = {"m0": "#4c78a8", "m1": "#f58518", "m2": "#54a24b", "ours": "#e45756"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def write_json(path: Path, payload: dict) -> None:
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def old_model_id(architecture: str, training_seed: int) -> str:
    if architecture == "m2":
        return "m2_seed0_pilot" if training_seed == 0 else f"m2_seed{training_seed}"
    return (
        "cell17_seed0_recovered" if training_seed == 0 else f"cell17_seed{training_seed}"
    )


def cell_directory(
    repo: Path, architecture: str, training_seed: int, context_seed: int, context_size: int
) -> tuple[Path, str]:
    if architecture in {"m2", "ours"} and context_size == 2000:
        run_id = (
            f"20260908-{old_model_id(architecture, training_seed)}-final-"
            f"ctx-s{context_seed}-drop10100-mc50"
        )
        return repo / "ml4phy-paper/runs" / run_id, "archived_reuse"
    if (architecture, training_seed, context_seed, context_size) == ("ours", 0, 100, 500):
        run_id = "20260909-ours-seed0-final-ctx-s100-n500-drop10100-mc50-pilot"
        return repo / "ml4phy-paper/runs/extension_eval" / run_id, "pilot_reuse"
    run_id = (
        f"20260909-{architecture}-seed{training_seed}-final-"
        f"ctx-s{context_seed}-n{context_size}-drop10100-mc50"
    )
    return repo / "ml4phy-paper/runs/extension_eval" / run_id, "new_evaluation"


def hierarchical_summary(data: pd.DataFrame, metrics: list[str]) -> list[dict]:
    rows = []
    for (architecture, context_size), group in data.groupby(
        ["architecture_id", "context_size"], sort=False
    ):
        seed_rows = []
        for training_seed, seed_group in group.groupby("training_seed", sort=True):
            if len(seed_group) != len(CONTEXT_SEEDS):
                raise ValueError(f"Incomplete context matrix for {architecture}, {training_seed}")
            seed_row = {"training_seed": int(training_seed)}
            for metric in metrics:
                values = seed_group[metric].to_numpy(dtype=np.float64)
                seed_row[f"{metric}_mean"] = float(values.mean())
                seed_row[f"{metric}_context_sd"] = float(values.std(ddof=1))
            seed_rows.append(seed_row)
        if len(seed_rows) != len(TRAINING_SEEDS):
            raise ValueError(f"Incomplete training-seed matrix for {architecture}")
        row = {
            "method": METHOD_NAMES[architecture],
            "architecture_id": architecture,
            "context_size": int(context_size),
            "training_seed_count": len(TRAINING_SEEDS),
            "context_draws_per_training_seed": len(CONTEXT_SEEDS),
        }
        for metric in metrics:
            seed_means = np.asarray([item[f"{metric}_mean"] for item in seed_rows])
            context_sds = np.asarray([item[f"{metric}_context_sd"] for item in seed_rows])
            row[f"{metric}_mean_of_seed_means"] = float(seed_means.mean())
            row[f"{metric}_sd_across_seed_means"] = float(seed_means.std(ddof=1))
            row[f"{metric}_mean_context_sd_within_seed"] = float(context_sds.mean())
        rows.append(row)
    return rows


def main() -> None:
    repo = Path(".").resolve()
    output_root = repo / "ml4phy-paper"
    outputs = {
        "cells": output_root / "tables/phase1_neural_cell_metrics.csv",
        "bins": output_root / "tables/phase1_neural_bin_error_cells.csv",
        "contrasts": output_root / "tables/phase1_neural_peak_contrast_cells.csv",
        "summary": output_root / "tables/phase1_neural_context_size_summary.csv",
        "variation": output_root / "tables/phase1_neural_context_variation.csv",
        "comparison": output_root / "tables/phase1_neural_method_comparison.csv",
        "curves": output_root / "tables/phase1_neural_acceptance_curves.csv",
        "efficiency_figure": output_root / "figures/phase1_neural_context_efficiency.png",
        "variation_figure": output_root / "figures/phase1_neural_context_variation.png",
        "full_figure": output_root / "figures/phase1_neural_acceptance_full.png",
        "local_figure": output_root / "figures/phase1_neural_acceptance_local.png",
        "report": output_root / "reports/phase1_neural_result.md",
        "manifest": output_root / "manifests/phase1_neural_result.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))

    protocol_path = output_root / "manifests/extension_protocol_v1.json"
    base_protocol_path = output_root / "manifests/frozen_protocol_v1.json"
    registry_path = output_root / "configs/extension_models_v1.json"
    campaign_path = output_root / "runs/campaigns/20260909-phase1-neural-v1/campaign_record.json"
    protocol = read_json(protocol_path)
    base_protocol = read_json(base_protocol_path)
    registry = read_json(registry_path)
    campaign = read_json(campaign_path)
    if campaign["status"] != "completed" or len(campaign["completed"]) != 419:
        raise ValueError("Neural campaign is not complete")
    if len(campaign["reused_pilot"]) != 1 or campaign["failures"]:
        raise ValueError("Unexpected neural campaign pilot/failure accounting")

    subset_hashes = {
        (item["phase"], item["context_seed"], item["context_size"]): item[
            "identity_sha256"
        ]
        for item in protocol["context_protocol"]["subsets"]
    }
    cell_rows: list[dict] = []
    bin_rows: list[dict] = []
    contrast_rows: list[dict] = []
    input_rows: list[dict] = []
    curve_values: dict[tuple[str, int, int], list[np.ndarray]] = {}
    reference: dict[str, np.ndarray] | None = None
    grid_energy: np.ndarray | None = None
    target_hash = base_protocol["roles"]["final_target"]["identity_sha256"]

    for architecture in ARCHITECTURES:
        for training_seed in TRAINING_SEEDS:
            model_spec = registry["models"][f"{architecture}_seed{training_seed}"]
            if sha256_file(repo / model_spec["checkpoint"]) != model_spec[
                "checkpoint_sha256"
            ]:
                raise ValueError(f"Checkpoint mismatch: {architecture}, {training_seed}")
            for context_size in CONTEXT_SIZES:
                curve_values[(architecture, training_seed, context_size)] = []
                for context_seed in CONTEXT_SEEDS:
                    directory, provenance = cell_directory(
                        repo, architecture, training_seed, context_seed, context_size
                    )
                    summary_path = directory / "summary.json"
                    event_path = directory / "event_predictions.npz"
                    curve_path = directory / "curve.npz"
                    summary = read_json(summary_path)
                    expected_context_hash = subset_hashes[("final", context_seed, context_size)]
                    if (
                        summary["status"] != "completed"
                        or summary["protocol"]["phase"] != "final"
                        or summary["randomness"]["training_seed"] != training_seed
                        or summary["randomness"]["context_seed"] != context_seed
                        or summary["randomness"]["dropout_seed"] != 10100
                        or summary["counts"]["context_draw"] != context_size
                        or summary["counts"]["context_per_mc_pass"] != context_size
                        or summary["counts"]["target"] != 114400
                        or summary["counts"]["mc_passes"] != 50
                        or summary["protocol"]["target_identity_sha256"] != target_hash
                        or summary["protocol"]["context_draw_identity_sha256"]
                        != expected_context_hash
                    ):
                        raise ValueError(f"Cell contract mismatch: {summary_path}")
                    for path, key in (
                        (event_path, "event_predictions"),
                        (curve_path, "curve"),
                    ):
                        if sha256_file(path) != summary["server_only_outputs"][key]["sha256"]:
                            raise ValueError(f"Server-only output mismatch: {path}")
                    if summary["source_worktree_status"]:
                        raise ValueError(f"Cell used a dirty source tree: {summary_path}")

                    events = {item["region"]: item for item in summary["metrics"]["event"]}
                    bins = {item["region"]: item for item in summary["metrics"]["bin_5kev"]}
                    peak_mae = 100.0 * float(np.mean([bins[key]["mae"] for key in PEAK_REGIONS]))
                    continuum_mae = 100.0 * float(
                        np.mean([bins[key]["mae"] for key in CONTINUUM_REGIONS])
                    )
                    cell_rows.append(
                        {
                            "method": METHOD_NAMES[architecture],
                            "architecture_id": architecture,
                            "training_seed": training_seed,
                            "context_seed": context_seed,
                            "context_size": context_size,
                            "provenance": provenance,
                            "peak_region_mean_mae_percentage_points": peak_mae,
                            "continuum_region_mean_mae_percentage_points": continuum_mae,
                            "sparse_tail_mae_percentage_points": 100.0
                            * bins["sparse_tail"]["mae"],
                            "global_brier": events["full"]["brier"],
                            "equal_region_brier": events["equal_region_mean"]["brier"],
                            "inference_seconds": summary["runtime"]["inference_seconds"],
                            "peak_memory_mib": summary["runtime"]["peak_memory_mib"],
                            "run_id": summary["run_id"],
                        }
                    )
                    for region, metric in bins.items():
                        bin_rows.append(
                            {
                                "method": METHOD_NAMES[architecture],
                                "architecture_id": architecture,
                                "training_seed": training_seed,
                                "context_seed": context_seed,
                                "context_size": context_size,
                                "region_id": region,
                                "region": REGION_NAMES[region],
                                "valid_bin_count": metric["n_valid_bins"],
                                "excluded_bin_count": metric["n_excluded_bins"],
                                "events_in_valid_bins": metric["n_events_in_valid_bins"],
                                "mae_percentage_points": 100.0 * metric["mae"],
                                "rmse_percentage_points": 100.0 * metric["rmse"],
                            }
                        )
                    for metric in summary["metrics"]["peak_sideband_contrast"]:
                        contrast_rows.append(
                            {
                                "method": METHOD_NAMES[architecture],
                                "architecture_id": architecture,
                                "training_seed": training_seed,
                                "context_seed": context_seed,
                                "context_size": context_size,
                                "feature_id": "Bi-212"
                                if metric["peak"] == "Bi-214"
                                else metric["peak"],
                                "feature": PEAK_NAMES[metric["peak"]],
                                "status": metric["status"],
                                "center_event_count": metric["center_events"],
                                "sideband_event_count": metric["sideband_events"],
                                "empirical_contrast_percentage_points": 100.0
                                * metric["empirical_contrast"],
                                "predicted_contrast_percentage_points": 100.0
                                * metric["predicted_contrast"],
                                "absolute_contrast_error_percentage_points": 100.0
                                * metric["absolute_contrast_error"],
                            }
                        )
                    with np.load(curve_path) as archive:
                        current_grid = archive["energy_kev"].astype(np.float64)
                        current_curve = archive["prediction"].astype(np.float64)
                        current_reference = {
                            "center": archive["bin_centers_kev"].astype(np.float64),
                            "count": archive["bin_counts"].astype(np.int64),
                            "acceptance": archive["empirical_rate"].astype(np.float64),
                        }
                    if grid_energy is None:
                        grid_energy = current_grid
                        reference = current_reference
                    else:
                        if not np.array_equal(grid_energy, current_grid):
                            raise ValueError(f"Grid mismatch: {curve_path}")
                        assert reference is not None
                        for key in reference:
                            if not np.allclose(
                                reference[key], current_reference[key], equal_nan=True
                            ):
                                raise ValueError(f"Reference mismatch: {curve_path}")
                    if context_size in {500, 2000}:
                        curve_values[(architecture, training_seed, context_size)].append(
                            current_curve
                        )
                    input_rows.append(
                        {
                            "run_id": summary["run_id"],
                            "provenance": provenance,
                            "summary_sha256": sha256_file(summary_path),
                            "event_predictions_sha256": sha256_file(event_path),
                            "curve_sha256": sha256_file(curve_path),
                        }
                    )

    if len(cell_rows) != 480 or len(input_rows) != 480:
        raise ValueError("Expected exactly 480 validated neural cells")
    cells = pd.DataFrame(cell_rows)
    metrics = [
        "peak_region_mean_mae_percentage_points",
        "continuum_region_mean_mae_percentage_points",
        "sparse_tail_mae_percentage_points",
        "global_brier",
        "equal_region_brier",
        "inference_seconds",
        "peak_memory_mib",
    ]
    summary_rows = hierarchical_summary(cells, metrics)
    summary_frame = pd.DataFrame(summary_rows)

    variation_rows = []
    for (architecture, context_size, training_seed), group in cells.groupby(
        ["architecture_id", "context_size", "training_seed"], sort=False
    ):
        row = {
            "method": METHOD_NAMES[architecture],
            "architecture_id": architecture,
            "context_size": int(context_size),
            "training_seed": int(training_seed),
            "overlapping_context_draw_count": len(group),
        }
        for metric in metrics[:5]:
            values = group[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_context_mean"] = float(values.mean())
            row[f"{metric}_context_sd"] = float(values.std(ddof=1))
            row[f"{metric}_context_range"] = float(values.max() - values.min())
        variation_rows.append(row)

    comparison_rows = []
    for context_size in (500, 2000):
        for architecture in ARCHITECTURES:
            row = summary_frame[
                (summary_frame["architecture_id"] == architecture)
                & (summary_frame["context_size"] == context_size)
            ].iloc[0]
            comparison_rows.append(
                {
                    "method": METHOD_NAMES[architecture],
                    "architecture_id": architecture,
                    "context_size": context_size,
                    "parameter_count": registry["models"][f"{architecture}_seed0"][
                        "parameter_count"
                    ],
                    "peak_mae_percentage_points_mean": row[
                        "peak_region_mean_mae_percentage_points_mean_of_seed_means"
                    ],
                    "peak_mae_percentage_points_seed_sd": row[
                        "peak_region_mean_mae_percentage_points_sd_across_seed_means"
                    ],
                    "continuum_mae_percentage_points_mean": row[
                        "continuum_region_mean_mae_percentage_points_mean_of_seed_means"
                    ],
                    "continuum_mae_percentage_points_seed_sd": row[
                        "continuum_region_mean_mae_percentage_points_sd_across_seed_means"
                    ],
                    "global_brier_mean": row["global_brier_mean_of_seed_means"],
                    "global_brier_seed_sd": row["global_brier_sd_across_seed_means"],
                }
            )

    assert grid_energy is not None and reference is not None
    curve_rows = []
    aggregate_curves: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for context_size in (500, 2000):
        for architecture in ARCHITECTURES:
            seed_means = []
            context_sds = []
            for training_seed in TRAINING_SEEDS:
                matrix = np.stack(curve_values[(architecture, training_seed, context_size)])
                if matrix.shape[0] != len(CONTEXT_SEEDS):
                    raise ValueError("Incomplete curve matrix")
                seed_means.append(matrix.mean(axis=0))
                context_sds.append(matrix.std(axis=0, ddof=1))
            seed_matrix = np.stack(seed_means)
            mean = seed_matrix.mean(axis=0)
            seed_sd = seed_matrix.std(axis=0, ddof=1)
            context_sd = np.stack(context_sds).mean(axis=0)
            aggregate_curves[(architecture, context_size)] = (mean, seed_sd)
            for energy, mean_value, seed_value, context_value in zip(
                grid_energy, mean, seed_sd, context_sd, strict=True
            ):
                curve_rows.append(
                    {
                        "method": METHOD_NAMES[architecture],
                        "architecture_id": architecture,
                        "context_size": context_size,
                        "energy_kev": energy,
                        "mean_of_training_seed_context_means": mean_value,
                        "sd_across_training_seed_context_means": seed_value,
                        "mean_context_sd_within_training_seed": context_value,
                        "training_seed_count": 3,
                        "context_draws_per_training_seed": 10,
                        "mc_passes_per_cell": 50,
                    }
                )

    write_csv(outputs["cells"], cell_rows)
    write_csv(outputs["bins"], bin_rows)
    write_csv(outputs["contrasts"], contrast_rows)
    write_csv(outputs["summary"], summary_rows)
    write_csv(outputs["variation"], variation_rows)
    write_csv(outputs["comparison"], comparison_rows)
    write_csv(outputs["curves"], curve_rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for axis, metric, title in (
        (
            axes[0],
            "peak_region_mean_mae_percentage_points",
            "Prespecified peak regions",
        ),
        (
            axes[1],
            "continuum_region_mean_mae_percentage_points",
            "Prespecified continuum regions",
        ),
    ):
        for architecture in ARCHITECTURES:
            group = summary_frame[summary_frame["architecture_id"] == architecture].sort_values(
                "context_size"
            )
            axis.errorbar(
                group["context_size"],
                group[f"{metric}_mean_of_seed_means"],
                yerr=group[f"{metric}_sd_across_seed_means"],
                marker="o",
                capsize=3,
                color=COLORS[architecture],
                label=METHOD_NAMES[architecture],
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks(CONTEXT_SIZES, labels=[str(value) for value in CONTEXT_SIZES])
        axis.set_xlabel("Context events")
        axis.set_ylabel("5-keV-bin MAE (percentage points)")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(outputs["efficiency_figure"], dpi=180)
    plt.close(fig)

    variation = pd.DataFrame(variation_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for axis, metric, title in (
        (
            axes[0],
            "peak_region_mean_mae_percentage_points_context_sd",
            "Peak-error context variation",
        ),
        (
            axes[1],
            "continuum_region_mean_mae_percentage_points_context_sd",
            "Continuum-error context variation",
        ),
    ):
        for architecture in ARCHITECTURES:
            grouped = (
                variation[variation["architecture_id"] == architecture]
                .groupby("context_size")[metric]
                .mean()
            )
            axis.plot(
                grouped.index,
                grouped.values,
                marker="o",
                color=COLORS[architecture],
                label=METHOD_NAMES[architecture],
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks(CONTEXT_SIZES, labels=[str(value) for value in CONTEXT_SIZES])
        axis.set_xlabel("Context events")
        axis.set_ylabel("Within-seed context SD (percentage points)")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(outputs["variation_figure"], dpi=180)
    plt.close(fig)

    supported = reference["count"] >= 4
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 6.8), constrained_layout=True)
    for axis, context_size in zip(axes, (500, 2000), strict=True):
        axis.scatter(
            reference["center"][supported],
            reference["acceptance"][supported],
            s=7,
            color="0.45",
            alpha=0.5,
            label="Reference (5-keV bins)",
        )
        for architecture in ARCHITECTURES:
            mean, _ = aggregate_curves[(architecture, context_size)]
            axis.plot(
                grid_energy,
                mean,
                color=COLORS[architecture],
                linewidth=1.1,
                label=METHOD_NAMES[architecture],
            )
        axis.set_title(f"Context size {context_size}")
        axis.set_xlim(500, 3000)
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel("Energy (keV)")
        axis.set_ylabel("Acceptance")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    fig.savefig(outputs["full_figure"], dpi=180)
    plt.close(fig)

    panels = [
        ("Tl-208 DEP and Bi-212", (1565.0, 1645.0)),
        ("Continuum", (1700.0, 2000.0)),
        ("Tl-208 SE", (2070.0, 2135.0)),
        ("Tl-208 FE", (2575.0, 2640.0)),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 6.6), constrained_layout=True)
    for row_index, context_size in enumerate((500, 2000)):
        for column_index, (title, bounds) in enumerate(panels):
            axis = axes[row_index, column_index]
            bin_mask = (
                supported
                & (reference["center"] >= bounds[0])
                & (reference["center"] <= bounds[1])
            )
            grid_mask = (grid_energy >= bounds[0]) & (grid_energy <= bounds[1])
            axis.scatter(
                reference["center"][bin_mask],
                reference["acceptance"][bin_mask],
                s=15,
                color="0.45",
                alpha=0.55,
            )
            for architecture in ARCHITECTURES:
                mean, _ = aggregate_curves[(architecture, context_size)]
                axis.plot(
                    grid_energy[grid_mask],
                    mean[grid_mask],
                    color=COLORS[architecture],
                    linewidth=1.1,
                    label=METHOD_NAMES[architecture],
                )
            axis.set_title(f"{title}\nn={context_size}")
            axis.set_xlim(bounds)
            axis.set_ylim(-0.02, 1.02)
            axis.set_xlabel("Energy (keV)")
            if column_index == 0:
                axis.set_ylabel("Acceptance")
            axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.savefig(outputs["local_figure"], dpi=180)
    plt.close(fig)

    def formatted(architecture: str, size: int, metric: str) -> str:
        row = summary_frame[
            (summary_frame["architecture_id"] == architecture)
            & (summary_frame["context_size"] == size)
        ].iloc[0]
        return (
            f"{row[f'{metric}_mean_of_seed_means']:.3f} ± "
            f"{row[f'{metric}_sd_across_seed_means']:.3f}"
        )

    table_lines = []
    for architecture in ARCHITECTURES:
        table_lines.append(
            "| "
            + " | ".join(
                [
                    METHOD_NAMES[architecture],
                    formatted(architecture, 500, "peak_region_mean_mae_percentage_points"),
                    formatted(
                        architecture, 500, "continuum_region_mean_mae_percentage_points"
                    ),
                    formatted(architecture, 2000, "peak_region_mean_mae_percentage_points"),
                    formatted(
                        architecture, 2000, "continuum_region_mean_mae_percentage_points"
                    ),
                ]
            )
            + " |"
        )
    report = """# Phase 1 neural context-size result

Status: complete. The matrix contains four architectures, three training seeds,
ten overlapping context draws, and four nested context sizes (480 cells). Sixty
compatible 2,000-context cells and one timing-pilot cell were reused; 419 cells
were newly evaluated. The campaign recorded no failed scientific runs.

## Main local-shape result

Values are mean ± SD across the three training-seed means, in percentage
points. Each seed mean averages the same ten context draws.

| Method | Peak MAE, n=500 | Continuum MAE, n=500 | Peak MAE, n=2000 | Continuum MAE, n=2000 |
|---|---:|---:|---:|---:|
""" + "\n".join(table_lines) + """

Peak MAE is the equal average of the prespecified DEP, 1,620.74-keV Bi-212,
SE, and FE regional 5-keV-bin MAEs. Continuum MAE equally averages the
prespecified 1,700--2,000 and 2,200--2,400-keV regions. Full regional values,
RMSE, support, excluded bins, contrast errors, sparse-tail results, and Brier
checks remain in the CSV files at full precision.

## Variation and uncertainty boundary

The context-variation table reports within-training-seed SD and range across
the ten overlapping contexts. Training-seed variation is reported separately
as the SD of three context-averaged seed estimates. These are sensitivity
diagnostics, not independent-dataset standard errors, because contexts overlap
and every cell shares the same 114,400-event target.

The saved 50-pass files contain means and pointwise standard deviations, not
the stochastic function draws required to aggregate uncertainty over a bin.
No binned coverage or interval-width result is therefore invented from these
files. Model posterior, repeated-context, and finite-reference uncertainty are
not treated as interchangeable.

## Interpretation

The reference acceptance is a finite noisy estimate, not exact ground truth.
This is a prospectively specified follow-up on historically exposed data. The
study measures context efficiency conditional on each disclosed pretrained
acceptance model and classifier; it is not evidence of lower total end-to-end
data use. Brier remains a secondary check rather than the main evidence for
local-shape reconstruction. Phase 2 was not started.
"""
    outputs["report"].write_text(report)

    source_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = {
        "schema_version": 1,
        "analysis": "Phase 1 neural context-size study",
        "status": "completed",
        "source_commit": source_commit,
        "script": "ml4phy-paper/scripts/aggregate_extension_neural.py",
        "script_sha256": sha256_file(Path(__file__)),
        "protocol_sha256": sha256_file(protocol_path),
        "model_registry_sha256": sha256_file(registry_path),
        "campaign_record_sha256": sha256_file(campaign_path),
        "cell_count": len(input_rows),
        "new_evaluation_cells": 419,
        "pilot_reuse_cells": 1,
        "archived_reuse_cells": 60,
        "failed_scientific_runs": [],
        "historical_data_exposure": protocol["analysis_status"],
        "interval_limitation": (
            "No binned neural coverage computed: stochastic function draws were not "
            "saved by the 50-pass cell evaluator."
        ),
        "inputs": input_rows,
        "outputs": {
            str(path.relative_to(repo)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for key, path in outputs.items()
            if key != "manifest"
        },
    }
    write_json(outputs["manifest"], manifest)


if __name__ == "__main__":
    main()
