#!/usr/bin/env python3
"""Aggregate the approved Phase 2 acceptance-training-budget study."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BUDGETS = (2000, 5000, 18866)
NEW_BUDGETS = (2000, 5000)
ARCHITECTURES = ("m0", "m1", "m2", "ours")
METHOD_NAMES = {
    "m0": "CNP",
    "m1": "Attentive CNP",
    "m2": "Attentive CNP + PE",
    "ours": "Density-guided CNP (ours)",
}
COLORS = {"m0": "#4c78a8", "m1": "#f58518", "m2": "#54a24b", "ours": "#e45756"}
TRAINING_SEEDS = (0, 1, 2)
CONTEXT_SEEDS = tuple(range(100, 110))
CONTEXT_SIZE = 500
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
ELIGIBLE_COUNTS = {2000: 1895, 5000: 4984, 18866: 18836}


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


def run_directory(
    repo: Path, budget: int, architecture: str, training_seed: int, context_seed: int
) -> Path:
    run_id = (
        f"20260909-phase2-b{budget}-{architecture}-seed{training_seed}-"
        f"final-ctx-s{context_seed}-n500-drop10100-mc50"
    )
    return repo / "ml4phy-paper/runs/phase2/evaluation" / run_id


def hierarchical_summary(cells: pd.DataFrame) -> list[dict]:
    metrics = (
        "peak_region_mean_mae_percentage_points",
        "continuum_region_mean_mae_percentage_points",
        "sparse_tail_mae_percentage_points",
        "global_brier",
        "equal_region_brier",
        "inference_seconds",
        "peak_memory_mib",
    )
    rows = []
    for (budget, architecture), group in cells.groupby(
        ["training_budget", "architecture_id"], sort=False
    ):
        seed_rows = []
        for training_seed, seed_group in group.groupby("training_seed", sort=True):
            if len(seed_group) != len(CONTEXT_SEEDS):
                raise ValueError(f"Incomplete contexts: {budget}, {architecture}, {training_seed}")
            seed_row = {"training_seed": int(training_seed)}
            for metric in metrics:
                values = seed_group[metric].to_numpy(dtype=np.float64)
                seed_row[f"{metric}_mean"] = float(values.mean())
                seed_row[f"{metric}_context_sd"] = float(values.std(ddof=1))
            seed_rows.append(seed_row)
        if len(seed_rows) != len(TRAINING_SEEDS):
            raise ValueError(f"Incomplete training seeds: {budget}, {architecture}")
        row = {
            "method": METHOD_NAMES[architecture],
            "architecture_id": architecture,
            "training_budget_nominal_events": int(budget),
            "training_budget_sampling_eligible_events": ELIGIBLE_COUNTS[int(budget)],
            "context_size": CONTEXT_SIZE,
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
    root = repo / "ml4phy-paper"
    outputs = {
        "cells": root / "tables/phase2_cell_metrics.csv",
        "bins": root / "tables/phase2_bin_error_cells.csv",
        "contrasts": root / "tables/phase2_peak_contrast_cells.csv",
        "summary": root / "tables/phase2_training_budget_summary.csv",
        "curves": root / "tables/phase2_acceptance_curves.csv",
        "efficiency_figure": root / "figures/phase2_training_budget_efficiency.png",
        "tail_figure": root / "figures/phase2_sparse_tail_diagnostic.png",
        "local_figure": root / "figures/phase2_density_guided_local_curves.png",
        "report": root / "reports/phase2_result.md",
        "manifest": root / "manifests/phase2_result.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))

    phase2_protocol_path = root / "manifests/phase2_protocol_v1.json"
    extension_protocol_path = root / "manifests/extension_protocol_v1.json"
    phase1_manifest_path = root / "manifests/phase1_neural_result.json"
    phase1_cells_path = root / "tables/phase1_neural_cell_metrics.csv"
    phase1_bins_path = root / "tables/phase1_neural_bin_error_cells.csv"
    phase1_contrasts_path = root / "tables/phase1_neural_peak_contrast_cells.csv"
    phase1_curves_path = root / "tables/phase1_neural_acceptance_curves.csv"
    training_manifest_path = root / "manifests/phase2_training_result.json"
    campaign_path = root / "runs/campaigns/20260909-phase2-eval-v1/campaign_record.json"
    phase2_protocol = read_json(phase2_protocol_path)
    extension_protocol = read_json(extension_protocol_path)
    phase1_manifest = read_json(phase1_manifest_path)
    training_manifest = read_json(training_manifest_path)
    campaign = read_json(campaign_path)
    if campaign["status"] != "completed" or len(campaign["completed"]) != 240:
        raise ValueError("Phase 2 evaluation campaign is incomplete")
    if campaign["failures"]:
        raise ValueError("Phase 2 evaluation campaign contains failures")
    if training_manifest["status"] != "completed":
        raise ValueError("Phase 2 training campaign is incomplete")
    for path in (phase1_cells_path, phase1_bins_path, phase1_contrasts_path, phase1_curves_path):
        relative = str(path.relative_to(repo))
        if sha256_file(path) != phase1_manifest["outputs"][relative]["sha256"]:
            raise ValueError(f"Phase 1 input hash mismatch: {path}")

    subset_hashes = {
        item["context_seed"]: item["identity_sha256"]
        for item in extension_protocol["context_protocol"]["subsets"]
        if item["phase"] == "final" and item["context_size"] == CONTEXT_SIZE
    }
    target_hash = read_json(root / "manifests/frozen_protocol_v1.json")["roles"][
        "final_target"
    ]["identity_sha256"]
    cell_rows: list[dict] = []
    bin_rows: list[dict] = []
    contrast_rows: list[dict] = []
    input_rows: list[dict] = []
    curve_arrays: dict[tuple[int, str, int], list[np.ndarray]] = {}
    grid_energy: np.ndarray | None = None
    reference: dict[str, np.ndarray] | None = None

    phase1_cells = pd.read_csv(phase1_cells_path)
    reused_cells = phase1_cells[phase1_cells["context_size"] == CONTEXT_SIZE].copy()
    if len(reused_cells) != 120:
        raise ValueError("Expected 120 reusable full-budget Phase 1 cells")
    reused_cells.insert(2, "training_budget", 18866)
    reused_cells["provenance"] = "phase1_full_budget_reuse"
    cell_rows.extend(reused_cells.to_dict("records"))

    for source_path, destination in (
        (phase1_bins_path, bin_rows),
        (phase1_contrasts_path, contrast_rows),
    ):
        frame = pd.read_csv(source_path)
        frame = frame[frame["context_size"] == CONTEXT_SIZE].copy()
        if len(frame) == 0:
            raise ValueError(f"No reusable full-budget rows in {source_path}")
        frame.insert(2, "training_budget", 18866)
        destination.extend(frame.to_dict("records"))

    for budget in NEW_BUDGETS:
        registry_path = root / f"configs/phase2_models_b{budget}_v1.json"
        registry = read_json(registry_path)
        if registry["training_budget"] != budget:
            raise ValueError(f"Registry budget mismatch: {registry_path}")
        for architecture in ARCHITECTURES:
            for training_seed in TRAINING_SEEDS:
                curve_arrays[(budget, architecture, training_seed)] = []
                model_id = f"b{budget}_{architecture}_seed{training_seed}"
                model_spec = registry["models"][model_id]
                if sha256_file(repo / model_spec["checkpoint"]) != model_spec["checkpoint_sha256"]:
                    raise ValueError(f"Checkpoint hash mismatch: {model_id}")
                for context_seed in CONTEXT_SEEDS:
                    directory = run_directory(
                        repo, budget, architecture, training_seed, context_seed
                    )
                    summary_path = directory / "summary.json"
                    event_path = directory / "event_predictions.npz"
                    curve_path = directory / "curve.npz"
                    summary = read_json(summary_path)
                    if (
                        summary["status"] != "completed"
                        or summary["model"]["id"] != model_id
                        or summary["randomness"]["training_seed"] != training_seed
                        or summary["randomness"]["context_seed"] != context_seed
                        or summary["randomness"]["dropout_seed"] != 10100
                        or summary["counts"]["context_draw"] != CONTEXT_SIZE
                        or summary["counts"]["context_per_mc_pass"] != CONTEXT_SIZE
                        or summary["counts"]["target"] != 114400
                        or summary["counts"]["mc_passes"] != 50
                        or summary["protocol"]["target_identity_sha256"] != target_hash
                        or summary["protocol"]["context_draw_identity_sha256"]
                        != subset_hashes[context_seed]
                        or summary["source_worktree_status"]
                    ):
                        raise ValueError(f"Cell contract mismatch: {summary_path}")
                    for path, key in ((event_path, "event_predictions"), (curve_path, "curve")):
                        if sha256_file(path) != summary["server_only_outputs"][key]["sha256"]:
                            raise ValueError(f"Server-only output hash mismatch: {path}")
                    events = {item["region"]: item for item in summary["metrics"]["event"]}
                    bins = {item["region"]: item for item in summary["metrics"]["bin_5kev"]}
                    cell_rows.append(
                        {
                            "method": METHOD_NAMES[architecture],
                            "architecture_id": architecture,
                            "training_budget": budget,
                            "training_seed": training_seed,
                            "context_seed": context_seed,
                            "context_size": CONTEXT_SIZE,
                            "provenance": "phase2_new_evaluation",
                            "peak_region_mean_mae_percentage_points": 100.0
                            * float(np.mean([bins[key]["mae"] for key in PEAK_REGIONS])),
                            "continuum_region_mean_mae_percentage_points": 100.0
                            * float(
                                np.mean([bins[key]["mae"] for key in CONTINUUM_REGIONS])
                            ),
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
                                "training_budget": budget,
                                "training_seed": training_seed,
                                "context_seed": context_seed,
                                "context_size": CONTEXT_SIZE,
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
                                "training_budget": budget,
                                "training_seed": training_seed,
                                "context_seed": context_seed,
                                "context_size": CONTEXT_SIZE,
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
                    elif not np.array_equal(grid_energy, current_grid):
                        raise ValueError(f"Grid mismatch: {curve_path}")
                    else:
                        assert reference is not None
                        for key in reference:
                            if not np.allclose(reference[key], current_reference[key], equal_nan=True):
                                raise ValueError(f"Reference mismatch: {curve_path}")
                    curve_arrays[(budget, architecture, training_seed)].append(current_curve)
                    input_rows.append(
                        {
                            "run_id": summary["run_id"],
                            "summary_sha256": sha256_file(summary_path),
                            "event_predictions_sha256": sha256_file(event_path),
                            "curve_sha256": sha256_file(curve_path),
                        }
                    )

    if len(cell_rows) != 360 or len(input_rows) != 240:
        raise ValueError("Unexpected Phase 2 cell accounting")
    cells = pd.DataFrame(cell_rows)
    summary_rows = hierarchical_summary(cells)
    summary_frame = pd.DataFrame(summary_rows)

    assert grid_energy is not None and reference is not None
    curve_rows: list[dict] = []
    aggregate_curves: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    phase1_curves = pd.read_csv(phase1_curves_path)
    phase1_curves = phase1_curves[phase1_curves["context_size"] == CONTEXT_SIZE]
    for architecture in ARCHITECTURES:
        group = phase1_curves[phase1_curves["architecture_id"] == architecture].sort_values(
            "energy_kev"
        )
        if not np.array_equal(group["energy_kev"].to_numpy(), grid_energy):
            raise ValueError(f"Phase 1 curve grid mismatch: {architecture}")
        aggregate_curves[(18866, architecture)] = (
            group["mean_of_training_seed_context_means"].to_numpy(),
            group["sd_across_training_seed_context_means"].to_numpy(),
        )
    for budget in NEW_BUDGETS:
        for architecture in ARCHITECTURES:
            seed_means = []
            context_sds = []
            for training_seed in TRAINING_SEEDS:
                matrix = np.stack(curve_arrays[(budget, architecture, training_seed)])
                if matrix.shape[0] != len(CONTEXT_SEEDS):
                    raise ValueError("Incomplete curve matrix")
                seed_means.append(matrix.mean(axis=0))
                context_sds.append(matrix.std(axis=0, ddof=1))
            seed_matrix = np.stack(seed_means)
            mean = seed_matrix.mean(axis=0)
            seed_sd = seed_matrix.std(axis=0, ddof=1)
            context_sd = np.stack(context_sds).mean(axis=0)
            aggregate_curves[(budget, architecture)] = (mean, seed_sd)
            for energy, mean_value, seed_value, context_value in zip(
                grid_energy, mean, seed_sd, context_sd, strict=True
            ):
                curve_rows.append(
                    {
                        "method": METHOD_NAMES[architecture],
                        "architecture_id": architecture,
                        "training_budget_nominal_events": budget,
                        "energy_kev": energy,
                        "mean_of_training_seed_context_means": mean_value,
                        "sd_across_training_seed_context_means": seed_value,
                        "mean_context_sd_within_training_seed": context_value,
                        "training_seed_count": 3,
                        "context_draws_per_training_seed": 10,
                        "mc_passes_per_cell": 50,
                    }
                )
    for architecture in ARCHITECTURES:
        group = phase1_curves[phase1_curves["architecture_id"] == architecture].sort_values(
            "energy_kev"
        )
        for energy, mean_value, seed_value, context_value in zip(
            group["energy_kev"],
            group["mean_of_training_seed_context_means"],
            group["sd_across_training_seed_context_means"],
            group["mean_context_sd_within_training_seed"],
            strict=True,
        ):
            curve_rows.append(
                {
                    "method": METHOD_NAMES[architecture],
                    "architecture_id": architecture,
                    "training_budget_nominal_events": 18866,
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
    write_csv(outputs["curves"], curve_rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for axis, metric, title in (
        (axes[0], "peak_region_mean_mae_percentage_points", "Prespecified peak regions"),
        (
            axes[1],
            "continuum_region_mean_mae_percentage_points",
            "Prespecified continuum regions",
        ),
    ):
        for architecture in ARCHITECTURES:
            group = summary_frame[summary_frame["architecture_id"] == architecture].sort_values(
                "training_budget_nominal_events"
            )
            axis.errorbar(
                group["training_budget_nominal_events"],
                group[f"{metric}_mean_of_seed_means"],
                yerr=group[f"{metric}_sd_across_seed_means"],
                marker="o",
                capsize=3,
                color=COLORS[architecture],
                label=METHOD_NAMES[architecture],
            )
        axis.set_xscale("log")
        axis.set_xticks(BUDGETS, labels=[f"{value:,}" for value in BUDGETS])
        axis.set_xlabel("Acceptance-training events (nominal)")
        axis.set_ylabel("5-keV-bin MAE (percentage points)")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(outputs["efficiency_figure"], dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    for architecture in ARCHITECTURES:
        group = summary_frame[summary_frame["architecture_id"] == architecture].sort_values(
            "training_budget_nominal_events"
        )
        axis.errorbar(
            group["training_budget_nominal_events"],
            group["sparse_tail_mae_percentage_points_mean_of_seed_means"],
            yerr=group["sparse_tail_mae_percentage_points_sd_across_seed_means"],
            marker="o",
            capsize=3,
            color=COLORS[architecture],
            label=METHOD_NAMES[architecture],
        )
    axis.set_xscale("log")
    axis.set_xticks(BUDGETS, labels=[f"{value:,}" for value in BUDGETS])
    axis.set_xlabel("Acceptance-training events (nominal)")
    axis.set_ylabel("Sparse-tail 5-keV-bin MAE (percentage points)")
    axis.set_title("Sparse-tail diagnostic (13 valid bins; 47 excluded)")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)
    fig.savefig(outputs["tail_figure"], dpi=180)
    plt.close(fig)

    supported = reference["count"] >= 4
    panels = [
        ("DEP and 1,620.74-keV feature", (1565.0, 1645.0)),
        ("Continuum", (1700.0, 2000.0)),
        ("Tl-208 SE", (2070.0, 2135.0)),
        ("Tl-208 FE", (2575.0, 2640.0)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.7), constrained_layout=True)
    line_styles = {2000: ":", 5000: "--", 18866: "-"}
    for axis, (title, bounds) in zip(axes, panels, strict=True):
        bin_mask = (
            supported
            & (reference["center"] >= bounds[0])
            & (reference["center"] <= bounds[1])
        )
        grid_mask = (grid_energy >= bounds[0]) & (grid_energy <= bounds[1])
        axis.scatter(
            reference["center"][bin_mask],
            reference["acceptance"][bin_mask],
            s=16,
            color="0.45",
            alpha=0.55,
            label="Reference (5-keV bins)",
        )
        for budget in BUDGETS:
            mean, _ = aggregate_curves[(budget, "ours")]
            axis.plot(
                grid_energy[grid_mask],
                mean[grid_mask],
                color=COLORS["ours"],
                linestyle=line_styles[budget],
                linewidth=1.3,
                label=f"Ours, {budget:,} training events",
            )
        axis.set_title(title)
        axis.set_xlim(bounds)
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel("Energy (keV)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Acceptance")
    axes[0].legend(frameon=False, fontsize=7)
    fig.savefig(outputs["local_figure"], dpi=180)
    plt.close(fig)

    def value(architecture: str, budget: int, metric: str) -> tuple[float, float]:
        row = summary_frame[
            (summary_frame["architecture_id"] == architecture)
            & (summary_frame["training_budget_nominal_events"] == budget)
        ].iloc[0]
        return (
            float(row[f"{metric}_mean_of_seed_means"]),
            float(row[f"{metric}_sd_across_seed_means"]),
        )

    table_lines = []
    for budget in BUDGETS:
        for architecture in ARCHITECTURES:
            peak = value(architecture, budget, "peak_region_mean_mae_percentage_points")
            continuum = value(
                architecture, budget, "continuum_region_mean_mae_percentage_points"
            )
            tail = value(architecture, budget, "sparse_tail_mae_percentage_points")
            table_lines.append(
                f"| {budget:,} | {METHOD_NAMES[architecture]} | "
                f"{peak[0]:.2f} ± {peak[1]:.2f} | {continuum[0]:.2f} ± {continuum[1]:.2f} | "
                f"{tail[0]:.2f} ± {tail[1]:.2f} |"
            )

    best_lines = []
    for budget in BUDGETS:
        budget_rows = summary_frame[
            summary_frame["training_budget_nominal_events"] == budget
        ]
        peak_best = budget_rows.loc[
            budget_rows["peak_region_mean_mae_percentage_points_mean_of_seed_means"].idxmin()
        ]
        continuum_best = budget_rows.loc[
            budget_rows[
                "continuum_region_mean_mae_percentage_points_mean_of_seed_means"
            ].idxmin()
        ]
        best_lines.append(
            f"- Among the four neural architectures at {budget:,} nominal events, "
            f"the lowest peak MAE is "
            f"{peak_best['method']} ({peak_best['peak_region_mean_mae_percentage_points_mean_of_seed_means']:.2f} pp); "
            f"the lowest continuum MAE is {continuum_best['method']} "
            f"({continuum_best['continuum_region_mean_mae_percentage_points_mean_of_seed_means']:.2f} pp)."
        )

    report = """# Phase 2 acceptance-training-budget result

Status: complete. All 24 approved 3,000-step training jobs and all 240 new
evaluation cells completed without scientific failure. The analysis combines
those cells with 120 compatible Phase 1 cells at the original 18,866-event
budget. Every cell uses a 500-event context, the same ten overlapping context
draws, 50 MC passes with fixed dropout seed 10100, and the fixed 114,400-event
target.

## Main result

Values are mean ± SD across three training-seed means, in percentage points;
each seed mean averages the same ten context draws.

| Nominal training events | Method | Peak MAE | Continuum MAE | Sparse-tail MAE |
|---:|---|---:|---:|---:|
""" + "\n".join(table_lines) + """

The prespecified peak metric equally averages DEP, the 1,620.74-keV Bi-212
feature, SE, and FE regional 5-keV-bin MAEs. The continuum metric equally
averages the 1,700--2,000 and 2,200--2,400-keV regions.

""" + "\n".join(best_lines) + """

The full-precision tables retain regional RMSE, support, excluded-bin counts,
peak/sideband contrasts, Brier checks, and all unfavorable outcomes. The
simpler CNP variants are more robust at the 2,000-event budget: the
density-guided model degrades to 9.40 pp peak MAE and 12.89 pp continuum MAE.
The density-guided advantage appears at 5,000 and improves further at 18,866,
but one frozen nested pool ordering does not identify a precise transition
budget or establish robustness to alternative training-pool draws.

The
sparse tail remains a separate diagnostic: only 13 target bins meet the
four-event rule and 47 are excluded. The 2,000- and 5,000-event training
subsets contain no sample-eligible sparse-tail event, so no broad all-region
superiority claim is supported.

## Data-efficiency boundary

The nominal acceptance-training budgets retain 1,895, 4,984, and 18,836
sample-eligible events, respectively. The smaller pools are outcome-blind
nested prefixes shared by all methods. For Density-guided CNP, the density
pool shrinks with the nominal pool and does not reconstruct the full pool.
Training uses a fixed 3,000-step schedule with repeated sampling, so this is a
fixed-compute acceptance-stage study rather than a per-method optimum study.

The classifier remains fixed and was trained on 18,866 events selected from
377,330 candidates. Therefore this study can support only acceptance-model
training-data efficiency conditional on the pretrained classifier; it cannot
support a lower total end-to-end training-data claim. The extra 500 context
events and the evaluation, calibration, and development costs remain
separately disclosed in the data-budget ledger.

## Statistical and provenance boundary

The ten contexts overlap and all cells share the same finite, noisy target;
they are not independent datasets. Training-seed SD and within-seed context
variation are kept separate in the summary table. Brier is a secondary check,
and the earlier dropout-coverage diagnostic remains uncalibrated and unsuitable
as a positive headline result. This is a prospectively specified follow-up on
historically exposed data, not an untouched test.

Dense GP was not run in Phase 2. Its separately proposed budget extension still
requires explicit approval; sparse GP and new uncertainty methods remain out
of scope.
"""
    outputs["report"].write_text(report)

    source_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = {
        "schema_version": 1,
        "analysis": "Phase 2 acceptance-training-budget study",
        "status": "completed",
        "source_commit": source_commit,
        "script": "ml4phy-paper/scripts/aggregate_phase2.py",
        "script_sha256": sha256_file(Path(__file__)),
        "phase2_protocol_sha256": sha256_file(phase2_protocol_path),
        "extension_protocol_sha256": sha256_file(extension_protocol_path),
        "phase2_training_manifest_sha256": sha256_file(training_manifest_path),
        "phase1_neural_manifest_sha256": sha256_file(phase1_manifest_path),
        "evaluation_campaign_record_sha256": sha256_file(campaign_path),
        "matrix": {
            "nominal_training_budgets": list(BUDGETS),
            "sampling_eligible_counts": ELIGIBLE_COUNTS,
            "architectures": list(ARCHITECTURES),
            "training_seeds": list(TRAINING_SEEDS),
            "context_seeds": list(CONTEXT_SEEDS),
            "context_size": CONTEXT_SIZE,
            "new_evaluation_cells": 240,
            "reused_full_budget_cells": 120,
            "total_analysis_cells": 360,
        },
        "training_failures": training_manifest["scientific_failures"],
        "evaluation_failures": campaign["failures"],
        "historical_data_exposure": phase2_protocol["historical_data_exposure"],
        "claim_boundary": (
            "Acceptance-model training-data efficiency conditional on the fixed "
            "classifier; not total end-to-end training-data efficiency."
        ),
        "inputs": input_rows,
        "reused_inputs": {
            str(path.relative_to(repo)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (
                phase1_cells_path,
                phase1_bins_path,
                phase1_contrasts_path,
                phase1_curves_path,
            )
        },
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
