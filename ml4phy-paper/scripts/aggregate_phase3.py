#!/usr/bin/env python3
"""Aggregate the approved Phase 3 neural and dense-GP campaigns."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUBSET_IDS = ("seed20260910_n5000", "seed20260911_n5000", "original_n10000")
ARCHITECTURES = ("m0", "m1", "m2", "ours")
TRAINING_SEEDS = (0, 1, 2)
CONTEXT_SEEDS = tuple(range(100, 110))
CONTEXT_SIZE = 500
PEAK_REGIONS = ("DEP", "Bi-214", "SE", "FE")
CONTINUUM_REGIONS = ("continuum_1700_2000", "continuum_2200_2400")
METHOD_NAMES = {
    "m0": "CNP",
    "m1": "Attentive CNP",
    "m2": "Attentive CNP + PE",
    "ours": "Density-guided CNP (ours)",
}
COLORS = {"m0": "#4c78a8", "m1": "#f58518", "m2": "#54a24b", "ours": "#e45756"}
BUDGET_LABELS = {2000: "2k", 5000: "5k", 10000: "10k", 18866: "18.9k"}
ORIGINAL_SUBSET_IDS = {
    2000: "original_n2000",
    5000: "original_n5000",
    10000: "original_n10000",
    18866: "full_n18866",
}
ELIGIBLE = {
    "original_n2000": 1895,
    "original_n5000": 4984,
    "seed20260910_n5000": 4981,
    "seed20260911_n5000": 4980,
    "original_n10000": 9980,
    "full_n18866": 18836,
}


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
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def phase3_run_dir(
    root: Path, subset_id: str, architecture: str, training_seed: int, context_seed: int
) -> Path:
    run_id = (
        f"20260909-phase3-{subset_id}-{architecture}-seed{training_seed}-"
        f"final-ctx-s{context_seed}-n500-drop10100-mc50"
    )
    return root / "runs/phase3/evaluation" / run_id


def summarize_cells(cells: pd.DataFrame) -> list[dict]:
    metrics = (
        "peak_mae_percentage_points",
        "continuum_mae_percentage_points",
        "sparse_tail_mae_percentage_points",
        "global_brier",
        "equal_region_brier",
        "inference_seconds",
        "peak_memory_mib",
    )
    rows = []
    for (subset_id, budget, architecture), group in cells.groupby(
        ["subset_id", "training_budget_nominal_events", "architecture_id"], sort=False
    ):
        seed_rows = []
        for seed, seed_group in group.groupby("training_seed", sort=True):
            if len(seed_group) != 10:
                raise ValueError(f"Incomplete contexts: {subset_id}, {architecture}, {seed}")
            item = {"training_seed": int(seed)}
            for metric in metrics:
                values = seed_group[metric].to_numpy(dtype=np.float64)
                item[f"{metric}_mean"] = float(values.mean())
                item[f"{metric}_context_sd"] = float(values.std(ddof=1))
            seed_rows.append(item)
        if len(seed_rows) != 3:
            raise ValueError(f"Incomplete training seeds: {subset_id}, {architecture}")
        row = {
            "subset_id": subset_id,
            "ordering_id": "original" if subset_id.startswith("original") or subset_id.startswith("full") else subset_id.removesuffix("_n5000"),
            "budget_label": BUDGET_LABELS[int(budget)],
            "training_budget_nominal_events": int(budget),
            "training_budget_sampling_eligible_events": ELIGIBLE[subset_id],
            "method": METHOD_NAMES[architecture],
            "architecture_id": architecture,
            "context_size": CONTEXT_SIZE,
            "training_seed_count": 3,
            "context_draws_per_training_seed": 10,
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
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--short"], text=True
    ).splitlines()
    if status:
        raise RuntimeError("Phase 3 aggregation requires a clean worktree: " + "; ".join(status))
    source_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    outputs = {
        "cells": root / "tables/phase3_cell_metrics.csv",
        "bins": root / "tables/phase3_bin_error_cells.csv",
        "contrasts": root / "tables/phase3_peak_contrast_cells.csv",
        "summary": root / "tables/phase3_training_budget_summary.csv",
        "cross_budget": root / "tables/phase3_cross_budget_comparison.csv",
        "subset": root / "tables/phase3_5k_subset_robustness.csv",
        "method": root / "tables/phase3_method_comparison_context500.csv",
        "gp_development": root / "tables/phase3_gp_development_selection.csv",
        "gp_final": root / "tables/phase3_gp_final_summary.csv",
        "budget_figure": root / "figures/phase3_training_budget_efficiency.png",
        "subset_figure": root / "figures/phase3_subset_robustness.png",
        "gp_figure": root / "figures/phase3_gp_context_efficiency.png",
        "report": root / "reports/phase3_result.md",
        "manifest": root / "manifests/phase3_result.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))

    protocol_path = root / "manifests/phase3_protocol_v1.json"
    training_manifest_path = root / "manifests/phase3_training_result.json"
    phase2_manifest_path = root / "manifests/phase2_result.json"
    phase2_cells_path = root / "tables/phase2_cell_metrics.csv"
    phase1_kernel_path = root / "tables/phase1_kernel_context_size_summary.csv"
    eval_campaign_path = root / "runs/phase3/campaigns/20260909-phase3-neural-evaluation-v1/campaign_record.json"
    gp_campaign_path = root / "runs/campaigns/20260909-phase3-dense-gp-v1/campaign_record.json"
    protocol = read_json(protocol_path)
    training_manifest = read_json(training_manifest_path)
    phase2_manifest = read_json(phase2_manifest_path)
    eval_campaign = read_json(eval_campaign_path)
    gp_campaign = read_json(gp_campaign_path)
    if training_manifest["status"] != "completed" or training_manifest["completed_jobs"] != 36:
        raise ValueError("Phase 3 training result is incomplete")
    if eval_campaign["status"] != "completed" or len(eval_campaign["completed"]) != 360:
        raise ValueError("Phase 3 neural evaluation is incomplete")
    if gp_campaign["status"] != "completed" or len(gp_campaign["completed"]) != 120:
        raise ValueError("Phase 3 dense GP campaign is incomplete")
    if eval_campaign["failures"] or gp_campaign["failures"]:
        raise ValueError("Phase 3 campaign contains a scientific failure")
    phase2_relative = str(phase2_cells_path.relative_to(repo))
    if sha256_file(phase2_cells_path) != phase2_manifest["outputs"][phase2_relative]["sha256"]:
        raise ValueError("Phase 2 cell table hash mismatch")

    subset_specs = {item["subset_id"]: item for item in read_json(root / "configs/phase3_training_v1.json")["subset_inputs"]}
    context_hashes = {
        item["context_seed"]: item["identity_sha256"]
        for item in read_json(root / "manifests/extension_protocol_v1.json")["context_protocol"]["subsets"]
        if item["phase"] == "final" and item["context_size"] == 500
    }
    target_hash = read_json(root / "manifests/frozen_protocol_v1.json")["roles"]["final_target"]["identity_sha256"]
    cell_rows = []
    bin_rows = []
    contrast_rows = []
    neural_inputs = []
    for subset_id in SUBSET_IDS:
        budget = subset_specs[subset_id]["nominal_unique_events"]
        for architecture in ARCHITECTURES:
            for training_seed in TRAINING_SEEDS:
                for context_seed in CONTEXT_SEEDS:
                    directory = phase3_run_dir(root, subset_id, architecture, training_seed, context_seed)
                    summary_path = directory / "summary.json"
                    summary = read_json(summary_path)
                    if (
                        summary["status"] != "completed"
                        or summary["model"]["id"] != f"p3_{subset_id}_{architecture}_seed{training_seed}"
                        or summary["randomness"]["training_seed"] != training_seed
                        or summary["randomness"]["context_seed"] != context_seed
                        or summary["randomness"]["dropout_seed"] != 10100
                        or summary["counts"]["context_draw"] != 500
                        or summary["counts"]["target"] != 114400
                        or summary["counts"]["mc_passes"] != 50
                        or summary["protocol"]["target_identity_sha256"] != target_hash
                        or summary["protocol"]["context_draw_identity_sha256"] != context_hashes[context_seed]
                        or summary["source_worktree_status"]
                    ):
                        raise ValueError(f"Neural cell contract mismatch: {summary_path}")
                    for key, metadata in summary["server_only_outputs"].items():
                        path = directory / metadata["file"]
                        if path.stat().st_size != metadata["bytes"] or sha256_file(path) != metadata["sha256"]:
                            raise ValueError(f"Neural output hash mismatch: {path}")
                    events = {item["region"]: item for item in summary["metrics"]["event"]}
                    bins = {item["region"]: item for item in summary["metrics"]["bin_5kev"]}
                    cell_rows.append({
                        "subset_id": subset_id,
                        "ordering_id": subset_specs[subset_id]["ordering_id"],
                        "training_budget_nominal_events": budget,
                        "training_budget_sampling_eligible_events": subset_specs[subset_id]["sampling_eligible_unique_events"],
                        "method": METHOD_NAMES[architecture],
                        "architecture_id": architecture,
                        "training_seed": training_seed,
                        "context_seed": context_seed,
                        "context_size": 500,
                        "peak_mae_percentage_points": 100.0 * float(np.mean([bins[key]["mae"] for key in PEAK_REGIONS])),
                        "continuum_mae_percentage_points": 100.0 * float(np.mean([bins[key]["mae"] for key in CONTINUUM_REGIONS])),
                        "sparse_tail_mae_percentage_points": 100.0 * bins["sparse_tail"]["mae"],
                        "global_brier": events["full"]["brier"],
                        "equal_region_brier": events["equal_region_mean"]["brier"],
                        "inference_seconds": summary["runtime"]["inference_seconds"],
                        "peak_memory_mib": summary["runtime"]["peak_memory_mib"],
                        "run_id": summary["run_id"],
                    })
                    for region, metric in bins.items():
                        bin_rows.append({
                            "subset_id": subset_id,
                            "training_budget_nominal_events": budget,
                            "method": METHOD_NAMES[architecture],
                            "architecture_id": architecture,
                            "training_seed": training_seed,
                            "context_seed": context_seed,
                            "region_id": region,
                            "valid_bin_count": metric["n_valid_bins"],
                            "excluded_bin_count": metric["n_excluded_bins"],
                            "events_in_valid_bins": metric["n_events_in_valid_bins"],
                            "mae_percentage_points": 100.0 * metric["mae"],
                            "rmse_percentage_points": 100.0 * metric["rmse"],
                        })
                    for metric in summary["metrics"]["peak_sideband_contrast"]:
                        contrast_rows.append({
                            "subset_id": subset_id,
                            "training_budget_nominal_events": budget,
                            "method": METHOD_NAMES[architecture],
                            "architecture_id": architecture,
                            "training_seed": training_seed,
                            "context_seed": context_seed,
                            "feature_id": "feature_1620kev" if metric["peak"] == "Bi-214" else metric["peak"],
                            "feature": "1,620.74-keV feature" if metric["peak"] == "Bi-214" else metric["peak"],
                            "status": metric["status"],
                            "center_event_count": metric["center_events"],
                            "sideband_event_count": metric["sideband_events"],
                            "empirical_contrast_percentage_points": 100.0 * metric["empirical_contrast"],
                            "predicted_contrast_percentage_points": 100.0 * metric["predicted_contrast"],
                            "absolute_contrast_error_percentage_points": 100.0 * metric["absolute_contrast_error"],
                        })
                    neural_inputs.append({
                        "run_id": summary["run_id"],
                        "summary_sha256": sha256_file(summary_path),
                        "server_only_output_hashes": {
                            key: item["sha256"] for key, item in summary["server_only_outputs"].items()
                        },
                    })

    if len(cell_rows) != 360:
        raise ValueError("Unexpected Phase 3 neural cell count")
    new_cells = pd.DataFrame(cell_rows)
    old_cells = pd.read_csv(phase2_cells_path).rename(columns={
        "training_budget": "training_budget_nominal_events",
        "peak_region_mean_mae_percentage_points": "peak_mae_percentage_points",
        "continuum_region_mean_mae_percentage_points": "continuum_mae_percentage_points",
    })
    old_cells["subset_id"] = old_cells["training_budget_nominal_events"].map(ORIGINAL_SUBSET_IDS)
    old_cells["ordering_id"] = "original"
    old_cells["training_budget_sampling_eligible_events"] = old_cells["subset_id"].map(ELIGIBLE)
    common_columns = list(new_cells.columns)
    all_cells = pd.concat([old_cells[common_columns], new_cells], ignore_index=True)
    if len(all_cells) != 720:
        raise ValueError("Unexpected combined neural cell count")
    summary_rows = summarize_cells(all_cells)
    summary = pd.DataFrame(summary_rows)

    nested = summary[summary["subset_id"].isin(ORIGINAL_SUBSET_IDS.values())].copy()
    nested = nested.sort_values(["training_budget_nominal_events", "architecture_id"])
    cross_rows = []
    for _, row in nested.iterrows():
        full = nested[
            (nested["architecture_id"] == row["architecture_id"])
            & (nested["training_budget_nominal_events"] == 18866)
        ].iloc[0]
        cross_rows.append({
            "budget_label": row["budget_label"],
            "training_budget_nominal_events": row["training_budget_nominal_events"],
            "sampling_eligible_unique_events": row["training_budget_sampling_eligible_events"],
            "subset_id": row["subset_id"],
            "method": row["method"],
            "architecture_id": row["architecture_id"],
            "peak_mae_percentage_points": row["peak_mae_percentage_points_mean_of_seed_means"],
            "peak_training_seed_sd": row["peak_mae_percentage_points_sd_across_seed_means"],
            "peak_context_sd": row["peak_mae_percentage_points_mean_context_sd_within_seed"],
            "peak_delta_vs_full_same_method_percentage_points": row["peak_mae_percentage_points_mean_of_seed_means"] - full["peak_mae_percentage_points_mean_of_seed_means"],
            "continuum_mae_percentage_points": row["continuum_mae_percentage_points_mean_of_seed_means"],
            "continuum_training_seed_sd": row["continuum_mae_percentage_points_sd_across_seed_means"],
            "continuum_context_sd": row["continuum_mae_percentage_points_mean_context_sd_within_seed"],
            "continuum_delta_vs_full_same_method_percentage_points": row["continuum_mae_percentage_points_mean_of_seed_means"] - full["continuum_mae_percentage_points_mean_of_seed_means"],
            "sparse_tail_mae_percentage_points": row["sparse_tail_mae_percentage_points_mean_of_seed_means"],
            "global_brier": row["global_brier_mean_of_seed_means"],
        })

    five_k = summary[summary["training_budget_nominal_events"] == 5000].copy()
    subset_rows = []
    for _, row in five_k.iterrows():
        subset_group = five_k[five_k["architecture_id"] == row["architecture_id"]]
        full = nested[
            (nested["architecture_id"] == row["architecture_id"])
            & (nested["training_budget_nominal_events"] == 18866)
        ].iloc[0]
        subset_rows.append({
            "subset_id": row["subset_id"],
            "ordering_id": row["ordering_id"],
            "nominal_events": 5000,
            "sampling_eligible_events": row["training_budget_sampling_eligible_events"],
            "method": row["method"],
            "architecture_id": row["architecture_id"],
            "peak_mae_percentage_points": row["peak_mae_percentage_points_mean_of_seed_means"],
            "peak_initialization_sd": row["peak_mae_percentage_points_sd_across_seed_means"],
            "peak_context_sd": row["peak_mae_percentage_points_mean_context_sd_within_seed"],
            "peak_between_subset_sd": subset_group["peak_mae_percentage_points_mean_of_seed_means"].std(ddof=1),
            "peak_delta_vs_full_same_method_percentage_points": row["peak_mae_percentage_points_mean_of_seed_means"] - full["peak_mae_percentage_points_mean_of_seed_means"],
            "continuum_mae_percentage_points": row["continuum_mae_percentage_points_mean_of_seed_means"],
            "continuum_initialization_sd": row["continuum_mae_percentage_points_sd_across_seed_means"],
            "continuum_context_sd": row["continuum_mae_percentage_points_mean_context_sd_within_seed"],
            "continuum_between_subset_sd": subset_group["continuum_mae_percentage_points_mean_of_seed_means"].std(ddof=1),
            "continuum_delta_vs_full_same_method_percentage_points": row["continuum_mae_percentage_points_mean_of_seed_means"] - full["continuum_mae_percentage_points_mean_of_seed_means"],
            "sparse_tail_mae_percentage_points": row["sparse_tail_mae_percentage_points_mean_of_seed_means"],
        })

    gp_completed = {item["run_id"]: item for item in gp_campaign["completed"]}
    gp_inputs = []
    gp_development_rows = gp_campaign["family_selection"]["family_scores"]
    gp_final_cells = []
    for item in gp_campaign["completed"]:
        directory = root / "runs/gp/phase3" / item["run_id"]
        summary_path = directory / "summary.json"
        gp = read_json(summary_path)
        if sha256_file(summary_path) != item["summary_sha256"]:
            raise ValueError(f"GP summary hash mismatch: {summary_path}")
        for metadata in gp["server_only_outputs"].values():
            path = directory / metadata["file"]
            if sha256_file(path) != metadata["sha256"]:
                raise ValueError(f"GP output hash mismatch: {path}")
        gp_inputs.append({
            "run_id": item["run_id"],
            "summary_sha256": item["summary_sha256"],
            "server_only_output_hashes": {
                key: value["sha256"] for key, value in gp["server_only_outputs"].items()
            },
        })
        if gp["phase"] != "final":
            continue
        events = {entry["region"]: entry for entry in gp["metrics"]["event"]}
        bins = {entry["region"]: entry for entry in gp["metrics"]["bin_5kev"]}
        gp_final_cells.append({
            "context_size": gp["context"]["size"],
            "context_seed": gp["context"]["seed"],
            "family": gp["estimator"]["family"],
            "peak_mae_percentage_points": 100.0 * float(np.mean([bins[key]["mae"] for key in PEAK_REGIONS])),
            "continuum_mae_percentage_points": 100.0 * float(np.mean([bins[key]["mae"] for key in CONTINUUM_REGIONS])),
            "sparse_tail_mae_percentage_points": 100.0 * bins["sparse_tail"]["mae"],
            "global_brier": events["full"]["brier"],
            "equal_region_brier": events["equal_region_mean"]["brier"],
            "fit_seconds": gp["runtime"]["fit_seconds"],
            "target_prediction_seconds": gp["runtime"]["target_prediction_seconds"],
            "peak_rss_mib": gp["runtime"]["peak_rss_mib"],
            "warning_count": len(gp["warnings"]),
        })
    if len(gp_final_cells) != 40:
        raise ValueError("Unexpected GP final cell count")
    gp_frame = pd.DataFrame(gp_final_cells)
    gp_rows = []
    for size, group in gp_frame.groupby("context_size", sort=True):
        row = {"context_size": int(size), "context_draws": len(group), "family": group["family"].iloc[0]}
        for metric in (
            "peak_mae_percentage_points", "continuum_mae_percentage_points",
            "sparse_tail_mae_percentage_points", "global_brier", "equal_region_brier",
            "fit_seconds", "target_prediction_seconds", "peak_rss_mib",
        ):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_context_sd"] = float(group[metric].std(ddof=1))
        row["total_warning_count"] = int(group["warning_count"].sum())
        gp_rows.append(row)
    gp_summary = pd.DataFrame(gp_rows)

    kernels = pd.read_csv(phase1_kernel_path)
    kernel500 = kernels[kernels["context_size"] == 500]
    method_rows = []
    for _, row in nested[nested["training_budget_nominal_events"] == 18866].iterrows():
        method_rows.append({
            "method": row["method"],
            "estimator_family": "pretrained neural",
            "context_size": 500,
            "acceptance_pretraining_nominal_events": 18866,
            "peak_mae_percentage_points": row["peak_mae_percentage_points_mean_of_seed_means"],
            "peak_variation": row["peak_mae_percentage_points_sd_across_seed_means"],
            "continuum_mae_percentage_points": row["continuum_mae_percentage_points_mean_of_seed_means"],
            "continuum_variation": row["continuum_mae_percentage_points_sd_across_seed_means"],
            "sparse_tail_mae_percentage_points": row["sparse_tail_mae_percentage_points_mean_of_seed_means"],
            "global_brier": row["global_brier_mean_of_seed_means"],
            "variation_definition": "SD across three initialization-seed means",
        })
    for _, row in kernel500.iterrows():
        method_rows.append({
            "method": row["method"],
            "estimator_family": "kernel regression / KDE ratio",
            "context_size": 500,
            "acceptance_pretraining_nominal_events": int(row["pretraining_pool_events"]),
            "peak_mae_percentage_points": row["peak_region_mean_mae_percentage_points_mean"],
            "peak_variation": row["peak_region_mean_mae_percentage_points_context_sd"],
            "continuum_mae_percentage_points": row["continuum_region_mean_mae_percentage_points_mean"],
            "continuum_variation": row["continuum_region_mean_mae_percentage_points_context_sd"],
            "sparse_tail_mae_percentage_points": row["sparse_tail_mae_percentage_points_mean"],
            "global_brier": row["global_brier_mean"],
            "variation_definition": "SD across ten overlapping contexts",
        })
    gp500 = gp_summary[gp_summary["context_size"] == 500].iloc[0]
    method_rows.append({
        "method": "Gaussian-process probability estimator",
        "estimator_family": "context-only dense Bernoulli GP",
        "context_size": 500,
        "acceptance_pretraining_nominal_events": 0,
        "peak_mae_percentage_points": gp500["peak_mae_percentage_points_mean"],
        "peak_variation": gp500["peak_mae_percentage_points_context_sd"],
        "continuum_mae_percentage_points": gp500["continuum_mae_percentage_points_mean"],
        "continuum_variation": gp500["continuum_mae_percentage_points_context_sd"],
        "sparse_tail_mae_percentage_points": gp500["sparse_tail_mae_percentage_points_mean"],
        "global_brier": gp500["global_brier_mean"],
        "variation_definition": "SD across ten overlapping contexts",
    })

    write_csv(outputs["cells"], cell_rows)
    write_csv(outputs["bins"], bin_rows)
    write_csv(outputs["contrasts"], contrast_rows)
    write_csv(outputs["summary"], summary_rows)
    write_csv(outputs["cross_budget"], cross_rows)
    write_csv(outputs["subset"], subset_rows)
    write_csv(outputs["method"], method_rows)
    write_csv(outputs["gp_development"], gp_development_rows)
    write_csv(outputs["gp_final"], gp_rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    for axis, metric, title in (
        (axes[0], "peak_mae_percentage_points", "Prespecified peak regions"),
        (axes[1], "continuum_mae_percentage_points", "Prespecified continuum regions"),
    ):
        for architecture in ARCHITECTURES:
            group = nested[nested["architecture_id"] == architecture].sort_values("training_budget_nominal_events")
            axis.errorbar(
                np.arange(4), group[f"{metric}_mean_of_seed_means"],
                yerr=group[f"{metric}_sd_across_seed_means"], marker="o", capsize=3,
                color=COLORS[architecture], label=METHOD_NAMES[architecture],
            )
        axis.set_xticks(np.arange(4), ["2k", "5k", "10k", "18.9k"])
        axis.set_xlabel("Acceptance-training budget (nominal events)")
        axis.set_ylabel("5-keV-bin MAE (percentage points)")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(outputs["budget_figure"], dpi=180)
    plt.close(fig)

    subset_order = ("original_n5000", "seed20260910_n5000", "seed20260911_n5000")
    subset_labels = ("Original", "Seed 20260910", "Seed 20260911")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    for axis, metric, title in (
        (axes[0], "peak_mae_percentage_points", "Peak-region error at 5k"),
        (axes[1], "continuum_mae_percentage_points", "Continuum error at 5k"),
    ):
        error_column = (
            "peak_initialization_sd"
            if metric.startswith("peak_")
            else "continuum_initialization_sd"
        )
        for architecture in ARCHITECTURES:
            rows = pd.DataFrame(subset_rows)
            group = rows[rows["architecture_id"] == architecture].set_index("subset_id").loc[list(subset_order)]
            axis.errorbar(
                np.arange(3), group[metric], yerr=group[error_column],
                marker="o", capsize=3, color=COLORS[architecture], label=METHOD_NAMES[architecture],
            )
            full = nested[(nested["architecture_id"] == architecture) & (nested["training_budget_nominal_events"] == 18866)].iloc[0]
            axis.axhline(full[f"{metric}_mean_of_seed_means"], color=COLORS[architecture], alpha=0.22, linewidth=1)
        axis.set_xticks(np.arange(3), subset_labels, rotation=12)
        axis.set_ylabel("5-keV-bin MAE (percentage points)")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(outputs["subset_figure"], dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=True)
    for axis, metric, title in (
        (axes[0], "peak_mae_percentage_points", "Dense GP: peak regions"),
        (axes[1], "continuum_mae_percentage_points", "Dense GP: continuum regions"),
    ):
        axis.errorbar(
            gp_summary["context_size"], gp_summary[f"{metric}_mean"],
            yerr=gp_summary[f"{metric}_context_sd"], marker="o", capsize=3, color="#7a5195",
        )
        axis.set_xscale("log", base=2)
        axis.set_xticks([250, 500, 1000, 2000], ["250", "500", "1,000", "2,000"])
        axis.set_xlabel("Context events")
        axis.set_ylabel("5-keV-bin MAE (percentage points)")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    fig.savefig(outputs["gp_figure"], dpi=180)
    plt.close(fig)

    def nested_value(budget: int, architecture: str, metric: str) -> float:
        row = nested[
            (nested["training_budget_nominal_events"] == budget)
            & (nested["architecture_id"] == architecture)
        ].iloc[0]
        return float(row[f"{metric}_mean_of_seed_means"])

    main_lines = []
    for budget in (2000, 5000, 10000, 18866):
        values = []
        for architecture in ARCHITECTURES:
            values.append(
                f"{nested_value(budget, architecture, 'peak_mae_percentage_points'):.2f}/"
                f"{nested_value(budget, architecture, 'continuum_mae_percentage_points'):.2f}"
            )
        main_lines.append(
            f"| {BUDGET_LABELS[budget]} ({budget:,}) | "
            + " | ".join(values)
            + " |"
        )
    ours_5k = pd.DataFrame(subset_rows)
    ours_5k = ours_5k[ours_5k["architecture_id"] == "ours"].set_index("subset_id").loc[list(subset_order)]
    gp500_peak = float(gp500["peak_mae_percentage_points_mean"])
    gp500_cont = float(gp500["continuum_mae_percentage_points_mean"])
    report = f"""# Phase 3 data-efficiency confirmation

Status: complete. The dense Bernoulli-GP campaign completed 120/120 cells, the neural campaign completed 36/36 training jobs and 360/360 evaluations, and no scientific run failed. Dense GP used {gp_campaign['accumulated_active_seconds']:.2f} seconds ({gp_campaign['accumulated_active_seconds'] / 60:.2f} minutes) under its four-hour cap. Neural training plus evaluation used {eval_campaign['total_neural_active_seconds']:.2f} seconds ({eval_campaign['total_neural_active_seconds'] / 60:.2f} minutes) under the separate two-hour cap.

## Nested acceptance-training budget result

Values below are peak/continuum MAE in percentage points, averaged hierarchically over three initialization seeds and the same ten overlapping 500-event contexts. The 2k, 5k, and 10k rows use nested prefixes of the original outcome-blind ordering; 18.9k is the exact 18,866-event full pool.

| Budget | CNP | Attentive CNP | Attentive CNP + PE | Density-guided CNP (ours) |
|---|---:|---:|---:|---:|
{chr(10).join(main_lines)}

The 2k degradation remains visible and no method is uniformly best across every region and budget. Sparse-tail results are retained in the CSV tables; the small training subsets have no sampling-eligible sparse-tail events, so peak-region findings must not be generalized to the tail.

## Five-thousand-event subset robustness

For Density-guided CNP, peak/continuum MAE across the original, seed-20260910, and seed-20260911 orderings is:

- Original: {ours_5k.iloc[0]['peak_mae_percentage_points']:.2f}/{ours_5k.iloc[0]['continuum_mae_percentage_points']:.2f} pp.
- Seed 20260910: {ours_5k.iloc[1]['peak_mae_percentage_points']:.2f}/{ours_5k.iloc[1]['continuum_mae_percentage_points']:.2f} pp.
- Seed 20260911: {ours_5k.iloc[2]['peak_mae_percentage_points']:.2f}/{ours_5k.iloc[2]['continuum_mae_percentage_points']:.2f} pp.

These are three overlapping random subsets of one finite parent pool, not independent datasets. The tables separate initialization-seed SD, mean within-seed context SD, and between-subset SD. They do not identify an exact minimum sample requirement or justify interpolation between budgets.

## Dense GP and method boundary

Development-only selection chose ConstantKernel × Matern(ν=1.5): mean global development Brier was {gp_campaign['family_selection']['family_scores'][1]['mean_global_brier']:.6f}, versus {gp_campaign['family_selection']['family_scores'][0]['mean_global_brier']:.6f} for RBF. At 500 context events the selected context-only GP has {gp500_peak:.2f} pp peak MAE and {gp500_cont:.2f} pp continuum MAE. It uses no acceptance pretraining, whereas the neural models are conditional on acceptance-model pretraining; this is not an equal-total-information comparison.

The compact method table includes the full-pool neural models, context-only and pooled-data kernel controls, and selected GP. Brier remains secondary. No dropout interval is presented as calibrated, and no MC smoothness claim is made.

## Claim boundary

The result tests acceptance-model training-data efficiency conditional on the classifier pretrained with 18,866 selected events from 377,330 candidates. It does not support end-to-end training on 5k events. All targets and contexts were historically exposed; this is a prospectively specified follow-up analysis, not an untouched test. The 1,620.74-keV structure is named by energy because its isotope identity remains unresolved in the audited sources.
"""
    outputs["report"].write_text(report)

    portable = [path for key, path in outputs.items() if key != "manifest"]
    write_json(outputs["manifest"], {
        "schema_version": 1,
        "analysis": "Phase 3 data-efficiency confirmation and dense GP completion",
        "status": "completed",
        "source_commit": source_commit,
        "script": str(Path(__file__).resolve().relative_to(repo)),
        "script_sha256": sha256_file(Path(__file__)),
        "phase3_protocol_sha256": sha256_file(protocol_path),
        "phase3_training_manifest_sha256": sha256_file(training_manifest_path),
        "phase2_result_manifest_sha256": sha256_file(phase2_manifest_path),
        "neural_evaluation_campaign_record_sha256": sha256_file(eval_campaign_path),
        "dense_gp_campaign_record_sha256": sha256_file(gp_campaign_path),
        "completion": {
            "dense_gp_development_cells": 80,
            "dense_gp_final_cells": 40,
            "neural_training_jobs": 36,
            "neural_evaluation_cells": 360,
            "scientific_failures": [],
        },
        "runtime": {
            "dense_gp_active_seconds": gp_campaign["accumulated_active_seconds"],
            "neural_training_active_seconds": training_manifest["accumulated_active_seconds"],
            "neural_evaluation_active_seconds": eval_campaign["accumulated_active_seconds"],
            "total_neural_active_seconds": eval_campaign["total_neural_active_seconds"],
            "maximum_neural_inference_memory_mib": max(item["peak_memory_mib"] for item in eval_campaign["completed"]),
        },
        "dense_gp_selection": gp_campaign["family_selection"],
        "matrix": {
            "budget_labels": ["2k", "5k", "10k", "18.9k"],
            "exact_nominal_budgets": [2000, 5000, 10000, 18866],
            "five_k_subset_ids": list(subset_order),
            "architectures": list(ARCHITECTURES),
            "training_seeds": list(TRAINING_SEEDS),
            "context_seeds": list(CONTEXT_SEEDS),
            "context_size": 500,
        },
        "historical_data_exposure": protocol["historical_data_exposure"],
        "claim_boundary": "Acceptance-model training-data efficiency conditional on the fixed pretrained classifier; not end-to-end data efficiency.",
        "neural_inputs": neural_inputs,
        "dense_gp_inputs": gp_inputs,
        "outputs": {
            str(path.relative_to(repo)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in portable
        },
    })


if __name__ == "__main__":
    main()
