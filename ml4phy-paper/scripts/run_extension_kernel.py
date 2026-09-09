#!/usr/bin/env python3
"""Run context-only and pooled-data Gaussian-kernel Phase 1 controls."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aggregate_development_baselines import BANDWIDTHS_KEV, CV_FOLDS, CV_SEED, nw_predict
from evaluate_fixed_protocol import bin_metrics, event_metrics, peak_contrasts

VARIANTS = ("context_only", "pooled_data")
VARIANT_NAMES = {
    "context_only": "Gaussian kernel regression (context only)",
    "pooled_data": "Gaussian kernel regression (pooled data + context)",
}
CONTEXT_SEEDS = tuple(range(100, 110))
CONTEXT_SIZES = (250, 500, 1000, 2000)
PEAK_REGIONS = ("DEP", "Bi-214", "SE", "FE")
CONTINUUM_REGIONS = ("continuum_1700_2000", "continuum_2200_2400")


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


def select_bandwidth(
    context_energy: np.ndarray,
    context_outcome: np.ndarray,
    pool_energy: np.ndarray | None = None,
    pool_outcome: np.ndarray | None = None,
) -> tuple[float, list[dict]]:
    rng = np.random.default_rng(CV_SEED)
    fold_id = np.empty(context_energy.size, dtype=np.int64)
    fold_id[rng.permutation(context_energy.size)] = np.arange(context_energy.size) % CV_FOLDS
    rows = []
    for bandwidth in BANDWIDTHS_KEV:
        scores = []
        for fold in range(CV_FOLDS):
            validation = fold_id == fold
            train_energy = context_energy[~validation]
            train_outcome = context_outcome[~validation]
            if pool_energy is not None and pool_outcome is not None:
                train_energy = np.concatenate([pool_energy, train_energy])
                train_outcome = np.concatenate([pool_outcome, train_outcome])
            prediction, _ = nw_predict(
                train_energy,
                train_outcome,
                context_energy[validation],
                float(bandwidth),
            )
            scores.append(float(np.mean((prediction - context_outcome[validation]) ** 2)))
        rows.append(
            {
                "bandwidth_kev": float(bandwidth),
                "mean_cv_brier": float(np.mean(scores)),
                "sd_cv_brier_across_folds": float(np.std(scores, ddof=1)),
            }
        )
    winner = min(rows, key=lambda row: (row["mean_cv_brier"], row["bandwidth_kev"]))
    return float(winner["bandwidth_kev"]), rows


def gaussian_components(
    train_energy: np.ndarray,
    train_outcome: np.ndarray,
    query_energy: np.ndarray,
    bandwidth_kev: float,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    numerator = np.empty(query_energy.size, dtype=np.float64)
    denominator = np.empty(query_energy.size, dtype=np.float64)
    for start in range(0, query_energy.size, chunk_size):
        stop = min(start + chunk_size, query_energy.size)
        scaled = (query_energy[start:stop, None] - train_energy[None, :]) / bandwidth_kev
        weight = np.exp(-0.5 * scaled**2)
        numerator[start:stop] = weight @ train_outcome
        denominator[start:stop] = weight.sum(axis=1)
    if np.any(denominator == 0):
        raise FloatingPointError("Pooled Gaussian-kernel denominator underflowed to zero")
    return numerator, denominator


def main() -> None:
    repo = Path(".").resolve()
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--short"], text=True
    ).splitlines()
    if status:
        raise RuntimeError("Kernel run requires a clean worktree: " + "; ".join(status))
    source_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    output_root = repo / "ml4phy-paper"
    outputs = {
        "bandwidths": output_root / "tables/phase1_kernel_bandwidth_selection.csv",
        "cells": output_root / "tables/phase1_kernel_cell_metrics.csv",
        "bins": output_root / "tables/phase1_kernel_bin_error_cells.csv",
        "contrasts": output_root / "tables/phase1_kernel_peak_contrast_cells.csv",
        "summary": output_root / "tables/phase1_kernel_context_size_summary.csv",
        "curves": output_root / "tables/phase1_kernel_acceptance_curves.csv",
        "figure": output_root / "figures/phase1_kernel_context_efficiency.png",
        "report": output_root / "reports/phase1_kernel_result.md",
        "manifest": output_root / "manifests/phase1_kernel_result.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))
    server_output = output_root / "runs/kernel/20260909-phase1-kernel-v1/predictions.npz"
    server_output.parent.mkdir(parents=True, exist_ok=False)

    extension_path = output_root / "manifests/extension_protocol_v1.json"
    base_path = output_root / "manifests/frozen_protocol_v1.json"
    context_archive_path = output_root / "local/protocol/extension_context_roles_v1.npz"
    base_archive_path = output_root / "local/protocol/frozen_roles_v1.npz"
    train_path = repo / "runs/small_data_configs/simple_cnn_small/eval_train/predictions.h5"
    archived_development_path = output_root / "tables/development_baseline_metrics.json"
    extension = read_json(extension_path)
    base = read_json(base_path)
    archived_development = read_json(archived_development_path)
    if sha256_file(context_archive_path) != extension["context_protocol"]["local_archive"][
        "sha256"
    ]:
        raise ValueError("Extension context archive hash mismatch")
    if sha256_file(base_archive_path) != base["local_role_archive"]["sha256"]:
        raise ValueError("Base role archive hash mismatch")
    if sha256_file(train_path) != "ea2eb5594acdd181ebc18926fd8deabc7b2f6910b117dec863d87c3b70057159":
        raise ValueError("Acceptance input prediction hash mismatch")

    development_h5 = repo / base["inputs"]["historical_development"]["logical_path"]
    final_h5 = repo / base["inputs"]["full_test"]["logical_path"]
    with np.load(base_archive_path) as archive:
        final_target_rows = archive["final_target_rows"].astype(np.int64)
        threshold = float(archive["fixed_threshold"])
    with h5py.File(train_path, "r") as handle:
        pool_energy = handle["energy"][:].astype(np.float64)
        pool_outcome = (handle["score"][:].astype(np.float64) >= threshold).astype(np.float64)
    if pool_energy.size != 18866:
        raise ValueError("Unexpected pooled-data event count")
    with h5py.File(development_h5, "r") as handle:
        development_contexts = {}
        with np.load(context_archive_path) as archive:
            for context_size in CONTEXT_SIZES:
                for context_seed in CONTEXT_SEEDS:
                    key = f"development_context_s{context_seed}_n{context_size}_rows"
                    rows = np.sort(archive[key].astype(np.int64))
                    development_contexts[(context_size, context_seed)] = (
                        handle["energy"][rows].astype(np.float64),
                        (handle["score"][rows].astype(np.float64) >= threshold).astype(np.float64),
                    )
    with h5py.File(final_h5, "r") as handle:
        final_energy = handle["energy"][final_target_rows].astype(np.float64)
        final_outcome = (
            handle["score"][final_target_rows].astype(np.float64) >= threshold
        ).astype(np.float64)
        final_contexts = {}
        with np.load(context_archive_path) as archive:
            for context_size in CONTEXT_SIZES:
                for context_seed in CONTEXT_SEEDS:
                    key = f"final_context_s{context_seed}_n{context_size}_rows"
                    rows = np.sort(archive[key].astype(np.int64))
                    final_contexts[(context_size, context_seed)] = (
                        handle["energy"][rows].astype(np.float64),
                        (handle["score"][rows].astype(np.float64) >= threshold).astype(np.float64),
                    )

    start = time.perf_counter()
    bandwidth_rows = []
    selected: dict[tuple[str, int, int], float] = {}
    for variant in VARIANTS:
        for context_size in CONTEXT_SIZES:
            for context_seed in CONTEXT_SEEDS:
                context_energy, context_outcome = development_contexts[
                    (context_size, context_seed)
                ]
                bandwidth, candidate_rows = select_bandwidth(
                    context_energy,
                    context_outcome,
                    pool_energy if variant == "pooled_data" else None,
                    pool_outcome if variant == "pooled_data" else None,
                )
                selected[(variant, context_size, context_seed)] = bandwidth
                for row in candidate_rows:
                    bandwidth_rows.append(
                        {
                            "variant": variant,
                            "method": VARIANT_NAMES[variant],
                            "context_size": context_size,
                            "context_seed": context_seed,
                            **row,
                            "selected": row["bandwidth_kev"] == bandwidth,
                            "selection_data": "five-fold CV within development context",
                            "cv_seed": CV_SEED,
                        }
                    )

    archived_bandwidths = {
        int(seed): float(value)
        for seed, value in archived_development["kernel"][
            "selected_bandwidth_kev_by_context_seed"
        ].items()
    }
    observed_archived = {
        seed: selected[("context_only", 2000, seed)] for seed in CONTEXT_SEEDS
    }
    if observed_archived != archived_bandwidths:
        raise ValueError(
            f"Original context-only bandwidth selections were not reproduced: "
            f"{observed_archived} != {archived_bandwidths}"
        )

    grid_energy = np.arange(500.0, 3000.0 + 0.5, 1.0)
    query_energy = np.concatenate([final_energy, grid_energy])
    pooled_components = {}
    for bandwidth in sorted(
        {selected[("pooled_data", size, seed)] for size in CONTEXT_SIZES for seed in CONTEXT_SEEDS}
    ):
        pooled_components[bandwidth] = gaussian_components(
            pool_energy, pool_outcome, query_energy, bandwidth
        )

    prediction_arrays = np.empty(
        (len(VARIANTS), len(CONTEXT_SIZES), len(CONTEXT_SEEDS), final_energy.size),
        dtype=np.float32,
    )
    curve_arrays = np.empty(
        (len(VARIANTS), len(CONTEXT_SIZES), len(CONTEXT_SEEDS), grid_energy.size),
        dtype=np.float32,
    )
    cell_rows = []
    bin_rows = []
    contrast_rows = []
    for variant_index, variant in enumerate(VARIANTS):
        for size_index, context_size in enumerate(CONTEXT_SIZES):
            for seed_index, context_seed in enumerate(CONTEXT_SEEDS):
                context_energy, context_outcome = final_contexts[(context_size, context_seed)]
                bandwidth = selected[(variant, context_size, context_seed)]
                if variant == "context_only":
                    prediction, effective_n = nw_predict(
                        context_energy, context_outcome, query_energy, bandwidth
                    )
                else:
                    pool_numerator, pool_denominator = pooled_components[bandwidth]
                    context_numerator, context_denominator = gaussian_components(
                        context_energy, context_outcome, query_energy, bandwidth
                    )
                    numerator = pool_numerator + context_numerator
                    denominator = pool_denominator + context_denominator
                    prediction = numerator / denominator
                    effective_n = np.full(prediction.size, np.nan)
                    if context_size == 250 and context_seed == 100:
                        check, _ = nw_predict(
                            np.concatenate([pool_energy, context_energy]),
                            np.concatenate([pool_outcome, context_outcome]),
                            query_energy[::1000],
                            bandwidth,
                        )
                        if not np.allclose(check, prediction[::1000], rtol=1e-12, atol=1e-12):
                            raise ValueError("Additive pooled-kernel implementation mismatch")
                target_prediction = prediction[: final_energy.size]
                curve_prediction = prediction[final_energy.size :]
                prediction_arrays[variant_index, size_index, seed_index] = target_prediction
                curve_arrays[variant_index, size_index, seed_index] = curve_prediction
                events = {
                    item["region"]: item
                    for item in event_metrics(final_energy, target_prediction, final_outcome)
                }
                bins, _ = bin_metrics(final_energy, target_prediction, final_outcome)
                bins_by_region = {item["region"]: item for item in bins}
                contrasts = peak_contrasts(final_energy, target_prediction, final_outcome)
                cell_rows.append(
                    {
                        "variant": variant,
                        "method": VARIANT_NAMES[variant],
                        "context_size": context_size,
                        "context_seed": context_seed,
                        "selected_bandwidth_kev": bandwidth,
                        "nominal_pretraining_pool_events": 0
                        if variant == "context_only"
                        else 18866,
                        "context_events": context_size,
                        "peak_region_mean_mae_percentage_points": 100.0
                        * float(np.mean([bins_by_region[key]["mae"] for key in PEAK_REGIONS])),
                        "continuum_region_mean_mae_percentage_points": 100.0
                        * float(
                            np.mean([bins_by_region[key]["mae"] for key in CONTINUUM_REGIONS])
                        ),
                        "sparse_tail_mae_percentage_points": 100.0
                        * bins_by_region["sparse_tail"]["mae"],
                        "global_brier": events["full"]["brier"],
                        "equal_region_brier": events["equal_region_mean"]["brier"],
                        "minimum_effective_n": float(np.nanmin(effective_n))
                        if variant == "context_only"
                        else None,
                    }
                )
                for metric in bins:
                    bin_rows.append(
                        {
                            "variant": variant,
                            "method": VARIANT_NAMES[variant],
                            "context_size": context_size,
                            "context_seed": context_seed,
                            "region": metric["region"],
                            "valid_bin_count": metric["n_valid_bins"],
                            "excluded_bin_count": metric["n_excluded_bins"],
                            "events_in_valid_bins": metric["n_events_in_valid_bins"],
                            "mae_percentage_points": 100.0 * metric["mae"],
                            "rmse_percentage_points": 100.0 * metric["rmse"],
                        }
                    )
                for metric in contrasts:
                    contrast_rows.append(
                        {
                            "variant": variant,
                            "method": VARIANT_NAMES[variant],
                            "context_size": context_size,
                            "context_seed": context_seed,
                            "feature": "Bi-212" if metric["peak"] == "Bi-214" else metric["peak"],
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

    np.savez(
        server_output,
        variants=np.asarray(VARIANTS),
        context_sizes=np.asarray(CONTEXT_SIZES),
        context_seeds=np.asarray(CONTEXT_SEEDS),
        final_target_rows=final_target_rows,
        final_energy_kev=final_energy,
        final_outcome=final_outcome.astype(np.int8),
        predictions=prediction_arrays,
        grid_energy_kev=grid_energy,
        curves=curve_arrays,
    )
    cells = pd.DataFrame(cell_rows)
    summary_rows = []
    for (variant, context_size), group in cells.groupby(["variant", "context_size"]):
        row = {
            "variant": variant,
            "method": VARIANT_NAMES[variant],
            "context_size": int(context_size),
            "overlapping_context_draw_count": len(group),
            "pretraining_pool_events": 0 if variant == "context_only" else 18866,
        }
        for metric in (
            "peak_region_mean_mae_percentage_points",
            "continuum_region_mean_mae_percentage_points",
            "sparse_tail_mae_percentage_points",
            "global_brier",
            "equal_region_brier",
        ):
            values = group[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_context_sd"] = float(values.std(ddof=1))
        summary_rows.append(row)
    summary_frame = pd.DataFrame(summary_rows)
    curve_rows = []
    for variant_index, variant in enumerate(VARIANTS):
        for context_size in (500, 2000):
            size_index = CONTEXT_SIZES.index(context_size)
            matrix = curve_arrays[variant_index, size_index].astype(np.float64)
            mean = matrix.mean(axis=0)
            sd = matrix.std(axis=0, ddof=1)
            for energy, mean_value, sd_value in zip(grid_energy, mean, sd, strict=True):
                curve_rows.append(
                    {
                        "variant": variant,
                        "method": VARIANT_NAMES[variant],
                        "context_size": context_size,
                        "energy_kev": energy,
                        "mean_across_contexts": mean_value,
                        "sd_across_overlapping_contexts": sd_value,
                    }
                )

    write_csv(outputs["bandwidths"], bandwidth_rows)
    write_csv(outputs["cells"], cell_rows)
    write_csv(outputs["bins"], bin_rows)
    write_csv(outputs["contrasts"], contrast_rows)
    write_csv(outputs["summary"], summary_rows)
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
        for variant in VARIANTS:
            group = summary_frame[summary_frame["variant"] == variant].sort_values(
                "context_size"
            )
            axis.errorbar(
                group["context_size"],
                group[f"{metric}_mean"],
                yerr=group[f"{metric}_context_sd"],
                marker="o",
                capsize=3,
                label=VARIANT_NAMES[variant],
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks(CONTEXT_SIZES, labels=[str(value) for value in CONTEXT_SIZES])
        axis.set_xlabel("Context events")
        axis.set_ylabel("5-keV-bin MAE (percentage points)")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(outputs["figure"], dpi=180)
    plt.close(fig)

    def value(variant: str, size: int, metric: str) -> str:
        row = summary_frame[
            (summary_frame["variant"] == variant) & (summary_frame["context_size"] == size)
        ].iloc[0]
        return f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_context_sd']:.3f}"

    table_lines = []
    for variant in VARIANTS:
        table_lines.append(
            f"| {VARIANT_NAMES[variant]} | "
            f"{value(variant, 500, 'peak_region_mean_mae_percentage_points')} | "
            f"{value(variant, 500, 'continuum_region_mean_mae_percentage_points')} | "
            f"{value(variant, 2000, 'peak_region_mean_mae_percentage_points')} | "
            f"{value(variant, 2000, 'continuum_region_mean_mae_percentage_points')} |"
        )
    report = """# Phase 1 Gaussian-kernel controls

Status: complete. The context-only estimator and the separately labeled
pooled-data control use the Nadaraya--Watson probability estimate. The same
formula is a common-bandwidth KDE ratio only when the passing-event density is
multiplied by the context pass fraction; it is not counted as a second method.

| Method | Peak MAE, n=500 | Continuum MAE, n=500 | Peak MAE, n=2000 | Continuum MAE, n=2000 |
|---|---:|---:|---:|---:|
""" + "\n".join(table_lines) + """

Values are mean ± SD across ten overlapping context draws in percentage
points. Bandwidth was selected independently for every development context by
fixed five-fold context CV with seed 20260908 over 2, 5, 10, 20, 50, and 100
keV, breaking exact score ties toward the smaller bandwidth. Final outcomes
were not used for selection. The original n=2,000 context-only bandwidth
choices were reproduced exactly.

The context-only method uses no acceptance pretraining data. The pooled-data
control uses all 18,866 nominal acceptance-input events plus the current
context, with no minimum-bin filter in the kernel calculation. It therefore
has a different and much larger information budget and must not be presented
as a context-only comparator. The server-only archive retains all event-level
predictions. The final reference is finite and noisy, and context draws are
overlapping rather than independent datasets.
"""
    outputs["report"].write_text(report)
    manifest = {
        "schema_version": 1,
        "analysis": "Phase 1 Gaussian-kernel context-only and pooled-data controls",
        "status": "completed",
        "source_commit": source_commit,
        "script": "ml4phy-paper/scripts/run_extension_kernel.py",
        "script_sha256": sha256_file(Path(__file__)),
        "protocol_sha256": sha256_file(extension_path),
        "base_protocol_sha256": sha256_file(base_path),
        "training_predictions_sha256": sha256_file(train_path),
        "archived_n2000_bandwidths_reproduced": True,
        "bandwidth_candidates_kev": list(map(float, BANDWIDTHS_KEV)),
        "cv_folds": CV_FOLDS,
        "cv_seed": CV_SEED,
        "context_only_pretraining_events": 0,
        "pooled_data_pretraining_events": 18866,
        "final_cell_count": len(cell_rows),
        "runtime_seconds": time.perf_counter() - start,
        "failed_runs": [],
        "server_only_predictions": {
            "path": str(server_output.relative_to(repo)),
            "bytes": server_output.stat().st_size,
            "sha256": sha256_file(server_output),
        },
        "outputs": {
            str(path.relative_to(repo)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for key, path in outputs.items()
            if key != "manifest"
        },
        "historical_data_exposure": extension["analysis_status"],
    }
    write_json(outputs["manifest"], manifest)


if __name__ == "__main__":
    main()
