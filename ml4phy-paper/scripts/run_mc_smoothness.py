#!/usr/bin/env python3
"""Measure nested dropout-MC noise, smoothness, error, and bin-interval width."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

MODELS = ("m2_seed0", "ours_seed0")
METHOD_NAMES = {
    "m2_seed0": "Attentive CNP + PE",
    "ours_seed0": "Density-guided CNP (ours)",
}
STREAM_SEEDS = (10100, 30100)
CONTINUA = {
    "continuum_1700_2000": (1700.0, 2000.0),
    "continuum_2200_2400": (2200.0, 2400.0),
}
PASS_LEVELS = (50, 200)


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


def binned_draw(
    target_values: np.ndarray, bin_index: np.ndarray, bin_count: np.ndarray
) -> np.ndarray:
    total = np.bincount(bin_index, weights=target_values, minlength=bin_count.size)
    return np.divide(total, bin_count, out=np.full(bin_count.size, np.nan), where=bin_count > 0)


def continuum_mae(
    target_energy: np.ndarray, target_prediction: np.ndarray, target_outcome: np.ndarray
) -> dict[str, float]:
    from evaluate_fixed_protocol import bin_metrics

    rows, _ = bin_metrics(target_energy, target_prediction, target_outcome)
    by_region = {row["region"]: row for row in rows}
    return {
        region: 100.0 * by_region[region]["mae"] for region in CONTINUA
    }


def normalized_curvature(grid: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    spacing = float(np.diff(grid).mean())
    output = {}
    for region, (lower, upper) in CONTINUA.items():
        mask = (grid >= lower) & (grid <= upper)
        output[region] = float(np.mean(np.abs(np.diff(prediction[mask], n=2))) / spacing**2)
    return output


def main() -> None:
    repo = Path(".").resolve()
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--short"], text=True
    ).splitlines()
    if status:
        raise RuntimeError("MC diagnostic requires a clean worktree: " + "; ".join(status))
    source_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    source_root = repo / "ml4phy-paper/local/resum-flex-edba6a"
    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(repo))
    os.environ["PYTHONPATH"] = os.pathsep.join([str(source_root), str(repo)])
    from schemas import InputMode, StandardBatch

    from majorana_acp.cut_acceptance.config import load_config
    from majorana_acp.cut_acceptance.positional_encoding import encode_phi
    from scripts.diagnostics.cnp_test_inference import _load_cnp

    output_root = repo / "ml4phy-paper"
    outputs = {
        "estimates": output_root / "tables/mc_smoothness_estimates.csv",
        "streams": output_root / "tables/mc_smoothness_stream_comparison.csv",
        "coverage": output_root / "tables/mc_smoothness_coverage_width.csv",
        "figure": output_root / "figures/mc_smoothness_diagnostic.png",
        "report": output_root / "reports/mc_smoothness_result.md",
        "manifest": output_root / "manifests/mc_smoothness_result.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))
    server_output = output_root / "runs/mc/20260909-smoothness-v1/diagnostics.npz"
    server_output.parent.mkdir(parents=True, exist_ok=False)

    config_path = output_root / "configs/mc_smoothness_protocol_v1.json"
    registry_path = output_root / "configs/extension_models_v1.json"
    protocol_path = output_root / "manifests/frozen_protocol_v1.json"
    extension_path = output_root / "manifests/extension_protocol_v1.json"
    role_path = output_root / "local/protocol/frozen_roles_v1.npz"
    context_path = output_root / "local/protocol/extension_context_roles_v1.npz"
    diagnostic_config = read_json(config_path)
    registry = read_json(registry_path)
    protocol = read_json(protocol_path)
    extension = read_json(extension_path)
    if diagnostic_config["status"] != "frozen_before_mc_smoothness_execution":
        raise ValueError("MC diagnostic protocol is not frozen")
    if sha256_file(role_path) != protocol["local_role_archive"]["sha256"]:
        raise ValueError("Base role archive hash mismatch")
    if sha256_file(context_path) != extension["context_protocol"]["local_archive"][
        "sha256"
    ]:
        raise ValueError("Extension context archive hash mismatch")
    with np.load(role_path) as archive:
        target_rows = archive["final_target_rows"].astype(np.int64)
        threshold = float(archive["fixed_threshold"])
    with np.load(context_path) as archive:
        context_rows = np.sort(archive["final_context_s100_n2000_rows"].astype(np.int64))
    h5_path = repo / protocol["inputs"]["full_test"]["logical_path"]
    with h5py.File(h5_path, "r") as handle:
        context_energy = handle["energy"][context_rows].astype(np.float64)
        context_score = handle["score"][context_rows].astype(np.float64)
        target_energy = handle["energy"][target_rows].astype(np.float64)
        target_outcome = (
            handle["score"][target_rows].astype(np.float64) >= threshold
        ).astype(np.float64)
    grid_05 = np.arange(500.0, 3000.0 + 0.25, 0.5)
    grid_10 = grid_05[::2]
    query_energy = np.concatenate([target_energy, grid_05])
    edges = np.arange(500.0, 3000.0 + 5.0, 5.0)
    bin_index = np.searchsorted(edges, target_energy, side="right") - 1
    bin_count = np.bincount(bin_index, minlength=edges.size - 1)
    empirical = binned_draw(target_outcome, bin_index, bin_count)
    valid_bins = bin_count >= 4

    estimate_rows = []
    stream_rows = []
    coverage_rows = []
    saved_arrays: dict[str, np.ndarray] = {
        "grid_energy_0p5_kev": grid_05,
        "target_rows": target_rows,
        "target_energy_kev": target_energy,
        "target_outcome": target_outcome.astype(np.int8),
        "bin_counts": bin_count,
        "bin_empirical_acceptance": empirical,
    }
    timings = []
    extensions = {}

    for model_id in MODELS:
        specification = registry["models"][model_id]
        checkpoint_path = repo / specification["checkpoint"]
        if sha256_file(checkpoint_path) != specification["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint mismatch for {model_id}")
        cfg = load_config(repo / specification["config"])
        cnp = _load_cnp(cfg, checkpoint_path)
        device = next(cnp.parameters()).device
        e_lo, e_hi = cfg.energy_range
        t_lo, t_hi = cfg.threshold_range
        t_norm = float((threshold - t_lo) / (t_hi - t_lo))
        query_norm = (query_energy - e_lo) / (e_hi - e_lo)
        target_phi = np.stack(
            [query_norm, np.full_like(query_norm, t_norm)], axis=-1
        )[None, :, :]
        target_phi = encode_phi(
            target_phi, cfg.positional_encoding, energy_range_kev=cfg.energy_range
        )
        target_batch = StandardBatch(
            mode=InputMode.EVENT_ONLY,
            theta=None,
            phi=target_phi,
            labels=np.zeros((1, query_energy.size), dtype=np.int8),
        )
        context_norm = (context_energy - e_lo) / (e_hi - e_lo)
        context_phi = np.stack(
            [context_norm, np.full_like(context_norm, t_norm)], axis=-1
        )[None, :, :]
        context_phi = encode_phi(
            context_phi, cfg.positional_encoding, energy_range_kev=cfg.energy_range
        )
        context_batch = StandardBatch(
            mode=InputMode.EVENT_ONLY,
            theta=None,
            phi=context_phi,
            labels=(context_score >= threshold).astype(np.int8)[None, :],
        )

        cnp.eval()
        with torch.no_grad():
            deterministic = cnp.predict_beta(context_batch, target_batch).cpu().numpy()[0]
        deterministic_target = deterministic[: target_energy.size]
        deterministic_grid = deterministic[target_energy.size :]
        saved_arrays[f"{model_id}_dropout_disabled_target"] = deterministic_target
        saved_arrays[f"{model_id}_dropout_disabled_grid"] = deterministic_grid
        deterministic_error = continuum_mae(
            target_energy, deterministic_target, target_outcome
        )
        for spacing, grid, curve in (
            (0.5, grid_05, deterministic_grid),
            (1.0, grid_10, deterministic_grid[::2]),
        ):
            curvature = normalized_curvature(grid, curve)
            for region in CONTINUA:
                estimate_rows.append(
                    {
                        "model_id": model_id,
                        "method": METHOD_NAMES[model_id],
                        "estimator": "dropout_disabled",
                        "stream_seed": None,
                        "passes": 0,
                        "grid_spacing_kev": spacing,
                        "region": region,
                        "continuum_mae_percentage_points": deterministic_error[region],
                        "normalized_mean_absolute_curvature": curvature[region],
                    }
                )

        stream_results = {}

        def run_stream(
            stream_seed: int,
            passes: int,
            model=cnp,
            fixed_context_batch=context_batch,
            fixed_target_batch=target_batch,
            model_device=device,
        ) -> dict:
            torch.manual_seed(stream_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(stream_seed)
            sums = {
                level: np.zeros(query_energy.size, dtype=np.float64)
                for level in (50, 200, 800)
                if level <= passes
            }
            running = np.zeros(query_energy.size, dtype=np.float64)
            bin_draws = np.empty((passes, bin_count.size), dtype=np.float32)
            model.train()
            start = time.perf_counter()
            try:
                with torch.no_grad():
                    for index in range(passes):
                        prediction = model.predict_beta(
                            fixed_context_batch, fixed_target_batch
                        ).cpu().numpy()[0]
                        running += prediction
                        bin_draws[index] = binned_draw(
                            prediction[: target_energy.size], bin_index, bin_count
                        )
                        level = index + 1
                        if level in sums:
                            sums[level] = running / level
            finally:
                model.eval()
            if model_device.type == "cuda":
                torch.cuda.synchronize(model_device)
            elapsed = time.perf_counter() - start
            return {"means": sums, "bin_draws": bin_draws, "seconds": elapsed}

        for stream_seed in STREAM_SEEDS:
            result = run_stream(stream_seed, 200)
            timings.append(
                {
                    "model_id": model_id,
                    "stream_seed": stream_seed,
                    "passes": 200,
                    "seconds": result["seconds"],
                }
            )
            stream_results[stream_seed] = result
            for passes in PASS_LEVELS:
                mean = result["means"][passes]
                target_mean = mean[: target_energy.size]
                grid_mean = mean[target_energy.size :]
                saved_arrays[f"{model_id}_stream{stream_seed}_mc{passes}_target"] = target_mean
                saved_arrays[f"{model_id}_stream{stream_seed}_mc{passes}_grid"] = grid_mean
                error = continuum_mae(target_energy, target_mean, target_outcome)
                for spacing, grid, curve in (
                    (0.5, grid_05, grid_mean),
                    (1.0, grid_10, grid_mean[::2]),
                ):
                    curvature = normalized_curvature(grid, curve)
                    for region in CONTINUA:
                        estimate_rows.append(
                            {
                                "model_id": model_id,
                                "method": METHOD_NAMES[model_id],
                                "estimator": "dropout_mc",
                                "stream_seed": stream_seed,
                                "passes": passes,
                                "grid_spacing_kev": spacing,
                                "region": region,
                                "continuum_mae_percentage_points": error[region],
                                "normalized_mean_absolute_curvature": curvature[region],
                            }
                        )
                draws = result["bin_draws"][:passes]
                lower, upper = np.nanquantile(draws, (0.16, 0.84), axis=0)
                for region, (low_energy, high_energy) in CONTINUA.items():
                    centers = 0.5 * (edges[:-1] + edges[1:])
                    use = (
                        valid_bins & (centers >= low_energy) & (centers < high_energy)
                    )
                    coverage_rows.append(
                        {
                            "model_id": model_id,
                            "method": METHOD_NAMES[model_id],
                            "stream_seed": stream_seed,
                            "passes": passes,
                            "region": region,
                            "valid_bin_count": int(use.sum()),
                            "empirical_bin_coverage": float(
                                np.mean((empirical[use] >= lower[use]) & (empirical[use] <= upper[use]))
                            ),
                            "mean_interval_width_percentage_points": 100.0
                            * float(np.mean(upper[use] - lower[use])),
                            "interval_definition": "central 68% dropout function-draw bin interval",
                            "coverage_target": "finite empirical bin acceptance",
                        }
                    )

        trigger = False
        for passes in PASS_LEVELS:
            left = stream_results[STREAM_SEEDS[0]]["means"][passes]
            right = stream_results[STREAM_SEEDS[1]]["means"][passes]
            left_grid = left[target_energy.size :]
            right_grid = right[target_energy.size :]
            left_curvature = normalized_curvature(grid_05, left_grid)
            right_curvature = normalized_curvature(grid_05, right_grid)
            for region, (lower, upper) in CONTINUA.items():
                mask = (grid_05 >= lower) & (grid_05 <= upper)
                difference = left_grid[mask] - right_grid[mask]
                relative_curvature = abs(
                    left_curvature[region] - right_curvature[region]
                ) / max(0.5 * (left_curvature[region] + right_curvature[region]), 1e-15)
                rmse_pp = 100.0 * float(np.sqrt(np.mean(difference**2)))
                stream_rows.append(
                    {
                        "model_id": model_id,
                        "method": METHOD_NAMES[model_id],
                        "passes": passes,
                        "region": region,
                        "independent_stream_rmse_percentage_points": rmse_pp,
                        "independent_stream_max_abs_percentage_points": 100.0
                        * float(np.max(np.abs(difference))),
                        "relative_normalized_curvature_difference": relative_curvature,
                    }
                )
                if passes == 200 and (rmse_pp > 0.25 or relative_curvature > 0.10):
                    trigger = True
        seconds_per_pass = max(
            result["seconds"] / result["passes"]
            for result in timings
            if result["model_id"] == model_id
        )
        projected_extension_seconds = 2.0 * 800.0 * seconds_per_pass
        extensions[model_id] = {
            "triggered": trigger,
            "projected_two_stream_800_seconds": projected_extension_seconds,
            "affordable_under_1800_seconds": projected_extension_seconds <= 1800.0,
            "executed": False,
        }
        if trigger and projected_extension_seconds <= 1800.0:
            extended = {}
            for stream_seed in STREAM_SEEDS:
                result = run_stream(stream_seed, 800)
                extended[stream_seed] = result
                original_200 = stream_results[stream_seed]["means"][200]
                if not np.allclose(
                    result["means"][200], original_200, rtol=0.0, atol=1e-12
                ):
                    raise ValueError("Repeated stream did not preserve the nested 200-pass prefix")
                mean = result["means"][800]
                saved_arrays[f"{model_id}_stream{stream_seed}_mc800_target"] = mean[
                    : target_energy.size
                ]
                saved_arrays[f"{model_id}_stream{stream_seed}_mc800_grid"] = mean[
                    target_energy.size :
                ]
                error = continuum_mae(
                    target_energy, mean[: target_energy.size], target_outcome
                )
                for spacing, grid, curve in (
                    (0.5, grid_05, mean[target_energy.size :]),
                    (1.0, grid_10, mean[target_energy.size :][::2]),
                ):
                    curvature = normalized_curvature(grid, curve)
                    for region in CONTINUA:
                        estimate_rows.append(
                            {
                                "model_id": model_id,
                                "method": METHOD_NAMES[model_id],
                                "estimator": "dropout_mc",
                                "stream_seed": stream_seed,
                                "passes": 800,
                                "grid_spacing_kev": spacing,
                                "region": region,
                                "continuum_mae_percentage_points": error[region],
                                "normalized_mean_absolute_curvature": curvature[region],
                            }
                        )
                draws = result["bin_draws"]
                lower_draw, upper_draw = np.nanquantile(draws, (0.16, 0.84), axis=0)
                centers = 0.5 * (edges[:-1] + edges[1:])
                for region, (low_energy, high_energy) in CONTINUA.items():
                    use = (
                        valid_bins & (centers >= low_energy) & (centers < high_energy)
                    )
                    coverage_rows.append(
                        {
                            "model_id": model_id,
                            "method": METHOD_NAMES[model_id],
                            "stream_seed": stream_seed,
                            "passes": 800,
                            "region": region,
                            "valid_bin_count": int(use.sum()),
                            "empirical_bin_coverage": float(
                                np.mean(
                                    (empirical[use] >= lower_draw[use])
                                    & (empirical[use] <= upper_draw[use])
                                )
                            ),
                            "mean_interval_width_percentage_points": 100.0
                            * float(np.mean(upper_draw[use] - lower_draw[use])),
                            "interval_definition": "central 68% dropout function-draw bin interval",
                            "coverage_target": "finite empirical bin acceptance",
                        }
                    )
            left = extended[STREAM_SEEDS[0]]["means"][800][target_energy.size :]
            right = extended[STREAM_SEEDS[1]]["means"][800][target_energy.size :]
            left_curvature = normalized_curvature(grid_05, left)
            right_curvature = normalized_curvature(grid_05, right)
            for region, (lower, upper) in CONTINUA.items():
                mask = (grid_05 >= lower) & (grid_05 <= upper)
                difference = left[mask] - right[mask]
                stream_rows.append(
                    {
                        "model_id": model_id,
                        "method": METHOD_NAMES[model_id],
                        "passes": 800,
                        "region": region,
                        "independent_stream_rmse_percentage_points": 100.0
                        * float(np.sqrt(np.mean(difference**2))),
                        "independent_stream_max_abs_percentage_points": 100.0
                        * float(np.max(np.abs(difference))),
                        "relative_normalized_curvature_difference": abs(
                            left_curvature[region] - right_curvature[region]
                        )
                        / max(
                            0.5 * (left_curvature[region] + right_curvature[region]),
                            1e-15,
                        ),
                    }
                )
            extensions[model_id]["executed"] = True
        del cnp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    np.savez(server_output, **saved_arrays)
    write_csv(outputs["estimates"], estimate_rows)
    write_csv(outputs["streams"], stream_rows)
    write_csv(outputs["coverage"], coverage_rows)

    stream_frame = pd.DataFrame(stream_rows)
    coverage_frame = pd.DataFrame(coverage_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for model_id in MODELS:
        group = stream_frame[
            (stream_frame["model_id"] == model_id)
            & (stream_frame["region"] == "continuum_1700_2000")
        ]
        axes[0].plot(
            group["passes"],
            group["independent_stream_rmse_percentage_points"],
            marker="o",
            label=METHOD_NAMES[model_id],
        )
        width = (
            coverage_frame[
                (coverage_frame["model_id"] == model_id)
                & (coverage_frame["region"] == "continuum_1700_2000")
            ]
            .groupby("passes")["mean_interval_width_percentage_points"]
            .mean()
        )
        axes[1].plot(width.index, width.values, marker="o", label=METHOD_NAMES[model_id])
    axes[0].axhline(0.25, color="0.4", linestyle="--", label="800-pass trigger")
    axes[0].set_xlabel("MC dropout passes")
    axes[0].set_ylabel("Independent-stream RMSE (percentage points)")
    axes[0].set_title("MC estimator noise")
    axes[1].set_xlabel("MC dropout passes")
    axes[1].set_ylabel("Mean 68% bin-interval width (percentage points)")
    axes[1].set_title("Dropout interval width")
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(outputs["figure"], dpi=180)
    plt.close(fig)

    final_passes = {
        model_id: 800 if extensions[model_id]["executed"] else 200 for model_id in MODELS
    }
    result_lines = []
    for model_id in MODELS:
        passes = final_passes[model_id]
        for region in CONTINUA:
            row = next(
                item
                for item in stream_rows
                if item["model_id"] == model_id
                and item["passes"] == passes
                and item["region"] == region
            )
            widths = [
                item["mean_interval_width_percentage_points"]
                for item in coverage_rows
                if item["model_id"] == model_id
                and item["passes"] == passes
                and item["region"] == region
            ]
            coverages = [
                item["empirical_bin_coverage"]
                for item in coverage_rows
                if item["model_id"] == model_id
                and item["passes"] == passes
                and item["region"] == region
            ]
            result_lines.append(
                f"| {METHOD_NAMES[model_id]} | {passes} | {region} | "
                f"{row['independent_stream_rmse_percentage_points']:.3f} | "
                f"{np.mean(coverages):.3f} | {np.mean(widths):.3f} |"
            )
    report = """# MC-noise and smoothness diagnostic

Status: complete for the two seed-0 neural models on final context seed 100 at
n=2,000. This fixed diagnostic does not select a favorable context or training
seed.

| Method | Final passes | Region | Two-stream RMSE (pp) | Empirical-bin coverage | Mean interval width (pp) |
|---|---:|---|---:|---:|---:|
""" + "\n".join(result_lines) + """

The 50-pass estimate is the exact prefix of each 200-pass stream. An 800-pass
extension was run only where the frozen numerical-noise trigger fired and the
measured projection was below 30 minutes. Every pass evaluated all 114,400
actual target energies and the complete 0.5-keV grid in one forward call. No
energy chunking was used. Ordinary elementwise dropout was retained; masks
were not forced to be identical across energies, while each query retained its
intended marginal estimator. The 1-keV diagnostic is a subset of the same
0.5-keV function draws, and curvature is divided by squared grid spacing.

Dropout-disabled predictions are reported as a separate deterministic network
estimator, not as the infinite-MC limit. Continuum MAE is listed beside every
roughness value in the CSV. The interval is the central 68% distribution of
bin means formed by averaging each stochastic function draw over actual events
in the bin before taking quantiles. Coverage checks the finite empirical bin
acceptance and is only a consistency diagnostic, not calibrated confidence
coverage. Width is reported beside coverage; higher coverage alone is not
interpreted as better.
"""
    outputs["report"].write_text(report)
    manifest = {
        "schema_version": 1,
        "analysis": "nested dropout-MC noise and normalized smoothness diagnostic",
        "status": "completed",
        "source_commit": source_commit,
        "script": "ml4phy-paper/scripts/run_mc_smoothness.py",
        "script_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(config_path),
        "model_registry_sha256": sha256_file(registry_path),
        "protocol_sha256": sha256_file(extension_path),
        "context_seed": 100,
        "context_size": 2000,
        "stream_seeds": list(STREAM_SEEDS),
        "timings": timings,
        "extensions": extensions,
        "energy_chunking": diagnostic_config["energy_chunking"],
        "dropout_mask_coupling": diagnostic_config["dropout_mask_coupling"],
        "failed_runs": [],
        "server_only_diagnostics": {
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
