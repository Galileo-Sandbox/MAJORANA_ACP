#!/usr/bin/env python3
"""Export local-shape metrics and curves from saved final neural predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ARCHITECTURES = ("m2", "cell17")
METHOD_NAMES = {
    "m2": "Attentive CNP + PE",
    "cell17": "Density-guided CNP (ours)",
}
TRAINING_SEEDS = (0, 1, 2)
CONTEXT_SEEDS = tuple(range(100, 110))
FIXED_DROPOUT_SEED = 10100
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def display_path(path: Path, repo: Path) -> str:
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def model_id(architecture: str, training_seed: int) -> str:
    if architecture == "m2":
        return "m2_seed0_pilot" if training_seed == 0 else f"m2_seed{training_seed}"
    return "cell17_seed0_recovered" if training_seed == 0 else f"cell17_seed{training_seed}"


def run_dir(repo: Path, architecture: str, training_seed: int, context_seed: int) -> Path:
    return (
        repo
        / "ml4phy-paper/runs"
        / f"20260908-{model_id(architecture, training_seed)}-final-ctx-s{context_seed}-drop10100-mc50"
    )


def hierarchical_summary(rows: list[dict], value: str, group: str) -> list[dict]:
    output = []
    group_values = []
    for row in rows:
        if row[group] not in group_values:
            group_values.append(row[group])
    for architecture in ARCHITECTURES:
        for group_value in group_values:
            seed_means = []
            context_sds = []
            for training_seed in TRAINING_SEEDS:
                values = np.asarray(
                    [
                        row[value]
                        for row in rows
                        if row["architecture_id"] == architecture
                        and row["training_seed"] == training_seed
                        and row[group] == group_value
                    ],
                    dtype=np.float64,
                )
                if values.size != len(CONTEXT_SEEDS):
                    raise ValueError(
                        f"Incomplete context matrix for {(architecture, training_seed, group_value)}"
                    )
                seed_means.append(values.mean())
                context_sds.append(values.std(ddof=1))
            seed_means_array = np.asarray(seed_means)
            output.append(
                {
                    "method": METHOD_NAMES[architecture],
                    "architecture_id": architecture,
                    group: group_value,
                    "training_seed_count": len(TRAINING_SEEDS),
                    "context_draws_per_seed": len(CONTEXT_SEEDS),
                    f"{value}_mean_of_training_seed_means": float(seed_means_array.mean()),
                    f"{value}_sd_across_training_seed_means": float(
                        seed_means_array.std(ddof=1)
                    ),
                    f"{value}_mean_context_sd_within_seed": float(np.mean(context_sds)),
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("ml4phy-paper"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    output_root = (repo / args.output_root).resolve()
    outputs = {
        "bin_cells": output_root / "tables/final_reused_bin_error_cells.csv",
        "bin_summary": output_root / "tables/final_reused_bin_error_summary.csv",
        "contrast_cells": output_root / "tables/final_reused_peak_contrast_cells.csv",
        "contrast_summary": output_root
        / "tables/final_reused_peak_contrast_summary.csv",
        "support": output_root / "tables/final_reference_support.csv",
        "reference": output_root / "tables/final_reference_bins_5kev.csv",
        "curves": output_root / "tables/final_reused_acceptance_curves.csv",
        "figure": output_root / "figures/final_reused_acceptance_curves.png",
        "report": output_root / "reports/reused_final_shape_metrics.md",
        "manifest": output_root / "manifests/reused_final_shape_metrics.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        parser.error("Refusing to overwrite existing outputs: " + ", ".join(existing))
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    protocol_path = repo / "ml4phy-paper/manifests/frozen_protocol_v1.json"
    registry_path = repo / "ml4phy-paper/configs/trained_models_v1.json"
    protocol = read_json(protocol_path)
    registry = read_json(registry_path)
    bin_cells = []
    contrast_cells = []
    curves: dict[tuple[str, int], list[np.ndarray]] = {}
    input_manifest = []
    reference = None
    support = None
    target_hashes = set()

    for architecture in ARCHITECTURES:
        for training_seed in TRAINING_SEEDS:
            spec = registry["models"][model_id(architecture, training_seed)]
            checkpoint_path = repo / spec["checkpoint"]
            if sha256_file(checkpoint_path) != spec["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint hash mismatch for {(architecture, training_seed)}")
            curves[(architecture, training_seed)] = []
            for context_seed in CONTEXT_SEEDS:
                directory = run_dir(repo, architecture, training_seed, context_seed)
                summary_path = directory / "summary.json"
                curve_path = directory / "curve.npz"
                summary = read_json(summary_path)
                if (
                    summary["status"] != "completed"
                    or summary["protocol"]["phase"] != "final"
                    or summary["randomness"]["training_seed"] != training_seed
                    or summary["randomness"]["context_seed"] != context_seed
                    or summary["randomness"]["dropout_seed"] != FIXED_DROPOUT_SEED
                    or summary["counts"]["target"] != 114400
                    or summary["counts"]["context_per_mc_pass"] != 2000
                    or summary["counts"]["mc_passes"] != 50
                ):
                    raise ValueError(f"Final run contract mismatch: {summary_path}")
                if sha256_file(curve_path) != summary["server_only_outputs"]["curve"]["sha256"]:
                    raise ValueError(f"Curve hash mismatch: {curve_path}")
                target_hashes.add(summary["protocol"]["target_identity_sha256"])
                input_manifest.append(
                    {
                        "run_id": summary["run_id"],
                        "summary_sha256": sha256_file(summary_path),
                        "curve_sha256": sha256_file(curve_path),
                    }
                )
                for metric in summary["metrics"]["bin_5kev"]:
                    bin_cells.append(
                        {
                            "method": METHOD_NAMES[architecture],
                            "architecture_id": architecture,
                            "training_seed": training_seed,
                            "context_seed": context_seed,
                            "region_id": metric["region"],
                            "region": REGION_NAMES[metric["region"]],
                            "valid_bin_count": metric["n_valid_bins"],
                            "excluded_bin_count": metric["n_excluded_bins"],
                            "events_in_valid_bins": metric["n_events_in_valid_bins"],
                            "mae_percentage_points": 100.0 * metric["mae"]
                            if metric["mae"] is not None
                            else None,
                            "rmse_percentage_points": 100.0 * metric["rmse"]
                            if metric["rmse"] is not None
                            else None,
                        }
                    )
                for metric in summary["metrics"]["peak_sideband_contrast"]:
                    contrast_cells.append(
                        {
                            "method": METHOD_NAMES[architecture],
                            "architecture_id": architecture,
                            "training_seed": training_seed,
                            "context_seed": context_seed,
                            "feature_id": "Bi-212" if metric["peak"] == "Bi-214" else metric["peak"],
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
                    energy = archive["energy_kev"].astype(np.float64)
                    prediction = archive["prediction"].astype(np.float64)
                    current_reference = {
                        "bin_center_kev": archive["bin_centers_kev"].astype(np.float64),
                        "event_count": archive["bin_counts"].astype(np.int64),
                        "empirical_acceptance": archive["empirical_rate"].astype(np.float64),
                    }
                curves[(architecture, training_seed)].append(prediction)
                if reference is None:
                    reference = current_reference
                    support = summary["metrics"]
                else:
                    for key in reference:
                        if not np.allclose(
                            reference[key], current_reference[key], rtol=0.0, atol=0.0, equal_nan=True
                        ):
                            raise ValueError(f"Reference bins differ in {directory.name}")

    if target_hashes != {protocol["roles"]["final_target"]["identity_sha256"]}:
        raise ValueError("Final target identity differs from the frozen protocol.")
    if reference is None or support is None:
        raise RuntimeError("No reusable final runs were found.")

    bin_summary = hierarchical_summary(bin_cells, "mae_percentage_points", "region")
    rmse_summary = hierarchical_summary(bin_cells, "rmse_percentage_points", "region")
    rmse_by_key = {
        (row["architecture_id"], row["region"]): row for row in rmse_summary
    }
    for row in bin_summary:
        match = rmse_by_key[(row["architecture_id"], row["region"])]
        row.update(
            {
                key: value
                for key, value in match.items()
                if key.startswith("rmse_percentage_points")
            }
        )
    contrast_summary = hierarchical_summary(
        contrast_cells, "absolute_contrast_error_percentage_points", "feature"
    )

    support_rows = []
    for metric in support["event"]:
        if metric["region"] == "equal_region_mean":
            continue
        support_rows.append(
            {
                "region_id": metric["region"],
                "region": REGION_NAMES[metric["region"]],
                "status": metric["status"],
                "target_event_count": metric["n_events"],
                "minimum_supported_event_count": support["minimum_region_events"],
            }
        )

    reference_rows = []
    for center, count, empirical in zip(
        reference["bin_center_kev"],
        reference["event_count"],
        reference["empirical_acceptance"],
        strict=True,
    ):
        reference_rows.append(
            {
                "bin_center_kev": center,
                "event_count": count,
                "supported_minimum_four_events": bool(count >= 4),
                "empirical_acceptance": empirical,
            }
        )

    curve_rows = []
    architecture_curves = {}
    for architecture in ARCHITECTURES:
        seed_means = []
        context_sds = []
        for training_seed in TRAINING_SEEDS:
            matrix = np.stack(curves[(architecture, training_seed)])
            seed_means.append(matrix.mean(axis=0))
            context_sds.append(matrix.std(axis=0, ddof=1))
        seed_means_array = np.stack(seed_means)
        mean = seed_means_array.mean(axis=0)
        seed_sd = seed_means_array.std(axis=0, ddof=1)
        mean_context_sd = np.stack(context_sds).mean(axis=0)
        architecture_curves[architecture] = (mean, seed_sd)
        for query_energy, mean_value, seed_sd_value, context_sd_value in zip(
            energy, mean, seed_sd, mean_context_sd, strict=True
        ):
            curve_rows.append(
                {
                    "method": METHOD_NAMES[architecture],
                    "architecture_id": architecture,
                    "energy_kev": query_energy,
                    "mean_of_training_seed_context_means": mean_value,
                    "sd_across_training_seed_context_means": seed_sd_value,
                    "mean_context_sd_within_training_seed": context_sd_value,
                    "training_seed_count": len(TRAINING_SEEDS),
                    "context_draws_per_seed": len(CONTEXT_SEEDS),
                    "mc_passes_per_cell": 50,
                }
            )

    write_csv(outputs["bin_cells"], bin_cells)
    write_csv(outputs["bin_summary"], bin_summary)
    write_csv(outputs["contrast_cells"], contrast_cells)
    write_csv(outputs["contrast_summary"], contrast_summary)
    write_csv(outputs["support"], support_rows)
    write_csv(outputs["reference"], reference_rows)
    write_csv(outputs["curves"], curve_rows)

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.2), constrained_layout=True)
    panels = [
        ("Full range", (500.0, 3000.0)),
        ("Tl-208 DEP and Bi-212", (1565.0, 1645.0)),
        ("Continuum", (1700.0, 2000.0)),
        ("Tl-208 SE", (2070.0, 2135.0)),
        ("Tl-208 FE", (2575.0, 2640.0)),
    ]
    colors = {"m2": "tab:blue", "cell17": "tab:orange"}
    valid_reference = reference["event_count"] >= 4
    for axis, (title, bounds) in zip(axes.flat, panels, strict=False):
        bin_mask = (
            valid_reference
            & (reference["bin_center_kev"] >= bounds[0])
            & (reference["bin_center_kev"] <= bounds[1])
        )
        axis.scatter(
            reference["bin_center_kev"][bin_mask],
            reference["empirical_acceptance"][bin_mask],
            s=8 if title == "Full range" else 18,
            color="0.45",
            alpha=0.55,
            label="Reference (5-keV bins)",
            zorder=1,
        )
        query_mask = (energy >= bounds[0]) & (energy <= bounds[1])
        for architecture in ARCHITECTURES:
            mean, seed_sd = architecture_curves[architecture]
            axis.plot(
                energy[query_mask],
                mean[query_mask],
                color=colors[architecture],
                linewidth=1.25,
                label=METHOD_NAMES[architecture],
                zorder=2,
            )
            axis.fill_between(
                energy[query_mask],
                mean[query_mask] - seed_sd[query_mask],
                mean[query_mask] + seed_sd[query_mask],
                color=colors[architecture],
                alpha=0.12,
                linewidth=0,
            )
        axis.set_title(title)
        axis.set_xlabel("Energy (keV)")
        axis.set_ylabel("Acceptance")
        axis.set_xlim(bounds)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False)
    fig.savefig(outputs["figure"], dpi=180)
    plt.close(fig)

    def summary_row(rows: list[dict], architecture: str, group: str) -> dict:
        return next(
            row
            for row in rows
            if row["architecture_id"] == architecture
            and (row.get("region") == group or row.get("feature") == group)
        )

    report_lines = [
        "# Reused final local-shape evidence",
        "",
        "Status: exported from the 60 saved final neural prediction cells; no inference or training was rerun. This is a prospectively specified follow-up analysis on historically exposed data.",
        "",
        "## Five-keV bin errors",
        "",
        "Predictions were first evaluated at each reference event energy and then averaged within bins. Values below are percentage points and use training seeds as the top-level replication unit.",
        "",
        "| Region | Attentive CNP + PE MAE | Density-guided CNP MAE | Attentive CNP + PE RMSE | Density-guided CNP RMSE | Valid/excluded bins |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for region in REGION_NAMES.values():
        m2 = summary_row(bin_summary, "m2", region)
        ours = summary_row(bin_summary, "cell17", region)
        cell = next(row for row in bin_cells if row["region"] == region)
        report_lines.append(
            f"| {region} | {m2['mae_percentage_points_mean_of_training_seed_means']:.3f} ± {m2['mae_percentage_points_sd_across_training_seed_means']:.3f} | "
            f"{ours['mae_percentage_points_mean_of_training_seed_means']:.3f} ± {ours['mae_percentage_points_sd_across_training_seed_means']:.3f} | "
            f"{m2['rmse_percentage_points_mean_of_training_seed_means']:.3f} ± {m2['rmse_percentage_points_sd_across_training_seed_means']:.3f} | "
            f"{ours['rmse_percentage_points_mean_of_training_seed_means']:.3f} ± {ours['rmse_percentage_points_sd_across_training_seed_means']:.3f} | "
            f"{cell['valid_bin_count']}/{cell['excluded_bin_count']} |"
        )
    report_lines.extend(
        [
            "",
            "## Peak/sideband contrast error",
            "",
            "| Feature | Attentive CNP + PE absolute error (pp) | Density-guided CNP absolute error (pp) | Center/sideband events |",
            "|---|---:|---:|---:|",
        ]
    )
    for feature in PEAK_NAMES.values():
        m2 = summary_row(contrast_summary, "m2", feature)
        ours = summary_row(contrast_summary, "cell17", feature)
        cell = next(row for row in contrast_cells if row["feature"] == feature)
        report_lines.append(
            f"| {feature} | {m2['absolute_contrast_error_percentage_points_mean_of_training_seed_means']:.3f} ± {m2['absolute_contrast_error_percentage_points_sd_across_training_seed_means']:.3f} | "
            f"{ours['absolute_contrast_error_percentage_points_mean_of_training_seed_means']:.3f} ± {ours['absolute_contrast_error_percentage_points_sd_across_training_seed_means']:.3f} | "
            f"{cell['center_event_count']}/{cell['sideband_event_count']} |"
        )
    report_lines.extend(
        [
            "",
            "## Limits",
            "",
            "- The 114,400-event reference is finite and noisy, not an exact acceptance function.",
            "- The exported curve band is the standard deviation across three training-seed means after averaging ten contexts. It is not a calibrated confidence or posterior interval.",
            "- Contexts overlap and share a target. Full cell-level values and context dispersion are retained in the CSV files.",
            "- The original 50-pass roughness statistic remains unsuitable for a smoothness claim because MC-estimator noise was material.",
            "- These two methods are the only matched neural pair already complete. Missing matched CNP and Attentive CNP baselines must not be filled with confounded historical checkpoints.",
            "",
        ]
    )
    outputs["report"].write_text("\n".join(report_lines))

    manifest = {
        "schema_version": 1,
        "analysis": "reuse saved final predictions for local-shape exports",
        "source_commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "script": "ml4phy-paper/scripts/export_reused_final_shape_metrics.py",
        "script_sha256": sha256_file(Path(__file__)),
        "protocol_sha256": sha256_file(protocol_path),
        "registry_sha256": sha256_file(registry_path),
        "input_run_count": len(input_manifest),
        "inputs": input_manifest,
        "estimator": {
            "mc_passes": 50,
            "dropout_seed": FIXED_DROPOUT_SEED,
            "context_count": 2000,
            "training_seeds": list(TRAINING_SEEDS),
            "context_seeds": list(CONTEXT_SEEDS),
            "curve_aggregation": "context mean within seed, then mean and SD across seeds",
            "bin_estimator": "average event-level predictions at actual reference energies",
        },
        "historical_data_exposure": protocol["exposure_statement"],
        "outputs": {},
    }
    for name, path in outputs.items():
        if name == "manifest":
            continue
        manifest["outputs"][display_path(path, repo)] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    with outputs["manifest"].open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
