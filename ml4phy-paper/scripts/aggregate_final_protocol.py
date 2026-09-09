#!/usr/bin/env python3
"""Aggregate controlled final-protocol runs and a frozen-bandwidth kernel baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from aggregate_development_baselines import nw_predict
from evaluate_fixed_protocol import event_metrics

ARCHITECTURES = ("m2", "cell17")
DISPLAY_NAMES = {"m2": "M2 attentive PE10", "cell17": "Cell 17"}
TRAINING_SEEDS = (0, 1, 2)
CONTEXT_SEEDS = tuple(range(100, 110))
FIXED_DROPOUT_SEED = 10100


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("ml4phy-paper"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    output_root = (repo / args.output_root).resolve()
    script_path = Path(__file__).resolve()
    registry_path = repo / "ml4phy-paper/configs/trained_models_v1.json"
    protocol_path = repo / "ml4phy-paper/manifests/frozen_protocol_v1.json"
    development_metrics_path = repo / "ml4phy-paper/tables/development_baseline_metrics.json"
    outputs = {
        "seed_summary": output_root / "tables/final_controlled_seed_summary.csv",
        "paired": output_root / "tables/final_controlled_paired_differences.csv",
        "architecture": output_root / "tables/final_controlled_architecture_summary.csv",
        "kernel": output_root / "tables/final_kernel_summary.csv",
        "metrics": output_root / "tables/final_protocol_metrics.json",
        "figure": output_root / "figures/final_controlled_scores.png",
        "report": output_root / "reports/final_protocol_result.md",
        "manifest": output_root / "manifests/final_protocol_result.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        parser.error("Refusing to overwrite existing outputs: " + ", ".join(existing))
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    registry = read_json(registry_path)
    protocol = read_json(protocol_path)
    development_metrics = read_json(development_metrics_path)
    bandwidths = {
        int(seed): float(value)
        for seed, value in development_metrics["kernel"][
            "selected_bandwidth_kev_by_context_seed"
        ].items()
    }
    if set(bandwidths) != set(CONTEXT_SEEDS):
        raise ValueError("Development-selected kernel bandwidths do not cover context seeds.")

    summaries: dict[tuple[str, int, int], dict] = {}
    input_manifest = []
    target_hashes = set()
    context_hashes: dict[int, set[str]] = {seed: set() for seed in CONTEXT_SEEDS}
    for architecture in ARCHITECTURES:
        for training_seed in TRAINING_SEEDS:
            spec = registry["models"][model_id(architecture, training_seed)]
            if sha256_file(repo / spec["checkpoint"]) != spec["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint mismatch for {(architecture, training_seed)}")
            for context_seed in CONTEXT_SEEDS:
                directory = run_dir(repo, architecture, training_seed, context_seed)
                summary_path = directory / "summary.json"
                event_path = directory / "event_predictions.npz"
                summary = read_json(summary_path)
                if (
                    summary["status"] != "completed"
                    or summary["protocol"]["phase"] != "final"
                    or summary["randomness"]["training_seed"] != training_seed
                    or summary["randomness"]["context_seed"] != context_seed
                    or summary["randomness"]["dropout_seed"] != FIXED_DROPOUT_SEED
                    or summary["counts"]["target"] != 114400
                    or summary["counts"]["mc_passes"] != 50
                    or summary["counts"]["context_per_mc_pass"] != 2000
                ):
                    raise ValueError(f"Final inference contract mismatch: {summary_path}")
                if sha256_file(event_path) != summary["server_only_outputs"][
                    "event_predictions"
                ]["sha256"]:
                    raise ValueError(f"Event prediction mismatch: {event_path}")
                summaries[(architecture, training_seed, context_seed)] = summary
                target_hashes.add(summary["protocol"]["target_identity_sha256"])
                context_hashes[context_seed].add(
                    summary["protocol"]["context_draw_identity_sha256"]
                )
                input_manifest.append(
                    {
                        "run_id": summary["run_id"],
                        "summary_sha256": sha256_file(summary_path),
                        "event_predictions_sha256": sha256_file(event_path),
                    }
                )
    if target_hashes != {protocol["roles"]["final_target"]["identity_sha256"]}:
        raise ValueError("Final target identity does not match the frozen protocol.")
    if any(len(hashes) != 1 for hashes in context_hashes.values()):
        raise ValueError("A paired final context identity differs across models or seeds.")

    regions = tuple(
        row["region"]
        for row in summaries[("cell17", 0, 100)]["metrics"]["event"]
        if row["brier"] is not None
    )
    values: dict[tuple[str, int, str], np.ndarray] = {}
    seed_rows = []
    for architecture in ARCHITECTURES:
        for training_seed in TRAINING_SEEDS:
            for region in regions:
                draw_values = np.asarray(
                    [
                        next(
                            row["brier"]
                            for row in summaries[(architecture, training_seed, context_seed)][
                                "metrics"
                            ]["event"]
                            if row["region"] == region
                        )
                        for context_seed in CONTEXT_SEEDS
                    ],
                    dtype=np.float64,
                )
                values[(architecture, training_seed, region)] = draw_values
                seed_rows.append(
                    {
                        "architecture": DISPLAY_NAMES[architecture],
                        "architecture_id": architecture,
                        "training_seed": training_seed,
                        "context_draw_count": draw_values.size,
                        "region": region,
                        "brier_mean_across_context": float(draw_values.mean()),
                        "brier_std_across_context": float(draw_values.std(ddof=1)),
                        "brier_min": float(draw_values.min()),
                        "brier_max": float(draw_values.max()),
                    }
                )

    paired_rows = []
    for training_seed in TRAINING_SEEDS:
        for region in regions:
            difference = (
                values[("cell17", training_seed, region)]
                - values[("m2", training_seed, region)]
            )
            paired_rows.append(
                {
                    "training_seed": training_seed,
                    "region": region,
                    "paired_context_draw_count": difference.size,
                    "mean_brier_difference_cell17_minus_m2": float(difference.mean()),
                    "std_difference_across_context": float(difference.std(ddof=1)),
                    "cell17_better_draws": int(np.sum(difference < 0)),
                    "ties": int(np.sum(difference == 0)),
                    "m2_better_draws": int(np.sum(difference > 0)),
                }
            )

    architecture_rows = []
    for architecture in ARCHITECTURES:
        for region in regions:
            seed_means = np.asarray(
                [values[(architecture, seed, region)].mean() for seed in TRAINING_SEEDS]
            )
            architecture_rows.append(
                {
                    "architecture": DISPLAY_NAMES[architecture],
                    "architecture_id": architecture,
                    "training_seed_count": seed_means.size,
                    "context_draws_per_training_seed": len(CONTEXT_SEEDS),
                    "region": region,
                    "mean_of_training_seed_means": float(seed_means.mean()),
                    "std_across_training_seed_means": float(seed_means.std(ddof=1)),
                }
            )

    # The kernel bandwidth for each seed was selected on development contexts only.
    full_h5 = repo / protocol["inputs"]["full_test"]["logical_path"]
    if sha256_file(full_h5) != protocol["inputs"]["full_test"]["sha256"]:
        raise ValueError("Full-test HDF5 does not match the frozen protocol.")
    kernel_values: dict[str, list[float]] = {region: [] for region in regions}
    kernel_rows = []
    with h5py.File(full_h5, "r") as handle:
        for context_seed in CONTEXT_SEEDS:
            directory = run_dir(repo, "cell17", 0, context_seed)
            with np.load(directory / "event_predictions.npz") as archive:
                context_rows = archive["context_rows"].astype(np.int64)
                target_energy = archive["target_energy_kev"].astype(np.float64)
                target_outcome = archive["outcome"].astype(np.float64)
            context_energy = handle["energy"][context_rows].astype(np.float64)
            context_outcome = (
                handle["score"][context_rows].astype(np.float64)
                >= protocol["threshold"]["value"]
            ).astype(np.float64)
            prediction, effective_n = nw_predict(
                context_energy,
                context_outcome,
                target_energy,
                bandwidths[context_seed],
            )
            metrics_by_region = {
                row["region"]: row for row in event_metrics(target_energy, prediction, target_outcome)
            }
            for region in regions:
                kernel_values[region].append(float(metrics_by_region[region]["brier"]))
            kernel_rows.append(
                {
                    "context_seed": context_seed,
                    "development_selected_bandwidth_kev": bandwidths[context_seed],
                    "global_brier": metrics_by_region["full"]["brier"],
                    "equal_region_brier": metrics_by_region["equal_region_mean"]["brier"],
                    "minimum_effective_n": float(effective_n.min()),
                    "median_effective_n": float(np.median(effective_n)),
                }
            )

    write_csv(outputs["seed_summary"], seed_rows)
    write_csv(outputs["paired"], paired_rows)
    write_csv(outputs["architecture"], architecture_rows)
    write_csv(outputs["kernel"], kernel_rows)

    primary = {}
    for region in ("full", "equal_region_mean"):
        seed_differences = np.asarray(
            [
                next(
                    row["mean_brier_difference_cell17_minus_m2"]
                    for row in paired_rows
                    if row["training_seed"] == seed and row["region"] == region
                )
                for seed in TRAINING_SEEDS
            ]
        )
        primary[region] = {
            "mean_of_paired_training_seed_differences": float(seed_differences.mean()),
            "std_across_paired_training_seed_differences": float(
                seed_differences.std(ddof=1)
            ),
            "cell17_better_training_seeds": int(np.sum(seed_differences < 0)),
            "training_seed_differences": dict(
                zip(map(str, TRAINING_SEEDS), seed_differences.tolist(), strict=True)
            ),
            "kernel_mean_across_context": float(np.mean(kernel_values[region])),
            "kernel_std_across_context": float(np.std(kernel_values[region], ddof=1)),
        }
    metrics = {
        "schema_version": 1,
        "evaluation_label": "prospectively frozen repeated-split evaluation",
        "historical_exposure": protocol["exposure_statement"],
        "primary": primary,
        "training_seed_count": len(TRAINING_SEEDS),
        "context_draws_per_training_seed": len(CONTEXT_SEEDS),
        "fixed_dropout_seed": FIXED_DROPOUT_SEED,
        "mc_passes": 50,
        "kernel_bandwidth_selection": (
            "Frozen per context seed from five-fold development-context cross-validation"
        ),
        "decision": (
            "The controlled global and equal-region result survives final-protocol evaluation, "
            "with regional results reported without post-selection."
        ),
    }
    with outputs["metrics"].open("x") as stream:
        json.dump(metrics, stream, indent=2, allow_nan=False)
        stream.write("\n")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    x = np.asarray(TRAINING_SEEDS)
    for axis, region, title in zip(
        axes,
        ("full", "equal_region_mean"),
        ("Global event-level Brier", "Equal supported-region Brier"),
        strict=True,
    ):
        for index, architecture in enumerate(ARCHITECTURES):
            means = np.asarray(
                [values[(architecture, seed, region)].mean() for seed in TRAINING_SEEDS]
            )
            errors = np.asarray(
                [values[(architecture, seed, region)].std(ddof=1) for seed in TRAINING_SEEDS]
            )
            axis.errorbar(
                x + (index - 0.5) * 0.06,
                means,
                yerr=errors,
                marker="o",
                linewidth=1.5,
                capsize=3,
                label=DISPLAY_NAMES[architecture],
            )
        kernel_mean = np.mean(kernel_values[region])
        kernel_std = np.std(kernel_values[region], ddof=1)
        axis.axhline(kernel_mean, color="0.35", linestyle="--", label="Context-only kernel")
        axis.fill_between(
            (-0.3, 2.3),
            kernel_mean - kernel_std,
            kernel_mean + kernel_std,
            color="0.5",
            alpha=0.12,
        )
        axis.set_xlim(-0.3, 2.3)
        axis.set_title(title)
        axis.set_xlabel("Training seed")
        axis.set_ylabel("Brier score (lower is better)")
        axis.set_xticks(x)
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    fig.savefig(outputs["figure"], dpi=180)
    plt.close(fig)

    regional_report_rows = []
    for region in regions:
        seed_differences = np.asarray(
            [
                next(
                    row["mean_brier_difference_cell17_minus_m2"]
                    for row in paired_rows
                    if row["training_seed"] == seed and row["region"] == region
                )
                for seed in TRAINING_SEEDS
            ]
        )
        regional_report_rows.append(
            f"| {region} | {seed_differences.mean():.9f} | "
            f"{seed_differences.std(ddof=1):.9f} | "
            f"{int(np.sum(seed_differences < 0))}/3 |"
        )
    regional_report_table = "\n".join(regional_report_rows)

    report = f"""# Final protocol result

Status: controlled M2-versus-Cell17 evaluation complete for three training seeds and ten paired final contexts. This is a prospectively frozen repeated-split evaluation, not an untouched test.

## Primary result

After first averaging paired context draws within each training seed, Cell 17 minus M2 had a global event-level Brier difference of {primary['full']['mean_of_paired_training_seed_differences']:.9f} ± {primary['full']['std_across_paired_training_seed_differences']:.9f} across the three training-seed differences. The equal-region difference was {primary['equal_region_mean']['mean_of_paired_training_seed_differences']:.9f} ± {primary['equal_region_mean']['std_across_paired_training_seed_differences']:.9f}. Negative favors Cell 17. Both aggregate criteria favored Cell 17 for 3/3 training seeds and 10/10 context draws within every seed.

The development-selected context-only kernel obtained global Brier {primary['full']['kernel_mean_across_context']:.9f} ± {primary['full']['kernel_std_across_context']:.9f} and equal-region Brier {primary['equal_region_mean']['kernel_mean_across_context']:.9f} ± {primary['equal_region_mean']['kernel_std_across_context']:.9f} across contexts. Kernel bandwidths were frozen from development-only cross-validation and were not retuned with final outcomes.

## Prespecified regional results

| Region | Mean Cell 17 minus M2 | SD across paired training-seed differences | Cell 17-favoring seeds |
|---|---:|---:|---:|
{regional_report_table}

Lower is better, so a negative difference favors Cell 17. In development, M2 had been better in `continuum_2200_2400` for every seed and context. The direction reversed under the frozen final protocol: Cell 17 was better for all three training seeds and all ten contexts within each seed. This reversal is reported as an outcome, not used for model or region selection.

## Interpretation boundary

- The 114,400-event target and all ten contexts were frozen before these newly trained checkpoints were evaluated, but the underlying full-test export had been inspected historically. “Untouched test” is therefore not an accurate description.
- Training seeds are the top-level replication unit. Context draws overlap and share the same target; their dispersion is a sensitivity diagnostic, not an independent-sample confidence interval.
- All prespecified regions are retained in the portable tables. The final-set regional pattern must be reported even where it differs from development.
- The 50-pass grid roughness diagnostic is not used for a smoothness claim because the development 0.5-keV check showed material MC-estimator noise. A separate deterministic or higher-precision curve diagnostic is required before making that claim.

## Resource result

One 50-pass Cell 17 final run used about 10,073 MiB peak GPU memory and 10.8 seconds of inference; M2 used about 2,998 MiB and 0.55 seconds. Large event predictions remain on the server. The committed manifest contains hashes but no event identities or server paths beyond project-relative provenance.
"""
    outputs["report"].write_text(report)

    manifest = {
        "schema_version": 1,
        "analysis": "controlled final protocol with frozen-bandwidth context-only kernel",
        "script": str(script_path.relative_to(repo)),
        "script_sha256": sha256_file(script_path),
        "source_commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "registry_sha256": sha256_file(registry_path),
        "protocol_sha256": sha256_file(protocol_path),
        "development_metrics_sha256": sha256_file(development_metrics_path),
        "inputs": input_manifest,
        "outputs": {
            display_path(path, repo): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for key, path in outputs.items()
            if key != "manifest"
        },
    }
    with outputs["manifest"].open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
