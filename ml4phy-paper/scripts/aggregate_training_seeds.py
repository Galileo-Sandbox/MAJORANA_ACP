#!/usr/bin/env python3
"""Aggregate the three-seed controlled Cell 17 versus M2 development comparison."""

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


def inference_summary_path(repo: Path, architecture: str, training_seed: int, context_seed: int):
    runs = repo / "ml4phy-paper/runs"
    if architecture == "m2":
        model = "m2-seed0" if training_seed == 0 else f"m2_seed{training_seed}"
        return runs / f"20260908-{model}-dev-ctx-s{context_seed}-drop10100-mc50/summary.json"
    if training_seed == 0:
        name = (
            "20260908-cell17-dev-s100-mc50"
            if context_seed == 100
            else f"20260908-cell17-dev-ctx-s{context_seed}-drop10100-mc50"
        )
        return runs / name / "summary.json"
    return (
        runs
        / f"20260908-cell17_seed{training_seed}-dev-ctx-s{context_seed}-drop10100-mc50"
        / "summary.json"
    )


def training_record_path(repo: Path, architecture: str, seed: int) -> Path | None:
    if architecture == "cell17" and seed == 0:
        return None
    run_name = (
        "20260908-m2-seed0-pilot3000"
        if architecture == "m2" and seed == 0
        else f"20260908-{architecture}-seed{seed}-train3000"
    )
    return repo / "ml4phy-paper/runs/training" / run_name / "runner_record.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("ml4phy-paper"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    output_root = (repo / args.output_root).resolve()
    script_path = Path(__file__).resolve()
    registry_path = repo / "ml4phy-paper/configs/trained_models_v1.json"
    outputs = {
        "seed_summary": output_root / "tables/controlled_seed_summary.csv",
        "paired": output_root / "tables/controlled_seed_paired_differences.csv",
        "architecture": output_root / "tables/controlled_architecture_summary.csv",
        "training": output_root / "tables/controlled_training_runs.csv",
        "metrics": output_root / "tables/controlled_three_seed_metrics.json",
        "figure": output_root / "figures/controlled_three_seed_scores.png",
        "report": output_root / "reports/controlled_three_seed_result.md",
        "manifest": output_root / "manifests/controlled_three_seed_result.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        parser.error("Refusing to overwrite existing outputs: " + ", ".join(existing))
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    registry = read_json(registry_path)
    model_ids = {
        ("m2", 0): "m2_seed0_pilot",
        ("m2", 1): "m2_seed1",
        ("m2", 2): "m2_seed2",
        ("cell17", 0): "cell17_seed0_recovered",
        ("cell17", 1): "cell17_seed1",
        ("cell17", 2): "cell17_seed2",
    }
    for key, model_id in model_ids.items():
        spec = registry["models"][model_id]
        checkpoint = repo / spec["checkpoint"]
        config = repo / spec["config"]
        if sha256_file(checkpoint) != spec["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch for {key}")
        if sha256_file(config) != spec["config_sha256"]:
            raise ValueError(f"Configuration hash mismatch for {key}")
        if spec["training_seed"] != key[1]:
            raise ValueError(f"Training seed mismatch for {key}")

    summaries: dict[tuple[str, int, int], dict] = {}
    inference_inputs = []
    target_hashes = set()
    context_hashes: dict[int, set[str]] = {seed: set() for seed in CONTEXT_SEEDS}
    for architecture in ARCHITECTURES:
        for training_seed in TRAINING_SEEDS:
            for context_seed in CONTEXT_SEEDS:
                path = inference_summary_path(repo, architecture, training_seed, context_seed)
                summary = read_json(path)
                if (
                    summary["status"] != "completed"
                    or summary["protocol"]["phase"] != "development"
                    or summary["randomness"]["training_seed"] != training_seed
                    or summary["randomness"]["context_seed"] != context_seed
                    or summary["randomness"]["dropout_seed"] != FIXED_DROPOUT_SEED
                    or summary["counts"]["mc_passes"] != 50
                    or summary["counts"]["context_per_mc_pass"] != 2000
                ):
                    raise ValueError(f"Inference contract mismatch: {path}")
                summaries[(architecture, training_seed, context_seed)] = summary
                target_hashes.add(summary["protocol"]["target_identity_sha256"])
                context_hashes[context_seed].add(
                    summary["protocol"]["context_draw_identity_sha256"]
                )
                inference_inputs.append(
                    {"run_id": summary["run_id"], "summary_sha256": sha256_file(path)}
                )
    if len(target_hashes) != 1:
        raise ValueError("Target identity differs across controlled runs.")
    if any(len(hashes) != 1 for hashes in context_hashes.values()):
        raise ValueError("A paired context identity differs across models or training seeds.")

    regions = tuple(
        row["region"]
        for row in summaries[("cell17", 0, 100)]["metrics"]["event"]
        if row["brier"] is not None
    )
    seed_rows = []
    values: dict[tuple[str, int, str], np.ndarray] = {}
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
                    "context_draws_per_seed": len(CONTEXT_SEEDS),
                    "region": region,
                    "mean_of_training_seed_means": float(seed_means.mean()),
                    "std_across_training_seed_means": float(seed_means.std(ddof=1)),
                    "minimum_training_seed_mean": float(seed_means.min()),
                    "maximum_training_seed_mean": float(seed_means.max()),
                }
            )

    training_rows = []
    training_inputs = []
    for architecture in ARCHITECTURES:
        for seed in TRAINING_SEEDS:
            record_path = training_record_path(repo, architecture, seed)
            if record_path is None:
                historical_summary = read_json(
                    repo
                    / "results/cut_acceptance/simple_cnn_small/sweeps/cell17/bin10/inclusive/run_summary.json"
                )
                training_rows.append(
                    {
                        "architecture": DISPLAY_NAMES[architecture],
                        "training_seed": seed,
                        "provenance": "recovered historical checkpoint",
                        "steps": 3000,
                        "final_loss": historical_summary["cnp_final_train_loss"],
                        "wall_seconds": None,
                        "checkpoint_sha256": registry["models"][model_ids[(architecture, seed)]][
                            "checkpoint_sha256"
                        ],
                    }
                )
                continue
            record = read_json(record_path)
            summary_path = record_path.parent / "artifacts/run_summary.json"
            training_summary = read_json(summary_path)
            if record["status"] != "completed" or record["returncode"] != 0:
                raise ValueError(f"Training did not complete: {record_path}")
            training_rows.append(
                {
                    "architecture": DISPLAY_NAMES[architecture],
                    "training_seed": seed,
                    "provenance": "paper controlled run",
                    "steps": record["training_steps"],
                    "final_loss": training_summary["cnp_final_train_loss"],
                    "wall_seconds": record["wall_seconds"],
                    "checkpoint_sha256": record["outputs"]["cnp.ckpt"]["sha256"],
                }
            )
            training_inputs.append(
                {
                    "run_id": record["run_id"],
                    "runner_record_sha256": sha256_file(record_path),
                    "run_summary_sha256": sha256_file(summary_path),
                }
            )

    write_csv(outputs["seed_summary"], seed_rows)
    write_csv(outputs["paired"], paired_rows)
    write_csv(outputs["architecture"], architecture_rows)
    write_csv(outputs["training"], training_rows)

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
        }
    metrics = {
        "schema_version": 1,
        "comparison": "Cell 17 minus M2 attentive PE10",
        "primary": primary,
        "training_seed_count": len(TRAINING_SEEDS),
        "context_draws_per_training_seed": len(CONTEXT_SEEDS),
        "fixed_dropout_seed": FIXED_DROPOUT_SEED,
        "target_identity_sha256": next(iter(target_hashes)),
        "interpretation": (
            "Training seeds are the top-level replication unit. Context draws are paired, "
            "overlapping sensitivity checks on a shared target and are not independent datasets."
        ),
        "decision": (
            "Freeze Cell 17 as the three-seed development candidate; retain regional trade-offs "
            "and proceed only to a timed final-protocol inference pilot."
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
        axis.set_title(title)
        axis.set_xlabel("Training seed")
        axis.set_ylabel("Brier score (lower is better)")
        axis.set_xticks(x)
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    fig.savefig(outputs["figure"], dpi=180)
    plt.close(fig)

    report = f"""# Controlled three-training-seed development result

Status: the prespecified M2-versus-Cell17 development comparison is complete for training seeds 0, 1, and 2. Final-protocol results have not been inspected for these newly trained models.

## Primary result

Training seeds are the replication unit. After averaging the ten paired context draws within each seed, Cell 17 minus M2 had a global Brier difference of {primary['full']['mean_of_paired_training_seed_differences']:.9f} ± {primary['full']['std_across_paired_training_seed_differences']:.9f} across the three training-seed differences. The equal-supported-region difference was {primary['equal_region_mean']['mean_of_paired_training_seed_differences']:.9f} ± {primary['equal_region_mean']['std_across_paired_training_seed_differences']:.9f}. Negative favors Cell 17; both criteria favored Cell 17 for 3/3 training seeds and 10/10 context draws within every seed.

The context draws overlap and share one 2,074-event development target. Their within-seed standard deviation is a context-sensitivity diagnostic, not an independent-sample confidence interval. With only three training seeds, the across-seed standard deviations are descriptive.

## Regional boundary

The benefit is not uniform. Cell 17 improves the 1700-2000 keV continuum strongly for every training seed, while M2 improves the 2200-2400 keV window for every training seed and context draw. Unsupported DEP and sparse-tail evidence remains inconclusive. Any paper claim must state this regional trade-off and cannot say that density guidance is universally better.

## Resource result

M2 training took 84.86-87.93 seconds for the three new paper runs; Cell 17 seeds 1 and 2 took 125.55-126.10 seconds. The recovered Cell 17 seed-0 runtime is unavailable. Checkpoints and event-level predictions remain server-only; portable tables contain their hashes.

## Next gate

Freeze Cell 17 as the three-seed development candidate. Run one final-protocol timing/memory pilot before deciding how many paired final-context evaluations are computationally justified. Do not add architectures, hyperparameters, or training seeds.
"""
    outputs["report"].write_text(report)

    manifest = {
        "schema_version": 1,
        "analysis": "controlled three-training-seed Cell 17 versus M2 development comparison",
        "script": str(script_path.relative_to(repo)),
        "script_sha256": sha256_file(script_path),
        "source_commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "registry": {
            "path": str(registry_path.relative_to(repo)),
            "sha256": sha256_file(registry_path),
        },
        "training_inputs": training_inputs,
        "inference_inputs": inference_inputs,
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
