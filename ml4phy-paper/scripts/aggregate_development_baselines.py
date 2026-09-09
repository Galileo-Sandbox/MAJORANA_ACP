#!/usr/bin/env python3
"""Compare recovered models and a context-only kernel on frozen development roles."""

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
from evaluate_fixed_protocol import event_metrics

MODEL_IDS = ("v9", "v5", "cell17", "true_cnp", "base1", "base2", "base3")
DISPLAY_NAMES = {
    "v9": "Cell 15 v9",
    "v5": "Cell 15 v5",
    "cell17": "Cell 17",
    "true_cnp": "True CNP",
    "base1": "Base 1",
    "base2": "Base 2",
    "base3": "Base 3",
    "kernel": "Kernel",
}
CONTEXT_SEEDS = tuple(range(100, 110))
FIXED_DROPOUT_SEED = 10100
BANDWIDTHS_KEV = (2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
CV_FOLDS = 5
CV_SEED = 20260908
PRIMARY_REGIONS = ("full", "equal_region_mean")


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
    if not rows:
        raise ValueError(f"No rows available for {path}")
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_dir_name(model: str, seed: int) -> str:
    if model in {"v9", "v5", "cell17"} and seed == 100:
        return f"20260908-{model}-dev-s100-mc50"
    if model in {"v9", "v5", "cell17"}:
        return f"20260908-{model}-dev-ctx-s{seed}-drop10100-mc50"
    return f"20260908-{model}-dev-ctx-s{seed}-drop10100-mc50"


def nw_predict(
    train_energy: np.ndarray,
    train_outcome: np.ndarray,
    query_energy: np.ndarray,
    bandwidth_kev: float,
    *,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.empty(query_energy.size, dtype=np.float64)
    effective_n = np.empty(query_energy.size, dtype=np.float64)
    for start in range(0, query_energy.size, chunk_size):
        stop = min(start + chunk_size, query_energy.size)
        scaled = (query_energy[start:stop, None] - train_energy[None, :]) / bandwidth_kev
        log_weight = -0.5 * scaled**2
        log_weight -= np.max(log_weight, axis=1, keepdims=True)
        weight = np.exp(log_weight)
        weight_sum = weight.sum(axis=1)
        predictions[start:stop] = (weight @ train_outcome) / weight_sum
        effective_n[start:stop] = weight_sum**2 / np.square(weight).sum(axis=1)
    return predictions, effective_n


def select_bandwidth(energy: np.ndarray, outcome: np.ndarray) -> tuple[float, list[dict]]:
    rng = np.random.default_rng(CV_SEED)
    fold_id = np.empty(energy.size, dtype=np.int64)
    fold_id[rng.permutation(energy.size)] = np.arange(energy.size) % CV_FOLDS
    rows = []
    for bandwidth in BANDWIDTHS_KEV:
        scores = []
        for fold in range(CV_FOLDS):
            valid = fold_id == fold
            prediction, _ = nw_predict(
                energy[~valid], outcome[~valid], energy[valid], bandwidth
            )
            scores.append(float(np.mean((prediction - outcome[valid]) ** 2)))
        rows.append(
            {
                "bandwidth_kev": bandwidth,
                "mean_cv_brier": float(np.mean(scores)),
                "std_cv_brier_across_folds": float(np.std(scores, ddof=1)),
            }
        )
    winner = min(rows, key=lambda row: (row["mean_cv_brier"], row["bandwidth_kev"]))
    return float(winner["bandwidth_kev"]), rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--runs-root", type=Path, default=Path("ml4phy-paper/runs"))
    parser.add_argument("--output-root", type=Path, default=Path("ml4phy-paper"))
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=Path("ml4phy-paper/manifests/frozen_protocol_v1.json"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    runs_root = (repo / args.runs_root).resolve()
    output_root = (repo / args.output_root).resolve()
    protocol_path = (repo / args.protocol_manifest).resolve()
    script_path = Path(__file__).resolve()

    outputs = {
        "summary": output_root / "tables/development_baseline_summary.csv",
        "paired": output_root / "tables/development_baseline_paired_differences.csv",
        "kernel": output_root / "tables/development_kernel_bandwidths.csv",
        "metrics": output_root / "tables/development_baseline_metrics.json",
        "figure": output_root / "figures/development_baseline_scores.png",
        "report": output_root / "reports/development_baselines.md",
        "manifest": output_root / "manifests/development_baselines.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        parser.error("Refusing to overwrite existing outputs: " + ", ".join(existing))
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    protocol = read_json(protocol_path)
    h5_path = repo / protocol["inputs"]["historical_development"]["logical_path"]
    if sha256_file(h5_path) != protocol["inputs"]["historical_development"]["sha256"]:
        raise ValueError("Development HDF5 hash does not match the frozen protocol.")
    threshold = float(protocol["threshold"]["value"])

    summaries: dict[str, dict[int, dict]] = {model: {} for model in MODEL_IDS}
    input_manifest = []
    event_archives: dict[int, dict[str, np.ndarray]] = {}
    target_hashes = set()
    source_commits = set()
    for model in MODEL_IDS:
        for seed in CONTEXT_SEEDS:
            run_dir = runs_root / run_dir_name(model, seed)
            summary_path = run_dir / "summary.json"
            event_path = run_dir / "event_predictions.npz"
            if not summary_path.is_file() or not event_path.is_file():
                parser.error(f"Missing required output under {run_dir}")
            summary = read_json(summary_path)
            expected = (
                "completed",
                model,
                "development",
                seed,
                FIXED_DROPOUT_SEED,
                2000,
                50,
            )
            observed = (
                summary["status"],
                summary["model"]["id"],
                summary["protocol"]["phase"],
                summary["randomness"]["context_seed"],
                summary["randomness"]["dropout_seed"],
                summary["counts"]["context_per_mc_pass"],
                summary["counts"]["mc_passes"],
            )
            if observed != expected:
                raise ValueError(f"Run contract mismatch for {run_dir.name}: {observed}")
            if summary["source_worktree_status"]:
                raise ValueError(f"Run used a dirty worktree: {run_dir.name}")
            if sha256_file(event_path) != summary["server_only_outputs"]["event_predictions"]["sha256"]:
                raise ValueError(f"Event prediction hash mismatch for {run_dir.name}")
            target_hashes.add(summary["protocol"]["target_identity_sha256"])
            source_commits.add(summary["source_commit"])
            summaries[model][seed] = summary
            if model == "cell17":
                with np.load(event_path) as archive:
                    event_archives[seed] = {key: archive[key].copy() for key in archive.files}
            input_manifest.append(
                {
                    "run_id": run_dir.name,
                    "summary_sha256": sha256_file(summary_path),
                    "event_predictions_sha256": sha256_file(event_path),
                }
            )
    if len(target_hashes) != 1:
        raise ValueError("Runs do not share one target identity.")

    model_draw_metrics: dict[str, dict[int, dict[str, float]]] = {
        model: {} for model in (*MODEL_IDS, "kernel")
    }
    for model in MODEL_IDS:
        for seed in CONTEXT_SEEDS:
            model_draw_metrics[model][seed] = {
                row["region"]: row["brier"]
                for row in summaries[model][seed]["metrics"]["event"]
            }

    bandwidth_rows = []
    kernel_support = {}
    with h5py.File(h5_path, "r") as handle:
        for seed in CONTEXT_SEEDS:
            archive = event_archives[seed]
            context_rows = archive["context_rows"].astype(np.int64)
            context_energy = handle["energy"][context_rows].astype(np.float64)
            context_score = handle["score"][context_rows].astype(np.float64)
            context_outcome = (context_score >= threshold).astype(np.float64)
            target_energy = archive["target_energy_kev"].astype(np.float64)
            target_outcome = archive["outcome"].astype(np.float64)
            selected, cv_rows = select_bandwidth(context_energy, context_outcome)
            prediction, effective_n = nw_predict(
                context_energy, context_outcome, target_energy, selected
            )
            model_draw_metrics["kernel"][seed] = {
                row["region"]: row["brier"]
                for row in event_metrics(target_energy, prediction, target_outcome)
            }
            kernel_support[seed] = {
                "minimum_effective_n": float(effective_n.min()),
                "median_effective_n": float(np.median(effective_n)),
            }
            for row in cv_rows:
                bandwidth_rows.append(
                    {
                        "context_seed": seed,
                        "selected": row["bandwidth_kev"] == selected,
                        **row,
                    }
                )

    regions = tuple(
        region
        for region, value in model_draw_metrics["cell17"][100].items()
        if value is not None
    )
    summary_rows = []
    for model in (*MODEL_IDS, "kernel"):
        for region in regions:
            values = np.asarray(
                [model_draw_metrics[model][seed][region] for seed in CONTEXT_SEEDS],
                dtype=np.float64,
            )
            summary_rows.append(
                {
                    "model": DISPLAY_NAMES[model],
                    "model_id": model,
                    "comparison_role": (
                        "frozen_candidate"
                        if model == "cell17"
                        else "context_only_baseline"
                        if model == "kernel"
                        else "recovered_historical_model"
                    ),
                    "training_seed_count": 0 if model == "kernel" else 1,
                    "context_draw_count": len(CONTEXT_SEEDS),
                    "region": region,
                    "brier_mean": float(values.mean()),
                    "brier_std_across_context": float(values.std(ddof=1)),
                    "brier_min": float(values.min()),
                    "brier_max": float(values.max()),
                }
            )

    paired_rows = []
    for baseline in (*MODEL_IDS, "kernel"):
        if baseline == "cell17":
            continue
        for region in regions:
            differences = np.asarray(
                [
                    model_draw_metrics["cell17"][seed][region]
                    - model_draw_metrics[baseline][seed][region]
                    for seed in CONTEXT_SEEDS
                ],
                dtype=np.float64,
            )
            paired_rows.append(
                {
                    "model_a": "Cell 17",
                    "model_b": DISPLAY_NAMES[baseline],
                    "region": region,
                    "paired_draw_count": differences.size,
                    "mean_brier_difference_a_minus_b": float(differences.mean()),
                    "std_brier_difference_across_context": float(differences.std(ddof=1)),
                    "a_better_draws": int(np.sum(differences < 0)),
                    "ties": int(np.sum(differences == 0)),
                    "b_better_draws": int(np.sum(differences > 0)),
                }
            )

    write_csv(outputs["summary"], summary_rows)
    write_csv(outputs["paired"], paired_rows)
    write_csv(outputs["kernel"], bandwidth_rows)

    selected_bandwidths = [
        row["bandwidth_kev"] for row in bandwidth_rows if row["selected"]
    ]
    primary_means = {
        model: {
            region: next(
                row["brier_mean"]
                for row in summary_rows
                if row["model_id"] == model and row["region"] == region
            )
            for region in PRIMARY_REGIONS
        }
        for model in (*MODEL_IDS, "kernel")
    }
    metrics = {
        "schema_version": 1,
        "frozen_candidate": "cell17",
        "primary_means": primary_means,
        "kernel": {
            "kind": "Gaussian Nadaraya-Watson regression of binary passage flags",
            "information_budget": "2,000 context events only",
            "candidate_bandwidths_kev": list(BANDWIDTHS_KEV),
            "cv_folds": CV_FOLDS,
            "cv_seed": CV_SEED,
            "selected_bandwidth_kev_by_context_seed": dict(
                zip(map(str, CONTEXT_SEEDS), selected_bandwidths, strict=True)
            ),
            "effective_support_by_context_seed": kernel_support,
        },
        "variation_sources": {
            "training_seed": "one recovered seed for each neural model",
            "context_seed": list(CONTEXT_SEEDS),
            "dropout_seed": FIXED_DROPOUT_SEED,
        },
        "evaluation_source_commits": sorted(source_commits),
        "interpretation": (
            "Recovered neural baselines are secondary comparisons, not controlled ablations, "
            "because their historical training configurations differ."
        ),
    }
    with outputs["metrics"].open("x") as stream:
        json.dump(metrics, stream, indent=2, allow_nan=False)
        stream.write("\n")

    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for axis, region, title in zip(
        axes,
        PRIMARY_REGIONS,
        ("Global event-level Brier", "Equal supported-region Brier"),
        strict=True,
    ):
        for index, model in enumerate((*MODEL_IDS, "kernel")):
            values = [model_draw_metrics[model][seed][region] for seed in CONTEXT_SEEDS]
            axis.plot(
                CONTEXT_SEEDS,
                values,
                marker="o",
                ms=3,
                linewidth=1.3,
                color=colors[index],
                label=DISPLAY_NAMES[model],
            )
        axis.set_title(title)
        axis.set_xlabel("Context seed")
        axis.set_ylabel("Brier score (lower is better)")
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.grid(alpha=0.22)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(outputs["figure"], dpi=180)
    plt.close(fig)

    candidate_global = primary_means["cell17"]["full"]
    candidate_equal = primary_means["cell17"]["equal_region_mean"]
    kernel_global = primary_means["kernel"]["full"]
    kernel_equal = primary_means["kernel"]["equal_region_mean"]
    best_recovered = min(
        (model for model in MODEL_IDS if model not in {"v9", "v5", "cell17"}),
        key=lambda model: primary_means[model]["full"],
    )
    report = f"""# Development baseline comparison

Status: recovered-model and inference-only baseline evaluation complete on the frozen development protocol. No new model training contributed to this comparison.

## Result

Cell 17 remains the frozen candidate. Its mean global Brier score is {candidate_global:.9f}, compared with {kernel_global:.9f} for the context-only kernel and {primary_means[best_recovered]['full']:.9f} for the best recovered historical baseline, {DISPLAY_NAMES[best_recovered]}. The corresponding equal-supported-region means are {candidate_equal:.9f}, {kernel_equal:.9f}, and {primary_means[best_recovered]['equal_region_mean']:.9f}.

Each value is the mean over ten paired context draws with a fixed target. Neural models use dropout seed {FIXED_DROPOUT_SEED} and 50 MC passes. The kernel bandwidth is selected independently for each draw by {CV_FOLDS}-fold context-only cross-validation over {', '.join(f'{value:g}' for value in BANDWIDTHS_KEV)} keV; target outcomes are never used for bandwidth selection.

## Interpretation boundary

- True CNP and Base 1/2/3 are recovered historical models with different samplers, context ranges, attention settings, positional encodings, or other training choices. Their comparison with Cell 17 is informative but confounded and is not a controlled architecture ablation.
- All neural models have only training seed 0. The ten overlapping context draws share one 2,074-event development target and do not establish training-seed uncertainty.
- DEP and the sparse tail remain unsupported in the development target. The equal-region summary excludes unsupported regions by the frozen rule.
- The context-only kernel has a matched 2,000-event inference budget. Its bandwidth tuning cost and effective support are recorded in the tables and metrics JSON.

## Training gate

The recovered controls and the eligible kernel do not close the controlled M2-versus-density-guided comparison. Together with the candidate decision, this satisfies the gap-decision condition for one M2 training pilot at seed 0. A full ladder or three-seed campaign remains unauthorized until the pilot is checked for configuration parity, runtime, and a non-degenerate learning result.
"""
    outputs["report"].write_text(report)

    manifest = {
        "schema_version": 1,
        "analysis": "frozen-development recovered baselines and context-only kernel",
        "script": str(script_path.relative_to(repo)),
        "script_sha256": sha256_file(script_path),
        "source_commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "protocol": {
            "path": str(protocol_path.relative_to(repo)),
            "sha256": sha256_file(protocol_path),
        },
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
