#!/usr/bin/env python3
"""Aggregate the fixed-dropout development context sweep and freeze a candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODELS = ("v9", "v5", "cell17")
DISPLAY_NAMES = {"v9": "Cell 15 v9", "v5": "Cell 15 v5", "cell17": "Cell 17"}
MODEL_COLORS = {"v9": "#4C78A8", "v5": "#F58518", "cell17": "#54A24B"}
CONTEXT_SEEDS = tuple(range(100, 110))
FIXED_DROPOUT_SEED = 10100
PRIMARY_METRICS = ("full", "equal_region_mean")
SUPPORTED_REGIONS = (
    "full",
    "Bi-214",
    "continuum_1700_2000",
    "SE",
    "continuum_2200_2400",
    "FE",
    "equal_region_mean",
)


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
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_dir_name(model: str, seed: int) -> str:
    if seed == 100:
        return f"20260908-{model}-dev-s100-mc50"
    return f"20260908-{model}-dev-ctx-s{seed}-drop10100-mc50"


def metric_map(summary: dict, family: str, key: str) -> dict[str, float | None]:
    return {row[family]: row.get(key) for row in summary["metrics"][family]}


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--runs-root", type=Path, default=Path("ml4phy-paper/runs"))
    parser.add_argument("--output-root", type=Path, default=Path("ml4phy-paper"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    runs_root = (repo / args.runs_root).resolve()
    output_root = (repo / args.output_root).resolve()
    script_path = Path(__file__).resolve()

    outputs = {
        "draws": output_root / "tables/development_context_draws.csv",
        "summary": output_root / "tables/development_context_summary.csv",
        "paired": output_root / "tables/development_context_paired_differences.csv",
        "contrasts": output_root / "tables/development_peak_contrasts.csv",
        "roughness": output_root / "tables/development_continuum_roughness.csv",
        "metrics": output_root / "tables/development_context_metrics.json",
        "score_figure": output_root / "figures/development_context_scores.png",
        "curve_figure": output_root / "figures/development_context_curves.png",
        "report": output_root / "reports/development_candidate.md",
        "manifest": output_root / "manifests/development_context_sweep.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        parser.error("Refusing to overwrite existing outputs: " + ", ".join(existing))
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict[int, dict]] = {model: {} for model in MODELS}
    curves: dict[str, dict[int, dict[str, np.ndarray]]] = {model: {} for model in MODELS}
    input_manifest = []
    target_hashes = set()
    context_hashes: dict[int, set[str]] = {seed: set() for seed in CONTEXT_SEEDS}
    checkpoint_hashes: dict[str, set[str]] = {model: set() for model in MODELS}
    source_commits = set()

    for model in MODELS:
        for seed in CONTEXT_SEEDS:
            run_dir = runs_root / run_dir_name(model, seed)
            summary_path = run_dir / "summary.json"
            curve_path = run_dir / "curve.npz"
            if not summary_path.is_file() or not curve_path.is_file():
                parser.error(f"Missing required run output under {run_dir}")
            summary = read_json(summary_path)
            expected = {
                "status": "completed",
                "model": model,
                "phase": "development",
                "context_seed": seed,
                "dropout_seed": FIXED_DROPOUT_SEED,
                "mc_passes": 50,
                "context_draw": 2000,
                "context_per_mc_pass": 2000,
            }
            observed = {
                "status": summary["status"],
                "model": summary["model"]["id"],
                "phase": summary["protocol"]["phase"],
                "context_seed": summary["randomness"]["context_seed"],
                "dropout_seed": summary["randomness"]["dropout_seed"],
                "mc_passes": summary["counts"]["mc_passes"],
                "context_draw": summary["counts"]["context_draw"],
                "context_per_mc_pass": summary["counts"]["context_per_mc_pass"],
            }
            if observed != expected:
                raise ValueError(f"Run contract mismatch for {run_dir.name}: {observed}")
            if summary["source_worktree_status"]:
                raise ValueError(f"Run used a dirty worktree: {run_dir.name}")
            if sha256_file(curve_path) != summary["server_only_outputs"]["curve"]["sha256"]:
                raise ValueError(f"Curve hash mismatch for {run_dir.name}")
            target_hashes.add(summary["protocol"]["target_identity_sha256"])
            context_hashes[seed].add(summary["protocol"]["context_draw_identity_sha256"])
            checkpoint_hashes[model].add(summary["model"]["checkpoint_sha256"])
            source_commits.add(summary["source_commit"])
            summaries[model][seed] = summary
            with np.load(curve_path) as archive:
                curves[model][seed] = {
                    "energy_kev": archive["energy_kev"].copy(),
                    "prediction": archive["prediction"].copy(),
                }
            input_manifest.append(
                {
                    "run_id": run_dir.name,
                    "summary_sha256": sha256_file(summary_path),
                    "curve_sha256": sha256_file(curve_path),
                }
            )

    if len(target_hashes) != 1:
        raise ValueError(f"Target identities differ across runs: {target_hashes}")
    for seed, hashes in context_hashes.items():
        if len(hashes) != 1:
            raise ValueError(f"Context identity differs across models for seed {seed}: {hashes}")
    if any(len(hashes) != 1 for hashes in checkpoint_hashes.values()):
        raise ValueError(f"A model changed checkpoint within the sweep: {checkpoint_hashes}")
    if len(source_commits) != 1:
        raise ValueError(f"Runs used multiple source commits: {source_commits}")

    draw_rows = []
    for model in MODELS:
        for seed in CONTEXT_SEEDS:
            summary = summaries[model][seed]
            events = {row["region"]: row for row in summary["metrics"]["event"]}
            for region in SUPPORTED_REGIONS:
                row = events[region]
                draw_rows.append(
                    {
                        "model": DISPLAY_NAMES[model],
                        "model_id": model,
                        "training_seed": summary["randomness"]["training_seed"],
                        "context_seed": seed,
                        "dropout_seed": FIXED_DROPOUT_SEED,
                        "region": region,
                        "n_events": row["n_events"],
                        "brier": row["brier"],
                        "log_loss": row["log_loss"],
                    }
                )

    summary_rows = []
    for model in MODELS:
        for region in SUPPORTED_REGIONS:
            rows = [row for row in draw_rows if row["model_id"] == model and row["region"] == region]
            brier = [float(row["brier"]) for row in rows]
            brier_mean, brier_std = mean_std(brier)
            log_loss = [float(row["log_loss"]) for row in rows if row["log_loss"] is not None]
            log_mean, log_std = mean_std(log_loss) if log_loss else (None, None)
            summary_rows.append(
                {
                    "model": DISPLAY_NAMES[model],
                    "model_id": model,
                    "training_seed_count": 1,
                    "context_draw_count": len(rows),
                    "fixed_dropout_seed": FIXED_DROPOUT_SEED,
                    "region": region,
                    "brier_mean": brier_mean,
                    "brier_std_across_context": brier_std,
                    "brier_min": min(brier),
                    "brier_max": max(brier),
                    "log_loss_mean": log_mean,
                    "log_loss_std_across_context": log_std,
                }
            )

    paired_rows = []
    pairs = (("cell17", "v5"), ("cell17", "v9"), ("v5", "v9"))
    for model_a, model_b in pairs:
        for region in SUPPORTED_REGIONS:
            differences = []
            for seed in CONTEXT_SEEDS:
                event_a = {row["region"]: row for row in summaries[model_a][seed]["metrics"]["event"]}
                event_b = {row["region"]: row for row in summaries[model_b][seed]["metrics"]["event"]}
                differences.append(float(event_a[region]["brier"] - event_b[region]["brier"]))
            difference_mean, difference_std = mean_std(differences)
            paired_rows.append(
                {
                    "model_a": DISPLAY_NAMES[model_a],
                    "model_b": DISPLAY_NAMES[model_b],
                    "region": region,
                    "paired_draw_count": len(differences),
                    "mean_brier_difference_a_minus_b": difference_mean,
                    "std_brier_difference_across_context": difference_std,
                    "a_better_draws": sum(value < 0 for value in differences),
                    "ties": sum(value == 0 for value in differences),
                    "b_better_draws": sum(value > 0 for value in differences),
                }
            )

    contrast_rows = []
    roughness_rows = []
    for model in MODELS:
        peak_names = [row["peak"] for row in summaries[model][100]["metrics"]["peak_sideband_contrast"]]
        for peak in peak_names:
            source = [
                next(
                    row
                    for row in summaries[model][seed]["metrics"]["peak_sideband_contrast"]
                    if row["peak"] == peak
                )
                for seed in CONTEXT_SEEDS
            ]
            values = [float(row["absolute_contrast_error"]) for row in source]
            value_mean, value_std = mean_std(values)
            contrast_rows.append(
                {
                    "model": DISPLAY_NAMES[model],
                    "model_id": model,
                    "peak": peak,
                    "status": source[0]["status"],
                    "center_events": source[0]["center_events"],
                    "sideband_events": source[0]["sideband_events"],
                    "absolute_contrast_error_mean": value_mean,
                    "absolute_contrast_error_std_across_context": value_std,
                }
            )
        for region in ("continuum_1700_2000", "continuum_2200_2400"):
            values = [
                float(
                    next(
                        row
                        for row in summaries[model][seed]["metrics"]["continuum_roughness"]
                        if row["region"] == region
                    )["mean_absolute_second_difference"]
                )
                for seed in CONTEXT_SEEDS
            ]
            value_mean, value_std = mean_std(values)
            roughness_rows.append(
                {
                    "model": DISPLAY_NAMES[model],
                    "model_id": model,
                    "region": region,
                    "grid_spacing_kev": 1.0,
                    "masd_mean": value_mean,
                    "masd_std_across_context": value_std,
                }
            )

    # Candidate rule: lowest mean on both prespecified primary Brier summaries,
    # with directionally favorable paired differences on every draw against each alternative.
    primary_means = {
        model: {
            region: next(
                row["brier_mean"]
                for row in summary_rows
                if row["model_id"] == model and row["region"] == region
            )
            for region in PRIMARY_METRICS
        }
        for model in MODELS
    }
    winner_by_metric = {
        region: min(MODELS, key=lambda model: primary_means[model][region])
        for region in PRIMARY_METRICS
    }
    if len(set(winner_by_metric.values())) != 1:
        raise RuntimeError(f"Primary metrics do not identify one candidate: {winner_by_metric}")
    candidate = next(iter(winner_by_metric.values()))
    for alternative in MODELS:
        if alternative == candidate:
            continue
        for region in PRIMARY_METRICS:
            paired = next(
                row
                for row in paired_rows
                if row["model_a"] == DISPLAY_NAMES[candidate]
                and row["model_b"] == DISPLAY_NAMES[alternative]
                and row["region"] == region
            )
            if paired["a_better_draws"] != len(CONTEXT_SEEDS):
                raise RuntimeError(
                    f"Candidate is not directionally consistent against {alternative} in {region}."
                )

    write_csv(outputs["draws"], draw_rows)
    write_csv(outputs["summary"], summary_rows)
    write_csv(outputs["paired"], paired_rows)
    write_csv(outputs["contrasts"], contrast_rows)
    write_csv(outputs["roughness"], roughness_rows)

    metric_payload = {
        "schema_version": 1,
        "candidate": candidate,
        "candidate_display_name": DISPLAY_NAMES[candidate],
        "selection_rule": (
            "Lowest mean development Brier on both global and equal-supported-region summaries, "
            "with favorable paired differences on all ten fixed-dropout context draws."
        ),
        "primary_means": primary_means,
        "winner_by_metric": winner_by_metric,
        "training_seed_count": 1,
        "context_seeds": list(CONTEXT_SEEDS),
        "fixed_dropout_seed": FIXED_DROPOUT_SEED,
        "mc_passes": 50,
        "target_identity_sha256": next(iter(target_hashes)),
        "context_identity_sha256_by_seed": {
            str(seed): next(iter(context_hashes[seed])) for seed in CONTEXT_SEEDS
        },
        "checkpoint_sha256": {
            model: next(iter(checkpoint_hashes[model])) for model in MODELS
        },
        "source_commit": next(iter(source_commits)),
        "caveats": [
            "All three recovered models have only training seed 0.",
            "Context draws overlap and share one development target; they are paired sensitivity checks, not independent datasets.",
            "The fixed dropout seed isolates context selection but does not quantify dropout-estimator variation.",
            "DEP has 17 events and the sparse tail has zero events in the development target; both are inconclusive.",
        ],
    }
    with outputs["metrics"].open("x") as stream:
        json.dump(metric_payload, stream, indent=2, allow_nan=False)
        stream.write("\n")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    for axis, region, title in zip(
        axes,
        PRIMARY_METRICS,
        ("Global event-level Brier", "Equal supported-region Brier"),
        strict=True,
    ):
        for model in MODELS:
            values = [
                next(
                    float(row["brier"])
                    for row in draw_rows
                    if row["model_id"] == model
                    and row["context_seed"] == seed
                    and row["region"] == region
                )
                for seed in CONTEXT_SEEDS
            ]
            axis.plot(CONTEXT_SEEDS, values, marker="o", ms=4, label=DISPLAY_NAMES[model], color=MODEL_COLORS[model])
        axis.set_title(title)
        axis.set_xlabel("Context seed")
        axis.set_ylabel("Brier score (lower is better)")
        axis.grid(alpha=0.25)
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0].legend(frameon=False)
    fig.savefig(outputs["score_figure"], dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.8), constrained_layout=True)
    windows = ((1500.0, 1665.0), (2025.0, 2675.0))
    for axis, (lower, upper) in zip(axes, windows, strict=True):
        for model in MODELS:
            energy = curves[model][100]["energy_kev"]
            prediction = np.stack([curves[model][seed]["prediction"] for seed in CONTEXT_SEEDS])
            mean = prediction.mean(axis=0)
            std = prediction.std(axis=0, ddof=1)
            mask = (energy >= lower) & (energy <= upper)
            axis.plot(energy[mask], mean[mask], label=DISPLAY_NAMES[model], color=MODEL_COLORS[model])
            axis.fill_between(energy[mask], (mean - std)[mask], (mean + std)[mask], color=MODEL_COLORS[model], alpha=0.16)
        axis.set_xlim(lower, upper)
        axis.set_ylabel("Predicted acceptance")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, loc="lower right", ncol=3)
    axes[1].set_xlabel("Energy (keV)")
    fig.savefig(outputs["curve_figure"], dpi=180)
    plt.close(fig)

    full = {model: primary_means[model]["full"] for model in MODELS}
    equal = {model: primary_means[model]["equal_region_mean"] for model in MODELS}
    cell17_v5 = {
        row["region"]: row
        for row in paired_rows
        if row["model_a"] == "Cell 17" and row["model_b"] == "Cell 15 v5"
    }
    report = f"""# Development candidate decision

Status: candidate frozen from the development protocol. No new model training contributed to this decision.

## Decision

Cell 17 is the frozen density-guided candidate for the next conditional experiment. It had the lowest mean global and equal-supported-region event-level Brier score under every one of the ten paired context draws while the dropout seed was held fixed at {FIXED_DROPOUT_SEED}.

| Model | Global Brier, mean | Equal-region Brier, mean | Training seeds | Context draws |
|---|---:|---:|---:|---:|
| Cell 15 v9 | {full['v9']:.9f} | {equal['v9']:.9f} | 1 | 10 |
| Cell 15 v5 | {full['v5']:.9f} | {equal['v5']:.9f} | 1 | 10 |
| Cell 17 | {full['cell17']:.9f} | {equal['cell17']:.9f} | 1 | 10 |

Relative to Cell 15 v5, Cell 17's mean paired Brier difference was {cell17_v5['full']['mean_brier_difference_a_minus_b']:.9f} globally and {cell17_v5['equal_region_mean']['mean_brier_difference_a_minus_b']:.9f} for the equal-region summary; negative favors Cell 17. Both comparisons favored Cell 17 in 10/10 paired context draws. The result satisfies the prespecified preference rule because the learnable Cell 17 variant shows a consistent development benefit over the simpler fixed-floor alternatives.

## Scope and limitations

- The comparison uses one recovered training seed per model. Context-draw dispersion is reported separately and cannot substitute for training-seed replication.
- The ten context draws overlap, share the same 2,074-event development target, and use one fixed dropout seed. They are paired sensitivity checks, not ten independent datasets.
- DEP has only 17 development-target events and the sparse tail has none. Those regions remain inconclusive and were not used to make a favorable exception.
- Regional results are mixed: candidate selection uses the two frozen aggregate Brier criteria, while the peak-contrast and continuum-roughness tables preserve local trade-offs.
- A second campaign that varied context and dropout seeds together exists server-side but is excluded from the context-only dispersion reported here.

## Next gate

Run a fixed-context dropout-estimator sensitivity check and a 0.5-keV grid convergence check for Cell 17. Evaluate recovered CNP controls and the context-only kernel baseline under the same development protocol. If Cell 17 remains competitive, the gap decision authorizes one controlled M2 training pilot at training seed 0; it does not authorize a full training grid.
"""
    outputs["report"].write_text(report)

    manifest = {
        "schema_version": 1,
        "analysis": "fixed-dropout development context sweep",
        "script": str(script_path.relative_to(repo)),
        "script_sha256": sha256_file(script_path),
        "source_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "source_worktree_status": subprocess.check_output(["git", "-C", str(repo), "status", "--short"], text=True).splitlines(),
        "inputs": input_manifest,
        "outputs": {
            display_path(path, repo): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for key, path in outputs.items()
            if key != "manifest"
        },
        "candidate": candidate,
        "variation_sources": {
            "training_seed": "fixed at recovered seed 0",
            "context_seed": list(CONTEXT_SEEDS),
            "dropout_seed": f"fixed at {FIXED_DROPOUT_SEED}",
        },
    }
    with outputs["manifest"].open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
