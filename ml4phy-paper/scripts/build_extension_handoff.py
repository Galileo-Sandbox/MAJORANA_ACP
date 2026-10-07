#!/usr/bin/env python3
"""Build the Phase 1 extension synthesis, data workflow, and Phase 2 cost gate."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


METHOD_ORDER = (
    "CNP",
    "Attentive CNP",
    "Attentive CNP + PE",
    "Density-guided CNP (ours)",
    "Gaussian kernel regression (context only)",
    "Gaussian kernel regression (pooled data + context)",
)
COLORS = {
    "CNP": "#4c78a8",
    "Attentive CNP": "#f58518",
    "Attentive CNP + PE": "#54a24b",
    "Density-guided CNP (ours)": "#e45756",
    "Gaussian kernel regression (context only)": "#72b7b2",
    "Gaussian kernel regression (pooled data + context)": "#b279a2",
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def add_box(ax, xy, width, height, text, facecolor, fontsize=10.5):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.015",
        linewidth=1.4,
        edgecolor="#333333",
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.25,
    )


def add_arrow(ax, start, end, label=None, offset=(0, 0)):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.4,
            color="#555555",
            connectionstyle="arc3,rad=0.0",
        )
    )
    if label:
        ax.text(
            (start[0] + end[0]) / 2 + offset[0],
            (start[1] + end[1]) / 2 + offset[1],
            label,
            ha="center",
            va="center",
            fontsize=9,
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
        )


def workflow_figure(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Data and information budget for the ML4PS extension", fontsize=20, pad=18)

    add_box(ax, (0.03, 0.72), 0.19, 0.14, "Eligible train split\n377,330 unique events\n500–3,000 keV", "#dbe9f6")
    add_box(ax, (0.30, 0.72), 0.19, 0.14, "Classifier training\n18,866 unique events\nfixed 5%, seed 0", "#bdd7ee")
    add_box(ax, (0.57, 0.72), 0.18, 0.14, "Classifier exposure\n50 epochs\n943,300 repeated draws", "#d9e6f2")
    add_arrow(ax, (0.22, 0.79), (0.30, 0.79), "without replacement", (0, 0.035))
    add_arrow(ax, (0.49, 0.79), (0.57, 0.79), "weighted sampling", (0, 0.035))

    add_box(ax, (0.30, 0.47), 0.19, 0.14, "Acceptance input\n18,866 same identities\n18,836 sample-eligible", "#fce1c5")
    add_box(ax, (0.57, 0.47), 0.18, 0.14, "Acceptance training\n3,000 steps per seed\nwith replacement", "#f8cfa6")
    add_box(ax, (0.79, 0.47), 0.18, 0.14, "Density pool (ours)\n18,866 same identities\nenergy-only buffer", "#f8cfa6")
    add_arrow(ax, (0.395, 0.72), (0.395, 0.61), "same events", (0.052, 0))
    add_arrow(ax, (0.49, 0.54), (0.57, 0.54), "training trials", (0, 0.035))
    add_arrow(ax, (0.49, 0.50), (0.79, 0.50), "density estimation", (0, -0.035))

    add_box(ax, (0.03, 0.21), 0.19, 0.14, "Disjoint test split\n141,474 unique events\nall classifier-scored", "#dff0d8")
    add_box(ax, (0.29, 0.24), 0.17, 0.10, "Threshold calibration\n2,000 events\nT = 0.540643573", "#cce8c5", 9.8)
    add_box(ax, (0.50, 0.24), 0.18, 0.10, "Development\n3,000 context reservoir\n2,074 targets", "#cce8c5", 9.8)
    add_box(ax, (0.72, 0.24), 0.24, 0.10, "Final follow-up\n20,000 context reservoir\n114,400 fixed targets", "#cce8c5", 9.8)
    add_arrow(ax, (0.22, 0.28), (0.29, 0.28))
    add_arrow(ax, (0.22, 0.28), (0.50, 0.28))
    add_arrow(ax, (0.22, 0.28), (0.72, 0.28))

    ax.text(
        0.5,
        0.085,
        "Unique measured-event union used for classifier training or score evaluation: "
        "160,340 = 18,866 + 141,474 (disjoint).\n"
        "Roles reuse identities and must not be summed. Evaluation is not training, but its cost is disclosed.",
        ha="center",
        va="center",
        fontsize=11.5,
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#f3f3f3", "edgecolor": "#777777"},
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def combined_figure(path: Path, rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharex=True)
    metrics = (
        ("peak_mae_percentage_points_mean", "Prespecified peak-region MAE"),
        ("continuum_mae_percentage_points_mean", "Prespecified continuum-region MAE"),
    )
    for ax, (column, title) in zip(axes, metrics):
        for method in METHOD_ORDER:
            selected = [r for r in rows if r["method"] == method and r["status"] == "completed"]
            selected.sort(key=lambda item: int(item["context_size"]))
            ax.plot(
                [int(r["context_size"]) for r in selected],
                [float(r[column]) for r in selected],
                marker="o",
                linewidth=2,
                markersize=6,
                color=COLORS[method],
                label=method,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks([250, 500, 1000, 2000], labels=["250", "500", "1,000", "2,000"])
        ax.set_xlabel("Context events")
        ax.set_ylabel("MAE (percentage points)")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9.5)
    fig.text(
        0.5,
        0.075,
        "Lines show means only; neural training-seed and context variation are kept separate from kernel context variation in the CSV.",
        ha="center",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_comparison(neural_rows: list[dict], kernel_rows: list[dict]) -> list[dict]:
    rows = []
    for item in neural_rows:
        rows.append(
            {
                "status": "completed",
                "method": item["method"],
                "estimator_family": "pretrained neural",
                "context_size": int(item["context_size"]),
                "classifier_training_unique_events": 18866,
                "acceptance_pretraining_unique_events": 18866,
                "final_target_events": 114400,
                "replication": "3 training seeds x 10 overlapping contexts",
                "peak_mae_percentage_points_mean": item["peak_region_mean_mae_percentage_points_mean_of_seed_means"],
                "peak_training_seed_sd": item["peak_region_mean_mae_percentage_points_sd_across_seed_means"],
                "peak_context_sd": item["peak_region_mean_mae_percentage_points_mean_context_sd_within_seed"],
                "continuum_mae_percentage_points_mean": item["continuum_region_mean_mae_percentage_points_mean_of_seed_means"],
                "continuum_training_seed_sd": item["continuum_region_mean_mae_percentage_points_sd_across_seed_means"],
                "continuum_context_sd": item["continuum_region_mean_mae_percentage_points_mean_context_sd_within_seed"],
                "sparse_tail_mae_percentage_points_mean": item["sparse_tail_mae_percentage_points_mean_of_seed_means"],
                "global_brier_mean": item["global_brier_mean_of_seed_means"],
                "information_budget_note": "conditional on classifier and acceptance-model pretraining",
            }
        )
    for item in kernel_rows:
        rows.append(
            {
                "status": "completed",
                "method": item["method"],
                "estimator_family": "Gaussian kernel probability estimator",
                "context_size": int(item["context_size"]),
                "classifier_training_unique_events": 18866,
                "acceptance_pretraining_unique_events": int(item["pretraining_pool_events"]),
                "final_target_events": 114400,
                "replication": "10 overlapping contexts",
                "peak_mae_percentage_points_mean": item["peak_region_mean_mae_percentage_points_mean"],
                "peak_training_seed_sd": "",
                "peak_context_sd": item["peak_region_mean_mae_percentage_points_context_sd"],
                "continuum_mae_percentage_points_mean": item["continuum_region_mean_mae_percentage_points_mean"],
                "continuum_training_seed_sd": "",
                "continuum_context_sd": item["continuum_region_mean_mae_percentage_points_context_sd"],
                "sparse_tail_mae_percentage_points_mean": item["sparse_tail_mae_percentage_points_mean"],
                "global_brier_mean": item["global_brier_mean"],
                "information_budget_note": (
                    "context only; no acceptance pretraining"
                    if item["variant"] == "context_only"
                    else "pooled acceptance-training events plus context"
                ),
            }
        )
    rows.append(
        {
            "status": "pilot_only_campaign_stopped_by_cost_gate",
            "method": "Gaussian-process probability estimator",
            "estimator_family": "dense Bernoulli GP with Laplace approximation",
            "context_size": 2000,
            "classifier_training_unique_events": 18866,
            "acceptance_pretraining_unique_events": 0,
            "final_target_events": 114400,
            "replication": "one RBF timing pilot; no final scoring",
            "peak_mae_percentage_points_mean": "",
            "peak_training_seed_sd": "",
            "peak_context_sd": "",
            "continuum_mae_percentage_points_mean": "",
            "continuum_training_seed_sd": "",
            "continuum_context_sd": "",
            "sparse_tail_mae_percentage_points_mean": "",
            "global_brier_mean": "",
            "information_budget_note": "context only; final campaign requires amended-protocol approval",
        }
    )
    return rows


def method_table(rows: list[dict], context_size: int) -> str:
    lines = [
        "| Method | Peak MAE (pp) | Continuum MAE (pp) | Acceptance pretraining |",
        "|---|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        item = next(
            row
            for row in rows
            if row["method"] == method and row["context_size"] == context_size
        )
        lines.append(
            f"| {method} | {float(item['peak_mae_percentage_points_mean']):.3f} | "
            f"{float(item['continuum_mae_percentage_points_mean']):.3f} | "
            f"{item['acceptance_pretraining_unique_events']:,} |"
        )
    return "\n".join(lines)


def main() -> None:
    repo = Path(".").resolve()
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--short"], text=True
    ).splitlines()
    if status:
        raise RuntimeError("Synthesis requires a clean worktree: " + "; ".join(status))
    source_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    root = repo / "ml4phy-paper"
    outputs = {
        "comparison": root / "tables/phase1_extension_method_comparison.csv",
        "comparison_figure": root / "figures/phase1_extension_context_efficiency.png",
        "workflow_figure": root / "figures/extension_data_workflow.png",
        "handoff": root / "reports/phase1_extension_handoff.md",
        "phase2": root / "reports/phase2_approval_request.md",
        "manifest": root / "manifests/phase1_extension_handoff.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))

    input_paths = {
        "data_budget": root / "manifests/data_budget_ledger.json",
        "training": root / "manifests/extension_training_result.json",
        "neural": root / "manifests/phase1_neural_result.json",
        "kernel": root / "manifests/phase1_kernel_result.json",
        "gp": root / "manifests/gp_pilot_gate.json",
        "mc": root / "manifests/mc_smoothness_result.json",
    }
    manifests = {name: read_json(path) for name, path in input_paths.items()}
    expected = {
        "training": "completed",
        "neural": "completed",
        "kernel": "completed",
        "gp": "campaign_stopped_for_approval",
        "mc": "completed",
    }
    for name, value in expected.items():
        if manifests[name]["status"] != value:
            raise ValueError(f"Unexpected {name} status: {manifests[name]['status']}")
    if manifests["data_budget"]["key_findings"]["classifier_training_unique"] != 18866:
        raise ValueError("Unexpected classifier-training count in data-budget manifest")

    neural_rows = read_csv(root / "tables/phase1_neural_context_size_summary.csv")
    kernel_rows = read_csv(root / "tables/phase1_kernel_context_size_summary.csv")
    comparison = build_comparison(neural_rows, kernel_rows)
    write_csv(outputs["comparison"], comparison)
    combined_figure(outputs["comparison_figure"], comparison)
    workflow_figure(outputs["workflow_figure"])

    training_rows = read_csv(root / "tables/extension_training_runs.csv")
    controlled_rows = read_csv(root / "tables/controlled_training_runs.csv")
    means = {
        "CNP": sum(float(r["wall_seconds"]) for r in training_rows if r["architecture"] == "m0") / 3,
        "Attentive CNP": sum(float(r["wall_seconds"]) for r in training_rows if r["architecture"] == "m1") / 3,
        "Attentive CNP + PE": sum(
            float(r["wall_seconds"])
            for r in controlled_rows
            if r["architecture"] == "M2 attentive PE10"
        ) / 3,
        "Density-guided CNP (ours)": sum(
            float(r["wall_seconds"])
            for r in controlled_rows
            if r["architecture"] == "Cell 17" and r["wall_seconds"]
        ) / 2,
    }
    phase2_training_seconds = 2 * 3 * sum(means.values())
    phase1_campaign_seconds = 2169.45
    phase2_evaluation_cells = 2 * 4 * 3 * 10
    phase2_evaluation_seconds = phase1_campaign_seconds / 419 * phase2_evaluation_cells
    phase2_total_seconds = phase2_training_seconds + phase2_evaluation_seconds
    phase2_contingency_seconds = 1.25 * phase2_total_seconds

    handoff = f"""# Phase 1 extension handoff

Status: Phase 1 is complete for the four neural architectures and both Gaussian-kernel information budgets. The dense Gaussian-process campaign was stopped by its frozen two-hour approval gate. Phase 2 has not started.

All results are prospectively specified follow-up analyses on historically exposed data, not an untouched test. The fixed classifier, threshold, target identities, original endpoints, and commit `48a59d0b2d93a6c76d2001522abcb98ac74432fb` remain preserved.

## Reused evidence

- The fixed classifier and threshold, all original target roles, 60 compatible neural cells at context size 2,000, and saved event predictions for the original local-shape export were reused.
- The archived Gaussian-kernel estimator and original development bandwidth choices were reproduced exactly before extending the context-size matrix.
- No classifier retraining, new dataset, optional gate control, or Phase 2 training was performed.

## New execution

- Six matched baseline models were trained: CNP and Attentive CNP, three 3,000-step seeds each. All six completed without warnings.
- The 480-cell neural matrix is complete: 60 archived cells, one reused pilot cell, and 419 new evaluations. Twenty-five cells whose provenance recorded a transient dirty worktree were rerun cleanly; arrays agreed to absolute tolerance 1e-12.
- Eighty Gaussian-kernel cells were evaluated across context-only and pooled-data budgets. The pooled control adds all 18,866 acceptance-input events to each context.
- One full 2,000-context dense Bernoulli-GP timing pilot completed. The scaling-aware campaign estimate was 2.41 hours, exceeding the frozen two-hour gate, so no complete GP comparison was launched.
- The nested MC diagnostic extended Attentive CNP + PE to 800 passes when its trigger fired. Material stream disagreement and grid dependence remained, so a smoother-curve claim is rejected.

## Completed method comparison

At context size 500:

{method_table(comparison, 500)}

At context size 2,000:

{method_table(comparison, 2000)}

Neural uncertainty columns separate the SD across three training-seed means from the mean within-seed SD across ten overlapping contexts. Kernel rows have context SD only. These contexts share targets and overlap; they are sensitivity replicates, not independent datasets.

## Supported claims and boundaries

- Among completed methods, Density-guided CNP has the lowest prespecified peak-region MAE at both headline context sizes. At n=500 it is 3.557 percentage points, compared with 5.996--9.612 for the other completed estimators.
- Its continuum MAE is 4.072 percentage points at n=500. The pooled-data kernel control is numerically similar at 4.056 but consumes the additional 18,866-event pool; this is not evidence that either method wins under equal total information.
- Neural errors change little from 250 to 2,000 context events. The result supports retained performance with a small context conditional on disclosed pretraining, not a strong context-scaling improvement and not total-data efficiency.
- The context-only kernel improves with more context but remains worse on the peak endpoint. The pooled kernel's strong sparse-tail result is retained and reported rather than suppressed.
- Brier remains secondary evidence. The local-error and fixed contrast analyses support localized reconstruction; the MC diagnostic does not support a smoother-curve claim.
- Dropout intervals cover only 0--3.8% of eligible empirical bins in the reported diagnostic and are not calibrated confidence intervals.

## Unresolved items

- No final Gaussian-process row exists because the agreed cost gate fired. A separately labeled sparse variational Bernoulli-GP protocol requires approval and resource testing.
- Common uncertainty coverage across dropout, kernel bootstrap, GP posterior, and finite-reference uncertainty remains undefined and was not invented after inspection.
- Acceptance-training-size efficiency remains untested. The exact Phase 2 slice and measured projection are in `phase2_approval_request.md`.
- Recovered seed-0 provenance limitations remain as recorded in the data-budget report.
"""
    outputs["handoff"].write_text(handoff)

    timing_lines = "\n".join(
        f"| {method} | {seconds:.3f} |" for method, seconds in means.items()
    )
    phase2 = f"""# Phase 2 approval request: acceptance-training-size slice

Status: awaiting explicit approval. No Phase 2 subset was materialized and no Phase 2 model was trained.

## Proposed frozen slice

- Keep the classifier, threshold, final targets, 3,000-step schedule, optimizer, and context size 500 fixed.
- Use acceptance-training budgets 2,000, 5,000, and 18,866. Reuse the completed 18,866-budget models when all protocol fields match.
- Before training, create one identity-hash-based, outcome-blind ordering of the 18,866-event input pool. Use nested 2,000- and 5,000-event prefixes for every architecture and seed.
- Recompute effective retained counts after the minimum-four-events-per-bin sampler filter. Do not remove target regions when a smaller pool lacks support.
- For Density-guided CNP, construct the density buffer only from the matching prefix and verify its identity hash; do not reconstruct the full pool.
- Train four architectures with three initialization seeds at each of the two smaller budgets: 24 new jobs. Evaluate the same ten existing context draws at n=500 for 240 full-target cells.
- Treat the result as a fixed-compute data-budget study, not an exhaustive per-method optimum. Preserve unfavorable outcomes.

## Measured cost basis

| Architecture | Mean measured 3,000-step training time (s) |
|---|---:|
{timing_lines}

The fixed-step training projection is **{phase2_training_seconds:.2f} seconds ({phase2_training_seconds / 60:.2f} minutes)** for 24 jobs. It assumes training time is approximately pool-size independent because the step and sampled trial-size schedules stay fixed; the actual times will still be recorded.

The completed Phase 1 neural campaign used 2,169.45 wall-seconds for 419 new cells. Scaling that measured campaign rate to 240 Phase 2 cells projects **{phase2_evaluation_seconds:.2f} seconds ({phase2_evaluation_seconds / 60:.2f} minutes)** for evaluation. Training plus evaluation is **{phase2_total_seconds:.2f} seconds ({phase2_total_seconds / 60:.2f} minutes)** sequentially before aggregation, or **{phase2_contingency_seconds:.2f} seconds ({phase2_contingency_seconds / 60:.2f} minutes)** with a 25% operational allowance. The slow-path measured GPU allocation was 6,726 MiB.

## Approval decision requested

Approve or reject the 24-job fixed slice above. Approval would authorize subset freezing, a short input-validation dry run, 24 sequential training jobs, and the matching 240-cell evaluation only. It would not authorize a full training-size by context-size factorial grid, classifier retraining, a gate control, or a GP campaign.
"""
    outputs["phase2"].write_text(phase2)

    output_records = {}
    for name, path in outputs.items():
        if name == "manifest":
            continue
        output_records[str(path.relative_to(repo))] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema_version": 1,
        "analysis": "Phase 1 extension synthesis and Phase 2 approval gate",
        "status": "phase1_completed_phase2_awaiting_approval",
        "source_commit": source_commit,
        "script": str(Path(__file__).resolve().relative_to(repo)),
        "script_sha256": sha256_file(Path(__file__)),
        "inputs": {
            str(path.relative_to(repo)): sha256_file(path)
            for path in input_paths.values()
        },
        "execution_summary": {
            "new_baseline_training_jobs": 6,
            "neural_cells_total": 480,
            "neural_cells_new": 419,
            "neural_cells_pilot_reused": 1,
            "neural_cells_archived_reused": 60,
            "kernel_cells": 80,
            "dense_gp_campaign_completed": False,
            "dense_gp_projected_hours": manifests["gp"]["projection"]["scaled_total_hours"],
            "smoother_curve_claim_supported": False,
        },
        "phase2_gate": {
            "status": "awaiting_explicit_approval",
            "new_training_jobs": 24,
            "evaluation_cells": phase2_evaluation_cells,
            "projected_training_seconds": phase2_training_seconds,
            "projected_evaluation_seconds": phase2_evaluation_seconds,
            "projected_total_seconds": phase2_total_seconds,
            "projected_total_with_25_percent_allowance_seconds": phase2_contingency_seconds,
            "campaign_launched": False,
        },
        "outputs": output_records,
        "historical_data_exposure": "Prospectively specified follow-up analyses on historically exposed data; not an untouched test.",
    }
    write_json(outputs["manifest"], manifest)


if __name__ == "__main__":
    main()
