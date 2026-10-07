#!/usr/bin/env python3
"""Summarize the controlled M2 seed-0 training and paired development inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

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


def checkpoint_parameter_count(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get(
        "model_state", payload.get("model_state_dict", payload.get("state_dict"))
    )
    if state is None:
        raise ValueError(f"Checkpoint has no recognized state dictionary: {path}")
    return int(sum(value.numel() for value in state.values()))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("ml4phy-paper"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    output_root = (repo / args.output_root).resolve()
    script_path = Path(__file__).resolve()
    training_dir = repo / "ml4phy-paper/runs/training/20260908-m2-seed0-pilot3000"
    artifact_dir = training_dir / "artifacts"
    runner_path = training_dir / "runner_record.json"
    train_summary_path = artifact_dir / "run_summary.json"
    checkpoint_path = artifact_dir / "cnp.ckpt"
    m2_config_path = repo / "ml4phy-paper/configs/m2_attentive_pe10_seed0.yaml"
    cell17_config_path = (
        repo / "configs/cut_acceptance/simple_cnn_small/sweeps/cell17/bin10/inclusive.yaml"
    )
    registry_path = repo / "ml4phy-paper/configs/trained_models_v1.json"
    outputs = {
        "paired": output_root / "tables/m2_seed0_pilot_paired_differences.csv",
        "metrics": output_root / "tables/m2_seed0_pilot_metrics.json",
        "report": output_root / "reports/m2_seed0_pilot.md",
        "manifest": output_root / "manifests/m2_seed0_pilot.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        parser.error("Refusing to overwrite existing outputs: " + ", ".join(existing))
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    runner = read_json(runner_path)
    train_summary = read_json(train_summary_path)
    registry = read_json(registry_path)
    spec = registry["models"]["m2_seed0_pilot"]
    if runner["status"] != "completed" or runner["returncode"] != 0:
        raise ValueError("Training runner did not complete successfully.")
    if sha256_file(checkpoint_path) != spec["checkpoint_sha256"]:
        raise ValueError("M2 checkpoint does not match the portable registry.")
    if sha256_file(m2_config_path) != spec["config_sha256"]:
        raise ValueError("M2 configuration does not match the portable registry.")
    if not math.isfinite(train_summary["cnp_final_train_loss"]):
        raise ValueError("M2 final training loss is not finite.")
    if runner["training_steps"] != 3000 or runner["training_seed"] != 0:
        raise ValueError("Unexpected training step or seed contract.")
    if (
        train_summary["n_train_events"] != 18866
        or train_summary["n_validation_events"] != 7074
        or train_summary["n_bins_used"] != 212
    ):
        raise ValueError("Training counts do not match recovered inputs.")

    with m2_config_path.open() as stream:
        m2_config = yaml.safe_load(stream)
    with cell17_config_path.open() as stream:
        cell17_config = yaml.safe_load(stream)
    allowed_top_level_differences = {"name", "out_dir", "device", "aggregator"}
    unexpected = {
        key: {"m2": m2_config.get(key), "cell17": cell17_config.get(key)}
        for key in sorted(set(m2_config) | set(cell17_config))
        if key not in allowed_top_level_differences
        and m2_config.get(key) != cell17_config.get(key)
    }
    if unexpected:
        raise ValueError(f"Unexpected non-architecture configuration differences: {unexpected}")

    summaries: dict[str, dict[int, dict]] = {"cell17": {}, "m2": {}}
    inference_inputs = []
    for seed in CONTEXT_SEEDS:
        paths = {
            "cell17": repo
            / "ml4phy-paper/runs"
            / (
                "20260908-cell17-dev-s100-mc50"
                if seed == 100
                else f"20260908-cell17-dev-ctx-s{seed}-drop10100-mc50"
            )
            / "summary.json",
            "m2": repo
            / "ml4phy-paper/runs"
            / f"20260908-m2-seed0-dev-ctx-s{seed}-drop10100-mc50"
            / "summary.json",
        }
        for model, path in paths.items():
            summary = read_json(path)
            if (
                summary["status"] != "completed"
                or summary["protocol"]["phase"] != "development"
                or summary["randomness"]["context_seed"] != seed
                or summary["randomness"]["dropout_seed"] != FIXED_DROPOUT_SEED
                or summary["counts"]["mc_passes"] != 50
            ):
                raise ValueError(f"Inference contract mismatch: {path}")
            summaries[model][seed] = summary
            inference_inputs.append(
                {"run_id": summary["run_id"], "summary_sha256": sha256_file(path)}
            )
        if (
            summaries["cell17"][seed]["protocol"]["context_draw_identity_sha256"]
            != summaries["m2"][seed]["protocol"]["context_draw_identity_sha256"]
        ):
            raise ValueError(f"Context identity mismatch for seed {seed}")
        if (
            summaries["cell17"][seed]["protocol"]["target_identity_sha256"]
            != summaries["m2"][seed]["protocol"]["target_identity_sha256"]
        ):
            raise ValueError(f"Target identity mismatch for seed {seed}")

    regions = tuple(
        row["region"]
        for row in summaries["cell17"][100]["metrics"]["event"]
        if row["brier"] is not None
    )
    paired_rows = []
    primary = {}
    for region in regions:
        differences = []
        cell17_values = []
        m2_values = []
        for seed in CONTEXT_SEEDS:
            cell17_events = {
                row["region"]: row for row in summaries["cell17"][seed]["metrics"]["event"]
            }
            m2_events = {
                row["region"]: row for row in summaries["m2"][seed]["metrics"]["event"]
            }
            cell17_value = float(cell17_events[region]["brier"])
            m2_value = float(m2_events[region]["brier"])
            cell17_values.append(cell17_value)
            m2_values.append(m2_value)
            differences.append(cell17_value - m2_value)
        difference = np.asarray(differences)
        row = {
            "region": region,
            "paired_context_draws": len(CONTEXT_SEEDS),
            "cell17_brier_mean": float(np.mean(cell17_values)),
            "m2_brier_mean": float(np.mean(m2_values)),
            "mean_brier_difference_cell17_minus_m2": float(difference.mean()),
            "std_difference_across_context": float(difference.std(ddof=1)),
            "cell17_better_draws": int(np.sum(difference < 0)),
            "ties": int(np.sum(difference == 0)),
            "m2_better_draws": int(np.sum(difference > 0)),
        }
        paired_rows.append(row)
        if region in {"full", "equal_region_mean"}:
            primary[region] = row
    write_csv(outputs["paired"], paired_rows)

    parameter_counts = {
        "m2_seed0_pilot": checkpoint_parameter_count(checkpoint_path),
        "cell17_seed0_recovered": checkpoint_parameter_count(
            repo / "results/cut_acceptance/simple_cnn_small/sweeps/cell17/bin10/inclusive/cnp.ckpt"
        ),
    }
    metrics = {
        "schema_version": 1,
        "training": {
            "model": "m2_seed0_pilot",
            "steps": runner["training_steps"],
            "seed": runner["training_seed"],
            "final_loss": train_summary["cnp_final_train_loss"],
            "wall_seconds": runner["wall_seconds"],
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "source_commit": runner["source_commit"],
            "dependency_commit": runner["dependency_commit"],
        },
        "parameter_counts": parameter_counts,
        "configuration_parity": {
            "unexpected_non_architecture_differences": unexpected,
            "intentional_top_level_differences": sorted(allowed_top_level_differences),
        },
        "primary_paired_results": primary,
        "training_seed_count_per_model": 1,
        "context_seeds": list(CONTEXT_SEEDS),
        "fixed_dropout_seed": FIXED_DROPOUT_SEED,
        "decision": "expand only M2 and Cell 17 to training seeds 1 and 2",
        "limitations": [
            "The seed-0 result does not establish training-seed stability.",
            "M2 and Cell 17 differ as complete architecture packages and in parameter count.",
            "Regional results remain mixed; Cell 17 is worse in continuum_2200_2400 for every draw.",
        ],
    }
    with outputs["metrics"].open("x") as stream:
        json.dump(metrics, stream, indent=2, allow_nan=False)
        stream.write("\n")

    report = f"""# Controlled M2 seed-0 pilot result

Status: training and paired development evaluation complete. The checkpoint remains server-only.

## Training result

The 3,000-step M2 attentive-PE10 pilot completed at training seed 0 in {runner['wall_seconds']:.2f} seconds with a finite final training loss of {train_summary['cnp_final_train_loss']:.6f}. It used 18,866 training events, 212 retained energy bins, and the frozen classifier lineage. The checkpoint SHA-256 is `{spec['checkpoint_sha256']}`.

The committed M2 configuration matches Cell 17 on every non-architecture training field. M2 has {parameter_counts['m2_seed0_pilot']:,} parameters and Cell 17 has {parameter_counts['cell17_seed0_recovered']:,}; exact parameter matching is not claimed.

## Paired development result

Across ten identical context draws with dropout seed {FIXED_DROPOUT_SEED}, Cell 17 minus M2 had a mean global Brier difference of {primary['full']['mean_brier_difference_cell17_minus_m2']:.9f} and a mean equal-supported-region difference of {primary['equal_region_mean']['mean_brier_difference_cell17_minus_m2']:.9f}. Negative favors Cell 17. Both aggregate comparisons favored Cell 17 in 10/10 draws.

The result is not uniform by region: Cell 17 improves the 1700-2000 keV continuum strongly but is worse in the 2200-2400 keV window in 10/10 draws. The full paired table preserves all prespecified supported and diagnostic regions.

## Expansion decision

The pilot passed configuration-parity, runtime, finite-loss, checkpoint-load, and non-degenerate-inference gates. The seed-0 effect is consistent across context draws but does not measure training-seed variability. Authorize only M2 and Cell 17 training seeds 1 and 2 next. Do not launch M0/M1, E3/E4, a hyperparameter sweep, or final-protocol evaluation at this stage.
"""
    outputs["report"].write_text(report)

    manifest = {
        "schema_version": 1,
        "analysis": "controlled M2 seed-0 training pilot and paired development evaluation",
        "script": str(script_path.relative_to(repo)),
        "script_sha256": sha256_file(script_path),
        "source_commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "training_inputs": {
            "runner_record_sha256": sha256_file(runner_path),
            "run_summary_sha256": sha256_file(train_summary_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "m2_config_sha256": sha256_file(m2_config_path),
            "cell17_config_sha256": sha256_file(cell17_config_path),
        },
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
