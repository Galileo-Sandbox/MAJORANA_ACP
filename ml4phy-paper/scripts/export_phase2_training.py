#!/usr/bin/env python3
"""Validate Phase 2 training and export lightweight registries and provenance."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import torch
import yaml

BUDGETS = (2000, 5000)
ARCHITECTURES = ("m0", "m1", "m2", "ours")
TRAINING_SEEDS = (0, 1, 2)
PARAMETER_COUNTS = {"m0": 116482, "m1": 149250, "m2": 151682, "ours": 130693}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def checkpoint_summary(path: Path, architecture: str) -> tuple[int, float]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    count = sum(int(tensor.numel()) for tensor in payload["model_state"].values())
    history = payload["history"]
    if len(history["step"]) != 3000 or int(history["step"][-1]) != 2999:
        raise ValueError(f"Incomplete checkpoint history: {path}")
    loss = float(history["loss"][-1])
    if count != PARAMETER_COUNTS[architecture] or not math.isfinite(loss):
        raise ValueError(f"Checkpoint parameter count or loss mismatch: {path}")
    return count, loss


def write_json(path: Path, payload: dict) -> None:
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    repo = Path(".").resolve()
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--short"], text=True
    ).splitlines()
    if status:
        raise RuntimeError("Training export requires a clean worktree: " + "; ".join(status))
    source_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    root = repo / "ml4phy-paper"
    campaign_path = root / "runs/phase2/campaigns/20260909-phase2-training-v1/campaign_record.json"
    phase2_config_path = root / "configs/phase2_training_v1.json"
    protocol_path = root / "manifests/phase2_protocol_v1.json"
    output_paths = {
        "registry_2000": root / "configs/phase2_models_b2000_v1.json",
        "registry_5000": root / "configs/phase2_models_b5000_v1.json",
        "table": root / "tables/phase2_training_runs.csv",
        "report": root / "reports/phase2_training_result.md",
        "manifest": root / "manifests/phase2_training_result.json",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))

    campaign = read_json(campaign_path)
    phase2_config = read_json(phase2_config_path)
    protocol = read_json(protocol_path)
    if campaign["status"] != "completed" or len(campaign["completed"]) != 24:
        raise ValueError("Phase 2 training campaign is incomplete")
    if campaign["failures"]:
        raise ValueError("Phase 2 campaign has a scientific failure")

    completed = {item["run_id"]: item for item in campaign["completed"]}
    rows = []
    runner_inputs = []
    registries = {}
    for budget in BUDGETS:
        subset = next(item for item in protocol["subsets"] if item["budget"] == budget)
        registry_models = {}
        common_pool_hash = None
        for architecture in ARCHITECTURES:
            for seed in TRAINING_SEEDS:
                base_id = f"20260909-phase2-b{budget}-{architecture}-seed{seed}-train3000"
                candidates = [run_id for run_id in completed if run_id == base_id or run_id.startswith(base_id + "-attempt")]
                if len(candidates) != 1:
                    raise ValueError(f"Expected one completed run for {base_id}, got {candidates}")
                run_id = candidates[0]
                run_dir = root / "runs/phase2/training" / run_id
                record_path = run_dir / "runner_record.json"
                config_path = run_dir / "resolved_config.yaml"
                checkpoint_path = run_dir / "artifacts/cnp.ckpt"
                pool_path = run_dir / "artifacts/training_pool.npz"
                summary_path = run_dir / "artifacts/run_summary.json"
                log_path = run_dir / "training.log"
                record = read_json(record_path)
                summary = read_json(summary_path)
                with config_path.open() as stream:
                    resolved = yaml.safe_load(stream)
                if record["status"] != "completed" or record["returncode"] != 0:
                    raise ValueError(f"Run did not complete: {run_id}")
                if record["subset_file_sha256"] != subset["sha256"]:
                    raise ValueError(f"Subset file mismatch: {run_id}")
                if record["subset_identity_sha256"] != subset["identity_sha256"]:
                    raise ValueError(f"Subset identity mismatch: {run_id}")
                if resolved["train_predictions_path"] != subset["logical_path"]:
                    raise ValueError(f"Resolved subset path mismatch: {run_id}")
                if int(summary["n_train_events"]) != budget:
                    raise ValueError(f"Nominal count mismatch: {run_id}")
                if record["post_run_validation"]["training_pool_retained_events"] != subset["sampling_eligible_unique_events"]:
                    raise ValueError(f"Effective count mismatch: {run_id}")
                for name, metadata in record["outputs"].items():
                    output = run_dir / "artifacts" / name
                    if output.stat().st_size != metadata["bytes"] or sha256_file(output) != metadata["sha256"]:
                        raise ValueError(f"Output hash mismatch: {output}")
                parameter_count, final_loss = checkpoint_summary(checkpoint_path, architecture)
                warning_lines = [
                    line.strip()
                    for line in log_path.read_text().splitlines()
                    if "warning" in line.lower() or "nan" in line.lower()
                ]
                if warning_lines:
                    raise ValueError(f"Unexpected training warning: {run_id}: {warning_lines}")
                pool_hash = sha256_file(pool_path)
                if common_pool_hash is None:
                    common_pool_hash = pool_hash
                elif pool_hash != common_pool_hash:
                    raise ValueError(f"Training pool summary differs within budget {budget}")
                model_id = f"b{budget}_{architecture}_seed{seed}"
                registry_models[model_id] = {
                    "display_name": f"{phase2_config['method_names'][architecture]} (budget {budget:,}, seed {seed})",
                    "architecture": architecture,
                    "config": str(config_path.relative_to(repo)),
                    "config_sha256": sha256_file(config_path),
                    "checkpoint": str(checkpoint_path.relative_to(repo)),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "training_seed": seed,
                    "training_budget": budget,
                    "sampling_eligible_unique_events": subset["sampling_eligible_unique_events"],
                    "density_pool_unique_events": budget,
                    "parameter_count": parameter_count,
                    "comparison_role": "phase2_fixed_compute_training_budget",
                    "runner_record": str(record_path.relative_to(repo)),
                    "runner_record_sha256": sha256_file(record_path),
                }
                rows.append(
                    {
                        "training_budget": budget,
                        "sampling_eligible_unique_events": subset["sampling_eligible_unique_events"],
                        "architecture": architecture,
                        "method": phase2_config["method_names"][architecture],
                        "training_seed": seed,
                        "steps": 3000,
                        "parameter_count": parameter_count,
                        "final_training_loss": format(final_loss, ".17g"),
                        "wall_seconds": format(float(record["wall_seconds"]), ".17g"),
                        "status": "completed",
                        "warning_count": 0,
                        "density_pool_unique_events": budget,
                        "checkpoint_sha256": sha256_file(checkpoint_path),
                        "run_id": run_id,
                    }
                )
                runner_inputs.append(
                    {
                        "run_id": run_id,
                        "runner_record_sha256": sha256_file(record_path),
                        "resolved_config_sha256": sha256_file(config_path),
                        "training_log_sha256": sha256_file(log_path),
                        "run_summary_sha256": sha256_file(summary_path),
                        "checkpoint_sha256": sha256_file(checkpoint_path),
                        "training_pool_sha256": pool_hash,
                    }
                )
        registries[budget] = {
            "schema_version": 1,
            "status": "phase2_budget_model_registry",
            "source_commit": source_commit,
            "training_budget": budget,
            "sampling_eligible_unique_events": subset["sampling_eligible_unique_events"],
            "density_pool_unique_events": budget,
            "training_pool_sha256": common_pool_hash,
            "training_predictions": subset["logical_path"],
            "training_predictions_sha256": subset["sha256"],
            "training_identity_sha256": subset["identity_sha256"],
            "classifier_config": phase2_config["fixed_classifier"],
            "classifier_config_sha256": "ad8f147aabe1fcd13f79932eaaa2950a0054253120f10b175692420c0236429f",
            "phase2_protocol": str(protocol_path.relative_to(repo)),
            "phase2_protocol_sha256": sha256_file(protocol_path),
            "models": registry_models,
        }

    write_json(output_paths["registry_2000"], registries[2000])
    write_json(output_paths["registry_5000"], registries[5000])
    write_csv(output_paths["table"], rows)

    table_lines = "\n".join(
        f"| {row['training_budget']:,} | {row['method']} | {row['training_seed']} | "
        f"{float(row['final_training_loss']):.6f} | {float(row['wall_seconds']):.2f} |"
        for row in rows
    )
    report = f"""# Phase 2 acceptance-model training result

Status: all 24 approved jobs completed. The summed job wall time was {campaign['summed_job_wall_seconds']:.2f} seconds ({campaign['summed_job_wall_seconds'] / 60:.2f} minutes), below the pre-execution 35.33-minute projection.

| Budget | Method | Seed | Final loss | Wall time (s) |
|---:|---|---:|---:|---:|
{table_lines}

The 2,000-event nominal pool retained 1,895 sampling-eligible events in 159 bins. The 5,000-event nominal pool retained 4,984 events in 205 bins. All jobs used the fixed 3,000-step schedule, sampled with replacement, and produced complete checkpoint histories through step 2,999. No training log contained a warning or non-finite value.

Every resolved configuration points to its matching server-only subset HDF5. For Density-guided CNP, this also forces the reconstructed density buffer to contain exactly 2,000 or 5,000 nominal events; no smaller-budget run references the original full pool. Checkpoints and event-level subset files remain server-only.

One initial Density-guided CNP budget-2,000 seed-0 attempt stopped before optimization because the default execution sandbox hid the CUDA device. Its directory and record are preserved. The uniquely named attempt 2 completed normally and is the only version admitted to evaluation.

This is fixed-compute training-budget evidence on historically exposed data. Final performance requires the prespecified n=500 evaluation and must retain unfavorable outcomes.
"""
    output_paths["report"].write_text(report)

    portable_outputs = [
        output_paths["registry_2000"],
        output_paths["registry_5000"],
        output_paths["table"],
        output_paths["report"],
    ]
    manifest = {
        "schema_version": 1,
        "analysis": "Phase 2 fixed-compute acceptance-model training",
        "status": "completed",
        "source_commit": source_commit,
        "script": str(Path(__file__).resolve().relative_to(repo)),
        "script_sha256": sha256_file(Path(__file__)),
        "phase2_protocol_sha256": sha256_file(protocol_path),
        "phase2_config_sha256": sha256_file(phase2_config_path),
        "campaign_record_sha256": sha256_file(campaign_path),
        "completed_jobs": len(rows),
        "scientific_failures": campaign["failures"],
        "preserved_preoptimization_failed_attempts": campaign["failed_attempts"],
        "summed_job_wall_seconds": campaign["summed_job_wall_seconds"],
        "warnings": [],
        "runner_inputs": runner_inputs,
        "outputs": {
            str(path.relative_to(repo)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in portable_outputs
        },
        "historical_data_exposure": "Prospectively specified follow-up analysis on historically exposed data; not an untouched test.",
    }
    write_json(output_paths["manifest"], manifest)


if __name__ == "__main__":
    main()
