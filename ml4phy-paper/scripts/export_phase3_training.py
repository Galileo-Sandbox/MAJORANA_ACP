#!/usr/bin/env python3
"""Validate Phase 3 training and export lightweight model registries."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import torch
import yaml

SUBSET_IDS = ("seed20260910_n5000", "seed20260911_n5000", "original_n10000")
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


def write_json(path: Path, payload: dict) -> None:
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


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


def main() -> None:
    repo = Path(".").resolve()
    root = repo / "ml4phy-paper"
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--short"], text=True
    ).splitlines()
    if status:
        raise RuntimeError("Training export requires a clean worktree: " + "; ".join(status))
    source_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    campaign_path = root / "runs/phase3/campaigns/20260909-phase3-neural-training-v1/campaign_record.json"
    config_path = root / "configs/phase3_training_v1.json"
    protocol_path = root / "manifests/phase3_protocol_v1.json"
    table_path = root / "tables/phase3_training_runs.csv"
    report_path = root / "reports/phase3_training_result.md"
    manifest_path = root / "manifests/phase3_training_result.json"
    registry_paths = {
        subset_id: root / f"configs/phase3_models_{subset_id}_v1.json"
        for subset_id in SUBSET_IDS
    }
    outputs = [table_path, report_path, manifest_path, *registry_paths.values()]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))

    campaign = read_json(campaign_path)
    config = read_json(config_path)
    if campaign["status"] != "completed" or len(campaign["completed"]) != 36:
        raise ValueError("Phase 3 training campaign is incomplete")
    if campaign["failures"]:
        raise ValueError("Phase 3 training campaign contains failures")
    completed = {item["run_id"]: item for item in campaign["completed"]}

    rows = []
    runner_inputs = []
    registries = {}
    for subset_id in SUBSET_IDS:
        subset = next(item for item in config["subset_inputs"] if item["subset_id"] == subset_id)
        models = {}
        common_pool_hash = None
        for architecture in ARCHITECTURES:
            for seed in TRAINING_SEEDS:
                base_id = f"20260909-phase3-{subset_id}-{architecture}-seed{seed}-train3000"
                candidates = [
                    run_id for run_id in completed
                    if run_id == base_id or run_id.startswith(base_id + "-attempt")
                ]
                if len(candidates) != 1:
                    raise ValueError(f"Expected one completed run for {base_id}: {candidates}")
                run_id = candidates[0]
                run_dir = root / "runs/phase3/training" / run_id
                record_path = run_dir / "runner_record.json"
                resolved_path = run_dir / "resolved_config.yaml"
                checkpoint_path = run_dir / "artifacts/cnp.ckpt"
                pool_path = run_dir / "artifacts/training_pool.npz"
                summary_path = run_dir / "artifacts/run_summary.json"
                log_path = run_dir / "training.log"
                record = read_json(record_path)
                summary = read_json(summary_path)
                with resolved_path.open() as stream:
                    resolved = yaml.safe_load(stream)
                if record["status"] != "completed" or record["returncode"] != 0:
                    raise ValueError(f"Run did not complete: {run_id}")
                if record["subset_file_sha256"] != subset["sha256"]:
                    raise ValueError(f"Subset file mismatch: {run_id}")
                if record["subset_identity_sha256"] != subset["identity_sha256"]:
                    raise ValueError(f"Subset identity mismatch: {run_id}")
                if resolved["train_predictions_path"] != subset["logical_path"]:
                    raise ValueError(f"Resolved subset path mismatch: {run_id}")
                if resolved["training"]["batch_size"] != 16 or resolved["training"]["n_steps"] != 3000:
                    raise ValueError(f"Training schedule mismatch: {run_id}")
                if int(summary["n_train_events"]) != subset["nominal_unique_events"]:
                    raise ValueError(f"Nominal count mismatch: {run_id}")
                validation = record["post_run_validation"]
                if validation["training_pool_retained_events"] != subset["sampling_eligible_unique_events"]:
                    raise ValueError(f"Effective count mismatch: {run_id}")
                for name, metadata in record["outputs"].items():
                    output = run_dir / "artifacts" / name
                    if output.stat().st_size != metadata["bytes"] or sha256_file(output) != metadata["sha256"]:
                        raise ValueError(f"Output hash mismatch: {output}")
                parameter_count, final_loss = checkpoint_summary(checkpoint_path, architecture)
                warning_lines = [
                    line.strip() for line in log_path.read_text().splitlines()
                    if "warning" in line.lower() or "nan" in line.lower()
                ]
                if warning_lines:
                    raise ValueError(f"Unexpected training warning: {run_id}: {warning_lines}")
                pool_hash = sha256_file(pool_path)
                if common_pool_hash is None:
                    common_pool_hash = pool_hash
                elif common_pool_hash != pool_hash:
                    raise ValueError(f"Training pool differs within subset {subset_id}")

                model_id = f"p3_{subset_id}_{architecture}_seed{seed}"
                models[model_id] = {
                    "display_name": f"{config['method_names'][architecture]} ({subset_id}, seed {seed})",
                    "architecture": architecture,
                    "config": str(resolved_path.relative_to(repo)),
                    "config_sha256": sha256_file(resolved_path),
                    "checkpoint": str(checkpoint_path.relative_to(repo)),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "training_seed": seed,
                    "training_budget": subset["nominal_unique_events"],
                    "training_subset_id": subset_id,
                    "training_ordering_id": subset["ordering_id"],
                    "sampling_eligible_unique_events": subset["sampling_eligible_unique_events"],
                    "density_pool_unique_events": subset["density_pool_unique_events"],
                    "parameter_count": parameter_count,
                    "comparison_role": "phase3_fixed_compute_training_data_efficiency",
                    "runner_record": str(record_path.relative_to(repo)),
                    "runner_record_sha256": sha256_file(record_path),
                }
                rows.append({
                    "subset_id": subset_id,
                    "ordering_id": subset["ordering_id"],
                    "training_budget_nominal_events": subset["nominal_unique_events"],
                    "sampling_eligible_unique_events": subset["sampling_eligible_unique_events"],
                    "density_pool_unique_events": subset["density_pool_unique_events"],
                    "architecture_id": architecture,
                    "method": config["method_names"][architecture],
                    "training_seed": seed,
                    "steps": 3000,
                    "batch_size": 16,
                    "parameter_count": parameter_count,
                    "final_training_loss": format(final_loss, ".17g"),
                    "wall_seconds": format(float(record["wall_seconds"]), ".17g"),
                    "peak_gpu_memory_mib": record.get("peak_gpu_memory_mib"),
                    "warning_count": 0,
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "run_id": run_id,
                })
                runner_inputs.append({
                    "run_id": run_id,
                    "runner_record_sha256": sha256_file(record_path),
                    "resolved_config_sha256": sha256_file(resolved_path),
                    "training_log_sha256": sha256_file(log_path),
                    "run_summary_sha256": sha256_file(summary_path),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "training_pool_sha256": pool_hash,
                })
        registries[subset_id] = {
            "schema_version": 1,
            "status": "phase3_model_registry",
            "source_commit": source_commit,
            "training_subset_id": subset_id,
            "training_ordering_id": subset["ordering_id"],
            "training_budget": subset["nominal_unique_events"],
            "sampling_eligible_unique_events": subset["sampling_eligible_unique_events"],
            "density_pool_unique_events": subset["density_pool_unique_events"],
            "training_pool_sha256": common_pool_hash,
            "training_predictions": subset["logical_path"],
            "training_predictions_sha256": subset["sha256"],
            "training_identity_sha256": subset["identity_sha256"],
            "classifier_config": config["fixed_classifier"],
            "classifier_config_sha256": "ad8f147aabe1fcd13f79932eaaa2950a0054253120f10b175692420c0236429f",
            "phase3_protocol": str(protocol_path.relative_to(repo)),
            "phase3_protocol_sha256": sha256_file(protocol_path),
            "models": models,
        }

    for subset_id, registry in registries.items():
        write_json(registry_paths[subset_id], registry)
    write_csv(table_path, rows)
    maximum_vram = max(int(row["peak_gpu_memory_mib"] or 0) for row in rows)
    report_path.write_text(f"""# Phase 3 neural training result

Status: all 36 approved jobs completed without scientific failure. The campaign used {campaign['accumulated_active_seconds']:.2f} seconds ({campaign['accumulated_active_seconds'] / 60:.2f} minutes) of controlled wall time under the separate two-hour cap. Four independent CUDA jobs ran concurrently after a four-job measured pilot; the maximum recorded per-process VRAM was {maximum_vram:,} MiB.

Every job used the frozen 3,000-step, batch-16 schedule. Batch size was not increased because that would change sampled exposure and the optimizer update definition. Parallelism was across independent approved jobs. Checkpoint histories end at step 2,999, all losses are finite, and no training log contains a warning.

The two new 5k subsets retain 4,981 and 4,980 sampling-eligible events. The 10k prefix retains 9,980. Each resolved configuration and reconstructed density buffer uses only its matching subset; none uses the 18,866-event full pool. Checkpoints and event-level training artifacts remain server-only.

These completed models are ready for the frozen 360-cell, context-size-500 evaluation. No performance claim is made from training loss.
""")
    portable = [table_path, report_path, *registry_paths.values()]
    write_json(manifest_path, {
        "schema_version": 1,
        "analysis": "Phase 3 fixed-compute neural training",
        "status": "completed",
        "source_commit": source_commit,
        "script": str(Path(__file__).resolve().relative_to(repo)),
        "script_sha256": sha256_file(Path(__file__)),
        "phase3_protocol_sha256": sha256_file(protocol_path),
        "phase3_config_sha256": sha256_file(config_path),
        "campaign_record_sha256": sha256_file(campaign_path),
        "completed_jobs": len(rows),
        "scientific_failures": campaign["failures"],
        "incomplete_attempts": campaign["incomplete_attempts"],
        "accumulated_active_seconds": campaign["accumulated_active_seconds"],
        "summed_job_wall_seconds": campaign["summed_job_wall_seconds"],
        "maximum_per_process_vram_mib": maximum_vram,
        "runner_inputs": runner_inputs,
        "outputs": {
            str(path.relative_to(repo)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in portable
        },
    })


if __name__ == "__main__":
    main()
