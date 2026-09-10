#!/usr/bin/env python3
"""Run the frozen mechanism-control training pilot or remaining matrix."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

MODES = ("global_gate", "global_attention", "global_both", "density_free_global")
SEEDS = (0, 1, 2)
HARD_SECONDS = 7200.0
CAMPAIGN_ID = "20260910-mechanism-controls-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "remaining"), required=True)
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.max_workers <= 4:
        parser.error("max-workers must be 1--4")
    repo = Path(".").resolve()
    root = repo / "ml4phy-paper/review_followup_20260910"
    if subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True).strip():
        raise RuntimeError("Campaign requires a clean worktree")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    protocol_path = root / "configs/protocol_v1.json"
    runner = root / "train_control.py"
    campaign_path = root / "runs/campaigns" / CAMPAIGN_ID / "campaign_record.json"
    if campaign_path.exists():
        campaign = json.loads(campaign_path.read_text())
        if campaign["source_commit"] != source_commit:
            raise ValueError("Campaign implementation commit changed")
    else:
        if args.stage != "pilot":
            raise ValueError("Pilot must run before the remaining matrix")
        campaign = {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "status": "pilot_running",
            "source_commit": source_commit,
            "protocol_sha256": sha256_file(protocol_path),
            "runner_sha256": sha256_file(runner),
            "scientific_start_unix": time.time(),
            "scientific_start_time": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "hard_limit_seconds": HARD_SECONDS,
            "completed_training": [],
            "failures": [],
            "retries": [],
        }
    elapsed = time.time() - campaign["scientific_start_unix"]
    if elapsed >= HARD_SECONDS:
        raise TimeoutError("Scientific hard wall-clock limit already reached")
    completed = {(item["mode"], item["training_seed"]) for item in campaign["completed_training"]}
    matrix = [(mode, seed) for mode in MODES for seed in SEEDS]
    pilot = ("global_gate", 0)
    pending = [pilot] if args.stage == "pilot" and pilot not in completed else []
    if args.stage == "remaining":
        if pilot not in completed:
            raise ValueError("Successful pilot is absent")
        pilot_record = next(
            item
            for item in campaign["completed_training"]
            if (item["mode"], item["training_seed"]) == pilot
        )
        historical_eval_seconds = 856.10 / 240.0
        remaining_training = len(matrix) - len(completed)
        projected = 1.25 * (
            math.ceil(remaining_training / args.max_workers) * pilot_record["wall_seconds"]
            + math.ceil(120 / args.max_workers) * historical_eval_seconds
        )
        campaign["resource_forecast"] = {
            "pilot_wall_seconds": pilot_record["wall_seconds"],
            "pilot_peak_gpu_memory_mib": pilot_record["peak_gpu_memory_mib"],
            "historical_matched_evaluation_seconds_per_cell": historical_eval_seconds,
            "remaining_training_jobs": remaining_training,
            "new_evaluation_cells": 120,
            "max_workers": args.max_workers,
            "margin_factor": 1.25,
            "projected_remaining_seconds_with_margin": projected,
            "remaining_hard_limit_seconds_at_forecast": HARD_SECONDS - elapsed,
        }
        if projected > HARD_SECONDS - elapsed:
            campaign["status"] = "stopped_forecast_exceeds_cap"
            write_json(campaign_path, campaign)
            raise TimeoutError("Pilot forecast plus margin exceeds remaining cap")
        pending = [item for item in matrix if item not in completed]
        campaign["status"] = "training_running"
    write_json(campaign_path, campaign)

    def execute(item):
        mode, seed = item
        run_id = f"20260910-mechanism-{mode}-seed{seed}-train3000"
        command = [
            str(repo / ".venv/bin/python"),
            str(runner),
            "--repo",
            str(repo),
            "--mode",
            mode,
            "--training-seed",
            str(seed),
            "--run-id",
            run_id,
        ]
        remaining = HARD_SECONDS - (time.time() - campaign["scientific_start_unix"])
        process = subprocess.run(
            command,
            cwd=repo,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            text=True,
            capture_output=True,
            timeout=max(1.0, remaining),
        )
        return item, run_id, process

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(execute, item) for item in pending]
        for future in concurrent.futures.as_completed(futures):
            item, run_id, process = future.result()
            print(process.stdout, end="")
            if process.stderr:
                print(process.stderr, end="")
            if process.returncode:
                campaign["failures"].append({"run_id": run_id, "returncode": process.returncode})
                campaign["status"] = "stopped_technical_failure"
                write_json(campaign_path, campaign)
                raise RuntimeError(f"Training failed: {run_id}")
            run_record = json.loads(
                (root / "runs/training" / run_id / "runner_record.json").read_text()
            )
            campaign["completed_training"].append(
                {
                    "run_id": run_id,
                    "mode": item[0],
                    "training_seed": item[1],
                    "wall_seconds": run_record["wall_seconds"],
                    "peak_gpu_memory_mib": run_record["peak_gpu_memory_mib"],
                    "checkpoint_sha256": run_record["outputs"]["control.ckpt"]["sha256"],
                }
            )
            write_json(campaign_path, campaign)
    campaign["status"] = "pilot_complete" if args.stage == "pilot" else "training_complete"
    campaign["scientific_elapsed_seconds"] = time.time() - campaign["scientific_start_unix"]
    write_json(campaign_path, campaign)
    print(
        json.dumps(
            {
                "status": campaign["status"],
                "completed": len(campaign["completed_training"]),
                "elapsed": campaign["scientific_elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
