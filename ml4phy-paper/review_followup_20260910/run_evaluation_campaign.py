#!/usr/bin/env python3
"""Run the 120 frozen mechanism-control evaluation cells under the shared cap."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

MODES = ("global_gate", "global_attention", "global_both", "density_free_global")
HARD_SECONDS = 7200.0
CAMPAIGN_ID = "20260910-mechanism-controls-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-workers", type=int, default=4)
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
    campaign_path = root / "runs/campaigns" / CAMPAIGN_ID / "campaign_record.json"
    campaign = json.loads(campaign_path.read_text())
    if len(campaign["completed_training"]) != 12:
        raise ValueError("Training campaign is incomplete")
    protocol = json.loads((root / "configs/protocol_v1.json").read_text())
    control_path = root / "control_models.py"
    if sha256_file(control_path) != protocol["hashes"]["sources"][
        "ml4phy-paper/review_followup_20260910/control_models.py"
    ]:
        raise ValueError("Frozen control implementation changed after training")
    if time.time() - campaign["scientific_start_unix"] >= HARD_SECONDS:
        raise TimeoutError("Scientific hard wall-clock limit reached before evaluation")
    runner = root / "evaluate_control.py"
    campaign["evaluation_runner_sha256"] = sha256_file(runner)
    campaign["evaluation_source_commit"] = source_commit
    training_runs = {
        (item["mode"], item["training_seed"]): item for item in campaign["completed_training"]
    }
    campaign.setdefault("completed_evaluations", [])
    campaign["status"] = "evaluation_running"
    campaign["evaluation_max_workers"] = args.max_workers
    write_json(campaign_path, campaign)
    completed = {
        (item["mode"], item["training_seed"], item["context_seed"])
        for item in campaign["completed_evaluations"]
    }
    pending = [
        (mode, seed, context)
        for mode in MODES
        for seed in (0, 1, 2)
        for context in range(100, 110)
        if (mode, seed, context) not in completed
    ]

    def execute(item):
        mode, seed, context = item
        training_run = training_runs[(mode, seed)]
        run_id = f"20260910-mechanism-{mode}-seed{seed}-ctx{context}-n500-mc50-drop10100"
        command = [
            str(repo / ".venv/bin/python"),
            str(runner),
            "--repo",
            str(repo),
            "--mode",
            mode,
            "--training-seed",
            str(seed),
            "--training-run-id",
            training_run["run_id"],
            "--context-seed",
            str(context),
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
            print(process.stdout, end="", flush=True)
            if process.stderr:
                print(process.stderr, end="", flush=True)
            if process.returncode:
                campaign["failures"].append({"run_id": run_id, "returncode": process.returncode})
                campaign["status"] = "stopped_technical_failure"
                write_json(campaign_path, campaign)
                raise RuntimeError(f"Evaluation failed: {run_id}")
            directory = root / "runs/evaluation" / run_id
            summary = json.loads((directory / "summary.json").read_text())
            campaign["completed_evaluations"].append(
                {
                    "run_id": run_id,
                    "mode": item[0],
                    "training_seed": item[1],
                    "context_seed": item[2],
                    "inference_seconds": summary["inference_seconds"],
                    "peak_gpu_memory_mib": summary["peak_gpu_memory_mib"],
                    "event_predictions_sha256": sha256_file(directory / "event_predictions.npz"),
                    "curve_sha256": sha256_file(directory / "curve.npz"),
                }
            )
            write_json(campaign_path, campaign)
    campaign["status"] = "scientific_matrix_complete"
    campaign["scientific_end_time"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    campaign["scientific_elapsed_seconds"] = time.time() - campaign["scientific_start_unix"]
    write_json(campaign_path, campaign)
    print(
        json.dumps(
            {
                "status": campaign["status"],
                "evaluations": len(campaign["completed_evaluations"]),
                "elapsed": campaign["scientific_elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
