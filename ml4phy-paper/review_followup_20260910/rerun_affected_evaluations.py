#!/usr/bin/env python3
"""Rerun only global-gate cells invalidated by the checkpoint-loader reset."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

AFFECTED = ("global_gate", "global_both", "density_free_global")
HARD_SECONDS = 7200.0


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
    repo = Path(".").resolve()
    root = repo / "ml4phy-paper/review_followup_20260910"
    if subprocess.check_output(["git", "status", "--short"], text=True).strip():
        raise RuntimeError("Correction rerun requires a clean reviewed implementation commit")
    campaign_path = root / "runs/campaigns/20260910-mechanism-controls-v1/campaign_record.json"
    campaign = json.loads(campaign_path.read_text())
    if time.time() - campaign["scientific_start_unix"] >= HARD_SECONDS:
        raise TimeoutError("Scientific hard wall-clock limit reached")
    if not any(row["mode"] in AFFECTED for row in campaign["completed_evaluations"]):
        raise ValueError("No affected evaluation cells are present")
    invalid = [row for row in campaign["completed_evaluations"] if row["mode"] in AFFECTED]
    retained = [row for row in campaign["completed_evaluations"] if row["mode"] not in AFFECTED]
    if len(invalid) != 90 or len(retained) != 30:
        raise ValueError("Expected 90 affected and 30 unaffected new cells")
    campaign.setdefault("invalid_evaluations_checkpoint_restore_reset", []).extend(invalid)
    campaign["completed_evaluations"] = retained
    campaign["status"] = "checkpoint_restore_correction_running"
    campaign["checkpoint_restore_fix_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    campaign["checkpoint_restore_fix_sha256"] = sha256_file(root / "control_models.py")
    campaign["checkpoint_restore_correction_started_at"] = (
        datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )
    write_json(campaign_path, campaign)
    training = {
        (row["mode"], row["training_seed"]): row["run_id"] for row in campaign["completed_training"]
    }
    jobs = [
        (mode, seed, context)
        for mode in AFFECTED
        for seed in range(3)
        for context in range(100, 110)
    ]
    runner = root / "evaluate_control.py"

    def execute(job):
        mode, seed, context = job
        run_id = f"20260910-mechanism-{mode}-seed{seed}-ctx{context}-n500-mc50-drop10100-restorefix"
        remaining = HARD_SECONDS - (time.time() - campaign["scientific_start_unix"])
        process = subprocess.run(
            [
                str(repo / ".venv/bin/python"),
                str(runner),
                "--repo",
                str(repo),
                "--mode",
                mode,
                "--training-seed",
                str(seed),
                "--training-run-id",
                training[(mode, seed)],
                "--context-seed",
                str(context),
                "--run-id",
                run_id,
            ],
            cwd=repo,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            text=True,
            capture_output=True,
            timeout=max(1.0, remaining),
        )
        return job, run_id, process

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(execute, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            job, run_id, process = future.result()
            print(process.stdout, end="", flush=True)
            if process.stderr:
                print(process.stderr, end="", flush=True)
            if process.returncode:
                campaign["failures"].append(
                    {"run_id": run_id, "returncode": process.returncode, "stage": "restorefix"}
                )
                campaign["status"] = "stopped_checkpoint_restore_correction_failure"
                write_json(campaign_path, campaign)
                raise RuntimeError(f"Checkpoint-restoration correction failed: {run_id}")
            directory = root / "runs/evaluation" / run_id
            summary = json.loads((directory / "summary.json").read_text())
            for name in ("event_predictions.npz", "curve.npz"):
                if sha256_file(directory / name) != summary["outputs"][name]["sha256"]:
                    raise ValueError(f"Correction output hash mismatch: {run_id}/{name}")
            campaign["completed_evaluations"].append(
                {
                    "run_id": run_id,
                    "mode": job[0],
                    "training_seed": job[1],
                    "context_seed": job[2],
                    "inference_seconds": summary["inference_seconds"],
                    "peak_gpu_memory_mib": summary["peak_gpu_memory_mib"],
                    "event_predictions_sha256": summary["outputs"]["event_predictions.npz"][
                        "sha256"
                    ],
                    "curve_sha256": summary["outputs"]["curve.npz"]["sha256"],
                    "checkpoint_restore_corrected": True,
                }
            )
            write_json(campaign_path, campaign)
    if len(campaign["completed_evaluations"]) != 120:
        raise ValueError("Corrected evaluation inventory is incomplete")
    campaign["status"] = "scientific_matrix_complete"
    campaign["scientific_end_time"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    campaign["scientific_elapsed_seconds"] = time.time() - campaign["scientific_start_unix"]
    campaign["checkpoint_restore_correction_completed_cells"] = 90
    write_json(campaign_path, campaign)
    print(
        json.dumps(
            {
                "status": campaign["status"],
                "completed": 120,
                "invalid_retained": 90,
                "elapsed": campaign["scientific_elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
