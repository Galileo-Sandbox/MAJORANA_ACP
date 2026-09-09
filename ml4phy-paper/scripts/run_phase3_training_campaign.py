#!/usr/bin/env python3
"""Run or resume the approved Phase 3 neural training matrix under its cap."""

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

SUBSET_IDS = ("seed20260910_n5000", "seed20260911_n5000", "original_n10000")
ARCHITECTURES = ("m0", "m1", "m2", "ours")
TRAINING_SEEDS = (0, 1, 2)
CAMPAIGN_ID = "20260909-phase3-neural-training-v1"
HARD_LIMIT_SECONDS = 2 * 60 * 60


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


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
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def validate_completed(run_dir: Path, expected: tuple[str, str, int]) -> dict:
    record_path = run_dir / "runner_record.json"
    if not record_path.is_file():
        raise FileNotFoundError(f"Existing run has no record: {run_dir}")
    record = read_json(record_path)
    subset_id, architecture, seed = expected
    if record.get("status") != "completed" or record.get("returncode") != 0:
        raise ValueError(f"Existing run is not complete: {run_dir}")
    observed = (record["subset_id"], record["architecture"], record["training_seed"])
    if observed != expected:
        raise ValueError(f"Existing run metadata mismatch: {run_dir}")
    for name, item in record["outputs"].items():
        output_path = run_dir / "artifacts" / name
        if output_path.stat().st_size != item["bytes"] or sha256_file(output_path) != item["sha256"]:
            raise ValueError(f"Existing output hash mismatch: {output_path}")
    return record


def resolve_attempt(
    training_root: Path, base_run_id: str, expected: tuple[str, str, int]
) -> tuple[str, Path, dict | None, list[dict]]:
    incomplete = []
    attempt = 1
    while True:
        run_id = base_run_id if attempt == 1 else f"{base_run_id}-attempt{attempt}"
        run_dir = training_root / run_id
        if not run_dir.exists():
            return run_id, run_dir, None, incomplete
        record_path = run_dir / "runner_record.json"
        if record_path.is_file():
            record = read_json(record_path)
            if record.get("status") == "completed":
                return run_id, run_dir, validate_completed(run_dir, expected), incomplete
            incomplete.append({
                "run_id": run_id,
                "status": record.get("status", "missing"),
                "returncode": record.get("returncode"),
                "runner_record_sha256": sha256_file(record_path),
            })
        else:
            incomplete.append({
                "run_id": run_id,
                "status": "missing_runner_record",
                "returncode": None,
                "runner_record_sha256": None,
            })
        attempt += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--max-new-jobs", type=int, default=None)
    args = parser.parse_args()
    if args.max_workers < 1 or args.max_workers > 4:
        parser.error("--max-workers must be between one and four")
    if args.max_new_jobs is not None and args.max_new_jobs < 1:
        parser.error("--max-new-jobs must be positive")

    repo = Path(".").resolve()
    worktree = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--short"], text=True
    ).splitlines()
    if worktree:
        raise RuntimeError("Campaign requires a clean worktree: " + "; ".join(worktree))
    source_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    root = repo / "ml4phy-paper"
    runner = root / "scripts/run_phase3_training.py"
    config_path = root / "configs/phase3_training_v1.json"
    protocol_path = root / "manifests/phase3_protocol_v1.json"
    config = read_json(config_path)
    protocol = read_json(protocol_path)
    if config["status"] != "frozen" or protocol["status"] != "frozen_approved_not_started":
        raise ValueError("Phase 3 protocol is not approved and frozen")
    if config["neural_hard_limit_seconds"] != HARD_LIMIT_SECONDS:
        raise ValueError("Phase 3 neural hard limit mismatch")

    campaign_dir = root / "runs/phase3/campaigns" / CAMPAIGN_ID
    record_path = campaign_dir / "campaign_record.json"
    log_path = campaign_dir / "campaign.log"
    resuming = campaign_dir.exists()
    if resuming:
        record = read_json(record_path)
        if record["status"] == "completed":
            raise RuntimeError("Phase 3 training campaign is already complete")
        record.setdefault("resumes", []).append({
            "time": utc_now(),
            "source_commit": source_commit,
            "runner_sha256": sha256_file(runner),
            "campaign_script_sha256": sha256_file(Path(__file__)),
            "max_workers": args.max_workers,
            "max_new_jobs": args.max_new_jobs,
        })
        record["status"] = "running"
    else:
        campaign_dir.mkdir(parents=True, exist_ok=False)
        record = {
            "schema_version": 1,
            "status": "running",
            "campaign_id": CAMPAIGN_ID,
            "start_time": utc_now(),
            "source_commit": source_commit,
            "runner_sha256": sha256_file(runner),
            "campaign_script_sha256": sha256_file(Path(__file__)),
            "phase3_protocol_sha256": sha256_file(protocol_path),
            "phase3_config_sha256": sha256_file(config_path),
            "hard_limit_seconds": HARD_LIMIT_SECONDS,
            "projected_seconds_with_allowance": config["conservative_projected_seconds"],
            "matrix": {
                "subset_ids": list(SUBSET_IDS),
                "architectures": list(ARCHITECTURES),
                "training_seeds": list(TRAINING_SEEDS),
                "jobs": 36,
                "steps_per_job": 3000,
                "batch_size": 16,
            },
            "execution_settings": {
                "max_workers": args.max_workers,
                "max_new_jobs": args.max_new_jobs,
            },
            "accumulated_active_seconds": 0.0,
            "completed": [],
            "failures": [],
            "incomplete_attempts": [],
        }
    write_json(record_path, record)

    completed_cells = {
        (item["subset_id"], item["architecture"], item["training_seed"])
        for item in record["completed"]
    }
    prior = float(record.get("accumulated_active_seconds", 0.0))
    session_start = time.perf_counter()

    def elapsed() -> float:
        return prior + time.perf_counter() - session_start

    def save() -> None:
        record["accumulated_active_seconds"] = elapsed()
        write_json(record_path, record)

    pending = []
    training_root = root / "runs/phase3/training"
    for subset_id in SUBSET_IDS:
        for seed in TRAINING_SEEDS:
            for architecture in ARCHITECTURES:
                logical = (subset_id, architecture, seed)
                base_id = f"20260909-phase3-{subset_id}-{architecture}-seed{seed}-train3000"
                run_id, run_dir, existing, incomplete = resolve_attempt(
                    training_root, base_id, logical
                )
                known = {item["run_id"] for item in record["incomplete_attempts"]}
                for item in incomplete:
                    if item["run_id"] not in known:
                        record["incomplete_attempts"].append(item)
                        known.add(item["run_id"])
                if existing is not None:
                    if logical not in completed_cells:
                        record["completed"].append({
                            "run_id": run_id,
                            "subset_id": subset_id,
                            "architecture": architecture,
                            "training_seed": seed,
                            "runner_record_sha256": sha256_file(run_dir / "runner_record.json"),
                            "wall_seconds": existing["wall_seconds"],
                            "peak_gpu_memory_mib": existing.get("peak_gpu_memory_mib"),
                            "recovered_after_interruption": True,
                        })
                        completed_cells.add(logical)
                    continue
                pending.append({
                    "run_id": run_id,
                    "run_dir": run_dir,
                    "subset_id": subset_id,
                    "architecture": architecture,
                    "training_seed": seed,
                })
    if args.max_new_jobs is not None:
        pending = pending[: args.max_new_jobs]
    save()

    def execute(cell: dict) -> dict:
        remaining = HARD_LIMIT_SECONDS - elapsed()
        if remaining <= 0:
            return {**cell, "outcome": "time_limit", "returncode": None, "output": ""}
        command = [
            str(repo / ".venv/bin/python"),
            str(runner),
            "--repo", str(repo),
            "--architecture", cell["architecture"],
            "--subset-id", cell["subset_id"],
            "--training-seed", str(cell["training_seed"]),
            "--run-id", cell["run_id"],
        ]
        process = subprocess.Popen(
            command,
            cwd=repo,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            output, _ = process.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                output, _ = process.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                output, _ = process.communicate()
            return {**cell, "outcome": "time_limit", "returncode": process.returncode, "output": output}
        outcome = "completed" if process.returncode == 0 else "failed"
        return {**cell, "outcome": outcome, "returncode": process.returncode, "output": output}

    hit_limit = False
    log_mode = "a" if resuming else "x"
    with log_path.open(log_mode) as log, concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = [executor.submit(execute, cell) for cell in pending]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            output = result.pop("output")
            print(output, end="", flush=True)
            log.write(output)
            log.flush()
            outcome = result.pop("outcome")
            run_dir = result.pop("run_dir")
            if outcome == "time_limit":
                record.setdefault("terminated_at_limit", []).append(result)
                hit_limit = True
            elif outcome == "failed":
                record["failures"].append(result)
            else:
                expected = (
                    result["subset_id"],
                    result["architecture"],
                    result["training_seed"],
                )
                run_record = validate_completed(run_dir, expected)
                if expected not in completed_cells:
                    record["completed"].append({
                        "run_id": result["run_id"],
                        "subset_id": result["subset_id"],
                        "architecture": result["architecture"],
                        "training_seed": result["training_seed"],
                        "runner_record_sha256": sha256_file(run_dir / "runner_record.json"),
                        "wall_seconds": run_record["wall_seconds"],
                        "peak_gpu_memory_mib": run_record.get("peak_gpu_memory_mib"),
                        "recovered_after_interruption": False,
                    })
                    completed_cells.add(expected)
            save()

    if record["failures"]:
        record["status"] = "failed"
    elif hit_limit:
        record["status"] = "partial_time_limit"
    elif args.max_new_jobs is not None and len(record["completed"]) < 36:
        record["status"] = "paused_operator_limit"
    elif len(record["completed"]) == 36:
        record["status"] = "completed"
        record["summed_job_wall_seconds"] = sum(
            float(item["wall_seconds"]) for item in record["completed"]
        )
        record["log_sha256"] = sha256_file(log_path)
    else:
        record["status"] = "partial_time_limit"
    record["end_time"] = utc_now()
    save()
    print(json.dumps({
        "status": record["status"],
        "completed_jobs": len(record["completed"]),
        "failures": len(record["failures"]),
        "accumulated_active_seconds": record["accumulated_active_seconds"],
    }))
    if record["status"] == "failed":
        raise RuntimeError("One or more Phase 3 training jobs failed")


if __name__ == "__main__":
    main()
