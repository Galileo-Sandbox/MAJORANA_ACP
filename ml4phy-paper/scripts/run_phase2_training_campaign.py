#!/usr/bin/env python3
"""Run or safely resume the approved 24-job Phase 2 training campaign."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

BUDGETS = (2000, 5000)
ARCHITECTURES = ("m0", "m1", "m2", "ours")
TRAINING_SEEDS = (0, 1, 2)
CAMPAIGN_ID = "20260909-phase2-training-v1"


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


def validate_completed(run_dir: Path, expected: tuple[int, str, int]) -> dict:
    record_path = run_dir / "runner_record.json"
    if not record_path.is_file():
        raise FileNotFoundError(f"Existing run has no record: {run_dir}")
    record = read_json(record_path)
    budget, architecture, seed = expected
    if record["status"] != "completed" or record["returncode"] != 0:
        raise ValueError(f"Existing run is not complete: {run_dir}")
    if (record["training_budget"], record["architecture"], record["training_seed"]) != expected:
        raise ValueError(f"Existing run metadata mismatch: {run_dir}")
    for name, item in record["outputs"].items():
        output_path = run_dir / "artifacts" / name
        if sha256_file(output_path) != item["sha256"]:
            raise ValueError(f"Existing output hash mismatch: {output_path}")
    return record


def resolve_run_attempt(
    training_root: Path, base_run_id: str, expected: tuple[int, str, int]
) -> tuple[str, Path, dict | None, list[dict]]:
    """Return a completed attempt or the first unused uniquely named attempt."""
    failed_attempts = []
    attempt = 1
    while True:
        run_id = base_run_id if attempt == 1 else f"{base_run_id}-attempt{attempt}"
        run_dir = training_root / run_id
        if not run_dir.exists():
            return run_id, run_dir, None, failed_attempts
        record_path = run_dir / "runner_record.json"
        if record_path.is_file():
            run_record = read_json(record_path)
            if run_record.get("status") == "completed":
                return run_id, run_dir, validate_completed(run_dir, expected), failed_attempts
            failed_attempts.append(
                {
                    "run_id": run_id,
                    "status": run_record.get("status", "missing"),
                    "returncode": run_record.get("returncode"),
                    "failure_stage": "before optimization" if not (run_dir / "artifacts/cnp.ckpt").exists() else "training",
                    "reason": "CUDA device was hidden by the default execution sandbox" if run_record.get("returncode") == 1 else "see preserved runner record",
                    "runner_record_sha256": sha256_file(record_path),
                }
            )
        else:
            failed_attempts.append(
                {
                    "run_id": run_id,
                    "status": "missing_runner_record",
                    "returncode": None,
                    "failure_stage": "runner setup",
                    "reason": "see preserved server-only directory",
                    "runner_record_sha256": None,
                }
            )
        attempt += 1


def main() -> None:
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
    campaign_dir = root / "runs/phase2/campaigns" / CAMPAIGN_ID
    record_path = campaign_dir / "campaign_record.json"
    log_path = campaign_dir / "campaign.log"
    runner = root / "scripts/run_phase2_training.py"
    protocol_path = root / "manifests/phase2_protocol_v1.json"
    protocol = read_json(protocol_path)
    if protocol["status"] != "frozen_approved_not_started":
        raise ValueError("Phase 2 protocol is not approved and frozen")

    resuming = campaign_dir.exists()
    if resuming:
        record = read_json(record_path)
        if record["status"] == "completed":
            raise RuntimeError("Campaign is already complete")
        record.setdefault("resumes", []).append(
            {
                "time": utc_now(),
                "source_commit": source_commit,
                "runner_sha256": sha256_file(runner),
                "campaign_script_sha256": sha256_file(Path(__file__)),
            }
        )
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
            "protocol_sha256": sha256_file(protocol_path),
            "matrix": {
                "budgets": list(BUDGETS),
                "architectures": list(ARCHITECTURES),
                "training_seeds": list(TRAINING_SEEDS),
                "total_jobs": 24,
            },
            "completed": [],
            "failures": [],
            "failed_attempts": [],
        }
    write_json(record_path, record)
    completed_ids = {item["run_id"] for item in record["completed"]}
    session_timer = time.perf_counter()
    with log_path.open("a" if resuming else "x") as log:
        try:
            for budget in BUDGETS:
                for architecture in ARCHITECTURES:
                    for seed in TRAINING_SEEDS:
                        base_run_id = f"20260909-phase2-b{budget}-{architecture}-seed{seed}-train3000"
                        run_id, run_dir, run_record, failed_attempts = resolve_run_attempt(
                            root / "runs/phase2/training",
                            base_run_id,
                            (budget, architecture, seed),
                        )
                        known_failed = {item["run_id"] for item in record["failed_attempts"]}
                        for failed in failed_attempts:
                            if failed["run_id"] not in known_failed:
                                record["failed_attempts"].append(failed)
                                known_failed.add(failed["run_id"])
                        if run_record is not None:
                            if run_id not in completed_ids:
                                record["completed"].append(
                                    {
                                        "run_id": run_id,
                                        "runner_record_sha256": sha256_file(
                                            run_dir / "runner_record.json"
                                        ),
                                        "wall_seconds": run_record["wall_seconds"],
                                        "recovered_after_interruption": True,
                                    }
                                )
                                completed_ids.add(run_id)
                                write_json(record_path, record)
                            continue
                        command = [
                            str(repo / ".venv/bin/python"),
                            str(runner),
                            "--repo",
                            str(repo),
                            "--architecture",
                            architecture,
                            "--budget",
                            str(budget),
                            "--training-seed",
                            str(seed),
                            "--run-id",
                            run_id,
                        ]
                        process = subprocess.Popen(
                            command,
                            cwd=repo,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                        )
                        assert process.stdout is not None
                        for line in process.stdout:
                            print(line, end="", flush=True)
                            log.write(line)
                            log.flush()
                        returncode = process.wait()
                        if returncode != 0:
                            record["failures"].append(
                                {"run_id": run_id, "returncode": returncode}
                            )
                            raise RuntimeError(f"Phase 2 training failed: {run_id}")
                        run_record = validate_completed(
                            run_dir, (budget, architecture, seed)
                        )
                        record["completed"].append(
                            {
                                "run_id": run_id,
                                "runner_record_sha256": sha256_file(
                                    run_dir / "runner_record.json"
                                ),
                                "wall_seconds": run_record["wall_seconds"],
                            }
                        )
                        completed_ids.add(run_id)
                        write_json(record_path, record)
        except Exception:
            record["status"] = "failed"
            record["end_time"] = utc_now()
            record["last_session_wall_seconds"] = time.perf_counter() - session_timer
            write_json(record_path, record)
            raise

    if len(record["completed"]) != 24 or record["failures"]:
        raise RuntimeError("Campaign matrix is incomplete")
    record["status"] = "completed"
    record["end_time"] = utc_now()
    record["last_session_wall_seconds"] = time.perf_counter() - session_timer
    record["summed_job_wall_seconds"] = sum(
        float(item["wall_seconds"]) for item in record["completed"]
    )
    record["log_sha256"] = sha256_file(log_path)
    write_json(record_path, record)
    print(
        json.dumps(
            {
                "status": "completed",
                "jobs": len(record["completed"]),
                "summed_job_wall_seconds": record["summed_job_wall_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
