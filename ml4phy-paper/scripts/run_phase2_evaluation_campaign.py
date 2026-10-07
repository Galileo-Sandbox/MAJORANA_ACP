#!/usr/bin/env python3
"""Run the frozen Phase 2 n=500 neural evaluation matrix."""

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
CONTEXT_SEEDS = tuple(range(100, 110))
CONTEXT_SIZE = 500


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


def validate_existing_run(
    path: Path, expected: tuple[int, str, int, int]
) -> dict:
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Existing run lacks summary: {path}")
    summary = read_json(summary_path)
    budget, architecture, training_seed, context_seed = expected
    expected_model = f"b{budget}_{architecture}_seed{training_seed}"
    if summary["status"] != "completed":
        raise ValueError(f"Existing run is incomplete: {path}")
    if summary["model"]["id"] != expected_model:
        raise ValueError(f"Existing run model mismatch: {path}")
    if summary["model"]["training_seed"] != training_seed:
        raise ValueError(f"Existing run training-seed mismatch: {path}")
    if summary["randomness"]["context_seed"] != context_seed:
        raise ValueError(f"Existing run context-seed mismatch: {path}")
    if summary["counts"]["context_size"] != CONTEXT_SIZE:
        raise ValueError(f"Existing run context-size mismatch: {path}")
    if summary["randomness"]["dropout_seed"] != 10100:
        raise ValueError(f"Existing run dropout-seed mismatch: {path}")
    if summary["counts"]["mc_passes"] != 50:
        raise ValueError(f"Existing run MC budget mismatch: {path}")
    if summary["counts"]["target"] != 114400:
        raise ValueError(f"Existing run target count mismatch: {path}")
    expected_subset = f"final_context_s{context_seed}_n500_rows"
    if summary["extension_protocol"]["subset_key"] != expected_subset:
        raise ValueError(f"Existing run subset mismatch: {path}")
    for output in summary["server_only_outputs"].values():
        output_path = path / output["file"]
        if sha256_file(output_path) != output["sha256"]:
            raise ValueError(f"Server-only output hash mismatch: {output_path}")
    return summary


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
    evaluator = repo / "ml4phy-paper/scripts/evaluate_extension_neural.py"
    run_root = repo / "ml4phy-paper/runs/phase2/evaluation"
    campaign_dir = repo / "ml4phy-paper/runs/campaigns/20260909-phase2-eval-v1"
    record_path = campaign_dir / "campaign_record.json"
    log_path = campaign_dir / "campaign.log"
    resuming = campaign_dir.exists()
    if resuming:
        record = read_json(record_path)
        if record["status"] == "completed":
            raise RuntimeError("Campaign is already complete")
        record.setdefault("resumes", []).append(
            {
                "time": utc_now(),
                "source_commit": source_commit,
                "script_sha256": sha256_file(Path(__file__)),
            }
        )
        record["status"] = "running"
    else:
        campaign_dir.mkdir(parents=True, exist_ok=False)
        record = {
            "schema_version": 1,
            "status": "running",
            "start_time": utc_now(),
            "source_commit": source_commit,
            "script_sha256": sha256_file(Path(__file__)),
            "evaluator_sha256": sha256_file(evaluator),
            "matrix": {
                "training_budgets": list(BUDGETS),
                "architectures": list(ARCHITECTURES),
                "training_seeds": list(TRAINING_SEEDS),
                "context_seeds": list(CONTEXT_SEEDS),
                "context_size": CONTEXT_SIZE,
                "mc_passes": 50,
                "dropout_seed": 10100,
                "target_events": 114400,
                "campaign_cells": 240,
                "full_budget_reuse_cells": 120,
            },
            "completed": [],
            "failures": [],
        }
    write_json(record_path, record)
    campaign_start = time.perf_counter()
    completed_ids = {item["run_id"] for item in record["completed"]}

    with log_path.open("a" if resuming else "x") as log:
        try:
            for budget in BUDGETS:
                registry = repo / f"ml4phy-paper/configs/phase2_models_b{budget}_v1.json"
                registry_hash = sha256_file(registry)
                for architecture in ARCHITECTURES:
                    for training_seed in TRAINING_SEEDS:
                        for context_seed in CONTEXT_SEEDS:
                            cell = (budget, architecture, training_seed, context_seed)
                            run_id = (
                                f"20260909-phase2-b{budget}-{architecture}-seed{training_seed}-"
                                f"final-ctx-s{context_seed}-n500-drop10100-mc50"
                            )
                            output_dir = run_root / run_id
                            if output_dir.exists():
                                summary = validate_existing_run(output_dir, cell)
                                if run_id not in completed_ids:
                                    record["completed"].append(
                                        {
                                            "run_id": run_id,
                                            "summary_sha256": sha256_file(
                                                output_dir / "summary.json"
                                            ),
                                            "registry_sha256": registry_hash,
                                            "wall_seconds": None,
                                            "inference_seconds": summary["runtime"][
                                                "inference_seconds"
                                            ],
                                            "peak_memory_mib": summary["runtime"][
                                                "peak_memory_mib"
                                            ],
                                            "recovered_after_interruption": True,
                                        }
                                    )
                                    completed_ids.add(run_id)
                                    write_json(record_path, record)
                                continue
                            command = [
                                str(repo / ".venv/bin/python"),
                                str(evaluator),
                                "--repo",
                                str(repo),
                                "--model-id",
                                f"b{budget}_{architecture}_seed{training_seed}",
                                "--phase",
                                "final",
                                "--context-seed",
                                str(context_seed),
                                "--context-size",
                                str(CONTEXT_SIZE),
                                "--n-mc",
                                "50",
                                "--grid-spacing-kev",
                                "1.0",
                                "--run-kind",
                                "evaluation",
                                "--output-dir",
                                str(output_dir),
                                "--model-registry",
                                str(registry),
                            ]
                            cell_start = time.perf_counter()
                            process = subprocess.Popen(
                                command,
                                cwd=repo,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                bufsize=1,
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
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
                                raise RuntimeError(f"Evaluation failed: {run_id}")
                            summary = validate_existing_run(output_dir, cell)
                            record["completed"].append(
                                {
                                    "run_id": run_id,
                                    "summary_sha256": sha256_file(output_dir / "summary.json"),
                                    "registry_sha256": registry_hash,
                                    "wall_seconds": time.perf_counter() - cell_start,
                                    "inference_seconds": summary["runtime"]["inference_seconds"],
                                    "peak_memory_mib": summary["runtime"]["peak_memory_mib"],
                                }
                            )
                            completed_ids.add(run_id)
                            write_json(record_path, record)
        except Exception:
            record["status"] = "failed"
            record["end_time"] = utc_now()
            record["final_session_wall_seconds"] = time.perf_counter() - campaign_start
            write_json(record_path, record)
            raise

    record["status"] = "completed"
    record["end_time"] = utc_now()
    record["final_session_wall_seconds"] = time.perf_counter() - campaign_start
    record["measured_completed_cell_wall_seconds"] = sum(
        item["wall_seconds"]
        for item in record["completed"]
        if item["wall_seconds"] is not None
    )
    record["log_sha256"] = sha256_file(log_path)
    write_json(record_path, record)
    print(
        json.dumps(
            {
                "status": record["status"],
                "completed_cells": len(record["completed"]),
                "failures": len(record["failures"]),
                "final_session_wall_seconds": record["final_session_wall_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
