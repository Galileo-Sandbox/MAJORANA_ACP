#!/usr/bin/env python3
"""Run the frozen Phase 3 neural evaluation matrix under the shared cap."""

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
CONTEXT_SEEDS = tuple(range(100, 110))
CONTEXT_SIZE = 500
HARD_LIMIT_SECONDS = 2 * 60 * 60
CAMPAIGN_ID = "20260909-phase3-neural-evaluation-v1"


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


def validate_cell(path: Path, expected: tuple[str, str, int, int]) -> dict:
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Existing run lacks summary: {path}")
    summary = read_json(summary_path)
    subset_id, architecture, training_seed, context_seed = expected
    expected_model = f"p3_{subset_id}_{architecture}_seed{training_seed}"
    if (
        summary["status"] != "completed"
        or summary["model"]["id"] != expected_model
        or summary["model"]["training_seed"] != training_seed
        or summary["randomness"]["context_seed"] != context_seed
        or summary["randomness"]["dropout_seed"] != 10100
        or summary["counts"]["context_size"] != CONTEXT_SIZE
        or summary["counts"]["mc_passes"] != 50
        or summary["counts"]["target"] != 114400
        or summary["extension_protocol"]["subset_key"]
        != f"final_context_s{context_seed}_n500_rows"
        or summary["source_worktree_status"]
    ):
        raise ValueError(f"Evaluation cell contract mismatch: {summary_path}")
    for output in summary["server_only_outputs"].values():
        output_path = path / output["file"]
        if output_path.stat().st_size != output["bytes"] or sha256_file(output_path) != output["sha256"]:
            raise ValueError(f"Server-only output hash mismatch: {output_path}")
    return summary


def resolve_attempt(
    run_root: Path, base_id: str, expected: tuple[str, str, int, int]
) -> tuple[str, Path, dict | None, list[dict]]:
    incomplete = []
    attempt = 1
    while True:
        run_id = base_id if attempt == 1 else f"{base_id}-attempt{attempt}"
        path = run_root / run_id
        if not path.exists():
            return run_id, path, None, incomplete
        summary_path = path / "summary.json"
        if summary_path.is_file():
            return run_id, path, validate_cell(path, expected), incomplete
        incomplete.append({
            "run_id": run_id,
            "summary_present": False,
            "scientific_result_completed": False,
        })
        attempt += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--max-new-cells", type=int, default=None)
    args = parser.parse_args()
    if args.max_workers < 1 or args.max_workers > 4:
        parser.error("--max-workers must be between one and four")
    if args.max_new_cells is not None and args.max_new_cells < 1:
        parser.error("--max-new-cells must be positive")

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
    evaluator = root / "scripts/evaluate_extension_neural.py"
    config_path = root / "configs/phase3_training_v1.json"
    protocol_path = root / "manifests/phase3_protocol_v1.json"
    training_campaign_path = root / "runs/phase3/campaigns/20260909-phase3-neural-training-v1/campaign_record.json"
    config = read_json(config_path)
    protocol = read_json(protocol_path)
    training_campaign = read_json(training_campaign_path)
    if config["status"] != "frozen" or protocol["status"] != "frozen_approved_not_started":
        raise ValueError("Phase 3 protocol is not approved and frozen")
    if training_campaign["status"] != "completed" or len(training_campaign["completed"]) != 36:
        raise ValueError("Phase 3 neural training is incomplete")
    training_seconds = float(training_campaign["accumulated_active_seconds"])
    if training_seconds >= HARD_LIMIT_SECONDS:
        raise RuntimeError("Shared Phase 3 neural time cap was consumed by training")

    run_root = root / "runs/phase3/evaluation"
    campaign_dir = root / "runs/phase3/campaigns" / CAMPAIGN_ID
    record_path = campaign_dir / "campaign_record.json"
    log_path = campaign_dir / "campaign.log"
    resuming = campaign_dir.exists()
    if resuming:
        record = read_json(record_path)
        if record["status"] == "completed":
            raise RuntimeError("Phase 3 evaluation campaign is already complete")
        record.setdefault("resumes", []).append({
            "time": utc_now(),
            "source_commit": source_commit,
            "script_sha256": sha256_file(Path(__file__)),
            "max_workers": args.max_workers,
            "max_new_cells": args.max_new_cells,
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
            "script_sha256": sha256_file(Path(__file__)),
            "evaluator_sha256": sha256_file(evaluator),
            "phase3_protocol_sha256": sha256_file(protocol_path),
            "shared_neural_hard_limit_seconds": HARD_LIMIT_SECONDS,
            "training_active_seconds_charged": training_seconds,
            "matrix": {
                "subset_ids": list(SUBSET_IDS),
                "architectures": list(ARCHITECTURES),
                "training_seeds": list(TRAINING_SEEDS),
                "context_seeds": list(CONTEXT_SEEDS),
                "context_size": CONTEXT_SIZE,
                "mc_passes": 50,
                "dropout_seed": 10100,
                "target_events": 114400,
                "cells": 360,
            },
            "execution_settings": {
                "max_workers": args.max_workers,
                "max_new_cells": args.max_new_cells,
            },
            "accumulated_active_seconds": 0.0,
            "completed": [],
            "failures": [],
            "incomplete_attempts": [],
        }
    write_json(record_path, record)

    completed_cells = {
        (item["subset_id"], item["architecture"], item["training_seed"], item["context_seed"])
        for item in record["completed"]
    }
    prior = float(record.get("accumulated_active_seconds", 0.0))
    session_start = time.perf_counter()

    def evaluation_elapsed() -> float:
        return prior + time.perf_counter() - session_start

    def total_neural_elapsed() -> float:
        return training_seconds + evaluation_elapsed()

    def save() -> None:
        record["accumulated_active_seconds"] = evaluation_elapsed()
        record["total_neural_active_seconds"] = total_neural_elapsed()
        write_json(record_path, record)

    pending = []
    registry_paths = {
        subset_id: root / f"configs/phase3_models_{subset_id}_v1.json"
        for subset_id in SUBSET_IDS
    }
    registry_hashes = {key: sha256_file(path) for key, path in registry_paths.items()}
    for subset_id in SUBSET_IDS:
        for training_seed in TRAINING_SEEDS:
            for context_seed in CONTEXT_SEEDS:
                for architecture in ARCHITECTURES:
                    expected = (subset_id, architecture, training_seed, context_seed)
                    base_id = (
                        f"20260909-phase3-{subset_id}-{architecture}-seed{training_seed}-"
                        f"final-ctx-s{context_seed}-n500-drop10100-mc50"
                    )
                    run_id, output_dir, existing, incomplete = resolve_attempt(
                        run_root, base_id, expected
                    )
                    known = {item["run_id"] for item in record["incomplete_attempts"]}
                    for item in incomplete:
                        if item["run_id"] not in known:
                            record["incomplete_attempts"].append(item)
                            known.add(item["run_id"])
                    if existing is not None:
                        if expected not in completed_cells:
                            record["completed"].append({
                                "run_id": run_id,
                                "subset_id": subset_id,
                                "architecture": architecture,
                                "training_seed": training_seed,
                                "context_seed": context_seed,
                                "summary_sha256": sha256_file(output_dir / "summary.json"),
                                "registry_sha256": registry_hashes[subset_id],
                                "wall_seconds": None,
                                "inference_seconds": existing["runtime"]["inference_seconds"],
                                "peak_memory_mib": existing["runtime"]["peak_memory_mib"],
                                "recovered_after_interruption": True,
                            })
                            completed_cells.add(expected)
                        continue
                    pending.append({
                        "run_id": run_id,
                        "output_dir": output_dir,
                        "subset_id": subset_id,
                        "architecture": architecture,
                        "training_seed": training_seed,
                        "context_seed": context_seed,
                    })
    if args.max_new_cells is not None:
        pending = pending[: args.max_new_cells]
    save()

    def execute(cell: dict) -> dict:
        remaining = HARD_LIMIT_SECONDS - total_neural_elapsed()
        if remaining <= 0:
            return {**cell, "outcome": "time_limit", "returncode": None, "output": "", "wall_seconds": 0.0}
        model_id = f"p3_{cell['subset_id']}_{cell['architecture']}_seed{cell['training_seed']}"
        command = [
            str(repo / ".venv/bin/python"),
            str(evaluator),
            "--repo", str(repo),
            "--model-id", model_id,
            "--phase", "final",
            "--context-seed", str(cell["context_seed"]),
            "--context-size", str(CONTEXT_SIZE),
            "--n-mc", "50",
            "--grid-spacing-kev", "1.0",
            "--run-kind", "evaluation",
            "--output-dir", str(cell["output_dir"]),
            "--model-registry", str(registry_paths[cell["subset_id"]]),
        ]
        start = time.perf_counter()
        process = subprocess.Popen(
            command,
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
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
            return {
                **cell,
                "outcome": "time_limit",
                "returncode": process.returncode,
                "output": output,
                "wall_seconds": time.perf_counter() - start,
            }
        return {
            **cell,
            "outcome": "completed" if process.returncode == 0 else "failed",
            "returncode": process.returncode,
            "output": output,
            "wall_seconds": time.perf_counter() - start,
        }

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
            output_dir = result.pop("output_dir")
            if outcome == "time_limit":
                record.setdefault("terminated_at_limit", []).append(result)
                hit_limit = True
            elif outcome == "failed":
                record["failures"].append(result)
            else:
                expected = (
                    result["subset_id"], result["architecture"],
                    result["training_seed"], result["context_seed"],
                )
                summary = validate_cell(output_dir, expected)
                if expected not in completed_cells:
                    record["completed"].append({
                        "run_id": result["run_id"],
                        "subset_id": result["subset_id"],
                        "architecture": result["architecture"],
                        "training_seed": result["training_seed"],
                        "context_seed": result["context_seed"],
                        "summary_sha256": sha256_file(output_dir / "summary.json"),
                        "registry_sha256": registry_hashes[result["subset_id"]],
                        "wall_seconds": result["wall_seconds"],
                        "inference_seconds": summary["runtime"]["inference_seconds"],
                        "peak_memory_mib": summary["runtime"]["peak_memory_mib"],
                        "recovered_after_interruption": False,
                    })
                    completed_cells.add(expected)
            save()

    if record["failures"]:
        record["status"] = "failed"
    elif hit_limit:
        record["status"] = "partial_time_limit"
    elif args.max_new_cells is not None and len(record["completed"]) < 360:
        record["status"] = "paused_operator_limit"
    elif len(record["completed"]) == 360:
        record["status"] = "completed"
        record["measured_completed_cell_wall_seconds"] = sum(
            float(item["wall_seconds"] or 0.0) for item in record["completed"]
        )
        record["log_sha256"] = sha256_file(log_path)
    else:
        record["status"] = "partial_time_limit"
    record["end_time"] = utc_now()
    save()
    print(json.dumps({
        "status": record["status"],
        "completed_cells": len(record["completed"]),
        "failures": len(record["failures"]),
        "evaluation_active_seconds": record["accumulated_active_seconds"],
        "total_neural_active_seconds": record["total_neural_active_seconds"],
    }))
    if record["status"] == "failed":
        raise RuntimeError("One or more Phase 3 evaluation cells failed")


if __name__ == "__main__":
    main()
