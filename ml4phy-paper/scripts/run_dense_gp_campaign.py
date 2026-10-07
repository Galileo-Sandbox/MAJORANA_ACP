#!/usr/bin/env python3
"""Run or resume the frozen dense Bernoulli-GP campaign under a hard cap."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

FAMILIES = ("rbf", "matern32")
CONTEXT_SIZES = (250, 500, 1000, 2000)
CONTEXT_SEEDS = tuple(range(100, 110))
HARD_LIMIT_SECONDS = 4 * 60 * 60
CAMPAIGN_ID = "20260909-phase3-dense-gp-v1"


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


def run_id(phase: str, family: str, context_size: int, context_seed: int) -> str:
    return (
        f"20260909-phase3-gp-{phase}-{family}-"
        f"ctx-s{context_seed}-n{context_size}"
    )


def validate_summary(
    directory: Path,
    phase: str,
    family: str,
    context_size: int,
    context_seed: int,
) -> dict:
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing completed summary: {summary_path}")
    summary = read_json(summary_path)
    if (
        summary["status"] != "completed"
        or summary["phase"] != phase
        or summary["estimator"]["family_id"] != family
        or summary["context"]["size"] != context_size
        or summary["context"]["seed"] != context_seed
    ):
        raise ValueError(f"Dense GP cell contract mismatch: {summary_path}")
    for output in summary["server_only_outputs"].values():
        output_path = directory / output["file"]
        if sha256_file(output_path) != output["sha256"]:
            raise ValueError(f"Dense GP output hash mismatch: {output_path}")
    return summary


def resolve_attempt(
    run_root: Path,
    base_identifier: str,
    phase: str,
    family: str,
    size: int,
    seed: int,
) -> tuple[str, Path, dict | None, list[dict]]:
    incomplete = []
    attempt = 1
    while True:
        identifier = (
            base_identifier if attempt == 1 else f"{base_identifier}-attempt{attempt}"
        )
        directory = run_root / identifier
        if not directory.exists():
            return identifier, directory, None, incomplete
        if (directory / "summary.json").is_file():
            return (
                identifier,
                directory,
                validate_summary(directory, phase, family, size, seed),
                incomplete,
            )
        state_path = directory / "runner_state.json"
        incomplete.append(
            {
                "run_id": identifier,
                "state_sha256": sha256_file(state_path) if state_path.is_file() else None,
                "state_status": read_json(state_path).get("status")
                if state_path.is_file()
                else "missing",
                "scientific_result_completed": False,
            }
        )
        attempt += 1


def select_family(run_root: Path) -> dict:
    scores = []
    for family in FAMILIES:
        global_scores = []
        regional_scores = []
        for size in CONTEXT_SIZES:
            for seed in CONTEXT_SEEDS:
                identifier = run_id("development", family, size, seed)
                _, _, summary, _ = resolve_attempt(
                    run_root,
                    identifier,
                    "development",
                    family,
                    size,
                    seed,
                )
                if summary is None:
                    raise FileNotFoundError(
                        f"No completed dense GP development cell: {identifier}"
                    )
                events = {item["region"]: item for item in summary["metrics"]["event"]}
                global_scores.append(float(events["full"]["brier"]))
                regional_scores.append(float(events["equal_region_mean"]["brier"]))
        scores.append(
            {
                "family": family,
                "development_cells": len(global_scores),
                "mean_global_brier": float(np.mean(global_scores)),
                "mean_equal_region_brier": float(np.mean(regional_scores)),
            }
        )
    rbf = next(item for item in scores if item["family"] == "rbf")
    matern = next(item for item in scores if item["family"] == "matern32")
    global_difference = abs(rbf["mean_global_brier"] - matern["mean_global_brier"])
    if global_difference > 1.0e-6:
        selected = min(scores, key=lambda item: item["mean_global_brier"])["family"]
        reason = "lowest mean global development Brier"
    else:
        regional_difference = abs(
            rbf["mean_equal_region_brier"] - matern["mean_equal_region_brier"]
        )
        if regional_difference > 0.0:
            selected = min(scores, key=lambda item: item["mean_equal_region_brier"])[
                "family"
            ]
            reason = "equal-region development Brier tie-break within 1e-6"
        else:
            selected = "rbf"
            reason = "RBF deterministic final tie-break"
    return {
        "selected_family": selected,
        "reason": reason,
        "global_brier_difference": global_difference,
        "family_scores": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument(
        "--max-new-cells",
        type=int,
        default=None,
        help="Operational pilot limit; omitted for the complete campaign",
    )
    args = parser.parse_args()
    if args.max_workers < 1 or args.threads_per_worker < 1:
        parser.error("worker and thread counts must be positive")
    if args.max_new_cells is not None and args.max_new_cells < 1:
        parser.error("--max-new-cells must be positive")
    repo = Path(".").resolve()
    worktree = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--short"], text=True
    ).splitlines()
    if worktree:
        raise RuntimeError("Dense GP campaign requires a clean worktree")
    source_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    root = repo / "ml4phy-paper"
    runner = root / "scripts/run_dense_gp_cell.py"
    protocol_path = root / "manifests/phase3_protocol_v1.json"
    gp_config_path = root / "configs/gp_protocol_v1.json"
    protocol = read_json(protocol_path)
    if protocol["status"] != "frozen_approved_not_started":
        raise ValueError("Phase 3 protocol is not approved and frozen")
    if protocol["resource_gates"]["dense_gp_hard_limit_seconds"] != HARD_LIMIT_SECONDS:
        raise ValueError("Dense GP hard limit mismatch")

    run_root = root / "runs/gp/phase3"
    campaign_dir = root / "runs/campaigns" / CAMPAIGN_ID
    record_path = campaign_dir / "campaign_record.json"
    log_path = campaign_dir / "campaign.log"
    resuming = campaign_dir.exists()
    if resuming:
        record = read_json(record_path)
        if record["status"] == "completed":
            raise RuntimeError("Dense GP campaign is already complete")
        record.setdefault("resumes", []).append(
            {
                "time": utc_now(),
                "source_commit": source_commit,
                "runner_sha256": sha256_file(runner),
                "campaign_script_sha256": sha256_file(Path(__file__)),
                "max_workers": args.max_workers,
                "threads_per_worker": args.threads_per_worker,
                "max_new_cells": args.max_new_cells,
            }
        )
        record["status"] = "running"
        record.setdefault("incomplete_attempts", [])
        if (
            not record["completed"]
            and float(record.get("accumulated_active_seconds", 0.0)) == 0.0
            and (run_root / run_id("development", "rbf", 250, 100)).exists()
        ):
            record["accumulated_active_seconds"] = 15.0
            record["unrecorded_interruption_recovery"] = {
                "reason": "Initial CLI smoke check invoked a campaign without argparse",
                "conservative_active_seconds_charged": 15.0,
                "result_completed": False,
            }
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
            "gp_config_sha256": sha256_file(gp_config_path),
            "hard_limit_seconds": HARD_LIMIT_SECONDS,
            "projected_seconds": protocol["resource_gates"][
                "dense_gp_projected_seconds"
            ],
            "deadline_utc_at_start": (
                datetime.now(UTC) + timedelta(seconds=HARD_LIMIT_SECONDS)
            ).isoformat().replace("+00:00", "Z"),
            "matrix": {
                "development_families": list(FAMILIES),
                "selected_final_family_count": 1,
                "context_sizes": list(CONTEXT_SIZES),
                "context_seeds": list(CONTEXT_SEEDS),
                "development_cells": 80,
                "final_cells": 40,
                "total_cells": 120,
                "pilot_cells_reused": 0,
            },
            "accumulated_active_seconds": 0.0,
            "completed": [],
            "failures": [],
            "terminated_at_limit": [],
            "incomplete_attempts": [],
            "family_selection": None,
            "execution_settings": {
                "max_workers": args.max_workers,
                "threads_per_worker": args.threads_per_worker,
                "max_new_cells": args.max_new_cells,
            },
        }
    write_json(record_path, record)
    completed_logical_cells = {
        (
            item["phase"],
            item["family"],
            item["context_size"],
            item["context_seed"],
        )
        for item in record["completed"]
    }
    session_start = time.perf_counter()

    def elapsed_total() -> float:
        return float(record.get("accumulated_before_session_seconds", 0.0)) + (
            time.perf_counter() - session_start
        )

    prior_accumulated = float(record.get("accumulated_active_seconds", 0.0))
    record["accumulated_before_session_seconds"] = prior_accumulated

    def save_progress() -> None:
        record["accumulated_active_seconds"] = elapsed_total()
        write_json(record_path, record)

    def prepare_cell(phase: str, family: str, size: int, seed: int) -> dict | None:
        base_identifier = run_id(phase, family, size, seed)
        identifier, directory, existing_summary, incomplete = resolve_attempt(
            run_root, base_identifier, phase, family, size, seed
        )
        known_incomplete = {item["run_id"] for item in record["incomplete_attempts"]}
        for item in incomplete:
            if item["run_id"] not in known_incomplete:
                record["incomplete_attempts"].append(item)
                known_incomplete.add(item["run_id"])
        if existing_summary is not None:
            summary = existing_summary
            logical_cell = (phase, family, size, seed)
            if logical_cell not in completed_logical_cells:
                record["completed"].append(
                    {
                        "run_id": identifier,
                        "phase": phase,
                        "family": family,
                        "context_size": size,
                        "context_seed": seed,
                        "summary_sha256": sha256_file(directory / "summary.json"),
                        "total_seconds": summary["runtime"]["total_seconds"],
                        "recovered_after_interruption": True,
                    }
                )
                completed_logical_cells.add(logical_cell)
                save_progress()
            return None
        return {
            "run_id": identifier,
            "directory": directory,
            "phase": phase,
            "family": family,
            "context_size": size,
            "context_seed": seed,
        }

    def execute_prepared(cell: dict) -> dict:
        remaining = HARD_LIMIT_SECONDS - elapsed_total()
        if remaining <= 0:
            return {**cell, "outcome": "time_limit", "returncode": None, "output": ""}
        command = [
            str(repo / ".venv/bin/python"),
            str(runner),
            "--repo",
            str(repo),
            "--phase",
            cell["phase"],
            "--family",
            cell["family"],
            "--context-size",
            str(cell["context_size"]),
            "--context-seed",
            str(cell["context_seed"]),
            "--run-id",
            cell["run_id"],
        ]
        thread_count = str(args.threads_per_worker)
        process = subprocess.Popen(
            command,
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": thread_count,
                "OPENBLAS_NUM_THREADS": thread_count,
                "MKL_NUM_THREADS": thread_count,
                "NUMEXPR_NUM_THREADS": thread_count,
                "VECLIB_MAXIMUM_THREADS": thread_count,
            },
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
            }
        if process.returncode != 0:
            return {
                **cell,
                "outcome": "failed",
                "returncode": process.returncode,
                "output": output,
            }
        return {**cell, "outcome": "completed", "returncode": 0, "output": output}

    new_cells_started = 0

    def execute_matrix(cells: list[tuple[str, str, int, int]], log) -> str:
        nonlocal new_cells_started
        prepared = []
        for phase, family, size, seed in cells:
            cell = prepare_cell(phase, family, size, seed)
            if cell is not None:
                prepared.append(cell)
        if args.max_new_cells is not None:
            remaining_slots = args.max_new_cells - new_cells_started
            if remaining_slots <= 0:
                return "operator_limit"
            prepared = prepared[:remaining_slots]
        if not prepared:
            return "complete"
        hit_time_limit = False
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.max_workers
        ) as executor:
            futures = []
            for cell in prepared:
                if elapsed_total() >= HARD_LIMIT_SECONDS:
                    hit_time_limit = True
                    break
                futures.append(executor.submit(execute_prepared, cell))
                new_cells_started += 1
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                output = result.pop("output")
                print(output, end="", flush=True)
                log.write(output)
                log.flush()
                outcome = result.pop("outcome")
                directory = result.pop("directory")
                if outcome == "time_limit":
                    record["terminated_at_limit"].append(result)
                    hit_time_limit = True
                elif outcome == "failed":
                    record["failures"].append(result)
                else:
                    summary = validate_summary(
                        directory,
                        result["phase"],
                        result["family"],
                        result["context_size"],
                        result["context_seed"],
                    )
                    logical_cell = (
                        result["phase"],
                        result["family"],
                        result["context_size"],
                        result["context_seed"],
                    )
                    if logical_cell not in completed_logical_cells:
                        record["completed"].append(
                            {
                                **{key: result[key] for key in (
                                    "run_id", "phase", "family", "context_size", "context_seed"
                                )},
                                "summary_sha256": sha256_file(directory / "summary.json"),
                                "total_seconds": summary["runtime"]["total_seconds"],
                                "recovered_after_interruption": False,
                            }
                        )
                        completed_logical_cells.add(logical_cell)
                save_progress()
        if record["failures"]:
            raise RuntimeError("One or more dense GP cells failed")
        if hit_time_limit:
            return "time_limit"
        if args.max_new_cells is not None and new_cells_started >= args.max_new_cells:
            return "operator_limit"
        return "complete"

    log_mode = "a" if resuming else "x"
    with log_path.open(log_mode) as log:
        try:
            development_cells = [
                ("development", family, size, seed)
                for size in CONTEXT_SIZES
                for family in FAMILIES
                for seed in CONTEXT_SEEDS
            ]
            outcome = execute_matrix(development_cells, log)
            if outcome != "complete":
                record["status"] = (
                    "partial_time_limit" if outcome == "time_limit" else "paused_operator_limit"
                )
                record["end_time"] = utc_now()
                save_progress()
                return
            if record["family_selection"] is None:
                record["family_selection"] = select_family(run_root)
                save_progress()
            selected = record["family_selection"]["selected_family"]
            final_cells = [
                ("final", selected, size, seed)
                for size in CONTEXT_SIZES
                for seed in CONTEXT_SEEDS
            ]
            outcome = execute_matrix(final_cells, log)
            if outcome != "complete":
                record["status"] = (
                    "partial_time_limit" if outcome == "time_limit" else "paused_operator_limit"
                )
                record["end_time"] = utc_now()
                save_progress()
                return
        except Exception:
            record["status"] = "failed"
            record["end_time"] = utc_now()
            save_progress()
            raise

    if len(record["completed"]) != 120 or record["failures"]:
        raise RuntimeError("Dense GP campaign matrix is incomplete")
    record["status"] = "completed"
    record["end_time"] = utc_now()
    record["log_sha256"] = sha256_file(log_path)
    save_progress()
    print(
        json.dumps(
            {
                "status": record["status"],
                "completed_cells": len(record["completed"]),
                "selected_family": record["family_selection"]["selected_family"],
                "accumulated_active_seconds": record["accumulated_active_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
