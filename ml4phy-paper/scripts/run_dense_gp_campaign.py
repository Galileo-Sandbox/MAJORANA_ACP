#!/usr/bin/env python3
"""Run or resume the frozen dense Bernoulli-GP campaign under a hard cap."""

from __future__ import annotations

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


def select_family(run_root: Path) -> dict:
    scores = []
    for family in FAMILIES:
        global_scores = []
        regional_scores = []
        for size in CONTEXT_SIZES:
            for seed in CONTEXT_SEEDS:
                identifier = run_id("development", family, size, seed)
                summary = validate_summary(
                    run_root / identifier, "development", family, size, seed
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
            "family_selection": None,
        }
    write_json(record_path, record)
    completed_ids = {item["run_id"] for item in record["completed"]}
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

    def execute_cell(phase: str, family: str, size: int, seed: int, log) -> bool:
        identifier = run_id(phase, family, size, seed)
        directory = run_root / identifier
        if identifier in completed_ids or (directory / "summary.json").is_file():
            summary = validate_summary(directory, phase, family, size, seed)
            if identifier not in completed_ids:
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
                completed_ids.add(identifier)
                save_progress()
            return True
        remaining = HARD_LIMIT_SECONDS - elapsed_total()
        if remaining <= 0:
            return False
        command = [
            str(repo / ".venv/bin/python"),
            str(runner),
            "--repo",
            str(repo),
            "--phase",
            phase,
            "--family",
            family,
            "--context-size",
            str(size),
            "--context-seed",
            str(seed),
            "--run-id",
            identifier,
        ]
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
            log.write(output)
            log.flush()
            record["terminated_at_limit"].append(
                {
                    "run_id": identifier,
                    "phase": phase,
                    "family": family,
                    "context_size": size,
                    "context_seed": seed,
                    "returncode": process.returncode,
                }
            )
            return False
        print(output, end="", flush=True)
        log.write(output)
        log.flush()
        if process.returncode != 0:
            record["failures"].append(
                {"run_id": identifier, "returncode": process.returncode}
            )
            save_progress()
            raise RuntimeError(f"Dense GP cell failed: {identifier}")
        summary = validate_summary(directory, phase, family, size, seed)
        record["completed"].append(
            {
                "run_id": identifier,
                "phase": phase,
                "family": family,
                "context_size": size,
                "context_seed": seed,
                "summary_sha256": sha256_file(directory / "summary.json"),
                "total_seconds": summary["runtime"]["total_seconds"],
                "recovered_after_interruption": False,
            }
        )
        completed_ids.add(identifier)
        save_progress()
        return True

    log_mode = "a" if resuming else "x"
    with log_path.open(log_mode) as log:
        try:
            for size in CONTEXT_SIZES:
                for family in FAMILIES:
                    for seed in CONTEXT_SEEDS:
                        if not execute_cell("development", family, size, seed, log):
                            record["status"] = "partial_time_limit"
                            record["end_time"] = utc_now()
                            save_progress()
                            return
            if record["family_selection"] is None:
                record["family_selection"] = select_family(run_root)
                save_progress()
            selected = record["family_selection"]["selected_family"]
            for size in CONTEXT_SIZES:
                for seed in CONTEXT_SEEDS:
                    if not execute_cell("final", selected, size, seed, log):
                        record["status"] = "partial_time_limit"
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
