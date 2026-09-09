#!/usr/bin/env python3
"""Re-evaluate dirty-provenance neural cells and compare numerical arrays."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ALLOWED_DIRTY_STATUS = {
    (" M ml4phy-paper/scripts/run_extension_neural_campaign.py",),
    (" M ml4phy-paper/scripts/evaluate_extension_neural.py",),
}


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
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def compare_archives(original: Path, rerun: Path) -> dict:
    comparisons = {}
    with np.load(original) as left, np.load(rerun) as right:
        if set(left.files) != set(right.files):
            raise ValueError(f"Archive fields differ: {original}, {rerun}")
        for key in left.files:
            left_value = left[key]
            right_value = right[key]
            exact = np.array_equal(left_value, right_value, equal_nan=True)
            if np.issubdtype(left_value.dtype, np.number):
                close = np.allclose(
                    left_value, right_value, rtol=0.0, atol=1e-12, equal_nan=True
                )
                maximum_difference = float(
                    np.nanmax(np.abs(left_value - right_value))
                ) if left_value.size else 0.0
            else:
                close = exact
                maximum_difference = 0.0
            if not close:
                raise ValueError(f"Numerical rerun mismatch for {original.name}:{key}")
            comparisons[key] = {
                "exact": bool(exact),
                "absolute_tolerance": 1e-12,
                "maximum_absolute_difference": maximum_difference,
            }
    return comparisons


def main() -> None:
    repo = Path(".").resolve()
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--short"], text=True
    ).splitlines()
    if status:
        raise RuntimeError("Clean reruns require a clean worktree: " + "; ".join(status))
    source_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    run_root = repo / "ml4phy-paper/runs/extension_eval"
    dirty = []
    for summary_path in sorted(run_root.glob("*/summary.json")):
        if summary_path.parent.name.endswith("-cleanrerun"):
            continue
        summary = read_json(summary_path)
        dirty_status = tuple(summary.get("source_worktree_status", []))
        if not dirty_status:
            continue
        if dirty_status not in ALLOWED_DIRTY_STATUS:
            raise ValueError(f"Unexpected dirty status in {summary_path}: {dirty_status}")
        if not summary["model"]["id"].startswith("m0_seed"):
            raise ValueError(f"Dirty cell is not an M0 cell: {summary_path}")
        dirty.append((summary_path.parent, summary))
    if len(dirty) != 25:
        raise ValueError(f"Expected 25 dirty-provenance cells, found {len(dirty)}")

    record_dir = repo / "ml4phy-paper/runs/campaigns/20260909-phase1-clean-reruns-v1"
    record_dir.mkdir(parents=True, exist_ok=False)
    record_path = record_dir / "record.json"
    evaluator = repo / "ml4phy-paper/scripts/evaluate_extension_neural.py"
    record = {
        "schema_version": 1,
        "status": "running",
        "start_time": utc_now(),
        "source_commit": source_commit,
        "script_sha256": sha256_file(Path(__file__)),
        "evaluator_sha256": sha256_file(evaluator),
        "expected_cells": len(dirty),
        "completed": [],
    }
    start = time.perf_counter()
    for original_dir, original_summary in dirty:
        output_dir = original_dir.with_name(original_dir.name + "-cleanrerun")
        command = [
            str(repo / ".venv/bin/python"),
            str(evaluator),
            "--repo",
            str(repo),
            "--model-id",
            original_summary["model"]["id"],
            "--phase",
            "final",
            "--context-seed",
            str(original_summary["randomness"]["context_seed"]),
            "--context-size",
            str(original_summary["counts"]["context_draw"]),
            "--n-mc",
            "50",
            "--grid-spacing-kev",
            "1.0",
            "--run-kind",
            "evaluation",
            "--output-dir",
            str(output_dir),
        ]
        subprocess.run(command, cwd=repo, check=True)
        rerun_summary_path = output_dir / "summary.json"
        rerun_summary = read_json(rerun_summary_path)
        if rerun_summary["source_worktree_status"]:
            raise ValueError(f"Clean rerun recorded a dirty tree: {output_dir}")
        if (
            rerun_summary["protocol"]["context_draw_identity_sha256"]
            != original_summary["protocol"]["context_draw_identity_sha256"]
        ):
            raise ValueError(f"Context identity changed in {output_dir}")
        archive_comparisons = {
            name: compare_archives(original_dir / name, output_dir / name)
            for name in ("event_predictions.npz", "curve.npz")
        }
        record["completed"].append(
            {
                "original_run_id": original_summary["run_id"],
                "original_worktree_status": original_summary["source_worktree_status"],
                "original_summary_sha256": sha256_file(original_dir / "summary.json"),
                "rerun_id": rerun_summary["run_id"],
                "rerun_summary_sha256": sha256_file(rerun_summary_path),
                "archive_comparisons": archive_comparisons,
            }
        )
        temporary = record_path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(record_path)
    record["status"] = "completed"
    record["end_time"] = utc_now()
    record["wall_seconds"] = time.perf_counter() - start
    temporary = record_path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(record_path)
    print(
        json.dumps(
            {
                "status": record["status"],
                "rerun_cells": len(record["completed"]),
                "wall_seconds": record["wall_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
