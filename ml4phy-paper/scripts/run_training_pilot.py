#!/usr/bin/env python3
"""Run one isolated paper training job with validated provenance and streamed logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml


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
    with path.open("w") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def validate_dependency_source(source_root: Path, compatibility: dict) -> None:
    for relative, expected in compatibility["source_hashes"].items():
        path = source_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"RESUM_FLEX source mismatch for {relative}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--n-steps-override", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.run_id.replace("-", "").replace("_", "").isalnum():
        parser.error("run-id may contain only letters, numbers, hyphens, and underscores.")
    if args.n_steps_override is not None and args.n_steps_override <= 0:
        parser.error("n-steps-override must be positive.")

    repo = args.repo.resolve()
    config_path = (repo / args.config).resolve()
    run_dir = repo / "ml4phy-paper/runs/training" / args.run_id
    artifact_dir = run_dir / "artifacts"
    source_root = repo / "ml4phy-paper/local/resum-flex-edba6a"
    compatibility_path = repo / "ml4phy-paper/manifests/resum_flex_compatibility.json"
    registry_path = repo / "ml4phy-paper/configs/recovered_models_v1.json"
    protocol_path = repo / "ml4phy-paper/manifests/frozen_protocol_v1.json"
    python = repo / ".venv/bin/python"
    if run_dir.exists():
        parser.error(f"Refusing to reuse run directory: {run_dir}")
    for path, description in (
        (config_path, "training configuration"),
        (source_root / "core/__init__.py", "RESUM_FLEX source overlay"),
        (compatibility_path, "dependency compatibility manifest"),
        (registry_path, "recovered model registry"),
        (protocol_path, "frozen protocol manifest"),
        (python, "repository Python"),
    ):
        if not path.is_file():
            parser.error(f"Missing {description}: {path}")

    compatibility = read_json(compatibility_path)
    registry = read_json(registry_path)
    protocol = read_json(protocol_path)
    validate_dependency_source(source_root, compatibility)
    for logical_name, relative, expected in (
        (
            "training predictions",
            registry["training_predictions"],
            registry["training_predictions_sha256"],
        ),
        (
            "classifier configuration",
            registry["classifier_config"],
            registry["classifier_config_sha256"],
        ),
    ):
        path = repo / relative
        if not path.is_file() or sha256_file(path) != expected:
            parser.error(f"{logical_name} does not match the recovered registry: {path}")
    validation_path = repo / protocol["inputs"]["historical_development"]["logical_path"]
    if (
        not validation_path.is_file()
        or sha256_file(validation_path)
        != protocol["inputs"]["historical_development"]["sha256"]
    ):
        parser.error("Validation predictions do not match the frozen protocol.")

    worktree_status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--short"], text=True
    ).splitlines()
    if worktree_status:
        parser.error("Training requires a clean worktree: " + "; ".join(worktree_status))
    source_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()

    with config_path.open() as stream:
        resolved = yaml.safe_load(stream)
    if resolved["train_predictions_path"] != registry["training_predictions"]:
        parser.error("Training configuration points to an unexpected training input.")
    if (
        resolved["validation_predictions_path"]
        != protocol["inputs"]["historical_development"]["logical_path"]
    ):
        parser.error("Training configuration points to an unexpected validation input.")
    if resolved["upstream_classifier_config"] != registry["classifier_config"]:
        parser.error("Training configuration points to an unexpected classifier configuration.")
    resolved["out_dir"] = str(artifact_dir.relative_to(repo))
    if args.n_steps_override is not None:
        resolved["training"]["n_steps"] = args.n_steps_override
    if resolved["training"]["seed"] != 0:
        parser.error("This authorized pilot is restricted to training seed 0.")
    if resolved.get("device") != "cuda":
        parser.error("The paper training pilot must fail loudly when CUDA is unavailable.")

    command = [
        str(python),
        "-m",
        "majorana_acp.cut_acceptance.cli",
        str(run_dir / "resolved_config.yaml"),
        "--seed",
        "0",
    ]
    preview = {
        "run_id": args.run_id,
        "config": str(config_path.relative_to(repo)),
        "config_sha256": sha256_file(config_path),
        "resolved_out_dir": resolved["out_dir"],
        "training_steps": resolved["training"]["n_steps"],
        "training_seed": resolved["training"]["seed"],
        "source_commit": source_commit,
        "dependency_commit": compatibility["commit"],
        "command": command,
    }
    if args.dry_run:
        print(json.dumps(preview, indent=2))
        return

    run_dir.mkdir(parents=True)
    resolved_config_path = run_dir / "resolved_config.yaml"
    with resolved_config_path.open("x") as stream:
        yaml.safe_dump(resolved, stream, sort_keys=False)

    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(source_root), str(repo), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    environment["MPLCONFIGDIR"] = str(run_dir / "matplotlib-cache")
    runner_record_path = run_dir / "runner_record.json"
    record = {
        "schema_version": 1,
        "status": "running",
        "start_time": utc_now(),
        **preview,
        "resolved_config_sha256": sha256_file(resolved_config_path),
        "dependency_source_hashes": compatibility["source_hashes"],
        "training_predictions_sha256": registry["training_predictions_sha256"],
        "classifier_config_sha256": registry["classifier_config_sha256"],
        "validation_predictions_sha256": protocol["inputs"]["historical_development"][
            "sha256"
        ],
    }
    write_json(runner_record_path, record)

    timer_start = time.perf_counter()
    log_path = run_dir / "training.log"
    with log_path.open("x") as log_stream:
        process = subprocess.Popen(
            command,
            cwd=repo,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_stream.write(line)
            log_stream.flush()
        returncode = process.wait()

    record["end_time"] = utc_now()
    record["wall_seconds"] = float(time.perf_counter() - timer_start)
    record["returncode"] = returncode
    record["status"] = "completed" if returncode == 0 else "failed"
    record["log_sha256"] = sha256_file(log_path)
    expected_outputs = ("cnp.ckpt", "training_pool.npz", "run_summary.json")
    record["outputs"] = {
        name: {
            "bytes": (artifact_dir / name).stat().st_size,
            "sha256": sha256_file(artifact_dir / name),
        }
        for name in expected_outputs
        if (artifact_dir / name).is_file()
    }
    write_json(runner_record_path, record)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "status": record["status"],
                "wall_seconds": record["wall_seconds"],
                "outputs": sorted(record["outputs"]),
            }
        )
    )
    if returncode != 0:
        raise SystemExit(returncode)
    if set(record["outputs"]) != set(expected_outputs):
        raise RuntimeError("Training returned successfully without all required outputs.")


if __name__ == "__main__":
    main()
