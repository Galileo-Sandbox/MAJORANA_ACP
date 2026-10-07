#!/usr/bin/env python3
"""Run one approved Phase 2 training job against an exact subset HDF5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import yaml

IDENTITY_FIELDS = ("run_number", "detector", "id", "tp0")
ARCHITECTURES = ("m0", "m1", "m2", "ours")
BUDGETS = (2000, 5000)
TRAINING_SEEDS = (0, 1, 2)


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


def identity_hash(path: Path) -> tuple[int, str]:
    with h5py.File(path, "r") as handle:
        identities = np.column_stack(
            [handle[field][:].astype("<i8", copy=False) for field in IDENTITY_FIELDS]
        )
    order = np.lexsort(
        tuple(identities[:, index] for index in reversed(range(identities.shape[1])))
    )
    payload = np.ascontiguousarray(identities[order], dtype="<i8").tobytes()
    return identities.shape[0], hashlib.sha256(payload).hexdigest()


def validate_dependency(source_root: Path, compatibility: dict) -> None:
    for relative, expected in compatibility["source_hashes"].items():
        path = source_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"RESUM_FLEX source mismatch for {relative}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--budget", type=int, choices=BUDGETS, required=True)
    parser.add_argument("--training-seed", type=int, choices=TRAINING_SEEDS, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.run_id.replace("-", "").replace("_", "").isalnum():
        parser.error("run-id may contain only letters, numbers, hyphens, and underscores")

    repo = args.repo.resolve()
    root = repo / "ml4phy-paper"
    run_dir = root / "runs/phase2/training" / args.run_id
    artifact_dir = run_dir / "artifacts"
    config_path = root / "configs/phase2_training_v1.json"
    protocol_path = root / "manifests/phase2_protocol_v1.json"
    compatibility_path = root / "manifests/resum_flex_compatibility.json"
    base_protocol_path = root / "manifests/frozen_protocol_v1.json"
    source_root = root / "local/resum-flex-edba6a"
    python = repo / ".venv/bin/python"
    if run_dir.exists():
        parser.error(f"Refusing to reuse run directory: {run_dir}")
    for path, description in (
        (config_path, "Phase 2 configuration"),
        (protocol_path, "Phase 2 protocol"),
        (compatibility_path, "dependency compatibility manifest"),
        (base_protocol_path, "base protocol"),
        (source_root / "core/__init__.py", "RESUM_FLEX source overlay"),
        (python, "repository Python"),
    ):
        if not path.is_file():
            parser.error(f"Missing {description}: {path}")

    config = read_json(config_path)
    protocol = read_json(protocol_path)
    compatibility = read_json(compatibility_path)
    base_protocol = read_json(base_protocol_path)
    if config["status"] != "frozen" or protocol["status"] != "frozen_approved_not_started":
        parser.error("Phase 2 protocol is not in the approved frozen state")
    validate_dependency(source_root, compatibility)
    subset = next(item for item in protocol["subsets"] if item["budget"] == args.budget)
    subset_path = repo / subset["logical_path"]
    if not subset_path.is_file() or sha256_file(subset_path) != subset["sha256"]:
        parser.error("Phase 2 subset HDF5 is missing or has the wrong hash")
    count, observed_identity_hash = identity_hash(subset_path)
    if count != args.budget or observed_identity_hash != subset["identity_sha256"]:
        parser.error("Phase 2 subset count or identity hash mismatch")

    base_config_path = repo / config["base_configs"][args.architecture]
    if sha256_file(base_config_path) != config["base_config_sha256"][args.architecture]:
        parser.error("Base architecture configuration hash mismatch")
    validation_path = repo / base_protocol["inputs"]["historical_development"]["logical_path"]
    if sha256_file(validation_path) != base_protocol["inputs"]["historical_development"]["sha256"]:
        parser.error("Validation prediction input hash mismatch")
    classifier_path = repo / config["fixed_classifier"]
    classifier_hash = "ad8f147aabe1fcd13f79932eaaa2950a0054253120f10b175692420c0236429f"
    if sha256_file(classifier_path) != classifier_hash:
        parser.error("Classifier configuration hash mismatch")

    worktree = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--short"], text=True
    ).splitlines()
    if worktree:
        parser.error("Phase 2 training requires a clean worktree: " + "; ".join(worktree))
    source_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()

    with base_config_path.open() as stream:
        resolved = yaml.safe_load(stream)
    resolved["name"] = f"ml4ps_phase2_b{args.budget}_{args.architecture}_seed{args.training_seed}"
    resolved["out_dir"] = str(artifact_dir.relative_to(repo))
    resolved["train_predictions_path"] = str(subset_path.relative_to(repo))
    resolved["training"]["seed"] = args.training_seed
    if resolved["training"]["n_steps"] != 3000:
        parser.error("Phase 2 requires the fixed 3,000-step schedule")
    if resolved["validation_predictions_path"] != base_protocol["inputs"]["historical_development"]["logical_path"]:
        parser.error("Base configuration validation input mismatch")
    if resolved["upstream_classifier_config"] != config["fixed_classifier"]:
        parser.error("Base configuration classifier mismatch")
    if resolved.get("device") != "cuda":
        parser.error("Phase 2 training requires CUDA")

    resolved_path = run_dir / "resolved_config.yaml"
    command = [
        str(python),
        "-m",
        "majorana_acp.cut_acceptance.cli",
        str(resolved_path),
        "--seed",
        str(args.training_seed),
    ]
    preview = {
        "run_id": args.run_id,
        "architecture": args.architecture,
        "training_budget": args.budget,
        "training_seed": args.training_seed,
        "training_steps": 3000,
        "source_commit": source_commit,
        "base_config": str(base_config_path.relative_to(repo)),
        "base_config_sha256": sha256_file(base_config_path),
        "subset_input": str(subset_path.relative_to(repo)),
        "subset_file_sha256": subset["sha256"],
        "subset_identity_sha256": subset["identity_sha256"],
        "sampling_eligible_unique_events": subset["sampling_eligible_unique_events"],
        "density_pool_unique_events": args.budget,
        "command": command,
    }
    if args.dry_run:
        print(json.dumps(preview, indent=2))
        return

    run_dir.mkdir(parents=True, exist_ok=False)
    with resolved_path.open("x") as stream:
        yaml.safe_dump(resolved, stream, sort_keys=False)
    record_path = run_dir / "runner_record.json"
    record = {
        "schema_version": 1,
        "status": "running",
        "start_time": utc_now(),
        **preview,
        "resolved_config_sha256": sha256_file(resolved_path),
        "phase2_protocol_sha256": sha256_file(protocol_path),
        "phase2_config_sha256": sha256_file(config_path),
        "dependency_commit": compatibility["commit"],
        "dependency_source_hashes": compatibility["source_hashes"],
        "classifier_config_sha256": classifier_hash,
        "validation_predictions_sha256": sha256_file(validation_path),
    }
    write_json(record_path, record)

    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(source_root), str(repo), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    environment["MPLCONFIGDIR"] = str(run_dir / "matplotlib-cache")
    log_path = run_dir / "training.log"
    timer = time.perf_counter()
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
    record["wall_seconds"] = float(time.perf_counter() - timer)
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
    if returncode == 0 and set(record["outputs"]) == set(expected_outputs):
        summary = read_json(artifact_dir / "run_summary.json")
        with np.load(artifact_dir / "training_pool.npz") as pool:
            retained_count = int(pool["bin_event_counts"].sum())
            retained_bins = int(pool["bin_event_counts"].size)
        record["post_run_validation"] = {
            "summary_train_events": int(summary["n_train_events"]),
            "training_pool_retained_events": retained_count,
            "training_pool_kept_bins": retained_bins,
            "density_reconstruction_input": resolved["train_predictions_path"],
            "density_reconstruction_uses_full_pool": False,
        }
        if int(summary["n_train_events"]) != args.budget:
            record["status"] = "failed_validation"
        if retained_count != subset["sampling_eligible_unique_events"]:
            record["status"] = "failed_validation"
        if retained_bins != subset["kept_bin_count"]:
            record["status"] = "failed_validation"
    write_json(record_path, record)
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
    if returncode != 0 or record["status"] != "completed":
        raise SystemExit(returncode or 1)
    if set(record["outputs"]) != set(expected_outputs):
        raise RuntimeError("Training returned successfully without all required outputs")


if __name__ == "__main__":
    main()
