#!/usr/bin/env python3
"""Evaluate one neural cell with an explicitly frozen nested context subset."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

IDENTITY_FIELDS = ("run_number", "detector", "id", "tp0")
ALLOWED_CONTEXT_SIZES = (250, 500, 1000, 2000)
ALLOWED_CONTEXT_SEEDS = tuple(range(100, 110))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_matrix(handle: h5py.File, rows: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [handle[field][rows].astype("<i8", copy=False) for field in IDENTITY_FIELDS]
    )


def identity_hash(identities: np.ndarray) -> str:
    order = np.lexsort(
        tuple(identities[:, index] for index in reversed(range(identities.shape[1])))
    )
    return hashlib.sha256(
        np.ascontiguousarray(identities[order], dtype="<i8").tobytes()
    ).hexdigest()


def read_json(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def write_json(path: Path, payload: dict) -> None:
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--phase", choices=("development", "final"), required=True)
    parser.add_argument("--context-seed", type=int, choices=ALLOWED_CONTEXT_SEEDS, required=True)
    parser.add_argument("--context-size", type=int, choices=ALLOWED_CONTEXT_SIZES, required=True)
    parser.add_argument("--n-mc", type=int, default=50)
    parser.add_argument("--grid-spacing-kev", type=float, default=1.0)
    parser.add_argument("--run-kind", choices=("pilot", "evaluation"), default="evaluation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-registry",
        type=Path,
        default=Path("ml4phy-paper/configs/extension_models_v1.json"),
    )
    parser.add_argument(
        "--extension-protocol",
        type=Path,
        default=Path("ml4phy-paper/manifests/extension_protocol_v1.json"),
    )
    parser.add_argument(
        "--context-archive",
        type=Path,
        default=Path("ml4phy-paper/local/protocol/extension_context_roles_v1.npz"),
    )
    parser.add_argument(
        "--base-protocol",
        type=Path,
        default=Path("ml4phy-paper/manifests/frozen_protocol_v1.json"),
    )
    parser.add_argument(
        "--base-role-archive",
        type=Path,
        default=Path("ml4phy-paper/local/protocol/frozen_roles_v1.npz"),
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    output_dir = (repo / args.output_dir).resolve()
    if output_dir.exists():
        parser.error(f"Refusing to reuse output directory: {output_dir}")
    if args.n_mc <= 1 or args.grid_spacing_kev <= 0:
        parser.error("n-mc must exceed one and grid spacing must be positive")

    extension_path = (repo / args.extension_protocol).resolve()
    context_archive_path = (repo / args.context_archive).resolve()
    base_protocol_path = (repo / args.base_protocol).resolve()
    base_archive_path = (repo / args.base_role_archive).resolve()
    model_registry_path = (repo / args.model_registry).resolve()
    extension = read_json(extension_path)
    base = read_json(base_protocol_path)
    registry = read_json(model_registry_path)
    if args.model_id not in registry["models"]:
        parser.error(f"Unknown model ID {args.model_id!r}")
    archive_spec = extension["context_protocol"]["local_archive"]
    if sha256_file(context_archive_path) != archive_spec["sha256"]:
        raise ValueError("Extension context archive hash mismatch")
    if sha256_file(base_archive_path) != base["local_role_archive"]["sha256"]:
        raise ValueError("Base role archive hash mismatch")

    subset_key = (
        f"{args.phase}_context_s{args.context_seed}_n{args.context_size}_rows"
    )
    with np.load(context_archive_path) as archive:
        if subset_key not in archive:
            raise KeyError(f"Missing context subset {subset_key}")
        context_rows = archive[subset_key].astype(np.int64)
    with np.load(base_archive_path) as archive:
        target_rows = archive[f"{args.phase}_target_rows"].astype(np.int64)
        threshold = float(archive["fixed_threshold"])
    if context_rows.size != args.context_size or np.unique(context_rows).size != args.context_size:
        raise ValueError("Context subset size or uniqueness check failed")
    if threshold != 0.540643572807312:
        raise ValueError("Frozen threshold mismatch")

    phase_input = "historical_development" if args.phase == "development" else "full_test"
    h5_path = repo / base["inputs"][phase_input]["logical_path"]
    if sha256_file(h5_path) != base["inputs"][phase_input]["sha256"]:
        raise ValueError("Phase HDF5 hash mismatch")
    with h5py.File(h5_path, "r") as handle:
        observed_subset_hash = identity_hash(identity_matrix(handle, np.sort(context_rows)))
    expected_subset = next(
        item
        for item in extension["context_protocol"]["subsets"]
        if item["phase"] == args.phase
        and item["context_seed"] == args.context_seed
        and item["context_size"] == args.context_size
    )
    if observed_subset_hash != expected_subset["identity_sha256"]:
        raise ValueError("Context subset identity hash mismatch")

    run_id = output_dir.name
    temp_dir = repo / "ml4phy-paper/local/extension_eval_tmp" / run_id
    if temp_dir.exists():
        parser.error(f"Refusing to reuse temporary directory: {temp_dir}")
    temp_dir.mkdir(parents=True)
    synthetic_archive_path = temp_dir / "roles.npz"
    np.savez(
        synthetic_archive_path,
        fixed_threshold=np.array(threshold, dtype=np.float64),
        **{
            f"{args.phase}_context_reservoir_rows": np.sort(context_rows),
            f"{args.phase}_target_rows": target_rows,
        },
    )
    synthetic_protocol = copy.deepcopy(base)
    synthetic_protocol["source_commit"] = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    synthetic_protocol["local_role_archive"] = {
        "logical_path": str(synthetic_archive_path.relative_to(repo)),
        "sha256": sha256_file(synthetic_archive_path),
    }
    synthetic_protocol["roles"][f"{args.phase}_context_reservoir"] = {
        "n_events": args.context_size,
        "identity_sha256": observed_subset_hash,
        "construction": f"frozen extension subset {subset_key}",
    }
    synthetic_protocol_path = temp_dir / "protocol.json"
    write_json(synthetic_protocol_path, synthetic_protocol)

    evaluator_path = repo / "ml4phy-paper/scripts/evaluate_fixed_protocol.py"
    command = [
        str(repo / ".venv/bin/python"),
        str(evaluator_path),
        "--repo",
        str(repo),
        "--model-id",
        args.model_id,
        "--phase",
        args.phase,
        "--context-seed",
        str(args.context_seed),
        "--dropout-seed",
        "10100",
        "--n-context",
        str(args.context_size),
        "--n-mc",
        str(args.n_mc),
        "--max-context-per-pass",
        str(args.context_size),
        "--grid-spacing-kev",
        str(args.grid_spacing_kev),
        "--run-kind",
        args.run_kind,
        "--output-dir",
        str(output_dir),
        "--model-registry",
        str(model_registry_path),
        "--protocol-manifest",
        str(synthetic_protocol_path),
        "--role-archive",
        str(synthetic_archive_path),
    ]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["MPLCONFIGDIR"] = str(temp_dir / "matplotlib-cache")
    source_root = repo / "ml4phy-paper/local/resum-flex-edba6a"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(source_root), str(repo), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    subprocess.run(command, cwd=repo, env=environment, check=True)

    summary_path = output_dir / "summary.json"
    summary = read_json(summary_path)
    summary["wrapper_command"] = " ".join(shlex.quote(value) for value in sys.argv)
    summary["extension_protocol"] = {
        "manifest": str(extension_path.relative_to(repo)),
        "manifest_sha256": sha256_file(extension_path),
        "context_archive": str(context_archive_path.relative_to(repo)),
        "context_archive_sha256": sha256_file(context_archive_path),
        "subset_key": subset_key,
        "subset_identity_sha256": observed_subset_hash,
        "construction": extension["context_protocol"]["construction"],
        "membership_at_2000": extension["context_protocol"]["membership_at_2000"],
        "analysis_status": extension["analysis_status"],
    }
    summary["randomness"]["context_permutation"] = "frozen outcome-blind SHA-256 priority"
    summary["randomness"]["dropout_rule"] = "fixed 10100"
    summary["counts"]["context_size"] = args.context_size
    updated_summary_path = output_dir / "summary.updated.json"
    with updated_summary_path.open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    updated_summary_path.replace(summary_path)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "status": "extension_metadata_added",
                "subset_key": subset_key,
                "summary_sha256": sha256_file(summary_path),
            }
        )
    )


if __name__ == "__main__":
    main()
