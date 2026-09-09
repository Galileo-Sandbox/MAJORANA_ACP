#!/usr/bin/env python3
"""Freeze nested context subsets and model-selection rules for the extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import sklearn

IDENTITY_FIELDS = ("run_number", "detector", "id", "tp0")
CONTEXT_SEEDS = tuple(range(100, 110))
CONTEXT_SIZES = (250, 500, 1000, 2000)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def identity_matrix(handle: h5py.File, rows: np.ndarray) -> np.ndarray:
    order = np.argsort(rows)
    inverse = np.argsort(order)
    sorted_rows = rows[order]
    return np.column_stack(
        [
            handle[field][sorted_rows].astype("<i8", copy=False)[inverse]
            for field in IDENTITY_FIELDS
        ]
    )


def identity_hash(identities: np.ndarray) -> str:
    ordered = identities[
        np.lexsort(
            tuple(
                identities[:, index]
                for index in reversed(range(identities.shape[1]))
            )
        )
    ]
    return hashlib.sha256(np.ascontiguousarray(ordered, dtype="<i8").tobytes()).hexdigest()


def outcome_blind_order(identities: np.ndarray, phase: str, context_seed: int) -> np.ndarray:
    salt = f"ml4ps-extension-context-v1|{phase}|{context_seed}|".encode()
    priorities = [
        hashlib.sha256(salt + np.asarray(row, dtype="<i8").tobytes()).digest()
        for row in identities
    ]
    return np.asarray(sorted(range(len(priorities)), key=priorities.__getitem__), dtype=np.int64)


def development_run(repo: Path, seed: int) -> Path:
    if seed == 100:
        run_id = "20260908-cell17-dev-s100-mc50"
    else:
        run_id = f"20260908-cell17-dev-ctx-s{seed}-drop10100-mc50"
    return repo / "ml4phy-paper/runs" / run_id


def final_run(repo: Path, seed: int) -> Path:
    return (
        repo
        / "ml4phy-paper/runs"
        / f"20260908-cell17_seed0_recovered-final-ctx-s{seed}-drop10100-mc50"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument(
        "--local-output",
        type=Path,
        default=Path("ml4phy-paper/local/protocol/extension_context_roles_v1.npz"),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("ml4phy-paper/manifests/extension_protocol_v1.json"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    local_output = (repo / args.local_output).resolve()
    manifest_output = (repo / args.manifest_output).resolve()
    for output in (local_output, manifest_output):
        if output.exists():
            parser.error(f"Refusing to overwrite existing output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    original_protocol_path = repo / "ml4phy-paper/manifests/frozen_protocol_v1.json"
    original_role_archive_path = (
        repo / "ml4phy-paper/local/protocol/frozen_roles_v1.npz"
    )
    development_h5_path = (
        repo / "runs/small_data_configs/simple_cnn_small/eval/predictions.h5"
    )
    final_h5_path = (
        repo / "runs/small_data_configs/simple_cnn_small/eval_full_test/predictions.h5"
    )
    original_protocol = read_json(original_protocol_path)
    archive: dict[str, np.ndarray] = {}
    subset_manifest = []
    input_runs = []
    context_draw_unions = []
    with np.load(original_role_archive_path) as original_roles:
        reservoir_rows = {
            "development": original_roles[
                "development_context_reservoir_rows"
            ].astype(np.int64),
            "final": original_roles["final_context_reservoir_rows"].astype(np.int64),
        }

    with (
        h5py.File(development_h5_path, "r") as development_h5,
        h5py.File(final_h5_path, "r") as final_h5,
    ):
        for phase, source, run_locator, expected_reservoir_hash in (
            (
                "development",
                development_h5,
                development_run,
                original_protocol["roles"]["development_context_reservoir"][
                    "identity_sha256"
                ],
            ),
            (
                "final",
                final_h5,
                final_run,
                original_protocol["roles"]["final_context_reservoir"]["identity_sha256"],
            ),
        ):
            union_rows = []
            for context_seed in CONTEXT_SEEDS:
                run = run_locator(repo, context_seed)
                summary_path = run / "summary.json"
                event_path = run / "event_predictions.npz"
                summary = read_json(summary_path)
                with np.load(event_path) as prediction:
                    original_rows = prediction["context_rows"].astype(np.int64)
                if original_rows.size != 2000 or np.unique(original_rows).size != 2000:
                    raise ValueError(f"Original context is not 2,000 unique rows: {run.name}")
                if not np.all(np.isin(original_rows, reservoir_rows[phase])):
                    raise ValueError(f"Original context is outside its frozen reservoir: {run.name}")
                identities = identity_matrix(source, original_rows)
                if (
                    identity_hash(identities)
                    != summary["protocol"]["context_draw_identity_sha256"]
                ):
                    raise ValueError(f"Original context hash mismatch: {run.name}")
                order = outcome_blind_order(identities, phase, context_seed)
                ordered_rows = original_rows[order]
                for context_size in CONTEXT_SIZES:
                    rows = ordered_rows[:context_size]
                    key = f"{phase}_context_s{context_seed}_n{context_size}_rows"
                    archive[key] = rows
                    subset_manifest.append(
                        {
                            "phase": phase,
                            "context_seed": context_seed,
                            "context_size": context_size,
                            "identity_sha256": identity_hash(identity_matrix(source, rows)),
                            "nested_parent_size": next(
                                (
                                    size
                                    for size in CONTEXT_SIZES
                                    if size > context_size
                                ),
                                None,
                            ),
                        }
                    )
                union_rows.append(original_rows)
                input_runs.append(
                    {
                        "run_id": summary["run_id"],
                        "summary_sha256": sha256_file(summary_path),
                        "event_predictions_sha256": sha256_file(event_path),
                    }
                )
            union = np.unique(np.concatenate(union_rows))
            union_hash = identity_hash(identity_matrix(source, union))
            if phase == "development" and union_hash != expected_reservoir_hash:
                raise ValueError("Development context draws do not cover the full reservoir.")
            context_draw_unions.append(
                {
                    "phase": phase,
                    "unique_event_count": int(union.size),
                    "identity_sha256": union_hash,
                    "reservoir_count": int(reservoir_rows[phase].size),
                    "covers_full_reservoir": bool(
                        union.size == reservoir_rows[phase].size
                        and np.array_equal(np.sort(union), np.sort(reservoir_rows[phase]))
                    ),
                }
            )

    np.savez(local_output, **archive)
    neural_methods = [
        {
            "method": "CNP",
            "config": "ml4phy-paper/configs/m0_cnp_seed0.yaml",
            "training_seeds": [0, 1, 2],
            "checkpoint_status": "missing matched checkpoints; three jobs authorized",
        },
        {
            "method": "Attentive CNP",
            "config": "ml4phy-paper/configs/m1_attentive_cnp_seed0.yaml",
            "training_seeds": [0, 1, 2],
            "checkpoint_status": "missing matched checkpoints; three jobs authorized",
            "model_semantics": "deterministic cross-attention, not a latent ANP",
        },
        {
            "method": "Attentive CNP + PE",
            "config": "ml4phy-paper/configs/m2_attentive_pe10_seed0.yaml",
            "training_seeds": [0, 1, 2],
            "checkpoint_status": "reuse three controlled checkpoints",
        },
        {
            "method": "Density-guided CNP (ours)",
            "config": "ml4phy-paper/configs/cell17_controlled_seed0.yaml",
            "training_seeds": [0, 1, 2],
            "checkpoint_status": "reuse three controlled checkpoints",
        },
    ]
    for method in neural_methods:
        method["config_sha256"] = sha256_file(repo / method["config"])
    manifest = {
        "schema_version": 1,
        "status": "frozen_before_extension_training_or_inference",
        "source_commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "script": "ml4phy-paper/scripts/build_extension_protocol.py",
        "script_sha256": sha256_file(Path(__file__)),
        "original_result_commit": "48a59d0b2d93a6c76d2001522abcb98ac74432fb",
        "original_protocol": {
            "path": "ml4phy-paper/manifests/frozen_protocol_v1.json",
            "sha256": sha256_file(original_protocol_path),
            "preserved_unchanged": True,
        },
        "analysis_status": (
            "Prospectively specified follow-up analysis on historically exposed data; "
            "not an untouched test."
        ),
        "context_protocol": {
            "sizes": list(CONTEXT_SIZES),
            "context_seeds": list(CONTEXT_SEEDS),
            "construction": (
                "SHA-256-priority, outcome-blind deterministic permutation of each original "
                "2,000-event context membership; smaller contexts are nested prefixes."
            ),
            "membership_at_2000": "identical to the original executed context draw",
            "dropout_seed": 10100,
            "dropout_amendment": {
                "original_rule": "10000 + context_seed",
                "executed_rule": "fixed 10100",
                "decision_date": "2026-09-08",
                "reason": "isolate context-selection variation from dropout-MC variation",
            },
            "mc_passes": 50,
            "subsets": subset_manifest,
            "original_draw_unions": context_draw_unions,
            "local_archive": {
                "logical_path": str(args.local_output),
                "sha256": sha256_file(local_output),
                "git_policy": "ignored; contains source-HDF5 row indices",
            },
        },
        "neural_methods": neural_methods,
        "kernel": {
            "estimator": "Nadaraya-Watson / common-bandwidth KDE ratio with pass fraction",
            "bandwidth_candidates_kev": [2, 5, 10, 20, 50, 100],
            "selection_data": "development only, separately for each context size",
            "final_target_selection_forbidden": True,
            "variants": ["context-only", "acceptance-training-pool plus context"],
        },
        "gaussian_process": {
            "scikit_learn_version": sklearn.__version__,
            "likelihood": "Bernoulli-logistic classification",
            "approximation": "scikit-learn Laplace",
            "dtype": "float64",
            "covariances": ["ConstantKernel * RBF", "ConstantKernel * Matern(nu=1.5)"],
            "initial_length_scale_kev": 50,
            "length_scale_bounds_kev": [1, 1000],
            "initial_amplitude": 1,
            "amplitude_bounds": [0.01, 100],
            "optimizer_restarts": 2,
            "optimizer_max_iterations": 200,
            "laplace_max_iterations": 100,
            "family_selection": (
                "lowest mean global development Brier; equal-region tie-breaker within 1e-6; "
                "then prefer RBF"
            ),
            "pilot_gate": (
                "one full 2,000-context development fit plus event-level target prediction; "
                "stop before campaign if projected total exceeds two hours"
            ),
        },
        "phase2_gate": (
            "No acceptance-training-size campaign is authorized until Phase 1 results and a "
            "measured Phase 2 cost estimate are returned for approval."
        ),
        "inputs": input_runs,
    }
    with manifest_output.open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
