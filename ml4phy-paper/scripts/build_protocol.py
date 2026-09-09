#!/usr/bin/env python3
"""Freeze paper data roles and an independently calibrated classifier threshold."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

IDENTITY_FIELDS = ("run_number", "detector", "id", "tp0")
ENERGY_RANGE_KEV = (500.0, 3000.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_matrix(handle: h5py.File) -> np.ndarray:
    return np.column_stack(
        [handle[field][:].astype("<i8", copy=False) for field in IDENTITY_FIELDS]
    )


def identity_hash(identities: np.ndarray) -> str:
    ordered = identities[
        np.lexsort(tuple(identities[:, index] for index in reversed(range(identities.shape[1]))))
    ]
    return hashlib.sha256(np.ascontiguousarray(ordered, dtype="<i8").tobytes()).hexdigest()


def identity_tuples(identities: np.ndarray) -> list[tuple[int, ...]]:
    return [tuple(int(value) for value in row) for row in identities]


def stratified_sample(labels: np.ndarray, count: int, seed: int) -> np.ndarray:
    if count <= 0 or count >= labels.size:
        raise ValueError("Stratified sample count must be between zero and the population size.")
    classes, class_counts = np.unique(labels, return_counts=True)
    exact = count * class_counts / labels.size
    quotas = np.floor(exact).astype(int)
    remainder = count - int(quotas.sum())
    fractional_order = np.argsort(-(exact - quotas), kind="stable")
    quotas[fractional_order[:remainder]] += 1
    rng = np.random.default_rng(seed)
    selected = []
    for label, quota in zip(classes, quotas, strict=True):
        candidates = np.flatnonzero(labels == label)
        selected.append(rng.choice(candidates, size=int(quota), replace=False))
    return np.sort(np.concatenate(selected))


def role_summary(
    rows: np.ndarray,
    identities: np.ndarray,
    energies: np.ndarray,
    *,
    labels: np.ndarray | None = None,
) -> dict:
    role_energies = energies[rows]
    summary = {
        "count": int(rows.size),
        "identity_sha256": identity_hash(identities[rows]),
        "energy_min_kev": float(role_energies.min()),
        "energy_max_kev": float(role_energies.max()),
    }
    if labels is not None:
        role_labels = labels[rows]
        summary.update(positive=int(role_labels.sum()), negative=int(rows.size - role_labels.sum()))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument(
        "--development-h5",
        type=Path,
        default=Path("runs/small_data_configs/simple_cnn_small/eval/predictions.h5"),
    )
    parser.add_argument(
        "--full-test-h5",
        type=Path,
        default=Path("runs/small_data_configs/simple_cnn_small/eval_full_test/predictions.h5"),
    )
    parser.add_argument(
        "--local-output",
        type=Path,
        default=Path("ml4phy-paper/local/protocol/frozen_roles_v1.npz"),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("ml4phy-paper/manifests/frozen_protocol_v1.json"),
    )
    parser.add_argument("--split-seed", type=int, default=20260908)
    args = parser.parse_args()

    repo = args.repo.resolve()
    development_path = (repo / args.development_h5).resolve()
    full_test_path = (repo / args.full_test_h5).resolve()
    local_output = (repo / args.local_output).resolve()
    manifest_output = (repo / args.manifest_output).resolve()
    for output in (local_output, manifest_output):
        if output.exists():
            parser.error(f"Refusing to overwrite existing output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    with (
        h5py.File(development_path, "r") as development,
        h5py.File(full_test_path, "r") as full_test,
    ):
        for field in (*IDENTITY_FIELDS, "energy", "score", "label"):
            if field not in development or field not in full_test:
                parser.error(f"Required field {field!r} is missing from an input HDF5 file.")
        development_ids = identity_matrix(development)
        full_ids = identity_matrix(full_test)
        development_labels = development["label"][:].astype(np.int8)
        development_scores = development["score"][:].astype(np.float64)
        development_energies = development["energy"][:].astype(np.float64)
        full_energies = full_test["energy"][:].astype(np.float64)

    if len(set(identity_tuples(development_ids))) != development_ids.shape[0]:
        parser.error("Development identities are not unique.")
    if len(set(identity_tuples(full_ids))) != full_ids.shape[0]:
        parser.error("Full-test identities are not unique.")
    development_id_set = set(identity_tuples(development_ids))
    full_id_to_row = {identity: row for row, identity in enumerate(identity_tuples(full_ids))}
    if not development_id_set <= set(full_id_to_row):
        parser.error("Development identities are not a subset of full-test identities.")
    if not (
        np.all(
            (development_energies >= ENERGY_RANGE_KEV[0])
            & (development_energies <= ENERGY_RANGE_KEV[1])
        )
        and np.all((full_energies >= ENERGY_RANGE_KEV[0]) & (full_energies <= ENERGY_RANGE_KEV[1]))
    ):
        parser.error("Input exports contain rows outside the frozen energy range.")

    calibration_rows = stratified_sample(development_labels, count=2000, seed=args.split_seed)
    development_remaining = np.setdiff1d(np.arange(development_ids.shape[0]), calibration_rows)
    development_rng = np.random.default_rng(args.split_seed + 1)
    development_permutation = development_rng.permutation(development_remaining)
    development_context_rows = np.sort(development_permutation[:3000])
    development_target_rows = np.sort(development_permutation[3000:])

    development_rows_in_full = np.array(
        [full_id_to_row[identity] for identity in identity_tuples(development_ids)], dtype=np.int64
    )
    final_candidates = np.setdiff1d(np.arange(full_ids.shape[0]), development_rows_in_full)
    final_rng = np.random.default_rng(args.split_seed + 2)
    final_permutation = final_rng.permutation(final_candidates)
    final_context_rows = np.sort(final_permutation[:20000])
    final_target_rows = np.sort(final_permutation[20000:])

    if (
        np.union1d(
            np.union1d(calibration_rows, development_context_rows), development_target_rows
        ).size
        != development_ids.shape[0]
    ):
        parser.error("Development roles do not form a complete disjoint partition.")
    if np.intersect1d(calibration_rows, development_context_rows).size:
        parser.error("Calibration and development context roles overlap.")
    if np.intersect1d(calibration_rows, development_target_rows).size:
        parser.error("Calibration and development target roles overlap.")
    if np.intersect1d(development_context_rows, development_target_rows).size:
        parser.error("Development context and target roles overlap.")
    if np.union1d(final_context_rows, final_target_rows).size != final_candidates.size:
        parser.error("Final roles do not form a complete disjoint partition.")
    if np.intersect1d(final_context_rows, final_target_rows).size:
        parser.error("Final context and target roles overlap.")

    fpr, tpr, thresholds = roc_curve(
        development_labels[calibration_rows], development_scores[calibration_rows]
    )
    threshold_index = int(np.argmax(tpr - fpr))
    threshold = float(thresholds[threshold_index])

    np.savez(
        local_output,
        calibration_development_rows=calibration_rows,
        development_context_reservoir_rows=development_context_rows,
        development_target_rows=development_target_rows,
        final_context_reservoir_rows=final_context_rows,
        final_target_rows=final_target_rows,
        split_seed=np.int64(args.split_seed),
        fixed_threshold=np.float64(threshold),
    )

    roles = {
        "threshold_calibration": role_summary(
            calibration_rows,
            development_ids,
            development_energies,
            labels=development_labels,
        ),
        "development_context_reservoir": role_summary(
            development_context_rows, development_ids, development_energies
        ),
        "development_target": role_summary(
            development_target_rows, development_ids, development_energies
        ),
        "final_context_reservoir": role_summary(final_context_rows, full_ids, full_energies),
        "final_target": role_summary(final_target_rows, full_ids, full_energies),
    }
    manifest = {
        "schema_version": 1,
        "status": "frozen_before_paper_model_inference",
        "source_commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "script_sha256": sha256_file(Path(__file__)),
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python ml4phy-paper/scripts/build_protocol.py --repo .",
        "identity_fields": IDENTITY_FIELDS,
        "energy_range_kev": ENERGY_RANGE_KEV,
        "split_seed": args.split_seed,
        "inputs": {
            "historical_development": {
                "logical_path": str(args.development_h5),
                "bytes": development_path.stat().st_size,
                "sha256": sha256_file(development_path),
                "count": int(development_ids.shape[0]),
                "identity_sha256": identity_hash(development_ids),
            },
            "full_test": {
                "logical_path": str(args.full_test_h5),
                "bytes": full_test_path.stat().st_size,
                "sha256": sha256_file(full_test_path),
                "count": int(full_ids.shape[0]),
                "identity_sha256": identity_hash(full_ids),
            },
        },
        "subset_checks": {
            "development_is_exact_identity_subset_of_full_test": True,
            "full_test_minus_development_count": int(final_candidates.size),
            "all_roles_pairwise_disjoint_within_each_evaluation_stage": True,
        },
        "roles": roles,
        "threshold": {
            "value": threshold,
            "method": "Youden-J maximum on threshold_calibration only",
            "calibration_count": int(calibration_rows.size),
            "calibration_identity_sha256": roles["threshold_calibration"]["identity_sha256"],
            "roc_auc": float(
                roc_auc_score(
                    development_labels[calibration_rows], development_scores[calibration_rows]
                )
            ),
            "target_outcomes_used": False,
        },
        "context_protocol": {
            "primary_context_count": 2000,
            "paired_context_seeds": list(range(100, 110)),
            "mc_passes_after_pilot": 50,
            "dropout_seed_rule": "10000 + context_seed",
        },
        "local_role_archive": {
            "logical_path": str(args.local_output),
            "sha256": sha256_file(local_output),
            "git_policy": "ignored; contains HDF5 row indices",
        },
        "exposure_statement": "The final target was historically inspected, including for the talk. This is a prospectively frozen repeated-split evaluation, not an untouched test.",
    }
    with manifest_output.open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        f"Frozen threshold {threshold:.15f}; development target {development_target_rows.size}; "
        f"final target {final_target_rows.size}."
    )


if __name__ == "__main__":
    main()
