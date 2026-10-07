#!/usr/bin/env python3
"""Fit and evaluate one cell of the frozen dense Bernoulli-GP campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import minimize
from sklearn import __version__ as sklearn_version
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern

IDENTITY_FIELDS = ("run_number", "detector", "id", "tp0")
CONTEXT_SIZES = (250, 500, 1000, 2000)
CONTEXT_SEEDS = tuple(range(100, 110))
FAMILIES = ("rbf", "matern32")


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


def normalized_energy(energy_kev: np.ndarray) -> np.ndarray:
    return ((energy_kev.astype(np.float64) - 500.0) / 2500.0)[:, None]


def optimizer_seed(context_size: int, context_seed: int) -> int:
    return 31000 + 100 * CONTEXT_SIZES.index(context_size) + context_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--phase", choices=("development", "final"), required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--context-size", type=int, choices=CONTEXT_SIZES, required=True)
    parser.add_argument("--context-seed", type=int, choices=CONTEXT_SEEDS, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id.replace("-", "").replace("_", "").isalnum():
        parser.error("run-id may contain only letters, numbers, hyphens, and underscores")

    repo = args.repo.resolve()
    root = repo / "ml4phy-paper"
    output_dir = root / "runs/gp/phase3" / args.run_id
    if output_dir.exists():
        parser.error(f"Refusing to reuse output directory: {output_dir}")
    worktree = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--short"], text=True
    ).splitlines()
    if worktree:
        parser.error("Dense GP cell requires a clean worktree: " + "; ".join(worktree))
    source_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    config_path = root / "configs/gp_protocol_v1.json"
    extension_path = root / "manifests/extension_protocol_v1.json"
    base_path = root / "manifests/frozen_protocol_v1.json"
    context_archive_path = root / "local/protocol/extension_context_roles_v1.npz"
    base_archive_path = root / "local/protocol/frozen_roles_v1.npz"
    config = read_json(config_path)
    extension = read_json(extension_path)
    base = read_json(base_path)
    if config["status"] != "frozen_before_gp_pilot":
        raise ValueError("Dense GP protocol is not frozen")
    if sklearn_version != extension["gaussian_process"]["scikit_learn_version"]:
        raise ValueError("Installed scikit-learn version differs from frozen protocol")
    if (
        sha256_file(context_archive_path)
        != extension["context_protocol"]["local_archive"]["sha256"]
    ):
        raise ValueError("Extension context archive hash mismatch")
    if sha256_file(base_archive_path) != base["local_role_archive"]["sha256"]:
        raise ValueError("Base role archive hash mismatch")

    subset_key = f"{args.phase}_context_s{args.context_seed}_n{args.context_size}_rows"
    with np.load(context_archive_path) as archive:
        context_rows = archive[subset_key].astype(np.int64)
    with np.load(base_archive_path) as archive:
        target_rows = archive[f"{args.phase}_target_rows"].astype(np.int64)
        threshold = float(archive["fixed_threshold"])
    expected_context_hash = next(
        item["identity_sha256"]
        for item in extension["context_protocol"]["subsets"]
        if item["phase"] == args.phase
        and item["context_seed"] == args.context_seed
        and item["context_size"] == args.context_size
    )
    input_key = "historical_development" if args.phase == "development" else "full_test"
    h5_path = repo / base["inputs"][input_key]["logical_path"]
    if sha256_file(h5_path) != base["inputs"][input_key]["sha256"]:
        raise ValueError("Phase HDF5 hash mismatch")
    with h5py.File(h5_path, "r") as handle:
        sorted_context_rows = np.sort(context_rows)
        sorted_target_rows = np.sort(target_rows)
        context_ids = identity_matrix(handle, sorted_context_rows)
        target_ids = identity_matrix(handle, sorted_target_rows)
        context_energy = handle["energy"][sorted_context_rows].astype(np.float64)
        context_score = handle["score"][sorted_context_rows].astype(np.float64)
        target_energy = handle["energy"][sorted_target_rows].astype(np.float64)
        target_score = handle["score"][sorted_target_rows].astype(np.float64)
    if identity_hash(context_ids) != expected_context_hash:
        raise ValueError("Context identity hash mismatch")
    expected_target_hash = base["roles"][f"{args.phase}_target"]["identity_sha256"]
    if identity_hash(target_ids) != expected_target_hash:
        raise ValueError("Target identity hash mismatch")
    if threshold != 0.540643572807312:
        raise ValueError("Frozen threshold mismatch")

    output_dir.mkdir(parents=True, exist_ok=False)
    state_path = output_dir / "runner_state.json"
    prediction_path = output_dir / "predictions.npz"
    curve_path = output_dir / "curve.npz"
    summary_path = output_dir / "summary.json"
    seed = optimizer_seed(args.context_size, args.context_seed)
    state = {
        "schema_version": 1,
        "status": "running",
        "run_id": args.run_id,
        "start_time": utc_now(),
        "source_commit": source_commit,
        "script_sha256": sha256_file(Path(__file__)),
        "phase": args.phase,
        "family": args.family,
        "context_size": args.context_size,
        "context_seed": args.context_seed,
        "optimizer_seed": seed,
    }
    write_json(state_path, state)
    total_start = time.perf_counter()
    try:
        context_outcome = (context_score >= threshold).astype(np.int64)
        target_outcome = (target_score >= threshold).astype(np.float64)
        optimization_records: list[dict] = []

        def bounded_optimizer(objective, initial_theta, bounds):
            start = time.perf_counter()
            result = minimize(
                objective,
                initial_theta,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": 200},
            )
            optimization_records.append(
                {
                    "success": bool(result.success),
                    "status": int(result.status),
                    "message": str(result.message),
                    "iterations": int(result.nit),
                    "function_evaluations": int(result.nfev),
                    "objective": float(result.fun),
                    "wall_seconds": time.perf_counter() - start,
                }
            )
            return result.x, result.fun

        length_scale = 50.0 / 2500.0
        length_bounds = (1.0 / 2500.0, 1000.0 / 2500.0)
        base_kernel = (
            RBF(length_scale, length_bounds)
            if args.family == "rbf"
            else Matern(length_scale, length_bounds, nu=1.5)
        )
        kernel = ConstantKernel(1.0, (0.01, 100.0)) * base_kernel
        classifier = GaussianProcessClassifier(
            kernel=kernel,
            optimizer=bounded_optimizer,
            n_restarts_optimizer=2,
            max_iter_predict=100,
            warm_start=False,
            copy_X_train=True,
            random_state=seed,
            n_jobs=None,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit_start = time.perf_counter()
            classifier.fit(normalized_energy(context_energy), context_outcome)
            fit_seconds = time.perf_counter() - fit_start
            prediction_start = time.perf_counter()
            target_probability = classifier.predict_proba(normalized_energy(target_energy))[:, 1]
            target_prediction_seconds = time.perf_counter() - prediction_start
            grid_energy = np.arange(500.0, 3000.0 + 0.5, 1.0)
            grid_start = time.perf_counter()
            grid_probability = classifier.predict_proba(normalized_energy(grid_energy))[:, 1]
            grid_prediction_seconds = time.perf_counter() - grid_start
        warning_records = [
            {"category": item.category.__name__, "message": str(item.message)} for item in caught
        ]

        sys.path.insert(0, str(root / "scripts"))
        from evaluate_fixed_protocol import (  # noqa: PLC0415
            bin_metrics,
            event_metrics,
            grid_roughness,
            peak_contrasts,
        )

        event_rows = event_metrics(target_energy, target_probability, target_outcome)
        bin_rows, binned = bin_metrics(target_energy, target_probability, target_outcome)
        contrast_rows = peak_contrasts(target_energy, target_probability, target_outcome)
        roughness_rows = grid_roughness(grid_energy, grid_probability)
        np.savez(
            prediction_path,
            target_rows=sorted_target_rows,
            target_energy_kev=target_energy,
            outcome=target_outcome.astype(np.int8),
            prediction=target_probability,
            context_rows=sorted_context_rows,
        )
        np.savez(
            curve_path,
            energy_kev=grid_energy,
            prediction=grid_probability,
            bin_centers_kev=binned["centers_kev"],
            bin_counts=binned["counts"],
            empirical_rate=binned["empirical_rate"],
            bin_prediction_mean=binned["prediction_mean"],
        )
        summary = {
            "schema_version": 1,
            "status": "completed",
            "run_id": args.run_id,
            "start_time": state["start_time"],
            "end_time": utc_now(),
            "source_commit": source_commit,
            "source_worktree_status": worktree,
            "script_sha256": sha256_file(Path(__file__)),
            "metric_script": "ml4phy-paper/scripts/evaluate_fixed_protocol.py",
            "metric_script_sha256": sha256_file(root / "scripts/evaluate_fixed_protocol.py"),
            "config": "ml4phy-paper/configs/gp_protocol_v1.json",
            "config_sha256": sha256_file(config_path),
            "scikit_learn_version": sklearn_version,
            "phase": args.phase,
            "context": {
                "seed": args.context_seed,
                "size": args.context_size,
                "identity_sha256": expected_context_hash,
                "passing": int(context_outcome.sum()),
            },
            "target": {
                "events": int(target_energy.size),
                "identity_sha256": expected_target_hash,
            },
            "threshold": threshold,
            "estimator": {
                "family_id": args.family,
                "family": "ConstantKernel * RBF"
                if args.family == "rbf"
                else "ConstantKernel * Matern(nu=1.5)",
                "likelihood": "Bernoulli-logistic",
                "approximation": "Laplace",
                "prediction": "posterior-integrated predict_proba",
                "dtype": "float64",
                "optimizer_random_seed": seed,
                "optimizer_restarts": 2,
                "optimizer_max_iterations": 200,
                "laplace_max_iterations": 100,
                "optimized_kernel": str(classifier.kernel_),
                "optimized_amplitude": float(classifier.kernel_.k1.constant_value),
                "optimized_length_scale_kev": float(classifier.kernel_.k2.length_scale * 2500.0),
                "log_marginal_likelihood": float(classifier.log_marginal_likelihood_value_),
                "optimization_records": optimization_records,
            },
            "metrics": {
                "event": event_rows,
                "bin_5kev": bin_rows,
                "peak_sideband_contrast": contrast_rows,
                "roughness": roughness_rows,
            },
            "runtime": {
                "fit_seconds": fit_seconds,
                "target_prediction_seconds": target_prediction_seconds,
                "grid_prediction_seconds": grid_prediction_seconds,
                "total_seconds": time.perf_counter() - total_start,
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            },
            "warnings": warning_records,
            "server_only_outputs": {
                "predictions": {
                    "file": prediction_path.name,
                    "bytes": prediction_path.stat().st_size,
                    "sha256": sha256_file(prediction_path),
                },
                "curve": {
                    "file": curve_path.name,
                    "bytes": curve_path.stat().st_size,
                    "sha256": sha256_file(curve_path),
                },
            },
            "historical_data_exposure": extension["analysis_status"],
        }
        write_json(summary_path, summary)
        state.update(
            {
                "status": "completed",
                "end_time": summary["end_time"],
                "summary_sha256": sha256_file(summary_path),
            }
        )
        write_json(state_path, state)
        print(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "status": "completed",
                    "fit_seconds": fit_seconds,
                    "total_seconds": summary["runtime"]["total_seconds"],
                    "global_brier": event_rows[0]["brier"],
                }
            ),
            flush=True,
        )
    except Exception as error:
        state.update(
            {
                "status": "failed",
                "end_time": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "wall_seconds": time.perf_counter() - total_start,
            }
        )
        write_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
