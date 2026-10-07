#!/usr/bin/env python3
"""Run the required dense Bernoulli-GP timing pilot before any GP campaign."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import minimize
from sklearn import __version__ as sklearn_version
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

IDENTITY_FIELDS = ("run_number", "detector", "id", "tp0")
REGIONS = {
    "DEP": (1577.0, 1606.0),
    "Bi-212": (1606.0, 1635.0),
    "continuum_1700_2000": (1700.0, 2000.0),
    "SE": (2088.0, 2118.0),
    "continuum_2200_2400": (2200.0, 2400.0),
    "FE": (2599.0, 2629.0),
    "sparse_tail": (2700.0, 3000.0),
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


def main() -> None:
    repo = Path(".").resolve()
    worktree = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--short"], text=True
    ).splitlines()
    if worktree:
        raise RuntimeError("GP pilot requires a clean worktree: " + "; ".join(worktree))
    source_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    config_path = repo / "ml4phy-paper/configs/gp_protocol_v1.json"
    extension_path = repo / "ml4phy-paper/manifests/extension_protocol_v1.json"
    base_path = repo / "ml4phy-paper/manifests/frozen_protocol_v1.json"
    context_archive_path = repo / "ml4phy-paper/local/protocol/extension_context_roles_v1.npz"
    base_archive_path = repo / "ml4phy-paper/local/protocol/frozen_roles_v1.npz"
    output_dir = repo / "ml4phy-paper/runs/gp/20260909-rbf-dev-s100-n2000-pilot-attempt2"
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "summary.json"
    predictions_path = output_dir / "predictions.npz"
    config = read_json(config_path)
    extension = read_json(extension_path)
    base = read_json(base_path)
    if config["status"] != "frozen_before_gp_pilot" or config["pilot"] != {
        "covariance_family": "RBF",
        "phase": "development",
        "context_seed": 100,
        "context_size": 2000,
        "optimizer_random_seed": 31000,
        "include_development_target_prediction": True,
        "include_final_target_prediction_timing": True,
    }:
        raise ValueError("Unexpected GP pilot configuration")
    if sklearn_version != extension["gaussian_process"]["scikit_learn_version"]:
        raise ValueError("Installed scikit-learn version differs from the frozen protocol")
    if sha256_file(context_archive_path) != extension["context_protocol"]["local_archive"][
        "sha256"
    ]:
        raise ValueError("Extension context archive hash mismatch")
    if sha256_file(base_archive_path) != base["local_role_archive"]["sha256"]:
        raise ValueError("Base role archive hash mismatch")

    with np.load(context_archive_path) as archive:
        context_rows = archive["development_context_s100_n2000_rows"].astype(np.int64)
    with np.load(base_archive_path) as archive:
        development_target_rows = archive["development_target_rows"].astype(np.int64)
        final_target_rows = archive["final_target_rows"].astype(np.int64)
        threshold = float(archive["fixed_threshold"])
    development_h5 = repo / base["inputs"]["historical_development"]["logical_path"]
    final_h5 = repo / base["inputs"]["full_test"]["logical_path"]
    if sha256_file(development_h5) != base["inputs"]["historical_development"]["sha256"]:
        raise ValueError("Development HDF5 hash mismatch")
    if sha256_file(final_h5) != base["inputs"]["full_test"]["sha256"]:
        raise ValueError("Final HDF5 hash mismatch")
    with h5py.File(development_h5, "r") as handle:
        sorted_context_rows = np.sort(context_rows)
        context_energy = handle["energy"][sorted_context_rows].astype(np.float64)
        context_score = handle["score"][sorted_context_rows].astype(np.float64)
        context_ids = identity_matrix(handle, sorted_context_rows)
        development_energy = handle["energy"][development_target_rows].astype(np.float64)
        development_score = handle["score"][development_target_rows].astype(np.float64)
    with h5py.File(final_h5, "r") as handle:
        final_energy = handle["energy"][final_target_rows].astype(np.float64)
    expected_context_hash = next(
        item["identity_sha256"]
        for item in extension["context_protocol"]["subsets"]
        if item["phase"] == "development"
        and item["context_seed"] == 100
        and item["context_size"] == 2000
    )
    if identity_hash(context_ids) != expected_context_hash:
        raise ValueError("Pilot context identity mismatch")
    context_outcome = (context_score >= threshold).astype(np.int64)
    development_outcome = (development_score >= threshold).astype(np.float64)

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
    kernel = ConstantKernel(1.0, (0.01, 100.0)) * RBF(length_scale, length_bounds)
    classifier = GaussianProcessClassifier(
        kernel=kernel,
        optimizer=bounded_optimizer,
        n_restarts_optimizer=2,
        max_iter_predict=100,
        warm_start=False,
        copy_X_train=True,
        random_state=31000,
        n_jobs=None,
    )
    start_time = utc_now()
    total_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_start = time.perf_counter()
        classifier.fit(normalized_energy(context_energy), context_outcome)
        fit_seconds = time.perf_counter() - fit_start
        development_start = time.perf_counter()
        development_probability = classifier.predict_proba(
            normalized_energy(development_energy)
        )[:, 1]
        development_prediction_seconds = time.perf_counter() - development_start
        final_start = time.perf_counter()
        final_probability = classifier.predict_proba(normalized_energy(final_energy))[:, 1]
        final_prediction_seconds = time.perf_counter() - final_start
    warning_records = [
        {
            "category": item.category.__name__,
            "message": str(item.message),
        }
        for item in caught
    ]
    np.savez(
        predictions_path,
        development_target_rows=development_target_rows,
        development_probability=development_probability,
        final_target_rows=final_target_rows,
        final_probability=final_probability,
    )
    regional_brier = {}
    for name, (lower, upper) in REGIONS.items():
        mask = (development_energy >= lower) & (development_energy < upper)
        regional_brier[name] = {
            "events": int(mask.sum()),
            "brier": float(
                np.mean((development_probability[mask] - development_outcome[mask]) ** 2)
            )
            if mask.any()
            else None,
        }
    supported_brier = [
        item["brier"]
        for item in regional_brier.values()
        if item["events"] >= 20 and item["brier"] is not None
    ]
    optimized_length_scale_kev = float(classifier.kernel_.k2.length_scale * 2500.0)
    summary = {
        "schema_version": 1,
        "status": "completed",
        "run_id": output_dir.name,
        "start_time": start_time,
        "end_time": utc_now(),
        "source_commit": source_commit,
        "source_worktree_status": worktree,
        "script_sha256": sha256_file(Path(__file__)),
        "config": "ml4phy-paper/configs/gp_protocol_v1.json",
        "config_sha256": sha256_file(config_path),
        "scikit_learn_version": sklearn_version,
        "context": {
            "phase": "development",
            "seed": 100,
            "size": 2000,
            "identity_sha256": expected_context_hash,
            "passing": int(context_outcome.sum()),
        },
        "threshold": threshold,
        "estimator": {
            "family": "ConstantKernel * RBF",
            "likelihood": "Bernoulli-logistic",
            "approximation": "Laplace",
            "prediction": "posterior-integrated predict_proba",
            "optimizer_random_seed": 31000,
            "optimizer_restarts": 2,
            "optimizer_max_iterations": 200,
            "laplace_max_iterations": 100,
            "optimized_kernel": str(classifier.kernel_),
            "optimized_amplitude": float(classifier.kernel_.k1.constant_value),
            "optimized_length_scale_kev": optimized_length_scale_kev,
            "log_marginal_likelihood": float(classifier.log_marginal_likelihood_value_),
            "optimization_records": optimization_records,
        },
        "metrics": {
            "development_global_brier": float(
                np.mean((development_probability - development_outcome) ** 2)
            ),
            "development_equal_supported_region_brier": float(np.mean(supported_brier)),
            "development_regions": regional_brier,
        },
        "runtime": {
            "fit_seconds": fit_seconds,
            "development_target_events": int(development_energy.size),
            "development_prediction_seconds": development_prediction_seconds,
            "final_target_events": int(final_energy.size),
            "final_prediction_seconds": final_prediction_seconds,
            "total_seconds": time.perf_counter() - total_start,
        },
        "warnings": warning_records,
        "server_only_predictions": {
            "path": str(predictions_path.relative_to(repo)),
            "bytes": predictions_path.stat().st_size,
            "sha256": sha256_file(predictions_path),
        },
        "historical_data_exposure": extension["analysis_status"],
    }
    write_json(summary_path, summary)
    print(json.dumps(summary["runtime"]))


if __name__ == "__main__":
    main()
