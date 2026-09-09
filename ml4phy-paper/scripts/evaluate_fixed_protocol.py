#!/usr/bin/env python3
"""Evaluate one recovered CNP with fixed roles, threshold, and randomness."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np

IDENTITY_FIELDS = ("run_number", "detector", "id", "tp0")
REGIONS = {
    "full": (500.0, 3000.0),
    "DEP": (1577.0, 1606.0),
    "Bi-214": (1606.0, 1635.0),
    "continuum_1700_2000": (1700.0, 2000.0),
    "SE": (2088.0, 2118.0),
    "continuum_2200_2400": (2200.0, 2400.0),
    "FE": (2599.0, 2629.0),
    "sparse_tail": (2700.0, 3000.0),
}
PEAKS = {
    "DEP": (1592.0, 1577.0, 1606.0),
    "Bi-214": (1620.0, 1606.0, 1635.0),
    "SE": (2103.0, 2088.0, 2118.0),
    "FE": (2614.0, 2599.0, 2629.0),
}
PROBABILITY_CLIP = 1.0e-6
MIN_REGION_EVENTS = 20
MIN_BIN_EVENTS = 4


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


def verify_file(path: Path, expected_sha256: str, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ValueError(
            f"{description} hash mismatch: expected {expected_sha256}, observed {observed}"
        )


def identity_matrix(handle: h5py.File, rows: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [handle[field][rows].astype("<i8", copy=False) for field in IDENTITY_FIELDS]
    )


def identity_hash(identities: np.ndarray) -> str:
    ordered = identities[
        np.lexsort(tuple(identities[:, index] for index in reversed(range(identities.shape[1]))))
    ]
    return hashlib.sha256(np.ascontiguousarray(ordered, dtype="<i8").tobytes()).hexdigest()


def region_mask(energy: np.ndarray, bounds: tuple[float, float], *, full: bool = False):
    lower, upper = bounds
    return (energy >= lower) & (energy <= upper if full else energy < upper)


def predictive_metrics(prediction: np.ndarray, outcome: np.ndarray) -> dict[str, float | None]:
    if outcome.size == 0:
        return {"brier": None, "log_loss": None}
    clipped = np.clip(prediction, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
    return {
        "brier": float(np.mean((prediction - outcome) ** 2)),
        "log_loss": float(
            -np.mean(outcome * np.log(clipped) + (1.0 - outcome) * np.log(1.0 - clipped))
        ),
    }


def event_metrics(energy: np.ndarray, prediction: np.ndarray, outcome: np.ndarray) -> list[dict]:
    rows = []
    region_brier = []
    for region, bounds in REGIONS.items():
        mask = region_mask(energy, bounds, full=region == "full")
        status = (
            "supported"
            if region == "full" or int(mask.sum()) >= MIN_REGION_EVENTS
            else "inconclusive"
        )
        metrics = predictive_metrics(prediction[mask], outcome[mask])
        if region != "full" and status == "supported" and metrics["brier"] is not None:
            region_brier.append(metrics["brier"])
        rows.append(
            {
                "region": region,
                "status": status,
                "n_events": int(mask.sum()),
                **metrics,
            }
        )
    rows.append(
        {
            "region": "equal_region_mean",
            "status": "supported_regions_only",
            "n_events": None,
            "brier": float(np.mean(region_brier)),
            "log_loss": None,
        }
    )
    return rows


def bin_metrics(
    energy: np.ndarray, prediction: np.ndarray, outcome: np.ndarray, bin_width_kev: float = 5.0
) -> tuple[list[dict], dict[str, np.ndarray]]:
    edges = np.arange(500.0, 3000.0 + bin_width_kev, bin_width_kev)
    centers = 0.5 * (edges[:-1] + edges[1:])
    index = np.searchsorted(edges, energy, side="right") - 1
    valid_event = (index >= 0) & (index < centers.size)
    counts = np.bincount(index[valid_event], minlength=centers.size)
    observed = np.bincount(index[valid_event], weights=outcome[valid_event], minlength=centers.size)
    predicted = np.bincount(
        index[valid_event], weights=prediction[valid_event], minlength=centers.size
    )
    empirical_rate = np.divide(
        observed, counts, out=np.full(centers.size, np.nan), where=counts > 0
    )
    prediction_mean = np.divide(
        predicted, counts, out=np.full(centers.size, np.nan), where=counts > 0
    )
    rows = []
    for region, bounds in REGIONS.items():
        in_region = region_mask(centers, bounds, full=region == "full")
        valid = in_region & (counts >= MIN_BIN_EVENTS)
        residual = prediction_mean[valid] - empirical_rate[valid]
        rows.append(
            {
                "region": region,
                "n_valid_bins": int(valid.sum()),
                "n_excluded_bins": int(in_region.sum() - valid.sum()),
                "n_events_in_valid_bins": int(counts[valid].sum()),
                "mae": float(np.mean(np.abs(residual))) if valid.any() else None,
                "rmse": float(np.sqrt(np.mean(residual**2))) if valid.any() else None,
            }
        )
    return rows, {
        "centers_kev": centers,
        "counts": counts,
        "empirical_rate": empirical_rate,
        "prediction_mean": prediction_mean,
    }


def peak_contrasts(energy: np.ndarray, prediction: np.ndarray, outcome: np.ndarray) -> list[dict]:
    rows = []
    for name, (peak, lower, upper) in PEAKS.items():
        center = (energy >= peak - 5.0) & (energy < peak + 5.0)
        sideband = (
            (energy >= lower) & (energy < upper) & ((energy < peak - 5.0) | (energy >= peak + 5.0))
        )
        if not center.any() or not sideband.any():
            rows.append(
                {
                    "peak": name,
                    "status": "inconclusive",
                    "center_events": int(center.sum()),
                    "sideband_events": int(sideband.sum()),
                }
            )
            continue
        empirical_contrast = float(outcome[center].mean() - outcome[sideband].mean())
        predicted_contrast = float(prediction[center].mean() - prediction[sideband].mean())
        rows.append(
            {
                "peak": name,
                "status": "supported"
                if int(center.sum() + sideband.sum()) >= MIN_REGION_EVENTS
                else "inconclusive",
                "center_events": int(center.sum()),
                "sideband_events": int(sideband.sum()),
                "empirical_contrast": empirical_contrast,
                "predicted_contrast": predicted_contrast,
                "absolute_contrast_error": abs(predicted_contrast - empirical_contrast),
            }
        )
    return rows


def grid_roughness(grid_energy: np.ndarray, grid_prediction: np.ndarray) -> list[dict]:
    spacing = float(np.diff(grid_energy).mean())
    rows = []
    for region in ("continuum_1700_2000", "continuum_2200_2400"):
        mask = region_mask(grid_energy, REGIONS[region])
        rows.append(
            {
                "region": region,
                "grid_spacing_kev": spacing,
                "mean_absolute_second_difference": float(
                    np.mean(np.abs(np.diff(grid_prediction[mask], n=2)))
                ),
            }
        )
    return rows


def validate_dependency_source(source_root: Path, compatibility: dict) -> dict:
    import core
    import schemas

    observed_root = Path(core.__file__).resolve().parent.parent
    if observed_root != source_root.resolve():
        raise RuntimeError(
            f"Imported RESUM_FLEX from {observed_root}, expected source overlay {source_root.resolve()}"
        )
    if Path(schemas.__file__).resolve().parent.parent != observed_root:
        raise RuntimeError("core and schemas resolved from different dependency roots.")
    for relative, expected in compatibility["source_hashes"].items():
        verify_file(source_root / relative, expected, f"RESUM_FLEX source {relative}")
    return {
        "commit": compatibility["commit"],
        "source_hashes": compatibility["source_hashes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--phase", choices=("development", "final"), required=True)
    parser.add_argument("--context-seed", type=int, required=True)
    parser.add_argument("--dropout-seed", type=int)
    parser.add_argument("--n-context", type=int, default=2000)
    parser.add_argument("--n-mc", type=int, default=50)
    parser.add_argument("--max-context-per-pass", type=int, default=2000)
    parser.add_argument("--grid-spacing-kev", type=float, default=1.0)
    parser.add_argument("--run-kind", choices=("pilot", "evaluation"), default="evaluation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-registry",
        type=Path,
        default=Path("ml4phy-paper/configs/recovered_models_v1.json"),
    )
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=Path("ml4phy-paper/manifests/frozen_protocol_v1.json"),
    )
    parser.add_argument(
        "--role-archive",
        type=Path,
        default=Path("ml4phy-paper/local/protocol/frozen_roles_v1.npz"),
    )
    parser.add_argument(
        "--dependency-manifest",
        type=Path,
        default=Path("ml4phy-paper/manifests/resum_flex_compatibility.json"),
    )
    parser.add_argument(
        "--resum-source-root",
        type=Path,
        default=Path("ml4phy-paper/local/resum-flex-edba6a"),
    )
    args = parser.parse_args()
    start_time = utc_now()
    timer_start = time.perf_counter()

    repo = args.repo.resolve()
    output_dir = (repo / args.output_dir).resolve()
    if output_dir.exists():
        parser.error(f"Refusing to reuse output directory: {output_dir}")
    if args.n_context <= 0 or args.n_mc <= 1 or args.grid_spacing_kev <= 0:
        parser.error(
            "n-context must be positive, n-mc must exceed one, and grid spacing must be positive."
        )
    if args.max_context_per_pass < args.n_context:
        parser.error(
            "max-context-per-pass must be at least n-context so dropout variation is not mixed with context subsampling."
        )

    registry_path = (repo / args.model_registry).resolve()
    protocol_path = (repo / args.protocol_manifest).resolve()
    role_archive_path = (repo / args.role_archive).resolve()
    dependency_path = (repo / args.dependency_manifest).resolve()
    source_root = (repo / args.resum_source_root).resolve()
    registry = read_json(registry_path)
    protocol = read_json(protocol_path)
    compatibility = read_json(dependency_path)
    if args.model_id not in registry["models"]:
        parser.error(f"Unknown model ID {args.model_id!r}")
    model_spec = registry["models"][args.model_id]
    config_path = repo / model_spec["config"]
    checkpoint_path = repo / model_spec["checkpoint"]
    training_pool_path = checkpoint_path.parent / "training_pool.npz"
    training_predictions_path = repo / registry["training_predictions"]
    classifier_config_path = repo / registry["classifier_config"]
    verify_file(config_path, model_spec["config_sha256"], "model configuration")
    verify_file(checkpoint_path, model_spec["checkpoint_sha256"], "model checkpoint")
    verify_file(training_pool_path, registry["training_pool_sha256"], "training pool")
    verify_file(
        training_predictions_path,
        registry["training_predictions_sha256"],
        "training classifier predictions",
    )
    verify_file(
        classifier_config_path,
        registry["classifier_config_sha256"],
        "classifier configuration",
    )
    verify_file(
        role_archive_path,
        protocol["local_role_archive"]["sha256"],
        "frozen role archive",
    )
    dependency = validate_dependency_source(source_root, compatibility)

    phase_input = "historical_development" if args.phase == "development" else "full_test"
    h5_path = repo / protocol["inputs"][phase_input]["logical_path"]
    verify_file(h5_path, protocol["inputs"][phase_input]["sha256"], f"{args.phase} HDF5")
    reservoir_key = f"{args.phase}_context_reservoir_rows"
    target_key = f"{args.phase}_target_rows"
    with np.load(role_archive_path) as role_archive:
        threshold = float(role_archive["fixed_threshold"])
        if not math.isclose(threshold, protocol["threshold"]["value"], rel_tol=0.0, abs_tol=0.0):
            raise ValueError("Role archive and protocol manifest thresholds differ.")
        reservoir_rows = role_archive[reservoir_key].astype(np.int64)
        target_rows = role_archive[target_key].astype(np.int64)
    if args.n_context > reservoir_rows.size:
        parser.error(
            f"Requested {args.n_context} context events from a {reservoir_rows.size}-event reservoir."
        )
    context_rng = np.random.default_rng(args.context_seed)
    context_rows = np.sort(context_rng.choice(reservoir_rows, size=args.n_context, replace=False))
    dropout_seed = args.dropout_seed
    if dropout_seed is None:
        dropout_seed = 10000 + args.context_seed

    with h5py.File(h5_path, "r") as handle:
        context_ids = identity_matrix(handle, context_rows)
        target_ids = identity_matrix(handle, target_rows)
        context_energy = handle["energy"][context_rows].astype(np.float64)
        context_score = handle["score"][context_rows].astype(np.float64)
        target_energy = handle["energy"][target_rows].astype(np.float64)
        target_score = handle["score"][target_rows].astype(np.float64)
    expected_reservoir_hash = protocol["roles"][f"{args.phase}_context_reservoir"][
        "identity_sha256"
    ]
    with h5py.File(h5_path, "r") as handle:
        reservoir_hash = identity_hash(identity_matrix(handle, reservoir_rows))
    if reservoir_hash != expected_reservoir_hash:
        raise ValueError("Context reservoir identity hash does not match the frozen protocol.")
    expected_target_hash = protocol["roles"][f"{args.phase}_target"]["identity_sha256"]
    if identity_hash(target_ids) != expected_target_hash:
        raise ValueError("Target identity hash does not match the frozen protocol.")

    import torch

    from majorana_acp.cut_acceptance.config import load_config
    from scripts.diagnostics.cnp_test_inference import _cnp_infer_global, _load_cnp

    cfg = load_config(config_path)
    cnp = _load_cnp(cfg, checkpoint_path)
    device = next(cnp.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(dropout_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(dropout_seed)

    grid_energy = np.arange(
        protocol["energy_range_kev"][0],
        protocol["energy_range_kev"][1] + 0.5 * args.grid_spacing_kev,
        args.grid_spacing_kev,
    )
    query_energy = np.concatenate([target_energy, grid_energy])
    inference_start = time.perf_counter()
    prediction, prediction_std, n_context_used = _cnp_infer_global(
        cnp,
        cfg,
        query_energy,
        context_energy,
        context_score,
        threshold,
        n_mc=args.n_mc,
        seed=dropout_seed,
        max_context_per_pass=args.max_context_per_pass,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_start
    target_prediction = prediction[: target_energy.size]
    target_prediction_std = prediction_std[: target_energy.size]
    grid_prediction = prediction[target_energy.size :]
    grid_prediction_std = prediction_std[target_energy.size :]
    outcome = (target_score >= threshold).astype(np.float64)
    event_rows = event_metrics(target_energy, target_prediction, outcome)
    bin_rows, binned = bin_metrics(target_energy, target_prediction, outcome)
    contrast_rows = peak_contrasts(target_energy, target_prediction, outcome)
    roughness_rows = grid_roughness(grid_energy, grid_prediction)

    output_dir.mkdir(parents=True)
    event_path = output_dir / "event_predictions.npz"
    curve_path = output_dir / "curve.npz"
    summary_path = output_dir / "summary.json"
    np.savez(
        event_path,
        target_rows=target_rows,
        target_energy_kev=target_energy,
        outcome=outcome.astype(np.int8),
        prediction=target_prediction,
        prediction_std=target_prediction_std,
        context_rows=context_rows,
    )
    np.savez(
        curve_path,
        energy_kev=grid_energy,
        prediction=grid_prediction,
        prediction_std=grid_prediction_std,
        bin_centers_kev=binned["centers_kev"],
        bin_counts=binned["counts"],
        empirical_rate=binned["empirical_rate"],
        bin_prediction_mean=binned["prediction_mean"],
    )
    peak_memory_mib = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2))
        if device.type == "cuda"
        else None
    )
    summary = {
        "schema_version": 1,
        "run_id": output_dir.name,
        "run_kind": args.run_kind,
        "status": "completed",
        "start_time": start_time,
        "end_time": utc_now(),
        "source_commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_worktree_status": subprocess.check_output(
            ["git", "-C", str(repo), "status", "--short"], text=True
        ).splitlines(),
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "model": {
            "id": args.model_id,
            "display_name": model_spec["display_name"],
            "training_seed": model_spec["training_seed"],
            "comparison_role": model_spec["comparison_role"],
            "config": model_spec["config"],
            "config_sha256": model_spec["config_sha256"],
            "checkpoint": model_spec["checkpoint"],
            "checkpoint_sha256": model_spec["checkpoint_sha256"],
            "training_pool_sha256": registry["training_pool_sha256"],
        },
        "dependency": dependency,
        "protocol": {
            "manifest": str(args.protocol_manifest),
            "manifest_sha256": sha256_file(protocol_path),
            "role_archive_sha256": sha256_file(role_archive_path),
            "phase": args.phase,
            "exposure_statement": protocol["exposure_statement"],
            "threshold": threshold,
            "threshold_method": protocol["threshold"]["method"],
            "threshold_calibration_identity_sha256": protocol["threshold"][
                "calibration_identity_sha256"
            ],
            "context_reservoir_identity_sha256": reservoir_hash,
            "context_draw_identity_sha256": identity_hash(context_ids),
            "target_identity_sha256": expected_target_hash,
        },
        "randomness": {
            "training_seed": model_spec["training_seed"],
            "context_seed": args.context_seed,
            "dropout_seed": dropout_seed,
        },
        "counts": {
            "context_reservoir": int(reservoir_rows.size),
            "context_draw": int(context_rows.size),
            "context_per_mc_pass": int(n_context_used),
            "target": int(target_rows.size),
            "mc_passes": args.n_mc,
            "grid_points": int(grid_energy.size),
        },
        "metrics": {
            "event": event_rows,
            "bin_5kev": bin_rows,
            "peak_sideband_contrast": contrast_rows,
            "continuum_roughness": roughness_rows,
            "probability_clip": PROBABILITY_CLIP,
            "minimum_region_events": MIN_REGION_EVENTS,
            "minimum_bin_events": MIN_BIN_EVENTS,
        },
        "runtime": {
            "total_seconds": float(time.perf_counter() - timer_start),
            "inference_seconds": float(inference_seconds),
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "peak_memory_mib": peak_memory_mib,
        },
        "server_only_outputs": {
            "event_predictions": {
                "file": event_path.name,
                "bytes": event_path.stat().st_size,
                "sha256": sha256_file(event_path),
            },
            "curve": {
                "file": curve_path.name,
                "bytes": curve_path.stat().st_size,
                "sha256": sha256_file(curve_path),
            },
        },
    }
    with summary_path.open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "run_id": summary["run_id"],
                "status": summary["status"],
                "model": args.model_id,
                "phase": args.phase,
                "brier": event_rows[0]["brier"],
                "inference_seconds": inference_seconds,
                "peak_memory_mib": peak_memory_mib,
            }
        )
    )


if __name__ == "__main__":
    main()
