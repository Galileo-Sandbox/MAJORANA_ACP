#!/usr/bin/env python3
"""Evaluate one frozen mechanism-control checkpoint with the old protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from control_models import load_control_checkpoint  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--mode", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--context-seed", type=int, choices=range(100, 110), required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    repo = args.repo.resolve()
    root = repo / "ml4phy-paper/review_followup_20260910"
    output = root / "runs/evaluation" / args.run_id
    if output.exists():
        parser.error(f"Refusing existing output {output}")
    training_id = f"20260910-mechanism-{args.mode}-seed{args.training_seed}-train3000"
    training = root / "runs/training" / training_id
    record = json.loads((training / "runner_record.json").read_text())
    if record["status"] != "completed" or record["control_mode"] != args.mode:
        raise ValueError("Training record is incomplete or mode-mismatched")
    checkpoint = training / "artifacts/control.ckpt"
    if sha256_file(checkpoint) != record["outputs"]["control.ckpt"]["sha256"]:
        raise ValueError("Control checkpoint hash mismatch")
    protocol = json.loads((root / "configs/protocol_v1.json").read_text())
    config_path = (
        repo
        / "ml4phy-paper/runs/phase2/training/20260909-phase2-b5000-ours-seed0-train3000/resolved_config.yaml"
    )
    from majorana_acp.cut_acceptance.config import load_config
    from majorana_acp.cut_acceptance.pipeline import build_local_cnp
    from majorana_acp.cut_acceptance.positional_encoding import phi_dim
    from scripts.diagnostics.cnp_test_inference import _cnp_infer_global

    cfg = load_config(config_path)
    cfg.train_predictions_path = (
        repo / protocol["fixed_scientific_settings"]["training_subset"]["logical_path"]
    )
    torch.manual_seed(args.training_seed)
    base = build_local_cnp(cfg, dim_phi=phi_dim(cfg.positional_encoding))
    model, payload = load_control_checkpoint(checkpoint, base, expected_mode=args.mode)
    model.to("cuda")
    model.eval()
    extension_path = repo / "ml4phy-paper/local/protocol/extension_context_roles_v1.npz"
    roles_path = repo / "ml4phy-paper/local/protocol/frozen_roles_v1.npz"
    frozen_path = repo / "ml4phy-paper/manifests/frozen_protocol_v1.json"
    frozen = json.loads(frozen_path.read_text())
    with np.load(extension_path) as archive:
        context_rows = archive[f"final_context_s{args.context_seed}_n500_rows"].astype(np.int64)
    with np.load(roles_path) as archive:
        target_rows = archive["final_target_rows"].astype(np.int64)
        threshold = float(archive["fixed_threshold"])
    if threshold != 0.540643572807312 or context_rows.size != 500 or target_rows.size != 114400:
        raise ValueError("Frozen evaluation roles changed")
    h5_path = repo / frozen["inputs"]["full_test"]["logical_path"]
    with h5py.File(h5_path) as handle:
        context_sorted = np.sort(context_rows)
        context_energy = handle["energy"][context_sorted].astype(np.float64)
        context_score = handle["score"][context_sorted].astype(np.float64)
        target_energy = handle["energy"][target_rows].astype(np.float64)
        target_score = handle["score"][target_rows].astype(np.float64)
    grid_energy = np.arange(500.0, 3000.0 + 0.5, 1.0)
    query_energy = np.concatenate([target_energy, grid_energy])
    torch.manual_seed(10100)
    torch.cuda.manual_seed_all(10100)
    torch.cuda.reset_peak_memory_stats()
    inference_start = time.perf_counter()
    prediction, prediction_std, n_context = _cnp_infer_global(
        model,
        cfg,
        query_energy,
        context_energy,
        context_score,
        threshold,
        n_mc=50,
        seed=10100,
        max_context_per_pass=500,
    )
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_start
    target_prediction = prediction[: target_rows.size]
    target_std = prediction_std[: target_rows.size]
    grid_prediction = prediction[target_rows.size :]
    grid_std = prediction_std[target_rows.size :]
    outcome = (target_score >= threshold).astype(np.int8)
    edges = np.arange(500.0, 3005.0, 5.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    index = np.searchsorted(edges, target_energy, side="right") - 1
    valid = (index >= 0) & (index < centers.size)
    counts = np.bincount(index[valid], minlength=centers.size)
    passing = np.bincount(index[valid], weights=outcome[valid], minlength=centers.size)
    predicted = np.bincount(index[valid], weights=target_prediction[valid], minlength=centers.size)
    empirical = np.divide(passing, counts, out=np.full(centers.size, np.nan), where=counts > 0)
    bin_prediction = np.divide(
        predicted, counts, out=np.full(centers.size, np.nan), where=counts > 0
    )
    output.mkdir(parents=True)
    np.savez(
        output / "event_predictions.npz",
        target_rows=target_rows,
        target_energy_kev=target_energy,
        outcome=outcome,
        prediction=target_prediction,
        prediction_std=target_std,
        context_rows=context_sorted,
    )
    np.savez(
        output / "curve.npz",
        energy_kev=grid_energy,
        prediction=grid_prediction,
        prediction_std=grid_std,
        bin_centers_kev=centers,
        bin_counts=counts,
        empirical_rate=empirical,
        bin_prediction_mean=bin_prediction,
    )
    summary = {
        "schema_version": 1,
        "status": "completed",
        "run_id": args.run_id,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "control_mode": args.mode,
        "training_seed": args.training_seed,
        "context_seed": args.context_seed,
        "context_size": n_context,
        "dropout_seed": 10100,
        "mc_passes": 50,
        "threshold": threshold,
        "target_events": int(target_rows.size),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_mode": payload["control_mode"],
        "inference_seconds": inference_seconds,
        "total_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated() / 1024**2),
        "outputs": {},
        "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    for name in ("event_predictions.npz", "curve.npz"):
        path = output / name
        summary["outputs"][name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
