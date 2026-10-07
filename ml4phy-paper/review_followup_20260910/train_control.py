#!/usr/bin/env python3
"""Train one frozen 5k mechanism-control checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from control_models import (  # noqa: E402
    MODES,
    apply_control_mode,
    parameter_inventory,
    save_control_checkpoint,
    task_schedule,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--mode", choices=MODES[1:], required=True)
    parser.add_argument("--training-seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    repo = args.repo.resolve()
    root = repo / "ml4phy-paper/review_followup_20260910"
    protocol_path = root / "configs/protocol_v1.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol["status"] != "frozen_approved_pretraining_checks_passed":
        raise ValueError("Mechanism protocol is not frozen and approved")
    worktree = subprocess.check_output(
        ["git", "status", "--short"], cwd=repo, text=True
    ).splitlines()
    if worktree:
        raise RuntimeError("Training requires the frozen implementation commit and clean worktree")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    run_dir = root / "runs/training" / args.run_id
    if run_dir.exists():
        parser.error(f"Refusing existing run directory {run_dir}")
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True)
    record_path = run_dir / "runner_record.json"
    config_path = (
        repo
        / "ml4phy-paper/runs/phase2/training/20260909-phase2-b5000-ours-seed0-train3000/resolved_config.yaml"
    )

    from core.surrogate_cnp import cnp_loss, split_context_target

    from majorana_acp.cut_acceptance.config import load_config
    from majorana_acp.cut_acceptance.event_sampler import EventSampler, load_events
    from majorana_acp.cut_acceptance.pipeline import build_local_cnp
    from majorana_acp.cut_acceptance.positional_encoding import phi_dim

    cfg = load_config(config_path)
    if cfg.training.n_steps != 3000 or cfg.training.batch_size != 16:
        raise ValueError("Frozen 3000-step/batch-16 schedule changed")
    expected_subset = protocol["fixed_scientific_settings"]["training_subset"]
    subset_path = repo / expected_subset["logical_path"]
    if sha256_file(subset_path) != expected_subset["sha256"]:
        raise ValueError("Frozen 5k subset hash mismatch")
    cfg = cfg.model_copy(
        update={
            "training": cfg.training.model_copy(update={"seed": args.training_seed}),
            "train_predictions_path": subset_path,
        }
    )
    energy, score = load_events(
        cfg.train_predictions_path,
        target_class=cfg.target_class,
        energy_range=cfg.energy_range,
    )
    if energy.size != 5000:
        raise ValueError("Density input is not exactly the frozen 5k pool")
    sampler = EventSampler(
        energy,
        score,
        energy_range=cfg.energy_range,
        energy_bin_width=cfg.energy_bin_width,
        threshold_range=cfg.threshold_range,
        min_events_per_bin=cfg.min_events_per_bin,
        t_sampling="boundary_mix",
        sampling_pattern=cfg.sampling_pattern,
        zoom_window_width_kev=cfg.zoom_window_width_kev,
        local_event_fraction=cfg.local_event_fraction,
        n_clusters=cfg.n_clusters,
        physics_peaks_kev=list(cfg.physics_peaks_kev),
        positional_encoding=cfg.positional_encoding,
        density_sampling=cfg.density_sampling,
        density_kde_radius_kev=cfg.density_kde_radius_kev,
    )
    if int(sampler.bin_event_counts.sum()) != 4984:
        raise ValueError("Sampler does not retain exactly 4,984 events")
    np.savez(
        artifact_dir / "training_pool.npz",
        bin_centers=sampler.bin_centers,
        bin_event_counts=sampler.bin_event_counts,
        n_events_total=np.int64(energy.size),
    )
    torch.manual_seed(args.training_seed)
    model = build_local_cnp(cfg, dim_phi=phi_dim(cfg.positional_encoding))
    apply_control_mode(model, args.mode)
    device = torch.device("cuda")
    model.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    rng = np.random.default_rng(args.training_seed)
    history = {"step": [], "loss": []}
    task_trace = []
    start_time = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    record = {
        "schema_version": 1,
        "status": "running",
        "run_id": args.run_id,
        "control_mode": args.mode,
        "training_seed": args.training_seed,
        "source_commit": source_commit,
        "start_time": start_time,
        "protocol_sha256": sha256_file(protocol_path),
        "implementation_sha256": sha256_file(HERE / "control_models.py"),
        "subset_sha256": sha256_file(subset_path),
        "subset_identity_sha256": expected_subset["identity_sha256"],
        "nominal_events": 5000,
        "sampling_eligible_events": 4984,
        "density_buffer_events": 5000,
        "training_steps": 3000,
        "batch_size": 16,
        "parameter_inventory": parameter_inventory(model, args.mode),
    }
    write_json(record_path, record)
    model.train()
    for step in range(cfg.training.n_steps):
        n_events = int(rng.integers(cfg.n_trial_events_min, cfg.n_trial_events_max + 1))
        n_context = int(rng.integers(128, min(512, n_events - 1) + 1))
        sampler_seed = int(rng.integers(0, 2**31 - 1))
        split_seed = int(rng.integers(0, 2**31 - 1))
        task_trace.append((n_events, n_context, sampler_seed, split_seed))
        batch = sampler.generate(n_trials=16, n_events=n_events, seed=sampler_seed)
        context, target = split_context_target(batch, n_context=n_context, seed=split_seed)
        result = model(context, target)
        labels = torch.as_tensor(target.labels, dtype=torch.float32, device=device)
        loss = cnp_loss(result, labels, n_mc_samples=cfg.training.n_mc_samples)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        optimizer.step()
        history["step"].append(step)
        history["loss"].append(float(loss.detach()))
        if (step + 1) % 250 == 0:
            print(json.dumps({"step": step + 1, "loss": history["loss"][-1]}), flush=True)
    if task_trace != task_schedule(args.training_seed, 3000):
        raise ValueError("Executed training task schedule differs from frozen schedule")
    torch.cuda.synchronize(device)
    peak_memory = float(torch.cuda.max_memory_allocated(device) / 1024**2)
    model.to("cpu")
    checkpoint = artifact_dir / "control.ckpt"
    save_control_checkpoint(
        checkpoint,
        model,
        mode=args.mode,
        history=history,
        metadata={
            "source_commit": source_commit,
            "protocol_sha256": sha256_file(protocol_path),
            "training_seed": args.training_seed,
            "task_schedule_sha256": protocol["pretraining_checks"]["training_task_schedule_sha256"][
                str(args.training_seed)
            ],
            "final_checkpoint_step": 3000,
        },
    )
    record.update(
        {
            "status": "completed",
            "end_time": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "wall_seconds": time.perf_counter() - started,
            "peak_gpu_memory_mib": peak_memory,
            "final_loss": history["loss"][-1],
            "outputs": {
                "control.ckpt": {
                    "sha256": sha256_file(checkpoint),
                    "bytes": checkpoint.stat().st_size,
                },
                "training_pool.npz": {
                    "sha256": sha256_file(artifact_dir / "training_pool.npz"),
                    "bytes": (artifact_dir / "training_pool.npz").stat().st_size,
                },
            },
        }
    )
    write_json(record_path, record)
    print(
        json.dumps(
            {
                key: record[key]
                for key in ("run_id", "status", "wall_seconds", "peak_gpu_memory_mib", "final_loss")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
