#!/usr/bin/env python3
"""Audit the frozen mechanism slice and write its pre-training protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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

from control_models import (  # noqa: E402
    GLOBAL_ATTENTION_MODES,
    GLOBAL_GATE_MODES,
    MODES,
    apply_control_mode,
    canonical_state_dict,
    load_control_checkpoint,
    parameter_inventory,
    save_control_checkpoint,
    task_schedule,
)

IDENTITY_FIELDS = ("run_number", "detector", "id", "tp0")
SOURCE_FILES = (
    "majorana_acp/models/attentive_cnp.py",
    "majorana_acp/cut_acceptance/pipeline.py",
    "majorana_acp/cut_acceptance/binned_sampler.py",
    "ml4phy-paper/local/resum-flex-edba6a/core/training.py",
    "ml4phy-paper/local/resum-flex-edba6a/core/surrogate_cnp.py",
    "ml4phy-paper/scripts/run_phase2_training.py",
    "ml4phy-paper/scripts/evaluate_fixed_protocol.py",
    "ml4phy-paper/scripts/export_paper_presentation.py",
    "ml4phy-paper/scripts/export_paper_coverage.py",
    "ml4phy-paper/review_followup_20260910/control_models.py",
    "ml4phy-paper/review_followup_20260910/audit_and_freeze.py",
    "ml4phy-paper/review_followup_20260910/export_existing.py",
    "ml4phy-paper/review_followup_20260910/train_control.py",
    "ml4phy-paper/review_followup_20260910/evaluate_control.py",
    "ml4phy-paper/review_followup_20260910/run_training_campaign.py",
    "ml4phy-paper/review_followup_20260910/run_evaluation_campaign.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def identity_hash(path: Path) -> tuple[int, str]:
    with h5py.File(path) as handle:
        identities = np.column_stack(
            [handle[field][:].astype("<i8", copy=False) for field in IDENTITY_FIELDS]
        )
    order = np.lexsort(tuple(identities[:, i] for i in reversed(range(4))))
    payload = np.ascontiguousarray(identities[order], dtype="<i8").tobytes()
    return identities.shape[0], hashlib.sha256(payload).hexdigest()


def identity_tuples(handle: h5py.File, rows=None) -> set[tuple[int, ...]]:
    values = [
        handle[field][:] if rows is None else handle[field][rows] for field in IDENTITY_FIELDS
    ]
    return set(
        zip(
            *(np.asarray(value, dtype=np.int64).tolist() for value in values),
            strict=True,
        )
    )


def build_base(cfg, seed: int):
    from majorana_acp.cut_acceptance.pipeline import build_local_cnp
    from majorana_acp.cut_acceptance.positional_encoding import phi_dim

    torch.manual_seed(seed)
    return build_local_cnp(cfg, dim_phi=phi_dim(cfg.positional_encoding))


def load_original(cfg, checkpoint: Path, seed: int):
    model = build_base(cfg, seed)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    return model, payload


def smoke_batches(cfg):
    from core.surrogate_cnp import split_context_target

    from majorana_acp.cut_acceptance.event_sampler import EventSampler, load_events

    energy, score = load_events(
        cfg.train_predictions_path,
        target_class=cfg.target_class,
        energy_range=cfg.energy_range,
    )
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
    batch = sampler.generate(n_trials=2, n_events=96, seed=314159)
    return split_context_target(batch, n_context=48, seed=271828)


def outputs(model, ctx, target, *, train: bool, seed: int):
    model.train(train)
    torch.manual_seed(seed)
    result = model(ctx, target)
    return result.mu_logit.detach().clone(), result.log_sigma.detach().clone()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    args = parser.parse_args()
    started = time.perf_counter()
    repo = args.repo.resolve()
    root = repo / "ml4phy-paper"
    output = root / "review_followup_20260910"
    base_config_path = (
        root
        / "runs/phase2/training/20260909-phase2-b5000-ours-seed0-train3000/resolved_config.yaml"
    )
    checkpoint = (
        root / "runs/phase2/training/20260909-phase2-b5000-ours-seed0-train3000/artifacts/cnp.ckpt"
    )
    phase2_config_path = root / "configs/phase2_training_v1.json"
    phase2_protocol_path = root / "manifests/phase2_protocol_v1.json"
    frozen_protocol_path = root / "manifests/frozen_protocol_v1.json"
    extension_protocol_path = root / "manifests/extension_protocol_v1.json"
    compatibility_path = root / "manifests/resum_flex_compatibility.json"
    presentation_manifest_path = root / "manifests/paper_presentation_export.json"
    coverage_manifest_path = root / "manifests/paper_coverage_export.json"
    required = [
        base_config_path,
        checkpoint,
        phase2_config_path,
        phase2_protocol_path,
        frozen_protocol_path,
        extension_protocol_path,
        compatibility_path,
        presentation_manifest_path,
        coverage_manifest_path,
        *[repo / name for name in SOURCE_FILES],
    ]
    missing = [str(path.relative_to(repo)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required historical sources: {missing}")

    phase2_config = read_json(phase2_config_path)
    phase2_protocol = read_json(phase2_protocol_path)
    frozen = read_json(frozen_protocol_path)
    extension = read_json(extension_protocol_path)
    compatibility = read_json(compatibility_path)
    presentation = read_json(presentation_manifest_path)
    coverage = read_json(coverage_manifest_path)
    if phase2_config["status"] != "frozen":
        raise ValueError("Phase 2 configuration is not frozen")
    if frozen["threshold"]["value"] != 0.540643572807312:
        raise ValueError("Frozen threshold changed")
    if extension["context_protocol"]["local_archive"]["sha256"] is None:
        raise ValueError("Extension context protocol has no archive hash")
    subset = next(item for item in phase2_protocol["subsets"] if item["budget"] == 5000)
    subset_path = repo / subset["logical_path"]
    count, observed_identity = identity_hash(subset_path)
    if (
        sha256_file(subset_path) != subset["sha256"]
        or count != 5000
        or observed_identity != subset["identity_sha256"]
    ):
        raise ValueError("Original 5k subset hash/count/identity mismatch")
    if (
        subset["sampling_eligible_unique_events"] != 4984
        or subset["density_pool_unique_events"] != 5000
    ):
        raise ValueError("Original 5k effective or density count mismatch")
    if (
        presentation["matrix"]["neural_recovered"] != 600
        or coverage["checks"]["source_record_count"] != 630
    ):
        raise ValueError("Existing portable inventory mismatch")
    if (
        coverage["checks"]["overall_supported_bins"] != 442
        or coverage["checks"]["overall_excluded_bins"] != 58
    ):
        raise ValueError("Existing support audit mismatch")
    role_path = repo / frozen["local_role_archive"]["logical_path"]
    full_test_path = repo / frozen["inputs"]["full_test"]["logical_path"]
    if sha256_file(role_path) != frozen["local_role_archive"]["sha256"]:
        raise ValueError("Frozen role archive hash mismatch")
    with np.load(role_path) as roles:
        target_rows = roles["final_target_rows"].astype(np.int64)
        context_rows = roles["final_context_reservoir_rows"].astype(np.int64)
    with h5py.File(subset_path) as train_handle, h5py.File(full_test_path) as test_handle:
        train_ids = identity_tuples(train_handle)
        target_overlap = len(train_ids & identity_tuples(test_handle, target_rows))
        context_overlap = len(train_ids & identity_tuples(test_handle, context_rows))
    if target_overlap or context_overlap:
        raise ValueError("Training subset overlaps frozen target or context reservoir")

    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    from majorana_acp.cut_acceptance.config import load_config

    cfg = load_config(base_config_path)
    original, payload = load_original(cfg, checkpoint, 0)
    full, _ = load_original(cfg, checkpoint, 0)
    apply_control_mode(full, "full_density")
    ctx, target = smoke_batches(cfg)
    parity = {}
    for train, label in ((False, "deterministic"), (True, "fixed_rng_stochastic")):
        left = outputs(original, ctx, target, train=train, seed=424242)
        right = outputs(full, ctx, target, train=train, seed=424242)
        differences = [float(torch.max(torch.abs(a - b))) for a, b in zip(left, right, strict=True)]
        parity[label] = {
            "mu_logit_max_abs_difference": differences[0],
            "log_sigma_max_abs_difference": differences[1],
        }
        if max(differences) != 0.0:
            raise ValueError(f"Full-mode parity failed in {label}: {differences}")

    parameter_counts = {}
    shared_initialization = {}
    roundtrip = {}
    gradients = {}
    tmp = output / "local/audit"
    tmp.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        base = build_base(cfg, 0)
        reference_state = {key: value.detach().clone() for key, value in base.state_dict().items()}
        apply_control_mode(base, mode)
        canonical = canonical_state_dict(base)
        changed = [
            key for key in reference_state if not torch.equal(reference_state[key], canonical[key])
        ]
        expected_changed = ["attention.kappa_raw"] if mode in GLOBAL_GATE_MODES else []
        if changed != expected_changed:
            raise ValueError(f"Unexpected initialization changes for {mode}: {changed}")
        shared_initialization[mode] = {"changed_tensors": changed, "expected": expected_changed}
        parameter_counts[mode] = parameter_inventory(base, mode)
        if mode == "full_density":
            continue
        base.train()
        result = base(ctx, target)
        loss = result.mu_logit.square().mean() + result.log_sigma.square().mean()
        loss.backward()
        attention = base.attention.base
        checks = {}
        if mode in GLOBAL_GATE_MODES:
            checks["global_gate_phi_gradient"] = float(attention.kappa_raw.grad.detach().abs())
        if mode in GLOBAL_ATTENTION_MODES:
            checks["global_bandwidth_output_bias_gradient"] = float(
                attention.pool_sfn_net.network[-1].bias.grad.detach().abs().sum()
            )
            checks["global_temperature_output_bias_gradient"] = float(
                attention.pool_tau_net.network[-1].bias.grad.detach().abs().sum()
            )
        if not checks or any(not math.isfinite(value) or value <= 0 for value in checks.values()):
            raise ValueError(f"Finite nonzero control-gradient check failed for {mode}: {checks}")
        gradients[mode] = checks
        checkpoint_path = tmp / f"{mode}.ckpt"
        save_control_checkpoint(
            checkpoint_path,
            base,
            mode=mode,
            history={"loss": [float(loss.detach())]},
            metadata={"audit_only": True},
        )
        restored, saved = load_control_checkpoint(
            checkpoint_path, build_base(cfg, 999), expected_mode=mode
        )
        left = outputs(base, ctx, target, train=False, seed=7)
        right = outputs(restored, ctx, target, train=False, seed=7)
        difference = max(
            float(torch.max(torch.abs(a - b))) for a, b in zip(left, right, strict=True)
        )
        if difference != 0.0 or saved["control_mode"] != mode:
            raise ValueError(f"Checkpoint round-trip failed for {mode}: {difference}")
        roundtrip[mode] = difference

    # The custom schedule is compared with an independent transcription of the
    # executed trainer's RNG order, not merely with a shared random seed.
    task_checks = {}
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        independent = []
        for _ in range(3000):
            n_events = int(rng.integers(640, 1025))
            n_context = int(rng.integers(128, min(512, n_events - 1) + 1))
            independent.append(
                (
                    n_events,
                    n_context,
                    int(rng.integers(0, 2**31 - 1)),
                    int(rng.integers(0, 2**31 - 1)),
                )
            )
        if independent != task_schedule(seed, 3000):
            raise ValueError(f"Training-task schedule mismatch for seed {seed}")
        task_checks[str(seed)] = sha256_json(independent)

    source_hashes = {name: sha256_file(repo / name) for name in SOURCE_FILES}
    input_hashes = {
        str(path.relative_to(repo)): sha256_file(path)
        for path in (
            base_config_path,
            checkpoint,
            phase2_config_path,
            phase2_protocol_path,
            frozen_protocol_path,
            extension_protocol_path,
            compatibility_path,
            presentation_manifest_path,
            coverage_manifest_path,
            subset_path,
        )
    }
    protocol = {
        "schema_version": 1,
        "status": "frozen_approved_pretraining_checks_passed",
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "historical_exposure": "Post-review extension on historically exposed frozen data; not an untouched test.",
        "scope": {
            "modes": list(MODES),
            "reused_full_density_training_jobs": 3,
            "new_training_jobs": 12,
            "new_evaluation_cells": 120,
            "mechanism_cells_including_reuse": 150,
            "excluded": [
                "new data",
                "classifier training",
                "new threshold",
                "new budget or subset ordering",
                "baseline retraining",
                "GP or KDE fitting",
                "sparse GP",
                "uncertainty calibration",
                "high-MC study",
                "hyperparameter sweep",
            ],
        },
        "fixed_scientific_settings": {
            "training_subset": subset,
            "training_seeds": [0, 1, 2],
            "context_seeds": list(range(100, 110)),
            "context_size": 500,
            "training_steps": 3000,
            "batch_size": 16,
            "threshold": 0.540643572807312,
            "target_events": 114400,
            "mc_passes": 50,
            "dropout_seed": 10100,
            "checkpoint_rule": "final 3000-step checkpoint",
            "classifier_training_events": 18866,
        },
        "control_parameterizations": {
            "global_gate": "lambda=1+9*sigmoid(phi), phi_init=log(2/7), original kappa_raw repurposed, alpha=5, bands=0..9",
            "global_attention": "existing pool_sfn_net and pool_tau_net evaluated at constant Z0=(0,0)",
            "density_free_global": "global gate and attention plus direct decoder contrast_ratio replaced by zero",
        },
        "resource_gates": {
            "existing_export_seconds": 2700,
            "scientific_campaign_seconds": 7200,
            "pilot_margin_factor": 1.25,
            "maximum_concurrent_jobs": 4,
            "maximum_retry_per_technical_failure": 1,
            "portable_target_bytes": 5 * 1024**2,
            "portable_hard_limit_bytes": 20 * 1024**2,
        },
        "expected_inventory": {
            "existing_cells": {
                "neural": 600,
                "kernel_context_only": 10,
                "kernel_pooled": 10,
                "bernoulli_gp": 10,
            },
            "new_training": [
                {"mode": mode, "training_seed": seed} for mode in MODES[1:] for seed in (0, 1, 2)
            ],
            "new_evaluation": [
                {"mode": mode, "training_seed": seed, "context_seed": context}
                for mode in MODES[1:]
                for seed in (0, 1, 2)
                for context in range(100, 110)
            ],
        },
        "pretraining_checks": {
            "full_mode_parity": parity,
            "checkpoint_roundtrip_max_abs_difference": roundtrip,
            "finite_nonzero_gradients": gradients,
            "shared_initialization": shared_initialization,
            "training_task_schedule_sha256": task_checks,
            "parameter_counts": parameter_counts,
            "target_overlap": target_overlap,
            "context_reservoir_overlap": context_overlap,
            "density_buffer_fields_used": "energy only; outcomes/scores are not passed to the model builder",
        },
        "hashes": {"inputs": input_hashes, "sources": source_hashes},
        "dependency": compatibility,
        "existing_export_checks": coverage["checks"],
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(output / "configs/protocol_v1.json", protocol)
    report = f"""# Pre-training mechanism-control audit

Status: passed. No scientific training or evaluation was run during this audit.

- Base commit: `{protocol["base_commit"]}`.
- Full-mode deterministic and fixed-RNG stochastic maximum differences: exactly 0.
- Original 5k subset: 5,000 nominal, 4,984 sampling-eligible, 5,000 density-buffer events; identity and file hashes passed.
- Existing inventory: 600 neural plus 30 classical/control cells; frozen target support remains 442/500 bins with 101 events in 58 excluded bins.
- Control checkpoint round trips, finite intended gradients, global parameterization tests, shared initialization, and all three 3,000-step task-schedule hashes passed.
- The frozen target and context reservoirs are identity-disjoint from the training subset according to the Phase 2/extension manifests.

The implementation retains all original model tensors. Global-gate modes
repurpose `kappa_raw` as `phi`; global-attention modes retain both original
networks but their two first-layer input-weight columns are structurally inactive
under `Z0=(0,0)`. The density-free mode additionally makes the decoder's direct
R-input weight column inactive. Exact total/effectively participating parameter
counts are in `configs/protocol_v1.json`.

The full-mode parity gate is satisfied, so the authorized single-job resource
pilot may proceed only after this protocol and implementation are committed.
"""
    (output / "reports").mkdir(parents=True, exist_ok=True)
    (output / "reports/pretraining_audit.md").write_text(report)
    print(
        json.dumps({"status": "passed", "runtime_seconds": protocol["runtime_seconds"]}, indent=2)
    )


if __name__ == "__main__":
    main()
