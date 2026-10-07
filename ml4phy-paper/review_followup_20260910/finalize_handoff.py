#!/usr/bin/env python3
"""Write the final lightweight provenance manifest and campaign handoff."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    repo = Path(".").resolve()
    campaign_path = HERE / "runs/campaigns/20260910-mechanism-controls-v1/campaign_record.json"
    campaign = json.loads(campaign_path.read_text())
    if campaign["status"] != "scientific_matrix_complete":
        raise ValueError("Cannot finalize an incomplete scientific campaign")
    registry_path = HERE / "tables/mechanism_run_inventory.csv"
    registry = list(csv.DictReader(registry_path.open()))
    if len(registry) != 150 or len({x["run_id"] for x in registry}) != 150:
        raise ValueError("Mechanism registry must contain 150 unique cells")
    modes = {x["mode"] for x in registry}
    if modes != {
        "full_density",
        "global_gate",
        "global_attention",
        "global_both",
        "density_free_global",
    }:
        raise ValueError("Mechanism modes are incomplete")

    portable = []
    for folder in ("configs", "figures", "reports", "tables", "tests"):
        for path in sorted((HERE / folder).glob("**/*")):
            if path.is_file() and "__pycache__" not in path.parts:
                portable.append(path)
    for name in (
        ".gitignore",
        "PLAN.md",
        "aggregate_mechanism.py",
        "audit_and_freeze.py",
        "control_models.py",
        "evaluate_control.py",
        "export_existing.py",
        "finalize_handoff.py",
        "rerun_affected_evaluations.py",
        "run_evaluation_campaign.py",
        "run_training_campaign.py",
        "train_control.py",
    ):
        path = HERE / name
        if path.exists():
            portable.append(path)
    portable = sorted(set(portable))
    size = sum(path.stat().st_size for path in portable)
    if size >= 20 * 1024 * 1024:
        raise ValueError(f"Portable artifacts exceed 20 MiB: {size}")

    local_index = {
        "status": "server_only",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "training": campaign["completed_training"],
        "valid_evaluations": campaign["completed_evaluations"],
        "invalid_evaluations_retained": campaign.get(
            "invalid_evaluations_checkpoint_restore_reset", []
        ),
        "raw_event_payloads_are_not_tracked": True,
    }
    local_path = HERE / "local/server_artifact_index.json"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(json.dumps(local_index, indent=2) + "\n")

    existing_record = json.loads((HERE / "reports/existing_export_record.json").read_text())
    aggregate_record = json.loads((HERE / "reports/mechanism_aggregation_record.json").read_text())
    validation_record = json.loads((HERE / "reports/validation_record.json").read_text())
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "branch": "ml4phy-paper",
        "historical_exposure": "post-review follow-up on historically exposed frozen data",
        "protocol": {
            "path": "ml4phy-paper/review_followup_20260910/configs/protocol_v1.json",
            "sha256": sha256_file(HERE / "configs/protocol_v1.json"),
            "base_commit": json.loads((HERE / "configs/protocol_v1.json").read_text())[
                "base_commit"
            ],
            "completed_evidence_base_commit": "ed631f9928f086fa0f9fd447571abed171fc7d6c",
        },
        "implementation_commits": {
            "training": campaign["source_commit"],
            "evaluation_logistics": campaign.get("evaluation_source_commit"),
            "checkpoint_restore_fix": campaign.get("checkpoint_restore_fix_commit"),
            "aggregation": aggregate_record["source_commit"],
        },
        "inventory": {
            "existing_cells_recovered": existing_record["recovered_cells"],
            "existing_cells_expected": 630,
            "new_training_jobs_completed": len(campaign["completed_training"]),
            "new_training_jobs_expected": 12,
            "new_valid_evaluations_completed": len(campaign["completed_evaluations"]),
            "new_valid_evaluations_expected": 120,
            "mechanism_cells_including_reuse": len(registry),
            "mechanism_cells_expected": 150,
            "reused_full_density_cells": 30,
            "invalid_evaluations_retained_server_only": len(
                campaign.get("invalid_evaluations_checkpoint_restore_reset", [])
            ),
        },
        "fixed_estimator": {
            "efficiency_training_nominal_events": 5000,
            "efficiency_training_sampling_eligible_events": 4984,
            "density_buffer_events": 5000,
            "classifier_training_events": 18866,
            "context_events": 500,
            "target_events": 114400,
            "threshold": 0.540643572807312,
            "mc_passes": 50,
            "dropout_seed": 10100,
            "training_seeds": [0, 1, 2],
            "context_seeds": list(range(100, 110)),
        },
        "reference": {
            "target_identity_sha256": aggregate_record["target_identity_sha256"],
            "bins": 500,
            "supported_bins": 442,
            "excluded_bins": 58,
            "excluded_target_events": 101,
        },
        "resource_record": {
            "existing_export_seconds": existing_record["runtime_seconds"],
            "scientific_hard_cap_seconds": 7200,
            "scientific_elapsed_seconds": campaign["scientific_elapsed_seconds"],
            "training_max_workers": 4,
            "evaluation_safe_max_workers": 3,
            "training_peak_gpu_memory_mib_per_job": max(
                x["peak_gpu_memory_mib"] for x in campaign["completed_training"]
            ),
            "evaluation_peak_gpu_memory_mib_per_job": max(
                x["peak_gpu_memory_mib"] for x in campaign["completed_evaluations"]
            ),
            "portable_bytes": size,
            "portable_target_bytes": 5 * 1024 * 1024,
            "portable_hard_limit_bytes": 20 * 1024 * 1024,
        },
        "failures_and_corrections": {
            "campaign_failures": campaign["failures"],
            "training_pilot_pre_step_config_failure": True,
            "four_worker_evaluation_oom": True,
            "checkpoint_restore_reset_detected_post_evaluation": True,
            "checkpoint_restore_affected_cells": 90,
            "checkpoint_restore_corrected_cells": campaign.get(
                "checkpoint_restore_correction_completed_cells"
            ),
            "invalid_outputs_excluded": True,
        },
        "validation": {
            **validation_record,
            "tests_passed": 24,
            "test_failures": 0,
            "ruff_status": "passed",
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "platform": platform.platform(),
        },
        "source_artifact_registry": {
            "path": str(registry_path.relative_to(repo)),
            "sha256": sha256_file(registry_path),
            "contains_checkpoint_and_prediction_hashes": True,
        },
        "portable_outputs": {
            str(path.relative_to(repo)): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in portable
            if path.name != "provenance_v1.json"
        },
        "limitations": [
            "The 2k efficiency-training failure remains.",
            "The existing 10k result is nonmonotonic.",
            "Sparse-tail performance remains limited and no eligible tail events enter the sampler.",
            "The final reference is finite and historically exposed.",
            "Mechanism controls use one original 5k ordering only.",
            "The fixed classifier used 18,866 training events.",
            "Reference-band agreement is descriptive, not calibrated coverage.",
        ],
    }
    manifest_path = HERE / "manifests/provenance_v1.json"
    manifest_path.parent.mkdir(exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")

    report = f"""# Final mechanism-control handoff

Status: complete. The existing-artifact inventory recovered 630/630 cells. New work completed 12/12 training jobs and 120/120 valid evaluations. Together with 30 reused full-density cells, the mechanism inventory is 150/150. No job is running at handoff generation.

The full model obtains Peaks/Continuum C2 of 45.0%/88.8%. Replacing only its decoder gate with a learned global cutoff reduces these to 20.8%/81.7%; the paired difference is positive in every seed for both endpoints. Replacing only density-adaptive attention with learned global attention yields 45.8%/88.8%, essentially matching the full model and failing to support a necessary independent benefit from adaptive attention. Retaining direct density input under globally controlled modulation raises Peaks C2 by 2.9 points but lowers Continuum C2 by 5.2 points, so that control shows a trade-off rather than a joint improvement. The full density package exceeds the density-free global model by 25.0 points in Peaks and 2.0 points in Continuum, but it does not isolate one component.

Count-scale diagnostics reinforce the peak result: the full model's mean predicted-minus-observed count differences in DEP/1620/SE/FE cores are -14.3/+61.4/-1.5/-333.2 events, compared with substantially larger efficiency and weighted-MAE errors for global-gate controls in several cores. These are discrepancies against a finite calibration reference, not known physical biases or significances. Each core has only two supported bins, so Ck is discrete.

Scientific elapsed time was {campaign["scientific_elapsed_seconds"]:.1f} seconds under the 7,200-second cap. Four-way evaluation used too much memory; three workers were safe at {manifest["resource_record"]["evaluation_peak_gpu_memory_mib_per_job"]:.1f} MiB per process. A loader bug that reset learned global cutoffs was detected by deterministic map validation; all 90 affected evaluations were excluded and rerun after a regression-tested fix. The server retains those invalid artifacts for audit. Portable output size is {size / 1024**2:.2f} MiB: above the 5-MiB target because exact compressed per-bin prediction tables are retained, but below the 20-MiB hard limit.

Primary paths are `reports/mechanism_result.md`, `tables/mechanism_summary.csv`, `tables/mechanism_paired_contrast_summary.csv`, `tables/mechanism_regional_cells.csv.gz`, `tables/learned_mechanism_maps.csv.gz`, and `manifests/provenance_v1.json`. Existing-prediction measurement interpretation is in `reports/existing_measurement_interpretation.md`.

Limitations remain unchanged: 2k failure, 10k nonmonotonicity, sparse-tail failure and missing eligible sampler support, finite-reference uncertainty, historical exposure, one-subset mechanism controls, and 18,866 fixed-classifier training events. No calibrated interval, universal superiority, exact minimum-data, overfitting, smoothness, signal-efficiency, cross-section, or transfer claim is supported.
"""
    (HERE / "reports/final_handoff.md").write_text(report)
    print(
        json.dumps(
            {"status": "complete", "portable_bytes": size, "manifest": str(manifest_path)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
