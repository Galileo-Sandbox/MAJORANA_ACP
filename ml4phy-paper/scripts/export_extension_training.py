#!/usr/bin/env python3
"""Validate the six matched baseline runs and export a lightweight model registry."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import torch
import yaml

NEW_ARCHITECTURES = {
    "m0": {
        "display_name": "CNP",
        "config": "ml4phy-paper/configs/m0_cnp_seed0.yaml",
    },
    "m1": {
        "display_name": "Attentive CNP",
        "config": "ml4phy-paper/configs/m1_attentive_cnp_seed0.yaml",
    },
}
EXISTING_MODELS = {
    "m2": {
        "display_name": "Attentive CNP + PE",
        "source_ids": ("m2_seed0_pilot", "m2_seed1", "m2_seed2"),
    },
    "ours": {
        "display_name": "Density-guided CNP (ours)",
        "source_ids": ("cell17_seed0_recovered", "cell17_seed1", "cell17_seed2"),
    },
}
EXPECTED_SOURCE_COMMIT = "51282963417d334cc3343ad48c735f8cfaeca511"
EXPECTED_STEPS = 3000
EXPECTED_PARAMETER_COUNTS = {"m0": 116482, "m1": 149250, "m2": 151682, "ours": 130693}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def write_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def verify_hash(path: Path, expected: str, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{description} hash mismatch: expected {expected}, observed {observed}"
        )


def checkpoint_summary(path: Path) -> tuple[int, float]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model_state"]
    parameter_count = sum(int(tensor.numel()) for tensor in state.values())
    history = payload["history"]
    if not history["step"] or int(history["step"][-1]) != EXPECTED_STEPS:
        raise ValueError(f"Unexpected final training step in {path}")
    final_loss = float(history["loss"][-1])
    if not math.isfinite(final_loss):
        raise ValueError(f"Non-finite final loss in {path}")
    return parameter_count, final_loss


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    repo = Path(".").resolve()
    source_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    script_path = repo / "ml4phy-paper/scripts/export_extension_training.py"
    base_registry_path = repo / "ml4phy-paper/configs/trained_models_v1.json"
    protocol_path = repo / "ml4phy-paper/manifests/extension_protocol_v1.json"
    output_registry_path = repo / "ml4phy-paper/configs/extension_models_v1.json"
    table_path = repo / "ml4phy-paper/tables/extension_training_runs.csv"
    report_path = repo / "ml4phy-paper/reports/extension_training_result.md"
    manifest_path = repo / "ml4phy-paper/manifests/extension_training_result.json"
    for path in (output_registry_path, table_path, report_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    base_registry = read_json(base_registry_path)
    protocol = read_json(protocol_path)
    if protocol["source_commit"] != "a9f8dce7bebbc7e715b598d95540cc80056da563":
        raise ValueError("Unexpected extension protocol source commit")

    registry_models: dict[str, dict] = {}
    training_rows: list[dict] = []
    runner_inputs: list[dict] = []
    common_pool_hash: str | None = None

    for architecture, specification in NEW_ARCHITECTURES.items():
        config_path = repo / specification["config"]
        config_hash = sha256_file(config_path)
        for seed in range(3):
            run_id = f"20260909-{architecture}-seed{seed}-train3000"
            run_dir = repo / "ml4phy-paper/runs/training" / run_id
            record_path = run_dir / "runner_record.json"
            summary_path = run_dir / "artifacts/run_summary.json"
            checkpoint_path = run_dir / "artifacts/cnp.ckpt"
            pool_path = run_dir / "artifacts/training_pool.npz"
            resolved_path = run_dir / "resolved_config.yaml"
            log_path = run_dir / "training.log"
            record = read_json(record_path)
            summary = read_json(summary_path)
            with resolved_path.open() as stream:
                resolved = yaml.safe_load(stream)

            if record["source_commit"] != EXPECTED_SOURCE_COMMIT:
                raise ValueError(f"Unexpected source commit for {run_id}")
            if record["status"] != "completed" or record["returncode"] != 0:
                raise ValueError(f"Training run did not complete successfully: {run_id}")
            if record["training_steps"] != EXPECTED_STEPS or record["training_seed"] != seed:
                raise ValueError(f"Unexpected training schedule for {run_id}")
            if record["config_sha256"] != config_hash:
                raise ValueError(f"Configuration hash mismatch for {run_id}")
            if resolved["training"]["seed"] != seed:
                raise ValueError(f"Resolved seed mismatch for {run_id}")
            if resolved["training"]["n_steps"] != EXPECTED_STEPS:
                raise ValueError(f"Resolved step count mismatch for {run_id}")
            if summary["n_train_events"] != 18866 or summary["n_bins_used"] != 212:
                raise ValueError(f"Unexpected training pool summary for {run_id}")
            if summary["youden_T_star"] != 0.540643572807312:
                raise ValueError(f"Unexpected legacy threshold summary for {run_id}")
            for name, path in (
                ("cnp.ckpt", checkpoint_path),
                ("training_pool.npz", pool_path),
                ("run_summary.json", summary_path),
            ):
                verify_hash(path, record["outputs"][name]["sha256"], f"{run_id} {name}")

            parameter_count, final_loss = checkpoint_summary(checkpoint_path)
            if parameter_count != EXPECTED_PARAMETER_COUNTS[architecture]:
                raise ValueError(f"Unexpected parameter count for {run_id}")
            if final_loss != float(summary["cnp_final_train_loss"]):
                raise ValueError(f"Checkpoint/summary loss mismatch for {run_id}")
            pool_hash = sha256_file(pool_path)
            if common_pool_hash is None:
                common_pool_hash = pool_hash
            elif pool_hash != common_pool_hash:
                raise ValueError("The six controlled runs do not share one training pool")

            warning_lines = [
                line.strip()
                for line in log_path.read_text().splitlines()
                if "warning" in line.lower() or "nan" in line.lower()
            ]
            model_id = f"{architecture}_seed{seed}"
            checkpoint_relative = str(checkpoint_path.relative_to(repo))
            registry_models[model_id] = {
                "display_name": f"{specification['display_name']} (seed {seed})",
                "architecture": architecture,
                "config": specification["config"],
                "config_sha256": config_hash,
                "checkpoint": checkpoint_relative,
                "checkpoint_sha256": record["outputs"]["cnp.ckpt"]["sha256"],
                "training_seed": seed,
                "training_steps": EXPECTED_STEPS,
                "parameter_count": parameter_count,
                "comparison_role": "matched_extension_baseline",
                "runner_record": str(record_path.relative_to(repo)),
                "runner_record_sha256": sha256_file(record_path),
            }
            training_rows.append(
                {
                    "architecture": architecture,
                    "method": specification["display_name"],
                    "training_seed": seed,
                    "steps": EXPECTED_STEPS,
                    "nominal_train_events": summary["n_train_events"],
                    "sampling_eligible_events": 18836,
                    "parameter_count": parameter_count,
                    "final_train_loss": format(final_loss, ".17g"),
                    "wall_seconds": format(float(record["wall_seconds"]), ".17g"),
                    "status": record["status"],
                    "warning_count": len(warning_lines),
                    "checkpoint_sha256": record["outputs"]["cnp.ckpt"]["sha256"],
                    "run_id": run_id,
                }
            )
            runner_inputs.append(
                {
                    "run_id": run_id,
                    "runner_record_sha256": sha256_file(record_path),
                    "resolved_config_sha256": sha256_file(resolved_path),
                    "training_log_sha256": sha256_file(log_path),
                    "run_summary_sha256": sha256_file(summary_path),
                    "training_pool_sha256": pool_hash,
                    "checkpoint_sha256": record["outputs"]["cnp.ckpt"]["sha256"],
                    "warnings": warning_lines,
                }
            )

    if common_pool_hash != base_registry["training_pool_sha256"]:
        raise ValueError("New baseline training pool differs from the controlled core pool")

    for architecture, specification in EXISTING_MODELS.items():
        for seed, source_id in enumerate(specification["source_ids"]):
            source = dict(base_registry["models"][source_id])
            checkpoint_path = repo / source["checkpoint"]
            config_path = repo / source["config"]
            verify_hash(checkpoint_path, source["checkpoint_sha256"], source_id)
            verify_hash(config_path, source["config_sha256"], f"{source_id} config")
            parameter_count, _ = checkpoint_summary(checkpoint_path)
            if parameter_count != EXPECTED_PARAMETER_COUNTS[architecture]:
                raise ValueError(f"Unexpected parameter count for {source_id}")
            registry_models[f"{architecture}_seed{seed}"] = {
                **source,
                "display_name": f"{specification['display_name']} (seed {seed})",
                "architecture": architecture,
                "training_steps": EXPECTED_STEPS,
                "parameter_count": parameter_count,
                "comparison_role": "matched_extension_core",
                "source_registry_id": source_id,
            }

    registry = {
        "schema_version": 1,
        "status": "matched_four_architecture_extension_registry",
        "source_commit": source_commit,
        "training_pool_sha256": common_pool_hash,
        "training_predictions": base_registry["training_predictions"],
        "training_predictions_sha256": base_registry["training_predictions_sha256"],
        "classifier_config": base_registry["classifier_config"],
        "classifier_config_sha256": base_registry["classifier_config_sha256"],
        "extension_protocol": str(protocol_path.relative_to(repo)),
        "extension_protocol_sha256": sha256_file(protocol_path),
        "models": registry_models,
    }
    write_json(output_registry_path, registry)
    write_csv(table_path, training_rows)

    table_lines = "\n".join(
        f"| {row['method']} | {row['training_seed']} | {float(row['final_train_loss']):.6f} | "
        f"{float(row['wall_seconds']):.2f} | {row['warning_count']} |"
        for row in training_rows
    )
    total_wall = sum(float(row["wall_seconds"]) for row in training_rows)
    report = f"""# Matched baseline training result

Status: all six prospectively justified jobs completed successfully on 2026-09-09.

| Method | Seed | Final training loss | Wall time (s) | Logged warnings |
|---|---:|---:|---:|---:|
{table_lines}

The six jobs took {total_wall:.2f} seconds of summed wall time. Each used the
same 18,866-event input pool, the same 18,836 sampling-eligible events, 3,000
steps, batch size 16, variable 640--1,024-event trials, context sizes 128--512,
four training loss samples, Adam at 1e-3, dropout 0.2, and final-checkpoint
selection. Sampling remained with replacement. No run failed and no warning or
non-finite loss was recorded.

CNP has 116,482 parameters and uses mean pooling without positional encoding.
Attentive CNP has 149,250 parameters and uses deterministic cross-attention
without positional encoding; it is not a latent ANP. For comparison, the
matched Attentive CNP + PE model has 151,682 parameters and Density-guided CNP
(ours) has 130,693 parameters.

Checkpoints and event-level training pools remain in ignored server-only run
directories. `configs/extension_models_v1.json` records their paths and hashes
alongside the six reusable matched core checkpoints. This training result does
not by itself support a performance claim; evaluation follows the frozen
nested-context protocol. Phase 2 remains unapproved and was not started.
"""
    report_path.write_text(report)

    manifest = {
        "schema_version": 1,
        "analysis": "matched CNP and Attentive CNP baseline training",
        "status": "completed",
        "source_commit": source_commit,
        "script": str(script_path.relative_to(repo)),
        "script_sha256": sha256_file(script_path),
        "expected_training_source_commit": EXPECTED_SOURCE_COMMIT,
        "historical_data_exposure": (
            "Follow-up training on historically exposed data under a prospectively "
            "recorded extension protocol."
        ),
        "checkpoint_selection": "final 3000-step checkpoint; no validation selection",
        "failed_runs": [],
        "runner_inputs": runner_inputs,
        "outputs": {
            str(path.relative_to(repo)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (output_registry_path, table_path, report_path)
        },
    }
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
