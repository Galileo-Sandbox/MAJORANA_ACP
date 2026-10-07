#!/usr/bin/env python3
"""Freeze outcome-blind nested Phase 2 training subsets and provenance."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np

IDENTITY_FIELDS = ("run_number", "detector", "id", "tp0")
BUDGETS = (2000, 5000)
ARCHITECTURES = ("m0", "m1", "m2", "ours")
TRAINING_SEEDS = (0, 1, 2)
CONTEXT_SEEDS = tuple(range(100, 110))
NAMESPACE = b"ml4ps-phase2-training-order-v1\0"
BASE_CONFIGS = {
    "m0": "ml4phy-paper/configs/m0_cnp_seed0.yaml",
    "m1": "ml4phy-paper/configs/m1_attentive_cnp_seed0.yaml",
    "m2": "ml4phy-paper/configs/m2_attentive_pe10_seed0.yaml",
    "ours": "ml4phy-paper/configs/cell17_controlled_seed0.yaml",
}
METHOD_NAMES = {
    "m0": "CNP",
    "m1": "Attentive CNP",
    "m2": "Attentive CNP + PE",
    "ours": "Density-guided CNP (ours)",
}
REGIONS = {
    "DEP": (1577.0, 1606.0),
    "Bi-212 1620.74 keV": (1606.0, 1635.0),
    "continuum 1700-2000 keV": (1700.0, 2000.0),
    "SE": (2088.0, 2118.0),
    "continuum 2200-2400 keV": (2200.0, 2400.0),
    "FE": (2599.0, 2629.0),
    "sparse tail": (2700.0, 3000.0),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_hash(identities: np.ndarray) -> str:
    order = np.lexsort(
        tuple(identities[:, index] for index in reversed(range(identities.shape[1])))
    )
    payload = np.ascontiguousarray(identities[order], dtype="<i8").tobytes()
    return hashlib.sha256(payload).hexdigest()


def priority(identity: np.ndarray) -> bytes:
    payload = np.ascontiguousarray(identity, dtype="<i8").tobytes()
    return hashlib.sha256(NAMESPACE + payload).digest()


def effective_pool(energy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(500.0, 3000.0 + 5.0, 10.0, dtype=np.float64)
    bin_index = np.clip(np.digitize(energy, edges) - 1, 0, edges.size - 2)
    counts = np.bincount(bin_index, minlength=edges.size - 1)
    kept_bins = counts >= 4
    return kept_bins[bin_index], counts


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    repo = Path(".").resolve()
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--short"], text=True
    ).splitlines()
    if status:
        raise RuntimeError("Phase 2 freezing requires a clean worktree: " + "; ".join(status))
    source_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    root = repo / "ml4phy-paper"
    source_path = repo / "runs/small_data_configs/simple_cnn_small/eval_train/predictions.h5"
    local_dir = root / "local/phase2/protocol-v1"
    output_paths = {
        "config": root / "configs/phase2_training_v1.json",
        "ledger": root / "tables/phase2_training_subset_ledger.csv",
        "support": root / "tables/phase2_training_subset_region_support.csv",
        "report": root / "reports/phase2_protocol_freeze.md",
        "manifest": root / "manifests/phase2_protocol_v1.json",
        "order": local_dir / "training_order_rows.npy",
        **{
            f"subset_{budget}": local_dir / f"training_budget_{budget}.h5"
            for budget in BUDGETS
        },
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))
    expected_source_hash = "ea2eb5594acdd181ebc18926fd8deabc7b2f6910b117dec863d87c3b70057159"
    if sha256_file(source_path) != expected_source_hash:
        raise ValueError("Full acceptance-input HDF5 hash mismatch")

    with h5py.File(source_path, "r") as source:
        arrays = {name: source[name][:] for name in source}
    identities = np.column_stack(
        [arrays[name].astype("<i8", copy=False) for name in IDENTITY_FIELDS]
    )
    if identities.shape != (18866, 4) or np.unique(identities, axis=0).shape[0] != 18866:
        raise ValueError("Unexpected full-pool identity count or duplicate identity")
    priorities = [priority(identity) for identity in identities]
    order = np.asarray(
        sorted(
            range(identities.shape[0]),
            key=lambda index: (
                priorities[index],
                tuple(int(value) for value in identities[index]),
            ),
        ),
        dtype=np.int64,
    )
    if np.unique(order).size != 18866:
        raise ValueError("Outcome-blind ordering is not a permutation")

    local_dir.mkdir(parents=True, exist_ok=False)
    np.save(output_paths["order"], order)
    order_hash = hashlib.sha256(
        np.ascontiguousarray(identities[order], dtype="<i8").tobytes()
    ).hexdigest()
    ledger_rows = []
    support_rows = []
    subset_specs = []
    for budget in BUDGETS:
        selected_rows = order[:budget]
        subset_path = output_paths[f"subset_{budget}"]
        with h5py.File(subset_path, "x") as target:
            for name, values in arrays.items():
                target.create_dataset(name, data=values[selected_rows], track_times=False)
        subset_identities = identities[selected_rows]
        energy = arrays["energy"][selected_rows].astype(np.float64)
        effective_mask, bin_counts = effective_pool(energy)
        effective_identities = subset_identities[effective_mask]
        spec = {
            "budget": budget,
            "logical_path": str(subset_path.relative_to(repo)),
            "bytes": subset_path.stat().st_size,
            "sha256": sha256_file(subset_path),
            "identity_sha256": identity_hash(subset_identities),
            "effective_identity_sha256": identity_hash(effective_identities),
            "nominal_unique_events": budget,
            "sampling_eligible_unique_events": int(effective_mask.sum()),
            "sampling_excluded_unique_events": int((~effective_mask).sum()),
            "kept_bin_count": int((bin_counts >= 4).sum()),
            "nonempty_bin_count": int((bin_counts > 0).sum()),
            "density_pool_unique_events": budget,
            "density_pool_identity_sha256": identity_hash(subset_identities),
        }
        subset_specs.append(spec)
        ledger_rows.append(
            {
                "training_budget": budget,
                "nominal_unique_events": budget,
                "sampling_eligible_unique_events": spec["sampling_eligible_unique_events"],
                "sampling_excluded_unique_events": spec["sampling_excluded_unique_events"],
                "nonempty_10kev_bins": spec["nonempty_bin_count"],
                "kept_minimum_four_event_bins": spec["kept_bin_count"],
                "density_pool_unique_events": budget,
                "nominal_identity_sha256": spec["identity_sha256"],
                "effective_identity_sha256": spec["effective_identity_sha256"],
                "density_identity_sha256": spec["density_pool_identity_sha256"],
                "subset_file_sha256": spec["sha256"],
            }
        )
        for region, (lower, upper) in REGIONS.items():
            in_region = (energy >= lower) & (energy < upper)
            support_rows.append(
                {
                    "training_budget": budget,
                    "region": region,
                    "nominal_events": int(in_region.sum()),
                    "sampling_eligible_events": int((in_region & effective_mask).sum()),
                    "region_definition_kev": f"[{lower}, {upper})",
                    "target_region_removed": False,
                }
            )

    if not set(order[: BUDGETS[0]]).issubset(set(order[: BUDGETS[1]])):
        raise ValueError("Training subsets are not nested")
    write_csv(output_paths["ledger"], ledger_rows)
    write_csv(output_paths["support"], support_rows)

    base_config_hashes = {
        architecture: sha256_file(repo / relative)
        for architecture, relative in BASE_CONFIGS.items()
    }
    config = {
        "schema_version": 1,
        "analysis": "approved Phase 2 acceptance-training-size slice",
        "status": "frozen",
        "budgets": list(BUDGETS),
        "reused_full_budget": 18866,
        "architectures": list(ARCHITECTURES),
        "method_names": METHOD_NAMES,
        "training_seeds": list(TRAINING_SEEDS),
        "context_size": 500,
        "context_seeds": list(CONTEXT_SEEDS),
        "training_steps": 3000,
        "base_configs": BASE_CONFIGS,
        "base_config_sha256": base_config_hashes,
        "subset_inputs": subset_specs,
        "fixed_classifier": "configs/small_data_configs/simple_cnn_small.yaml",
        "fixed_threshold": 0.540643572807312,
        "fixed_final_target_events": 114400,
        "dropout_seed": 10100,
        "mc_passes": 50,
        "new_training_job_count": 24,
        "new_evaluation_cell_count": 240,
    }
    with output_paths["config"].open("x") as stream:
        json.dump(config, stream, indent=2, allow_nan=False)
        stream.write("\n")

    rows_text = "\n".join(
        f"| {row['training_budget']:,} | {row['sampling_eligible_unique_events']:,} | "
        f"{row['sampling_excluded_unique_events']:,} | {row['kept_minimum_four_event_bins']:,} |"
        for row in ledger_rows
    )
    report = f"""# Phase 2 protocol freeze

Status: frozen after explicit approval on 2026-09-09; no training was started by this freeze step.

The two laptop-only planning documents were not present in the server checkout at commit `{source_commit}`. The approved user instruction, Phase 1 handoff, existing frozen protocol, and preserved repository plans therefore govern this execution.

## Outcome-blind nested training pools

The ordering is determined only from the four event identity fields (`run_number`, `detector`, `id`, and `tp0`). Each identity receives SHA-256 priority under the fixed namespace `ml4ps-phase2-training-order-v1`; labels, classifier scores, energies, and final outcomes do not enter the ordering. The 2,000-event pool is the exact prefix of the 5,000-event pool. The 18,866-event budget reuses the completed Phase 1 models.

| Nominal budget | Sampling-eligible events | Excluded events | Kept 10-keV bins |
|---:|---:|---:|---:|
{rows_text}

Minimum-four-event filtering affects only which input events can be sampled during training. It does not change any final target identity or evaluation-region definition. Training remains with replacement for 3,000 steps, so this is a fixed-compute budget study.

For Density-guided CNP, the configured `train_predictions_path` will point to the matching subset HDF5. Its density pool is therefore exactly the nominal prefix ({BUDGETS[0]:,} or {BUDGETS[1]:,} identities), including events excluded by the minimum-bin sampler. The runner must verify this path and identity hash before training, and evaluation must rebuild from the same resolved configuration.

The server-only subset HDF5 files and row ordering remain under `ml4phy-paper/local/phase2/protocol-v1/`. Portable ledgers disclose counts, region support, overlaps, and hashes without exporting identities or event-level predictions.
"""
    output_paths["report"].write_text(report)

    portable_outputs = [
        output_paths["config"],
        output_paths["ledger"],
        output_paths["support"],
        output_paths["report"],
    ]
    manifest = {
        "schema_version": 1,
        "analysis": "Phase 2 outcome-blind nested training-pool freeze",
        "status": "frozen_approved_not_started",
        "source_commit": source_commit,
        "script": str(Path(__file__).resolve().relative_to(repo)),
        "script_sha256": sha256_file(Path(__file__)),
        "approval_basis": "User explicitly approved Phase 2 after reviewing commit 2800bb9.",
        "ordering": {
            "identity_fields": list(IDENTITY_FIELDS),
            "namespace": NAMESPACE.rstrip(b"\0").decode(),
            "rule": "ascending SHA-256(namespace || little-endian int64 identity), identity tuple tie-break",
            "ordered_identity_sha256": order_hash,
            "server_only_rows_path": str(output_paths["order"].relative_to(repo)),
            "server_only_rows_sha256": sha256_file(output_paths["order"]),
        },
        "full_input": {
            "logical_path": str(source_path.relative_to(repo)),
            "bytes": source_path.stat().st_size,
            "sha256": expected_source_hash,
            "unique_events": 18866,
            "identity_sha256": identity_hash(identities),
        },
        "subsets": subset_specs,
        "overlaps": {
            "intersection_2000_5000": 2000,
            "union_2000_5000": 5000,
            "nested_prefix_verified": True,
        },
        "density_pool_rule": "exact nominal subset; never reconstruct the full 18,866-event pool for smaller budgets",
        "final_target_regions_changed": False,
        "outputs": {
            str(path.relative_to(repo)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in portable_outputs
        },
        "historical_data_exposure": "Prospectively specified follow-up analysis on historically exposed data; not an untouched test.",
    }
    with output_paths["manifest"].open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
