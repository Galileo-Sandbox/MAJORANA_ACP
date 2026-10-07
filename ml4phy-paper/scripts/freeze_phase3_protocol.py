#!/usr/bin/env python3
"""Freeze the approved Phase 3 training subsets and resource ledger."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np

IDENTITY_FIELDS = ("run_number", "detector", "id", "tp0")
ORDERING_SEEDS = (20260910, 20260911)
ARCHITECTURES = ("m0", "m1", "m2", "ours")
TRAINING_SEEDS = (0, 1, 2)
CONTEXT_SEEDS = tuple(range(100, 110))
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


def read_json(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def identity_matrix(arrays: dict[str, np.ndarray], rows: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [arrays[field][rows].astype("<i8", copy=False) for field in IDENTITY_FIELDS]
    )


def identity_hash(identities: np.ndarray, *, preserve_order: bool = False) -> str:
    if preserve_order:
        ordered = identities
    else:
        order = np.lexsort(
            tuple(identities[:, index] for index in reversed(range(identities.shape[1])))
        )
        ordered = identities[order]
    return hashlib.sha256(np.ascontiguousarray(ordered, dtype="<i8").tobytes()).hexdigest()


def identity_set(identities: np.ndarray) -> set[tuple[int, ...]]:
    return {tuple(int(value) for value in row) for row in identities}


def seeded_priority(identity: np.ndarray, seed: int) -> bytes:
    namespace = f"ml4ps-phase3-training-order-v1:seed={seed}\0".encode()
    payload = np.ascontiguousarray(identity, dtype="<i8").tobytes()
    return hashlib.sha256(namespace + payload).digest()


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


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    repo = Path(".").resolve()
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--short"], text=True
    ).splitlines()
    if status:
        raise RuntimeError("Phase 3 freeze requires a clean worktree: " + "; ".join(status))
    source_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    root = repo / "ml4phy-paper"
    source_path = repo / "runs/small_data_configs/simple_cnn_small/eval_train/predictions.h5"
    phase2_manifest_path = root / "manifests/phase2_protocol_v1.json"
    base_protocol_path = root / "manifests/frozen_protocol_v1.json"
    base_roles_path = root / "local/protocol/frozen_roles_v1.npz"
    original_order_path = root / "local/phase2/protocol-v1/training_order_rows.npy"
    phase2_training_runs_path = root / "tables/phase2_training_runs.csv"
    phase2_evaluation_runs_path = root / "tables/phase2_evaluation_runs.csv"
    local_dir = root / "local/phase3/protocol-v1"
    output_paths = {
        "config": root / "configs/phase3_training_v1.json",
        "summary": root / "tables/phase3_training_subset_summary.csv",
        "support": root / "tables/phase3_training_subset_region_support.csv",
        "overlaps": root / "tables/phase3_training_subset_overlaps.csv",
        "report": root / "reports/phase3_protocol.md",
        "manifest": root / "manifests/phase3_protocol_v1.json",
        "original_10k": local_dir / "training_original_n10000.h5",
        **{
            f"order_{seed}": local_dir / f"training_order_seed{seed}_rows.npy"
            for seed in ORDERING_SEEDS
        },
        **{
            f"seed_{seed}_5k": local_dir / f"training_seed{seed}_n5000.h5"
            for seed in ORDERING_SEEDS
        },
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))
    phase2 = read_json(phase2_manifest_path)
    base = read_json(base_protocol_path)
    if sha256_file(source_path) != phase2["full_input"]["sha256"]:
        raise ValueError("Full acceptance-training HDF5 hash mismatch")
    if sha256_file(original_order_path) != phase2["ordering"]["server_only_rows_sha256"]:
        raise ValueError("Original Phase 2 ordering hash mismatch")
    if sha256_file(base_roles_path) != base["local_role_archive"]["sha256"]:
        raise ValueError("Frozen role archive hash mismatch")

    with h5py.File(source_path, "r") as source:
        arrays = {name: source[name][:] for name in source}
    identities = identity_matrix(arrays, np.arange(arrays[IDENTITY_FIELDS[0]].size))
    if identities.shape != (18866, 4) or np.unique(identities, axis=0).shape[0] != 18866:
        raise ValueError("Unexpected parent-pool identities")
    parent_set = identity_set(identities)
    original_order = np.load(original_order_path).astype(np.int64)
    if np.unique(original_order).size != 18866:
        raise ValueError("Original ordering is not a complete permutation")
    if (
        identity_hash(identities[original_order], preserve_order=True)
        != phase2["ordering"]["ordered_identity_sha256"]
    ):
        raise ValueError("Original ordered identity hash mismatch")
    for spec in phase2["subsets"]:
        expected = original_order[: spec["budget"]]
        if identity_hash(identities[expected]) != spec["identity_sha256"]:
            raise ValueError(f"Original {spec['budget']} prefix changed")

    final_path = repo / base["inputs"]["full_test"]["logical_path"]
    if sha256_file(final_path) != base["inputs"]["full_test"]["sha256"]:
        raise ValueError("Final HDF5 hash mismatch")
    with np.load(base_roles_path) as roles:
        final_target_rows = roles["final_target_rows"].astype(np.int64)
        final_context_rows = roles["final_context_reservoir_rows"].astype(np.int64)
    with h5py.File(final_path, "r") as handle:
        final_arrays = {field: handle[field][:] for field in IDENTITY_FIELDS}
    target_ids = identity_matrix(final_arrays, final_target_rows)
    context_ids = identity_matrix(final_arrays, final_context_rows)
    target_set = identity_set(target_ids)
    context_set = identity_set(context_ids)
    if parent_set & target_set or parent_set & context_set:
        raise ValueError("Acceptance-training identities overlap final target/context identities")

    local_dir.mkdir(parents=True, exist_ok=False)
    orders: dict[str, np.ndarray] = {"original": original_order}
    order_specs = {
        "original": {
            "ordering_seed": None,
            "namespace": phase2["ordering"]["namespace"],
            "rule": phase2["ordering"]["rule"],
            "rows_path": str(original_order_path.relative_to(repo)),
            "rows_sha256": sha256_file(original_order_path),
            "ordered_identity_sha256": phase2["ordering"]["ordered_identity_sha256"],
            "reused_without_regeneration": True,
        }
    }
    for seed in ORDERING_SEEDS:
        priorities = [seeded_priority(identity, seed) for identity in identities]
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
            raise ValueError(f"Seed {seed} ordering is not a permutation")
        order_path = output_paths[f"order_{seed}"]
        np.save(order_path, order)
        orders[f"seed{seed}"] = order
        order_specs[f"seed{seed}"] = {
            "ordering_seed": seed,
            "namespace": f"ml4ps-phase3-training-order-v1:seed={seed}",
            "rule": "ascending SHA-256(namespace || NUL || little-endian int64 identity), identity tuple tie-break",
            "rows_path": str(order_path.relative_to(repo)),
            "rows_sha256": sha256_file(order_path),
            "ordered_identity_sha256": identity_hash(identities[order], preserve_order=True),
            "reused_without_regeneration": False,
        }

    subset_requests = (
        ("original", 10000, output_paths["original_10k"]),
        ("seed20260910", 5000, output_paths["seed_20260910_5k"]),
        ("seed20260911", 5000, output_paths["seed_20260911_5k"]),
    )
    subset_specs = []
    summary_rows = []
    support_rows = []
    subset_sets: dict[str, set[tuple[int, ...]]] = {
        "original_n5000": identity_set(identities[original_order[:5000]])
    }
    for ordering_id, budget, subset_path in subset_requests:
        selected_rows = orders[ordering_id][:budget]
        with h5py.File(subset_path, "x") as target:
            for name, values in arrays.items():
                target.create_dataset(name, data=values[selected_rows], track_times=False)
        selected_ids = identities[selected_rows]
        energy = arrays["energy"][selected_rows].astype(np.float64)
        effective_mask, bin_counts = effective_pool(energy)
        subset_id = f"{ordering_id}_n{budget}"
        subset_sets[subset_id] = identity_set(selected_ids)
        spec = {
            "subset_id": subset_id,
            "ordering_id": ordering_id,
            "ordering_seed": order_specs[ordering_id]["ordering_seed"],
            "budget": budget,
            "logical_path": str(subset_path.relative_to(repo)),
            "bytes": subset_path.stat().st_size,
            "sha256": sha256_file(subset_path),
            "identity_sha256": identity_hash(selected_ids),
            "effective_identity_sha256": identity_hash(selected_ids[effective_mask]),
            "nominal_unique_events": budget,
            "sampling_eligible_unique_events": int(effective_mask.sum()),
            "sampling_excluded_unique_events": int((~effective_mask).sum()),
            "kept_bin_count": int((bin_counts >= 4).sum()),
            "nonempty_bin_count": int((bin_counts > 0).sum()),
            "density_pool_unique_events": budget,
            "density_pool_identity_sha256": identity_hash(selected_ids),
            "overlap_with_final_target": len(subset_sets[subset_id] & target_set),
            "overlap_with_final_context_reservoir": len(subset_sets[subset_id] & context_set),
        }
        subset_specs.append(spec)
        summary_rows.append(
            {
                "subset_id": subset_id,
                "ordering_seed": spec["ordering_seed"],
                "nominal_unique_events": budget,
                "sampling_eligible_unique_events": spec["sampling_eligible_unique_events"],
                "sampling_excluded_unique_events": spec["sampling_excluded_unique_events"],
                "nonempty_10kev_bins": spec["nonempty_bin_count"],
                "kept_minimum_four_event_bins": spec["kept_bin_count"],
                "density_pool_unique_events": budget,
                "overlap_with_final_target": spec["overlap_with_final_target"],
                "overlap_with_final_context_reservoir": spec[
                    "overlap_with_final_context_reservoir"
                ],
                "nominal_identity_sha256": spec["identity_sha256"],
                "effective_identity_sha256": spec["effective_identity_sha256"],
                "subset_file_sha256": spec["sha256"],
            }
        )
        for region, (lower, upper) in REGIONS.items():
            in_region = (energy >= lower) & (energy < upper)
            support_rows.append(
                {
                    "subset_id": subset_id,
                    "ordering_seed": spec["ordering_seed"],
                    "training_budget": budget,
                    "region": region,
                    "nominal_events": int(in_region.sum()),
                    "sampling_eligible_events": int((in_region & effective_mask).sum()),
                    "region_definition_kev": f"[{lower}, {upper})",
                    "target_region_removed": False,
                }
            )

    original_10k = subset_sets["original_n10000"]
    if not subset_sets["original_n5000"] <= original_10k:
        raise ValueError("Original 5k prefix is not nested in original 10k")
    if identity_hash(identities[original_order[:5000]]) != phase2["subsets"][1]["identity_sha256"]:
        raise ValueError("Original Phase 2 5k prefix was not preserved")
    overlap_rows = []
    names = sorted(subset_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index:]:
            intersection = len(subset_sets[left] & subset_sets[right])
            overlap_rows.append(
                {
                    "left_subset": left,
                    "right_subset": right,
                    "intersection_unique_events": intersection,
                    "union_unique_events": len(subset_sets[left] | subset_sets[right]),
                    "left_fraction_in_intersection": intersection / len(subset_sets[left]),
                    "right_fraction_in_intersection": intersection / len(subset_sets[right]),
                }
            )
    three_5k = [
        subset_sets["original_n5000"],
        subset_sets["seed20260910_n5000"],
        subset_sets["seed20260911_n5000"],
    ]
    five_k_union = set().union(*three_5k)
    five_k_triple = set.intersection(*three_5k)

    write_csv(output_paths["summary"], summary_rows)
    write_csv(output_paths["support"], support_rows)
    write_csv(output_paths["overlaps"], overlap_rows)
    base_config_hashes = {
        architecture: sha256_file(repo / relative)
        for architecture, relative in BASE_CONFIGS.items()
    }

    measured_5k_training_seconds = sum(
        float(row["wall_seconds"])
        for row in csv_rows(phase2_training_runs_path)
        if int(row["training_budget"]) == 5000
    )
    measured_5k_evaluation_seconds = sum(
        float(row["wall_seconds"])
        for row in csv_rows(phase2_evaluation_runs_path)
        if "-b5000-" in row["run_id"]
    )
    if measured_5k_training_seconds <= 0 or measured_5k_evaluation_seconds <= 0:
        raise ValueError("Missing Phase 2 5k timing evidence")
    ten_k_scaling_factor = 1.5
    training_seconds = (2.0 + ten_k_scaling_factor) * measured_5k_training_seconds
    evaluation_seconds = (2.0 + ten_k_scaling_factor) * measured_5k_evaluation_seconds
    projected_seconds = 1.25 * (training_seconds + evaluation_seconds)
    config = {
        "schema_version": 1,
        "analysis": "approved Phase 3 acceptance-training data-efficiency confirmation",
        "status": "frozen",
        "new_subsets": [item["subset_id"] for item in subset_specs],
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
        "new_training_job_count": 36,
        "new_evaluation_cell_count": 360,
        "neural_hard_limit_seconds": 7200,
        "conservative_projected_seconds": projected_seconds,
    }
    with output_paths["config"].open("x") as stream:
        json.dump(config, stream, indent=2, allow_nan=False)
        stream.write("\n")

    table = "\n".join(
        f"| {row['subset_id']} | {row['nominal_unique_events']:,} | "
        f"{row['sampling_eligible_unique_events']:,} | "
        f"{row['sampling_excluded_unique_events']:,} | "
        f"{row['kept_minimum_four_event_bins']:,} |"
        for row in summary_rows
    )
    report = f"""# Phase 3 protocol and resource gate

Status: frozen after explicit author approval; no Phase 3 fit or neural
training was started by this step. Source commit: `{source_commit}`.

## Reuse and required work

The original Phase 2 5k subset, its 12 neural checkpoints, and 120 matching
evaluation cells are reusable as the first of three 5k subset realizations.
The original 2k and 18.9k results remain unchanged. Phase 3 adds exactly 36
training jobs and 360 neural evaluation cells: two new 5k orderings plus the
10k prefix of the original ordering.

The dense-GP pilot is reusable for timing but not as a campaign cell: its
explicit optimizer seed 31000 differs from the frozen campaign seed rule,
which gives 31400 for the n=2,000, context-seed-100 cell. The required matrix
therefore remains 80 development fits and 40 final fits. Its existing
scaling-aware estimate is 8,662.74 seconds (2.41 hours), below the separately
approved four-hour hard limit.

## Frozen neural subsets

New orderings retain the Phase 2 outcome-blind SHA-256 priority algorithm but
encode the approved numeric ordering seed in a distinct namespace. The 10k
subset reuses the original Phase 2 row ordering byte-for-byte; the original 2k
and 5k prefixes were verified and were not regenerated.

| Subset | Nominal events | Sampling-eligible | Excluded | Kept 10-keV bins |
|---|---:|---:|---:|---:|
{table}

All three new subsets have zero identity overlap with the fixed 114,400-event
final target and the 20,000-event final context reservoir. Each density buffer
must use its exact matching nominal subset. No target region is removed when a
training subset lacks support.

The three 5k subsets have a union of {len(five_k_union):,} unique events and a
three-way intersection of {len(five_k_triple):,}. Pairwise overlaps and exact
hashes are exported in the accompanying tables and manifest. These are subset
draws from one finite parent pool, not independent datasets.

## Neural resource gate

Phase 2 measurements imply approximately {(training_seconds + evaluation_seconds) / 60:.1f}
minutes before allowance. Charging a 25% margin gives
{projected_seconds / 60:.1f} minutes, below the independent two-hour neural
cap. It charges each new 5k realization at the measured Phase 2 5k cost and
the 10k slice at 1.5 times that cost, consistent with the observed fixed-step
training and density-inference scaling. This is not a runtime guarantee. The
campaign runner must stop at 7,200 seconds,
preserve partial results, and never shrink the matrix.

The fixed classifier still used 18,866 selected events from 377,330 candidates.
This campaign therefore tests acceptance-model training-data efficiency only,
conditional on that classifier; it is not an end-to-end low-data experiment.
"""
    output_paths["report"].write_text(report)

    portable = [
        output_paths["config"],
        output_paths["summary"],
        output_paths["support"],
        output_paths["overlaps"],
        output_paths["report"],
    ]
    manifest = {
        "schema_version": 1,
        "analysis": "Phase 3 subset and resource freeze",
        "status": "frozen_approved_not_started",
        "source_commit": source_commit,
        "script": str(Path(__file__).resolve().relative_to(repo)),
        "script_sha256": sha256_file(Path(__file__)),
        "approval_commit": "2e4abd3953151acb554db51cfefef824ea2cfd9e",
        "parent_pool": phase2["full_input"],
        "orderings": order_specs,
        "subsets": subset_specs,
        "overlaps": {
            "pairwise_table": str(output_paths["overlaps"].relative_to(repo)),
            "five_k_union_unique_events": len(five_k_union),
            "five_k_triple_intersection_unique_events": len(five_k_triple),
            "original_5k_nested_in_original_10k": True,
            "all_training_subsets_disjoint_from_final_target": True,
            "all_training_subsets_disjoint_from_final_context_reservoir": True,
        },
        "resource_gates": {
            "dense_gp_hard_limit_seconds": 14400,
            "dense_gp_projected_seconds": 8662.744420958916,
            "dense_gp_pilot_campaign_cells_reusable": 0,
            "dense_gp_pilot_reuse_limitation": (
                "Timing evidence only: pilot optimizer seed 31000 differs from "
                "campaign seed 31400 for the corresponding cell."
            ),
            "neural_hard_limit_seconds": 7200,
            "neural_projected_seconds_with_25_percent_allowance": projected_seconds,
            "phase2_measured_5k_training_seconds": measured_5k_training_seconds,
            "phase2_measured_5k_evaluation_seconds": measured_5k_evaluation_seconds,
            "ten_k_scaling_factor_from_5k": ten_k_scaling_factor,
            "timing_inputs": {
                str(phase2_training_runs_path.relative_to(repo)): sha256_file(
                    phase2_training_runs_path
                ),
                str(phase2_evaluation_runs_path.relative_to(repo)): sha256_file(
                    phase2_evaluation_runs_path
                ),
            },
        },
        "fixed_protocol": {
            "classifier_training_events": 18866,
            "classifier_candidate_events": 377330,
            "threshold": 0.540643572807312,
            "context_size": 500,
            "context_seeds": list(CONTEXT_SEEDS),
            "final_target_events": 114400,
            "mc_passes": 50,
            "dropout_seed": 10100,
            "training_steps": 3000,
        },
        "outputs": {
            str(path.relative_to(repo)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in portable
        },
        "historical_data_exposure": (
            "Prospectively specified follow-up analysis on historically exposed data; "
            "not an untouched test."
        ),
    }
    with output_paths["manifest"].open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
