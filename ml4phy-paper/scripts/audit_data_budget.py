#!/usr/bin/env python3
"""Audit unique event budgets, overlaps, and executed estimator semantics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

IDENTITY_FIELDS = ("run_number", "detector", "id", "tp0")
ENERGY_RANGE_KEV = (500.0, 3000.0)
CONTEXT_SEEDS = tuple(range(100, 110))
TRAINING_SEEDS = (0, 1, 2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def display_path(path: Path, repo: Path) -> str:
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def identity_matrix(handle: h5py.File, rows: np.ndarray | None = None) -> np.ndarray:
    if rows is None:
        return np.column_stack(
            [handle[field][:].astype("<i8", copy=False) for field in IDENTITY_FIELDS]
        )
    return np.column_stack(
        [handle[field][rows].astype("<i8", copy=False) for field in IDENTITY_FIELDS]
    )


def identity_keys(identities: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(identities, dtype="<i8")
    return contiguous.view(np.dtype((np.void, contiguous.dtype.itemsize * 4))).reshape(-1)


def unique_keys(identities: np.ndarray) -> np.ndarray:
    keys = np.unique(identity_keys(identities))
    if keys.size != identities.shape[0]:
        raise ValueError("Expected unique composite event identities.")
    return keys


def identity_hash_from_keys(keys: np.ndarray) -> str:
    identities = np.ascontiguousarray(keys).view("<i8").reshape(-1, 4)
    ordered = identities[
        np.lexsort(
            tuple(
                identities[:, index]
                for index in reversed(range(identities.shape[1]))
            )
        )
    ]
    return hashlib.sha256(np.ascontiguousarray(ordered, dtype="<i8").tobytes()).hexdigest()


def identity_hash(identities: np.ndarray) -> str:
    return identity_hash_from_keys(unique_keys(identities))


def read_prediction(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return {
            "identities": identity_matrix(handle),
            "energy": handle["energy"][:].astype(np.float64),
            "score": handle["score"][:].astype(np.float64),
            "label": handle["label"][:].astype(bool),
        }


def read_raw_filtered(files: list[Path], target_label: str) -> dict[str, np.ndarray]:
    identities = []
    energies = []
    labels = []
    for path in files:
        with h5py.File(path, "r") as handle:
            energy = handle["energy_label"][:].astype(np.float64)
            keep = (energy >= ENERGY_RANGE_KEV[0]) & (energy < ENERGY_RANGE_KEV[1])
            rows = np.flatnonzero(keep)
            identities.append(identity_matrix(handle, rows))
            energies.append(energy[rows])
            labels.append(handle[target_label][rows].astype(bool))
    return {
        "identities": np.concatenate(identities),
        "energy": np.concatenate(energies),
        "label": np.concatenate(labels),
    }


def deterministic_subset(population: dict[str, np.ndarray], fraction: float, seed: int) -> dict:
    count = max(1, int(round(population["energy"].size * fraction)))
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(population["energy"].size, size=count, replace=False))
    return {key: value[rows] for key, value in population.items()}


def assert_prediction_matches(name: str, reconstructed: dict, prediction: dict) -> None:
    if not np.array_equal(reconstructed["identities"], prediction["identities"]):
        raise ValueError(f"{name} identities do not match the reconstructed dataset.")
    if not np.allclose(reconstructed["energy"], prediction["energy"], rtol=0.0, atol=1e-3):
        raise ValueError(f"{name} energies do not match the reconstructed dataset.")
    if not np.array_equal(reconstructed["label"], prediction["label"]):
        raise ValueError(f"{name} labels do not match the reconstructed dataset.")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def training_exposure(seed: int) -> dict[str, int]:
    """Reproduce the training-loop RNG calls that set N and context size."""
    rng = np.random.default_rng(seed)
    event_counts = []
    context_counts = []
    for _ in range(3000):
        event_counts.append(int(rng.integers(640, 1025)))
        context_counts.append(int(rng.integers(128, 513)))
        rng.integers(0, 2**31 - 1)  # EventSampler seed.
        rng.integers(0, 2**31 - 1)  # Context/target permutation seed.
    total_trial_events = sum(event_counts)
    total_context = sum(context_counts)
    return {
        "sum_events_per_single_trial_over_steps": total_trial_events,
        "event_draws_across_batch": 16 * total_trial_events,
        "context_uses_across_batch": 16 * total_context,
        "target_label_uses_across_batch": 16 * (total_trial_events - total_context),
        "mc_nll_terms": 4 * 16 * (total_trial_events - total_context),
    }


def development_run(repo: Path, seed: int) -> Path:
    if seed == 100:
        run_id = "20260908-cell17-dev-s100-mc50"
    else:
        run_id = f"20260908-cell17-dev-ctx-s{seed}-drop10100-mc50"
    return repo / "ml4phy-paper/runs" / run_id


def final_run(repo: Path, seed: int) -> Path:
    run_id = f"20260908-cell17_seed0_recovered-final-ctx-s{seed}-drop10100-mc50"
    return repo / "ml4phy-paper/runs" / run_id


def load_context_sets(
    repo: Path,
    development: dict[str, np.ndarray],
    full_test: dict[str, np.ndarray],
) -> tuple[list[dict], dict[str, np.ndarray]]:
    rows = []
    sets: dict[str, list[np.ndarray]] = {"development": [], "final": []}
    for phase, source, run_locator in (
        ("development", development, development_run),
        ("final", full_test, final_run),
    ):
        for seed in CONTEXT_SEEDS:
            run = run_locator(repo, seed)
            summary_path = run / "summary.json"
            event_path = run / "event_predictions.npz"
            summary = read_json(summary_path)
            with np.load(event_path) as archive:
                source_rows = archive["context_rows"].astype(np.int64)
            identities = source["identities"][source_rows]
            keys = unique_keys(identities)
            expected_hash = summary["protocol"]["context_draw_identity_sha256"]
            if identity_hash_from_keys(keys) != expected_hash:
                raise ValueError(f"Context identity mismatch in {run.name}")
            if summary["randomness"]["context_seed"] != seed:
                raise ValueError(f"Context seed mismatch in {run.name}")
            rows.append(
                {
                    "phase": phase,
                    "context_seed": seed,
                    "unique_event_count": keys.size,
                    "identity_sha256": expected_hash,
                    "run_id": summary["run_id"],
                    "executed_dropout_seed": summary["randomness"]["dropout_seed"],
                }
            )
            sets[phase].append(keys)
    return rows, {
        f"{phase}_context_draw_union": np.unique(np.concatenate(draws))
        for phase, draws in sets.items()
    }


def overlap_rows(sets: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    names = list(sets)
    for index, left_name in enumerate(names):
        for right_name in names[index:]:
            left = sets[left_name]
            right = sets[right_name]
            intersection = np.intersect1d(left, right, assume_unique=True).size
            union = left.size + right.size - intersection
            rows.append(
                {
                    "set_a": left_name,
                    "set_b": right_name,
                    "count_a": left.size,
                    "count_b": right.size,
                    "intersection_count": intersection,
                    "union_count": union,
                    "jaccard": intersection / union,
                }
            )
    return rows


def checkpoint_kappa(path: Path) -> tuple[float, float, float]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw = float(payload["model_state"]["attention.kappa_raw"].item())
    kappa = 1.0 + 4.0 / (1.0 + math.exp(-raw))
    contrast_transition_slope = (10.0 - kappa) * 10.0 / 4.0
    return raw, kappa, contrast_transition_slope


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("ml4phy-paper"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    output_root = (repo / args.output_root).resolve()
    outputs = {
        "ledger": output_root / "tables/data_budget_ledger.csv",
        "overlap": output_root / "tables/data_identity_overlap.csv",
        "context": output_root / "tables/context_draw_ledger.csv",
        "exposure": output_root / "tables/training_exposure_ledger.csv",
        "report": output_root / "reports/data_budget_ledger.md",
        "manifest": output_root / "manifests/data_budget_ledger.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        parser.error("Refusing to overwrite existing outputs: " + ", ".join(existing))
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    classifier_config_path = repo / "configs/small_data_configs/simple_cnn_small.yaml"
    classifier_metadata_path = repo / "runs/small_data_configs/simple_cnn_small/metadata.json"
    classifier_checkpoint_path = repo / "runs/small_data_configs/simple_cnn_small/epoch_050.pt"
    train_prediction_path = (
        repo / "runs/small_data_configs/simple_cnn_small/eval_train/predictions.h5"
    )
    development_path = repo / "runs/small_data_configs/simple_cnn_small/eval/predictions.h5"
    full_test_path = (
        repo / "runs/small_data_configs/simple_cnn_small/eval_full_test/predictions.h5"
    )
    protocol_path = repo / "ml4phy-paper/manifests/frozen_protocol_v1.json"
    role_archive_path = repo / "ml4phy-paper/local/protocol/frozen_roles_v1.npz"
    classifier_config = yaml.safe_load(classifier_config_path.read_text())
    classifier_metadata = read_json(classifier_metadata_path)
    if classifier_metadata["completed_epochs"] != 50:
        raise ValueError("Recovered classifier metadata does not record 50 completed epochs.")
    protocol = read_json(protocol_path)
    data_config = classifier_config["data"]
    raw_root = Path(data_config["data_dir"])
    raw_train_files = sorted(
        raw_root.glob("MJD_Train_*.hdf5"), key=lambda path: int(path.stem.rsplit("_", 1)[1])
    )
    raw_test_files = sorted(
        raw_root.glob("MJD_Test_*.hdf5"), key=lambda path: int(path.stem.rsplit("_", 1)[1])
    )
    if len(raw_train_files) != 16 or len(raw_test_files) != 6:
        raise ValueError("The resolved raw train/test file counts differ from the executed config.")

    raw_train = read_raw_filtered(raw_train_files, data_config["target_label"])
    raw_test = read_raw_filtered(raw_test_files, data_config["target_label"])
    classifier_train = deterministic_subset(
        raw_train, data_config["subset_portion"], data_config["subset_seed"]
    )
    classifier_monitor = deterministic_subset(
        raw_test, data_config["subset_portion"], data_config["subset_seed"]
    )
    train_prediction = read_prediction(train_prediction_path)
    development = read_prediction(development_path)
    full_test = read_prediction(full_test_path)
    assert_prediction_matches("classifier training export", classifier_train, train_prediction)
    assert_prediction_matches("classifier monitoring export", classifier_monitor, development)
    assert_prediction_matches("full-test export", raw_test, full_test)

    # Reproduce the acceptance sampler's 10-keV, minimum-four-event filter.
    energy = train_prediction["energy"]
    edges = np.arange(500.0, 3000.0 + 5.0, 10.0)
    bin_index = np.clip(np.digitize(energy, edges) - 1, 0, edges.size - 2)
    bin_counts = np.bincount(bin_index, minlength=edges.size - 1)
    kept_bins = bin_counts >= 4
    acceptance_effective_rows = np.flatnonzero(kept_bins[bin_index])
    excluded_rows = np.flatnonzero(~kept_bins[bin_index])
    if acceptance_effective_rows.size != 18836 or excluded_rows.size != 30:
        raise ValueError("Acceptance-sampler effective count differs from the recovered artifact.")

    with np.load(role_archive_path) as archive:
        calibration_rows = archive["calibration_development_rows"].astype(np.int64)
        development_context_rows = archive[
            "development_context_reservoir_rows"
        ].astype(np.int64)
        development_target_rows = archive["development_target_rows"].astype(np.int64)
        final_context_rows = archive["final_context_reservoir_rows"].astype(np.int64)
        final_target_rows = archive["final_target_rows"].astype(np.int64)

    role_sets = {
        "threshold_calibration": unique_keys(development["identities"][calibration_rows]),
        "development_context_reservoir": unique_keys(
            development["identities"][development_context_rows]
        ),
        "development_target": unique_keys(development["identities"][development_target_rows]),
        "final_context_reservoir": unique_keys(full_test["identities"][final_context_rows]),
        "final_target": unique_keys(full_test["identities"][final_target_rows]),
    }
    for role, keys in role_sets.items():
        if identity_hash_from_keys(keys) != protocol["roles"][role]["identity_sha256"]:
            raise ValueError(f"Frozen role hash mismatch for {role}")

    context_rows, context_unions = load_context_sets(repo, development, full_test)
    sets = {
        "classifier_training_denominator": unique_keys(raw_train["identities"]),
        "classifier_training_subset": unique_keys(classifier_train["identities"]),
        "classifier_monitoring_subset": unique_keys(classifier_monitor["identities"]),
        "acceptance_input_pool": unique_keys(train_prediction["identities"]),
        "acceptance_effective_sampling_pool": unique_keys(
            train_prediction["identities"][acceptance_effective_rows]
        ),
        "density_pool": unique_keys(train_prediction["identities"]),
        **role_sets,
        **context_unions,
    }

    expected_full_partition = np.unique(
        np.concatenate(
            [
                role_sets["threshold_calibration"],
                role_sets["development_context_reservoir"],
                role_sets["development_target"],
                role_sets["final_context_reservoir"],
                role_sets["final_target"],
            ]
        )
    )
    if not np.array_equal(expected_full_partition, unique_keys(full_test["identities"])):
        raise ValueError("Frozen evaluation roles do not partition the full test export.")

    fixed_threshold = float(protocol["threshold"]["value"])
    ledger_rows = [
        {
            "role": "classifier_training_denominator",
            "unique_event_count": sets["classifier_training_denominator"].size,
            "identity_sha256": identity_hash_from_keys(sets["classifier_training_denominator"]),
            "parent": "16 raw MAJORANA train files",
            "selection_or_filter": "500 <= energy < 3000 keV before fixed 5% selection",
            "information_used": "identity, energy; establishes the subset denominator",
        },
        {
            "role": "classifier_training_subset",
            "unique_event_count": sets["classifier_training_subset"].size,
            "identity_sha256": identity_hash_from_keys(sets["classifier_training_subset"]),
            "parent": "classifier_training_denominator",
            "selection_or_filter": "round(0.05 * 377330), NumPy seed 0, without replacement",
            "information_used": "waveform, psd_label_low_avse, energy, identity",
        },
        {
            "role": "classifier_monitoring_subset",
            "unique_event_count": sets["classifier_monitoring_subset"].size,
            "identity_sha256": identity_hash_from_keys(sets["classifier_monitoring_subset"]),
            "parent": "141474-event filtered test split",
            "selection_or_filter": "round(0.05 * 141474), NumPy seed 0, without replacement",
            "information_used": "per-epoch weighted monitoring; not checkpoint selection",
        },
        {
            "role": "acceptance_input_pool",
            "unique_event_count": sets["acceptance_input_pool"].size,
            "identity_sha256": identity_hash_from_keys(sets["acceptance_input_pool"]),
            "parent": "classifier_training_subset",
            "selection_or_filter": "inclusive target class and 500 <= energy <= 3000 keV",
            "information_used": "energy and frozen-classifier score",
        },
        {
            "role": "acceptance_effective_sampling_pool",
            "unique_event_count": sets["acceptance_effective_sampling_pool"].size,
            "identity_sha256": identity_hash_from_keys(
                sets["acceptance_effective_sampling_pool"]
            ),
            "parent": "acceptance_input_pool",
            "selection_or_filter": "events in 10-keV bins with at least four pool events",
            "information_used": "sampled with replacement to construct meta-learning trials",
        },
        {
            "role": "density_pool",
            "unique_event_count": sets["density_pool"].size,
            "identity_sha256": identity_hash_from_keys(sets["density_pool"]),
            "parent": "acceptance_input_pool",
            "selection_or_filter": "no minimum-bin filter; exact full acceptance input pool",
            "information_used": "energy only; fixed nonpersistent density buffer for our model",
        },
    ]
    for role in (
        "threshold_calibration",
        "development_context_reservoir",
        "development_context_draw_union",
        "development_target",
        "final_context_reservoir",
        "final_context_draw_union",
        "final_target",
    ):
        is_draw_union = role.endswith("draw_union")
        ledger_rows.append(
            {
                "role": role,
                "unique_event_count": sets[role].size,
                "identity_sha256": identity_hash_from_keys(sets[role]),
                "parent": "historical 7074-event subset"
                if role.startswith(("threshold", "development"))
                else "full test minus historical 7074-event subset",
                "selection_or_filter": (
                    "union of ten nested-study source draws at n_context=2000"
                    if is_draw_union
                    else "frozen role from split seed 20260908"
                ),
                "information_used": (
                    "fixed-threshold outcomes used for conditioning"
                    if "context" in role
                    else "classifier labels for threshold selection"
                    if role == "threshold_calibration"
                    else "fixed-threshold outcomes used for evaluation"
                ),
            }
        )

    exposure_rows = [
        {
            "stage": "classifier_training",
            "seed": 0,
            "steps_or_epochs": 50,
            "eligible_unique_events": 18866,
            "repeated_event_draws": 18866 * 50,
            "context_uses": "",
            "target_label_uses": 18866 * 50,
            "mc_nll_terms": "",
            "sampling": "weighted replacement; class and 10-keV energy weights multiplied",
        },
        {
            "stage": "classifier_monitoring",
            "seed": 0,
            "steps_or_epochs": 50,
            "eligible_unique_events": 7074,
            "repeated_event_draws": 7074 * 50,
            "context_uses": "",
            "target_label_uses": 7074 * 50,
            "mc_nll_terms": "",
            "sampling": "weighted replacement per epoch; monitoring only",
        },
    ]
    for seed in TRAINING_SEEDS:
        exposure = training_exposure(seed)
        exposure_rows.append(
            {
                "stage": "acceptance_model_training",
                "seed": seed,
                "steps_or_epochs": 3000,
                "eligible_unique_events": 18836,
                "repeated_event_draws": exposure["event_draws_across_batch"],
                "context_uses": exposure["context_uses_across_batch"],
                "target_label_uses": exposure["target_label_uses_across_batch"],
                "mc_nll_terms": exposure["mc_nll_terms"],
                "sampling": "flat 10-keV-bin stratified replacement; variable N=640..1024",
            }
        )

    write_csv(outputs["ledger"], ledger_rows)
    write_csv(outputs["overlap"], overlap_rows(sets))
    write_csv(outputs["context"], context_rows)
    write_csv(outputs["exposure"], exposure_rows)

    total_scored_union = np.union1d(
        sets["classifier_training_subset"], unique_keys(full_test["identities"])
    ).size
    source_population_union = np.union1d(
        sets["classifier_training_denominator"], unique_keys(full_test["identities"])
    ).size
    development_draw_union = sets["development_context_draw_union"].size
    final_draw_union = sets["final_context_draw_union"].size

    cell17_paths = [
        repo / "results/cut_acceptance/simple_cnn_small/sweeps/cell17/bin10/inclusive/cnp.ckpt",
        repo
        / "ml4phy-paper/runs/training/20260908-cell17-seed1-train3000/artifacts/cnp.ckpt",
        repo
        / "ml4phy-paper/runs/training/20260908-cell17-seed2-train3000/artifacts/cnp.ckpt",
    ]
    gate_rows = [checkpoint_kappa(path) for path in cell17_paths]
    gate_table = "\n".join(
        f"| {seed} | {raw:.9f} | {kappa:.9f} | {slope:.9f} |"
        for seed, (raw, kappa, slope) in zip(TRAINING_SEEDS, gate_rows, strict=True)
    )
    exposure_table = "\n".join(
        f"| {row['seed']} | {row['repeated_event_draws']:,} | "
        f"{int(row['context_uses']):,} | {int(row['target_label_uses']):,} | "
        f"{int(row['mc_nll_terms']):,} |"
        for row in exposure_rows
        if row["stage"] == "acceptance_model_training"
    )
    report = f"""# Data-budget and executed-method ledger

Audit date: 2026-09-09 (America/Los_Angeles)
Status: complete before any extension training. The two uncommitted laptop plans named in the extension prompt were absent from this checkout.

## Unique-event accounting

The classifier's 5% budget is **18,866 unique events selected from 377,330 eligible raw train-split events**, not a percentage of the acceptance export. The denominator is the concatenation of all 16 configured train files after applying `500 <= energy < 3000 keV`; `round(0.05 * 377330) = 18866`. Reconstructing NumPy's seed-0 selection from the raw files reproduces the 18,866-row classifier train prediction export identity-for-identity, including order, energy, and labels.

The acceptance model reads those same 18,866 score evaluations. Its `target_class: all` and energy filter remove none, but the sampler retains only bins with at least four events: **18,836 unique events are sampling-eligible and 30 are excluded across 21 sparse nonempty bins**. Sampling is with replacement. The density-guided model separately reconstructs a nonpersistent energy-only density buffer from all **18,866** input events before the minimum-bin sampler filter. Therefore the density-pool/acceptance-input overlap is 18,866/18,866, while the sampling-effective/density overlap is 18,836/18,866.

The classifier train split is identity-disjoint from every evaluation role. The 7,074-event classifier monitoring subset is exactly partitioned into 2,000 threshold-calibration events, a 3,000-event development context reservoir, and a 2,074-event development target. The remainder of the full 141,474-event test export is exactly partitioned into the 20,000-event final context reservoir and 114,400-event final target.

The ten executed 2,000-event development contexts have a unique union of **{development_draw_union:,}** events within their 3,000-event reservoir. The ten final contexts have a unique union of **{final_draw_union:,}** within their 20,000-event reservoir. Context overlaps are sensitivity replications, not independent additional data.

The union of unique waveform events that were either used for classifier training or score-evaluated in the full follow-up data is **{total_scored_union:,} = 18,866 train + 141,474 disjoint test**. If the full filtered classifier-training denominator is disclosed as the source population considered for the fixed subset, its union with the full test export is **{source_population_union:,} = 377,330 + 141,474**. Acceptance pretraining, density estimation, calibration, conditioning, and evaluation reuse these events; their row counts must not be summed as if disjoint.

The machine-readable ledger and complete pairwise intersections are in `tables/data_budget_ledger.csv` and `tables/data_identity_overlap.csv`. No event identity is committed; only counts and order-independent SHA-256 hashes are exported.

## Classifier execution

- Configuration: fixed subset seed 0, `subset_portion=0.05`, 50 epochs, batch size 256, `train_portion=1.0`.
- Preprocessing: subtract the mean of the first 500 waveform samples, divide by the positive maximum, align to the first 90%-rise sample, crop 200 samples before and 2,000 after with zero padding, then cast to float32.
- Label: raw `psd_label_low_avse`. The classifier loss is unweighted `BCEWithLogitsLoss`; `pos_weight: auto` is inactive because `loss.type` is `bce`.
- Sampling: class-balanced and 10-keV energy-balanced weights are multiplied; `WeightedRandomSampler` draws 18,866 samples with replacement per epoch. This gives 943,300 repeated train draws over 50 epochs. The exact realized unique coverage was not recorded and cannot be reconstructed from the saved checkpoints because model/dropout RNG consumption shared the PyTorch generator.
- Monitoring: a separately reconstructed 7,074-event fixed 5% test subset is sampled with replacement under the same weights each epoch, for 353,700 monitoring draws. It is not used for gradient updates.
- Checkpoint rule: every epoch is saved. Directory-based evaluation selects the lexicographically latest checkpoint, epoch 50. There is no best-validation checkpoint selection; epoch 49 has the highest recorded monitoring ROC AUC, while epoch 50 was executed and retained as the frozen classifier.

## Acceptance-model execution

The executed output is a logistic-normal Bernoulli-probability parameterization. The decoder emits unconstrained `mu_logit` and `log_sigma`; `sigma = softplus(log_sigma)`. For each target event and each of four MC samples, training draws standard-normal noise and forms `beta = sigmoid(mu_logit + sigma * epsilon)`, clipped to `[1e-6, 1 - 1e-6]`. The loss is Bernoulli negative log likelihood with one final unweighted mean over MC samples, batch trials, and target events. At deterministic evaluation, the pinned implementation uses `sigmoid(mu_logit)`; the paper's 50-pass estimator instead averages stochastic network evaluations with dropout active.

`mixup_alpha=0.01` is present in the validated configuration but **no executed training code reads it**, so mixup is inactive. Each training seed uses Adam at 1e-3, gradient-norm clipping at 1.0, 3,000 steps, batch size 16, variable trial size 640-1,024, and context size 128-512. The exact repeated exposure implied by the training-loop RNG is:

| Training seed | Pool-event draws | Context uses | Target-label uses | Four-sample NLL terms |
|---:|---:|---:|---:|---:|
{exposure_table}

Only the final 3,000-step acceptance checkpoint is written; `eval_every=0`, so the 7,074-event validation path does not select checkpoints or affect gradients. The legacy pipeline computes a Youden threshold from that file after training for its summary, but the paper evaluator ignores it and uses the separately frozen calibration threshold.

## Decoder gate

For Density-guided CNP, the executed contrast gate is `lambda(R) = kappa + (10 - kappa) * sigmoid(10 * (R - 3))`. Thus the contrast threshold is 3 and the sigmoid argument slope is exactly 10 per unit density contrast. The derivative `d lambda / d R` at `R=3` depends on the learned continuum floor:

| Training seed | Saved raw kappa | Executed kappa | Transition derivative at R=3 |
|---:|---:|---:|---:|
{gate_table}

The subsequent per-Fourier-band decoder weight is `sigmoid(5 * (lambda - band_index))`; its logit slope is 5 per band-index unit and its maximum weight derivative with respect to `lambda` is 1.25.

## Threshold, randomness, and provenance limitations

The threshold is **{fixed_threshold:.15f}**, selected by Youden-J on only the frozen 2,000-event calibration role. No development or final target outcome entered threshold selection.

The original protocol manifest committed at 2026-09-08 20:37:55 PDT specified `dropout_seed = 10000 + context_seed`. Before candidate selection, the executed design changed to a fixed dropout seed of **10100** so the ten context draws isolate context variation. The fixed-seed result was committed at 2026-09-08 20:54:30 PDT; the earlier context-and-dropout-varying campaign remains server-side and was excluded. The original frozen manifest is preserved unchanged, and this report records the dated prospective execution amendment rather than silently rewriting it.

Recovered seed-0 Cell 17 has a readable final checkpoint, configuration, loss history, and colocated training pool, but lacks an exact MAJORANA source commit, clean/dirty worktree state, runtime, peak memory, and realized event-draw log. Seeds 1 and 2 have paper-runner provenance. This asymmetry must remain disclosed.

## 1620-keV feature identity

The repository's historical `Bi-214 1620` label is inconsistent with its stated thorium-228 calibration spectrum. The IAEA recommended gamma-ray data for thorium-228 with daughters identify a **1,620.74-keV gamma from bismuth-212**. New artifacts therefore use `Bi-212 1620.74 keV`; archived labels remain untouched. Source: https://www-nds.iaea.org/publications/tecdocs/sti-pub-1287_Vol2.pdf.

## Interpretation boundary

Classifier pretraining uses waveforms and physical PSD labels. Acceptance-model pretraining reuses classifier scores and energies from the same 18,866 identities; it does not add new measured events. Context outcomes add conditional information at inference. Target events are evaluation data, not training data, but their classifier scoring and outcome/reference cost are disclosed. Consequently, a context-size experiment supports conditional context efficiency only, not total-data efficiency.
"""
    outputs["report"].write_text(report)

    source_files = [
        classifier_config_path,
        classifier_metadata_path,
        classifier_checkpoint_path,
        train_prediction_path,
        development_path,
        full_test_path,
        protocol_path,
        role_archive_path,
        repo / "majorana_acp/data/dataset.py",
        repo / "majorana_acp/training/trainer.py",
        repo / "majorana_acp/cut_acceptance/event_sampler.py",
        repo / "majorana_acp/cut_acceptance/pipeline.py",
        repo / "majorana_acp/models/attentive_cnp.py",
        repo / "ml4phy-paper/local/resum-flex-edba6a/core/surrogate_cnp.py",
    ]
    manifest = {
        "schema_version": 1,
        "analysis": "extension Phase 0 data-budget and executed-method audit",
        "source_commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "script": "ml4phy-paper/scripts/audit_data_budget.py",
        "script_sha256": sha256_file(Path(__file__)),
        "identity_fields": list(IDENTITY_FIELDS),
        "historical_data_exposure": protocol["exposure_statement"],
        "raw_inputs": {
            "train_files": [
                {"logical_name": path.name, "bytes": path.stat().st_size}
                for path in raw_train_files
            ],
            "test_files": [
                {"logical_name": path.name, "bytes": path.stat().st_size}
                for path in raw_test_files
            ],
            "large_file_policy": (
                "Raw multi-gigabyte HDF5 files are bound by reconstructed identity hashes; "
                "portable byte hashes are recorded for the compact prediction exports."
            ),
        },
        "input_hashes": {
            str(path.relative_to(repo)): sha256_file(path)
            for path in source_files
            if path.is_file() and path.stat().st_size < 100_000_000
        },
        "key_findings": {
            "classifier_training_denominator": 377330,
            "classifier_training_unique": 18866,
            "acceptance_input_unique": 18866,
            "acceptance_sampling_effective_unique": 18836,
            "density_pool_unique": 18866,
            "development_context_draw_union": int(development_draw_union),
            "final_context_draw_union": int(final_draw_union),
            "full_test_unique": 141474,
            "end_to_end_scored_unique_union": int(total_scored_union),
            "eligible_source_population_union": int(source_population_union),
            "fixed_threshold": fixed_threshold,
            "executed_dropout_seed": 10100,
        },
        "outputs": {},
    }
    for name, path in outputs.items():
        if name == "manifest":
            continue
        manifest["outputs"][display_path(path, repo)] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    with outputs["manifest"].open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
