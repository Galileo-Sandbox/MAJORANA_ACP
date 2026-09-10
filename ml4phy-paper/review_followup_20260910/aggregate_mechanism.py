#!/usr/bin/env python3
"""Aggregate the frozen 150-cell mechanism comparison and mechanism maps."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import lzma
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
for path in (HERE, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from control_models import (  # noqa: E402
    GLOBAL_ATTENTION_MODES,
    GLOBAL_GATE_MODES,
    apply_control_mode,
    base_attention,
    load_control_checkpoint,
    parameter_inventory,
)
from export_existing import REGIONS, composite_rows, mask, write_csv  # noqa: E402
from export_paper_coverage import (  # noqa: E402
    BIN_CENTERS,
    MIN_BIN_EVENTS,
    bin_events,
    coverage_metrics,
    sha256_file,
)

MODES = ("full_density", "global_gate", "global_attention", "global_both", "density_free_global")
MODE_LABELS = {
    "full_density": "Full density guidance",
    "global_gate": "Global decoder gate",
    "global_attention": "Global attention",
    "global_both": "Global gate + attention",
    "density_free_global": "Density-free global",
}
PRIMARY = ("peaks_equal_four_cores", "continuum_equal_two_windows")


def write_gzip(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz,
        io.TextIOWrapper(gz, newline="") as text,
    ):
        writer = csv.DictWriter(text, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_xz(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with lzma.open(path, "wt", preset=9, newline="") as text:
        writer = csv.DictWriter(text, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha_rows(rows: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(rows, dtype="<i8").tobytes()).hexdigest()


def records(repo: Path, campaign: dict) -> list[dict]:
    old = json.loads((repo / "ml4phy-paper/manifests/paper_presentation_export.json").read_text())
    full = [
        {
            "mode": "full_density",
            "mode_label": MODE_LABELS["full_density"],
            "run_id": row["run_id"],
            "training_seed": int(row["training_seed"]),
            "context_seed": int(row["context_seed"]),
            "event_path": row["event_predictions_path"],
            "event_sha256": row["event_predictions_sha256"],
            "curve_path": row["curve_path"],
            "curve_sha256": row["curve_sha256"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "provenance": "reused_original_5k_full_density",
        }
        for row in old["source_artifacts"]
        if row["subset_id"] == "original_n5000" and row["architecture_id"] == "ours"
    ]
    new = []
    for row in campaign["completed_evaluations"]:
        run = HERE / "runs/evaluation" / row["run_id"]
        new.append(
            {
                "mode": row["mode"],
                "mode_label": MODE_LABELS[row["mode"]],
                "run_id": row["run_id"],
                "training_seed": int(row["training_seed"]),
                "context_seed": int(row["context_seed"]),
                "event_path": str((run / "event_predictions.npz").relative_to(repo)),
                "event_sha256": row["event_predictions_sha256"],
                "curve_path": str((run / "curve.npz").relative_to(repo)),
                "curve_sha256": row["curve_sha256"],
                "checkpoint_sha256": next(
                    x["checkpoint_sha256"]
                    for x in campaign["completed_training"]
                    if x["mode"] == row["mode"] and x["training_seed"] == row["training_seed"]
                ),
                "provenance": "new_mechanism_control",
            }
        )
    result = full + new
    expected = {
        (mode, seed, context) for mode in MODES for seed in range(3) for context in range(100, 110)
    }
    actual = {(x["mode"], x["training_seed"], x["context_seed"]) for x in result}
    if len(result) != 150 or actual != expected:
        raise ValueError(
            f"Mechanism inventory mismatch: {len(result)} cells, {len(expected - actual)} missing"
        )
    return sorted(
        result, key=lambda x: (MODES.index(x["mode"]), x["training_seed"], x["context_seed"])
    )


def aggregate_cells(repo: Path, inventory: list[dict]):
    registry, bins, regional = [], [], []
    reference_rows = None
    reference = None
    target_hash = None
    for cell_index, record in enumerate(inventory):
        path = repo / record["event_path"]
        if sha256_file(path) != record["event_sha256"]:
            raise ValueError(f"Event prediction hash mismatch: {record['run_id']}")
        curve_path = repo / record["curve_path"]
        if sha256_file(curve_path) != record["curve_sha256"]:
            raise ValueError(f"Curve hash mismatch: {record['run_id']}")
        with np.load(path) as archive:
            rows = archive["target_rows"].astype(np.int64)
            energy = archive["target_energy_kev"].astype(np.float64)
            outcome = archive["outcome"].astype(np.int8)
            prediction = archive["prediction"].astype(np.float64)
        current_hash = sha_rows(rows)
        if reference_rows is None:
            reference_rows = rows
            target_hash = current_hash
            reference = bin_events(energy, outcome)
        elif current_hash != target_hash or not np.array_equal(rows, reference_rows):
            raise ValueError(f"Target identity mismatch: {record['run_id']}")
        binned = bin_events(energy, outcome, prediction)
        with np.load(curve_path) as curve:
            if not np.array_equal(curve["bin_counts"], reference["counts"]):
                raise ValueError(f"Curve bin-count mismatch: {record['run_id']}")
            delta = np.nanmax(np.abs(curve["bin_prediction_mean"] - binned["prediction_mean"]))
            if delta > 1e-12:
                raise ValueError(
                    f"Saved/recomputed bin means differ by {delta}: {record['run_id']}"
                )
        registry.append(
            {"cell_index": cell_index, **record, "target_identity_sha256": current_hash}
        )
        for index, value in enumerate(binned["prediction_mean"]):
            bins.append(
                {
                    "cell_index": cell_index,
                    "bin_index": index,
                    "mean_predicted_probability": value,
                    "sum_predicted_passing_probabilities": value * reference["counts"][index]
                    if np.isfinite(value)
                    else None,
                }
            )
        residual = binned["prediction_mean"] - reference["fraction"]
        for region_id, definition in REGIONS.items():
            event_mask = mask(energy, definition)
            bin_mask = mask(BIN_CENTERS, definition)
            support = bin_mask & (reference["counts"] >= MIN_BIN_EVENTS)
            nonempty = bin_mask & (reference["counts"] > 0)
            n_events = int(event_mask.sum())
            observed = int(outcome[event_mask].sum())
            predicted = float(prediction[event_mask].sum())
            difference = predicted - observed
            ck = coverage_metrics(reference, binned["prediction_mean"], bin_mask)
            regional.append(
                {
                    "cell_index": cell_index,
                    "run_id": record["run_id"],
                    "mode": record["mode"],
                    "mode_label": record["mode_label"],
                    "training_seed": record["training_seed"],
                    "context_seed": record["context_seed"],
                    "region_id": region_id,
                    "region_label": definition["label"],
                    "physical_label": definition["physical_label"],
                    "region_kind": definition["kind"],
                    "N_R_all_events": n_events,
                    "Y_R_all_events": observed,
                    "P_R_all_events": predicted,
                    "passing_count_difference_all_events": difference,
                    "efficiency_difference_pp_all_events": 100.0 * difference / n_events
                    if n_events
                    else None,
                    "relative_passing_difference_percent_all_events": 100.0 * difference / observed
                    if observed
                    else None,
                    "supported_bin_count": int(support.sum()),
                    "excluded_bin_count": int(bin_mask.sum() - support.sum()),
                    "supported_target_events": int(reference["counts"][support].sum()),
                    "unsupported_nonempty_target_events": int(
                        reference["counts"][nonempty & ~support].sum()
                    ),
                    "weighted_bin_MAE_pp_supported": 100.0
                    * float(
                        np.average(np.abs(residual[support]), weights=reference["counts"][support])
                    )
                    if support.any()
                    else None,
                    "weighted_bin_MAE_pp_all_nonempty": 100.0
                    * float(
                        np.average(
                            np.abs(residual[nonempty]), weights=reference["counts"][nonempty]
                        )
                    )
                    if nonempty.any()
                    else None,
                    "component_regions": None,
                    "weighting": "event weighted for MAE; equal supported bins for Ck",
                    **{
                        key: ck[key]
                        for key in (
                            "hit_count_C1",
                            "hit_count_C2",
                            "hit_count_C3",
                            "C1_percent",
                            "C2_percent",
                            "C3_percent",
                        )
                    },
                }
            )
    assert reference is not None
    regional.extend(composite_rows(regional))
    return registry, bins, regional, reference, target_hash


def summarize(regional: list[dict]):
    metrics = (
        "passing_count_difference_all_events",
        "efficiency_difference_pp_all_events",
        "weighted_bin_MAE_pp_supported",
        "C1_percent",
        "C2_percent",
        "C3_percent",
    )
    grouped = defaultdict(list)
    for row in regional:
        grouped[(row["mode"], row["region_id"], row["training_seed"])].append(row)
    seed_rows = []
    for (mode, region, seed), rows in grouped.items():
        out = {
            "mode": mode,
            "mode_label": MODE_LABELS[mode],
            "region_id": region,
            "training_seed": seed,
            "context_cells": len(rows),
        }
        for metric in metrics:
            values = np.array([x[metric] for x in rows if x[metric] is not None], dtype=float)
            out[f"{metric}_context_mean"] = float(values.mean()) if values.size else None
            out[f"{metric}_context_sd"] = float(values.std(ddof=1)) if values.size > 1 else None
        seed_rows.append(out)
    summary = []
    grouped_seed = defaultdict(list)
    for row in seed_rows:
        grouped_seed[(row["mode"], row["region_id"])].append(row)
    for (mode, region), rows in grouped_seed.items():
        out = {
            "mode": mode,
            "mode_label": MODE_LABELS[mode],
            "region_id": region,
            "training_seeds": len(rows),
            "contexts_per_seed": 10,
        }
        for metric in metrics:
            values = np.array(
                [
                    x[f"{metric}_context_mean"]
                    for x in rows
                    if x[f"{metric}_context_mean"] is not None
                ],
                dtype=float,
            )
            context_sd = np.array(
                [x[f"{metric}_context_sd"] for x in rows if x[f"{metric}_context_sd"] is not None],
                dtype=float,
            )
            out[f"{metric}_mean"] = float(values.mean()) if values.size else None
            out[f"{metric}_across_seed_sd"] = float(values.std(ddof=1)) if values.size > 1 else None
            out[f"{metric}_mean_within_seed_context_sd"] = (
                float(context_sd.mean()) if context_sd.size else None
            )
        summary.append(out)
    return seed_rows, summary


def contrasts(regional: list[dict]) -> list[dict]:
    index = {
        (x["mode"], x["training_seed"], x["context_seed"], x["region_id"]): x for x in regional
    }
    pairs = (
        ("full_density", "global_gate", "adaptive decoder gate"),
        ("full_density", "global_attention", "adaptive attention"),
        ("global_gate", "global_both", "adaptive attention with global gate"),
        ("global_attention", "global_both", "adaptive gate with global attention"),
        ("global_both", "density_free_global", "direct density input under global modulation"),
        ("full_density", "density_free_global", "complete density package"),
    )
    rows = []
    for left, right, estimand in pairs:
        for seed in range(3):
            for context in range(100, 110):
                for region in PRIMARY + (
                    "overall",
                    "sparse_tail_2700_3000",
                    "FE_core",
                    "SE_core",
                    "DEP_core",
                    "feature_1620_core",
                ):
                    a, b = (
                        index[(left, seed, context, region)],
                        index[(right, seed, context, region)],
                    )
                    rows.append(
                        {
                            "left_mode": left,
                            "right_mode": right,
                            "estimand": estimand,
                            "training_seed": seed,
                            "context_seed": context,
                            "region_id": region,
                            "C1_difference_pp_left_minus_right": a["C1_percent"] - b["C1_percent"],
                            "C2_difference_pp_left_minus_right": a["C2_percent"] - b["C2_percent"],
                            "C3_difference_pp_left_minus_right": a["C3_percent"] - b["C3_percent"],
                            "weighted_MAE_difference_pp_left_minus_right": a[
                                "weighted_bin_MAE_pp_supported"
                            ]
                            - b["weighted_bin_MAE_pp_supported"],
                            "efficiency_difference_contrast_pp_left_minus_right": a[
                                "efficiency_difference_pp_all_events"
                            ]
                            - b["efficiency_difference_pp_all_events"]
                            if a["efficiency_difference_pp_all_events"] is not None
                            else None,
                        }
                    )
    return rows


def contrast_summary(rows: list[dict]) -> list[dict]:
    """Preserve context variation and the three paired seed means separately."""
    metrics = (
        "C1_difference_pp_left_minus_right",
        "C2_difference_pp_left_minus_right",
        "C3_difference_pp_left_minus_right",
        "weighted_MAE_difference_pp_left_minus_right",
        "efficiency_difference_contrast_pp_left_minus_right",
    )
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["left_mode"],
            row["right_mode"],
            row["estimand"],
            row["region_id"],
            row["training_seed"],
        )
        grouped[key].append(row)
    result = []
    by_comparison = defaultdict(list)
    for key, group in grouped.items():
        out = {
            "aggregation_level": "contexts_within_seed",
            "left_mode": key[0],
            "right_mode": key[1],
            "estimand": key[2],
            "region_id": key[3],
            "training_seed": key[4],
            "component_count": len(group),
        }
        for metric in metrics:
            values = np.array([x[metric] for x in group if x[metric] is not None], dtype=float)
            out[f"{metric}_mean"] = float(values.mean()) if values.size else None
            out[f"{metric}_descriptive_sd"] = float(values.std(ddof=1)) if values.size > 1 else None
        result.append(out)
        by_comparison[key[:4]].append(out)
    for key, group in by_comparison.items():
        out = {
            "aggregation_level": "equal_seed_mean",
            "left_mode": key[0],
            "right_mode": key[1],
            "estimand": key[2],
            "region_id": key[3],
            "training_seed": None,
            "component_count": len(group),
        }
        for metric in metrics:
            values = np.array(
                [x[f"{metric}_mean"] for x in group if x[f"{metric}_mean"] is not None], dtype=float
            )
            out[f"{metric}_mean"] = float(values.mean()) if values.size else None
            out[f"{metric}_descriptive_sd"] = float(values.std(ddof=1)) if values.size > 1 else None
        result.append(out)
    return result


def build_model(repo: Path, mode: str, seed: int, campaign: dict):
    from majorana_acp.cut_acceptance.config import load_config
    from majorana_acp.cut_acceptance.pipeline import build_local_cnp
    from majorana_acp.cut_acceptance.positional_encoding import phi_dim

    config_path = (
        repo
        / f"ml4phy-paper/runs/phase2/training/20260909-phase2-b5000-ours-seed{seed}-train3000/resolved_config.yaml"
    )
    if not config_path.exists():
        config_path = (
            repo
            / "ml4phy-paper/runs/phase2/training/20260909-phase2-b5000-ours-seed0-train3000/resolved_config.yaml"
        )
    cfg = load_config(config_path)
    cfg = cfg.model_copy(
        update={
            "training": cfg.training.model_copy(update={"seed": seed}),
            "train_predictions_path": repo
            / "ml4phy-paper/local/phase2/protocol-v1/training_budget_5000.h5",
        }
    )
    torch.manual_seed(seed)
    model = build_local_cnp(cfg, dim_phi=phi_dim(cfg.positional_encoding))
    if mode == "full_density":
        checkpoint = (
            repo
            / f"ml4phy-paper/runs/phase2/training/20260909-phase2-b5000-ours-seed{seed}-train3000/artifacts/cnp.ckpt"
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model_state", payload.get("state_dict", payload))
        model.load_state_dict(state, strict=True)
        apply_control_mode(model, mode)
    else:
        entry = next(
            x
            for x in campaign["completed_training"]
            if x["mode"] == mode and x["training_seed"] == seed
        )
        checkpoint = HERE / "runs/training" / entry["run_id"] / "artifacts/control.ckpt"
        model, _ = load_control_checkpoint(checkpoint, model, expected_mode=mode)
    return model.eval(), checkpoint


def mechanism_maps(repo: Path, campaign: dict):
    grid = torch.arange(500.0, 3000.0 + 1.0, 1.0, dtype=torch.float32)
    e_norm = (grid - 500.0) / 2500.0
    rows, checkpoints = [], []
    for mode in MODES:
        for seed in range(3):
            model, checkpoint = build_model(repo, mode, seed, campaign)
            attn = base_attention(model)
            pool = attn.pool_energies_norm.cpu()
            rho_l = torch.zeros_like(e_norm)
            rho_g = torch.zeros_like(e_norm)
            for start in range(0, pool.numel(), 1024):
                delta2 = (e_norm[:, None] - pool[start : start + 1024][None, :]).square()
                rho_l += torch.exp(-delta2 / (2 * attn.pool_density_sfn_sigma_local_norm**2)).sum(1)
                rho_g += torch.exp(-delta2 / (2 * attn.pool_density_sfn_sigma_global_norm**2)).sum(
                    1
                )
            z = torch.stack(
                (
                    torch.log(rho_l + attn.pool_density_sfn_epsilon),
                    torch.log(rho_g + attn.pool_density_sfn_epsilon),
                ),
                dim=-1,
            )
            contrast = (
                (attn.pool_density_sfn_sigma_global_norm / attn.pool_density_sfn_sigma_local_norm)
                * rho_l
                / (rho_g + attn.pool_density_sfn_epsilon)
            )
            zin = torch.zeros_like(z) if mode in GLOBAL_ATTENTION_MODES else z
            with torch.no_grad():
                bandwidth = 2500.0 * (
                    attn.pool_density_sfn_sigma_min_norm
                    + (attn.pool_density_sfn_sigma_max_norm - attn.pool_density_sfn_sigma_min_norm)
                    * torch.sigmoid(attn.pool_sfn_net(zin))
                ).reshape(-1)
                temperature = (
                    attn.pool_density_sfn_tau_min
                    + (attn.pool_density_sfn_tau_max - attn.pool_density_sfn_tau_min)
                    * torch.sigmoid(attn.pool_tau_net(zin))
                ).reshape(-1)
                if mode in GLOBAL_GATE_MODES:
                    cutoff = torch.full_like(grid, 1.0 + 9.0 * torch.sigmoid(attn.kappa_raw).item())
                else:
                    kappa = 1.0 + 4.0 * torch.sigmoid(attn.kappa_raw)
                    cutoff = kappa + (10.0 - kappa) * torch.sigmoid(10.0 * (contrast - 3.0))
                weights = torch.sigmoid(5.0 * (cutoff[:, None] - torch.arange(10.0)[None, :]))
            inv = parameter_inventory(model, mode)
            checkpoints.append(
                {
                    "mode": mode,
                    "training_seed": seed,
                    "checkpoint_path": str(checkpoint.relative_to(repo)),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    **inv,
                }
            )
            for i, energy in enumerate(grid.tolist()):
                row = {
                    "mode": mode,
                    "training_seed": seed,
                    "energy_kev": energy,
                    "density_contrast_R": float(contrast[i]),
                    "direct_decoder_density_input": 0.0
                    if mode == "density_free_global"
                    else float(contrast[i]),
                    "decoder_cutoff_lambda": float(cutoff[i]),
                    "attention_bandwidth_kev": float(bandwidth[i]),
                    "attention_temperature": float(temperature[i]),
                }
                row.update(
                    {f"decoder_band_weight_{band}": float(weights[i, band]) for band in range(10)}
                )
                rows.append(row)
    return rows, checkpoints


def figures(output: Path, summary: list[dict], maps: list[dict]) -> None:
    figdir = output / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    lookup = {(x["mode"], x["region_id"]): x for x in summary}
    x = np.arange(len(MODES))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for ax, region, title in zip(
        axes, PRIMARY, ("Four peak cores", "Two continuum windows"), strict=True
    ):
        values = [lookup[(mode, region)]["C2_percent_mean"] for mode in MODES]
        ax.bar(x, values, color=["#0072B2", "#E69F00", "#56B4E9", "#CC79A7", "#777777"])
        ax.set_title(title)
        ax.set_ylabel("Reference-band C2 (%)")
        ax.set_xticks(x, [MODE_LABELS[m] for m in MODES], rotation=35, ha="right")
        ax.set_ylim(0, 100)
    fig.savefig(figdir / "mechanism_primary_endpoints.pdf")
    fig.savefig(figdir / "mechanism_primary_endpoints.png", dpi=180)
    plt.close(fig)

    count_regions = ("DEP_core", "feature_1620_core", "SE_core", "FE_core")
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    width = 0.16
    positions = np.arange(len(count_regions))
    for offset, mode in enumerate(MODES):
        values = [
            lookup[(mode, region)]["passing_count_difference_all_events_mean"]
            for region in count_regions
        ]
        ax.bar(
            positions + (offset - 2) * width,
            values,
            width,
            label=MODE_LABELS[mode],
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(
        positions,
        ["Tl-208 DEP", "Bi-212 1620", "Tl-208 SE", "Tl-208 FE"],
    )
    ax.set_ylabel("Predicted - observed passing count")
    ax.set_title("Finite-reference peak-core count discrepancies")
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(figdir / "regional_passing_count_differences.pdf")
    fig.savefig(figdir / "regional_passing_count_differences.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, constrained_layout=True)
    for mode in MODES:
        subset = [r for r in maps if r["mode"] == mode and r["training_seed"] == 0]
        energy = [r["energy_kev"] for r in subset]
        axes[0].plot(energy, [r["decoder_cutoff_lambda"] for r in subset], label=MODE_LABELS[mode])
        axes[1].plot(energy, [r["attention_bandwidth_kev"] for r in subset])
        axes[2].plot(energy, [r["attention_temperature"] for r in subset])
    axes[0].set_ylabel("Decoder cutoff")
    axes[1].set_ylabel("Bandwidth (keV)")
    axes[2].set_ylabel("Temperature")
    axes[2].set_xlabel("Energy (keV)")
    axes[0].legend(ncol=2, fontsize=8)
    fig.savefig(figdir / "learned_mechanism_maps_seed0.pdf")
    fig.savefig(figdir / "learned_mechanism_maps_seed0.png", dpi=180)
    plt.close(fig)


def main() -> None:
    started = time.perf_counter()
    repo = Path(".").resolve()
    campaign_path = HERE / "runs/campaigns/20260910-mechanism-controls-v1/campaign_record.json"
    campaign = json.loads(campaign_path.read_text())
    if (
        campaign["status"] != "scientific_matrix_complete"
        or len(campaign["completed_evaluations"]) != 120
    ):
        raise ValueError("Scientific matrix is incomplete")
    inventory = records(repo, campaign)
    registry, bin_rows, regional, reference, target_hash = aggregate_cells(repo, inventory)
    seed_rows, summary = summarize(regional)
    paired = contrasts(regional)
    paired_summary = contrast_summary(paired)
    maps, checkpoints = mechanism_maps(repo, campaign)
    tables = HERE / "tables"
    write_csv(tables / "mechanism_run_inventory.csv", registry)
    write_xz(tables / "mechanism_cell_bins.csv.xz", bin_rows)
    write_gzip(tables / "mechanism_regional_cells.csv.gz", regional)
    write_gzip(tables / "mechanism_seed_summary.csv.gz", seed_rows)
    write_csv(tables / "mechanism_summary.csv", summary)
    write_gzip(tables / "mechanism_paired_contrasts.csv.gz", paired)
    write_csv(tables / "mechanism_paired_contrast_summary.csv", paired_summary)
    write_csv(tables / "mechanism_checkpoint_inventory.csv", checkpoints)
    write_gzip(tables / "learned_mechanism_maps.csv.gz", maps)
    write_csv(tables / "mechanism_full_region_sensitivity.csv", summary)
    figures(HERE, summary, maps)
    lookup = {(x["mode"], x["region_id"]): x for x in summary}
    primary_lines = []
    for mode in MODES:
        peaks = lookup[(mode, PRIMARY[0])]
        continuum = lookup[(mode, PRIMARY[1])]
        primary_lines.append(
            f"| {MODE_LABELS[mode]} | {peaks['C2_percent_mean']:.1f} | {continuum['C2_percent_mean']:.1f} | {peaks['weighted_bin_MAE_pp_supported_mean']:.2f} | {continuum['weighted_bin_MAE_pp_supported_mean']:.2f} |"
        )
    full_gate_peak = (
        lookup[("full_density", PRIMARY[0])]["C2_percent_mean"]
        - lookup[("global_gate", PRIMARY[0])]["C2_percent_mean"]
    )
    full_gate_cont = (
        lookup[("full_density", PRIMARY[1])]["C2_percent_mean"]
        - lookup[("global_gate", PRIMARY[1])]["C2_percent_mean"]
    )
    full_global_attention_peak = (
        lookup[("full_density", PRIMARY[0])]["C2_percent_mean"]
        - lookup[("global_attention", PRIMARY[0])]["C2_percent_mean"]
    )
    full_global_attention_cont = (
        lookup[("full_density", PRIMARY[1])]["C2_percent_mean"]
        - lookup[("global_attention", PRIMARY[1])]["C2_percent_mean"]
    )
    direct_peak = (
        lookup[("global_both", PRIMARY[0])]["C2_percent_mean"]
        - lookup[("density_free_global", PRIMARY[0])]["C2_percent_mean"]
    )
    direct_cont = (
        lookup[("global_both", PRIMARY[1])]["C2_percent_mean"]
        - lookup[("density_free_global", PRIMARY[1])]["C2_percent_mean"]
    )
    package_peak = (
        lookup[("full_density", PRIMARY[0])]["C2_percent_mean"]
        - lookup[("density_free_global", PRIMARY[0])]["C2_percent_mean"]
    )
    package_cont = (
        lookup[("full_density", PRIMARY[1])]["C2_percent_mean"]
        - lookup[("density_free_global", PRIMARY[1])]["C2_percent_mean"]
    )
    tail_full = lookup[("full_density", "sparse_tail_2700_3000")]["C2_percent_mean"]
    tail_gate = lookup[("global_gate", "sparse_tail_2700_3000")]["C2_percent_mean"]
    tail_attention = lookup[("global_attention", "sparse_tail_2700_3000")]["C2_percent_mean"]
    report = (
        """# Mechanism-control result

This post-review analysis uses the historically exposed frozen final reference. It is not an untouched-test experiment. All 150 mechanism cells use the original 5k efficiency-training ordering, 500 conditioning events, the fixed classifier and threshold, and the same reference. Scores were computed per cell before averaging contexts within each seed and then the three seed means.

## Primary endpoints

| Variant | Peaks C2 (%) | Continuum C2 (%) | Peaks weighted MAE (pp) | Continuum weighted MAE (pp) |
| --- | ---: | ---: | ---: | ---: |
"""
        + "\n".join(primary_lines)
        + f"""

Reference-band C2 is the fraction of supported 5-keV bins within two fixed Wilson-reference half-widths of the finite empirical fraction. It is not calibrated interval coverage. Peaks are the equal-feature mean of the four frozen narrow cores; Continuum is the equal-window mean of the two frozen continuum regions. No scalar combines these endpoints.

The regional count difference is a sum of predicted passing probabilities minus the observed passing count. It interprets the efficiency discrepancy on a count scale but is neither an independently known physical bias nor a calibrated significance. Event-weighted bin MAE is retained beside signed differences to reveal cancellation.

The paired-contrast table reports every prespecified comparison by initialization seed and context. Contexts overlap and share training/reference data, so the 30 cells per variant are not treated as independent experiments. Learned maps are deterministic checkpoint diagnostics, not MC-noise-corrected predictive smoothness evidence.

The complete-minus-global-gate differences are {full_gate_peak:+.1f} percentage points for Peaks C2 and {full_gate_cont:+.1f} points for Continuum C2. The complete-minus-global-attention differences are {full_global_attention_peak:+.1f} and {full_global_attention_cont:+.1f} points. With both modulation rules global, the direct-density-input contrast (`global_both` minus `density_free_global`) is {direct_peak:+.1f} points for Peaks and {direct_cont:+.1f} points for Continuum. The full-package contrast is {package_peak:+.1f} and {package_cont:+.1f} points. These contrasts must be interpreted jointly with their three seed means in the paired summary; the full-package comparison cannot uniquely attribute its difference to one component.

Sparse-tail behavior remains adverse and variable: full-density C2 is {tail_full:.1f}%, global-gate C2 is {tail_gate:.1f}%, and global-attention C2 is {tail_attention:.1f}%. No universal-superiority or tail-mechanism claim follows.

All individual core, broad historical peak, continuum, Overall, and sparse-tail results and C1/C2/C3 sensitivity values are preserved in the cell and summary tables. The mechanism slice has one training-subset ordering; it does not establish subset robustness for controls. The fixed classifier used 18,866 events, so this remains efficiency-model data efficiency conditional on pretraining, not end-to-end 5k training.
"""
    )
    reports = HERE / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "mechanism_result.md").write_text(report)
    runtime = time.perf_counter() - started
    result = {
        "status": "complete",
        "runtime_seconds": runtime,
        "mechanism_cells": len(inventory),
        "reused_cells": 30,
        "new_cells": 120,
        "target_events": int(reference["counts"].sum()),
        "target_identity_sha256": target_hash,
        "supported_bins": int((reference["counts"] >= 4).sum()),
        "excluded_bins": int((reference["counts"] < 4).sum()),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    (reports / "mechanism_aggregation_record.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
