#!/usr/bin/env python3
"""Reanalyze the five historical prediction caches without model inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODEL_PATHS = {
    "True CNP": "cache/inference/simple_cnn_small/true_cnp/inclusive_bin10.npz",
    "Base 1": "cache/inference/simple_cnn_small/sweeps/base1_matched/inclusive_bin10.npz",
    "Base 3": "cache/inference/simple_cnn_small/sweeps/base3_matched/inclusive_bin10.npz",
    "Cell 15 v5": "cache/inference/simple_cnn_small/sweeps/cell15_v5/inclusive_bin10.npz",
    "Cell 17": "cache/inference/simple_cnn_small/sweeps/cell17/inclusive_bin10.npz",
}

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

BANDWIDTHS_KEV = (2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
CV_SEED = 20260908
PROBABILITY_CLIP = 1.0e-6
MIN_REGION_EVENTS = 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_fingerprint(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def region_mask(energy: np.ndarray, bounds: tuple[float, float], *, full: bool = False):
    lo, hi = bounds
    return (energy >= lo) & (energy <= hi if full else energy < hi)


def predictive_metrics(prediction: np.ndarray, outcome: np.ndarray) -> dict[str, float]:
    clipped = np.clip(prediction, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
    return {
        "brier": float(np.mean((prediction - outcome) ** 2)),
        "log_loss": float(
            -np.mean(outcome * np.log(clipped) + (1 - outcome) * np.log(1 - clipped))
        ),
    }


def nw_predict(
    train_energy: np.ndarray,
    train_outcome: np.ndarray,
    query_energy: np.ndarray,
    bandwidth_kev: float,
    *,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian Nadaraya-Watson prediction and effective sample size."""
    predictions = np.empty(query_energy.size, dtype=np.float64)
    effective_n = np.empty(query_energy.size, dtype=np.float64)
    for start in range(0, query_energy.size, chunk_size):
        stop = min(start + chunk_size, query_energy.size)
        scaled = (query_energy[start:stop, None] - train_energy[None, :]) / bandwidth_kev
        log_weight = -0.5 * scaled**2
        log_weight -= np.max(log_weight, axis=1, keepdims=True)
        weight = np.exp(log_weight)
        weight_sum = weight.sum(axis=1)
        predictions[start:stop] = (weight @ train_outcome) / weight_sum
        effective_n[start:stop] = weight_sum**2 / np.square(weight).sum(axis=1)
    return predictions, effective_n


def select_kernel_bandwidth(energy: np.ndarray, outcome: np.ndarray):
    rng = np.random.default_rng(CV_SEED)
    fold_id = np.empty(energy.size, dtype=np.int64)
    fold_id[rng.permutation(energy.size)] = np.arange(energy.size) % 5
    rows = []
    for bandwidth in BANDWIDTHS_KEV:
        fold_scores = []
        for fold in range(5):
            valid = fold_id == fold
            prediction, _ = nw_predict(energy[~valid], outcome[~valid], energy[valid], bandwidth)
            fold_scores.append(float(np.mean((prediction - outcome[valid]) ** 2)))
        rows.append(
            {
                "bandwidth_kev": bandwidth,
                "mean_cv_brier": float(np.mean(fold_scores)),
                "std_cv_brier": float(np.std(fold_scores, ddof=1)),
                **{f"fold_{index}_brier": value for index, value in enumerate(fold_scores)},
            }
        )
    selected = min(rows, key=lambda row: (row["mean_cv_brier"], row["bandwidth_kev"]))
    return selected["bandwidth_kev"], rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("ml4phy-paper"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    output_root = (repo / args.output_root).resolve()

    outputs = {
        "event_csv": output_root / "tables/cache_reanalysis_event_metrics.csv",
        "bin_csv": output_root / "tables/cache_reanalysis_bin_metrics.csv",
        "support_csv": output_root / "tables/cache_reanalysis_support.csv",
        "kernel_csv": output_root / "tables/cache_reanalysis_kernel_bandwidth_cv.csv",
        "metrics_json": output_root / "tables/cache_reanalysis_metrics.json",
        "full_figure": output_root / "figures/cache_full_range_historical.png",
        "region_figure": output_root / "figures/cache_regions_historical.png",
        "report": output_root / "reports/cache_reanalysis.md",
        "manifest": output_root / "manifests/cache_reanalysis.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        parser.error("Refusing to overwrite existing outputs: " + ", ".join(existing))
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    caches = {}
    input_manifest = {}
    for model, relative in MODEL_PATHS.items():
        path = repo / relative
        if not path.is_file():
            parser.error(f"Missing cache for {model}: {path}")
        with np.load(path) as archive:
            caches[model] = {key: archive[key].copy() for key in archive.files}
        input_manifest[model] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    reference = caches["True CNP"]
    shared_keys = (
        "e_C",
        "s_C",
        "e_T",
        "s_T",
        "e_train",
        "s_train",
        "bin_centers",
        "dense_energies",
    )
    shared_fingerprints = {}
    for key in shared_keys:
        fingerprints = {model: array_fingerprint(cache[key]) for model, cache in caches.items()}
        if len(set(fingerprints.values())) != 1:
            raise ValueError(f"Caches do not share {key}: {fingerprints}")
        shared_fingerprints[key] = next(iter(fingerprints.values()))
    scalar_keys = ("T_star", "n_total", "n_context_total", "n_target_total", "n_context_per_pass")
    for key in scalar_keys:
        values = {model: cache[key].item() for model, cache in caches.items()}
        if len(set(values.values())) != 1:
            raise ValueError(f"Caches do not share {key}: {values}")

    threshold = float(reference["T_star"])
    context_outcome = (reference["s_C"] >= threshold).astype(np.float64)
    target_outcome = (reference["s_T"] >= threshold).astype(np.float64)
    selected_bandwidth, kernel_cv_rows = select_kernel_bandwidth(reference["e_C"], context_outcome)
    kernel_target, kernel_target_ess = nw_predict(
        reference["e_C"], context_outcome, reference["e_T"], selected_bandwidth
    )
    kernel_dense, _ = nw_predict(
        reference["e_C"], context_outcome, reference["dense_energies"], selected_bandwidth
    )

    target_predictions = {
        model: np.interp(reference["e_T"], cache["dense_energies"], cache["dense_beta"])
        for model, cache in caches.items()
    }
    target_predictions[f"Kernel (context-only, h={selected_bandwidth:g} keV)"] = kernel_target

    event_rows = []
    for model, prediction in target_predictions.items():
        region_brier = []
        for region, bounds in REGIONS.items():
            mask = region_mask(reference["e_T"], bounds, full=region == "full")
            metrics = predictive_metrics(prediction[mask], target_outcome[mask])
            status = (
                "supported"
                if region == "full" or int(mask.sum()) >= MIN_REGION_EVENTS
                else "inconclusive"
            )
            if region != "full" and status == "supported":
                region_brier.append(metrics["brier"])
            event_rows.append(
                {
                    "model": model,
                    "region": region,
                    "status": status,
                    "n_events": int(mask.sum()),
                    "brier": metrics["brier"],
                    "log_loss": metrics["log_loss"],
                    "prediction_source": "dense-grid linear interpolation"
                    if model in caches
                    else "event-query kernel",
                }
            )
        event_rows.append(
            {
                "model": model,
                "region": "equal_region_mean",
                "status": "supported_regions_only",
                "n_events": "",
                "brier": float(np.mean(region_brier)),
                "log_loss": "",
                "prediction_source": "unweighted mean across predefined regions with at least 20 target events",
            }
        )

    bin_rows = []
    for model, cache in caches.items():
        for region, bounds in REGIONS.items():
            mask = region_mask(cache["bin_centers"], bounds, full=region == "full")
            valid = mask & np.isfinite(cache["rate"]) & np.isfinite(cache["beta"])
            residual = cache["beta"][valid] - cache["rate"][valid]
            bin_rows.append(
                {
                    "model": model,
                    "region": region,
                    "n_valid_bins": int(valid.sum()),
                    "n_target_events_in_valid_bins": int(cache["n_target_per_bin"][valid].sum()),
                    "mae": float(np.mean(np.abs(residual))) if valid.any() else "",
                    "rmse": float(np.sqrt(np.mean(residual**2))) if valid.any() else "",
                    "prediction_location": "bin center",
                }
            )

    expected_centers = np.arange(505.0, 3000.0, 10.0)
    support_rows = []
    for region, bounds in REGIONS.items():
        expected = region_mask(expected_centers, bounds, full=region == "full")
        retained = region_mask(reference["bin_centers"], bounds, full=region == "full")
        target = region_mask(reference["e_T"], bounds, full=region == "full")
        context = region_mask(reference["e_C"], bounds, full=region == "full")
        support_rows.append(
            {
                "region": region,
                "lower_kev": bounds[0],
                "upper_kev": bounds[1],
                "expected_10kev_bins": int(expected.sum()),
                "retained_bins": int(retained.sum()),
                "omitted_bins": int(expected.sum() - retained.sum()),
                "context_events": int(context.sum()),
                "target_events": int(target.sum()),
                "kernel_target_events_ess_below_5": int(np.sum(target & (kernel_target_ess < 5.0))),
                "status": "supported"
                if region == "full" or int(target.sum()) >= MIN_REGION_EVENTS
                else "inconclusive",
            }
        )

    roughness = []
    for model, cache in caches.items():
        spacing = float(np.diff(cache["dense_energies"]).mean())
        for region in ("continuum_1700_2000", "continuum_2200_2400"):
            mask = region_mask(cache["dense_energies"], REGIONS[region])
            values = cache["dense_beta"][mask]
            roughness.append(
                {
                    "model": model,
                    "region": region,
                    "grid_spacing_kev": spacing,
                    "mean_absolute_second_difference": float(np.mean(np.abs(np.diff(values, n=2)))),
                }
            )

    write_csv(outputs["event_csv"], event_rows)
    write_csv(outputs["bin_csv"], bin_rows)
    write_csv(outputs["support_csv"], support_rows)
    write_csv(outputs["kernel_csv"], kernel_cv_rows)
    metrics_payload = {
        "evidence_status": ["historical", "fixed_context", "interpolated_where_applicable"],
        "threshold": threshold,
        "threshold_provenance": "Legacy Youden-J selection on the same 7,074-event evaluation file; not independent.",
        "counts": {
            "training": int(reference["e_train"].size),
            "context": int(reference["e_C"].size),
            "target": int(reference["e_T"].size),
            "retained_bins": int(reference["bin_centers"].size),
            "dense_grid": int(reference["dense_energies"].size),
        },
        "kernel": {
            "selection": "five-fold context-only cross-validation by Brier score",
            "cv_seed": CV_SEED,
            "candidate_bandwidths_kev": BANDWIDTHS_KEV,
            "selected_bandwidth_kev": selected_bandwidth,
        },
        "event_metrics": event_rows,
        "bin_metrics": bin_rows,
        "support": support_rows,
        "continuum_roughness": roughness,
    }
    with outputs["metrics_json"].open("x") as stream:
        json.dump(metrics_payload, stream, indent=2, allow_nan=False)
        stream.write("\n")

    palette = plt.get_cmap("tab10").colors
    colors = {model: palette[index] for index, model in enumerate(caches)}
    fig, axis = plt.subplots(figsize=(10.5, 4.8))
    valid = np.isfinite(reference["rate"])
    axis.errorbar(
        reference["bin_centers"][valid],
        reference["rate"][valid],
        yerr=np.vstack(
            [
                reference["rate"][valid] - reference["rate_lo"][valid],
                reference["rate_hi"][valid] - reference["rate"][valid],
            ]
        ),
        fmt=".",
        color="0.55",
        alpha=0.65,
        markersize=3,
        linewidth=0.6,
        label="Historical target empirical rate",
    )
    for model, cache in caches.items():
        axis.plot(
            cache["dense_energies"],
            cache["dense_beta"],
            label=model,
            color=colors[model],
            linewidth=1.35,
        )
    axis.plot(
        reference["dense_energies"],
        kernel_dense,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Kernel, context-only (h={selected_bandwidth:g} keV)",
    )
    axis.set(
        xlabel="Energy (keV)", ylabel="Inclusive acceptance", xlim=(500, 3000), ylim=(-0.02, 1.02)
    )
    axis.set_title("Historical fixed-context cache comparison (interpolated curves)")
    axis.grid(alpha=0.2)
    axis.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(outputs["full_figure"], dpi=220)
    plt.close(fig)

    panel_regions = [
        "DEP",
        "Bi-214",
        "continuum_1700_2000",
        "SE",
        "continuum_2200_2400",
        "FE",
        "sparse_tail",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2), sharey=True)
    for axis, region in zip(axes.flat, panel_regions, strict=False):
        lo, hi = REGIONS[region]
        bin_mask = region_mask(
            reference["bin_centers"], (lo, hi), full=region == "sparse_tail"
        ) & np.isfinite(reference["rate"])
        axis.errorbar(
            reference["bin_centers"][bin_mask],
            reference["rate"][bin_mask],
            yerr=np.vstack(
                [
                    reference["rate"][bin_mask] - reference["rate_lo"][bin_mask],
                    reference["rate_hi"][bin_mask] - reference["rate"][bin_mask],
                ]
            ),
            fmt=".",
            color="0.5",
            markersize=4,
            linewidth=0.7,
        )
        for model, cache in caches.items():
            dense_mask = region_mask(
                cache["dense_energies"], (lo, hi), full=region == "sparse_tail"
            )
            axis.plot(
                cache["dense_energies"][dense_mask],
                cache["dense_beta"][dense_mask],
                color=colors[model],
                linewidth=1.1,
            )
        kernel_mask = region_mask(
            reference["dense_energies"], (lo, hi), full=region == "sparse_tail"
        )
        axis.plot(
            reference["dense_energies"][kernel_mask],
            kernel_dense[kernel_mask],
            color="black",
            linestyle="--",
            linewidth=1.0,
        )
        axis.set_title(region.replace("_", " "))
        axis.set_xlim(lo, hi)
        axis.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    for axis in axes[-1, :3]:
        axis.set_xlabel("Energy (keV)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Inclusive acceptance")
    handles = [plt.Line2D([], [], color=colors[name], label=name) for name in caches]
    handles.append(plt.Line2D([], [], color="black", linestyle="--", label="Context-only kernel"))
    axes.flat[-1].legend(handles=handles, loc="center", fontsize=9)
    fig.suptitle("Historical fixed-context regional diagnostics", y=1.01)
    fig.tight_layout()
    fig.savefig(outputs["region_figure"], dpi=220, bbox_inches="tight")
    plt.close(fig)

    full_rows = {row["model"]: row for row in event_rows if row["region"] == "full"}
    equal_rows = {row["model"]: row for row in event_rows if row["region"] == "equal_region_mean"}
    ranking = sorted(full_rows, key=lambda model: full_rows[model]["brier"])
    report_lines = [
        "# Cache-Only Reanalysis",
        "",
        "Status tags: **historical**, **fixed context**, and **interpolated where applicable**. No model inference or training was run.",
        "",
        "## Protocol",
        "",
        f"All five caches share 18,866 training, 2,000 context, and 5,074 target energy-score pairs. Predictions at target event energies are linear interpolations of the saved 800-point dense grid. The fixed threshold is `{threshold:.15f}` and was historically selected by Youden-J on the same 7,074-event evaluation file; these numbers are development evidence, not independent final-test evidence.",
        "",
        f"The context-only Gaussian Nadaraya-Watson baseline selected `{selected_bandwidth:g} keV` from {list(BANDWIDTHS_KEV)} by five-fold context-only Brier score (seed {CV_SEED}), then refit all 2,000 context events. Target outcomes were not used for bandwidth selection.",
        "",
        "## Full-range and equal-region Brier score",
        "",
        "| Model | Full-range Brier | Equal-region mean Brier | Full-range log loss |",
        "|---|---:|---:|---:|",
    ]
    for model in ranking:
        report_lines.append(
            f"| {model} | {full_rows[model]['brier']:.6f} | {equal_rows[model]['brier']:.6f} | {full_rows[model]['log_loss']:.6f} |"
        )
    report_lines.extend(
        [
            "",
            "The full-range ranking is descriptive only. Regional errors, event support, and continuum roughness must be considered jointly; a single favorable peak bin or legacy p-value is not treated as evidence of shape recovery.",
            "",
            "## Support and exclusions",
            "",
            "| Region | Status | Context events | Target events | Expected 10-keV bins | Retained bins | Omitted bins |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in support_rows:
        report_lines.append(
            f"| {row['region']} | {row['status']} | {row['context_events']} | {row['target_events']} | {row['expected_10kev_bins']} | {row['retained_bins']} | {row['omitted_bins']} |"
        )
    report_lines.extend(
        [
            "",
            f"Regions with fewer than {MIN_REGION_EVENTS} target events are marked inconclusive and excluded from the equal-region summary. The cached bin predictions are evaluated at bin centers, while event-level scores use interpolation. Neither reconstructs unsaved sub-grid structure. The caches contain no event IDs, class labels, independent threshold calibration, per-MC draws, or checkpoint hashes.",
            "",
            "## Artifacts",
            "",
            "- `tables/cache_reanalysis_event_metrics.csv`: event-level Brier score and log loss by region.",
            "- `tables/cache_reanalysis_bin_metrics.csv`: bin-center MAE/RMSE with explicit valid-bin counts.",
            "- `tables/cache_reanalysis_support.csv`: event and retained-bin support, including the sparse tail.",
            "- `tables/cache_reanalysis_kernel_bandwidth_cv.csv`: context-only bandwidth selection scores.",
            "- `tables/cache_reanalysis_metrics.json`: machine-readable aggregate results and roughness diagnostics.",
            "- `figures/cache_full_range_historical.png` and `figures/cache_regions_historical.png`: figures with historical status in their labels.",
            "",
        ]
    )
    with outputs["report"].open("x") as stream:
        stream.write("\n".join(report_lines))

    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    output_hashes = {}
    for key, path in outputs.items():
        if key == "manifest":
            continue
        try:
            display_path = str(path.relative_to(repo))
        except ValueError:
            display_path = str(path)
        output_hashes[display_path] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {
        "status": "completed",
        "evidence_status": ["historical", "fixed_context", "interpolated_where_applicable"],
        "source_commit": head,
        "command": "MPLCONFIGDIR=ml4phy-paper/local/matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python ml4phy-paper/scripts/cache_reanalysis.py --repo . --output-root ml4phy-paper",
        "script_sha256": sha256_file(Path(__file__)),
        "inputs": input_manifest,
        "shared_array_fingerprints": shared_fingerprints,
        "threshold": threshold,
        "threshold_provenance": "Legacy Youden-J selection on the same evaluation file.",
        "counts": metrics_payload["counts"],
        "kernel_selection": metrics_payload["kernel"],
        "outputs": output_hashes,
    }
    with outputs["manifest"].open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")

    print(
        f"Wrote {len(outputs)} cache-reanalysis artifacts; selected kernel bandwidth {selected_bandwidth:g} keV."
    )


if __name__ == "__main__":
    main()
