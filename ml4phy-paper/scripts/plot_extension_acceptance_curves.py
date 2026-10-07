#!/usr/bin/env python3
"""Plot completed Phase 1 acceptance curves at the two prespecified sizes."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONTEXT_SIZES = (500, 2000)
METHOD_ORDER = (
    "CNP",
    "Attentive CNP",
    "Attentive CNP + PE",
    "Density-guided CNP (ours)",
    "Gaussian kernel regression (context only)",
    "Gaussian kernel regression (pooled data + context)",
)
COLORS = {
    "CNP": "#4c78a8",
    "Attentive CNP": "#f58518",
    "Attentive CNP + PE": "#54a24b",
    "Density-guided CNP (ours)": "#e45756",
    "Gaussian kernel regression (context only)": "#72b7b2",
    "Gaussian kernel regression (pooled data + context)": "#b279a2",
}
LINESTYLES = {
    "CNP": "-",
    "Attentive CNP": "-",
    "Attentive CNP + PE": "-",
    "Density-guided CNP (ours)": "-",
    "Gaussian kernel regression (context only)": "--",
    "Gaussian kernel regression (pooled data + context)": ":",
}
LOCAL_WINDOWS = (
    (1565.0, 1645.0, "Tl-208 DEP and Bi-212 1620.74 keV"),
    (1700.0, 2000.0, "Continuum 1700–2000 keV"),
    (2068.0, 2135.0, "Tl-208 SE"),
    (2575.0, 2640.0, "Tl-208 FE"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_curves(neural_path: Path, kernel_path: Path) -> dict[tuple[str, int], tuple[list[float], list[float]]]:
    curves: dict[tuple[str, int], tuple[list[float], list[float]]] = {}
    neural = read_csv(neural_path)
    kernel = read_csv(kernel_path)
    for method in METHOD_ORDER[:4]:
        for size in CONTEXT_SIZES:
            selected = [
                row for row in neural
                if row["method"] == method and int(row["context_size"]) == size
            ]
            curves[(method, size)] = (
                [float(row["energy_kev"]) for row in selected],
                [float(row["mean_of_training_seed_context_means"]) for row in selected],
            )
    for method in METHOD_ORDER[4:]:
        for size in CONTEXT_SIZES:
            selected = [
                row for row in kernel
                if row["method"] == method and int(row["context_size"]) == size
            ]
            curves[(method, size)] = (
                [float(row["energy_kev"]) for row in selected],
                [float(row["mean_across_contexts"]) for row in selected],
            )
    for key, (energy, value) in curves.items():
        if len(energy) != 2501 or len(value) != 2501:
            raise ValueError(f"Incomplete 1-keV curve for {key}: {len(energy)} rows")
    return curves


def plot_reference(ax, reference: list[dict[str, str]], low: float, high: float) -> None:
    selected = [
        row for row in reference
        if low <= float(row["bin_center_kev"]) <= high
        and row["supported_minimum_four_events"] == "True"
    ]
    ax.scatter(
        [float(row["bin_center_kev"]) for row in selected],
        [float(row["empirical_acceptance"]) for row in selected],
        s=13,
        color="#777777",
        alpha=0.50,
        label="Finite reference (5-keV bins)",
        zorder=1,
    )


def plot_methods(ax, curves, size: int, low: float, high: float) -> None:
    for method in METHOD_ORDER:
        energy, value = curves[(method, size)]
        selected = [index for index, point in enumerate(energy) if low <= point <= high]
        ax.plot(
            [energy[index] for index in selected],
            [value[index] for index in selected],
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=1.25 if method == "Attentive CNP + PE" else 1.7,
            alpha=0.90,
            label=method,
            zorder=2,
        )


def main() -> None:
    repo = Path(".").resolve()
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--short"], text=True
    ).splitlines()
    if status:
        raise RuntimeError("Curve plotting requires a clean worktree: " + "; ".join(status))
    source_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    root = repo / "ml4phy-paper"
    neural_path = root / "tables/phase1_neural_acceptance_curves.csv"
    kernel_path = root / "tables/phase1_kernel_acceptance_curves.csv"
    reference_path = root / "tables/final_reference_bins_5kev.csv"
    output_paths = {
        "full": root / "figures/phase1_extension_acceptance_full.png",
        "local": root / "figures/phase1_extension_acceptance_local.png",
        "manifest": root / "manifests/phase1_extension_acceptance_curves.json",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))

    curves = load_curves(neural_path, kernel_path)
    reference = read_csv(reference_path)

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True, sharey=True)
    for ax, size in zip(axes, CONTEXT_SIZES):
        plot_reference(ax, reference, 500.0, 3000.0)
        plot_methods(ax, curves, size, 500.0, 3000.0)
        ax.set_title(f"Context size {size:,}")
        ax.set_ylabel("Acceptance")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.22)
    axes[-1].set_xlabel("Energy (keV)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9.2)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(output_paths["full"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
    for row_index, size in enumerate(CONTEXT_SIZES):
        for column_index, (low, high, title) in enumerate(LOCAL_WINDOWS):
            ax = axes[row_index, column_index]
            plot_reference(ax, reference, low, high)
            plot_methods(ax, curves, size, low, high)
            ax.set_xlim(low, high)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{title}\nn={size:,}", fontsize=11)
            ax.set_xlabel("Energy (keV)")
            if column_index == 0:
                ax.set_ylabel("Acceptance")
            ax.grid(alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9.2)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(output_paths["local"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    outputs = {
        str(path.relative_to(repo)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in output_paths.items()
        if name != "manifest"
    }
    manifest = {
        "schema_version": 1,
        "analysis": "completed-method full-range and local acceptance curves",
        "status": "completed",
        "source_commit": source_commit,
        "script": str(Path(__file__).resolve().relative_to(repo)),
        "script_sha256": sha256_file(Path(__file__)),
        "inputs": {
            str(path.relative_to(repo)): sha256_file(path)
            for path in (neural_path, kernel_path, reference_path)
        },
        "context_sizes": list(CONTEXT_SIZES),
        "methods": list(METHOD_ORDER),
        "curve_summary": "Neural lines average three training-seed means, each over ten contexts; kernel lines average ten contexts.",
        "uncertainty_note": "No uncertainty bands are drawn because neural training/context variation and kernel context variation are not the same quantity.",
        "reference_note": "Reference points are finite 5-keV-bin estimates with at least four events, not exact truth.",
        "outputs": outputs,
        "historical_data_exposure": "Prospectively specified follow-up analysis on historically exposed data; not an untouched test.",
    }
    with output_paths["manifest"].open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
