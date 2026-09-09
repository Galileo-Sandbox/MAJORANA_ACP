#!/usr/bin/env python3
"""Export a validated lightweight result bundle from a server-only run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

MAX_CURVE_BYTES = 2 * 1024 * 1024
REQUIRED_CURVE_ARRAYS = {
    "energy_kev",
    "prediction",
    "prediction_std",
    "bin_centers_kev",
    "bin_counts",
    "empirical_rate",
    "bin_prediction_mean",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    run_dir = (repo / args.run_dir).resolve()
    output_dir = (repo / args.output_dir).resolve()
    if output_dir.exists():
        parser.error(f"Refusing to reuse export directory: {output_dir}")
    summary_path = run_dir / "summary.json"
    curve_path = run_dir / "curve.npz"
    if not summary_path.is_file() or not curve_path.is_file():
        parser.error("Run is missing summary.json or curve.npz.")
    with summary_path.open() as stream:
        summary = json.load(stream)
    if summary.get("status") != "completed":
        parser.error(f"Run status is not completed: {summary.get('status')!r}")
    recorded_curve = summary["server_only_outputs"]["curve"]
    if sha256_file(curve_path) != recorded_curve["sha256"]:
        parser.error("Curve hash does not match the completed run summary.")
    if curve_path.stat().st_size > MAX_CURVE_BYTES:
        parser.error(
            f"Curve is {curve_path.stat().st_size} bytes; limit is {MAX_CURVE_BYTES} bytes."
        )
    with np.load(curve_path) as curve:
        if set(curve.files) != REQUIRED_CURVE_ARRAYS:
            parser.error(
                f"Unexpected curve schema: expected {sorted(REQUIRED_CURVE_ARRAYS)}, observed {sorted(curve.files)}"
            )
        if not (
            curve["energy_kev"].shape == curve["prediction"].shape == curve["prediction_std"].shape
        ):
            parser.error("Dense curve array shapes differ.")
        if not (
            curve["bin_centers_kev"].shape
            == curve["bin_counts"].shape
            == curve["empirical_rate"].shape
            == curve["bin_prediction_mean"].shape
        ):
            parser.error("Binned curve array shapes differ.")

    output_dir.mkdir(parents=True)
    export_curve_path = output_dir / "curve.npz"
    shutil.copyfile(curve_path, export_curve_path)
    portable_summary = {
        key: summary[key]
        for key in (
            "schema_version",
            "run_id",
            "run_kind",
            "status",
            "start_time",
            "end_time",
            "source_commit",
            "model",
            "dependency",
            "protocol",
            "randomness",
            "counts",
            "metrics",
            "runtime",
        )
    }
    portable_summary["reproduction"] = {
        "runner": "ml4phy-paper/scripts/run_experiment.py",
        "model_id": summary["model"]["id"],
        "phase": summary["protocol"]["phase"],
        "context_seed": summary["randomness"]["context_seed"],
        "dropout_seed": summary["randomness"]["dropout_seed"],
        "n_context": summary["counts"]["context_draw"],
        "n_mc": summary["counts"]["mc_passes"],
        "max_context_per_pass": summary["counts"]["context_per_mc_pass"],
    }
    portable_summary["exported_curve"] = {
        "file": "curve.npz",
        "bytes": export_curve_path.stat().st_size,
        "sha256": sha256_file(export_curve_path),
        "contains_event_level_data": False,
    }
    export_summary_path = output_dir / "summary.json"
    with export_summary_path.open("x") as stream:
        json.dump(portable_summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    export_manifest = {
        "schema_version": 1,
        "status": "completed",
        "source_run_id": summary["run_id"],
        "source_run_summary_sha256": sha256_file(summary_path),
        "files": {
            "curve.npz": {
                "bytes": export_curve_path.stat().st_size,
                "sha256": sha256_file(export_curve_path),
            },
            "summary.json": {
                "bytes": export_summary_path.stat().st_size,
                "sha256": sha256_file(export_summary_path),
            },
        },
        "excluded": [
            "event-level predictions and outcomes",
            "event row indices and identities",
            "machine-specific absolute paths",
            "verbose stdout and stderr logs",
            "model checkpoints and training pools",
        ],
    }
    with (output_dir / "manifest.json").open("x") as stream:
        json.dump(export_manifest, stream, indent=2)
        stream.write("\n")
    print(
        f"Exported run {summary['run_id']} to {output_dir}; "
        f"curve size {export_curve_path.stat().st_size} bytes."
    )


if __name__ == "__main__":
    main()
