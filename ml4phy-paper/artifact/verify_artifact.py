#!/usr/bin/env python3
"""Verify portable paper outputs and optional server-only source artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import lzma
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO / "ml4phy-paper/manifests/artifact_release_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_inside_repo(relative: str) -> Path:
    candidate = (REPO / relative).resolve()
    if not candidate.is_relative_to(REPO.resolve()):
        raise ValueError(f"Path escapes repository: {relative}")
    return candidate


def verify_record(relative: str, record: dict[str, Any], errors: list[str]) -> None:
    path = resolve_inside_repo(relative)
    if not path.is_file():
        errors.append(f"missing: {relative}")
        return
    observed = sha256(path)
    if observed != record["sha256"]:
        errors.append(f"hash mismatch: {relative}: {observed}")
    if "bytes" in record and path.stat().st_size != int(record["bytes"]):
        errors.append(f"size mismatch: {relative}: {path.stat().st_size}")


def verify_output_manifest(
    relative: str, expected_hash: str, output_key: str, errors: list[str]
) -> int:
    path = resolve_inside_repo(relative)
    if not path.is_file():
        errors.append(f"missing output manifest: {relative}")
        return 0
    observed = sha256(path)
    if observed != expected_hash:
        errors.append(f"manifest hash mismatch: {relative}: {observed}")
        return 0
    payload = json.loads(path.read_text())
    outputs = payload.get(output_key)
    if not isinstance(outputs, dict):
        errors.append(f"invalid {output_key} mapping: {relative}")
        return 0
    for output_path, record in outputs.items():
        verify_record(output_path, record, errors)
    return len(outputs)


def verify_inventory_expectations(manifest: dict[str, Any], errors: list[str]) -> None:
    coverage = json.loads(
        resolve_inside_repo("ml4phy-paper/manifests/paper_coverage_export.json").read_text()
    )
    mechanism = json.loads(
        resolve_inside_repo(
            "ml4phy-paper/review_followup_20260910/manifests/provenance_v1.json"
        ).read_text()
    )
    expected = manifest["expected_inventory"]
    checks = coverage["checks"]
    if checks["source_record_count"] != expected["presentation_cells"]:
        errors.append("presentation cell inventory changed")
    inventory = mechanism["inventory"]
    pairs = {
        "mechanism_training_jobs": "new_training_jobs_completed",
        "mechanism_evaluations": "new_valid_evaluations_completed",
        "mechanism_cells": "mechanism_cells_including_reuse",
    }
    for expected_key, observed_key in pairs.items():
        if inventory[observed_key] != expected[expected_key]:
            errors.append(f"mechanism inventory changed: {observed_key}")
    validation = mechanism["validation"]
    if validation["status"] != "passed" or validation["tests_passed"] != 24:
        errors.append("recorded mechanism validation is not complete")


def verify_server_artifacts(manifest: dict[str, Any], errors: list[str]) -> int:
    presentation_path = resolve_inside_repo("ml4phy-paper/manifests/paper_presentation_export.json")
    presentation = json.loads(presentation_path.read_text())
    checked: dict[str, str] = {}
    for record in presentation["source_artifacts"]:
        checked[record["event_predictions_path"]] = record["event_predictions_sha256"]
        checked[record["curve_path"]] = record["curve_sha256"]

    registry_path = resolve_inside_repo(
        "ml4phy-paper/review_followup_20260910/tables/mechanism_run_inventory.csv"
    )
    with registry_path.open(newline="") as stream:
        for record in csv.DictReader(stream):
            checked[record["event_path"]] = record["event_sha256"]
            checked[record["curve_path"]] = record["curve_sha256"]

    checkpoint_path = resolve_inside_repo(
        "ml4phy-paper/review_followup_20260910/tables/mechanism_checkpoint_inventory.csv"
    )
    with checkpoint_path.open(newline="") as stream:
        for record in csv.DictReader(stream):
            checked[record["checkpoint_path"]] = record["checkpoint_sha256"]

    for relative, record in manifest["external_server_inputs"].items():
        checked[relative] = record["sha256"]

    for relative, expected in sorted(checked.items()):
        verify_record(relative, {"sha256": expected}, errors)
    return len(checked)


def verify_compressed_mechanism_rows(errors: list[str]) -> int:
    path = resolve_inside_repo(
        "ml4phy-paper/review_followup_20260910/tables/mechanism_cell_bins.csv.xz"
    )
    counts: dict[int, int] = {}
    with lzma.open(path, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            index = int(row["cell_index"])
            counts[index] = counts.get(index, 0) + 1
    if set(counts) != set(range(150)):
        errors.append("mechanism compressed table does not contain cells 0 through 149")
    if set(counts.values()) != {500}:
        errors.append("mechanism compressed table does not contain 500 bins per cell")
    return sum(counts.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--server",
        action="store_true",
        help="also hash server-only predictions and checkpoints",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    errors: list[str] = []

    for relative, record in manifest["portable_anchors"].items():
        verify_record(relative, record, errors)

    verified_outputs = 0
    for collection in manifest["portable_output_collections"]:
        verified_outputs += verify_output_manifest(
            collection["manifest"],
            collection["sha256"],
            collection["output_key"],
            errors,
        )

    verify_inventory_expectations(manifest, errors)
    mechanism_bin_rows = verify_compressed_mechanism_rows(errors)
    server_files = verify_server_artifacts(manifest, errors) if args.server else 0

    result = {
        "status": "failed" if errors else "passed",
        "release_schema_version": manifest["schema_version"],
        "portable_anchors": len(manifest["portable_anchors"]),
        "portable_outputs": verified_outputs,
        "mechanism_bin_rows": mechanism_bin_rows,
        "server_mode": args.server,
        "server_files": server_files,
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
