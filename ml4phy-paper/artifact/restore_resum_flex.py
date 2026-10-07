#!/usr/bin/env python3
"""Validate and safely extract the pinned historical RESUM_FLEX snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[2]
COMPATIBILITY = REPO / "ml4phy-paper/manifests/resum_flex_compatibility.json"
DEFAULT_DESTINATION = REPO / "ml4phy-paper/local/resum-flex-edba6a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_member(member: tarfile.TarInfo) -> None:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive path: {member.name}")
    if member.issym() or member.islnk() or member.isdev():
        raise ValueError(f"Unsupported archive member type: {member.name}")


def verify_sources(root: Path, expected: dict[str, str]) -> None:
    for relative, digest in expected.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"Required source is missing after extraction: {relative}")
        observed = sha256(path)
        if observed != digest:
            raise ValueError(f"Source hash mismatch for {relative}: {observed}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--check-only", action="store_true", help="validate without extracting")
    args = parser.parse_args()

    archive = args.archive.resolve()
    destination = args.destination.resolve()
    compatibility = json.loads(COMPATIBILITY.read_text())
    observed_archive_hash = sha256(archive)
    expected_archive_hash = compatibility["local_archive_sha256"]
    if observed_archive_hash != expected_archive_hash:
        raise ValueError(
            f"Archive hash mismatch: expected {expected_archive_hash}, "
            f"observed {observed_archive_hash}"
        )

    with tarfile.open(archive, "r:*") as bundle:
        members = bundle.getmembers()
        for member in members:
            validate_member(member)
        with tempfile.TemporaryDirectory(prefix="resum-flex-") as temp:
            temp_root = Path(temp)
            bundle.extractall(temp_root, members=members, filter="data")
            verify_sources(temp_root, compatibility["source_hashes"])
            if not args.check_only:
                if destination.exists():
                    raise FileExistsError(f"Destination already exists: {destination}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(temp_root), str(destination))

    print(
        json.dumps(
            {
                "status": "passed",
                "archive": str(archive),
                "archive_sha256": observed_archive_hash,
                "historical_commit": compatibility["commit"],
                "extracted": not args.check_only,
                "destination": str(destination) if not args.check_only else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
