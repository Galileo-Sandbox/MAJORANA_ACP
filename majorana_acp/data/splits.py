"""Resolve a (split, indices) spec to concrete HDF5 file paths.

The Majorana data release names files ``MJD_Train_{N}.hdf5``,
``MJD_Test_{N}.hdf5``, and ``MJD_NPML_{N}.hdf5``. This module exposes a
single helper that turns a list of integer indices (or the shortcut
``"all"``) into an ordered list of paths the Dataset can consume.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

Split = Literal["train", "test", "npml"]
FileIndices = list[int] | Literal["all"]

_PREFIX: dict[Split, str] = {
    "train": "MJD_Train_",
    "test": "MJD_Test_",
    "npml": "MJD_NPML_",
}


def resolve_files(
    data_dir: Path | str,
    split: Split,
    indices: FileIndices = "all",
) -> list[Path]:
    """Return HDF5 file paths for the requested split and indices.

    Files are matched by the standard naming convention
    (e.g. ``MJD_Train_5.hdf5`` for ``split="train"``, ``indices=[5]``).
    The output is ordered to match ``indices``, or numerically when
    ``indices="all"``.

    Args:
        data_dir: Directory containing the ``.hdf5`` files.
        split: Which split to draw from (``"train"`` / ``"test"`` / ``"npml"``).
        indices: Either a list of file indices to load (e.g. ``[0, 1, 2]``)
            or the literal string ``"all"`` to load every available file
            for the split.

    Returns:
        Ordered list of resolved file paths.

    Raises:
        FileNotFoundError: ``data_dir`` does not exist, or one of the
            requested indices has no matching file.
        ValueError: ``indices`` contains negative numbers or duplicates.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    prefix = _PREFIX[split]
    available: dict[int, Path] = {}
    for path in data_dir.glob(f"{prefix}*.hdf5"):
        try:
            available[_parse_index(path, prefix)] = path
        except ValueError:
            # Files that match the glob but don't end in an integer
            # (unlikely in practice) are silently skipped.
            continue

    if indices == "all":
        return [available[k] for k in sorted(available)]

    if any(i < 0 for i in indices):
        raise ValueError(f"indices must be non-negative, got {indices}")
    if len(set(indices)) != len(indices):
        raise ValueError(f"indices contain duplicates: {indices}")

    missing = [i for i in indices if i not in available]
    if missing:
        raise FileNotFoundError(
            f"requested {split} indices not found in {data_dir}: {missing}"
        )
    return [available[i] for i in indices]


def _parse_index(path: Path, prefix: str) -> int:
    return int(path.stem[len(prefix) :])
