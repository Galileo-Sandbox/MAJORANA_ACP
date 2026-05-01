"""Shared pytest fixtures.

Synthetic HDF5 files mimic the real Majorana data-release schema but are
small enough to make tests fast. Real data lives at
``/home/klz/Data/MAJORANA`` and is never touched by the unit tests.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest


def _write_synthetic_hdf5(
    path: Path,
    n_events: int,
    *,
    low_avse_pass_rate: float = 0.5,
    id_offset: int = 0,
    seed: int | None = None,
) -> Path:
    """Write a Majorana-shaped HDF5 file with ``n_events`` synthetic events.

    Each waveform has a baseline near 1000 ADC with small Gaussian noise
    and a step at sample 1000 of height 500, so preprocessing should
    yield a baseline ~0 and peak == 1.
    """
    rng = np.random.default_rng(seed if seed is not None else hash(str(path)) % (2**32))

    wf = rng.normal(1000.0, 5.0, size=(n_events, 3800)).astype(np.float64)
    wf[:, 1000:] += 500.0

    with h5py.File(path, "w") as f:
        f.create_dataset("raw_waveform", data=wf)
        f.create_dataset(
            "energy_label",
            data=rng.uniform(100.0, 2700.0, size=n_events).astype(np.float64),
        )
        f.create_dataset(
            "psd_label_low_avse",
            data=(rng.random(n_events) < low_avse_pass_rate).astype(bool),
        )
        for k in ("psd_label_high_avse", "psd_label_dcr", "psd_label_lq"):
            f.create_dataset(k, data=np.ones(n_events, dtype=bool))
        f.create_dataset("tp0", data=np.full(n_events, 1000, dtype=np.int64))
        f.create_dataset(
            "detector",
            data=rng.integers(100, 300, size=n_events, dtype=np.int64),
        )
        f.create_dataset("run_number", data=np.full(n_events, 12345, dtype=np.int64))
        f.create_dataset(
            "id",
            data=np.arange(id_offset, id_offset + n_events, dtype=np.int64),
        )
    return path


@pytest.fixture
def tiny_train_file(tmp_path: Path) -> Path:
    """Single-file fixture: ``MJD_Train_0.hdf5`` with 8 events.

    Uses a fixed seed so the synthetic ``psd_label_low_avse`` always has
    both classes present — otherwise tests like the balanced sampler can
    fail intermittently under Python's hash randomization.
    """
    return _write_synthetic_hdf5(tmp_path / "MJD_Train_0.hdf5", n_events=8, seed=0)


@pytest.fixture
def tiny_train_dir(tmp_path: Path) -> Path:
    """Multi-file fixture exercising cross-file indexing.

    Layout:
      MJD_Train_0.hdf5 — 5 events, ids 0..4
      MJD_Train_1.hdf5 — 7 events, ids 5..11
      MJD_Train_2.hdf5 — 3 events, ids 12..14
      MJD_Test_0.hdf5  — 2 events  (off-split, used to verify glob filtering)
    """
    sizes = [5, 7, 3]
    offset = 0
    for i, n in enumerate(sizes):
        _write_synthetic_hdf5(
            tmp_path / f"MJD_Train_{i}.hdf5",
            n_events=n,
            id_offset=offset,
            seed=i,
        )
        offset += n
    # Test file: 16 events with mixed labels so eval / ROC-AUC tests have
    # both classes present. Pass rate ~0.6 to nudge a non-degenerate split.
    _write_synthetic_hdf5(
        tmp_path / "MJD_Test_0.hdf5",
        n_events=16,
        low_avse_pass_rate=0.6,
        seed=99,
    )
    return tmp_path
