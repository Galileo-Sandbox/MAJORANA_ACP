"""Tests for ``majorana_acp.data``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from torch.utils.data import DataLoader

from majorana_acp.data import (
    DatasetConfig,
    MajoranaWaveformDataset,
    resolve_files,
)

# --- splits.resolve_files --------------------------------------------


def test_resolve_files_all(tiny_train_dir: Path) -> None:
    files = resolve_files(tiny_train_dir, "train", "all")
    assert [f.name for f in files] == [
        "MJD_Train_0.hdf5",
        "MJD_Train_1.hdf5",
        "MJD_Train_2.hdf5",
    ]


def test_resolve_files_indices_preserves_order(tiny_train_dir: Path) -> None:
    files = resolve_files(tiny_train_dir, "train", [2, 0])
    assert [f.name for f in files] == [
        "MJD_Train_2.hdf5",
        "MJD_Train_0.hdf5",
    ]


def test_resolve_files_split_isolation(tiny_train_dir: Path) -> None:
    train = resolve_files(tiny_train_dir, "train", "all")
    test = resolve_files(tiny_train_dir, "test", "all")
    assert all("Train" in f.name for f in train)
    assert all("Test" in f.name for f in test)
    assert len(test) == 1


def test_resolve_files_missing_data_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_files(tmp_path / "does_not_exist", "train", "all")


def test_resolve_files_missing_index(tiny_train_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_files(tiny_train_dir, "train", [0, 99])


def test_resolve_files_duplicate_indices(tiny_train_dir: Path) -> None:
    with pytest.raises(ValueError):
        resolve_files(tiny_train_dir, "train", [0, 0])


def test_resolve_files_negative_index(tiny_train_dir: Path) -> None:
    with pytest.raises(ValueError):
        resolve_files(tiny_train_dir, "train", [-1])


# --- DatasetConfig validation ----------------------------------------


def test_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        DatasetConfig(files=[tmp_path / "no.hdf5"])


def test_config_rejects_empty_file_list() -> None:
    with pytest.raises(ValidationError):
        DatasetConfig(files=[])


def test_config_rejects_zero_baseline(tiny_train_file: Path) -> None:
    with pytest.raises(ValidationError):
        DatasetConfig(files=[tiny_train_file], baseline_samples=0)


def test_config_rejects_unknown_target_label(tiny_train_file: Path) -> None:
    with pytest.raises(ValidationError):
        DatasetConfig(files=[tiny_train_file], target_label="not_a_label")


# --- MajoranaWaveformDataset -----------------------------------------


def test_dataset_length_single_file(tiny_train_file: Path) -> None:
    ds = MajoranaWaveformDataset(DatasetConfig(files=[tiny_train_file]))
    assert len(ds) == 8


def test_dataset_length_multi_file(tiny_train_dir: Path) -> None:
    files = resolve_files(tiny_train_dir, "train", "all")
    ds = MajoranaWaveformDataset(DatasetConfig(files=files))
    assert len(ds) == 5 + 7 + 3


def test_dataset_item_keys_and_dtypes(tiny_train_file: Path) -> None:
    ds = MajoranaWaveformDataset(DatasetConfig(files=[tiny_train_file]))
    item = ds[0]

    assert set(item.keys()) == {
        "waveform",
        "label",
        "energy",
        "tp0",
        "detector",
        "run_number",
        "id",
    }
    assert item["waveform"].dtype == torch.float32
    assert item["waveform"].shape == (3800,)
    assert item["label"].dtype == torch.float32
    assert item["energy"].dtype == torch.float32
    for k in ("tp0", "detector", "run_number", "id"):
        assert item[k].dtype == torch.int64


def test_dataset_label_is_zero_or_one(tiny_train_file: Path) -> None:
    ds = MajoranaWaveformDataset(DatasetConfig(files=[tiny_train_file]))
    for i in range(len(ds)):
        assert ds[i]["label"].item() in (0.0, 1.0)


def test_dataset_preprocessing_baseline_and_peak(tiny_train_file: Path) -> None:
    """Synthetic waveforms have flat baseline + step; preprocessing should
    leave the first 500 samples near 0 and the peak at exactly 1.0."""
    ds = MajoranaWaveformDataset(DatasetConfig(files=[tiny_train_file]))
    wf = ds[0]["waveform"].numpy()
    # Baseline (first 500) ~ 0 — small noise allowed.
    assert abs(wf[:500].mean()) < 0.05
    # Max-normalized peak == 1.
    assert wf.max() == pytest.approx(1.0, abs=1e-6)


def test_dataset_target_label_switch(tiny_train_file: Path) -> None:
    """Switching ``target_label`` selects a different HDF5 dataset.

    The fixture sets ``psd_label_high_avse`` to all-True, so changing the
    target should make every label 1.0.
    """
    ds = MajoranaWaveformDataset(
        DatasetConfig(files=[tiny_train_file], target_label="psd_label_high_avse")
    )
    for i in range(len(ds)):
        assert ds[i]["label"].item() == 1.0


def test_dataset_cross_file_indexing(tiny_train_dir: Path) -> None:
    """Global indices map across files; ids should be a contiguous range."""
    files = resolve_files(tiny_train_dir, "train", "all")
    ds = MajoranaWaveformDataset(DatasetConfig(files=files))
    ids = [ds[i]["id"].item() for i in range(len(ds))]
    assert ids == list(range(15))


def test_dataset_negative_index(tiny_train_file: Path) -> None:
    ds = MajoranaWaveformDataset(DatasetConfig(files=[tiny_train_file]))
    assert ds[-1]["id"].item() == ds[len(ds) - 1]["id"].item()


def test_dataset_index_out_of_range(tiny_train_file: Path) -> None:
    ds = MajoranaWaveformDataset(DatasetConfig(files=[tiny_train_file]))
    with pytest.raises(IndexError):
        _ = ds[len(ds)]
    with pytest.raises(IndexError):
        _ = ds[-len(ds) - 1]


def test_dataset_align_t90_off_keeps_full_length(tiny_train_file: Path) -> None:
    ds = MajoranaWaveformDataset(DatasetConfig(files=[tiny_train_file]))
    assert ds[0]["waveform"].shape == (3800,)


def test_dataset_align_t90_default_window_is_2200(tiny_train_file: Path) -> None:
    """Default t90_pre=200 + t90_post=2000 produces length-2200 waveforms."""
    ds = MajoranaWaveformDataset(DatasetConfig(files=[tiny_train_file], align_t90=True))
    assert ds[0]["waveform"].shape == (2200,)


def test_dataset_align_t90_custom_window_shape(tiny_train_file: Path) -> None:
    ds = MajoranaWaveformDataset(
        DatasetConfig(files=[tiny_train_file], align_t90=True, t90_pre=100, t90_post=300)
    )
    assert ds[0]["waveform"].shape == (400,)


def test_dataset_align_t90_places_crossing_at_pre(tiny_train_file: Path) -> None:
    """The first sample crossing 0.9 in the aligned waveform should land
    at index ``t90_pre``."""
    pre, post = 150, 200
    ds = MajoranaWaveformDataset(
        DatasetConfig(files=[tiny_train_file], align_t90=True, t90_pre=pre, t90_post=post)
    )
    wf = ds[0]["waveform"].numpy()
    above = np.where(wf >= 0.9)[0]
    assert above.size > 0
    # The synthetic fixture's rising edge is a clean step at sample 1000,
    # so the first crossing in the original waveform is at sample 1000;
    # after alignment, that lands exactly at index `pre`.
    assert above[0] == pre


def test_dataset_works_with_dataloader(tiny_train_dir: Path) -> None:
    """End-to-end: default DataLoader collation stacks dict items."""
    files = resolve_files(tiny_train_dir, "train", "all")
    ds = MajoranaWaveformDataset(DatasetConfig(files=files))
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    assert batch["waveform"].shape == (4, 3800)
    assert batch["waveform"].dtype == torch.float32
    assert batch["label"].shape == (4,)
    assert batch["id"].dtype == torch.int64
