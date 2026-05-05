"""HDF5-backed PyTorch Dataset for Majorana Demonstrator waveforms.

The Dataset spans one or more HDF5 files (each holding up to 65 000
events) and yields per-event dicts of tensors. Preprocessing — baseline
subtraction, peak normalization, and ``float32`` cast — is applied
inside ``__getitem__``.

File handles are opened lazily and tracked per worker so the Dataset is
safe to use with ``DataLoader(num_workers > 0)``: ``h5py`` file handles
do not survive ``fork()``.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Literal, TypedDict

import h5py
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch.utils.data import Dataset, get_worker_info

PSDLabel = Literal[
    "psd_label_low_avse",
    "psd_label_high_avse",
    "psd_label_dcr",
    "psd_label_lq",
]


class WaveformItem(TypedDict):
    """One sample yielded by ``MajoranaWaveformDataset.__getitem__``.

    All values are 0-d or 1-d ``torch.Tensor`` so the default
    ``DataLoader`` collation just stacks them along a new batch axis.
    """

    waveform: (
        torch.Tensor
    )  # float32 — preprocessed; default shape (3800,), or (t90_pre + t90_post,) when align_t90=True
    label: torch.Tensor  # float32, scalar — 0.0 or 1.0
    energy: torch.Tensor  # float32, scalar — keV
    tp0: torch.Tensor  # int64, scalar — rising-edge sample index
    detector: torch.Tensor  # int64, scalar
    run_number: torch.Tensor  # int64, scalar
    id: torch.Tensor  # int64, scalar — global event ID


class DatasetConfig(BaseModel):
    """Validated configuration for ``MajoranaWaveformDataset``."""

    model_config = ConfigDict(frozen=True)

    files: list[Path] = Field(
        min_length=1,
        description="HDF5 files to load (read in the order given).",
    )
    target_label: PSDLabel = Field(
        default="psd_label_low_avse",
        description="Which PSD label to expose as the training target.",
    )
    baseline_samples: int = Field(
        default=500,
        gt=0,
        description="Number of leading samples averaged for the baseline.",
    )
    align_t90: bool = Field(
        default=False,
        description=(
            "If True, crop a fixed-length window around the 90% rising-edge "
            "sample (after baseline subtraction and max-normalization). "
            "Output length becomes t90_pre + t90_post. Useful for models "
            "without translation invariance (e.g. MLP)."
        ),
    )
    t90_pre: int = Field(
        default=200,
        gt=0,
        description="Samples before t90 to keep when align_t90=True.",
    )
    t90_post: int = Field(
        default=2000,
        gt=0,
        description="Samples after t90 to keep when align_t90=True.",
    )

    @field_validator("files")
    @classmethod
    def _files_must_exist(cls, files: list[Path]) -> list[Path]:
        for f in files:
            if not f.is_file():
                raise ValueError(f"HDF5 file not found: {f}")
        return files


class MajoranaWaveformDataset(Dataset[WaveformItem]):
    """PyTorch ``Dataset`` over one or more Majorana HDF5 files.

    Indexing is global across all configured files. A cumulative offset
    table is built once at construction from the per-file event counts.

    Per-event preprocessing in ``__getitem__``:

    1. Subtract baseline = mean of the first ``baseline_samples`` samples.
    2. Divide by the max of the baseline-subtracted waveform.
    3. Cast to ``float32``.
    """

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self._lengths = self._scan_lengths()
        self._cumulative = np.cumsum([0, *self._lengths])
        # File handles are opened lazily on first access and reset when
        # execution moves to a new DataLoader worker (see _ensure_handles).
        self._handles: dict[int, h5py.File] = {}
        self._handles_worker_id: int = -2  # sentinel: never opened

    def __len__(self) -> int:
        return int(self._cumulative[-1])

    def __getitem__(self, idx: int) -> WaveformItem:
        self._ensure_handles_for_current_worker()
        file_idx, local_idx = self._locate(idx)
        f = self._handle(file_idx)

        wf = self._preprocess(f["raw_waveform"][local_idx])
        label = float(f[self.config.target_label][local_idx])

        return {
            "waveform": torch.from_numpy(wf),
            "label": torch.tensor(label, dtype=torch.float32),
            "energy": torch.tensor(float(f["energy_label"][local_idx]), dtype=torch.float32),
            "tp0": torch.tensor(int(f["tp0"][local_idx]), dtype=torch.int64),
            "detector": torch.tensor(int(f["detector"][local_idx]), dtype=torch.int64),
            "run_number": torch.tensor(int(f["run_number"][local_idx]), dtype=torch.int64),
            "id": torch.tensor(int(f["id"][local_idx]), dtype=torch.int64),
        }

    def __getstate__(self) -> dict:
        # h5py.File is not picklable; drop handles so the Dataset can be
        # pickled for the spawn multiprocessing context.
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_handles_worker_id"] = -2
        return state

    def __del__(self) -> None:
        for h in self._handles.values():
            with contextlib.suppress(Exception):
                h.close()

    # -- internals -----------------------------------------------------

    def _scan_lengths(self) -> list[int]:
        lengths: list[int] = []
        for path in self.config.files:
            with h5py.File(path, "r") as f:
                lengths.append(int(f["energy_label"].shape[0]))
        return lengths

    def _ensure_handles_for_current_worker(self) -> None:
        info = get_worker_info()
        worker_id = info.id if info is not None else -1
        if worker_id != self._handles_worker_id:
            for h in self._handles.values():
                with contextlib.suppress(Exception):
                    h.close()
            self._handles = {}
            self._handles_worker_id = worker_id

    def _handle(self, file_idx: int) -> h5py.File:
        if file_idx not in self._handles:
            self._handles[file_idx] = h5py.File(self.config.files[file_idx], "r")
        return self._handles[file_idx]

    def _locate(self, idx: int) -> tuple[int, int]:
        n = len(self)
        if idx < 0:
            idx += n
        if not 0 <= idx < n:
            raise IndexError(idx)
        file_idx = int(np.searchsorted(self._cumulative, idx, side="right") - 1)
        local_idx = idx - int(self._cumulative[file_idx])
        return file_idx, local_idx

    def _preprocess(self, wf: np.ndarray) -> np.ndarray:
        # Compute baseline / peak in float64 (HDF5's native dtype) for
        # precision, then cast at the end to keep training in float32.
        baseline = float(wf[: self.config.baseline_samples].mean())
        wf = wf - baseline
        peak = float(wf.max())
        if peak > 0:
            wf = wf / peak
        if self.config.align_t90:
            wf = self._align_to_t90(wf)
        return wf.astype(np.float32, copy=False)

    def _align_to_t90(self, wf: np.ndarray) -> np.ndarray:
        """Crop ``[t90 - pre, t90 + post)`` zero-padded as needed.

        ``t90`` is the first sample at or above 0.9 in the
        max-normalized waveform — the end of the rising edge. Aligning
        to it makes the rising-edge / decay-tail boundary land at the
        same index across events.
        """
        above = np.where(wf >= 0.9)[0]
        t90 = int(above[0]) if above.size else int(np.argmax(wf))
        pre = self.config.t90_pre
        post = self.config.t90_post

        cropped = wf[max(0, t90 - pre) : min(len(wf), t90 + post)]
        front_pad = max(0, pre - t90)
        back_pad = max(0, (t90 + post) - len(wf))
        return np.pad(cropped, (front_pad, back_pad))
