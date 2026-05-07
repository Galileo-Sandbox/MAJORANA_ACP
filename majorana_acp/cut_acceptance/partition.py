"""Partition test-set events into LF / HF-train / HF-holdout fidelity tiers.

Rule (matches the project plan; see ``CutAcceptanceConfig``):

- Apply ``energy_range`` first, then keep only events whose true label
  matches ``target_class`` (1 = signal acceptance, 0 = background
  rejection).
- Events whose energy lies inside *any* peak window are "peak events".
  Everything else is "continuum".
- Continuum events go to LF.
- Peak events are split 10 / 30 / 60 (configurable via
  ``CutAcceptanceConfig.peak_split``) into LF / HF train / HF holdout.
- The three sets are guaranteed disjoint by construction; we assert it.

The same ``partition_seed`` is intentionally reused across the
signal-acceptance and background-rejection pipelines so the LF / HF /
holdout partition is reproducible across both runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from majorana_acp.cut_acceptance.config import CutAcceptanceConfig, PeakWindow


@dataclass(frozen=True)
class PartitionedIndices:
    """Index arrays into the *energy- and class-filtered* event list.

    ``base_mask`` is the boolean mask we applied to the raw HDF5 arrays
    to get the filtered list; ``lf``, ``hf_train``, ``hf_holdout`` are
    indices into the filtered list (0..N_filtered-1).
    """

    base_mask: np.ndarray  # shape (N_raw,), bool — events that survived the energy + class filter
    lf: np.ndarray  # shape (n_lf,), int — indices into the filtered array
    hf_train: np.ndarray  # shape (n_hf_train,), int
    hf_holdout: np.ndarray  # shape (n_hf_holdout,), int

    @property
    def n_filtered(self) -> int:
        return int(self.base_mask.sum())

    def disjoint(self) -> bool:
        a = set(self.lf.tolist())
        b = set(self.hf_train.tolist())
        c = set(self.hf_holdout.tolist())
        return not (a & b) and not (a & c) and not (b & c)


def _in_any_window(energy: np.ndarray, windows: list[PeakWindow]) -> np.ndarray:
    """Return a boolean mask: True where energy is inside *any* of the windows."""
    out = np.zeros_like(energy, dtype=bool)
    for w in windows:
        out |= (energy >= w.lo) & (energy <= w.hi)
    return out


def partition_events(
    energy: np.ndarray,
    label: np.ndarray,
    cfg: CutAcceptanceConfig,
) -> PartitionedIndices:
    """Build the LF / HF-train / HF-holdout index sets for one run.

    Parameters
    ----------
    energy : (N_raw,) float array — per-event energy in keV
    label  : (N_raw,) int array — true binary label per event
    cfg    : ``CutAcceptanceConfig``

    Returns
    -------
    ``PartitionedIndices`` with disjoint LF / HF train / HF holdout indices.
    """
    if energy.shape != label.shape:
        raise ValueError(f"energy / label shape mismatch: {energy.shape} vs {label.shape}")

    e_lo, e_hi = cfg.energy_range
    base_mask = (energy >= e_lo) & (energy <= e_hi) & (label == cfg.target_class)
    n_filtered = int(base_mask.sum())
    if n_filtered == 0:
        raise ValueError(
            f"no events survive energy_range={cfg.energy_range} and target_class={cfg.target_class}"
        )

    energy_f = energy[base_mask]
    in_peak = _in_any_window(energy_f, cfg.peak_windows)

    continuum_idx = np.flatnonzero(~in_peak)
    peak_idx = np.flatnonzero(in_peak)

    n_peak = peak_idx.size
    if n_peak < 3:
        raise ValueError(
            f"only {n_peak} peak event(s) survived the filter; cannot split 10/30/60. "
            "Loosen energy_range or peak_windows, or lower the test-set "
            "subset_portion in the upstream eval."
        )

    # Reproducible peak split using the configured seed.
    rng = np.random.default_rng(cfg.partition_seed)
    perm = rng.permutation(n_peak)
    n_lf_peak = int(round(n_peak * cfg.peak_split.lf))
    n_hf_train = int(round(n_peak * cfg.peak_split.hf_train))
    # holdout = the remainder, so the three counts sum to n_peak exactly.
    n_hf_holdout = n_peak - n_lf_peak - n_hf_train
    if n_hf_holdout < 0:
        raise ValueError(
            f"rounding gave negative holdout count: lf={n_lf_peak} train={n_hf_train} "
            f"out of {n_peak} — adjust peak_split fractions"
        )

    lf_peak = peak_idx[perm[:n_lf_peak]]
    hf_train = peak_idx[perm[n_lf_peak : n_lf_peak + n_hf_train]]
    hf_holdout = peak_idx[perm[n_lf_peak + n_hf_train :]]

    # LF = continuum ∪ lf_peak.
    lf = np.concatenate([continuum_idx, lf_peak])
    lf.sort()

    out = PartitionedIndices(
        base_mask=base_mask,
        lf=lf.astype(np.int64),
        hf_train=hf_train.astype(np.int64),
        hf_holdout=hf_holdout.astype(np.int64),
    )
    if not out.disjoint():
        raise AssertionError("partition_events produced overlapping LF/HF/HF-holdout sets")
    n_total = out.lf.size + out.hf_train.size + out.hf_holdout.size
    if n_total != n_filtered:
        raise AssertionError(
            f"partition_events lost or duplicated events: {n_total} placed vs {n_filtered} filtered"
        )
    return out
