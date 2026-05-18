"""Binned-energy sampler that produces RESUM_FLEX ``StandardBatch`` objects.

The CNP consumes 2-D design parameters θ = (E, T):

* ``E`` is a discrete energy-bin center (``cfg.energy_bin_width`` wide),
  normalized to ``[0, 1]`` against ``cfg.energy_range`` before being
  handed to the encoder. We do *not* sample E continuously — the whole
  point of switching to binning is to remove the nearest-neighbor
  smoothing the continuous-θ pipeline used.
* ``T`` is the score threshold, sampled with a "boundary mix" so the
  CNP sees plenty of trials near ``T=0`` and ``T=1`` (the bimodal
  CNN-score distribution makes the empirical β(T) plateau across the
  interior and cliff at both endpoints; uniform-T training under-samples
  the cliffs).

Per trial: pick a bin (uniformly over eligible bins), pick ``T_k``,
draw ``n_per_trial`` events with replacement from that bin's pool, and
set ``X_ki = 1[score_i ≥ T_k]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import h5py
import numpy as np

# RESUM_FLEX schemas — installed as a library, imported directly.
from schemas.data_models import InputMode, StandardBatch

TSamplingStrategy = Literal["uniform", "boundary_mix"]


def load_events(
    predictions_path: Path | str,
    *,
    target_class: int | str,
    energy_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Load (energy, score) arrays from a classifier's predictions.h5.

    Filters by ``target_class`` (0 / 1 / "all") and by ``energy_range``.
    """
    with h5py.File(Path(predictions_path), "r") as f:
        energy = f["energy"][:].astype(np.float64)
        score = f["score"][:].astype(np.float64)
        label = f["label"][:].astype(np.int64)

    e_lo, e_hi = energy_range
    keep = (energy >= e_lo) & (energy <= e_hi)
    if target_class != "all":
        keep &= label == int(target_class)
    return energy[keep], score[keep]


class _BinIndex:
    """Pre-computed map from bin index → event indices in the pool.

    Keeps only bins with ``>= min_events_per_bin``.
    """

    def __init__(
        self,
        energy: np.ndarray,
        score: np.ndarray,
        *,
        energy_range: tuple[float, float],
        bin_width: float,
        min_events_per_bin: int,
    ) -> None:
        e_lo, e_hi = energy_range
        if bin_width <= 0:
            raise ValueError(f"bin_width must be > 0, got {bin_width}")
        if not e_hi > e_lo:
            raise ValueError(f"energy_range must satisfy hi > lo, got {energy_range}")
        if energy.shape != score.shape:
            raise ValueError(f"energy / score shape mismatch: {energy.shape} vs {score.shape}")

        edges = np.arange(e_lo, e_hi + 0.5 * bin_width, bin_width, dtype=np.float64)
        if edges[-1] < e_hi:
            edges = np.append(edges, e_hi)
        bin_idx = np.clip(np.digitize(energy, edges) - 1, 0, edges.size - 2)

        kept_bins: list[int] = []
        bin_events: list[np.ndarray] = []
        for b in range(edges.size - 1):
            ev = np.flatnonzero(bin_idx == b)
            if ev.size >= min_events_per_bin:
                kept_bins.append(b)
                bin_events.append(ev)

        if not kept_bins:
            raise ValueError(
                f"No bins meet min_events_per_bin={min_events_per_bin} for "
                f"bin_width={bin_width}; pool has {energy.size} events."
            )

        self.energy = energy
        self.score = score
        self.edges = edges
        # Centers of *kept* bins, parallel to ``self.bin_events``.
        centers = 0.5 * (edges[:-1] + edges[1:])
        self.kept_bin_indices = np.asarray(kept_bins, dtype=np.int64)
        self.bin_centers = centers[self.kept_bin_indices]
        self.bin_events = bin_events

    def __len__(self) -> int:
        return self.kept_bin_indices.size


class BinnedSampler:
    """Per-bin batch generator for RESUM_FLEX ``train_cnp``.

    Pre-bins the input events and emits trials where each trial's events
    are drawn (with replacement) from one bin's pool. ``θ_k`` is the
    pair ``(E_bin_center_normalized, T_normalized)``.
    """

    mode = InputMode.DESIGN_ONLY
    dim_theta: int = 2
    dim_phi: None = None

    def __init__(
        self,
        energy: np.ndarray,
        score: np.ndarray,
        *,
        energy_range: tuple[float, float],
        energy_bin_width: float,
        threshold_range: tuple[float, float] = (0.0, 1.0),
        n_per_trial: int = 32,
        min_events_per_bin: int = 4,
        t_sampling: TSamplingStrategy = "boundary_mix",
    ) -> None:
        if n_per_trial < 1:
            raise ValueError(f"n_per_trial must be >= 1, got {n_per_trial}")
        if not threshold_range[1] > threshold_range[0]:
            raise ValueError(f"threshold_range must satisfy hi > lo, got {threshold_range}")
        if t_sampling not in ("uniform", "boundary_mix"):
            raise ValueError(f"t_sampling must be 'uniform' or 'boundary_mix', got {t_sampling!r}")

        self._index = _BinIndex(
            energy,
            score,
            energy_range=energy_range,
            bin_width=energy_bin_width,
            min_events_per_bin=min_events_per_bin,
        )
        self.energy_range = (float(energy_range[0]), float(energy_range[1]))
        self.energy_bin_width = float(energy_bin_width)
        self.threshold_range = (float(threshold_range[0]), float(threshold_range[1]))
        self.n_per_trial = int(n_per_trial)
        self.t_sampling: TSamplingStrategy = t_sampling

    # ----- introspection helpers (used by the pipeline + diagnostics) -----

    @property
    def n_bins(self) -> int:
        return len(self._index)

    @property
    def bin_centers(self) -> np.ndarray:
        return self._index.bin_centers

    @property
    def bin_event_counts(self) -> np.ndarray:
        return np.asarray([ev.size for ev in self._index.bin_events], dtype=np.int64)

    # ----- T sampler -----

    def _sample_t(self, rng: np.random.Generator, n_trials: int) -> np.ndarray:
        t_lo, t_hi = self.threshold_range
        if self.t_sampling == "uniform":
            return rng.uniform(t_lo, t_hi, size=n_trials)
        band = 0.05 * (t_hi - t_lo)
        which = rng.choice(3, size=n_trials, p=[0.5, 0.25, 0.25])
        t_k = np.empty(n_trials, dtype=np.float64)
        m0, m1, m2 = which == 0, which == 1, which == 2
        if m0.any():
            t_k[m0] = rng.uniform(t_lo, t_hi, size=int(m0.sum()))
        if m1.any():
            t_k[m1] = rng.uniform(t_lo, t_lo + band, size=int(m1.sum()))
        if m2.any():
            t_k[m2] = rng.uniform(t_hi - band, t_hi, size=int(m2.sum()))
        return t_k

    # ----- StandardBatch generator (RESUM_FLEX duck-type) -----

    def generate(self, n_trials: int, n_events: int, seed: int) -> StandardBatch:
        if n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {n_trials}")
        if n_events < 1:
            raise ValueError(f"n_events must be >= 1, got {n_events}")

        rng = np.random.default_rng(int(seed))
        # Pick one bin per trial uniformly over the eligible bins.
        bin_choices = rng.integers(0, self.n_bins, size=n_trials)
        e_k = self.bin_centers[bin_choices]
        t_k = self._sample_t(rng, n_trials)

        labels = np.zeros((n_trials, n_events), dtype=np.int8)
        for k in range(n_trials):
            ev = self._index.bin_events[int(bin_choices[k])]
            # Always sample with replacement — handles sparse bins uniformly
            # and gives the CNP a consistent context size N regardless of
            # how many events live in the bin.
            picks = rng.integers(0, ev.size, size=n_events)
            ev_idx = ev[picks]
            labels[k] = (self._index.score[ev_idx] >= t_k[k]).astype(np.int8)

        e_lo, e_hi = self.energy_range
        t_lo, t_hi = self.threshold_range
        theta = np.stack(
            [
                (e_k - e_lo) / (e_hi - e_lo),
                (t_k - t_lo) / (t_hi - t_lo),
            ],
            axis=-1,
        ).astype(np.float64)

        return StandardBatch(
            mode=InputMode.DESIGN_ONLY,
            theta=theta,
            phi=None,
            labels=labels,
        )
