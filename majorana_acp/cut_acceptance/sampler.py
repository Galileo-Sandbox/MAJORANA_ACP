"""Nearest-neighbor sampler that produces RESUM_FLEX ``StandardBatch`` objects.

We treat ``θ = (E, T)`` as continuous design parameters (``DESIGN_ONLY``,
2D), sample θ_k uniformly from a configured box during training, and
form one "trial" by collecting the ``n_per_trial`` events nearest to
``E_k`` in energy from the appropriate fidelity pool. ``X_ki`` is then
``1[score_i ≥ T_k]``.

This avoids fixed-energy-bin discretization while still satisfying the
RESUM_FLEX contract that all events in a trial share the same θ.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

# RESUM_FLEX schemas — installed as a library, imported directly.
from schemas.data_models import InputMode, StandardBatch


class EventPool:
    """Sorted (by energy) view of the (energy, score) pairs in one fidelity tier."""

    def __init__(self, energy: np.ndarray, score: np.ndarray) -> None:
        energy = np.asarray(energy, dtype=np.float64)
        score = np.asarray(score, dtype=np.float64)
        if energy.ndim != 1 or score.ndim != 1:
            raise ValueError(
                f"energy and score must be 1-D, got energy.shape={energy.shape}, "
                f"score.shape={score.shape}"
            )
        if energy.shape != score.shape:
            raise ValueError(f"energy / score shape mismatch: {energy.shape} vs {score.shape}")
        if energy.size == 0:
            raise ValueError("EventPool cannot be empty")
        order = np.argsort(energy, kind="stable")
        self.energy = np.ascontiguousarray(energy[order])
        self.score = np.ascontiguousarray(score[order])

    def __len__(self) -> int:
        return int(self.energy.size)

    def nearest_indices(self, E_k: float, n: int) -> np.ndarray:
        """Return up to ``n`` indices of events nearest to ``E_k`` in energy.

        If the pool has fewer than ``n`` events, returns all of them in
        ascending distance order (caller decides whether to pad).
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        n_cap = min(n, self.energy.size)
        # Pools are small (O(10^3) at full data, O(10^2) per HF tier) so an
        # explicit argsort is cheap and avoids window-expansion edge cases.
        d = np.abs(self.energy - float(E_k))
        return np.argsort(d, kind="stable")[:n_cap]


def load_event_pool(
    predictions_path: Path | str,
    indices: np.ndarray,
    *,
    base_mask: np.ndarray | None = None,
) -> EventPool:
    """Build an ``EventPool`` from a slice of a classifier's ``predictions.h5``.

    Parameters
    ----------
    predictions_path
        Path to a ``predictions.h5`` produced by ``majorana_acp.cli.evaluate``.
    indices
        Indices into the *filtered* event array (after applying ``base_mask``).
        Typically one of ``PartitionedIndices.{lf, hf_train, hf_holdout}``.
    base_mask
        Optional boolean mask over the full HDF5 arrays to apply before
        ``indices`` are interpreted (matches ``PartitionedIndices.base_mask``).
        If ``None``, ``indices`` are taken as raw indices into the file.
    """
    with h5py.File(Path(predictions_path), "r") as f:
        energy = f["energy"][:]
        score = f["score"][:]
    if base_mask is not None:
        if base_mask.shape != energy.shape:
            raise ValueError(
                f"base_mask shape {base_mask.shape} does not match energy shape {energy.shape}"
            )
        energy = energy[base_mask]
        score = score[base_mask]
    return EventPool(energy=energy[indices], score=score[indices])


class DesignOnlySampler:
    """Duck-typed batch generator for ``train_cnp`` / MFGP-prep paths.

    Implements the RESUM_FLEX generator interface
    (``mode``, ``dim_theta``, ``dim_phi``, ``generate(...)``).

    Each ``generate(B, N, seed)`` call samples ``B`` continuous
    ``θ_k = (E_k, T_k)`` uniformly from
    ``energy_box × threshold_box``, finds the ``N`` nearest events in
    the pool by energy, and binarizes their scores at ``T_k``.
    """

    mode = InputMode.DESIGN_ONLY
    dim_theta: int = 2
    dim_phi: None = None

    def __init__(
        self,
        pool: EventPool,
        energy_box: tuple[float, float],
        threshold_box: tuple[float, float],
        n_per_trial: int,
        *,
        allow_replacement: bool = True,
    ) -> None:
        if n_per_trial < 1:
            raise ValueError(f"n_per_trial must be >= 1, got {n_per_trial}")
        if not energy_box[1] > energy_box[0]:
            raise ValueError(f"energy_box must satisfy hi > lo, got {energy_box}")
        if not threshold_box[1] > threshold_box[0]:
            raise ValueError(f"threshold_box must satisfy hi > lo, got {threshold_box}")
        if not allow_replacement and len(pool) < n_per_trial:
            raise ValueError(
                f"pool has {len(pool)} events but n_per_trial={n_per_trial} requested "
                "and replacement is disabled — pass allow_replacement=True or reduce "
                "n_per_trial"
            )
        self.pool = pool
        self.energy_box = (float(energy_box[0]), float(energy_box[1]))
        self.threshold_box = (float(threshold_box[0]), float(threshold_box[1]))
        self.n_per_trial = int(n_per_trial)
        self.allow_replacement = bool(allow_replacement)

    def generate(self, n_trials: int, n_events: int, seed: int) -> StandardBatch:
        """Sample a ``StandardBatch`` of ``B = n_trials`` trials × ``N = n_events`` events.

        ``n_events`` is *requested*; if the pool has fewer events and
        ``allow_replacement=False`` the constructor would have already
        rejected. With replacement on, missing events are filled by
        uniform sampling with replacement from the full pool.
        """
        if n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {n_trials}")
        if n_events < 1:
            raise ValueError(f"n_events must be >= 1, got {n_events}")

        rng = np.random.default_rng(int(seed))
        e_lo, e_hi = self.energy_box
        t_lo, t_hi = self.threshold_box

        # θ_k = (E_k, T_k), one row per trial.
        e_k = rng.uniform(e_lo, e_hi, size=n_trials)
        t_k = rng.uniform(t_lo, t_hi, size=n_trials)
        theta = np.stack([e_k, t_k], axis=-1).astype(np.float64)

        labels = np.zeros((n_trials, n_events), dtype=np.int8)
        for k in range(n_trials):
            near = self.pool.nearest_indices(e_k[k], n_events)
            if near.size < n_events:
                # Pad with replacement — only reached when len(pool) < n_events.
                pad = rng.integers(0, len(self.pool), size=n_events - near.size)
                near = np.concatenate([near, pad])
            labels[k] = (self.pool.score[near] >= t_k[k]).astype(np.int8)

        return StandardBatch(
            mode=InputMode.DESIGN_ONLY,
            theta=theta,
            phi=None,
            labels=labels,
        )
