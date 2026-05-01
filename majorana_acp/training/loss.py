"""Loss functions and class-imbalance helpers.

Three strategies are supported by the ``loss`` config block (and can be
combined freely via ``balanced_sampler``):

* ``bce``          — vanilla BCEWithLogitsLoss.
* ``weighted_bce`` — BCEWithLogitsLoss with ``pos_weight`` (auto from
  training labels, or a fixed float).
* ``focal``        — focal loss with focusing parameter ``focal_gamma``.

``build_balanced_sampler`` produces a ``WeightedRandomSampler`` that
upsamples the minority class; it is orthogonal to the loss type.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import WeightedRandomSampler

from majorana_acp.training.config import LossConfig


class BinaryFocalLoss(nn.Module):
    """Binary focal loss with logits.

    ``L = mean( (1 - p_t)^gamma * BCE(logits, target) )``,
    where ``p_t`` is the predicted probability of the *correct* class.
    Reduces to BCE when ``gamma == 0``.
    """

    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * target + (1.0 - p) * (1.0 - target)
        return ((1.0 - p_t) ** self.gamma * bce).mean()


def _read_labels(files: list[Path], target_label: str) -> np.ndarray:
    """Read and concatenate the boolean target label across all files."""
    arrays: list[np.ndarray] = []
    for f in files:
        with h5py.File(f, "r") as h:
            arrays.append(h[target_label][:].astype(bool))
    return np.concatenate(arrays)


def compute_pos_weight(files: list[Path], target_label: str) -> float:
    """Return ``N_negative / N_positive`` over the labels in ``files``."""
    labels = _read_labels(files, target_label)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0:
        raise ValueError(f"No positive examples found for {target_label!r}")
    return n_neg / n_pos


def build_balanced_sampler(files: list[Path], target_label: str) -> WeightedRandomSampler:
    """Sampler that draws each class with equal expected frequency."""
    labels = _read_labels(files, target_label)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Cannot balance {target_label!r}: only one class present "
            f"(n_pos={n_pos}, n_neg={n_neg})"
        )
    weights = np.where(labels, 1.0 / n_pos, 1.0 / n_neg)
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=len(weights),
        replacement=True,
    )


def build_loss(
    cfg: LossConfig,
    files: list[Path] | None = None,
    target_label: str | None = None,
) -> nn.Module:
    """Instantiate the loss module specified by ``cfg``.

    For ``type='weighted_bce'`` with ``pos_weight='auto'``, ``files`` and
    ``target_label`` must be supplied so the weight can be computed from
    the training labels.
    """
    if cfg.type == "bce":
        return nn.BCEWithLogitsLoss()

    if cfg.type == "weighted_bce":
        if cfg.pos_weight == "auto":
            if files is None or target_label is None:
                raise ValueError("pos_weight='auto' requires files and target_label")
            pw = compute_pos_weight(files, target_label)
        else:
            pw = float(cfg.pos_weight)
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, dtype=torch.float32))

    if cfg.type == "focal":
        return BinaryFocalLoss(gamma=cfg.focal_gamma)

    raise ValueError(f"Unknown loss type: {cfg.type!r}")
