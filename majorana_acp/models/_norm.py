"""Helpers for swapping ``BatchNorm``, ``GroupNorm``, and ``LayerNorm``.

Three norm types are supported across the package:

- ``"batch"`` — ``BatchNorm1d``. Per-channel running statistics; the
  classic choice but introduces a train/eval discrepancy.
- ``"group"`` — ``GroupNorm`` over channel groups within each example.
  Only meaningful for ``(B, C, L)`` activations (1D conv models).
- ``"layer"`` — ``LayerNorm`` over the feature dim. Only meaningful for
  ``(B, F)`` activations (MLP / flat layers).

Each helper accepts the full ``NormType`` literal and raises a clear
``ValueError`` when the requested norm is not applicable to its target
shape — so a config like ``norm: group`` on an MLP or ``norm: layer``
on a conv block fails fast at model construction.
"""

from __future__ import annotations

from typing import Literal

from torch import nn

NormType = Literal["batch", "group", "layer"]


def make_norm_for_conv1d(
    num_features: int, norm: NormType = "batch", num_groups: int = 8
) -> nn.Module:
    """Norm layer for ``(B, C, L)`` activations.

    Supports ``"batch"`` and ``"group"``; rejects ``"layer"`` because
    the canonical "LayerNorm for ConvNets" choice (``GroupNorm(1, C)``)
    is just a special case of ``GroupNorm`` and we'd rather the user
    pick ``num_groups`` explicitly.
    """
    if norm == "batch":
        return nn.BatchNorm1d(num_features)
    if norm == "group":
        if num_features % num_groups != 0:
            raise ValueError(
                f"GroupNorm requires num_features ({num_features}) to be divisible by "
                f"num_groups ({num_groups})"
            )
        return nn.GroupNorm(num_groups, num_features)
    if norm == "layer":
        raise ValueError(
            "norm='layer' is not applicable to 1D conv layers (B, C, L). "
            "Use 'batch' or 'group' for conv models; 'layer' is reserved for "
            "flat (B, F) activations such as MLP."
        )
    raise ValueError(f"unknown norm type: {norm!r} (expected 'batch', 'group', or 'layer')")


def make_norm_for_flat(num_features: int, norm: NormType = "batch") -> nn.Module:
    """Norm layer for ``(B, F)`` activations (MLP / fully-connected blocks).

    Supports ``"batch"`` and ``"layer"``; rejects ``"group"`` because
    GroupNorm on a flat tensor would group features and is not a
    standard MLP normalization choice.
    """
    if norm == "batch":
        return nn.BatchNorm1d(num_features)
    if norm == "layer":
        return nn.LayerNorm(num_features)
    if norm == "group":
        raise ValueError(
            "norm='group' is not applicable to flat (B, F) activations. "
            "Use 'batch' or 'layer' for MLPs; 'group' is reserved for "
            "(B, C, L) activations such as 1D conv models."
        )
    raise ValueError(f"unknown norm type: {norm!r} (expected 'batch', 'group', or 'layer')")
