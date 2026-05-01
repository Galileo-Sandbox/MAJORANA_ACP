"""A multilayer perceptron baseline for waveform classification.

Input  : ``(B, L)`` float32 tensor — the preprocessed waveform from
         ``MajoranaWaveformDataset`` (default ``L=3800``). A
         ``(B, 1, L)`` tensor is also accepted and flattened.
Output : ``(B,)`` raw logits. No sigmoid — see CLAUDE.md / "Model output
         convention". Use ``BCEWithLogitsLoss`` for training and
         ``torch.sigmoid`` at inference for the [0, 1] score.

The architecture is a stack of ``Linear → BatchNorm1d → ReLU → Dropout``
blocks followed by a linear classifier head. Larger by default than
SimpleCNN because the first layer's weight matrix is
``input_dim × hidden_dims[0]``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from majorana_acp.models.registry import register_model


@register_model("mlp")
class MLP(nn.Module):
    """Stacked Linear + BatchNorm + ReLU + Dropout, then a linear head."""

    def __init__(
        self,
        input_dim: int = 3800,
        hidden_dims: Sequence[int] = (256, 128, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be a positive int, got {input_dim}")
        if not hidden_dims:
            raise ValueError("hidden_dims must be a non-empty sequence")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        hidden_dims = list(hidden_dims)
        self.input_dim = int(input_dim)

        layers: list[nn.Module] = []
        prev_dim = self.input_dim
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, L) or (B, 1, L); flatten to (B, L).
        if x.ndim == 3:
            x = x.flatten(1)
        if x.ndim != 2:
            raise ValueError(f"expected (B, L) or (B, 1, L), got shape {tuple(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"input length {x.shape[-1]} does not match configured input_dim={self.input_dim}"
            )
        return self.network(x).squeeze(-1)
