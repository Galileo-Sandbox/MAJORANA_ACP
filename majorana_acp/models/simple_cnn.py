"""A small 1D CNN baseline for waveform classification.

Input  : ``(B, L)`` float32 tensor — the preprocessed waveform from
         ``MajoranaWaveformDataset`` (baseline-subtracted, max-normalized).
Output : ``(B,)`` raw logits. No sigmoid — see CLAUDE.md / "Model output
         convention". Use ``BCEWithLogitsLoss`` for training and
         ``torch.sigmoid`` at inference for the [0, 1] score.

The architecture is a stack of Conv1d → BatchNorm → ReLU → MaxPool(4)
blocks followed by global average pooling and a linear head. Defaults
give 64x temporal downsampling over three blocks, suitable for the
3800-sample waveform.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from majorana_acp.models.registry import register_model


@register_model("simple_cnn")
class SimpleCNN(nn.Module):
    """Stacked Conv1d + BatchNorm + MaxPool, then a linear classifier head."""

    def __init__(
        self,
        channels: Sequence[int] = (16, 32, 64),
        kernel_size: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("channels must be a non-empty sequence")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd int, got {kernel_size}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        channels = list(channels)
        layers: list[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            layers.extend(
                [
                    nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=4),
                ]
            )
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[-1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, L) or (B, 1, L); standardize to (B, 1, L).
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = self.pool(x).flatten(1)  # (B, C_last)
        return self.classifier(x).squeeze(-1)  # (B,)
