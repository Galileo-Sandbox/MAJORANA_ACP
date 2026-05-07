"""A 1D ResNet for waveform classification.

Input  : ``(B, L)`` (when ``in_channels == 1``) or ``(B, C, L)`` for
         multi-channel input. Use ``in_channels=2`` together with
         ``data.use_derivative_channel: true`` to consume the derivative
         channel.
Output : ``(B,)`` raw logits — same single-logit convention as the other
         models in this package. Train with ``BCEWithLogitsLoss``;
         apply ``torch.sigmoid`` at inference for the [0, 1] score.

The architecture mirrors a slim ResNet-18 in 1D:

  Stem: Conv1d(7, stride=2) → Norm → ReLU → MaxPool(3, stride=2)
  4 stages of ``BasicBlock`` (Conv-Norm-ReLU + Conv-Norm + skip), each
    stage doubling channels and downsampling by 2 (except the first).
  AdaptiveAvgPool1d(1) → Dropout → Linear(C, 1).

Each ``BasicBlock`` keeps spatial resolution unless ``stride > 1`` is
requested at the start of a new stage; the skip connection then runs
through a 1×1 Conv1d to match the new channel/stride. ``norm`` selects
BatchNorm1d or GroupNorm for every normalization layer.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from majorana_acp.models._norm import NormType, make_norm_for_conv1d
from majorana_acp.models.registry import register_model


class _BasicBlock1D(nn.Module):
    """Two ``Conv1d`` layers + a skip connection."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        norm: NormType = "batch",
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1 = make_norm_for_conv1d(out_ch, norm=norm, num_groups=num_groups)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn2 = make_norm_for_conv1d(out_ch, norm=norm, num_groups=num_groups)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_ch != out_ch:
            self.downsample: nn.Module | None = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                make_norm_for_conv1d(out_ch, norm=norm, num_groups=num_groups),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


@register_model("resnet1d")
class ResNet1D(nn.Module):
    """ResNet-1D classifier."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        blocks_per_stage: Sequence[int] = (2, 2, 2, 2),
        kernel_size: int = 3,
        dropout: float = 0.2,
        norm: NormType = "batch",
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        if base_channels < 1:
            raise ValueError(f"base_channels must be >= 1, got {base_channels}")
        if not blocks_per_stage:
            raise ValueError("blocks_per_stage must be a non-empty sequence")
        if any(n < 1 for n in blocks_per_stage):
            raise ValueError(
                f"each entry in blocks_per_stage must be >= 1, got {list(blocks_per_stage)}"
            )
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd int, got {kernel_size}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.in_channels = in_channels
        self.norm = norm
        self.num_groups = num_groups

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            make_norm_for_conv1d(base_channels, norm=norm, num_groups=num_groups),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Residual stages: stage 0 keeps spatial res, later stages stride=2.
        layers: list[nn.Module] = []
        in_ch = base_channels
        for i, n_blocks in enumerate(blocks_per_stage):
            stage_ch = base_channels * (2**i)
            stride = 1 if i == 0 else 2
            layers.append(
                _BasicBlock1D(
                    in_ch, stage_ch, kernel_size, stride=stride, norm=norm, num_groups=num_groups
                )
            )
            for _ in range(n_blocks - 1):
                layers.append(
                    _BasicBlock1D(
                        stage_ch,
                        stage_ch,
                        kernel_size,
                        stride=1,
                        norm=norm,
                        num_groups=num_groups,
                    )
                )
            in_ch = stage_ch
        self.stages = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_ch, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d | nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Final classifier: Xavier on the linear (output is a logit, not pre-ReLU).
        last_linear = self.classifier[-1]
        assert isinstance(last_linear, nn.Linear)
        nn.init.xavier_uniform_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, L) only when in_channels=1; otherwise expect (B, C, L).
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x).squeeze(-1)
