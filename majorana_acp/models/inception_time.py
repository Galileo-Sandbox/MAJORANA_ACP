"""InceptionTime, a 1D time-series classifier with multi-scale convs.

Each :class:`InceptionModule` looks at its input through four branches:

  - a 1×1 *bottleneck* conv to compress channels (skipped when the input
    has only one channel),
  - three parallel ``Conv1d`` branches at *different* kernel sizes
    (default ``(10, 20, 40)``) so the model can attend to features at
    several temporal scales simultaneously,
  - a ``MaxPool1d`` branch followed by a 1×1 conv for local
    translation-invariance.

The four outputs are concatenated along the channel axis, BatchNormed,
and ReLU'd. ``padding='same'`` everywhere so the temporal length is
preserved through the deep stack — exactly what we need for waveforms
where the rising edge is at a fixed sample index.

:class:`InceptionResidualBlock` chains three modules and adds a skip
connection from the block's input to its output (with a 1×1 conv on
the skip path to match channels). Two residual blocks make up the
default backbone, followed by ``AdaptiveAvgPool1d(1)`` (GAP) and a
linear classifier head emitting a single logit per event — same output
convention as the other models in this package.

Reference: Fawaz et al., *InceptionTime: Finding AlexNet for time
series classification* (Data Min Knowl Disc, 2020).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from majorana_acp.models.registry import register_model


class InceptionModule(nn.Module):
    """One Inception block (bottleneck + multi-scale convs + maxpool branch)."""

    def __init__(
        self,
        in_channels: int,
        n_filters: int = 32,
        kernel_sizes: Sequence[int] = (10, 20, 40),
        bottleneck_channels: int = 32,
        use_bottleneck: bool = True,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        if n_filters < 1:
            raise ValueError(f"n_filters must be >= 1, got {n_filters}")
        if not kernel_sizes:
            raise ValueError("kernel_sizes must be a non-empty sequence")
        if any(k < 1 for k in kernel_sizes):
            raise ValueError(f"each kernel size must be >= 1, got {list(kernel_sizes)}")

        # Bottleneck only makes sense when we actually have channels to compress.
        self.use_bottleneck = bool(use_bottleneck) and in_channels > 1
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            branch_in = bottleneck_channels
        else:
            self.bottleneck = nn.Identity()
            branch_in = in_channels

        self.conv_branches = nn.ModuleList(
            [
                nn.Conv1d(branch_in, n_filters, kernel_size=int(k), padding="same", bias=False)
                for k in kernel_sizes
            ]
        )

        # Max-pool branch reads from the *original* input (not the bottlenecked
        # one) and uses a 1x1 conv to project to n_filters.
        self.maxpool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False),
        )

        out_channels = (len(kernel_sizes) + 1) * n_filters
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_b = self.bottleneck(x)
        branch_outputs = [branch(x_b) for branch in self.conv_branches]
        branch_outputs.append(self.maxpool_branch(x))
        z = torch.cat(branch_outputs, dim=1)
        return self.relu(self.bn(z))


class InceptionResidualBlock(nn.Module):
    """``modules_per_block`` Inception modules in sequence + a skip connection."""

    def __init__(
        self,
        in_channels: int,
        n_filters: int,
        kernel_sizes: Sequence[int],
        bottleneck_channels: int,
        use_bottleneck: bool,
        modules_per_block: int = 3,
    ) -> None:
        super().__init__()
        if modules_per_block < 1:
            raise ValueError(f"modules_per_block must be >= 1, got {modules_per_block}")

        modules: list[nn.Module] = []
        ch = in_channels
        out_per_module = (len(kernel_sizes) + 1) * n_filters
        for _ in range(modules_per_block):
            modules.append(
                InceptionModule(
                    in_channels=ch,
                    n_filters=n_filters,
                    kernel_sizes=kernel_sizes,
                    bottleneck_channels=bottleneck_channels,
                    use_bottleneck=use_bottleneck,
                )
            )
            ch = out_per_module
        self.inception_seq = nn.Sequential(*modules)

        # Skip path: 1x1 conv + BN to match channel count if necessary.
        if in_channels != out_per_module:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_per_module, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_per_module),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.out_channels = out_per_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.inception_seq(x) + self.shortcut(x))


@register_model("inception_time")
class InceptionTime(nn.Module):
    """Multi-scale 1D time-series classifier (InceptionTime).

    Input  : ``(B, L)`` (when ``in_channels == 1``) or ``(B, C, L)`` for
             multi-channel input.
    Output : ``(B,)`` raw logits — train with ``BCEWithLogitsLoss``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_filters: int = 32,
        kernel_sizes: Sequence[int] = (10, 20, 40),
        bottleneck_channels: int = 32,
        n_blocks: int = 2,
        modules_per_block: int = 3,
        use_bottleneck: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        if n_filters < 1:
            raise ValueError(f"n_filters must be >= 1, got {n_filters}")
        if not kernel_sizes:
            raise ValueError("kernel_sizes must be a non-empty sequence")
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")
        if modules_per_block < 1:
            raise ValueError(f"modules_per_block must be >= 1, got {modules_per_block}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.in_channels = in_channels

        blocks: list[nn.Module] = []
        ch = in_channels
        for _ in range(n_blocks):
            block = InceptionResidualBlock(
                in_channels=ch,
                n_filters=n_filters,
                kernel_sizes=kernel_sizes,
                bottleneck_channels=bottleneck_channels,
                use_bottleneck=use_bottleneck,
                modules_per_block=modules_per_block,
            )
            blocks.append(block)
            ch = block.out_channels
        self.backbone = nn.Sequential(*blocks)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(ch, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Final linear (logit) head: Xavier (no ReLU after).
        last_linear = self.classifier[-1]
        assert isinstance(last_linear, nn.Linear)
        nn.init.xavier_uniform_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.backbone(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x).squeeze(-1)
