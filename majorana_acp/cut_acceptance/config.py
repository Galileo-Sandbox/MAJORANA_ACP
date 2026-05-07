"""Pydantic config for the cut-acceptance pipeline.

One ``CutAcceptanceConfig`` fully specifies one signal-or-background run:
which classifier's scores to consume, which energy windows count as
"peak", how to split peak events into LF / HF train / HF holdout, the
θ-box for sampling, the per-trial event count per fidelity, the eval
grid, and a slot for nested CNP / MFGP / training configs from
RESUM_FLEX.

Two pipelines (signal vs background) share the same schema and the same
``partition_seed`` so the LF / HF / holdout split is reproducible across
both runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_yaml import parse_yaml_file_as

# RESUM_FLEX configs — re-used directly so we don't duplicate hyperparameters.
from schemas.config import CNPConfig, EncoderConfig, MFGPConfig, TrainingConfig


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PeakWindow(_Frozen):
    """A single energy window (keV) treated as a "peak region"."""

    lo: float = Field(..., description="Lower edge in keV (inclusive).")
    hi: float = Field(..., description="Upper edge in keV (inclusive).")

    @model_validator(mode="after")
    def _ordered(self) -> PeakWindow:
        if not self.hi > self.lo:
            raise ValueError(f"peak window must satisfy hi > lo, got [{self.lo}, {self.hi}]")
        return self


class PeakSplit(_Frozen):
    """Fractions for the LF / HF-train / HF-holdout split of *peak* events."""

    lf: float = Field(0.10, ge=0.0, le=1.0)
    hf_train: float = Field(0.30, ge=0.0, le=1.0)
    hf_holdout: float = Field(0.60, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _sums_to_one(self) -> PeakSplit:
        s = self.lf + self.hf_train + self.hf_holdout
        if abs(s - 1.0) > 1e-9:
            raise ValueError(
                f"peak split fractions must sum to 1.0, got {s:.6f} "
                f"(lf={self.lf}, hf_train={self.hf_train}, hf_holdout={self.hf_holdout})"
            )
        return self


class CutAcceptanceConfig(_Frozen):
    """Full config for one cut-acceptance run."""

    # ------------------------------------------------------------------
    # Identity / IO
    # ------------------------------------------------------------------
    name: str = Field(..., description="Run tag, used as the output subfolder.")
    predictions_path: Path = Field(
        ...,
        description="Path to a classifier's predictions.h5 "
        "(produced by majorana_acp.cli.evaluate).",
    )
    out_dir: Path = Field(..., description="Directory to write CNP / MFGP / plots into.")

    # ------------------------------------------------------------------
    # Data partition
    # ------------------------------------------------------------------
    target_class: Literal[0, 1, "all"] = Field(
        ...,
        description=(
            "Restrict to events with this true psd_label_low_avse. "
            "1 = signal acceptance (label==1), 0 = background rejection (label==0), "
            "'all' = no label filter (inclusive pass-rate; ignores ground truth and "
            "computes P(score >= T | E) marginalised over the natural class "
            "composition in the test set)."
        ),
    )
    energy_range: tuple[float, float] = Field(
        (500.0, 3000.0),
        description="Inclusive energy filter (keV) before any partitioning.",
    )
    peak_windows: list[PeakWindow] = Field(
        default_factory=lambda: [
            PeakWindow(lo=2605.0, hi=2620.0),
            PeakWindow(lo=2095.0, hi=2110.0),
            PeakWindow(lo=1615.0, hi=1625.0),
            PeakWindow(lo=1587.0, hi=1597.0),
        ],
        description="Energy windows treated as peak regions; the union is split "
        "into LF / HF train / HF holdout.",
    )
    peak_split: PeakSplit = Field(default_factory=PeakSplit)
    partition_seed: int = Field(
        0,
        description="RNG seed for the peak 10/30/60 split — keep equal across "
        "the signal and background runs so the partition is reproducible.",
    )

    # ------------------------------------------------------------------
    # θ-box and per-trial event count
    # ------------------------------------------------------------------
    threshold_range: tuple[float, float] = Field(
        (0.0, 1.0),
        description="θ_T sampling box; ranges where a sigmoid output is meaningful.",
    )
    n_per_trial_lf: int = Field(
        128, ge=1, description="Per-trial event count when sampling from the LF pool."
    )
    n_per_trial_hf: int = Field(
        32, ge=1, description="Per-trial event count when sampling from an HF pool."
    )
    min_pool_size: int = Field(
        8,
        ge=1,
        description="Refuse to construct a sampler if its event pool is smaller than this.",
    )

    # ------------------------------------------------------------------
    # Eval grid (final heatmap resolution)
    # ------------------------------------------------------------------
    energy_grid_step: float = Field(
        10.0, gt=0.0, description="Energy step (keV) of the β(E, T) eval grid."
    )
    threshold_grid_step: float = Field(
        0.02, gt=0.0, description="Threshold step of the β(E, T) eval grid."
    )

    # ------------------------------------------------------------------
    # Nested RESUM_FLEX configs (consumed by their training / fitting APIs).
    # Defaults mirror RESUM_FLEX's reference config.yaml so a YAML omitting
    # these blocks gets the same hyperparameters as the upstream demo.
    # ------------------------------------------------------------------
    encoder: EncoderConfig = Field(
        default_factory=lambda: EncoderConfig(
            type="mlp", latent_dim=64, hidden_dims=[128, 128], dropout=0.0
        )
    )
    cnp: CNPConfig = Field(
        default_factory=lambda: CNPConfig(
            n_context_min=16,
            n_context_max=64,
            output_activation="sigmoid",
            mixup_alpha=0.1,
        )
    )
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    mfgp: MFGPConfig = Field(default_factory=MFGPConfig)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("energy_range", "threshold_range")
    @classmethod
    def _ordered_pair(cls, v: tuple[float, float]) -> tuple[float, float]:
        if not v[1] > v[0]:
            raise ValueError(f"range must satisfy hi > lo, got {v}")
        return v


def load_config(path: Path | str) -> CutAcceptanceConfig:
    """Parse a YAML file into a validated ``CutAcceptanceConfig``."""
    return parse_yaml_file_as(CutAcceptanceConfig, Path(path))
