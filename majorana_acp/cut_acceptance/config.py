"""Pydantic config for the binned-CNP cut-acceptance pipeline.

The pipeline trains a single CNP that maps θ = (E_bin_center, T) →
β = P(score ≥ T | E). Training data is binned by energy and pulled
from the *train* split of the Majorana release (so the classifier
never saw it as training labels for the *CNP*'s purposes); the
empirical validation curve is computed from the *test* split. No MFGP,
no LF/HF partition, no peak-region split — all of that lived in the
RESuM-style fork and was removed when we pivoted to a CNP-only,
binned framing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_yaml import parse_yaml_file_as

# RESUM_FLEX configs — re-used directly so we don't duplicate hyperparameters.
from schemas.config import CNPConfig, EncoderConfig, TrainingConfig


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CutAcceptanceConfig(_Frozen):
    """Full config for one binned-CNP cut-acceptance run."""

    # ------------------------------------------------------------------
    # Identity / IO
    # ------------------------------------------------------------------
    name: str = Field(..., description="Run tag, used as the output subfolder.")
    train_predictions_path: Path = Field(
        ...,
        description=(
            "Path to a classifier's predictions.h5 over the **train** split "
            "(produced by majorana_acp.cli.evaluate --split train). "
            "Used as the CNP training pool: events are binned by energy and "
            "trials are drawn per (bin, T_sampled)."
        ),
    )
    validation_predictions_path: Path = Field(
        ...,
        description=(
            "Path to a classifier's predictions.h5 over the **test** split. "
            "Used only post-training to compute the empirical binned pass "
            "rate that the CNP is evaluated against — never seen by the CNP "
            "during training."
        ),
    )
    out_dir: Path = Field(..., description="Directory to write CNP + diagnostics into.")
    upstream_classifier_config: Path = Field(
        ...,
        description=(
            "Path (relative to repo root) to the YAML that trained the "
            "classifier whose ``predictions.h5`` outputs feed this run. "
            "The SHA256 of this file is recorded in ``run_summary.json`` "
            "at pipeline time, binding the cut-acceptance run to an exact "
            "upstream-classifier state. Mandatory — no default."
        ),
    )

    # ------------------------------------------------------------------
    # Class filter and energy binning
    # ------------------------------------------------------------------
    target_class: Literal[0, 1, "all"] = Field(
        ...,
        description=(
            "Restrict to events with this true psd_label_low_avse. "
            "1 = signal acceptance (label==1), 0 = background rejection "
            "(label==0), 'all' = no label filter (inclusive pass rate "
            "marginalised over the natural class composition)."
        ),
    )
    energy_range: tuple[float, float] = Field(
        (500.0, 3000.0),
        description="Inclusive energy range (keV) covered by the bin grid.",
    )
    energy_bin_width: float = Field(
        10.0,
        gt=0.0,
        description="Bin width in keV. The Resolution Sweep uses 5 / 10 / 20.",
    )
    threshold_range: tuple[float, float] = Field(
        (0.0, 1.0),
        description="θ_T sampling box (the CNN-score domain).",
    )

    # ------------------------------------------------------------------
    # Per-trial sampling
    # ------------------------------------------------------------------
    n_per_trial: int | None = Field(
        default=None,
        ge=1,
        description=(
            "DEPRECATED. The True-CNP pipeline reads events-per-trial from "
            "``training.n_events_per_trial`` instead. Kept here so legacy "
            "YAMLs (from the bin-isolated DESIGN_ONLY era) still parse — "
            "the value is ignored by the new pipeline. Remove from new "
            "configs."
        ),
    )
    min_events_per_bin: int = Field(
        4,
        ge=1,
        description=(
            "Bins with fewer events than this are excluded from the "
            "EventSampler's stratification pool — events in those bins "
            "do not contribute to training. For narrow binnings this "
            "drops rare-energy slabs that would otherwise dominate the "
            "stratified draw via tiny pools."
        ),
    )

    # ------------------------------------------------------------------
    # Nested RESUM_FLEX configs.
    # ------------------------------------------------------------------
    encoder: EncoderConfig = Field(
        default_factory=lambda: EncoderConfig(
            type="mlp", latent_dim=64, hidden_dims=[128, 128], dropout=0.0
        )
    )
    cnp: CNPConfig = Field(
        default_factory=lambda: CNPConfig(
            n_context_min=8,
            n_context_max=24,
            output_activation="sigmoid",
            mixup_alpha=0.01,
        )
    )
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    # ------------------------------------------------------------------
    # CNP decoder capacity. The deeper [128, 128, 128] decoder fixed the
    # β-saturation issue we saw with the default [128, 128].
    # ------------------------------------------------------------------
    decoder_hidden_dims: list[int] = Field(
        default_factory=lambda: [128, 128, 128],
        description="Hidden layer widths for the CNP decoder MLP.",
    )

    # ------------------------------------------------------------------
    # Inference-time MC Dropout. Requires encoder.dropout > 0 in the
    # nested EncoderConfig — otherwise every forward pass is identical
    # and the "uncertainty" collapses to context-resampling noise alone.
    # ------------------------------------------------------------------
    mc_dropout_samples: int = Field(
        50,
        ge=1,
        description=(
            "Number of forward passes with dropout active when evaluating "
            "the CNP on the (E_bin × T) grid. The sample mean is the β "
            "estimate; the sample std is σ_CNP."
        ),
    )

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
    """Parse a YAML file into a validated :class:`CutAcceptanceConfig`."""
    return parse_yaml_file_as(CutAcceptanceConfig, Path(path))
