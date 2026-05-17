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
    name: str | None = Field(
        default=None,
        description=(
            "Run tag, used in checkpoint metadata and the saved "
            "``run_summary.json``. Optional — when blank, the pipeline "
            "derives it from the (model, paradigm, bin, class) tuple."
        ),
    )
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
    out_dir: Path | None = Field(
        default=None,
        description=(
            "Directory to write CNP + diagnostics into. Optional — when "
            "blank, the pipeline derives "
            "``results/cut_acceptance/<model>/<paradigm_suffix>/bin{N}/{class}``."
        ),
    )
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
    # Hybrid-scale sampling — Axis A: trial size strategy.
    # Decoupled from Axis B (spatial pattern) so any combination is valid.
    # ------------------------------------------------------------------
    trial_size_strategy: Literal["fixed", "variable_uniform"] = Field(
        "fixed",
        description=(
            "How to choose N_total events per trial. ``fixed`` always "
            "uses ``training.n_events_per_trial`` (back-compat default). "
            "``variable_uniform`` resamples N per training step "
            "uniformly from ``[n_trial_events_min, n_trial_events_max]``. "
            "Per-step (not per-trial-within-batch) variation — within a "
            "single batch all trials share the same N to satisfy "
            "RESUM_FLEX's fixed-shape StandardBatch contract."
        ),
    )
    n_trial_events_min: int = Field(
        16,
        ge=4,
        description="Lower bound on N when trial_size_strategy='variable_uniform'.",
    )
    n_trial_events_max: int = Field(
        64,
        ge=4,
        description="Upper bound on N when trial_size_strategy='variable_uniform'.",
    )

    # ------------------------------------------------------------------
    # Hybrid-scale sampling — Axis B: spatial event-energy pattern.
    # ------------------------------------------------------------------
    sampling_pattern: Literal[
        "flat_stratified",
        "mixed_density",
        "random_clusters",
        "physics_anchored",
    ] = Field(
        "flat_stratified",
        description=(
            "Spatial distribution of context-event energies within one "
            "trial. ``flat_stratified`` is the current bin-uniform draw. "
            "The other three concentrate events near focus regions to "
            "give the CNP local detail without losing global coverage."
        ),
    )
    zoom_window_width_kev: float = Field(
        50.0,
        gt=0.0,
        description=(
            "Full width (keV) of the local zoom window used by "
            "``mixed_density``, ``random_clusters``, and ``physics_anchored``. "
            "Ignored for ``flat_stratified``."
        ),
    )
    local_event_fraction: float = Field(
        0.70,
        gt=0.0,
        lt=1.0,
        description=(
            "Fraction of a trial's events placed inside the focus window "
            "(``mixed_density`` / ``physics_anchored``). The remainder is "
            "drawn globally so the CNP still sees the full spectrum."
        ),
    )
    n_clusters: int = Field(
        2,
        ge=1,
        description=(
            "Number of disjoint focus windows for ``random_clusters``. "
            "Rejection sampling fails after 50 attempts if the requested "
            "count won't fit non-overlapping inside ``energy_range``."
        ),
    )
    physics_peaks_kev: list[float] = Field(
        default_factory=lambda: [1460.0, 2614.0],
        description=(
            "Candidate focus energies for ``physics_anchored``. Defaults "
            "are ⁴⁰K (1460 keV) and ²⁰⁸Tl FE (2614 keV) — both span the "
            "high-information regions for cut-acceptance. Peaks outside "
            "``energy_range`` are dropped at config-load time."
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

    @field_validator("n_trial_events_max")
    @classmethod
    def _max_ge_min(cls, v: int, info) -> int:
        n_min = info.data.get("n_trial_events_min")
        if n_min is not None and v < n_min:
            raise ValueError(
                f"n_trial_events_max ({v}) must be >= n_trial_events_min ({n_min})"
            )
        return v

    @field_validator("physics_peaks_kev")
    @classmethod
    def _peaks_inside_energy_range(cls, peaks: list[float], info) -> list[float]:
        rng = info.data.get("energy_range")
        if rng is None:
            return peaks
        lo, hi = rng
        kept = [p for p in peaks if lo <= p <= hi]
        # Silent filter is fine here (peaks outside range are just unusable
        # focus candidates); validate at pipeline time that *some* peaks
        # remain when sampling_pattern == "physics_anchored".
        return kept


def load_config(path: Path | str) -> CutAcceptanceConfig:
    """Parse a YAML file into a validated :class:`CutAcceptanceConfig`."""
    return parse_yaml_file_as(CutAcceptanceConfig, Path(path))
