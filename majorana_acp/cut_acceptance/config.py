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


class PositionalEncodingConfig(_Frozen):
    """1D Fourier-feature expansion of the energy coordinate.

    When ``enabled``, the per-event energy ``E`` is normalized into
    ``[0, 1]`` against ``[min_energy_kev, max_energy_kev]`` and lifted
    into a ``2 * num_bands``-dimensional sinusoidal basis:

    ``γ(E_norm) = [sin(2^k π E_norm), cos(2^k π E_norm)
                   for k in 0 .. num_bands-1]``

    The threshold dimension ``T`` is kept as a single scalar appended
    at the end of phi, so ``dim_phi = 2 * num_bands + 1`` when enabled
    and ``dim_phi = 2`` when disabled.

    Defaults to disabled — every existing YAML and trained checkpoint
    runs identically.
    """

    enabled: bool = Field(
        False,
        description=(
            "Master gate. MUST default false for backward compatibility "
            "with existing YAMLs and trained checkpoints."
        ),
    )
    num_bands: int = Field(
        10,
        ge=1,
        description=(
            "Number of frequency bands ``L``. The k-th band is "
            "``(sin(2^k π E_norm), cos(2^k π E_norm))``. The highest "
            "band has period ``2 / 2^(L-1)`` in the normalized "
            "coordinate; at L=10 over a 2500-keV range that's ~9.8 keV "
            "— deliberately matched to the bin10 grid scale."
        ),
    )
    min_energy_kev: float = Field(
        500.0,
        description=(
            "Lower bound for the energy normalization (E_norm = "
            "(E - min) / (max - min)). Usually matches energy_range[0] "
            "but kept independent so the PE window can be tuned "
            "without retraining the broader sampling stack."
        ),
    )
    max_energy_kev: float = Field(
        3000.0,
        description="Upper bound for the energy normalization.",
    )

    @field_validator("max_energy_kev")
    @classmethod
    def _max_gt_min(cls, v: float, info) -> float:
        lo = info.data.get("min_energy_kev")
        if lo is not None and v <= lo:
            raise ValueError(f"max_energy_kev ({v}) must exceed min_energy_kev ({lo})")
        return v


class DensityModulationConfig(_Frozen):
    """Pluggable Relative-Density modulation for the Gaussian Attention
    Bias (GAB).

    When ``enabled``, the GAB's per-pair penalty
    ``exp(log_gamma_h) · Δ²`` is multiplied by a target-conditioned
    ratio ``R_* ∈ [B, N_T]`` that measures the local-vs-global density
    of context events at the target query point ``E_*``:

        ρ̂_local(E_*)  = (1 / (σ_local √(2π))) · Σ_j K_local(E_*, E_j)
        ρ̂_global(E_*) = (1 / (σ_global √(2π))) · Σ_j K_global(E_*, E_j) + ε
        R_*           = ρ̂_local(E_*) / ρ̂_global(E_*)

    where ``K_σ(a, b) = exp(-(a−b)² / (2σ²))``. The kernel sums are
    divided by their respective normalisations so that, **in the limit
    of large N_C**, the ratio satisfies ``R_* → 1`` for flat densities.
    At finite N_C (e.g. variable-N training with N as low as 32),
    ``R_*`` can be substantially below 1 in flat regions simply because
    σ_local is so narrow that few or no events lie within one kernel
    width — the kernel-density estimator is sampling noise. What is
    robust across N_C is the **contrast**: a sharp peak cluster within
    σ_local of the query produces ``R_*`` larger than the surrounding
    continuum by roughly ``σ_global / σ_local``. The learnable
    per-head ``log_gamma`` absorbs the absolute scale.

    The penalty becomes ``exp(log_gamma_h) · R_* · Δ²``, so attention
    heads dynamically contract their effective receptive field in
    peak regions (pupil contracts at the peak) while keeping a wider
    smoothing window in the continuum.

    Sigmas are specified in keV in the YAML; the aggregator converts
    them to normalised units (divide by ``E_max − E_min`` at factory
    time) so the math runs in the same ``E_norm ∈ [0, 1]`` space as
    the rest of the CNP.

    Ignored unless ``aggregator.type == 'cross_attention'`` AND
    ``aggregator.gaussian_attention_bias == True`` — DM modifies the
    GAB penalty, so without GAB it has nothing to modify.
    """

    enabled: bool = Field(
        False,
        description=(
            "Master gate. Defaults false so every existing aggregator config parses unchanged."
        ),
    )
    sigma_local_kev: float = Field(
        10.0,
        gt=0.0,
        description=(
            "Half-width of the local Gaussian kernel (keV). Fits typical "
            "HPGe peak FWHM; 10 keV matches the bin10 grid scale and "
            "captures one peak's main lobe."
        ),
    )
    sigma_global_kev: float = Field(
        150.0,
        gt=0.0,
        description=(
            "Half-width of the global Gaussian kernel (keV). Spans the "
            "broad Compton continuum around any peak; 150 keV is wide "
            "enough that the kernel sum smoothly tracks the background "
            "density without latching onto narrow features."
        ),
    )
    epsilon: float = Field(
        1e-6,
        gt=0.0,
        description=(
            "Numerical floor added to ρ̂_global to keep the ratio finite "
            "when the global kernel sum vanishes (e.g. degenerate "
            "context with all events outside σ_global of E_*)."
        ),
    )

    @field_validator("sigma_global_kev")
    @classmethod
    def _global_gt_local(cls, v: float, info) -> float:
        lo = info.data.get("sigma_local_kev")
        if lo is not None and v <= lo:
            raise ValueError(
                f"sigma_global_kev ({v}) must exceed sigma_local_kev ({lo}) — "
                "the global kernel is meant to span a broader window than "
                "the local one so R_* contrasts peak vs continuum."
            )
        return v


class BoundedBandwidthConfig(_Frozen):
    """Pluggable Bounded Physical Bandwidth Network (BPBN) for the
    attention layer.

    BPBN **replaces** the abstract GAB scalar ``exp(log_gamma_h) · Δ²``
    with an explicit, bounded physical kernel:

        score_{*, i, h} = (Q_* K_i^T) / √d_k − Δ²_{*, i} / (2 σ_head²(E_*))

    where the per-head bandwidth ``σ_head(E_*)`` is rigidly bounded in
    ``[σ_min, σ_max]`` keV:

        σ_head(E_*) = clamp(σ_max / (1 + λ_head(E_*) · (R_* − 1)_+),
                            σ_min, σ_max)
        λ_head(E_*) = α_max · sigmoid( MLP_head(E_*) )
                    ∈ [0, α_max]
        R_*         = ρ̂_local(E_*) / ρ̂_global(E_*)   (dual-σ KDE; see
                       ``DensityModulationConfig`` for the kernel math)

    Operational regimes:

    * **Flat continuum (R_* ≈ 1)**: the denominator collapses to 1,
      ``σ_head → σ_max``. The attention window expands to cover ~200 keV,
      smoothing over Bernoulli noise that PE10's high-frequency basis
      would otherwise let the model fit point-by-point.
    * **Sharp peak (R_* ≫ 1)**: the denominator grows like
      ``λ · (R − 1)``, contracting ``σ_head`` toward ``σ_min``.
      Receptive fields tighten exclusively around the peak core,
      preserving physics-peak resolution.

    Why this fixes the GAB sawtooth diagnosis: the bare GAB penalty
    ``exp(log_gamma_h) · Δ²`` has no upper bound on ``γ`` — gradient
    descent drives ``γ → ∞`` toward 1-NN lookup, encouraging the
    decoder to fit individual context-event Bernoulli fluctuations
    rather than the smooth underlying β(E). BPBN clamps the effective
    kernel width in raw keV space, so the model literally cannot
    contract its attention below 5 keV (the HPGe resolution floor).

    BPBN replaces (and is mutually exclusive with) the per-head
    ``log_gamma`` parameter — when ``enabled``, ``log_gamma`` is NOT
    registered on the attention module. ``density_modulation`` is also
    bypassed; BPBN computes ``R_*`` itself with its own σ_local /
    σ_global. Existing GAB checkpoints continue to load through the
    plain GAB path because BPBN defaults ``enabled=False``.

    Ignored unless ``aggregator.type == 'cross_attention'`` AND
    ``aggregator.gaussian_attention_bias == True``.
    """

    enabled: bool = Field(
        False,
        description=(
            "Master gate. Defaults false so every existing aggregator "
            "config parses unchanged and every existing GAB checkpoint "
            "loads via the plain GAB path."
        ),
    )
    sigma_max_kev: float = Field(
        200.0,
        gt=0.0,
        description=(
            "Upper bound on the per-target attention bandwidth (keV). "
            "Sets the receptive-field width in flat-continuum regions "
            "where R_* ≈ 1 and the denominator collapses to 1. Default "
            "200 keV ≈ the Compton-continuum smoothing scale; large "
            "enough to average out PE10's bin-scale noise."
        ),
    )
    sigma_min_kev: float = Field(
        5.0,
        gt=0.0,
        description=(
            "Hard lower bound on the per-target attention bandwidth "
            "(keV). Enforced by ``torch.clamp`` so the kernel width can "
            "never collapse below the HPGe detector's resolution floor "
            "— the structural reason the original GAB collapsed into "
            "1-NN lookup is exactly this missing lower clamp. Default "
            "5 keV ≈ half the bin10 grid."
        ),
    )
    alpha_max: float = Field(
        40.0,
        gt=0.0,
        description=(
            "Hard upper bound on the per-head sensitivity ``λ_head(E_*)``. "
            "Together with sigmoid(MLP) ∈ [0, 1], this caps the maximum "
            "contraction rate (peak-aware bandwidth shrinkage). At "
            "α=40, R=15 (typical sharp peak), σ_head ≈ σ_max / (1 + 40·14) "
            "= σ_max / 561 → clamped to σ_min."
        ),
    )
    sensitivity_hidden_dim: int = Field(
        16,
        ge=1,
        description=(
            "Hidden width of the per-head sensitivity MLP: "
            "Linear(1 → hidden) → GELU → Linear(hidden → num_heads). "
            "Input is the normalised target energy ``E_target_norm``; "
            "output is the un-sigmoid'd λ for each head."
        ),
    )
    sigma_local_kev: float = Field(
        10.0,
        gt=0.0,
        description=(
            "Local Gaussian kernel half-width (keV) used to compute "
            "ρ̂_local in the R_* ratio. Matches DM's default."
        ),
    )
    sigma_global_kev: float = Field(
        150.0,
        gt=0.0,
        description=(
            "Global Gaussian kernel half-width (keV) used to compute "
            "ρ̂_global in the R_* ratio. Matches DM's default."
        ),
    )
    epsilon: float = Field(
        1e-6,
        gt=0.0,
        description="Numerical floor on ρ̂_global to keep R_* finite.",
    )

    @field_validator("sigma_min_kev")
    @classmethod
    def _min_below_max(cls, v: float, info) -> float:
        # ``sigma_max_kev`` is the previously-processed field, so it is
        # guaranteed present in ``info.data`` by the time this validator
        # runs. Cannot put this on ``sigma_max_kev`` directly — pydantic
        # v2 processes fields in declaration order and the sibling
        # would not yet be available.
        hi = info.data.get("sigma_max_kev")
        if hi is not None and hi <= v:
            raise ValueError(
                f"sigma_max_kev ({hi}) must exceed sigma_min_kev ({v}) — "
                "the clamp window must be non-degenerate."
            )
        return v

    @field_validator("sigma_global_kev")
    @classmethod
    def _global_gt_local(cls, v: float, info) -> float:
        lo = info.data.get("sigma_local_kev")
        if lo is not None and v <= lo:
            raise ValueError(f"sigma_global_kev ({v}) must exceed sigma_local_kev ({lo})")
        return v


class SfnModulationConfig(_Frozen):
    """Pluggable Bounded 2D Spectral Filter Network (SFN) for the
    attention layer.

    SFN is structurally similar to BPBN but replaces the closed-form
    ``σ_max / (1 + λ · (R−1)_+)`` formula with a fully-learned 2D MLP
    head:

        Z_*  = [E_target_norm, log(R_* + ε)]   ∈ R^{B, N_T, 2}
        σ_head(E_*, R_*) = σ_min + (σ_max − σ_min) · sigmoid(SFN_head(Z_*))
        score_{*, i, h}  = (Q_* K_i^T) / √d_k − Δ²_{*, i} / (2 σ_head²)

    where ``SFN`` is a small ``Linear(2 → 16) → GELU → Linear(16 → H)``
    network that takes the **macro** energy coordinate ``E_target_norm``
    and the **micro** density log-ratio ``log(R_*)`` and emits one
    bandwidth logit per attention head. ``R_*`` is the same self-
    normalised dual-σ KDE ratio used by DM/BPBN
    (``σ_local = 10 keV``, ``σ_global = 150 keV`` by default).

    Why this design fixes BPBN's residual sawtooth (Cell 5: ACF1 ≈
    −0.55 in the continuum even though σ_min clamped at 5 keV): the
    closed-form ``1 / (1 + λ(R−1))`` is **algebraically sensitive to
    R_*'s shot noise**. Small statistical fluctuations in the local
    kernel sum produce R_* wiggles that pass straight through into
    σ_head and contract the window erratically. Wrapping the same
    coordinates in an MLP exploits the **spectral bias of low-
    dimensional MLPs** — networks of this size preferentially learn
    smooth (low-frequency) functions of their inputs, acting as a
    non-linear low-pass filter that ignores the bin-scale noise in
    R_* while still tracking the persistent macro signature of a
    real γ-peak.

    Like BPBN, SFN **replaces** the GAB scalar ``log_gamma`` — when
    SFN is enabled, ``log_gamma`` is not registered on the attention
    module. SFN and BPBN are mutually exclusive (both replace the
    same parametrisation); the factory raises if both are set.

    Ignored unless ``aggregator.type == 'cross_attention'`` AND
    ``aggregator.gaussian_attention_bias == True``.
    """

    enabled: bool = Field(
        False,
        description=(
            "Master gate. Defaults false so every existing aggregator config parses unchanged."
        ),
    )
    sigma_max_kev: float = Field(
        200.0,
        gt=0.0,
        description=(
            "Upper bound on the per-target attention bandwidth (keV). "
            "Set the receptive field for continuum-vacuum regions where "
            "the SFN should crush PE10's bin-scale wiggles via long-"
            "window averaging."
        ),
    )
    sigma_min_kev: float = Field(
        5.0,
        gt=0.0,
        description=(
            "Lower bound on the per-target attention bandwidth (keV). "
            "Hard physical floor at the HPGe resolution scale; the "
            "sigmoid output saturating at 0 cannot push σ below this."
        ),
    )
    hidden_dim: int = Field(
        16,
        ge=1,
        description=(
            "Hidden width of the SFN MLP. "
            "Architecture: Linear(2 → hidden) → GELU → Linear(hidden → H). "
            "Default 16 is small enough for spectral bias to dominate "
            "(the network preferentially learns smooth functions of "
            "the 2D coordinate)."
        ),
    )
    sigma_local_kev: float = Field(
        10.0,
        gt=0.0,
        description="Local Gaussian kernel half-width for R_* (keV).",
    )
    sigma_global_kev: float = Field(
        150.0,
        gt=0.0,
        description="Global Gaussian kernel half-width for R_* (keV).",
    )
    epsilon: float = Field(
        1e-6,
        gt=0.0,
        description=(
            "Numerical floor on R_* before taking ``log``. Keeps the "
            "feature finite when ρ̂_local collapses to 0."
        ),
    )

    @field_validator("sigma_min_kev")
    @classmethod
    def _min_below_max(cls, v: float, info) -> float:
        # ``sigma_max_kev`` is declared first so it is already in
        # ``info.data`` by the time this validator runs.
        hi = info.data.get("sigma_max_kev")
        if hi is not None and hi <= v:
            raise ValueError(
                f"sigma_max_kev ({hi}) must exceed sigma_min_kev ({v}) — "
                "the clamp window must be non-degenerate."
            )
        return v

    @field_validator("sigma_global_kev")
    @classmethod
    def _global_gt_local(cls, v: float, info) -> float:
        lo = info.data.get("sigma_local_kev")
        if lo is not None and v <= lo:
            raise ValueError(f"sigma_global_kev ({v}) must exceed sigma_local_kev ({lo})")
        return v


class PoolDensitySfnConfig(_Frozen):
    """Pool-Based Absolute Density Spectral Filter Network (Cell 7).

    Cell 7's structural fix to the Cell 5/6 sawtooth: stop deriving the
    SFN's bandwidth-decision input from the *batch context*, and instead
    derive it from a **fixed precomputed buffer** of all kept events in
    the training pool. The same buffer is constructed at training and
    at inference (deterministic ``load_events`` on
    ``cfg.train_predictions_path``), so the density features the SFN
    sees are identical in both phases — the source of Cell 6's residual
    high-variance R_* (anchor bimodality + per-batch shot noise) is
    eliminated by construction.

    Math:

        E_pool ∈ R^{N_pool}                          (fixed buffer; kept events
                                                      of the training pool,
                                                      normalised to [0,1])
        ρ_local(E_*)  = Σ_{j ∈ pool} exp(-(E_* - E_pool_j)² / (2 σ_local²))
        ρ_global(E_*) = Σ_{j ∈ pool} exp(-(E_* - E_pool_j)² / (2 σ_global²))
        Z_*  = [log(ρ_local + ε), log(ρ_global + ε)]           ∈ R^{B, N_T, 2}
        σ_head(E_*) = σ_min + (σ_max − σ_min) · sigmoid(MLP(Z_*))
        score_{*, i, h}  = (Q_* K_i^T) / √d_k − Δ²_{*, i} / (2 σ_head²)

    Differences vs the Cell-6 SFN:

    * Kernel sums are **un-normalised** (absolute physical counts), so
      the MLP sees absolute statistical bounds — at sparse-data regions
      log ρ → −∞ and the network can learn "trust nothing, widen the
      attention window". The Cell-6 self-normalised ratio R_* hides this
      cue: a sparse-data trial and a dense-data trial both give R_* ≈ 1.
    * Computed against a **fixed pool buffer**, not the per-trial
      context. The pool of ~10k–20k events smooths shot noise to a
      negligible level vs Cell 6's σ_local-kernel-sum over a few hundred
      context events.
    * MLP input is **PE-free** — only the two log densities, no
      ``E_target_norm`` coordinate. The user-locked design choice: the
      absolute densities already implicitly encode where on the spectrum
      we are (peaks have ρ_local ≫ ρ_global, continuum is flat).

    Mutually exclusive with both ``bounded_bandwidth`` (Cell 5) and
    ``sfn_modulation`` (Cell 6) — all three replace the GAB
    ``log_gamma`` parametrisation. Factory raises if more than one is
    enabled simultaneously.

    Ignored unless ``aggregator.type == 'cross_attention'`` AND
    ``aggregator.gaussian_attention_bias == True``.
    """

    enabled: bool = Field(
        False,
        description=(
            "Master gate. Defaults false so every existing aggregator "
            "config parses unchanged and Cell 5/6 SFN checkpoints stay "
            "loadable through the same dispatcher."
        ),
    )
    sigma_max_kev: float = Field(
        200.0,
        gt=0.0,
        description="Upper bound on σ_head (keV). Same physical scale as Cell 5/6.",
    )
    sigma_min_kev: float = Field(
        5.0,
        gt=0.0,
        description="Lower bound on σ_head (keV). Same physical floor as Cell 5/6.",
    )
    hidden_dim: int = Field(
        16,
        ge=1,
        description=(
            "Hidden width of the PE-free MLP: "
            "Linear(2 → hidden) → GELU → Linear(hidden → H). Spectral "
            "bias preferentially learns smooth functions of the two "
            "log densities."
        ),
    )
    sigma_local_kev: float = Field(
        10.0,
        gt=0.0,
        description=(
            "Local Gaussian kernel half-width (keV) for ρ_local against "
            "the pool buffer. 10 keV ≈ bin10 grid + typical HPGe peak FWHM."
        ),
    )
    sigma_global_kev: float = Field(
        150.0,
        gt=0.0,
        description=(
            "Global Gaussian kernel half-width (keV) for ρ_global. "
            "Spans the Compton continuum smoothing scale."
        ),
    )
    epsilon: float = Field(
        1e-5,
        gt=0.0,
        description=(
            "Numerical floor inside ``log(ρ + ε)``. Prevents −∞ when a "
            "target query is far enough from every pool event that the "
            "kernel sum underflows to zero."
        ),
    )
    temperature_gating: bool = Field(
        False,
        description=(
            "Enable the Dual-Gated SFN (Cell 8). When True, a SECOND "
            "PE-free MLP head (same input ``[log ρ_local, log ρ_global]``) "
            "emits a per-target temperature τ_head ∈ [τ_min, τ_max] that "
            "scales the pre-softmax attention score:\n\n"
            "    Scores = (Q · K^T) / (√d_k · τ_head) − Δ² / (2 σ_head²)\n\n"
            "Closes the asymmetric loophole in Cell 7's σ-only gating: "
            "when σ_head → σ_max in a flat continuum the spatial "
            "penalty collapses to ~0, leaving the raw PE10-driven "
            "``Q·K^T`` free to produce sharp attention peaks against "
            "shot noise. Co-driving τ_head → τ_max simultaneously "
            "flattens the QK^T spikes — the model now has both a "
            "maximum bandwidth (the σ penalty) and an enforced minimum "
            "bandwidth (the temperature flooring) in vacuum regions."
        ),
    )
    tau_min_value: float = Field(
        1.0,
        gt=0.0,
        description=(
            "Lower bound on τ_head (dimensionless). Setting to 1.0 means "
            "the dual-gate degenerates to plain softmax in peak regions "
            "— the model can recover Cell 7's behaviour exactly when "
            "τ_head saturates at τ_min."
        ),
    )
    tau_max_value: float = Field(
        10.0,
        gt=0.0,
        description=(
            "Upper bound on τ_head (dimensionless). τ = 10 reduces "
            "pre-softmax score magnitudes by 10× — enough to flatten "
            "PE10's bin-scale ``Q·K^T`` fluctuations in vacuum regions "
            "without zeroing the attention entirely."
        ),
    )
    head_tied: bool = Field(
        False,
        description=(
            "Enable the Tied-Head DG-SFN (Cell 9). When True, the σ-MLP "
            "and τ-MLP each output a *single scalar* per target query "
            "instead of one scalar per attention head — the shared σ "
            "and τ are then broadcast across all H heads:\n\n"
            "    Scores_{h,*,i} = (Q_h · K_h^T) / (√d_k · τ(E_*))\n"
            "                       − Δ²_{*,i} / (2 σ(E_*)²)\n\n"
            "Closes the head-collapse escape route observed in Cell 8 "
            "(see analysis/cnp_audit/_cell8_diagnostic_head_collapse.md): "
            "with head-wise σ, training carved out a single narrow head "
            "doing effective 1-NN lookup while six others did global "
            "averaging — leaving σ_head with no position dependence and "
            "τ_head pinned at τ_min. Forcing head-consensus removes the "
            "single-narrow-head escape: a small σ now penalises every "
            "head simultaneously in the continuum, providing the NBLL "
            "gradient signal needed to drive σ AND τ upward jointly "
            "in vacuum regions.\n\n"
            "Requires ``temperature_gating=True``; raises otherwise."
        ),
    )

    @field_validator("head_tied")
    @classmethod
    def _head_tied_requires_temperature_gating(cls, v: bool, info) -> bool:
        if v and not info.data.get("temperature_gating", False):
            raise ValueError(
                "head_tied=True requires temperature_gating=True — head "
                "tying is meaningful only for the dual-gated variant."
            )
        return v

    pe_gated_decoder: bool = Field(
        False,
        description=(
            "Enable the PE-Gated Decoder Concat (Cell 13). When True, a "
            "THIRD PE-free MLP head (same input ``[log ρ_local, "
            "log ρ_global]`` as σ and τ) emits a per-target scalar "
            "η_head(E_*) ∈ [0, 1] that gates how much of the PE-encoded "
            "``z_phi_T`` latent leaks into the decoder's coordinate "
            "concat input:\n\n"
            "    decoder_input = [r_target, raw_E_norm, η · z_phi_T]\n\n"
            "In low-density regions (continuum) the SFN learns η → 0 "
            "and the decoder sees only the raw 2D coords → β̂ is forced "
            "smooth (Cell 11/12 regime, MASD ≈ 0.004). In high-density "
            "regions (peaks) the SFN learns η → 1 and the full PE10 "
            "latent reaches the decoder → β̂ can resolve sharp γ-line "
            "features (Cell 6 regime). The on/off switch sits under "
            "the SAME density-aware MLP family that drives σ and τ.\n\n"
            "Requires ``aggregator.decoder_coordinate_gating=True`` "
            "(so the raw coords stay in the decoder input) AND "
            "``temperature_gating=True`` (the SFN-machinery dependency). "
            "Decoder is built with input dim ``agg_dim + 2 + Z`` to "
            "accommodate the gated PE10 channel."
        ),
    )

    @field_validator("pe_gated_decoder")
    @classmethod
    def _pe_gate_requires_temperature_gating(cls, v: bool, info) -> bool:
        if v and not info.data.get("temperature_gating", False):
            raise ValueError(
                "pe_gated_decoder=True requires temperature_gating=True "
                "— the PE gate sits on the same SFN family as σ/τ."
            )
        return v

    band_filter: bool = Field(
        False,
        description=(
            "Enable the Density-Guided Bandwidth Filter (Cell 14). When "
            "True, a THIRD PE-free MLP head (same input ``[log ρ_local, "
            "log ρ_global]`` as σ and τ) emits a continuous frequency-"
            "cutoff oracle ``λ(E_*) ∈ [0, L]`` where ``L`` is the number "
            "of PE10 bands. Each Fourier band ``l ∈ {0..L-1}`` is then "
            "scaled by a soft cutoff weight\n\n"
            "    w_l(E_*) = sigmoid(α · (λ(E_*) − l))\n\n"
            "with ``α = band_filter_alpha`` (default 5.0, architectural "
            "constant). The filtered ``SAPE(E_*)`` vector replaces "
            "``z_phi_T`` in the decoder concat lane:\n\n"
            "    decoder_input = [r_target, raw_phi (E_norm, T_norm), SAPE(E_*)]\n\n"
            "In low-density regions (continuum) the SFN learns λ → 0 "
            "and every high-frequency band is physically zeroed at the "
            "source → β̂ is structurally incapable of drawing sub-bin "
            "sawtooth. At γ-peaks the SFN learns λ → L and the full "
            "Fourier basis is restored locally → β̂ resolves sharp "
            "single-bin features. Replaces (and is mutually exclusive "
            "with) ``pe_gated_decoder`` — both wire a third density-"
            "aware head into the decoder, but band_filter operates in "
            "frequency space while pe_gated_decoder is a global "
            "amplitude scalar.\n\n"
            "Requires ``aggregator.decoder_coordinate_gating=True`` "
            "(raw 2D coords stay in the decoder), "
            "``temperature_gating=True`` (the SFN family dependency), "
            "and ``positional_encoding.enabled=True`` (there must be "
            "Fourier bands to filter)."
        ),
    )
    band_filter_alpha: float = Field(
        5.0,
        gt=0.0,
        description=(
            "Sharpness constant α for the per-band cutoff sigmoid. "
            "Defaults to 5.0 — gives a sharp yet differentiable fence: "
            "one band above λ gets weight sigmoid(−5) ≈ 0.007 (zeroed), "
            "one band below λ gets weight sigmoid(+5) ≈ 0.993 (passed)."
        ),
    )

    @field_validator("band_filter")
    @classmethod
    def _band_filter_requirements(cls, v: bool, info) -> bool:
        if not v:
            return v
        if not info.data.get("temperature_gating", False):
            raise ValueError(
                "band_filter=True requires temperature_gating=True — "
                "the band-cutoff oracle sits on the same SFN family as σ/τ."
            )
        if info.data.get("pe_gated_decoder", False):
            raise ValueError(
                "band_filter=True is mutually exclusive with "
                "pe_gated_decoder=True — both wire a third density-aware "
                "head into the decoder concat. Enable at most one."
            )
        return v

    hard_filter: bool = Field(
        False,
        description=(
            "Enable the Hard-Gated Bandwidth Filter (Cell 15). Parameter-"
            "free variant of ``band_filter``: replaces the learnable "
            "λ-MLP with an explicit closed-form expression of the "
            "physical density contrast at ``E_*``:\n\n"
            "    R(E_*)  = (σ_global / σ_local) · ρ_local(E_*) / ρ_global(E_*)\n"
            "    λ(E_*)  = 1.0 + 9.0 · sigmoid(s · (R(E_*) − T))\n\n"
            "with ``T = hard_filter_contrast_threshold`` (default 5.0, "
            "from the HPGe physics rule that genuine γ-lines spike to "
            "≥5× the adjacent Compton continuum density) and ``s = "
            "hard_filter_sigmoid_steepness`` (default 10.0). The "
            "(σ_global/σ_local) prefactor normalises R so it reads ≈ 1 "
            "in flat regions and 5+ at γ-peaks — exact analogue of the "
            "self-normalised ratio used by ``DensityModulationConfig``.\n\n"
            "Behaviour:\n"
            "* R ≈ 1 in continuum → sigmoid argument ≈ −40 → λ → 1.0 → "
            "  only band 0 open → β̂ is structurally smooth (analogue of "
            "  Cell 11/12, MASD-clean).\n"
            "* R ≥ 5 at γ-peaks → sigmoid argument ≥ 0 → λ → 10.0 → all "
            "  10 PE bands open → β̂ can resolve 2-keV-wide SE/DEP/FE "
            "  lines.\n\n"
            "Zero trainable parameters. The model **cannot** drift to a "
            "compressed-λ degenerate solution (Cell 14 failure mode) "
            "because λ is no longer a function of weights to optimise. "
            "Mutually exclusive with ``band_filter`` and "
            "``pe_gated_decoder`` — all three are alternative drivers of "
            "the decoder-concat SAPE / η path.\n\n"
            "Requires ``aggregator.decoder_coordinate_gating=True``, "
            "``temperature_gating=True`` (the σ/τ SFN machinery is "
            "still learnable), and ``positional_encoding.enabled=True`` "
            "(there must be Fourier bands to filter)."
        ),
    )
    hard_filter_contrast_threshold: float = Field(
        5.0,
        gt=0.0,
        description=(
            "Density contrast threshold ``T`` for the hard filter. "
            "Default 5.0 encodes the empirical HPGe physics rule that "
            "real γ-lines display a local density at least 5× the "
            "adjacent Compton continuum baseline."
        ),
    )
    hard_filter_sigmoid_steepness: float = Field(
        10.0,
        gt=0.0,
        description=(
            "Sigmoid steepness ``s`` for the hard filter. Default 10.0 "
            "produces a near-step transition: 1 unit above threshold "
            "saturates λ → 10; 1 unit below saturates λ → 1."
        ),
    )

    @field_validator("hard_filter")
    @classmethod
    def _hard_filter_requirements(cls, v: bool, info) -> bool:
        if not v:
            return v
        if not info.data.get("temperature_gating", False):
            raise ValueError(
                "hard_filter=True requires temperature_gating=True — "
                "the σ/τ SFN machinery still drives the attention layer."
            )
        if info.data.get("pe_gated_decoder", False):
            raise ValueError(
                "hard_filter=True is mutually exclusive with "
                "pe_gated_decoder=True — both write into the decoder concat."
            )
        if info.data.get("band_filter", False):
            raise ValueError(
                "hard_filter=True is mutually exclusive with "
                "band_filter=True — both drive the per-band SAPE weights. "
                "Enable at most one."
            )
        return v

    @field_validator("sigma_min_kev")
    @classmethod
    def _min_below_max(cls, v: float, info) -> float:
        hi = info.data.get("sigma_max_kev")
        if hi is not None and hi <= v:
            raise ValueError(
                f"sigma_max_kev ({hi}) must exceed sigma_min_kev ({v}) — "
                "the clamp window must be non-degenerate."
            )
        return v

    @field_validator("sigma_global_kev")
    @classmethod
    def _global_gt_local(cls, v: float, info) -> float:
        lo = info.data.get("sigma_local_kev")
        if lo is not None and v <= lo:
            raise ValueError(f"sigma_global_kev ({v}) must exceed sigma_local_kev ({lo})")
        return v

    @field_validator("tau_max_value")
    @classmethod
    def _tau_max_gt_min(cls, v: float, info) -> float:
        lo = info.data.get("tau_min_value")
        if lo is not None and v <= lo:
            raise ValueError(
                f"tau_max_value ({v}) must exceed tau_min_value ({lo}) — "
                "the temperature clamp window must be non-degenerate."
            )
        return v


class AggregatorConfig(_Frozen):
    """Pluggable context-aggregator for the CNP.

    The legacy ``mean`` aggregator (`r_trial = mean(h_C, dim=1)`) is
    hard-coded inside RESUM_FLEX's ``ConditionalNeuralProcess``. To
    add target-conditioned cross-attention without modifying upstream,
    the pipeline dispatches to a local
    ``majorana_acp.models.attentive_cnp.AttentiveCNP`` when
    ``type == 'cross_attention'``. Defaulting to ``mean`` preserves
    byte-for-byte backward compatibility — every existing YAML and
    trained checkpoint routes through upstream unchanged.

    See ``## Proposed Phase: Attentive CNP Aggregation Layers`` in
    CLAUDE.md for the full math contract and the user-locked design
    settlements (``d_v = Z``; ``Q/K`` from ``z_φ``; dropout reused
    from ``encoder.dropout``; single attention layer).
    """

    type: Literal["mean", "cross_attention"] = Field(
        "mean",
        description=(
            "Which aggregator to use. ``mean`` (default) hits the "
            "upstream ``ConditionalNeuralProcess`` path verbatim — "
            "bitwise-identical to every pre-aggregator checkpoint. "
            "``cross_attention`` switches to the local ``AttentiveCNP`` "
            "with multi-head cross-attention over context events."
        ),
    )
    num_heads: int = Field(
        8,
        ge=1,
        description=(
            "Number of attention heads H. Used only when "
            "``type == 'cross_attention'``; ignored (kept for "
            "round-trip-safety) when ``type == 'mean'``. Must divide "
            "``attention_dim`` evenly so each head is ``attention_dim/H``."
        ),
    )
    attention_dim: int = Field(
        64,
        ge=1,
        description=(
            "Total Q/K/V projection width across all heads (per-head "
            "dim = ``attention_dim / num_heads``). Used only under "
            "cross-attention. Defaults to 64 to match the True-CNP "
            "encoder ``latent_dim`` so per-head dim = 8."
        ),
    )
    decoder_coordinate_gating: bool = Field(
        False,
        description=(
            "If True (cross-attention only), the decoder's coordinate "
            "concat input is the **raw** ``(E_norm, T_norm)`` 2D vector "
            "instead of the high-dimensional ``z_phi`` latent. The "
            "attention path keeps its high-frequency Q/K from the PE-"
            "encoded ``z_phi``; only the **decoder** is decoupled from "
            "the high-frequency channel so it can't fit bin-scale "
            "Bernoulli noise in smooth regions. Diagnostic motivation: "
            "PE10's highest band period (~9.8 keV) matches the 10 keV "
            "bin grid and the binomial-noise floor (σ ≈ 0.11), giving "
            "the unmodified decoder both the bandwidth and the loss "
            "incentive to fit pure noise into the global β(E) curve. "
            "Ignored when ``type == 'mean'``."
        ),
    )
    gaussian_attention_bias: bool = Field(
        False,
        description=(
            "If True (cross-attention only), inject a continuous "
            "relative-energy Gaussian penalty into the attention "
            "logits: ``scores = QK^T/√d - exp(log_gamma_h) · "
            "(E_target_norm − E_ctx_norm)^2`` with a learnable per-head "
            "``log_gamma`` (shape ``[num_heads, 1, 1]``, init zeros). "
            "Effect: the attention layer is biased toward a smooth, "
            "data-density-driven kernel over the continuous E-axis; "
            "each head learns its own effective receptive-field width. "
            "The raw E_norm is recovered from ``batch.phi`` via the "
            "same ``atan2`` inverse used by ``decoder_coordinate_gating``, "
            "so PE-on and PE-off both work without batch-schema changes. "
            "Ignored when ``type == 'mean'``."
        ),
    )
    density_modulation: DensityModulationConfig = Field(
        default_factory=DensityModulationConfig,
        description=(
            "Optional Relative-Density modulation of the GAB penalty. "
            "When ``enabled``, the penalty becomes "
            "``exp(log_gamma_h) · R_* · Δ²`` where ``R_*`` is the "
            "local-vs-global Gaussian kernel-density ratio at the "
            "target query point. See ``DensityModulationConfig`` for "
            "the math contract. No-op when ``enabled=False`` or when "
            "``gaussian_attention_bias=False``."
        ),
    )
    bounded_bandwidth: BoundedBandwidthConfig = Field(
        default_factory=BoundedBandwidthConfig,
        description=(
            "Optional Bounded Physical Bandwidth Network (BPBN). When "
            "``enabled`` (and ``gaussian_attention_bias=True``), the GAB "
            "scalar ``exp(log_gamma_h)`` is replaced by an explicit "
            "physical kernel width ``σ_head(E_*) ∈ [σ_min, σ_max]`` keV. "
            "See ``BoundedBandwidthConfig`` for the math. Mutually "
            "exclusive with the plain GAB ``log_gamma`` parameter — "
            "when BPBN is on, ``log_gamma`` is not registered on the "
            "attention module."
        ),
    )
    sfn_modulation: SfnModulationConfig = Field(
        default_factory=SfnModulationConfig,
        description=(
            "Optional Bounded 2D Spectral Filter Network (SFN). When "
            "``enabled`` (and ``gaussian_attention_bias=True``), the GAB "
            "scalar is replaced by ``σ_head = σ_min + (σ_max − σ_min) "
            "· sigmoid(MLP([E_norm, log R_*]))`` — a fully-learned bandwidth "
            "with the same physical [σ_min, σ_max] bounds as BPBN but "
            "no algebraic dependence on R_*'s shot noise. See "
            "``SfnModulationConfig`` for the math. Mutually exclusive "
            "with ``bounded_bandwidth`` (both replace the GAB parametrisation)."
        ),
    )
    pe_detach_qk: bool = Field(
        False,
        description=(
            "Blind the attention's Q and K projections to PE10. When "
            "True (cross-attention only), the W_Q and W_K linear layers "
            "take the **raw 2D (E_norm, T_norm) vector** as input "
            "instead of the PE-encoded ``z_phi`` latent. V keeps the "
            "PE10 latent via ``h_C`` from ``ContextPointEncoder``, and "
            "the decoder still receives ``z_phi_T`` (or raw coords if "
            "``decoder_coordinate_gating`` is on). Effect: ``Q · K^T`` "
            "becomes a smooth function of raw energy — the PE10 Fourier "
            "features can no longer produce bin-scale sharpness in the "
            "attention logits. Removes the ``Trojan horse'' pathway "
            "diagnosed in Cells 8/9 where σ/τ gates went trivial "
            "because the model could fit the bin grid via PE10 inside "
            "Q/K alone."
        ),
    )
    pool_density_sfn: PoolDensitySfnConfig = Field(
        default_factory=PoolDensitySfnConfig,
        description=(
            "Optional Pool-Based Absolute Density SFN (Cell 7). When "
            "``enabled`` (and ``gaussian_attention_bias=True``), the GAB "
            "scalar is replaced by ``σ_head = σ_min + (σ_max − σ_min) "
            "· sigmoid(MLP([log ρ_local, log ρ_global]))``, where the "
            "densities are *unnormalised* Gaussian-kernel sums against a "
            "fixed pool buffer of training-pool event energies (no per-"
            "trial shot noise). See ``PoolDensitySfnConfig`` for the math. "
            "Mutually exclusive with ``bounded_bandwidth`` and "
            "``sfn_modulation`` — all three replace the GAB parametrisation."
        ),
    )

    @field_validator("attention_dim")
    @classmethod
    def _divisible_by_heads(cls, v: int, info) -> int:
        h = info.data.get("num_heads")
        if h is not None and v % h != 0:
            raise ValueError(f"attention_dim ({v}) must be divisible by num_heads ({h})")
        return v


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
    # Hybrid-scale sampling — Axis C: bin-stratified vs continuous-density.
    # Controls whether the event draw inside any pattern (mixed_density,
    # physics_anchored, random_clusters, flat_stratified) uses the
    # bin-uniform stratification (default; legacy behaviour) or an
    # inverse-density continuous draw with a strict ±half_w absolute
    # filter for local allocation. ``energy_bin_width`` and
    # ``min_events_per_bin`` are still used by the diagnostic plotting
    # grid and by the kept-event filter, but they no longer determine the
    # draw probabilities under ``continuous``.
    # ------------------------------------------------------------------
    density_sampling: Literal["bin_stratified", "continuous"] = Field(
        "bin_stratified",
        description=(
            "Draw mechanism inside each sampling pattern. ``bin_stratified`` "
            "(default) keeps the legacy uniform-over-kept-bins → uniform-"
            "within-bin two-step draw — bitwise-identical to every existing "
            "config. ``continuous`` replaces step 1 with an inverse-density "
            "continuous weight precomputed over the kept-event pool, and "
            "step 2 with a strict ``|E_i − E_focus| ≤ zoom_window_width_kev/2`` "
            "absolute filter for the local fraction. The continuous mode "
            "removes the 10-keV staircase in the marginal P(E) of training "
            "context — useful for diagnosing whether sub-bin discontinuities "
            "drive high-frequency model artefacts."
        ),
    )
    density_kde_radius_kev: float = Field(
        10.0,
        gt=0.0,
        description=(
            "Half-width (keV) of the box kernel used to estimate per-event "
            "local density when ``density_sampling == 'continuous'``. For "
            "each kept event ``i`` the inverse-density weight is "
            "``W_i = 1 / |{j : |E_j − E_i| ≤ density_kde_radius_kev}|``, "
            "normalised to sum to 1 across the kept pool. Larger radii "
            "smooth more aggressively; 10 keV matches the typical bin "
            "width and is a sensible default."
        ),
    )

    # ------------------------------------------------------------------
    # 1D Fourier positional encoding of the energy coordinate.
    # Default-disabled — see PositionalEncodingConfig docstring.
    # ------------------------------------------------------------------
    positional_encoding: PositionalEncodingConfig = Field(
        default_factory=PositionalEncodingConfig,
        description=(
            "Fourier feature expansion of E before it enters the encoder/"
            "decoder MLPs. Off by default."
        ),
    )

    # ------------------------------------------------------------------
    # Compute device. ``auto`` picks CUDA when available, falling back
    # to CPU silently — works on shared-GPU machines as well as
    # CPU-only CI without YAML edits. Set explicitly to ``cuda`` to
    # fail loudly if GPU is unavailable, or to ``cpu`` to force CPU
    # (useful for byte-for-byte reproducibility against pre-GPU runs).
    # ------------------------------------------------------------------
    device: Literal["auto", "cuda", "cpu"] = Field(
        "auto",
        description=(
            "Compute device for training. ``auto`` (default) uses CUDA "
            "when available, falls back to CPU. ``cuda`` raises if GPU "
            "unavailable. ``cpu`` forces CPU."
        ),
    )

    # ------------------------------------------------------------------
    # Pluggable context aggregator. Default ``type='mean'`` routes
    # through the upstream ``ConditionalNeuralProcess`` unchanged.
    # ``type='cross_attention'`` switches to the local AttentiveCNP.
    # ------------------------------------------------------------------
    aggregator: AggregatorConfig = Field(
        default_factory=AggregatorConfig,
        description=(
            "Context aggregator. ``mean`` (default) is bit-for-bit "
            "identical to the legacy CNP; ``cross_attention`` enables "
            "target-conditioned multi-head attention."
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
            raise ValueError(f"n_trial_events_max ({v}) must be >= n_trial_events_min ({n_min})")
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
