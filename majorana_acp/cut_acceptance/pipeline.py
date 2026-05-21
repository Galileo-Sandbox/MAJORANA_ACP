"""True-CNP cut-acceptance pipeline — *training only*.

The pipeline trains a 1D-regression CNP in ``InputMode.EVENT_ONLY``:
each context event carries its own coordinates ``(E_i_norm, T_norm)``
in ``phi``, and there is no broadcasted trial-level theta. The bin
grid is used **only** as a sampling-stratification mechanism inside
:class:`majorana_acp.cut_acceptance.event_sampler.EventSampler` — the
CNP itself never sees bin boundaries.

For one ``run_pipeline(cfg)`` call we:

1. Load the train-split predictions, filter by ``target_class`` +
   ``energy_range``, and build an :class:`EventSampler`.
2. Train a RESUM_FLEX CNP (``dim_theta=None``, ``dim_phi=2``) on it.
3. Compute the Youden-J best T* once from the *test* labels so
   downstream diagnostics share a fixed reference threshold.
4. Save the checkpoint + a small ``run_summary.json``.

Outputs (all under ``cfg.out_dir``):

* ``cnp.ckpt``           — RESUM_FLEX CNP checkpoint.
* ``training_pool.npz``  — bin centers + per-bin event counts on the
  train split. Kept *only* so the diagnostics script can bin D_T on the
  same grid for the blue Wilson errorbars; not load-bearing for the CNP.
* ``run_summary.json``   — scalars (paths, counts, final loss, T*).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from core import (
    TrainingHistory,
    build_cnp,
    cnp_loss,
    save_checkpoint,
    split_context_target,
)
from sklearn.metrics import roc_curve

from majorana_acp.cut_acceptance.config import CutAcceptanceConfig
from majorana_acp.cut_acceptance.event_sampler import EventSampler, load_events
from majorana_acp.models.attentive_cnp import build_attentive_cnp


@dataclass(frozen=True)
class PipelineSummary:
    name: str
    target_class: int | str
    energy_bin_width: float
    out_dir: str
    cnp_ckpt: str
    n_train_events: int
    n_validation_events: int
    n_bins_used: int
    cnp_final_train_loss: float
    youden_T_star: float
    # Upstream-classifier lineage — binds this run to an exact
    # classifier-config state. The SHA256 is the source of truth; the
    # path is kept for human readability.
    upstream_classifier_config: str
    upstream_classifier_sha256: str
    # Hybrid-scale fingerprint — orthogonal sampling axes + the
    # canonical paradigm tag derived from them. Lets the registry
    # distinguish variants of the same (model, bin, class) cell
    # without having to crack open the YAML.
    sampling_pattern: str
    trial_size_strategy: str
    paradigm_path_suffix: str

    def to_json(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))


def _sha256_file(path: Path) -> str:
    """Hex SHA256 of a file's bytes (streamed; fine for YAMLs and beyond)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Path derivation — auto-generate self-describing out_dir / name when the YAML
# leaves them blank, so hybrid-scale variants don't clobber each other's
# outputs by sharing a path.
# ---------------------------------------------------------------------------


def _class_label(target_class: int | str) -> str:
    if target_class == "all":
        return "inclusive"
    if int(target_class) == 1:
        return "signal"
    if int(target_class) == 0:
        return "background"
    raise ValueError(f"unsupported target_class: {target_class!r}")


def paradigm_path_suffix(cfg: CutAcceptanceConfig) -> str:
    """Map (sampling_pattern, trial_size_strategy, hybrid params) → path tag.

    The True-CNP baseline (``flat_stratified`` + ``fixed``) gets the
    well-known ``true_cnp`` tag. Every other combination lives under
    ``hybrid_scale/<pattern>_<serialized_params>``, with parameter
    values serialized as underscore-separated tokens (``f0_70`` for
    ``local_event_fraction=0.70``, ``w50`` for 50 keV windows, etc.)
    so the path is shell-safe and the experiment is recoverable from
    the directory name alone.
    """
    is_baseline = (
        cfg.sampling_pattern == "flat_stratified"
        and cfg.trial_size_strategy == "fixed"
        and not cfg.positional_encoding.enabled
        and cfg.aggregator.type == "mean"
    )
    if is_baseline:
        return "true_cnp"

    bits: list[str] = [cfg.sampling_pattern]
    if cfg.sampling_pattern in ("mixed_density", "physics_anchored"):
        # f0_70 instead of f0.70 so the path token is shell-safe.
        bits.append(f"f{cfg.local_event_fraction:.2f}".replace(".", "_"))
        bits.append(f"w{int(cfg.zoom_window_width_kev)}")
    elif cfg.sampling_pattern == "random_clusters":
        bits.append(f"n{cfg.n_clusters}")
        bits.append(f"w{int(cfg.zoom_window_width_kev)}")
    # flat_stratified gets no extra tokens (only differs via trial size).
    if cfg.trial_size_strategy == "variable_uniform":
        bits.append(f"varN{cfg.n_trial_events_min}-{cfg.n_trial_events_max}")
    # Append _pe<L> when the Fourier feature expansion is enabled so the
    # canonical results path stays unique across PE-on / PE-off variants
    # of an otherwise-identical sampling paradigm.
    if cfg.positional_encoding.enabled:
        bits.append(f"pe{cfg.positional_encoding.num_bands}")
    # Append _attn<H>x<d> only for cross-attention. ``mean`` aggregator
    # stays unsuffixed so legacy paths never collide.
    if cfg.aggregator.type == "cross_attention":
        bits.append(f"attn{cfg.aggregator.num_heads}x{cfg.aggregator.attention_dim}")
        # Append _gated when the decoder receives raw 2D coords instead
        # of the high-frequency z_phi — keeps the path unique vs the
        # un-gated attentive variant.
        if cfg.aggregator.decoder_coordinate_gating:
            bits.append("gated")
        # Append _gab when the attention layer carries the learnable
        # continuous Gaussian relative-distance bias.
        if cfg.aggregator.gaussian_attention_bias:
            bits.append("gab")
            # Append _dense when DM modulates the GAB penalty by the
            # local-vs-global density ratio R_*. DM without GAB is a
            # no-op so the suffix only fires when both are on.
            if cfg.aggregator.density_modulation.enabled:
                bits.append("dense")
            # Append _bpbn when the Bounded Physical Bandwidth Network
            # replaces the GAB scalar with an explicit clamped σ_head.
            # Mutually exclusive with the un-bounded log_gamma path —
            # the suffix flags the architectural divergence.
            if cfg.aggregator.bounded_bandwidth.enabled:
                bits.append("bpbn")
            # Append _sfn when the 2D Spectral Filter Network replaces
            # log_gamma. Mutually exclusive with BPBN.
            if cfg.aggregator.sfn_modulation.enabled:
                bits.append("sfn")
            # Append _pdsfn (Pool-Density SFN, Cell 7) when the
            # absolute-density pool-based SFN replaces log_gamma.
            # ``_dgsfn`` (Dual-Gated SFN, Cell 8) when the same
            # pool-based bandwidth is paired with the dual-output
            # temperature head. ``_dgsfn_tied`` (Tied-Head, Cell 9)
            # when both σ and τ are shared across all attention heads.
            # All mutually exclusive with BPBN and SFN.
            if cfg.aggregator.pool_density_sfn.enabled:
                if cfg.aggregator.pool_density_sfn.temperature_gating:
                    bits.append("dgsfn")
                    if cfg.aggregator.pool_density_sfn.head_tied:
                        bits.append("tied")
                    # Cell 13 — third SFN head η gating PE10 into the
                    # decoder concat. Sits alongside the σ/τ heads.
                    if cfg.aggregator.pool_density_sfn.pe_gated_decoder:
                        bits.append("pegate")
                    # Cell 14 — λ-head emitting per-band soft cutoffs
                    # applied to the PE10 sin/cos pairs directly. The
                    # decoder receives a SAPE(E_*) vector instead of
                    # z_phi_T. Mutually exclusive with ``pegate``.
                    if cfg.aggregator.pool_density_sfn.band_filter:
                        bits.append("bandfilter")
                    # Cell 15 — parameter-free hard-gated variant of
                    # band_filter. λ is an explicit closed-form
                    # function of the normalised density contrast,
                    # threshold-locked at the empirical HPGe rule.
                    if cfg.aggregator.pool_density_sfn.hard_filter:
                        bits.append("hardfilter")
                        # Cell 15 re-run — explicit contrast-feature
                        # injection. The scalar R(E_*) is appended to
                        # the decoder concat so the decoder reads raw
                        # peak intensity directly.
                        if cfg.aggregator.pool_density_sfn.inject_contrast_feature:
                            bits.append("xfeed")
                else:
                    bits.append("pdsfn")
                # Encode a non-default ``sigma_local_kev`` so different
                # local-kernel widths land at distinct canonical paths.
                # The default 10 keV stays unsuffixed for back-compat;
                # any other value gets ``sl<int(σ_l)>`` (Cell 12 uses 2).
                sl_default = 10.0
                sl_kev = cfg.aggregator.pool_density_sfn.sigma_local_kev
                if abs(sl_kev - sl_default) > 1e-9:
                    # Encode floats compactly: integers as ``sl5``,
                    # fractional values as ``sl0p5`` (0.5 keV).
                    if abs(sl_kev - round(sl_kev)) < 1e-9:
                        bits.append(f"sl{int(round(sl_kev))}")
                    else:
                        bits.append(f"sl{str(sl_kev).replace('.', 'p')}")
                sg_default = 150.0
                sg_kev = cfg.aggregator.pool_density_sfn.sigma_global_kev
                if abs(sg_kev - sg_default) > 1e-9:
                    if abs(sg_kev - round(sg_kev)) < 1e-9:
                        bits.append(f"sg{int(round(sg_kev))}")
                    else:
                        bits.append(f"sg{str(sg_kev).replace('.', 'p')}")
    # Append _debinned when the sampler uses the continuous inverse-
    # density draw instead of the legacy bin-stratified loop.
    if cfg.density_sampling == "continuous":
        bits.append("debinned")
    # Append _pedetach (Cell 10) when the attention's Q/K projections
    # are blinded to PE10 — Q·K^T becomes a smooth function of raw
    # energy, forcing the SFN gates to be the only path to sharp
    # localization.
    if cfg.aggregator.type == "cross_attention" and cfg.aggregator.pe_detach_qk:
        bits.append("pedetach")
    return "hybrid_scale/" + "_".join(bits)


def _classifier_model_name(cfg: CutAcceptanceConfig) -> str:
    """Recover the model name (e.g. ``simple_cnn_small``) from the upstream
    classifier-config path stem. Used to seed the auto-derived out_dir
    so cells of different models live under separate subtrees."""
    return Path(cfg.upstream_classifier_config).stem


def resolve_out_dir(cfg: CutAcceptanceConfig) -> Path:
    """YAML-explicit ``out_dir`` wins; otherwise auto-derive from the
    canonical sibling-tree layout."""
    if cfg.out_dir is not None:
        return Path(cfg.out_dir)
    return Path(
        "results",
        "cut_acceptance",
        _classifier_model_name(cfg),
        paradigm_path_suffix(cfg),
        f"bin{int(cfg.energy_bin_width)}",
        _class_label(cfg.target_class),
    )


def resolve_name(cfg: CutAcceptanceConfig) -> str:
    """YAML-explicit ``name`` wins; otherwise build from the paradigm tag."""
    if cfg.name is not None:
        return cfg.name
    model = _classifier_model_name(cfg)
    paradigm = paradigm_path_suffix(cfg).replace("/", "__")
    return f"{model}_bin{int(cfg.energy_bin_width)}_{_class_label(cfg.target_class)}__{paradigm}"


# ---------------------------------------------------------------------------
# CNP build dispatcher — selects the upstream mean-aggregator CNP or our
# local ``AttentiveCNP`` based on ``cfg.aggregator.type``. The signature
# returned in both cases satisfies the upstream ``train_cnp`` /
# ``cnp_loss`` contract (``forward(ctx, tgt) → CnpOutput``), so the
# trainer never sees the dispatch.
# ---------------------------------------------------------------------------


def build_local_cnp(cfg: CutAcceptanceConfig, *, dim_phi: int):
    """Build a CNP whose aggregator matches ``cfg.aggregator.type``.

    ``mean`` (default) routes through upstream ``core.build_cnp`` exactly
    as the legacy code did — byte-identical to every pre-aggregator
    checkpoint. ``cross_attention`` builds our local ``AttentiveCNP``
    from ``majorana_acp.models.attentive_cnp.build_attentive_cnp``.
    """
    if cfg.aggregator.type == "mean":
        return build_cnp(
            cfg.encoder,
            dim_theta=None,
            dim_phi=dim_phi,
            decoder_hidden_dims=list(cfg.decoder_hidden_dims),
        )
    if cfg.aggregator.type == "cross_attention":
        dm = cfg.aggregator.density_modulation
        bb = cfg.aggregator.bounded_bandwidth
        sfn = cfg.aggregator.sfn_modulation
        pdsfn = cfg.aggregator.pool_density_sfn
        # ``energy_range_kev`` is needed by DM / BPBN / SFN / PD-SFN for
        # the keV → normalised σ conversion. Provide it whenever any
        # feature is on; the factory raises if missing when required.
        needs_range = dm.enabled or bb.enabled or sfn.enabled or pdsfn.enabled
        # Pool-Density SFN needs the kept-event energies of the training
        # pool as a fixed buffer. Re-load via the same call the sampler
        # uses (target_class + energy_range filter) so the buffer
        # reconstructs identically at train and inference time without
        # being shipped in the checkpoint.
        pool_energies_kev = None
        if pdsfn.enabled:
            pool_e, _ = load_events(
                cfg.train_predictions_path,
                target_class=cfg.target_class,
                energy_range=cfg.energy_range,
            )
            pool_energies_kev = pool_e
        return build_attentive_cnp(
            cfg.encoder,
            dim_theta=None,
            dim_phi=dim_phi,
            num_heads=cfg.aggregator.num_heads,
            attention_dim=cfg.aggregator.attention_dim,
            decoder_coordinate_gating=cfg.aggregator.decoder_coordinate_gating,
            gaussian_attention_bias=cfg.aggregator.gaussian_attention_bias,
            density_modulation_enabled=dm.enabled,
            density_sigma_local_kev=dm.sigma_local_kev if dm.enabled else None,
            density_sigma_global_kev=dm.sigma_global_kev if dm.enabled else None,
            density_epsilon=dm.epsilon,
            bounded_bandwidth_enabled=bb.enabled,
            bounded_sigma_max_kev=bb.sigma_max_kev if bb.enabled else None,
            bounded_sigma_min_kev=bb.sigma_min_kev if bb.enabled else None,
            bounded_alpha_max=bb.alpha_max if bb.enabled else None,
            bounded_sensitivity_hidden_dim=bb.sensitivity_hidden_dim,
            bounded_sigma_local_kev=bb.sigma_local_kev if bb.enabled else None,
            bounded_sigma_global_kev=bb.sigma_global_kev if bb.enabled else None,
            bounded_epsilon=bb.epsilon,
            sfn_modulation_enabled=sfn.enabled,
            sfn_sigma_max_kev=sfn.sigma_max_kev if sfn.enabled else None,
            sfn_sigma_min_kev=sfn.sigma_min_kev if sfn.enabled else None,
            sfn_hidden_dim=sfn.hidden_dim,
            sfn_sigma_local_kev=sfn.sigma_local_kev if sfn.enabled else None,
            sfn_sigma_global_kev=sfn.sigma_global_kev if sfn.enabled else None,
            sfn_epsilon=sfn.epsilon,
            pool_density_sfn_enabled=pdsfn.enabled,
            pool_density_sfn_sigma_max_kev=pdsfn.sigma_max_kev if pdsfn.enabled else None,
            pool_density_sfn_sigma_min_kev=pdsfn.sigma_min_kev if pdsfn.enabled else None,
            pool_density_sfn_hidden_dim=pdsfn.hidden_dim,
            pool_density_sfn_sigma_local_kev=pdsfn.sigma_local_kev if pdsfn.enabled else None,
            pool_density_sfn_sigma_global_kev=pdsfn.sigma_global_kev if pdsfn.enabled else None,
            pool_density_sfn_epsilon=pdsfn.epsilon,
            pool_density_sfn_temperature_gating=pdsfn.temperature_gating,
            pool_density_sfn_tau_min=pdsfn.tau_min_value,
            pool_density_sfn_tau_max=pdsfn.tau_max_value,
            pool_density_sfn_head_tied=pdsfn.head_tied,
            pool_density_sfn_pe_gated_decoder=pdsfn.pe_gated_decoder,
            pool_density_sfn_band_filter=pdsfn.band_filter,
            pool_density_sfn_band_filter_alpha=pdsfn.band_filter_alpha,
            pool_density_sfn_num_bands=(
                cfg.positional_encoding.num_bands
                if cfg.positional_encoding.enabled
                else 0
            ),
            pool_density_sfn_hard_filter=pdsfn.hard_filter,
            pool_density_sfn_hard_filter_contrast_threshold=pdsfn.hard_filter_contrast_threshold,
            pool_density_sfn_hard_filter_sigmoid_steepness=pdsfn.hard_filter_sigmoid_steepness,
            pool_density_sfn_hard_filter_lambda_min=pdsfn.hard_filter_lambda_min,
            pool_density_sfn_hard_filter_lambda_max=pdsfn.hard_filter_lambda_max,
            pool_density_sfn_inject_contrast_feature=pdsfn.inject_contrast_feature,
            pe_detach_qk=cfg.aggregator.pe_detach_qk,
            pool_energies_kev=pool_energies_kev,
            energy_range_kev=cfg.energy_range if needs_range else None,
            decoder_hidden_dims=list(cfg.decoder_hidden_dims),
        )
    raise ValueError(f"unknown aggregator.type: {cfg.aggregator.type!r}")


# ---------------------------------------------------------------------------
# Variable-N training wrapper — used only when
# ``trial_size_strategy == 'variable_uniform'``.
# ---------------------------------------------------------------------------


def _resolve_device(preference: str | None = None) -> torch.device:
    """Resolve a torch device from ``"auto" | "cuda" | "cpu"``.

    ``"auto"`` picks CUDA when available, falling back to CPU silently.
    Explicit ``"cuda"`` raises if CUDA isn't available so that GPU-only
    config files fail loudly rather than running 10× slower than the
    user expects. ``"cpu"`` always honors the explicit choice.
    """
    pref = (preference or "auto").lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' requested but torch.cuda.is_available() is "
                "False — install a CUDA build of torch or set device='cpu'."
            )
        return torch.device("cuda")
    if pref != "auto":
        raise ValueError(f"unknown device preference: {preference!r}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_cnp_local(
    cnp,
    sampler: EventSampler,
    *,
    cnp_config,
    training_config,
    device: torch.device,
    variable_n: bool = False,
    n_min: int | None = None,
    n_max: int | None = None,
) -> TrainingHistory:
    """Device-aware mirror of ``core.train_cnp`` with optional per-step variable N.

    Replaces both the upstream ``train_cnp`` (when ``variable_n=False``)
    and the previous ``train_cnp_variable_n_per_step`` wrapper. Behaviour
    is otherwise identical to the upstream loop except for two
    deliberate divergences:

    * ``x_target`` is moved to ``device`` before the loss call — upstream
      leaves the binary label tensor on CPU, which is fine when the
      model is also on CPU but raises a device mismatch on GPU.
    * When ``variable_n=True``, each step draws a fresh ``n_events``
      from ``Uniform[n_min, n_max]`` (uniform per-step, not per-trial,
      so ``StandardBatch.labels`` keeps its fixed ``[B, N]`` shape).

    The training loop assumes the caller has already moved ``cnp`` to
    ``device``. The sampler stays CPU-side and emits numpy arrays; the
    upstream encoder.forward picks up the device from its parameters
    and moves the batch tensors automatically.
    """
    if variable_n:
        if n_min is None or n_max is None:
            raise ValueError("variable_n=True requires n_min and n_max")
        if n_max < n_min:
            raise ValueError(f"n_max ({n_max}) < n_min ({n_min})")
        if n_min < 2:
            raise ValueError(f"n_min must be >= 2, got {n_min}")
    else:
        if training_config.n_events_per_trial < 2:
            raise ValueError(
                f"n_events_per_trial must be >= 2, got {training_config.n_events_per_trial}"
            )

    rng = np.random.default_rng(training_config.seed)
    torch.manual_seed(training_config.seed)
    optimizer = torch.optim.Adam(cnp.parameters(), lr=training_config.learning_rate)

    history: TrainingHistory = {
        "step": [],
        "loss": [],
        "eval_step": [],
        "eval_mae": [],
    }
    cnp_n_ctx_min = max(1, cnp_config.n_context_min)
    cnp_n_ctx_max_cfg = cnp_config.n_context_max

    if not variable_n:
        n_events_fixed = training_config.n_events_per_trial
        n_ctx_max_eff_fixed = min(cnp_n_ctx_max_cfg, n_events_fixed - 1)
        if n_ctx_max_eff_fixed < cnp_n_ctx_min:
            raise ValueError(
                f"Effective n_context range is empty: min={cnp_n_ctx_min}, "
                f"max={n_ctx_max_eff_fixed}; n_events_per_trial={n_events_fixed} "
                "must exceed n_context_min."
            )

    cnp.train()
    for step in range(training_config.n_steps):
        if variable_n:
            n_events = int(rng.integers(n_min, n_max + 1))
            n_ctx_max_eff = min(cnp_n_ctx_max_cfg, n_events - 1)
            if n_ctx_max_eff < cnp_n_ctx_min:
                continue
        else:
            n_events = n_events_fixed
            n_ctx_max_eff = n_ctx_max_eff_fixed
        n_ctx = int(rng.integers(cnp_n_ctx_min, n_ctx_max_eff + 1))

        batch = sampler.generate(
            n_trials=training_config.batch_size,
            n_events=n_events,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        ctx, tgt = split_context_target(
            batch, n_context=n_ctx, seed=int(rng.integers(0, 2**31 - 1))
        )
        out = cnp(ctx, tgt)
        x_target = torch.as_tensor(tgt.labels, dtype=torch.float32, device=device)
        loss = cnp_loss(out, x_target, n_mc_samples=training_config.n_mc_samples)

        optimizer.zero_grad()
        loss.backward()
        if training_config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(cnp.parameters(), training_config.grad_clip)
        optimizer.step()

        history["step"].append(step)
        history["loss"].append(float(loss.item()))

    return history


# ---------------------------------------------------------------------------
# Wilson score interval (1σ) for a binomial proportion.
# ---------------------------------------------------------------------------


def wilson_interval(
    k: np.ndarray, n: np.ndarray, *, z: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lower, upper) Wilson score interval at level ``z``.

    Handles ``k=0``, ``k=n`` and small ``n`` correctly — the naive
    ``√(p(1-p)/n)`` errorbar collapses to 0 in those cases and is
    symmetric where it shouldn't be. Returns NaN bounds where ``n=0``.
    Inputs broadcast; outputs follow numpy broadcasting rules.
    """
    k = np.asarray(k, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, k / n, np.nan)
        denom = 1.0 + (z * z) / n
        center = (p + (z * z) / (2.0 * n)) / denom
        half = (z * np.sqrt(p * (1.0 - p) / n + (z * z) / (4.0 * n * n))) / denom
    lo = np.clip(center - half, 0.0, 1.0)
    hi = np.clip(center + half, 0.0, 1.0)
    # FP guarantees: lo ≤ p ≤ hi (rounding can otherwise produce
    # ``hi = 0.9999…8`` against ``p = 1.0``, which breaks matplotlib
    # errorbars with negative half-widths).
    lo = np.minimum(lo, p)
    hi = np.maximum(hi, p)
    lo = np.where(n > 0, lo, np.nan)
    hi = np.where(n > 0, hi, np.nan)
    return lo, hi


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(cfg: CutAcceptanceConfig, *, seed: int = 0) -> PipelineSummary:
    """Train the CNP and save the checkpoint. No scoring / coverage."""
    out_dir = resolve_out_dir(cfg)
    resolved_name = resolve_name(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build the training sampler — thread the hybrid-scale knobs through.
    train_e, train_s = load_events(
        cfg.train_predictions_path,
        target_class=cfg.target_class,
        energy_range=cfg.energy_range,
    )
    sampler = EventSampler(
        train_e,
        train_s,
        energy_range=cfg.energy_range,
        energy_bin_width=cfg.energy_bin_width,
        threshold_range=cfg.threshold_range,
        min_events_per_bin=cfg.min_events_per_bin,
        t_sampling="boundary_mix",
        sampling_pattern=cfg.sampling_pattern,
        zoom_window_width_kev=cfg.zoom_window_width_kev,
        local_event_fraction=cfg.local_event_fraction,
        n_clusters=cfg.n_clusters,
        physics_peaks_kev=list(cfg.physics_peaks_kev),
        positional_encoding=cfg.positional_encoding,
        density_sampling=cfg.density_sampling,
        density_kde_radius_kev=cfg.density_kde_radius_kev,
    )
    np.savez(
        out_dir / "training_pool.npz",
        bin_centers=sampler.bin_centers,
        bin_event_counts=sampler.bin_event_counts,
        n_events_total=np.int64(train_e.size),
    )

    # 2. Train the CNP — EVENT_ONLY with per-event (E_i, T) in phi.
    #    Stock train_cnp for fixed N; variable-N wrapper otherwise.
    torch.manual_seed(cfg.training.seed)
    # dim_phi is dynamic: 2 (default) when PE is off, 2*L + 1 when PE is on.
    # The sampler computed it during __init__; we read it back here so the
    # CNP architecture matches what the StandardBatch will carry.
    # Dispatch by aggregator type — mean routes to upstream build_cnp
    # (byte-identical to legacy), cross_attention routes to our local
    # AttentiveCNP. Both return objects with the same
    # ``forward(ctx, tgt) → CnpOutput`` contract so the trainer below
    # is aggregator-agnostic.
    dim_phi = sampler.dim_phi
    cnp = build_local_cnp(cfg, dim_phi=dim_phi)
    # Resolve device once per run. The factory always builds CPU-side
    # so the optional pool buffer (register_buffer) is created CPU-side
    # too; ``.to(device)`` then moves params AND non-persistent buffers
    # together to GPU in one shot. Same call works on CPU-only machines.
    device = _resolve_device(getattr(cfg, "device", "auto"))
    cnp.to(device)
    variable_n = cfg.trial_size_strategy == "variable_uniform"
    history = train_cnp_local(
        cnp,
        sampler,
        cnp_config=cfg.cnp,
        training_config=cfg.training,
        device=device,
        variable_n=variable_n,
        n_min=cfg.n_trial_events_min if variable_n else None,
        n_max=cfg.n_trial_events_max if variable_n else None,
    )
    # Move the model back to CPU before saving so the checkpoint stays
    # device-agnostic (loadable on any machine regardless of CUDA).
    cnp.to("cpu")
    save_checkpoint(
        out_dir / "cnp.ckpt",
        cnp,
        encoder_config=cfg.encoder,
        dim_theta=None,
        dim_phi=dim_phi,
        history=history,
        metadata={
            "name": resolved_name,
            "target_class": cfg.target_class,
            "energy_bin_width": cfg.energy_bin_width,
            "train_predictions_path": str(cfg.train_predictions_path),
            "validation_predictions_path": str(cfg.validation_predictions_path),
            "decoder_hidden_dims": list(cfg.decoder_hidden_dims),
            "input_mode": "event_only",
            "sampling_pattern": cfg.sampling_pattern,
            "trial_size_strategy": cfg.trial_size_strategy,
            "paradigm_path_suffix": paradigm_path_suffix(cfg),
            "positional_encoding_enabled": cfg.positional_encoding.enabled,
            "positional_encoding_num_bands": cfg.positional_encoding.num_bands,
            "aggregator_type": cfg.aggregator.type,
            "aggregator_num_heads": cfg.aggregator.num_heads,
            "aggregator_attention_dim": cfg.aggregator.attention_dim,
            "decoder_coordinate_gating": cfg.aggregator.decoder_coordinate_gating,
            "gaussian_attention_bias": cfg.aggregator.gaussian_attention_bias,
            "density_modulation_enabled": cfg.aggregator.density_modulation.enabled,
            "density_modulation_sigma_local_kev": cfg.aggregator.density_modulation.sigma_local_kev,
            "density_modulation_sigma_global_kev": cfg.aggregator.density_modulation.sigma_global_kev,
            "bounded_bandwidth_enabled": cfg.aggregator.bounded_bandwidth.enabled,
            "bounded_sigma_max_kev": cfg.aggregator.bounded_bandwidth.sigma_max_kev,
            "bounded_sigma_min_kev": cfg.aggregator.bounded_bandwidth.sigma_min_kev,
            "bounded_alpha_max": cfg.aggregator.bounded_bandwidth.alpha_max,
            "sfn_modulation_enabled": cfg.aggregator.sfn_modulation.enabled,
            "sfn_sigma_max_kev": cfg.aggregator.sfn_modulation.sigma_max_kev,
            "sfn_sigma_min_kev": cfg.aggregator.sfn_modulation.sigma_min_kev,
            "sfn_hidden_dim": cfg.aggregator.sfn_modulation.hidden_dim,
            "pool_density_sfn_enabled": cfg.aggregator.pool_density_sfn.enabled,
            "pool_density_sfn_sigma_max_kev": cfg.aggregator.pool_density_sfn.sigma_max_kev,
            "pool_density_sfn_sigma_min_kev": cfg.aggregator.pool_density_sfn.sigma_min_kev,
            "pool_density_sfn_hidden_dim": cfg.aggregator.pool_density_sfn.hidden_dim,
            "pool_density_sfn_sigma_local_kev": cfg.aggregator.pool_density_sfn.sigma_local_kev,
            "pool_density_sfn_sigma_global_kev": cfg.aggregator.pool_density_sfn.sigma_global_kev,
            "pool_density_sfn_epsilon": cfg.aggregator.pool_density_sfn.epsilon,
            "pool_density_sfn_temperature_gating": cfg.aggregator.pool_density_sfn.temperature_gating,
            "pool_density_sfn_tau_min_value": cfg.aggregator.pool_density_sfn.tau_min_value,
            "pool_density_sfn_tau_max_value": cfg.aggregator.pool_density_sfn.tau_max_value,
            "pool_density_sfn_head_tied": cfg.aggregator.pool_density_sfn.head_tied,
            "pool_density_sfn_pe_gated_decoder": cfg.aggregator.pool_density_sfn.pe_gated_decoder,
            "pool_density_sfn_band_filter": cfg.aggregator.pool_density_sfn.band_filter,
            "pool_density_sfn_band_filter_alpha": cfg.aggregator.pool_density_sfn.band_filter_alpha,
            "pool_density_sfn_hard_filter": cfg.aggregator.pool_density_sfn.hard_filter,
            "pool_density_sfn_hard_filter_contrast_threshold": cfg.aggregator.pool_density_sfn.hard_filter_contrast_threshold,
            "pool_density_sfn_hard_filter_sigmoid_steepness": cfg.aggregator.pool_density_sfn.hard_filter_sigmoid_steepness,
            "pool_density_sfn_hard_filter_lambda_min": cfg.aggregator.pool_density_sfn.hard_filter_lambda_min,
            "pool_density_sfn_hard_filter_lambda_max": cfg.aggregator.pool_density_sfn.hard_filter_lambda_max,
            "pool_density_sfn_inject_contrast_feature": cfg.aggregator.pool_density_sfn.inject_contrast_feature,
            "pe_detach_qk": cfg.aggregator.pe_detach_qk,
            "density_sampling": cfg.density_sampling,
            "density_kde_radius_kev": cfg.density_kde_radius_kev,
            "device": str(device),
        },
    )
    final_train_loss = float(history["loss"][-1]) if history.get("loss") else float("nan")

    # 3. Youden-J best T* from the test labels — recorded once here so
    #    downstream diagnostics on the same (model, class) share a
    #    reference threshold without having to recompute it.
    with h5py.File(cfg.validation_predictions_path, "r") as f:
        val_score = f["score"][:].astype(np.float64)
        val_label = f["label"][:].astype(np.int64)
        val_energy = f["energy"][:].astype(np.float64)
    fpr, tpr, thr = roc_curve(val_label, val_score)
    T_star = float(thr[int(np.argmax(tpr - fpr))])

    # Count validation events that pass the (class, energy_range) filter.
    if cfg.target_class == "all":
        cls_mask = np.ones_like(val_label, dtype=bool)
    else:
        cls_mask = val_label == int(cfg.target_class)
    e_lo, e_hi = cfg.energy_range
    keep = cls_mask & (val_energy >= e_lo) & (val_energy <= e_hi)
    n_validation_events = int(keep.sum())

    # Upstream classifier lineage — fail loudly if the referenced YAML
    # is missing, since a wrong/stale reference invalidates the whole run.
    upstream_path = Path(cfg.upstream_classifier_config)
    if not upstream_path.is_file():
        raise FileNotFoundError(
            f"upstream_classifier_config does not exist: {upstream_path}. "
            "Configure a path relative to the repo root or fix the file."
        )
    upstream_sha = _sha256_file(upstream_path)

    summary = PipelineSummary(
        name=resolved_name,
        target_class=cfg.target_class,
        energy_bin_width=cfg.energy_bin_width,
        out_dir=str(out_dir),
        cnp_ckpt=str(out_dir / "cnp.ckpt"),
        n_train_events=int(train_e.size),
        n_validation_events=n_validation_events,
        n_bins_used=int(sampler.n_bins),
        cnp_final_train_loss=final_train_loss,
        youden_T_star=T_star,
        upstream_classifier_config=str(upstream_path),
        upstream_classifier_sha256=upstream_sha,
        sampling_pattern=cfg.sampling_pattern,
        trial_size_strategy=cfg.trial_size_strategy,
        paradigm_path_suffix=paradigm_path_suffix(cfg),
    )
    summary.to_json(out_dir / "run_summary.json")
    return summary
