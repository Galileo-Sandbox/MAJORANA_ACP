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
    train_cnp,
)
from sklearn.metrics import roc_curve

from majorana_acp.cut_acceptance.config import CutAcceptanceConfig
from majorana_acp.cut_acceptance.event_sampler import EventSampler, load_events


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
# Variable-N training wrapper — used only when
# ``trial_size_strategy == 'variable_uniform'``.
# ---------------------------------------------------------------------------


def train_cnp_variable_n_per_step(
    cnp,
    sampler: EventSampler,
    *,
    cnp_config,
    training_config,
    n_min: int,
    n_max: int,
) -> TrainingHistory:
    """Variant of ``core.train_cnp`` that resamples ``n_events`` per step
    uniformly from ``[n_min, n_max]``.

    Within a single batch all trials share the same N — required by
    ``StandardBatch.labels``'s fixed ``[B, N]`` shape and the encoder's
    lack of a padding mask. Across thousands of steps the model still
    sees the full N distribution. ``n_ctx_max`` is clamped per step
    to ``n_events − 1`` so ``split_context_target`` never receives an
    out-of-range request.

    Implementation mirrors ``core.train_cnp`` line-for-line *except*
    the per-step resampling of ``n_events``, so the upstream training
    loop stays untouched.
    """
    if n_max < n_min:
        raise ValueError(f"n_max ({n_max}) < n_min ({n_min})")
    if n_min < 2:
        raise ValueError(f"n_min must be >= 2, got {n_min}")

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
    cnp_n_ctx_max = cnp_config.n_context_max

    cnp.train()
    for step in range(training_config.n_steps):
        n_events = int(rng.integers(n_min, n_max + 1))
        n_ctx_max_eff = min(cnp_n_ctx_max, n_events - 1)
        if n_ctx_max_eff < cnp_n_ctx_min:
            # n_events too small to split — should be ruled out by the
            # n_min ≥ cnp.n_context_min + 1 invariant, but defend anyway.
            continue
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
        x_target = torch.as_tensor(tgt.labels, dtype=torch.float32)
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
    dim_phi = sampler.dim_phi
    cnp = build_cnp(
        cfg.encoder,
        dim_theta=None,
        dim_phi=dim_phi,
        decoder_hidden_dims=list(cfg.decoder_hidden_dims),
    )
    if cfg.trial_size_strategy == "variable_uniform":
        history = train_cnp_variable_n_per_step(
            cnp,
            sampler,
            cnp_config=cfg.cnp,
            training_config=cfg.training,
            n_min=cfg.n_trial_events_min,
            n_max=cfg.n_trial_events_max,
        )
    else:
        history = train_cnp(
            cnp,
            sampler,
            cnp_config=cfg.cnp,
            training_config=cfg.training,
        )
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
