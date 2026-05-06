"""End-to-end training driver.

``train(cfg)`` is the single entry point. Given a fully validated
:class:`ExperimentConfig` it builds the train + test datasets, model,
loss, and optimizer, runs the training loop, and writes the following
artifacts to ``cfg.train.out_dir``:

* ``metadata.json``         — config snapshot + runtime info
  (git SHA, host, versions, start/end times, completed epochs,
  final per-epoch metrics).
* ``training_history.json`` — per-epoch ``train_loss``, ``test_loss``,
  ``test_roc_auc`` so loss curves can be reconstructed.
* ``epoch_NNN.pt``          — model + optimizer checkpoint per epoch.
* ``train.log``             — log file scoped to this run.

The test set (``cfg.data.test_file_indices``) is used purely as a
held-out monitoring set during training. Train and test files come from
disjoint Zenodo files and are never mixed.
"""

from __future__ import annotations

import contextlib
import json
import logging
import random
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn import metrics as skm
from torch import nn, optim
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler

from majorana_acp.data import (
    DatasetConfig,
    MajoranaWaveformDataset,
    resolve_files,
)
from majorana_acp.models import build_model
from majorana_acp.training.config import (
    DataConfig,
    DeviceName,
    ExperimentConfig,
    OptimConfig,
    SamplerStrategy,
)
from majorana_acp.training.loss import build_loss

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed Python ``random``, NumPy, and torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: DeviceName) -> torch.device:
    """Map a config device string to a concrete ``torch.device``."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is not available")
    return torch.device(name)


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> optim.Optimizer:
    if cfg.optimizer == "adam":
        return optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            momentum=cfg.momentum,
        )
    raise ValueError(f"Unknown optimizer: {cfg.optimizer!r}")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    cfg: ExperimentConfig,
) -> None:
    """Write a checkpoint with model + optimizer state and the config."""
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "config": cfg.model_dump(mode="json"),
        },
        path,
    )


def train(cfg: ExperimentConfig) -> Path:
    """Run training end-to-end and return the output directory."""
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)

    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_handler = _attach_file_handler(out_dir / "train.log")

    # Initial metadata snapshot — written eagerly so even a crashed run
    # leaves a partial record on disk.
    metadata: dict = {
        "name": cfg.name,
        "config": cfg.model_dump(mode="json"),
        "runtime": _collect_runtime_info(device),
        "start_time": _utc_now_iso(),
        "end_time": None,
        "completed_epochs": 0,
        "final_metrics": None,
    }
    _write_json(out_dir / "metadata.json", metadata)

    history: list[dict] = []
    _write_json(out_dir / "training_history.json", history)

    try:
        logger.info("experiment=%s  device=%s  out_dir=%s", cfg.name, device, out_dir)

        # ---- Train data ------------------------------------------
        train_files = resolve_files(cfg.data.data_dir, "train", cfg.data.train_file_indices)
        logger.info("loaded %d training file(s)", len(train_files))

        train_ds = MajoranaWaveformDataset(
            DatasetConfig(
                files=train_files,
                target_label=cfg.data.target_label,
                baseline_samples=cfg.data.baseline_samples,
                align_t90=cfg.data.align_t90,
                t90_pre=cfg.data.t90_pre,
                t90_post=cfg.data.t90_post,
                use_derivative_channel=cfg.data.use_derivative_channel,
                energy_range=cfg.data.energy_range,
                subset_portion=cfg.data.subset_portion,
                subset_seed=cfg.data.subset_seed,
            )
        )
        logger.info("train set: %d events", len(train_ds))

        sampler = _build_train_sampler(
            dataset=train_ds,
            data_cfg=cfg.data,
            legacy_balanced_sampler=cfg.loss.balanced_sampler,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.data.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=cfg.data.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        # ---- Test data (held-out monitoring) ---------------------
        test_files = resolve_files(cfg.data.data_dir, "test", cfg.data.test_file_indices)
        test_ds = MajoranaWaveformDataset(
            DatasetConfig(
                files=test_files,
                target_label=cfg.data.target_label,
                baseline_samples=cfg.data.baseline_samples,
                align_t90=cfg.data.align_t90,
                t90_pre=cfg.data.t90_pre,
                t90_post=cfg.data.t90_post,
                use_derivative_channel=cfg.data.use_derivative_channel,
                energy_range=cfg.data.energy_range,
                subset_portion=cfg.data.subset_portion,
                subset_seed=cfg.data.subset_seed,
            )
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        logger.info("test set: %d events (used for per-epoch monitoring)", len(test_ds))

        # ---- Model ------------------------------------------------
        model = build_model(cfg.model.name, **cfg.model.params).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info("model=%s  params=%d", cfg.model.name, n_params)

        # ---- Loss + optimizer ------------------------------------
        loss_fn = build_loss(cfg.loss, files=train_files, target_label=cfg.data.target_label).to(
            device
        )
        optimizer = build_optimizer(model, cfg.optim)

        # ---- AMP --------------------------------------------------
        use_amp = cfg.train.amp and device.type == "cuda"
        scaler = torch.amp.GradScaler() if use_amp else None
        if use_amp:
            logger.info("mixed precision (AMP) enabled")

        # ---- Train loop ------------------------------------------
        for epoch in range(1, cfg.train.epochs + 1):
            train_loss = _train_one_epoch(
                model=model,
                loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                epoch=epoch,
                log_every=cfg.train.log_every_n_steps,
            )

            test_metrics = _eval_test_set(model, test_loader, loss_fn, device)

            entry = {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_metrics["loss"],
                "test_roc_auc": test_metrics["roc_auc"],
            }
            history.append(entry)
            _write_json(out_dir / "training_history.json", history)

            ckpt_path = out_dir / f"epoch_{epoch:03d}.pt"
            save_checkpoint(ckpt_path, model, optimizer, epoch, cfg)

            auc_str = (
                f"{test_metrics['roc_auc']:.4f}" if test_metrics["roc_auc"] is not None else "n/a"
            )
            logger.info(
                "epoch=%d  train_loss=%.4f  test_loss=%.4f  test_auc=%s  ckpt=%s",
                epoch,
                train_loss,
                test_metrics["loss"],
                auc_str,
                ckpt_path.name,
            )

            metadata["completed_epochs"] = epoch
            metadata["final_metrics"] = entry
            _write_json(out_dir / "metadata.json", metadata)

        metadata["end_time"] = _utc_now_iso()
        _write_json(out_dir / "metadata.json", metadata)
    finally:
        _detach_file_handler(file_handler)

    return out_dir


def _train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    log_every: int,
) -> float:
    model.train()
    loss_sum = 0.0
    n_batches = 0
    epoch_start = time.perf_counter()

    for step, batch in enumerate(loader, 1):
        wf = batch["waveform"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type=device.type):
                logits = model(wf)
                loss = loss_fn(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(wf)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()

        loss_val = float(loss.item())
        loss_sum += loss_val
        n_batches += 1

        if step % log_every == 0:
            logger.info("epoch=%d  step=%d  loss=%.4f", epoch, step, loss_val)

    elapsed = time.perf_counter() - epoch_start
    avg = loss_sum / max(n_batches, 1)
    logger.info("epoch=%d  done  steps=%d  elapsed=%.1fs", epoch, n_batches, elapsed)
    return avg


@torch.no_grad()
def _eval_test_set(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict:
    """Return mean loss + ROC-AUC over ``loader``. Used per-epoch.

    ROC-AUC is None if only one class is present in the test set.
    """
    model.eval()
    loss_sum = 0.0
    n_batches = 0
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in loader:
        wf = batch["waveform"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        logits = model(wf)
        loss = loss_fn(logits, target)

        loss_sum += float(loss.item())
        n_batches += 1
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(target.detach().cpu().numpy())

    avg_loss = loss_sum / max(n_batches, 1)
    logits_arr = np.concatenate(all_logits) if all_logits else np.array([], dtype=np.float32)
    labels_arr = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.float32)
    scores = 1.0 / (1.0 + np.exp(-logits_arr))

    auc: float | None = None
    n_pos = int(labels_arr.sum())
    n_neg = int(labels_arr.size) - n_pos
    if n_pos > 0 and n_neg > 0:
        auc = float(skm.roc_auc_score(labels_arr.astype(bool), scores))

    return {"loss": avg_loss, "roc_auc": auc}


def _build_train_sampler(
    *,
    dataset: MajoranaWaveformDataset,
    data_cfg: DataConfig,
    legacy_balanced_sampler: bool,
) -> RandomSampler | WeightedRandomSampler | None:
    """Build the train-side sampler.

    Composition rules:
    - ``data_cfg.sampler_strategies`` lists strategies whose per-event
      weights are multiplied together. Empty list = no weighting.
    - The legacy ``loss.balanced_sampler=True`` flag is folded in by
      ensuring ``"class_balanced"`` is part of the effective strategy list.
    - ``data_cfg.train_portion`` controls ``num_samples`` for the sampler.

    Returns ``None`` for the trivial case (no strategies and full data),
    a ``RandomSampler`` for sub-sampling without weighting, or a
    ``WeightedRandomSampler`` for any weighted case.
    """
    n = len(dataset)
    num_samples = int(n * data_cfg.train_portion)
    if num_samples < 1:
        raise ValueError(
            f"train_portion={data_cfg.train_portion} on dataset of size {n} yields 0 samples"
        )

    strategies: list[SamplerStrategy] = list(data_cfg.sampler_strategies)
    if legacy_balanced_sampler and "class_balanced" not in strategies:
        strategies.append("class_balanced")

    if not strategies:
        if data_cfg.train_portion < 1.0:
            logger.info(
                "subsampling training set: train_portion=%.3f -> %d / %d events",
                data_cfg.train_portion,
                num_samples,
                n,
            )
            return RandomSampler(dataset, replacement=False, num_samples=num_samples)
        return None

    logger.info(
        "weighted sampler  strategies=%s  num_samples=%d / %d",
        strategies,
        num_samples,
        n,
    )
    weights = _compute_sampler_weights(
        dataset=dataset,
        strategies=strategies,
        target_label=data_cfg.target_label,
        energy_bin_width_kev=data_cfg.energy_bin_width_kev,
    )
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=num_samples,
        replacement=True,
    )


def _read_dataset_metadata(
    dataset: MajoranaWaveformDataset, target_label: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, energies) aligned 1:1 with ``dataset[i]``.

    When ``dataset._index_map`` is set (energy_range filter and/or
    subset_portion < 1.0 active), the per-file label / energy arrays are
    indexed via the map so the output reflects exactly the events the
    Dataset exposes — including the subset selection.
    """
    if dataset._index_map is None:
        labels_chunks: list[np.ndarray] = []
        energies_chunks: list[np.ndarray] = []
        for path in dataset.config.files:
            with h5py.File(path, "r") as f:
                labels_chunks.append(f[target_label][:].astype(bool))
                energies_chunks.append(f["energy_label"][:].astype(np.float64))
        return np.concatenate(labels_chunks), np.concatenate(energies_chunks)

    # Indexed path: read each file once, then gather using the index map.
    n = dataset._index_map.shape[0]
    labels = np.empty(n, dtype=bool)
    energies = np.empty(n, dtype=np.float64)
    for fi, path in enumerate(dataset.config.files):
        rows = dataset._index_map[:, 0] == fi
        if not rows.any():
            continue
        with h5py.File(path, "r") as f:
            file_labels = f[target_label][:].astype(bool)
            file_energies = f["energy_label"][:].astype(np.float64)
        local = dataset._index_map[rows, 1]
        labels[rows] = file_labels[local]
        energies[rows] = file_energies[local]
    return labels, energies


def _compute_sampler_weights(
    *,
    dataset: MajoranaWaveformDataset,
    strategies: list[SamplerStrategy],
    target_label: str,
    energy_bin_width_kev: int,
) -> np.ndarray:
    """Per-event weight vector = product of weights from each strategy."""
    labels, energies = _read_dataset_metadata(dataset, target_label)
    if labels.size != len(dataset):
        raise RuntimeError(
            f"metadata length {labels.size} does not match dataset length "
            f"{len(dataset)} — this should never happen"
        )

    weights = np.ones(labels.size, dtype=np.float64)

    if "class_balanced" in strategies:
        n_pos = int(labels.sum())
        n_neg = int((~labels).sum())
        if n_pos == 0 or n_neg == 0:
            raise ValueError(f"class_balanced needs both classes (n_pos={n_pos}, n_neg={n_neg})")
        weights *= np.where(labels, 1.0 / n_pos, 1.0 / n_neg)

    if "energy_balanced" in strategies:
        if energies.size == 0:
            raise ValueError("energy_balanced needs at least one event")
        # Bin edges from 0 up through max energy, in steps of bin_width.
        max_e = float(energies.max())
        edges = np.arange(0.0, max_e + energy_bin_width_kev, energy_bin_width_kev)
        bin_idx = np.clip(np.digitize(energies, edges) - 1, 0, len(edges) - 2)
        bin_counts = np.bincount(bin_idx, minlength=len(edges) - 1).astype(np.float64)
        weights *= 1.0 / np.maximum(bin_counts[bin_idx], 1.0)

    return weights


# ---- Run metadata ----------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _collect_runtime_info(device: torch.device) -> dict:
    """Collect machine / git / library info for the metadata.json snapshot."""
    info: dict = {
        "hostname": socket.gethostname(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_used": str(device),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        with contextlib.suppress(Exception):
            info["gpu_name"] = torch.cuda.get_device_name(0)
    info.update(_collect_git_info())
    return info


def _collect_git_info() -> dict:
    """Best-effort git SHA + dirty flag. Empty dict if not in a repo."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {}
    dirty = (
        subprocess.call(
            ["git", "diff", "--quiet"],
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
        != 0
    )
    return {"git_sha": sha, "git_dirty": dirty}


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


# ---- Logging plumbing ------------------------------------------------


def _attach_file_handler(logfile: Path) -> logging.FileHandler:
    """Attach a per-run FileHandler to the ``majorana_acp`` package logger."""
    package_logger = logging.getLogger("majorana_acp")
    if package_logger.level == logging.NOTSET:
        package_logger.setLevel(logging.INFO)

    handler = logging.FileHandler(logfile)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    package_logger.addHandler(handler)
    return handler


def _detach_file_handler(handler: logging.FileHandler) -> None:
    package_logger = logging.getLogger("majorana_acp")
    package_logger.removeHandler(handler)
    handler.close()
