"""End-to-end training driver.

``train(cfg)`` is the single entry point. Given a fully validated
:class:`ExperimentConfig` it builds the dataset, model, loss, and
optimizer, runs the training loop, and saves a per-epoch checkpoint
plus a JSON snapshot of the config to ``cfg.train.out_dir``.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from majorana_acp.data import (
    DatasetConfig,
    MajoranaWaveformDataset,
    resolve_files,
)
from majorana_acp.models import build_model
from majorana_acp.training.config import (
    DeviceName,
    ExperimentConfig,
    OptimConfig,
)
from majorana_acp.training.loss import build_balanced_sampler, build_loss

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
    try:
        logger.info("experiment=%s  device=%s  out_dir=%s", cfg.name, device, out_dir)

        # ---- Data -------------------------------------------------
        train_files = resolve_files(cfg.data.data_dir, "train", cfg.data.train_file_indices)
        logger.info("loaded %d training file(s)", len(train_files))

        train_ds = MajoranaWaveformDataset(
            DatasetConfig(
                files=train_files,
                target_label=cfg.data.target_label,
                baseline_samples=cfg.data.baseline_samples,
            )
        )
        logger.info("dataset size: %d events", len(train_ds))

        sampler = None
        if cfg.loss.balanced_sampler:
            logger.info("building balanced sampler")
            sampler = build_balanced_sampler(train_files, cfg.data.target_label)

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.data.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=cfg.data.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

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

        # ---- Config snapshot for reproducibility ----------------
        (out_dir / "config.json").write_text(
            json.dumps(cfg.model_dump(mode="json"), indent=2, default=str)
        )

        # ---- Train loop ------------------------------------------
        for epoch in range(1, cfg.train.epochs + 1):
            avg_loss = _train_one_epoch(
                model=model,
                loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                epoch=epoch,
                log_every=cfg.train.log_every_n_steps,
            )
            ckpt_path = out_dir / f"epoch_{epoch:03d}.pt"
            save_checkpoint(ckpt_path, model, optimizer, epoch, cfg)
            logger.info(
                "epoch=%d  avg_loss=%.4f  checkpoint=%s",
                epoch,
                avg_loss,
                ckpt_path.name,
            )
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


# ---- Logging plumbing ------------------------------------------------


def _attach_file_handler(logfile: Path) -> logging.FileHandler:
    """Attach a per-run FileHandler to the ``majorana_acp`` package logger.

    The handler is removed (and the file closed) when the run finishes,
    so repeated calls to :func:`train` don't leak file descriptors.
    """
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
