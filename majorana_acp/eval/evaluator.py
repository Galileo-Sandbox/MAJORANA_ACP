"""Evaluation driver: load a checkpoint, score the test set, save metrics.

``evaluate(checkpoint, out_dir=None)`` is the single entry point. It
accepts either a ``.pt`` file or a run directory (in which case the
latest ``epoch_*.pt`` is used). Outputs are written to ``out_dir``,
defaulting to ``<checkpoint-parent>/eval/``:

* ``metrics.json``    — scalar metrics (counts, ROC-AUC, accuracy at 0.5, ...).
* ``predictions.h5``  — raw per-event scores, logits, labels and the
  auxiliary fields needed for stratified analysis (energy, tp0,
  detector, run_number, id).
* ``eval.log``        — log file scoped to this evaluation run.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn import metrics as skm
from torch.utils.data import DataLoader

from majorana_acp.data import (
    DatasetConfig,
    MajoranaWaveformDataset,
    resolve_files,
)
from majorana_acp.models import build_model
from majorana_acp.training.config import ExperimentConfig
from majorana_acp.training.trainer import resolve_device

logger = logging.getLogger(__name__)


# ---- public entry point --------------------------------------------


def evaluate(
    checkpoint: Path | str,
    out_dir: Path | str | None = None,
) -> Path:
    """Run evaluation end-to-end and return the output directory.

    Args:
        checkpoint: Path to a ``.pt`` file or to a run directory
            (the latest ``epoch_*.pt`` inside it is used).
        out_dir: Where to write outputs. Defaults to ``<ckpt-parent>/eval``.

    Returns:
        The output directory (a ``pathlib.Path``).
    """
    ckpt_path = _resolve_checkpoint_path(Path(checkpoint))
    out_dir = Path(out_dir) if out_dir is not None else ckpt_path.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    file_handler = _attach_file_handler(out_dir / "eval.log")
    try:
        logger.info("checkpoint=%s  out_dir=%s", ckpt_path, out_dir)

        model, cfg, ckpt_meta = load_checkpoint(ckpt_path)
        device = resolve_device(cfg.train.device)
        model = model.to(device).eval()
        logger.info(
            "loaded model=%s (epoch=%d) on device=%s",
            cfg.model.name,
            ckpt_meta["epoch"],
            device,
        )

        # ---- Build test dataset using the saved config ---------
        test_files = resolve_files(cfg.data.data_dir, "test", cfg.data.test_file_indices)
        logger.info("loaded %d test file(s)", len(test_files))

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
            )
        )
        logger.info("test set size: %d events", len(test_ds))

        loader = DataLoader(
            test_ds,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        # ---- Inference ----------------------------------------
        t0 = time.perf_counter()
        predictions = run_inference(model, loader, device)
        logger.info(
            "inference: %d events in %.1fs (%.0f events/s)",
            len(predictions["logit"]),
            time.perf_counter() - t0,
            len(predictions["logit"]) / max(time.perf_counter() - t0, 1e-9),
        )

        # ---- Save raw predictions ------------------------------
        save_predictions(out_dir / "predictions.h5", predictions)

        # ---- Compute and save scalar metrics ------------------
        scalar_metrics = compute_metrics(predictions)
        scalar_metrics["checkpoint"] = str(ckpt_path)
        scalar_metrics["epoch"] = ckpt_meta["epoch"]
        scalar_metrics["model_name"] = cfg.model.name
        (out_dir / "metrics.json").write_text(json.dumps(scalar_metrics, indent=2))

        for k, v in scalar_metrics.items():
            if isinstance(v, float):
                logger.info("metric  %s = %.4f", k, v)
            else:
                logger.info("metric  %s = %s", k, v)
    finally:
        _detach_file_handler(file_handler)

    return out_dir


# ---- helpers --------------------------------------------------------


def _resolve_checkpoint_path(path: Path) -> Path:
    """Accept either a ``.pt`` file or a run directory."""
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(path.glob("epoch_*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No epoch_*.pt files in {path}")
        return candidates[-1]
    raise FileNotFoundError(f"Checkpoint path does not exist: {path}")


def load_checkpoint(
    path: Path | str,
) -> tuple[torch.nn.Module, ExperimentConfig, dict]:
    """Load a checkpoint and rebuild the model with weights restored.

    Returns:
        (model, config, meta) where ``meta`` is a dict with ``epoch`` and
        any other non-state metadata stored in the checkpoint.
    """
    path = _resolve_checkpoint_path(Path(path))
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ExperimentConfig.model_validate(ckpt["config"])

    model = build_model(cfg.model.name, **cfg.model.params)
    model.load_state_dict(ckpt["model_state"])
    meta = {"epoch": int(ckpt["epoch"])}
    return model, cfg, meta


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Score every event in ``loader`` and return parallel arrays."""
    model.eval()
    chunks: dict[str, list[np.ndarray]] = {
        k: [] for k in ("logit", "label", "energy", "tp0", "detector", "run_number", "id")
    }
    for batch in loader:
        wf = batch["waveform"].to(device, non_blocking=True)
        logits = model(wf).cpu().numpy()
        chunks["logit"].append(logits)
        chunks["label"].append(batch["label"].numpy())
        chunks["energy"].append(batch["energy"].numpy())
        chunks["tp0"].append(batch["tp0"].numpy())
        chunks["detector"].append(batch["detector"].numpy())
        chunks["run_number"].append(batch["run_number"].numpy())
        chunks["id"].append(batch["id"].numpy())
    out = {k: np.concatenate(v) for k, v in chunks.items()}
    out["score"] = _sigmoid(out["logit"])
    return out


def compute_metrics(predictions: dict[str, np.ndarray]) -> dict:
    """Scalar classification metrics over the prediction arrays."""
    labels = predictions["label"].astype(bool)
    scores = predictions["score"]
    n = int(labels.size)
    n_pos = int(labels.sum())
    n_neg = n - n_pos

    out: dict = {
        "n_events": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "pass_rate": n_pos / n if n > 0 else 0.0,
        "accuracy_at_0.5": float(((scores > 0.5) == labels).mean()),
    }
    # ROC-AUC is undefined when only one class is present.
    if n_pos > 0 and n_neg > 0:
        out["roc_auc"] = float(skm.roc_auc_score(labels, scores))
    else:
        out["roc_auc"] = None
    return out


def save_predictions(path: Path, predictions: dict[str, np.ndarray]) -> None:
    """Write per-event arrays to an HDF5 file for downstream analysis."""
    with h5py.File(path, "w") as f:
        f.create_dataset("logit", data=predictions["logit"].astype(np.float32))
        f.create_dataset("score", data=predictions["score"].astype(np.float32))
        f.create_dataset("label", data=predictions["label"].astype(bool))
        f.create_dataset("energy", data=predictions["energy"].astype(np.float32))
        for k in ("tp0", "detector", "run_number", "id"):
            f.create_dataset(k, data=predictions[k].astype(np.int64))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ---- logging plumbing (mirrors trainer pattern) --------------------


def _attach_file_handler(logfile: Path) -> logging.FileHandler:
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
    logging.getLogger("majorana_acp").removeHandler(handler)
    handler.close()
