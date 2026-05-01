"""End-to-end and unit tests for ``majorana_acp.training.trainer``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from majorana_acp.training.config import (
    DataConfig,
    ExperimentConfig,
    LossConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
)
from majorana_acp.training.trainer import (
    build_optimizer,
    resolve_device,
    set_seed,
    train,
)


def _smoke_config(
    data_dir: Path,
    out_dir: Path,
    *,
    epochs: int = 1,
    loss: LossConfig | None = None,
) -> ExperimentConfig:
    """Build a minimal ExperimentConfig for synthetic-data smoke tests."""
    return ExperimentConfig(
        name="smoke",
        data=DataConfig(
            data_dir=data_dir,
            train_file_indices=[0, 1, 2],
            target_label="psd_label_low_avse",
            batch_size=4,
            num_workers=0,
            baseline_samples=500,
        ),
        model=ModelConfig(
            name="simple_cnn",
            params={"channels": [4, 8], "kernel_size": 3, "dropout": 0.0},
        ),
        optim=OptimConfig(lr=1e-3),
        loss=loss or LossConfig(type="bce"),
        train=TrainConfig(
            out_dir=out_dir,
            epochs=epochs,
            device="cpu",
            amp=False,
            log_every_n_steps=1,
        ),
    )


# --- helpers ---------------------------------------------------------


def test_set_seed_is_reproducible() -> None:
    set_seed(42)
    a = torch.rand(8)
    set_seed(42)
    b = torch.rand(8)
    assert torch.equal(a, b)


def test_resolve_device_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")


def test_resolve_device_cuda_unavailable_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        resolve_device("cuda")


def test_build_optimizer_adamw() -> None:
    model = nn.Linear(4, 1)
    opt = build_optimizer(model, OptimConfig(optimizer="adamw", lr=1e-3, weight_decay=1e-4))
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(1e-4)


def test_build_optimizer_sgd_uses_momentum() -> None:
    model = nn.Linear(4, 1)
    opt = build_optimizer(model, OptimConfig(optimizer="sgd", lr=1e-2, momentum=0.85))
    assert isinstance(opt, torch.optim.SGD)
    assert opt.param_groups[0]["momentum"] == pytest.approx(0.85)


def test_build_optimizer_adam() -> None:
    model = nn.Linear(4, 1)
    opt = build_optimizer(model, OptimConfig(optimizer="adam"))
    assert isinstance(opt, torch.optim.Adam)


# --- train() end-to-end ---------------------------------------------


def test_train_runs_and_saves_checkpoint(tiny_train_dir: Path, tmp_path: Path) -> None:
    cfg = _smoke_config(tiny_train_dir, tmp_path / "run")
    out_dir = train(cfg)

    ckpts = sorted(out_dir.glob("epoch_*.pt"))
    assert len(ckpts) == 1
    assert ckpts[0].name == "epoch_001.pt"

    config_json = out_dir / "config.json"
    assert config_json.is_file()
    snapshot = json.loads(config_json.read_text())
    assert snapshot["name"] == "smoke"

    log_file = out_dir / "train.log"
    assert log_file.is_file()
    assert log_file.stat().st_size > 0


def test_train_checkpoint_is_loadable(tiny_train_dir: Path, tmp_path: Path) -> None:
    cfg = _smoke_config(tiny_train_dir, tmp_path / "run")
    out_dir = train(cfg)
    ckpt_path = out_dir / "epoch_001.pt"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert {"model_state", "optimizer_state", "epoch", "config"} <= ckpt.keys()
    assert ckpt["epoch"] == 1


def test_train_multiple_epochs_produces_numbered_checkpoints(
    tiny_train_dir: Path, tmp_path: Path
) -> None:
    cfg = _smoke_config(tiny_train_dir, tmp_path / "run", epochs=3)
    out_dir = train(cfg)

    ckpts = sorted(out_dir.glob("epoch_*.pt"))
    assert [c.name for c in ckpts] == [
        "epoch_001.pt",
        "epoch_002.pt",
        "epoch_003.pt",
    ]


def test_train_with_weighted_bce_auto_pos_weight(tiny_train_dir: Path, tmp_path: Path) -> None:
    cfg = _smoke_config(
        tiny_train_dir,
        tmp_path / "run",
        loss=LossConfig(type="weighted_bce", pos_weight="auto"),
    )
    train(cfg)


def test_train_with_balanced_sampler(tiny_train_dir: Path, tmp_path: Path) -> None:
    cfg = _smoke_config(
        tiny_train_dir,
        tmp_path / "run",
        loss=LossConfig(type="bce", balanced_sampler=True),
    )
    train(cfg)


def test_train_with_focal_loss(tiny_train_dir: Path, tmp_path: Path) -> None:
    cfg = _smoke_config(
        tiny_train_dir,
        tmp_path / "run",
        loss=LossConfig(type="focal", focal_gamma=2.0),
    )
    train(cfg)


def test_train_does_not_leak_log_handlers(tiny_train_dir: Path, tmp_path: Path) -> None:
    """Calling train() multiple times shouldn't accumulate file handlers."""
    import logging as _logging

    pkg_logger = _logging.getLogger("majorana_acp")
    before = len(pkg_logger.handlers)

    cfg = _smoke_config(tiny_train_dir, tmp_path / "run1")
    train(cfg)
    cfg = _smoke_config(tiny_train_dir, tmp_path / "run2")
    train(cfg)

    after = len(pkg_logger.handlers)
    assert after == before
