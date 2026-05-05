"""Pydantic schemas for an end-to-end training experiment.

A single ``ExperimentConfig`` instance fully specifies one training run:
which data to load, which model architecture to instantiate, the
optimizer / loss / class-imbalance choices, and the training schedule.
Configs live as YAML under ``configs/`` and are loaded via
``load_config()``.

The model is referred to by registry key (``model.name``) plus a
free-form ``model.params`` dict, so swapping architectures is a one-line
YAML change once the new model is registered.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_yaml import parse_yaml_file_as

from majorana_acp.data import FileIndices, PSDLabel

OptimizerName = Literal["adam", "adamw", "sgd"]
LossType = Literal["bce", "weighted_bce", "focal"]
DeviceName = Literal["auto", "cuda", "cpu"]


class _Frozen(BaseModel):
    """Base for frozen, strict configs.

    ``extra="forbid"`` catches typos in YAML rather than silently
    discarding unknown keys.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class DataConfig(_Frozen):
    """Where to read the HDF5 files and how to feed them to the model."""

    data_dir: Path = Field(description="Directory containing the *.hdf5 files.")
    train_file_indices: FileIndices = Field(
        default="all",
        description='Train file indices, or "all".',
    )
    test_file_indices: FileIndices = Field(
        default="all",
        description='Test file indices used by the eval module, or "all".',
    )
    target_label: PSDLabel = Field(
        default="psd_label_low_avse",
        description="Which PSD label to predict.",
    )
    batch_size: int = Field(default=256, gt=0)
    num_workers: int = Field(default=4, ge=0)
    baseline_samples: int = Field(
        default=500,
        gt=0,
        description="Leading samples averaged for baseline subtraction.",
    )
    train_portion: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description=(
            "Fraction of the training set to draw per epoch (without "
            "replacement, reshuffled each epoch). Use < 1.0 to scan all "
            "files but spend less time per epoch."
        ),
    )
    align_t90: bool = Field(
        default=False,
        description=(
            "If True, the Dataset crops a fixed-length window around the "
            "90% rising-edge sample of each preprocessed waveform."
        ),
    )
    t90_pre: int = Field(
        default=200,
        gt=0,
        description="Samples before t90 to keep when align_t90=True.",
    )
    t90_post: int = Field(
        default=2000,
        gt=0,
        description="Samples after t90 to keep when align_t90=True.",
    )
    use_derivative_channel: bool = Field(
        default=False,
        description=(
            "If True, stack the first-difference (np.diff with leading "
            "zero pad) as a second channel. Output shape becomes (2, L). "
            "MLPs need input_dim = 2 * L; SimpleCNN needs in_channels = 2."
        ),
    )

    @field_validator("data_dir")
    @classmethod
    def _data_dir_must_exist(cls, v: Path) -> Path:
        if not v.is_dir():
            raise ValueError(f"data_dir not found: {v}")
        return v

    @field_validator("train_file_indices", "test_file_indices")
    @classmethod
    def _indices_clean(cls, v: FileIndices) -> FileIndices:
        if v == "all":
            return v
        if any(i < 0 for i in v):
            raise ValueError("file indices must be non-negative")
        if len(set(v)) != len(v):
            raise ValueError("file indices must be unique")
        return v


class ModelConfig(_Frozen):
    """Pluggable model spec: ``name`` + free-form keyword arguments.

    The trainer looks ``name`` up in the model registry and instantiates
    the resulting class with ``**params``. Swapping models is therefore
    a one-line YAML change (assuming the new model is registered).
    """

    name: str = Field(min_length=1, description="Registry key.")
    params: dict[str, Any] = Field(default_factory=dict)


class OptimConfig(_Frozen):
    optimizer: OptimizerName = "adamw"
    lr: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    momentum: float = Field(default=0.9, ge=0, lt=1, description="SGD only.")


class LossConfig(_Frozen):
    """Loss + class-imbalance handling.

    ``type`` selects the loss function. ``balanced_sampler`` is
    orthogonal — it can be combined with any loss to upsample the
    minority class via ``WeightedRandomSampler``.
    """

    type: LossType = "weighted_bce"
    pos_weight: Literal["auto"] | float = Field(
        default="auto",
        description='Used by weighted_bce. "auto" computes from training data.',
    )
    focal_gamma: float = Field(default=2.0, ge=0, description="Focal loss focusing parameter.")
    balanced_sampler: bool = Field(
        default=False,
        description="Upsample minority class via WeightedRandomSampler.",
    )

    @field_validator("pos_weight")
    @classmethod
    def _pos_weight_positive(cls, v: Literal["auto"] | float) -> Literal["auto"] | float:
        if isinstance(v, float) and v <= 0:
            raise ValueError("pos_weight must be > 0 or 'auto'")
        return v


class TrainConfig(_Frozen):
    epochs: int = Field(default=10, gt=0)
    seed: int = 0
    out_dir: Path = Field(description="Where checkpoints and logs are written.")
    log_every_n_steps: int = Field(default=50, gt=0)
    device: DeviceName = "auto"
    amp: bool = Field(default=True, description="Mixed-precision training.")


class ExperimentConfig(_Frozen):
    """Top-level config — one YAML, one training run."""

    name: str = Field(min_length=1, description="Human-readable experiment name.")
    data: DataConfig
    model: ModelConfig
    optim: OptimConfig = OptimConfig()
    loss: LossConfig = LossConfig()
    train: TrainConfig


def load_config(path: Path | str) -> ExperimentConfig:
    """Load and validate a YAML experiment config."""
    return parse_yaml_file_as(ExperimentConfig, Path(path))
