"""Name-based model registry.

Each concrete model registers itself via the ``@register_model("name")``
decorator at import time. The trainer asks for a model by name and the
registry instantiates the right ``nn.Module`` with whatever ``params``
the experiment YAML supplies. Swapping models is therefore a one-line
YAML change once the new model file is imported by the package.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from torch import nn

_MODEL_T = TypeVar("_MODEL_T", bound=type[nn.Module])

_REGISTRY: dict[str, type[nn.Module]] = {}


def register_model(name: str) -> Callable[[_MODEL_T], _MODEL_T]:
    """Decorator: register a model class under ``name``.

    Raises ``ValueError`` if ``name`` is already taken so silent
    overwrites can't sneak in via duplicated decorator calls.
    """

    def decorator(cls: _MODEL_T) -> _MODEL_T:
        if name in _REGISTRY:
            raise ValueError(
                f"Model {name!r} is already registered (existing: "
                f"{_REGISTRY[name].__name__}, new: {cls.__name__})"
            )
        _REGISTRY[name] = cls
        return cls

    return decorator


def build_model(name: str, **params: Any) -> nn.Module:
    """Instantiate the model registered under ``name`` with ``params``."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available models: {list_models()}")
    return _REGISTRY[name](**params)


def list_models() -> list[str]:
    """Return registered model names in sorted order."""
    return sorted(_REGISTRY)
