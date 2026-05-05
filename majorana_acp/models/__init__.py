"""Classifier model registry and concrete model implementations.

Importing this package side-effect-imports each concrete model module
so its ``@register_model`` decorator runs and the model becomes
discoverable via ``build_model(name)``.
"""

# Concrete model modules — imported for their @register_model side effect.
from majorana_acp.models import (
    mlp,  # noqa: F401
    resnet1d,  # noqa: F401
    simple_cnn,  # noqa: F401
)
from majorana_acp.models.registry import build_model, list_models, register_model

__all__ = ["build_model", "list_models", "register_model"]
