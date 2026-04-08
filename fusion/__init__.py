
"""
Fusion module registry and factory.

Usage:
    from fusion import build_fusion

    fusion = build_fusion("gated", dim=256, num_streams=2)
"""

from typing import Dict, Type
import torch.nn as nn

# Import base class
from .base import BaseFusion

# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

FUSION_REGISTRY: Dict[str, Type[BaseFusion]] = {}


def register_fusion(name: str):
    """
    Decorator to register a fusion module.

    Example:
        @register_fusion("concat")
        class ConcatFusion(BaseFusion):
            ...
    """
    def decorator(cls: Type[BaseFusion]):
        if name in FUSION_REGISTRY:
            raise ValueError(f"Fusion '{name}' is already registered.")

        if not issubclass(cls, BaseFusion):
            raise TypeError(f"{cls.__name__} must inherit from BaseFusion.")

        FUSION_REGISTRY[name] = cls
        return cls

    return decorator


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

def build_fusion(name: str, **kwargs) -> BaseFusion:
    """
    Factory function to build a fusion module.

    Args:
        name (str): Name of the fusion module.
        **kwargs: Arguments passed to the fusion class.

    Returns:
        BaseFusion: Instantiated fusion module.
    """
    if name not in FUSION_REGISTRY:
        available = ", ".join(FUSION_REGISTRY.keys())
        raise ValueError(
            f"Unknown fusion '{name}'. Available fusions: [{available}]"
        )

    fusion_cls = FUSION_REGISTRY[name]
    return fusion_cls(**kwargs)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def list_fusions():
    """Return a list of available fusion module names."""
    return list(FUSION_REGISTRY.keys())


# -----------------------------------------------------------------------------
# Auto-import modules to trigger registration
# -----------------------------------------------------------------------------

# IMPORTANT: these imports must come AFTER the registry definition
from . import concat  # noqa: F401
from . import gated   # noqa: F401
