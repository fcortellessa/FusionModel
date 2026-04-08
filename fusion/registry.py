"""
Registry for fusion methods.
"""

from .concat import ConcatFusion
from .gated import GatedFusion, SimpleGatedFusion
from .attention import AttentionFusion

FUSION_REGISTRY = {
    "concat": ConcatFusion,
    "gated": GatedFusion,
    "simple_gated": SimpleGatedFusion,
    "attention": AttentionFusion,
}

def build_fusion(name, dim, num_streams):
    if name not in FUSION_REGISTRY:
        raise ValueError(f"Unknown fusion: {name}")
    return FUSION_REGISTRY[name](dim, num_streams)