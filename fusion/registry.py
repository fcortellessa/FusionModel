"""
Registry for fusion methods.
"""

from .concat import ConcatFusion
from .gated import ARGatePlusFusion, SimpleGatedFusion, GatedSoftmaxFusion, GatedSigmoidFusion
from .attention import AttentionFusion

FUSION_REGISTRY = {
    "concat": ConcatFusion,
    "gated": GatedSoftmaxFusion,
    "simple_gated": SimpleGatedFusion,
    "gated_Softmax": GatedSoftmaxFusion,
    "gated_Sigmoid": GatedSigmoidFusion,
    "ARGate_Fusion": ARGatePlusFusion,
    "attention": AttentionFusion,
}

def build_fusion(name, dim, num_streams):
    if name not in FUSION_REGISTRY:
        raise ValueError(f"Unknown fusion: {name}")
    return FUSION_REGISTRY[name](dim, num_streams)