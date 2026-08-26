"""
Registry for fusion methods.
"""

from .concat import ConcatFusion
from .gated import ARGatePlusFusion, SimpleGatedFusion, GatedSoftmaxFusion, GatedSigmoidFusion
from .attention import AttentionFusion
from .spatial_gated import SpatialGatedFusion2D
from .gated_concat import (
    ChannelWiseGatedConcatFloorFusion,
    ChannelWiseGatedConcatFusion,
    ChannelWiseResidualGatedConcatFusion,
    GatedConcatFloorFusion,
    GatedConcatFusion,
    ResidualGatedConcatFusion,
)

FUSION_REGISTRY = {
    "concat": ConcatFusion,
    "gated": GatedSoftmaxFusion,
    "simple_gated": SimpleGatedFusion,
    "gated_Softmax": GatedSoftmaxFusion,
    "gated_Sigmoid": GatedSigmoidFusion,
    "ARGate_Fusion": ARGatePlusFusion,
    "attention": AttentionFusion,
    "spatial_gated_2d": SpatialGatedFusion2D,
    "gated_concat": GatedConcatFusion,
    "channelwise_gated_concat": ChannelWiseGatedConcatFusion,
    "residual_gated_concat": ResidualGatedConcatFusion,
    "channelwise_residual_gated_concat": ChannelWiseResidualGatedConcatFusion,
    "gated_concat_floor": GatedConcatFloorFusion,
    "channelwise_gated_concat_floor": ChannelWiseGatedConcatFloorFusion,
}

def build_fusion(name, dim, num_streams):
    # dim is the number of channels in each stream, in the case of spatial fusion, it is the number of channels per stream in the feature maps 
    # num_streams is the number of streams to fuse
    
    if name not in FUSION_REGISTRY:
        raise ValueError(f"Unknown fusion: {name}")
    return FUSION_REGISTRY[name](dim, num_streams)
