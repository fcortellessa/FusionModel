"""
Registry for fusion methods.
"""

import inspect

from .concat import ConcatFusion
from .gated import ARGatePlusFusion, SimpleGatedFusion, GatedSoftmaxFusion, GatedSigmoidFusion
from .attention import AttentionFusion
from .spatial_gated import SpatialGatedFusion2D
from .gated_cross_attention import GatedCrossAttentionFusion, SEGatedCrossAttentionFusion
from .gated_cross_attention_2d import (
    GatedCrossAttention2DFusion,
    SEGatedCrossAttention2DMapFusion,
    SEGatedCrossAttention2DNoTactilePosFusion,
    SEGatedCrossAttention2DPoolFusion,
)
from .gated_concat import (
    ChannelWiseGatedConcatFloorFusion,
    ChannelWiseGatedConcatFusion,
    ChannelWiseGatedConcatHybridFusion,
    ChannelWiseGatedConcatLocalFusion,
    ChannelWiseGatedConcatWideFusion,
    ChannelWiseResidualGatedConcatFusion,
    ChannelWiseResidualGatedConcatHybridFusion,
    ChannelWiseResidualGatedConcatLocalFusion,
    ChannelWiseResidualGatedConcatWideFusion,
    GatedConcatFloorFusion,
    GatedConcatFusion,
    ResidualGatedConcatFusion,
    ScaleGatedConcatFusion,
    SEChannelGatedConcatFusion,
    SEChannelGatedConcatRenormFusion,
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
    "gated_cross_attention": GatedCrossAttentionFusion,
    "se_gated_cross_attention": SEGatedCrossAttentionFusion,
    "se_gated_cross_attention_2d": SEGatedCrossAttention2DMapFusion,
    "gated_cross_attention_2d": GatedCrossAttention2DFusion,
    "se_gated_cross_attention_2d_pool": SEGatedCrossAttention2DPoolFusion,
    "se_gated_cross_attention_2d_notacpos_onlyRGBposenc": SEGatedCrossAttention2DNoTactilePosFusion,
    "gated_concat": GatedConcatFusion,
    "scale_gated_concat": ScaleGatedConcatFusion,
    "se_channel_gated_concat": SEChannelGatedConcatFusion,
    "se_channel_gated_concat_renorm": SEChannelGatedConcatRenormFusion,
    "channelwise_gated_concat": ChannelWiseGatedConcatFusion,
    # "channelwise_gated_concat" is a legacy misnomer (it's actually an element-wise
    # gate, not per-channel -- see CLAUDE.md). The key is frozen for backward
    # compatibility with existing configs/checkpoints; use "elementwise_gated_concat"
    # for new configs instead.
    "elementwise_gated_concat": ChannelWiseGatedConcatFusion,
    "channelwise_gated_concat_wide": ChannelWiseGatedConcatWideFusion,
    "channelwise_gated_concat_local": ChannelWiseGatedConcatLocalFusion,
    # Same legacy-misnomer situation as "channelwise_gated_concat" above -- frozen
    # for backward compatibility; use "elementwise_gated_concat_local" for new configs.
    "elementwise_gated_concat_local": ChannelWiseGatedConcatLocalFusion,
    "channelwise_gated_concat_hybrid": ChannelWiseGatedConcatHybridFusion,
    "residual_gated_concat": ResidualGatedConcatFusion,
    "channelwise_residual_gated_concat": ChannelWiseResidualGatedConcatFusion,
    "channelwise_residual_gated_concat_wide": ChannelWiseResidualGatedConcatWideFusion,
    "channelwise_residual_gated_concat_local": ChannelWiseResidualGatedConcatLocalFusion,
    "elementwise_residual_gated_concat_local": ChannelWiseResidualGatedConcatLocalFusion,
    "channelwise_residual_gated_concat_hybrid": ChannelWiseResidualGatedConcatHybridFusion,
    "gated_concat_floor": GatedConcatFloorFusion,
    "channelwise_gated_concat_floor": ChannelWiseGatedConcatFloorFusion,
}

def build_fusion(name, dim, num_streams, gate_init_prob=None, scale_dims=None, scale_channels=None):
    # dim is the number of channels in each stream, in the case of spatial fusion, it is the number of channels per stream in the feature maps
    # num_streams is the number of streams to fuse
    # gate_init_prob: optional initial sigmoid gate output (see fusion/gated_concat.py's
    # _sigmoid_init_bias). Only the gated_concat family accepts this kwarg -- forwarded
    # only when the target class declares it, so every other fusion type (concat,
    # attention, the softmax/sigmoid weighted-sum "gated" family, spatial_gated_2d)
    # is unaffected and this stays backward compatible by default (None -> each
    # accepting class's own default of 0.5, identical to the previous zero-bias init).
    # scale_dims: optional {"p3": n, "p4": n, "p5": n} flattened-size dict (see
    # PoseEstimationNet.stream_scale_dims). Only scale_gated_concat currently
    # declares this kwarg; forwarded the same way as gate_init_prob, so every
    # other fusion type is unaffected.
    # scale_channels: optional {"p3": C, "p4": C, "p5": C} channel counts per pyramid scale
    # (PoseEstimationNet.stream_scale_channels). Only the se_channel_* fusions declare it
    # (they need the channel/grid split of each scale's flattened block); forwarded the
    # same way, so every other fusion type is unaffected.

    if name not in FUSION_REGISTRY:
        raise ValueError(f"Unknown fusion: {name}")
    fusion_cls = FUSION_REGISTRY[name]
    params = inspect.signature(fusion_cls.__init__).parameters
    kwargs = {}
    if gate_init_prob is not None and "gate_init_prob" in params:
        kwargs["gate_init_prob"] = gate_init_prob
    if scale_dims is not None and "scale_dims" in params:
        kwargs["scale_dims"] = scale_dims
    if scale_channels is not None and "scale_channels" in params:
        kwargs["scale_channels"] = scale_channels
    return fusion_cls(dim, num_streams, **kwargs)
