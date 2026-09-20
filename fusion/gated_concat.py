"""Concatenation-preserving adaptive fusion modules.

Each module learns gates from all input streams but retains every stream in the
output.  This makes the modules suitable for tasks, such as rotation
estimation, where a weighted sum may discard useful modality-specific detail.
"""

import math

import torch
import torch.nn as nn

from . import register_fusion
from .base import BaseFusion


def _sigmoid_init_bias(gate_init_prob: float) -> float:
    """Convert a target initial sigmoid output into the bias/logit that produces it.

    gate_init_prob=0.5 (the default used everywhere below) reproduces the
    previous zero-bias init exactly, since sigmoid(0) == 0.5 -- so leaving
    gate_init_prob unset is fully backward compatible with every existing run.
    """
    if not 0.0 < gate_init_prob < 1.0:
        raise ValueError(f"gate_init_prob must be strictly between 0 and 1, got {gate_init_prob}")
    return math.log(gate_init_prob / (1.0 - gate_init_prob))


class _BaseGatedConcatFusion(BaseFusion):
    """Shared gate MLP and input handling for gated concatenation modules.

    `gate_hidden_dim` defaults to 32 -- the original width chosen (commit 5d1fd1a)
    for the *scalar* gate variants, where gate_outputs = num_streams (e.g. 2). The
    channel-wise variants (gate_outputs = num_streams * dim, e.g. 44800) inherited
    that same 32-unit width unchanged, which forces every one of the ~45k per-channel
    gate logits through a 32-dim bottleneck -- empirically this collapses ~80% of
    channels into two saturated clusters (near 0 / near 1) rather than learning
    independent per-channel trust (see q2_gating_granularity gate stats). The
    "_wide" channel-wise variants below widen this bottleneck to give channel-wise
    gating room to actually specialize per channel instead of per shared-latent-mode.
    """

    def __init__(self, dim: int, num_streams: int, gate_outputs: int, gate_hidden_dim: int = 32,
                 gate_init_prob: float = 0.5):
        super().__init__(dim, num_streams)
        self.gate_init_prob = gate_init_prob
        self.gate = nn.Sequential(
            nn.Linear(dim * num_streams, gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(gate_hidden_dim, gate_outputs),
        )

        # Begin with near-neutral gates while retaining gradients for learning.
        nn.init.uniform_(self.gate[3].weight, -1e-3, 1e-3)
        nn.init.constant_(self.gate[3].bias, _sigmoid_init_bias(gate_init_prob))

    def _gate_inputs(self, features_list):
        """Validate streams and return their joint representation for gating."""
        self._validate(features_list)
        return torch.cat(features_list, dim=1)


class _BaseFloorGatedConcatFusion(_BaseGatedConcatFusion):
    """Shared plain-gated fusion with the gate bounded away from zero.

    Plain (non-residual) gated-concat variants can drive a stream's sigmoid gate
    to ~0, which zeroes both that stream's contribution to the fused feature and
    the gradient flowing back into its backbone -- observed in practice as a
    training collapse on some seeds (val_abs_rot_x/val_abs_trans freezing within
    the first few epochs, e.g. q2_ScalarVsChannelwise seeds 48/50 for
    gated_concat and seed 49 for channelwise_gated_concat). Flooring the gate at
    MIN_GATE keeps a minimum pass-through/gradient path for every stream --
    unlike the residual_gated_concat variants, which additionally guarantee a
    pass-through of at least 1x -- while still allowing the gate to strongly
    (not fully) suppress a stream.
    """

    MIN_GATE = 0.1

    def _floored_sigmoid(self, logits):
        return self.MIN_GATE + (1.0 - self.MIN_GATE) * torch.sigmoid(logits)


@register_fusion("gated_concat")
class GatedConcatFusion(_BaseGatedConcatFusion):
    """Apply one independent sigmoid gate to each stream, then concatenate."""

    def __init__(self, dim: int, num_streams: int, gate_init_prob: float = 0.5):
        super().__init__(dim, num_streams, gate_outputs=num_streams, gate_init_prob=gate_init_prob)

    def forward(self, features_list):
        # Predict one gate for every stream from the combined multimodal input.
        combined = self._gate_inputs(features_list)
        weights = torch.sigmoid(self.gate(combined))  # (B, num_streams)

        # Preserve every feature vector instead of collapsing streams with a sum.
        weighted_features = [
            weights[:, stream_idx].unsqueeze(-1) * feature
            for stream_idx, feature in enumerate(features_list)
        ]
        out = torch.cat(weighted_features, dim=1)
        return out, weights


@register_fusion("scale_gated_concat")
class ScaleGatedConcatFusion(_BaseGatedConcatFusion):
    """gated_concat, but with one independent sigmoid gate per stream PER PYRAMID
    SCALE (P3/P4/P5) instead of a single scalar per stream -- a middle ground
    between gated_concat's one-gate-per-stream (structurally cannot distinguish
    P3 from P5 at all) and channelwise's one-gate-per-channel (can, but
    empirically doesn't: every channelwise run in the final_ablation gate
    diagnostics learned P3 ~= P4 ~= P5 gate means almost exactly, despite having
    the freedom to differ). This gives the gate exactly scale-level resolution,
    to test directly whether scale-specific trust is a real, useful signal that
    channelwise's per-channel freedom just happened not to find, or whether P3/P4/P5
    genuinely don't need to be treated differently for this task.

    Requires `scale_dims` (the {"p3": n, "p4": n, "p5": n} flattened-size dict
    computed once in PoseEstimationNet.__init__ as `stream_scale_dims`) so it
    knows where each stream's P3/P4/P5 blocks begin/end within the flattened
    `dim`-length feature vector it receives. build_fusion() forwards this
    automatically to any fusion class that declares a `scale_dims` constructor
    argument (same inspect.signature-based mechanism used for gate_init_prob),
    so every other fusion type is unaffected.
    """

    SCALE_ORDER = ("p3", "p4", "p5")

    def __init__(self, dim: int, num_streams: int, scale_dims: dict = None, gate_init_prob: float = 0.5):
        if not scale_dims:
            raise ValueError(
                "scale_gated_concat requires scale_dims (the P3/P4/P5 flattened-size "
                "dict from PoseEstimationNet.stream_scale_dims) -- it only makes sense "
                "for the 1D per-stream-vector pipeline, not standalone."
            )
        missing = [s for s in self.SCALE_ORDER if s not in scale_dims]
        if missing:
            raise ValueError(f"scale_dims is missing required key(s): {missing}")
        if sum(scale_dims[s] for s in self.SCALE_ORDER) != dim:
            raise ValueError(
                f"scale_dims {scale_dims} does not sum to dim={dim}; scale_gated_concat "
                "needs the exact per-scale split of the flattened per-stream feature vector."
            )
        super().__init__(dim, num_streams, gate_outputs=num_streams * len(self.SCALE_ORDER),
                          gate_init_prob=gate_init_prob)
        self.scale_dims = {s: int(scale_dims[s]) for s in self.SCALE_ORDER}
        # Precompute each scale's [start, end) slice within the flattened dim, in
        # the same P3-then-P4-then-P5 order forward_single_stream() concatenates.
        self._scale_slices = []
        offset = 0
        for s in self.SCALE_ORDER:
            size = self.scale_dims[s]
            self._scale_slices.append((offset, offset + size))
            offset += size

    def forward(self, features_list):
        combined = self._gate_inputs(features_list)
        weights = torch.sigmoid(
            self.gate(combined).reshape(-1, self.num_streams, len(self.SCALE_ORDER))
        )  # (B, num_streams, 3) -- one gate per stream per P3/P4/P5

        weighted_features = []
        for stream_idx, feature in enumerate(features_list):
            parts = [
                weights[:, stream_idx, scale_idx].unsqueeze(-1) * feature[:, start:end]
                for scale_idx, (start, end) in enumerate(self._scale_slices)
            ]
            weighted_features.append(torch.cat(parts, dim=1))

        out = torch.cat(weighted_features, dim=1)
        return out, weights


@register_fusion("residual_gated_concat")
class ResidualGatedConcatFusion(_BaseGatedConcatFusion):
    """Concatenate streams after residual scalar gating for each stream."""

    def __init__(self, dim: int, num_streams: int, gate_init_prob: float = 0.5):
        super().__init__(dim, num_streams, gate_outputs=num_streams, gate_init_prob=gate_init_prob)

    def forward(self, features_list):
        # Sigmoid gates are added to one, so each original stream is retained.
        combined = self._gate_inputs(features_list)
        weights = torch.sigmoid(self.gate(combined))  # (B, num_streams)

        weighted_features = [
            (1 + weights[:, stream_idx].unsqueeze(-1)) * feature
            for stream_idx, feature in enumerate(features_list)
        ]
        out = torch.cat(weighted_features, dim=1)
        return out, weights


@register_fusion("channelwise_gated_concat")
class ChannelWiseGatedConcatFusion(_BaseGatedConcatFusion):
    """Apply an independent sigmoid gate to every channel of every stream, then concatenate."""

    def __init__(self, dim: int, num_streams: int, gate_hidden_dim: int = 32, gate_init_prob: float = 0.5):
        super().__init__(dim, num_streams, gate_outputs=num_streams * dim, gate_hidden_dim=gate_hidden_dim,
                          gate_init_prob=gate_init_prob)

    def forward(self, features_list):
        # Reshape flat gate logits into one (num_streams, dim) gate map per item.
        combined = self._gate_inputs(features_list)
        weights = self.gate(combined).reshape(-1, self.num_streams, self.dim)
        weights = torch.sigmoid(weights)  # (B, num_streams, dim)

        weighted_features = [
            weights[:, stream_idx, :] * feature
            for stream_idx, feature in enumerate(features_list)
        ]
        out = torch.cat(weighted_features, dim=1)
        return out, weights


@register_fusion("channelwise_gated_concat_wide")
class ChannelWiseGatedConcatWideFusion(ChannelWiseGatedConcatFusion):
    """channelwise_gated_concat with the gate MLP's hidden bottleneck widened from
    32 to WIDE_HIDDEN_DIM, so ~44800 per-channel gate logits aren't all forced
    through the same 32-dim shared representation (see _BaseGatedConcatFusion
    docstring)."""

    WIDE_HIDDEN_DIM = 256

    def __init__(self, dim: int, num_streams: int, gate_init_prob: float = 0.5):
        super().__init__(dim, num_streams, gate_hidden_dim=self.WIDE_HIDDEN_DIM, gate_init_prob=gate_init_prob)


@register_fusion("gated_concat_floor")
class GatedConcatFloorFusion(_BaseFloorGatedConcatFusion):
    """gated_concat with the per-stream sigmoid gate floored at MIN_GATE."""

    def __init__(self, dim: int, num_streams: int, gate_init_prob: float = 0.5):
        super().__init__(dim, num_streams, gate_outputs=num_streams, gate_init_prob=gate_init_prob)

    def forward(self, features_list):
        combined = self._gate_inputs(features_list)
        weights = self._floored_sigmoid(self.gate(combined))  # (B, num_streams)

        weighted_features = [
            weights[:, stream_idx].unsqueeze(-1) * feature
            for stream_idx, feature in enumerate(features_list)
        ]
        out = torch.cat(weighted_features, dim=1)
        return out, weights


@register_fusion("channelwise_gated_concat_floor")
class ChannelWiseGatedConcatFloorFusion(_BaseFloorGatedConcatFusion):
    """channelwise_gated_concat with the per-channel sigmoid gate floored at MIN_GATE."""

    def __init__(self, dim: int, num_streams: int, gate_init_prob: float = 0.5):
        super().__init__(dim, num_streams, gate_outputs=num_streams * dim, gate_init_prob=gate_init_prob)

    def forward(self, features_list):
        combined = self._gate_inputs(features_list)
        weights = self.gate(combined).reshape(-1, self.num_streams, self.dim)
        weights = self._floored_sigmoid(weights)  # (B, num_streams, dim)

        weighted_features = [
            weights[:, stream_idx, :] * feature
            for stream_idx, feature in enumerate(features_list)
        ]
        out = torch.cat(weighted_features, dim=1)
        return out, weights


@register_fusion("channelwise_residual_gated_concat")
class ChannelWiseResidualGatedConcatFusion(_BaseGatedConcatFusion):
    """Concatenate streams after independently residual-gating every channel."""

    def __init__(self, dim: int, num_streams: int, gate_hidden_dim: int = 32, gate_init_prob: float = 0.5):
        super().__init__(dim, num_streams, gate_outputs=num_streams * dim, gate_hidden_dim=gate_hidden_dim,
                          gate_init_prob=gate_init_prob)

    def forward(self, features_list):
        # Reshape flat gate logits into one (num_streams, dim) gate map per item.
        combined = self._gate_inputs(features_list)
        weights = self.gate(combined).reshape(-1, self.num_streams, self.dim)
        weights = torch.sigmoid(weights)  # (B, num_streams, dim)

        # Residual channel-wise gates amplify without removing the base features.
        weighted_features = [
            (1 + weights[:, stream_idx, :]) * feature
            for stream_idx, feature in enumerate(features_list)
        ]
        out = torch.cat(weighted_features, dim=1)
        return out, weights


@register_fusion("channelwise_residual_gated_concat_wide")
class ChannelWiseResidualGatedConcatWideFusion(ChannelWiseResidualGatedConcatFusion):
    """channelwise_residual_gated_concat with the gate MLP's hidden bottleneck
    widened from 32 to WIDE_HIDDEN_DIM (see _BaseGatedConcatFusion docstring and
    ChannelWiseGatedConcatWideFusion)."""

    WIDE_HIDDEN_DIM = 256

    def __init__(self, dim: int, num_streams: int, gate_init_prob: float = 0.5):
        super().__init__(dim, num_streams, gate_hidden_dim=self.WIDE_HIDDEN_DIM, gate_init_prob=gate_init_prob)


class _PerChannelGatedConcatFusion(BaseFusion):
    """Per-channel gate computed by a tiny weight-shared function applied
    independently at every feature position (a 1x1 Conv1d over the `dim`
    "sequence" length), instead of _BaseGatedConcatFusion's single dense
    bottleneck shared by all `num_streams * dim` outputs.

    _BaseGatedConcatFusion (and its "_wide" variants) route every one of the
    ~44800 channel-gate decisions through the *same* small joint summary of the
    whole feature vector -- widening that summary gives it more room, but every
    channel's gate is still a function of the same shared numbers, so groups of
    channels can still be tied together and collapse in lockstep. Here, each
    channel gate is instead a function only of *that channel's own* per-stream
    values (optionally plus a small shared global-context vector, see
    `global_context_dim`), computed by one tiny function whose *weights* -- not
    its output -- are shared across all `dim` positions. Channels are then only
    coupled to each other through that tiny shared function's parameters, not
    through a shared finite-dimensional bottleneck, so the parameter count is
    independent of `dim`.

    Args:
        global_context_dim: 0 for a purely local gate (each channel sees only
            its own stream values). >0 adds a small global summary of the whole
            feature vector -- computed once per sample the same way
            _BaseGatedConcatFusion's bottleneck is, then broadcast to every
            position -- restoring cross-channel context at a parameter cost
            that's still independent of `dim` beyond that one summary layer.
        gate_init_prob: sigmoid gate's initial output before training (see
            _BaseGatedConcatFusion docstring); 0.5 (default) matches the
            previous zero-bias init exactly.
    """

    LOCAL_HIDDEN_DIM = 16

    def __init__(self, dim: int, num_streams: int, global_context_dim: int = 0, gate_init_prob: float = 0.5):
        super().__init__(dim, num_streams)
        self.global_context_dim = global_context_dim
        self.gate_init_prob = gate_init_prob

        if global_context_dim > 0:
            self.global_context = nn.Sequential(
                nn.Linear(dim * num_streams, global_context_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
            )

        per_position_in = num_streams + global_context_dim
        self.local_gate = nn.Sequential(
            nn.Conv1d(per_position_in, self.LOCAL_HIDDEN_DIM, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(self.LOCAL_HIDDEN_DIM, num_streams, kernel_size=1),
        )
        # Begin with near-neutral gates while retaining gradients for learning,
        # matching _BaseGatedConcatFusion's init.
        nn.init.uniform_(self.local_gate[2].weight, -1e-3, 1e-3)
        nn.init.constant_(self.local_gate[2].bias, _sigmoid_init_bias(gate_init_prob))

    def _gate_logits(self, features_list):
        """Returns per-channel gate logits, shape (B, num_streams, dim)."""
        self._validate(features_list)
        per_position = torch.stack(features_list, dim=1)  # (B, num_streams, dim)

        if self.global_context_dim > 0:
            combined = torch.cat(features_list, dim=1)  # (B, dim * num_streams)
            ctx = self.global_context(combined)  # (B, global_context_dim)
            ctx = ctx.unsqueeze(-1).expand(-1, -1, self.dim)  # (B, global_context_dim, dim)
            per_position = torch.cat([per_position, ctx], dim=1)  # (B, num_streams + ctx, dim)

        return self.local_gate(per_position)  # (B, num_streams, dim)


@register_fusion("channelwise_gated_concat_local")
class ChannelWiseGatedConcatLocalFusion(_PerChannelGatedConcatFusion):
    """channelwise_gated_concat with a purely local, weight-shared per-position
    gate (see _PerChannelGatedConcatFusion) instead of the shared dense
    bottleneck -- each channel's gate depends only on that channel's own
    RGB/tactile values, not on the rest of the feature vector."""

    def __init__(self, dim: int, num_streams: int, gate_init_prob: float = 0.5):
        super().__init__(dim, num_streams, global_context_dim=0, gate_init_prob=gate_init_prob)

    def forward(self, features_list):
        weights = torch.sigmoid(self._gate_logits(features_list))  # (B, num_streams, dim)
        weighted_features = [
            weights[:, stream_idx, :] * feature
            for stream_idx, feature in enumerate(features_list)
        ]
        out = torch.cat(weighted_features, dim=1)
        return out, weights


@register_fusion("channelwise_gated_concat_hybrid")
class ChannelWiseGatedConcatHybridFusion(_PerChannelGatedConcatFusion):
    """channelwise_gated_concat_local, plus a small shared global-context vector
    (summarizing the whole feature vector, same as _BaseGatedConcatFusion's
    bottleneck) broadcast into every position's otherwise-local gate decision."""

    GLOBAL_CONTEXT_DIM = 32

    def __init__(self, dim: int, num_streams: int, gate_init_prob: float = 0.5):
        super().__init__(dim, num_streams, global_context_dim=self.GLOBAL_CONTEXT_DIM, gate_init_prob=gate_init_prob)

    def forward(self, features_list):
        weights = torch.sigmoid(self._gate_logits(features_list))  # (B, num_streams, dim)
        weighted_features = [
            weights[:, stream_idx, :] * feature
            for stream_idx, feature in enumerate(features_list)
        ]
        out = torch.cat(weighted_features, dim=1)
        return out, weights


@register_fusion("channelwise_residual_gated_concat_local")
class ChannelWiseResidualGatedConcatLocalFusion(_PerChannelGatedConcatFusion):
    """channelwise_residual_gated_concat with a purely local, weight-shared
    per-position gate (see _PerChannelGatedConcatFusion)."""

    def __init__(self, dim: int, num_streams: int, gate_init_prob: float = 0.5):
        super().__init__(dim, num_streams, global_context_dim=0, gate_init_prob=gate_init_prob)

    def forward(self, features_list):
        weights = torch.sigmoid(self._gate_logits(features_list))  # (B, num_streams, dim)
        weighted_features = [
            (1 + weights[:, stream_idx, :]) * feature
            for stream_idx, feature in enumerate(features_list)
        ]
        out = torch.cat(weighted_features, dim=1)
        return out, weights


@register_fusion("channelwise_residual_gated_concat_hybrid")
class ChannelWiseResidualGatedConcatHybridFusion(_PerChannelGatedConcatFusion):
    """channelwise_residual_gated_concat_local, plus a small shared global-context
    vector broadcast into every position's gate decision (see
    ChannelWiseGatedConcatHybridFusion)."""

    GLOBAL_CONTEXT_DIM = 32

    def __init__(self, dim: int, num_streams: int, gate_init_prob: float = 0.5):
        super().__init__(dim, num_streams, global_context_dim=self.GLOBAL_CONTEXT_DIM, gate_init_prob=gate_init_prob)

    def forward(self, features_list):
        weights = torch.sigmoid(self._gate_logits(features_list))  # (B, num_streams, dim)
        weighted_features = [
            (1 + weights[:, stream_idx, :]) * feature
            for stream_idx, feature in enumerate(features_list)
        ]
        out = torch.cat(weighted_features, dim=1)
        return out, weights
