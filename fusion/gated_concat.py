"""Concatenation-preserving adaptive fusion modules.

Each module learns gates from all input streams but retains every stream in the
output.  This makes the modules suitable for tasks, such as rotation
estimation, where a weighted sum may discard useful modality-specific detail.
"""

import torch
import torch.nn as nn

from . import register_fusion
from .base import BaseFusion


class _BaseGatedConcatFusion(BaseFusion):
    """Shared gate MLP and input handling for gated concatenation modules."""

    def __init__(self, dim: int, num_streams: int, gate_outputs: int):
        super().__init__(dim, num_streams)
        self.gate = nn.Sequential(
            nn.Linear(dim * num_streams, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, gate_outputs),
        )

        # Begin with near-neutral gates while retaining gradients for learning.
        nn.init.uniform_(self.gate[3].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.gate[3].bias)

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

    def __init__(self, dim: int, num_streams: int):
        super().__init__(dim, num_streams, gate_outputs=num_streams)

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


@register_fusion("residual_gated_concat")
class ResidualGatedConcatFusion(_BaseGatedConcatFusion):
    """Concatenate streams after residual scalar gating for each stream."""

    def __init__(self, dim: int, num_streams: int):
        super().__init__(dim, num_streams, gate_outputs=num_streams)

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

    def __init__(self, dim: int, num_streams: int):
        super().__init__(dim, num_streams, gate_outputs=num_streams * dim)

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


@register_fusion("gated_concat_floor")
class GatedConcatFloorFusion(_BaseFloorGatedConcatFusion):
    """gated_concat with the per-stream sigmoid gate floored at MIN_GATE."""

    def __init__(self, dim: int, num_streams: int):
        super().__init__(dim, num_streams, gate_outputs=num_streams)

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

    def __init__(self, dim: int, num_streams: int):
        super().__init__(dim, num_streams, gate_outputs=num_streams * dim)

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

    def __init__(self, dim: int, num_streams: int):
        super().__init__(dim, num_streams, gate_outputs=num_streams * dim)

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
