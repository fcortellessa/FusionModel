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
