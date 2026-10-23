"""Cross-modal residual gated fusion modules.

The residual_gated_concat family computes (1 + g) * x, which only rescales a
stream by a factor in (1, 2); once the gates saturate this is a fixed per-channel
scale that the head absorbs, i.e. plain concatenation. Here the identity path is
the untouched stream and the gate controls how much of the *other* modality is
injected into it:

    out_i = x_i + g_i * P_(others -> i)(x_others)

A gate near 0 therefore means "stream i ignores the other modality", which the
head cannot undo with a fixed weight. Every cross-modal projection is
zero-initialised, so each module starts exactly as plain concatenation (or as the
anchor stream for `cross_gated`) and learns cross-modal mixing from there.

Gate weights are returned with the same shapes as the originals: (B, num_streams)
for scalar gates and (B, num_streams, dim) for channel-wise gates.
"""

import torch
import torch.nn as nn

from . import register_fusion
from .base import BaseFusion
from .gated_concat import _BaseGatedConcatFusion, _BaseSEChannelGatedConcat, _PerChannelGatedConcatFusion


class _CrossProjection(nn.Module):
    """Low-rank map from the other streams' concatenated features to one stream.

    A dense (num_others * dim -> dim) map would have billions of parameters at
    dim=44800, so it is factored through `rank` units. The up-projection is
    zero-initialised so the cross-modal term starts at exactly zero.
    """

    def __init__(self, dim: int, num_others: int, rank: int = 64):
        super().__init__()
        self.down = nn.Linear(dim * num_others, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, others):
        return self.up(self.down(torch.cat(others, dim=1)))


def _others(features_list, idx):
    return [f for j, f in enumerate(features_list) if j != idx]


class _CrossProjectionMixin:
    """Builds one low-rank cross-modal projection per receiving stream."""

    CROSS_RANK = 64

    def _build_projections(self, dim, num_streams):
        self.cross_proj = nn.ModuleList(
            [_CrossProjection(dim, num_streams - 1, self.CROSS_RANK) for _ in range(num_streams)]
        )

    def _cross_terms(self, features_list):
        return [self.cross_proj[i](_others(features_list, i)) for i in range(self.num_streams)]


@register_fusion("cross_gated")
class CrossGatedFusion(_CrossProjectionMixin, _BaseGatedConcatFusion):
    """Single-vector fusion: stream 0 is the anchor, the other streams enter as a
    sigmoid-gated correction. out = x_0 + g * P(x_1..). The returned weights are
    (B, num_streams); the anchor column is a constant 1."""

    def __init__(self, dim: int, num_streams: int, gate_init_prob: float = 0.5):
        super().__init__(dim, num_streams, gate_outputs=1, gate_init_prob=gate_init_prob)
        self.cross = _CrossProjection(dim, num_streams - 1, self.CROSS_RANK)

    def forward(self, features_list):
        combined = self._gate_inputs(features_list)
        g = torch.sigmoid(self.gate(combined))  # (B, 1)
        out = features_list[0] + g * self.cross(features_list[1:])
        weights = torch.cat([torch.ones_like(g), g.expand(-1, self.num_streams - 1)], dim=1)
        return out, weights


@register_fusion("cross_gated_concat")
class CrossGatedConcatFusion(_CrossProjectionMixin, _BaseGatedConcatFusion):
    """Concatenate streams, each enriched by a scalar-gated cross-modal term."""

    def __init__(self, dim: int, num_streams: int, gate_init_prob: float = 0.5):
        super().__init__(dim, num_streams, gate_outputs=num_streams, gate_init_prob=gate_init_prob)
        self._build_projections(dim, num_streams)

    def forward(self, features_list):
        combined = self._gate_inputs(features_list)
        weights = torch.sigmoid(self.gate(combined))  # (B, num_streams)
        cross = self._cross_terms(features_list)
        out = torch.cat(
            [f + weights[:, i].unsqueeze(-1) * c for i, (f, c) in enumerate(zip(features_list, cross))],
            dim=1,
        )
        return out, weights


@register_fusion("cross_channelwise_gated_concat")
class CrossChannelWiseGatedConcatFusion(_CrossProjectionMixin, _BaseGatedConcatFusion):
    """Concatenate streams, each enriched by a per-channel-gated cross-modal term."""

    def __init__(self, dim: int, num_streams: int, gate_hidden_dim: int = 32, gate_init_prob: float = 0.5):
        super().__init__(dim, num_streams, gate_outputs=num_streams * dim, gate_hidden_dim=gate_hidden_dim,
                         gate_init_prob=gate_init_prob)
        self._build_projections(dim, num_streams)

    def forward(self, features_list):
        combined = self._gate_inputs(features_list)
        weights = torch.sigmoid(self.gate(combined).reshape(-1, self.num_streams, self.dim))
        cross = self._cross_terms(features_list)
        out = torch.cat(
            [f + weights[:, i, :] * c for i, (f, c) in enumerate(zip(features_list, cross))],
            dim=1,
        )
        return out, weights


@register_fusion("cross_se_channel_gated_concat")
class CrossSEChannelGatedConcatFusion(_CrossProjectionMixin, _BaseSEChannelGatedConcat):
    """se_channel_gated_concat's squeeze-and-excitation gate (one gate per scale and
    channel, from pooled per-channel statistics of both streams), applied to the
    cross-modal term instead of rescaling the stream itself:
    out_i = x_i + g_i * P(x_others). Needs scale_dims / scale_channels like its parent."""

    def __init__(self, dim: int, num_streams: int, scale_dims: dict = None, scale_channels: dict = None,
                 gate_init_prob: float = 0.5):
        super().__init__(dim, num_streams, scale_dims=scale_dims, scale_channels=scale_channels,
                         gate_init_prob=gate_init_prob)
        self._build_projections(dim, num_streams)

    def forward(self, features_list):
        gates = self._channel_gates(features_list)
        weights = gates.index_select(2, self._expand_idx)  # (B, num_streams, dim)
        cross = self._cross_terms(features_list)
        out = torch.cat(
            [f + weights[:, i, :] * c for i, (f, c) in enumerate(zip(features_list, cross))],
            dim=1,
        )
        return out, weights


@register_fusion("cross_channelwise_gated_concat_local")
class CrossChannelWiseGatedConcatLocalFusion(_PerChannelGatedConcatFusion):
    """Position-wise version: the gate at position i depends only on that
    position's per-stream values (weight-shared Conv1d, see
    _PerChannelGatedConcatFusion) and the cross-modal term at position i is
    sum_j a[i_stream, j] * x_j[i] with learned scalars `a` (zero-init), so the
    parameter count is independent of `dim`."""

    def __init__(self, dim: int, num_streams: int, gate_init_prob: float = 0.5):
        super().__init__(dim, num_streams, global_context_dim=0, gate_init_prob=gate_init_prob)
        self.cross_scale = nn.Parameter(torch.zeros(num_streams, num_streams))

    def forward(self, features_list):
        weights = torch.sigmoid(self._gate_logits(features_list))  # (B, num_streams, dim)
        stacked = torch.stack(features_list, dim=1)  # (B, num_streams, dim)
        off_diag = self.cross_scale * (1 - torch.eye(self.num_streams, device=stacked.device))
        cross = torch.einsum("ij,bjd->bid", off_diag, stacked)
        out = (stacked + weights * cross).flatten(1)
        return out, weights
