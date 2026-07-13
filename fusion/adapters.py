"""
Task-specific feature adaptation modules applied AFTER fusion and BEFORE task heads.

This file implements two reusable nn.Modules that can be selected via config:
 - task_specific_residual_gate: Residual gated MLP producing a per-channel gate
 - task_specific_channel_attention: SE-style channel attention block

Both modules:
 - Accept a fused 1D feature tensor of shape [B, C]
 - Produce adapted features with residual modulation: out = x * (1 + gate)
 - Return both the adapted tensor and the gate/attention activations for logging

Notes/Requirements:
 - Keep concat fusion unchanged; these adapters run after fusion only.
 - No backbone changes.
 - Backward compatibility preserved: if adapter is not instantiated/used,
   model behavior is unchanged.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class TaskSpecificResidualGate(nn.Module):
    """Residual Gated Concat adapter for a single task.

    Architecture (per task):
        Linear(concat_dim, 32)
        ReLU
        Dropout(0.3)
        Linear(32, concat_dim)   # last layer initialized with very small weights and zero bias
        Sigmoid

    Forward computes a residual gating vector g in [0, 1] with shape [B, C]
    and returns adapted features: x * (1 + g).

    Returns a tuple: (adapted_features, gate_activations)
    """

    def __init__(self, concat_dim: int) -> None:
        super().__init__()
        self.concat_dim = int(concat_dim)

        self.net = nn.Sequential(
            nn.Linear(self.concat_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(32, self.concat_dim),
            nn.Sigmoid(),
        )

        # Initialize the last Linear with tiny weights (±1e-3) and zero bias
        # to start close to identity: out ≈ x * (1 + ~0)
        last_linear = self.net[3]
        assert isinstance(last_linear, nn.Linear)
        nn.init.uniform_(last_linear.weight, a=-1e-3, b=1e-3)
        nn.init.constant_(last_linear.bias, 0.0)

        # Store latest activations for external logging (TensorBoard)
        self.last_gate: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply residual gating.

        Args:
            x: Tensor of shape [B, C] (concatenated fused features)

        Returns:
            adapted: Tensor of shape [B, C]
            gate:    Tensor of shape [B, C] (values in [0, 1])
        """
        gate = self.net(x)
        adapted = x * (1.0 + gate)
        self.last_gate = gate.detach()
        return adapted, gate


class TaskSpecificChannelAttention(nn.Module):
    """SE-style channel attention adapter for a single task.

    Architecture (per task):
        Linear(concat_dim, concat_dim // 4)
        ReLU
        Linear(concat_dim // 4, concat_dim)
        Sigmoid

    Forward computes an attention vector a in [0, 1] with shape [B, C]
    and returns adapted features: x * (1 + a).

    Returns a tuple: (adapted_features, attention_vector)
    """

    def __init__(self, concat_dim: int) -> None:
        super().__init__()
        self.concat_dim = int(concat_dim)
        hidden = max(1, self.concat_dim // 4)

        self.net = nn.Sequential(
            nn.Linear(self.concat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.concat_dim),
            nn.Sigmoid(),
        )

        # No special init required beyond defaults; residual keeps it stable.
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply residual channel attention.

        Args:
            x: Tensor of shape [B, C] (concatenated fused features)

        Returns:
            adapted:   Tensor of shape [B, C]
            attention: Tensor of shape [B, C] (values in [0, 1])
        """
        attn = self.net(x)
        adapted = x * (1.0 + attn)
        self.last_attention = attn.detach()
        return adapted, attn


class TaskFeatureAdapter(nn.Module):
    """Task-specific channel attention adapter for a fused feature vector.

    Architecture:
        Linear(feature_dim, feature_dim // 4)
        ReLU
        Dropout(0.3)
        Linear(feature_dim // 4, feature_dim)   # tiny init, zero bias
        Sigmoid

    Forward computes a channel attention vector in [0, 1] with shape [B, feature_dim]
    and returns adapted features: x * (1 + attention).

    Returns a tuple: (output, attention)
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        hidden = max(1, self.feature_dim // 4)

        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden, self.feature_dim),
            nn.Sigmoid(),
        )

        # Initialize the last Linear with tiny weights (±1e-3) and zero bias
        # so the module starts close to identity: output ≈ x * (1 + ~0)
        last_linear = self.attention[3]
        assert isinstance(last_linear, nn.Linear)
        nn.init.uniform_(last_linear.weight, a=-1e-3, b=1e-3)
        nn.init.zeros_(last_linear.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply task-specific channel attention.

        Args:
            x: Fused feature tensor of shape (B, feature_dim)

        Returns:
            output:    Tensor of shape (B, feature_dim)
            attention: Tensor of shape (B, feature_dim) (values in [0, 1])
        """
        attention = self.attention(x)
        output = x * (1 + attention)
        return output, attention
