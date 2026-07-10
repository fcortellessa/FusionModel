"""
Task weighting strategies for balancing rotation and translation losses.
"""

from __future__ import annotations

from collections import deque

import torch
import torch.nn as nn


class DynamicWeightAverage(nn.Module):
    """
    Dynamic Weight Average (DWA) for two-task balancing.

    The module keeps the most recent epoch losses and updates the task weights
    from the loss ratio between the last two completed epochs:

        w_i(t) = 2 * softmax(r_i(t - 1) / T)

    where r_i(t - 1) = L_i(t - 1) / L_i(t - 2) and T is the temperature.

    The weights are applied to the next epoch, so the first two epochs use
    equal weights and the balance then adapts dynamically.
    """

    def __init__(self, temperature: float = 2.0, num_tasks: int = 2, eps: float = 1e-8):
        super().__init__()
        if num_tasks != 2:
            raise ValueError("DynamicWeightAverage currently supports exactly two losses.")

        self.temperature = float(temperature)
        self.num_tasks = int(num_tasks)
        self.eps = float(eps)

        self.register_buffer("task_weights", torch.ones(self.num_tasks, dtype=torch.float32))
        self._epoch_losses = deque(maxlen=2)
        self.register_buffer("latest_loss_ratios", torch.ones(self.num_tasks, dtype=torch.float32))

    def forward(self, loss_rot, loss_trans):
        """Combine rotation and translation losses using the current task weights."""
        weights = self.task_weights.to(device=loss_rot.device, dtype=loss_rot.dtype)
        return weights[0] * loss_rot + weights[1] * loss_trans

    @torch.no_grad()
    def update_epoch(self, loss_rot, loss_trans):
        """
        Update DWA weights from the completed epoch's average task losses.

        Call this once per epoch after collecting the epoch averages.
        """
        current_losses = torch.tensor(
            [float(loss_rot), float(loss_trans)],
            dtype=torch.float32,
            device=self.task_weights.device,
        )

        self._epoch_losses.append(current_losses)

        if len(self._epoch_losses) < 2:
            self.task_weights.copy_(torch.ones_like(self.task_weights))
            self.latest_loss_ratios.copy_(torch.ones_like(self.latest_loss_ratios))
            return self.task_weights

        previous_losses = self._epoch_losses[-2].clamp_min(self.eps)
        latest_losses = self._epoch_losses[-1].clamp_min(self.eps)
        loss_ratios = latest_losses / previous_losses

        self.latest_loss_ratios.copy_(loss_ratios)

        scaled = torch.softmax(loss_ratios / max(self.temperature, self.eps), dim=0)
        self.task_weights.copy_(self.num_tasks * scaled)
        return self.task_weights

    def get_weights(self):
        return self.task_weights

    def get_stats(self):
        weights = self.task_weights.detach().cpu()
        ratios = self.latest_loss_ratios.detach().cpu()
        return {
            "uncertainty_log_var_rot": "",
            "uncertainty_log_var_trans": "",
            "uncertainty_weight_rot": float(weights[0].item()),
            "uncertainty_weight_trans": float(weights[1].item()),
            "task_weighting_type": "dwa",
            "task_weighting_temperature": self.temperature,
            "task_weighting_loss_ratio_rot": float(ratios[0].item()),
            "task_weighting_loss_ratio_trans": float(ratios[1].item()),
        }