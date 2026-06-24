"""Relative-target regression loss.

Plain MSE on the per-pair targets. This is valid for the rotation channels too
because they are the ``(cos Δφ, sin Δφ)`` encoding — smooth and wrap-free — so we
never regress a raw angle.

Channels are grouped: ``0:2`` translation ``(Δx, Δy)``, ``2:4`` rotation
``(cos Δφ, sin Δφ)`` when present. ``w_xy`` / ``w_phi`` weight the two groups so
loss balancing is a single knob; with both at 1 the loss is the sum of the two
per-group means.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RelativeMSE(nn.Module):
    def __init__(self, w_xy: float = 1.0, w_phi: float = 1.0):
        super().__init__()
        self.w_xy = w_xy
        self.w_phi = w_phi

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """``outputs``/``targets``: ``(b, C, num_pairs)`` with ``C`` in {2, 4}."""
        assert outputs.shape == targets.shape, (outputs.shape, targets.shape)
        se = (outputs - targets) ** 2
        loss = self.w_xy * se[:, 0:2].mean()
        if outputs.shape[1] >= 4:
            loss = loss + self.w_phi * se[:, 2:4].mean()
        return loss
