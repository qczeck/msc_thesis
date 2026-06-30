"""Relative-target regression loss.

Plain (optionally scale-balanced) MSE on the per-pair targets. The ``(cos Δφ, sin
Δφ)`` rotation encoding is smooth and wrap-free, so MSE on it is valid; we never
regress a raw angle.

Channels are grouped: ``0:2`` translation ``(Δx, Δy)``, ``2:4`` rotation
``(cos Δφ, sin Δφ)`` when present. ``w_xy`` / ``w_phi`` weight the two groups.

**Why balancing matters.** Even with the translation deltas normalised by patch
size, ``Var(Δx, Δy)`` is order tens while the unit-circle ``(cos, sin)`` group is
bounded with variance ≤ 0.5 — and bounding the rotation range (e.g. ±30°) shrinks it
further. With ``balance="none"`` and ``w_xy = w_phi = 1`` the translation term is
~20–30× larger, so ``Δφ`` gets almost no gradient and collapses to the mean
predictor. ``balance="variance"`` divides each channel's squared error by that
channel's **detached batch target variance** (its mean-predictor floor), so every
group is measured as a *fraction of its own floor* — both O(1) regardless of scale
or rotation range, and ``w_xy``/``w_phi`` become genuine relative-importance knobs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RelativeMSE(nn.Module):
    """Grouped MSE over the relative-target channels.

    Args:
        w_xy: weight on the translation group ``(Δx, Δy)``.
        w_phi: weight on the rotation group ``(cos Δφ, sin Δφ)``.
        balance: ``"none"`` for raw grouped MSE (the default — keeps the
            translation-only baseline byte-for-byte unchanged); ``"variance"`` to
            normalise each channel by its detached batch target variance so the two
            groups are on a comparable scale.
    """

    def __init__(self, w_xy: float = 1.0, w_phi: float = 1.0, balance: str = "none"):
        super().__init__()
        assert balance in ("none", "variance"), balance
        self.w_xy = w_xy
        self.w_phi = w_phi
        self.balance = balance

    def _group(self, se: torch.Tensor, targets: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
        """Mean loss over channels ``lo:hi``; fraction-of-floor when balancing."""
        if self.balance == "none":
            return se[:, lo:hi].mean()
        eps = 1e-6
        mse_c = se[:, lo:hi].mean(dim=(0, 2))  # (g,) per-channel mean SE
        var_c = targets[:, lo:hi].var(dim=(0, 2), unbiased=False)  # (g,) mean-pred floor
        return (mse_c / var_c.detach().clamp_min(eps)).mean()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """``outputs``/``targets``: ``(b, C, num_pairs)`` with ``C`` in {2, 4}."""
        assert outputs.shape == targets.shape, (outputs.shape, targets.shape)
        se = (outputs - targets) ** 2
        loss = self.w_xy * self._group(se, targets, 0, 2)
        if outputs.shape[1] >= 4:
            loss = loss + self.w_phi * self._group(se, targets, 2, 4)
        return loss
