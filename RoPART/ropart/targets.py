"""Batched relative-target construction for training.

The single-image builders in ``helpers.targets`` are the reference; this is their
batched form, mirroring the dense ``[b, C, N, N]`` table that upstream
``engine.pretrain_model`` builds (v3 path), reduced to the channels we use:
``(Δx, Δy)`` for the baseline and ``+ (cos Δφ, sin Δφ)`` for RoPART.

Convention matches the helpers and upstream: index ``i`` is the reference patch,
``j`` the target, ``delta[..., i, j] = quantity_j - quantity_i``. Position is the
patch **centre** in the global frame.
"""

from __future__ import annotations

import torch


def build_targets(
    boxes: torch.Tensor,
    angles: torch.Tensor | None = None,
    *,
    rotates: bool = False,
    centered: bool = True,
    normalize_by: float | None = None,
) -> torch.Tensor:
    """Build the dense relative-target table over all ordered pairs.

    Args:
        boxes: ``(b, 4, N)`` integer boxes ``(x_s, y_s, x_e, y_e)``.
        angles: ``(b, N)`` per-patch orientations in radians (required if
            ``rotates``).
        rotates: append the ``(cos Δφ, sin Δφ)`` rotation channels.
        centered: use patch centres (default, the project convention) vs top-left
            corners for the translation channels.
        normalize_by: divide the translation deltas by this (e.g. ``patch_size``).

    Returns:
        ``(b, C, N, N)`` float tensor; ``C == 2`` (baseline) or ``4`` (rotation).
    """
    boxes = boxes.float()
    if centered:
        cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
        cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    else:
        cx, cy = boxes[:, 0], boxes[:, 1]
    pos = torch.stack((cx, cy), dim=1)  # (b, 2, N)

    # delta[b, c, i, j] = pos[b, c, j] - pos[b, c, i]
    delta = pos[:, :, None, :] - pos[:, :, :, None]  # (b, 2, N, N)
    if normalize_by is not None:
        delta = delta / float(normalize_by)

    if not rotates:
        return delta

    if angles is None:
        raise ValueError("rotates=True requires angles")
    angles = angles.float()
    dphi = angles[:, None, :] - angles[:, :, None]  # (b, N, N): [.,i,j] = φ_j - φ_i
    rot = torch.stack((torch.cos(dphi), torch.sin(dphi)), dim=1)  # (b, 2, N, N)
    return torch.cat((delta, rot), dim=1)  # (b, 4, N, N)


def gather_pairs(targets: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather the sampled ordered pairs from a dense ``[b, C, N, N]`` table.

    Args:
        targets: ``(b, C, N, N)`` dense table from :func:`build_targets`.
        indices: ``(2, num_pairs)`` reference/target patch indices (the model's
            sampled pairs); ``indices[0]`` = reference ``i``, ``indices[1]`` =
            target ``j``.

    Returns:
        ``(b, C, num_pairs)`` aligned with the model outputs.
    """
    return targets[:, :, indices[0], indices[1]]
