"""Relative targets between patch pairs.

The pretext supervision is *not* where a patch is, but the **relative transform**
between ordered patch pairs ``(i, j)``. Upstream builds this inside the training
step (``PART/engine.py::pretrain_model``, ``v3`` branch) by differencing the
absolute patch coordinates over all ordered pairs into a ``[2, N, N]`` table.

Here we implement only the **translation-only baseline** target ``(Δx, Δy)`` —
the scale channels ``(Δw, Δh)`` are dropped (scale/aspect is deferred to the
last build step).

Convention: in the ``[2, N, N]`` table, the first index ``i`` is the
**reference** patch and the second index ``j`` is the **target** patch, and

    delta[c, i, j] = pos_j - pos_i        (c in {x, y})

so the diagonal is zero and the table is anti-symmetric
(``delta[c, i, j] == -delta[c, j, i]`` — "negative symmetry").

**Position convention (project default): centred, global frame.** ``pos`` is the
patch **centre**, differenced in **global / image axes**. The centre is
the rotation-invariant anchor, keeping translation decoupled from ``Δφ`` once
rotation is added. This matches upstream's ``--centered_targets`` path. The
corner-based variant (upstream's non-centred default) stays reachable via
``centered=False`` for the ablation; at ``φ = 0`` / constant patch size the two
are numerically identical.
"""

from __future__ import annotations

import torch

from helpers.sampling import X_E, X_S, Y_E, Y_S


def patch_positions(boxes: torch.Tensor, centered: bool = True) -> torch.Tensor:
    """Reference point of each patch used to build the translation target.

    Args:
        boxes: ``(4, N)`` tensor with rows ``(x_s, y_s, x_e, y_e)``.
        centered: if ``True`` (the project default),
            use the patch **centre** ``((x_s+x_e)/2, (y_s+y_e)/2)``; if ``False``,
            the **top-left corner** ``(x_s, y_s)`` (upstream's non-centred path).

    Returns:
        ``(2, N)`` float tensor with rows ``(cx, cy)`` when centred, else
        ``(x_s, y_s)``.
    """
    if centered:
        cx = (boxes[X_S] + boxes[X_E]).float() / 2.0
        cy = (boxes[Y_S] + boxes[Y_E]).float() / 2.0
        return torch.stack((cx, cy), dim=0)
    return boxes[[X_S, Y_S]].float()


def relative_translation(
    boxes: torch.Tensor,
    normalize_by: float | None = None,
    centered: bool = True,
) -> torch.Tensor:
    """Build the ``(Δx, Δy)`` relative-translation target over all ordered pairs.

    For every ordered pair ``(i, j)`` the target is ``pos_j - pos_i`` where
    ``pos`` is the patch reference point (see :func:`patch_positions`), expressed
    in the **global / image frame**.

    Args:
        boxes: ``(4, N)`` tensor with rows ``(x_s, y_s, x_e, y_e)``.
        normalize_by: if given (e.g. ``patch_size``), divide the deltas by this,
            putting them in units of patch widths. Upstream normalises by the
            reference patch's width/height, which is ``patch_size`` for every
            patch here.
        centered: position convention (default ``True`` = centres; see
            :func:`patch_positions`).

    Returns:
        ``(2, N, N)`` float tensor; ``[c, i, j] = pos_j - pos_i``.
    """
    positions = patch_positions(boxes, centered=centered)  # (2, N)
    # delta[c, i, j] = positions[c, j] - positions[c, i]
    delta = positions[:, None, :] - positions[:, :, None]  # (2, N, N)
    if normalize_by is not None:
        delta = delta / float(normalize_by)
    return delta


def check_translation_invariants(delta: torch.Tensor, atol: float = 1e-5) -> None:
    """Assert the structural invariants of a ``(Δx, Δy)`` target table.

    * zero diagonal — a patch has zero relative translation to itself;
    * negative symmetry — ``delta[c, i, j] == -delta[c, j, i]``.

    Args:
        delta: ``(2, N, N)`` relative-translation table.
        atol: absolute tolerance for the float comparisons.
    """
    n = delta.shape[-1]
    diag = torch.diagonal(delta, dim1=-2, dim2=-1)  # (2, N)
    assert torch.allclose(diag, torch.zeros_like(diag), atol=atol), "non-zero diagonal"
    assert torch.allclose(
        delta, -delta.transpose(-2, -1), atol=atol
    ), "translation table is not anti-symmetric (negative symmetry violated)"
    assert delta.shape == (2, n, n)


# --------------------------------------------------------------------------- #
# Relative orientation (RoPART rotation channel-group)
# --------------------------------------------------------------------------- #


def relative_orientation(angles: torch.Tensor) -> torch.Tensor:
    """Build the relative-orientation target ``(cos Δφ, sin Δφ)`` over ordered pairs.

    For every ordered pair ``(i, j)`` the relative orientation is
    ``Δφ = φ_j - φ_i`` (reference ``i``, target ``j`` — the same convention as
    :func:`relative_translation`), encoded as ``(cos Δφ, sin Δφ)``. The
    ``(cos, sin)`` encoding is smooth and free of the ``±π`` wrap, so plain MSE on
    it is a valid angular loss; a raw angle is never regressed directly. This is the RoPART rotation channel-group.

    Args:
        angles: ``(N,)`` per-patch orientations in radians.

    Returns:
        ``(2, N, N)`` float tensor: channel 0 ``cos(φ_j - φ_i)``, channel 1
        ``sin(φ_j - φ_i)``.
    """
    angles = angles.float()
    dphi = angles[None, :] - angles[:, None]  # (N, N): [i, j] = φ_j - φ_i
    return torch.stack((torch.cos(dphi), torch.sin(dphi)), dim=0)  # (2, N, N)


def check_orientation_invariants(rot: torch.Tensor, atol: float = 1e-5) -> None:
    """Assert the structural invariants of a ``(cos Δφ, sin Δφ)`` target table.

    * diagonal ``(cos, sin) = (1, 0)`` — zero relative orientation to itself;
    * ``cos`` **symmetric** — ``cos Δφ(i,j) == cos Δφ(j,i)`` (cosine is even);
    * ``sin`` **anti-symmetric** — ``sin Δφ(i,j) == -sin Δφ(j,i)``. This is the
      rotational negative symmetry, the analogue of PART's translational result
      (Figure 6) and one of RoPART's intrinsic evaluation axes.

    Args:
        rot: ``(2, N, N)`` relative-orientation table from :func:`relative_orientation`.
        atol: absolute tolerance for the float comparisons.
    """
    cos, sin = rot[0], rot[1]
    n = cos.shape[-1]
    assert rot.shape == (2, n, n)
    assert torch.allclose(torch.diagonal(cos), torch.ones(n), atol=atol), "cos diagonal != 1"
    assert torch.allclose(torch.diagonal(sin), torch.zeros(n), atol=atol), "sin diagonal != 0"
    assert torch.allclose(cos, cos.transpose(-2, -1), atol=atol), "cos not symmetric"
    assert torch.allclose(sin, -sin.transpose(-2, -1), atol=atol), "sin not anti-symmetric"
