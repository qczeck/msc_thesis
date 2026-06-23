"""Visualisation helpers for off-grid patch sampling.

Small matplotlib wrappers to *see* what :mod:`helpers.sampling` produces: the
sampled boxes overlaid on the source image, the extracted patches as a grid, and
the row-major "packed" pseudo-image the ViT actually ingests.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle


def _to_hwc(image: np.ndarray | torch.Tensor) -> np.ndarray:
    """Coerce an image to an ``(H, W, C)`` array suitable for ``imshow``."""
    if isinstance(image, torch.Tensor):
        img = image.detach().cpu().float()
        if img.ndim == 3 and img.shape[0] in (1, 3):  # (C, H, W) -> (H, W, C)
            img = img.permute(1, 2, 0)
        img = img.numpy()
    else:
        img = image
    if img.dtype != np.uint8 and img.max() > 1.0:  # uint-ish floats
        img = img / 255.0
    return img


def show_image(image, ax=None, title: str | None = None, figsize=(5, 5)):
    """Show a single image (uint8 ``HWC`` or float ``CHW``/``HWC``)."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.imshow(_to_hwc(image))
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return ax


def draw_boxes(
    image,
    boxes: torch.Tensor,
    ax=None,
    title: str | None = None,
    figsize=(6, 6),
    number: bool = True,
    linewidth: float = 1.5,
):
    """Overlay sampled patch boxes on the source image.

    Args:
        image: source image (uint8 ``HWC``).
        boxes: ``(4, num_patches)`` tensor of ``(x_s, y_s, x_e, y_e)``.
        ax: optional existing axis.
        title: optional title.
        figsize: figure size when creating a new axis.
        number: annotate each box with its index.
        linewidth: rectangle edge width.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    show_image(image, ax=ax, title=title)

    boxes_t = boxes.T.tolist()
    cmap = plt.colormaps["hsv"]
    n = len(boxes_t)
    for i, (x_s, y_s, x_e, y_e) in enumerate(boxes_t):
        color = cmap(i / max(n, 1))
        ax.add_patch(
            Rectangle(
                (x_s, y_s),
                x_e - x_s,
                y_e - y_s,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
            )
        )
        if number:
            ax.text(
                x_s + 1,
                y_s + 1,
                str(i),
                color=color,
                fontsize=7,
                va="top",
                ha="left",
            )
    return ax


def show_patch_grid(
    patches: torch.Tensor,
    ncols: int | None = None,
    title: str | None = None,
    cell_size: float = 0.9,
    number: bool = True,
):
    """Show the extracted patches as a grid, in sampling order.

    Args:
        patches: ``(num_patches, C, ph, pw)`` tensor in ``[0, 1]``.
        ncols: columns in the grid (defaults to ``sqrt(num_patches)``).
        title: optional figure title.
        cell_size: size (inches) of each patch cell.
        number: annotate each cell with its index.
    """
    n = patches.shape[0]
    if ncols is None:
        ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * cell_size, nrows * cell_size))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.set_axis_off()
        if i < n:
            ax.imshow(_to_hwc(patches[i]))
            if number:
                ax.set_title(str(i), fontsize=6, pad=1)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def show_pair(
    image,
    boxes: torch.Tensor,
    i: int,
    j: int,
    patch_size: int | None = None,
    figsize=(13, 4.5),
):
    """Visualise one ordered patch pair and its relative translation.

    Three panels: the source image with the reference patch ``i`` (blue), the
    target patch ``j`` (red) and a translation arrow ``i -> j``; then the two
    cropped patches annotated with their centre coordinates.

    Args:
        image: source image (uint8 ``HWC``).
        boxes: ``(4, N)`` box tensor.
        i: reference patch index.
        j: target patch index.
        patch_size: if given, also report the delta normalised by this (the
            training target's units, patch widths).
        figsize: figure size.
    """
    from helpers.sampling import crop_patches
    from helpers.targets import patch_positions

    positions = patch_positions(boxes)  # patch centres (project default), matching the target
    pi = positions[:, i].tolist()
    pj = positions[:, j].tolist()
    dx_pix = pj[0] - pi[0]
    dy_pix = pj[1] - pi[1]

    pair = crop_patches(image, boxes[:, [i, j]])

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    show_image(image, ax=axes[0], title="reference i (blue) → target j (red)")
    for box, color, label in (
        (boxes[:, i], "tab:blue", f"i={i}"),
        (boxes[:, j], "tab:red", f"j={j}"),
    ):
        x_s, y_s, x_e, y_e = box.tolist()
        axes[0].add_patch(
            Rectangle((x_s, y_s), x_e - x_s, y_e - y_s, fill=False, edgecolor=color, linewidth=2)
        )
        axes[0].text(x_s + 1, y_s + 1, label, color=color, fontsize=9, va="top")
    # arrow between patch centres — the same vector the target encodes
    axes[0].annotate(
        "", xy=pj, xytext=pi, arrowprops=dict(arrowstyle="-|>", color="yellow", lw=2)
    )

    axes[1].imshow(_to_hwc(pair[0]))
    axes[1].set_axis_off()
    axes[1].set_title(f"reference i={i}\ncentre = ({pi[0]:.0f}, {pi[1]:.0f})")
    axes[2].imshow(_to_hwc(pair[1]))
    axes[2].set_axis_off()
    axes[2].set_title(f"target j={j}\ncentre = ({pj[0]:.0f}, {pj[1]:.0f})")

    suptitle = f"Δ(i→j) = (Δx={dx_pix:+.0f}, Δy={dy_pix:+.0f}) px"
    if patch_size:
        suptitle += (
            f"   |   normalised = "
            f"(Δx={dx_pix / patch_size:+.2f}, Δy={dy_pix / patch_size:+.2f}) patch-widths"
        )
    fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def show_sampling_overview(image, boxes, patches, retiled, figsize=(15, 5)):
    """One-row summary: boxes on image | retiled packing | (caller adds grid).

    Shows the two "whole image" views side by side — the sampled boxes on the
    original, and the row-major packed pseudo-image the ViT ingests — to make the
    scramble obvious. Use :func:`show_patch_grid` separately for the patch grid.
    """
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)
    draw_boxes(image, boxes, ax=ax0, title=f"{boxes.shape[1]} off-grid boxes on source")
    show_image(retiled, ax=ax1, title="row-major packed pseudo-image (ViT input)")
    fig.tight_layout()
    return fig
