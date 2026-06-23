"""Off-grid patch sampling for a single image.

A pared-down re-implementation of the upstream PART ``v3`` sampling path
(``PART/make_dataset.py``), keeping only what RoPART needs for the
translation-only baseline:

* **off-grid sampling** — patch positions are continuous (any pixel offset),
  not snapped to a ``patch_size`` grid;
* **constant patch size** — every patch is ``patch_size x patch_size``. This
  drops the variable width/height that upstream ``v3`` uses, i.e. it removes the
  scale/aspect ``(Δw, Δh)`` signal, which is exactly the baseline geometry
  (no scale channel; scale/aspect is deferred to the last build step).

Compared with upstream we deliberately drop: the precomputed candidate-box bank
(an efficiency trick for a full dataset — irrelevant for one image), the grid
variants (``v1``/``v2``/``v3_1``), ``skimage`` resizing (constant size needs no
resize), and all the dataset/dataloader machinery.

Everything is plain ``float32`` and device-agnostic so the same code runs under
MPS locally and CUDA on the cluster.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image

# Box rows, matching upstream's target layout: (x_start, y_start, x_end, y_end).
X_S, Y_S, X_E, Y_E = 0, 1, 2, 3


def load_image(path: str, img_size: int | None = None) -> np.ndarray:
    """Load an image as an ``(H, W, C)`` uint8 array, optionally square-resized.

    Args:
        path: path to the image file.
        img_size: if given, resize to ``(img_size, img_size)``.

    Returns:
        ``(H, W, C)`` uint8 array in RGB.
    """
    img = Image.open(path).convert("RGB")
    if img_size is not None:
        img = img.resize((img_size, img_size), Image.BILINEAR)
    return np.asarray(img)


def sample_offgrid_patches(
    img_size: int,
    patch_size: int,
    num_patches: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample ``num_patches`` off-grid, constant-size patch boxes.

    Each patch is ``patch_size x patch_size`` with its top-left corner drawn
    uniformly (and independently per axis) from the continuous range of valid
    pixel offsets ``[0, img_size - patch_size]``. Patches may overlap — that is
    expected for off-grid sampling.

    Upstream (``make_dataset.py::VectorizedCIFAR``, ``v3``) instead samples four
    independent coordinates and takes ``(min, max)``, giving *variable*-size
    boxes filtered to a width/height band. Fixing the size here is the only
    structural change.

    Args:
        img_size: side length of the (square) source image, in pixels.
        patch_size: side length of every patch, in pixels.
        num_patches: number of patches to sample.
        generator: optional ``torch.Generator`` for reproducible sampling.

    Returns:
        Integer tensor of shape ``(4, num_patches)`` with rows
        ``(x_start, y_start, x_end, y_end)``.
    """
    if patch_size > img_size:
        raise ValueError(f"patch_size ({patch_size}) must be <= img_size ({img_size})")

    high = img_size - patch_size + 1  # randint's high is exclusive
    x_s = torch.randint(0, high, (num_patches,), generator=generator)
    y_s = torch.randint(0, high, (num_patches,), generator=generator)
    x_e = x_s + patch_size
    y_e = y_s + patch_size
    return torch.stack((x_s, y_s, x_e, y_e), dim=0)  # [4, num_patches]


def crop_patches(image: np.ndarray, boxes: torch.Tensor) -> torch.Tensor:
    """Crop each box out of the image as a ``float32`` patch in ``[0, 1]``.

    This is the constant-size simplification of upstream
    ``patchify_and_resize``: because every box is already ``patch_size`` square,
    no resize is needed (upstream resizes only because its ``v3`` boxes vary in
    size).

    Args:
        image: ``(H, W, C)`` uint8 array.
        boxes: ``(4, num_patches)`` integer box tensor from
            :func:`sample_offgrid_patches`.

    Returns:
        ``float32`` tensor of shape ``(num_patches, C, patch_h, patch_w)`` in
        ``[0, 1]``.
    """
    img = torch.from_numpy(image.copy()).float().div(255.0)  # (H, W, C)
    img = img.permute(2, 0, 1)  # (C, H, W)

    patches = []
    for x_s, y_s, x_e, y_e in boxes.T.tolist():
        patches.append(img[:, y_s:y_e, x_s:x_e])
    return torch.stack(patches, dim=0)  # (num_patches, C, ph, pw)


def retile_patches(patches: torch.Tensor, img_size: int) -> torch.Tensor:
    """Pack patches row-major into a single ``(C, img_size, img_size)`` image.

    This mirrors the reshape/permute in upstream
    ``VectorizedCIFAR.__getitem__``: the sampled patches are laid out in a
    ``sqrt(num_patches)`` grid so a standard ViT ``patch_embed`` conv turns them
    straight back into tokens.

    The result is a *packing*, not a reconstruction — patch order has nothing to
    do with where each patch was sampled from. Visually it looks scrambled, which
    is the whole point: the network is given no positional information.

    Args:
        patches: ``(num_patches, C, patch_size, patch_size)`` tensor. ``patch_size``
            must equal ``img_size / sqrt(num_patches)`` and ``num_patches`` must be
            a perfect square.
        img_size: side length of the packed output image.

    Returns:
        ``(C, img_size, img_size)`` tensor.
    """
    num_patches, c, ph, pw = patches.shape
    patches_per_width = int(math.sqrt(num_patches))
    if patches_per_width**2 != num_patches:
        raise ValueError(f"num_patches ({num_patches}) must be a perfect square")
    if patches_per_width * ph != img_size:
        raise ValueError(
            f"patch_size ({ph}) * sqrt(num_patches) ({patches_per_width}) "
            f"!= img_size ({img_size})"
        )

    # (num_patches, C, ph, pw) -> (C, num_patches, ph, pw)
    out = patches.permute(1, 0, 2, 3)
    # -> (C, ppw, ppw, ph, pw) -> (C, ppw, ph, ppw, pw) -> (C, img, img)
    out = out.reshape(c, patches_per_width, patches_per_width, ph, pw)
    out = out.permute(0, 1, 3, 2, 4).reshape(c, img_size, img_size)
    return out
