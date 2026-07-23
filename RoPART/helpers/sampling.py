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

Rotation (the RoPART extension) is **additive**: :func:`sample_rotation_angles`
and :func:`crop_patches_rotated` add a uniformly random per-patch orientation,
rotating pixels about the patch **centre** (the rotation-invariant anchor). The
baseline path (:func:`sample_offgrid_patches`
with the default ``margin=0``, plus :func:`crop_patches`) is unchanged when
rotation is unused.

Everything is plain ``float32`` and device-agnostic so the same code runs under
MPS locally and CUDA on the cluster.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import gaussian_blur as _tv_gaussian_blur

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
    margin: int = 0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample ``num_patches`` off-grid, constant-size patch boxes.

    Each patch is ``patch_size x patch_size`` with its top-left corner drawn
    uniformly (and independently per axis) from the continuous range of valid
    pixel offsets ``[margin, img_size - patch_size - margin]``. Patches may
    overlap — that is expected for off-grid sampling.

    Upstream (``make_dataset.py::VectorizedCIFAR``, ``v3``) instead samples four
    independent coordinates and takes ``(min, max)``, giving *variable*-size
    boxes filtered to a width/height band. Fixing the size here is the only
    structural change.

    Args:
        img_size: side length of the (square) source image, in pixels.
        patch_size: side length of every patch, in pixels.
        num_patches: number of patches to sample.
        margin: keep every patch's top-left corner at least this many pixels from
            the image edges. ``0`` (default) is the baseline behaviour; pass
            :func:`rotation_margin` when patches will be rotated, so the
            diagonal-sized source window stays in-image.
        generator: optional ``torch.Generator`` for reproducible sampling.

    Returns:
        Integer tensor of shape ``(4, num_patches)`` with rows
        ``(x_start, y_start, x_end, y_end)``.
    """
    if patch_size > img_size:
        raise ValueError(f"patch_size ({patch_size}) must be <= img_size ({img_size})")

    high = img_size - patch_size - margin + 1  # randint's high is exclusive
    if high <= margin:
        raise ValueError(
            f"no valid positions: img_size={img_size}, patch_size={patch_size}, "
            f"margin={margin} leaves an empty range"
        )
    x_s = torch.randint(margin, high, (num_patches,), generator=generator)
    y_s = torch.randint(margin, high, (num_patches,), generator=generator)
    x_e = x_s + patch_size
    y_e = y_s + patch_size
    return torch.stack((x_s, y_s, x_e, y_e), dim=0)  # [4, num_patches]


def _image_to_chw_float(image: np.ndarray) -> torch.Tensor:
    """``(H, W, C)`` uint8 array -> contiguous ``(C, H, W)`` float32 tensor in ``[0, 1]``."""
    img = torch.from_numpy(image.copy()).float().div(255.0)  # (H, W, C)
    return img.permute(2, 0, 1).contiguous()  # (C, H, W)


def crop_patches(image: np.ndarray, boxes: torch.Tensor) -> torch.Tensor:
    """Crop each box out of the image as a ``float32`` patch in ``[0, 1]``.

    This is the constant-size simplification of upstream
    ``patchify_and_resize``: because every box is already ``patch_size`` square,
    no resize is needed (upstream resizes only because its ``v3`` boxes vary in
    size). Axis-aligned only — see :func:`crop_patches_rotated` for orientation.

    Args:
        image: ``(H, W, C)`` uint8 array.
        boxes: ``(4, num_patches)`` integer box tensor from
            :func:`sample_offgrid_patches`.

    Returns:
        ``float32`` tensor of shape ``(num_patches, C, patch_h, patch_w)`` in
        ``[0, 1]``.
    """
    img = _image_to_chw_float(image)  # (C, H, W)
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


# --------------------------------------------------------------------------- #
# Rotation (RoPART extension)
# --------------------------------------------------------------------------- #


def rotation_window_size(patch_size: int) -> int:
    """Side of the source window that fully contains a rotated ``patch_size`` square.

    A ``patch_size`` square rotated by any angle fits inside its diagonal,
    ``ceil(patch_size * sqrt(2))``. Reading from a window this large lets
    :func:`crop_patches_rotated` fill the rotated patch with real pixels (no
    empty corners).
    """
    return math.ceil(patch_size * math.sqrt(2))


def rotation_margin(patch_size: int) -> int:
    """Edge margin (px) a patch centre needs so its rotation window stays in-image.

    Pass this as ``margin`` to :func:`sample_offgrid_patches` when patches will be
    rotated, so the diagonal-sized source window never falls off the image (which
    would otherwise leave black slivers at oblique angles).
    """
    return (rotation_window_size(patch_size) - patch_size + 1) // 2


def sample_rotation_angles(
    num_patches: int,
    *,
    low: float = 0.0,
    high: float = 2.0 * math.pi,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one uniformly random orientation per patch.

    Args:
        num_patches: number of angles to sample.
        low, high: range in **radians** (default ``[0, 2π)`` — a uniformly random
            rotation in SO(2)).
        generator: optional ``torch.Generator`` for reproducible sampling.

    Returns:
        ``float32`` tensor of shape ``(num_patches,)`` in radians.
    """
    return (torch.rand(num_patches, generator=generator) * (high - low) + low).float()


def sample_quad_rotations(
    num_patches: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one right-angle orientation per patch from ``{0, 90, 180, 270}°``.

    The discrete (RotNet-style) restriction of :func:`sample_rotation_angles`:
    angles in **radians** drawn uniformly from ``{0, π/2, π, 3π/2}``. Pairs with the
    ``quad`` control (:func:`crop_patches_quad`), whose exact ``rot90`` extraction is
    interpolation-free, so the rotation carries no angle-correlated blur cue.

    Args:
        num_patches: number of angles to sample.
        generator: optional ``torch.Generator`` for reproducible sampling.

    Returns:
        ``float32`` tensor of shape ``(num_patches,)`` in radians, each a multiple
        of ``π/2``.
    """
    k = torch.randint(0, 4, (num_patches,), generator=generator)
    return (k.float() * (math.pi / 2.0)).float()


def crop_patches_rotated(
    image: np.ndarray,
    boxes: torch.Tensor,
    angles: torch.Tensor,
    *,
    fill_corners: bool = True,
    interpolation: str = "bilinear",
    supersample: int = 1,
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """Crop each box rotated by its angle, resampling the pixels once.

    For patch ``k`` the output pixel at patch-local offset ``(lx, ly)`` (centred,
    pixel-centre coordinates) reads the source at

        (sx, sy) = centre + R(φ_k) · (lx, ly),   R(φ) = [[cos, -sin], [sin, cos]]

    so the patch footprint on the source is the ``patch_size`` square rotated by
    ``φ_k`` about the patch **centre** (the rotation-invariant anchor). A single
    ``grid_sample`` does the resampling, i.e. exactly one interpolation.

    Args:
        image: ``(H, W, C)`` uint8 array.
        boxes: ``(4, N)`` integer box tensor (constant ``patch_size`` squares).
        angles: ``(N,)`` orientations in radians (see :func:`sample_rotation_angles`).
        fill_corners: if ``True`` (default), sample from the full image so the
            rotated patch has **no empty corners** (requires the centre to be at
            least :func:`rotation_margin` from the edges). If ``False``, rotate the
            axis-aligned ``patch_size`` crop in isolation, leaving the **empty
            (black) corners** the diagonal-window method exists to avoid — useful
            for *seeing* that artefact.
        interpolation: ``"bilinear"`` (default) or ``"nearest"``. Nearest avoids
            the resampling blur but aliases; bilinear is the standard choice, and
            its angle-dependent blur is the interpolation confound (Validity
            threat #1).
        supersample: anti-aliasing factor — control (c). With ``k > 1`` the source
            is bicubically upsampled by ``k`` before sampling, so the bilinear read
            happens on a ``k``× denser grid (fractional offsets shrink ~``1/k``).
            This makes the resampling near-lossless and its blur near
            angle-independent, removing the confound at the source. ``1`` (default)
            is the plain, confounded path.
        padding_mode: ``grid_sample`` padding for samples outside the source
            (``"zeros"``, ``"border"`` or ``"reflection"``).

    Returns:
        ``float32`` tensor of shape ``(N, C, patch_size, patch_size)`` in ``[0, 1]``.
    """
    if interpolation not in ("bilinear", "nearest"):
        raise ValueError(f"interpolation must be 'bilinear' or 'nearest', got {interpolation!r}")
    if supersample < 1:
        raise ValueError(f"supersample must be >= 1, got {supersample}")

    img = _image_to_chw_float(image)  # (C, H, W)
    h0, w0 = img.shape[-2], img.shape[-1]
    angles_l = angles.float().tolist()
    boxes_l = boxes.T.tolist()

    # Upsample the shared full image once (control (c)). We normalise grid
    # coordinates against the ORIGINAL extent and rely on align_corners=True so the
    # upsampled source maps to the same physical coordinates.
    full_src = img
    if fill_corners and supersample > 1:
        full_src = F.interpolate(
            img[None], scale_factor=supersample, mode="bicubic", align_corners=True
        ).clamp_(0.0, 1.0)[0]

    # Fast path: when every patch reads the SAME source (``fill_corners``) and is the
    # same size (the RoPART sampling invariant — constant ``patch_size``), the whole
    # batch of rotated reads is one ``grid_sample`` instead of ``N`` in a Python loop.
    # This is a pure performance refactor: the arithmetic and the ``grid_sample`` call
    # are bit-identical to the loop below (verified by
    # ``test_crop_patches_rotated_batched_equals_loop``). The upsample above is the
    # dominant cost and is untouched, so the win is ~10% — the loop's dispatch overhead.
    same_size = boxes_l and all((b[2] - b[0]) == (b[3] - b[1]) == (boxes_l[0][2] - boxes_l[0][0]) for b in boxes_l)
    if fill_corners and same_size:
        return _crop_rotated_batched(
            full_src, boxes_l, angles_l, w0, h0, interpolation, padding_mode
        )

    patches = []
    for (x_s, y_s, x_e, y_e), phi in zip(boxes_l, angles_l):
        ps = x_e - x_s
        cos, sin = math.cos(phi), math.sin(phi)

        # patch-local pixel-centre coordinates, centred on 0 (output is ps×ps;
        # supersampling raises the SOURCE resolution, not the output count)
        idx = torch.arange(ps, dtype=torch.float32) - (ps - 1) / 2.0
        ly, lx = torch.meshgrid(idx, idx, indexing="ij")

        if fill_corners:
            src = full_src
            # pixel-index centre (not the geometric (x_s+x_e)/2): with
            # align_corners=True an integer coordinate hits a pixel centre exactly,
            # so this keeps axis-aligned angles (φ = 0, 90°, …) interpolation-free.
            cx = (x_s + x_e - 1) / 2.0
            cy = (y_s + y_e - 1) / 2.0
            norm_w, norm_h = w0, h0
        else:
            src = img[:, y_s:y_e, x_s:x_e]
            if supersample > 1:
                src = F.interpolate(
                    src[None], scale_factor=supersample, mode="bicubic", align_corners=True
                ).clamp_(0.0, 1.0)[0]
            cx = cy = (ps - 1) / 2.0
            norm_w = norm_h = ps

        sx = cx + cos * lx - sin * ly
        sy = cy + sin * lx + cos * ly

        # normalise to [-1, 1] against the original extent for grid_sample(align_corners=True)
        gx = sx / ((norm_w - 1) / 2.0) - 1.0
        gy = sy / ((norm_h - 1) / 2.0) - 1.0
        grid = torch.stack((gx, gy), dim=-1)[None]  # (1, ps, ps, 2)

        patch = F.grid_sample(
            src[None], grid, mode=interpolation, padding_mode=padding_mode, align_corners=True
        )[0]
        patches.append(patch)
    return torch.stack(patches, dim=0)  # (N, C, ps, ps)


def _crop_rotated_batched(
    full_src: torch.Tensor,
    boxes_l: list[list[int]],
    angles_l: list[float],
    w0: int,
    h0: int,
    interpolation: str,
    padding_mode: str,
) -> torch.Tensor:
    """Batched shared-source rotated read — the fast path of :func:`crop_patches_rotated`.

    Assumes every box is the same ``ps x ps`` size and reads the one shared
    ``full_src`` (the ``fill_corners`` case). Builds all ``N`` sampling grids at once
    and issues a **single** ``grid_sample`` by stacking the grids along the output
    height (source batch stays 1, so the — possibly upsampled — ``full_src`` is never
    materialised ``N`` times). The per-pixel arithmetic mirrors the loop exactly:
    Python-float ``cos/sin/cx/cy`` become ``float32`` tensors, and
    ``python_float * float32_tensor`` is computed in ``float32`` either way, so the
    result is bit-identical (see the equality test).
    """
    c = full_src.shape[0]
    n = len(boxes_l)
    ps = boxes_l[0][2] - boxes_l[0][0]

    # patch-local pixel-centre coordinates, shared across all patches (constant size)
    idx = torch.arange(ps, dtype=torch.float32) - (ps - 1) / 2.0
    ly, lx = torch.meshgrid(idx, idx, indexing="ij")  # (ps, ps)

    cos = torch.tensor([math.cos(a) for a in angles_l], dtype=torch.float32).view(n, 1, 1)
    sin = torch.tensor([math.sin(a) for a in angles_l], dtype=torch.float32).view(n, 1, 1)
    # pixel-index centre (matches the loop's (x_s + x_e - 1) / 2 — keeps axis-aligned
    # angles interpolation-free under align_corners=True).
    cx = torch.tensor([(b[0] + b[2] - 1) / 2.0 for b in boxes_l], dtype=torch.float32).view(n, 1, 1)
    cy = torch.tensor([(b[1] + b[3] - 1) / 2.0 for b in boxes_l], dtype=torch.float32).view(n, 1, 1)

    sx = cx + cos * lx - sin * ly  # (n, ps, ps)
    sy = cy + sin * lx + cos * ly

    gx = sx / ((w0 - 1) / 2.0) - 1.0
    gy = sy / ((h0 - 1) / 2.0) - 1.0
    grid = torch.stack((gx, gy), dim=-1).reshape(1, n * ps, ps, 2)  # batch-along-height

    out = F.grid_sample(
        full_src[None], grid, mode=interpolation, padding_mode=padding_mode, align_corners=True
    )  # (1, C, n*ps, ps)
    return out.reshape(c, n, ps, ps).permute(1, 0, 2, 3).contiguous()  # (n, C, ps, ps)


def crop_patches_quad(
    image: np.ndarray,
    boxes: torch.Tensor,
    angles: torch.Tensor,
) -> torch.Tensor:
    """Axis-aligned crops rotated by an exact multiple of 90° — no interpolation.

    The discrete counterpart of :func:`crop_patches_rotated`. Each square patch is
    cropped axis-aligned (:func:`crop_patches`) then turned by
    ``k = round(angle / (π/2)) mod 4`` quarter-turns via :func:`torch.rot90` — an
    exact pixel permutation. Because a ``P×P`` square maps onto itself under
    ``rot90`` there is **no resampling** (hence no angle-correlated blur signature —
    the interpolation confound vanishes) and **no** diagonal source window / rotation
    margin is needed: boxes are sampled exactly like the translation baseline.

    ``angles`` must be (near) multiples of ``π/2`` — i.e. produced by
    :func:`sample_quad_rotations`. ``k`` counts counter-clockwise turns, defining the
    ``+φ`` direction for this control; ``build_targets`` differences the *same*
    ``angles``, so the ``(cos Δφ, sin Δφ)`` targets and the rotated pixels stay
    consistent.

    Args:
        image: ``(H, W, C)`` uint8 array.
        boxes: ``(4, num_patches)`` integer boxes (``P×P``, axis-aligned).
        angles: ``(num_patches,)`` per-patch orientations in radians.

    Returns:
        ``float32`` tensor ``(num_patches, C, P, P)`` in ``[0, 1]``.
    """
    patches = crop_patches(image, boxes)  # (N, C, P, P)
    ks = (torch.round(angles.float() / (math.pi / 2.0)).long() % 4).tolist()
    out = torch.empty_like(patches)
    for i, k in enumerate(ks):
        out[i] = torch.rot90(patches[i], k, dims=(-2, -1))
    return out


def _odd_kernel_for_sigma(sigma: float) -> int:
    """Smallest odd Gaussian kernel size covering ~3σ each side."""
    return max(int(2 * round(3.0 * sigma) + 1), 3)


def gaussian_blur_patches(
    patches: torch.Tensor,
    sigma: float | torch.Tensor,
) -> torch.Tensor:
    """Gaussian-blur patches, optionally with a per-patch ``sigma``.

    A building block for the **matched-blur control** of the interpolation
    confound (Validity threat #1): rotation blurs oblique angles more than
    axis-aligned ones, so adding compensating blur to the sharper patches can
    flatten the orientation-correlated sharpness cue. ``sigma <= 0`` leaves a
    patch unchanged.

    Args:
        patches: ``(N, C, H, W)`` batch or a single ``(C, H, W)`` patch.
        sigma: scalar (same blur for all) or ``(N,)`` tensor (per-patch), in pixels.

    Returns:
        Blurred tensor of the same shape as ``patches``.
    """
    single = patches.dim() == 3
    x = patches[None] if single else patches

    if torch.is_tensor(sigma):
        out = torch.stack(
            [
                p if s <= 0 else _tv_gaussian_blur(p, _odd_kernel_for_sigma(s), s)
                for p, s in zip(x, sigma.tolist())
            ]
        )
    else:
        out = x if sigma <= 0 else _tv_gaussian_blur(x, _odd_kernel_for_sigma(sigma), sigma)

    return out[0] if single else out
