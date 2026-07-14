"""Interpolation-control registry — the single seam for swapping controls.

Each control is a patch-*extraction* strategy: given a source image, the sampled
boxes and (optionally) per-patch angles, it returns the patch tensor the model
will ingest. The translation baseline crops axis-aligned patches; the rotation
controls rotate about the patch centre and differ only in how they handle the
interpolation confound (Validity threat #1: rotating and resampling a small patch
leaves a blur signature correlated with the angle).

Swapping or adding a control is a one-function change: write a module-level
``_extract_*`` function and register it with :func:`register_control`. The dataset
never needs to know which control is active — it calls the extractor returned by
:func:`build_extractor`. Extractors are module-level functions wrapped in
``functools.partial`` so they pickle cleanly for spawn-based DataLoader workers.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from helpers.sampling import (
    crop_patches,
    crop_patches_quad,
    crop_patches_rotated,
    gaussian_blur_patches,
)

# An extractor maps (image HWC uint8, boxes [4,N], angles [N] or None) -> patches
# [N, C, P, P] float32 in [0, 1].
Extractor = Callable[[np.ndarray, torch.Tensor, "torch.Tensor | None"], torch.Tensor]


# --------------------------------------------------------------------------- #
# Extraction functions (module-level so functools.partial is picklable)
# --------------------------------------------------------------------------- #


def _extract_translation(image, boxes, angles=None):
    """Axis-aligned crops — translation-only baseline (no rotation)."""
    return crop_patches(image, boxes)


def _extract_raw(image, boxes, angles):
    """Plain bilinear rotation — confounded by interpolation blur."""
    return crop_patches_rotated(image, boxes, angles)


def _extract_quad(image, boxes, angles):
    """Exact 90° (quad) rotation via rot90 — interpolation-free (RotNet-style)."""
    return crop_patches_quad(image, boxes, angles)


def _extract_quad_ch2(image, boxes, angles):
    """Quad-rotated pixels, but rotation is *unsupervised* (translation-only target).

    Identical pixel extraction to :func:`_extract_quad` (exact ``rot90``); the only
    difference is the registered ``num_channels=2``, so the target builder emits only
    ``(Δx, Δy)`` and no ``(cos Δφ, sin Δφ)`` channels. This isolates whether the
    translation collapse seen in the ch=4 rotation runs is caused by the rotated
    *input pixels* (nuisance) or by the ch=4 rotation *objective*.
    """
    return crop_patches_quad(image, boxes, angles)


def _extract_supersample(image, boxes, angles, *, factor=4):
    """(c) Supersampled rotation — bicubic source upsample before the read."""
    return crop_patches_rotated(image, boxes, angles, supersample=factor)


def _extract_dominant(image, boxes, angles, *, sigma=1.0):
    """(d) Dominant isotropic low-pass — one heavy blur on every rotated patch."""
    return gaussian_blur_patches(crop_patches_rotated(image, boxes, angles), sigma)


def _extract_randomised(image, boxes, angles, *, sigma_max=0.8):
    """(b) Randomised blur — σ drawn per patch independently of the angle."""
    patches = crop_patches_rotated(image, boxes, angles)
    sigmas = torch.rand(patches.shape[0]) * sigma_max
    return gaussian_blur_patches(patches, sigmas)


def _extract_matched(image, boxes, angles):
    """(a) Matched blur — per-angle σ(φ). Needs a precomputed schedule (not wired)."""
    raise NotImplementedError(
        "control 'matched' needs a precomputed sigma(phi) schedule; not yet implemented"
    )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Control:
    fn: Callable
    rotates: bool
    num_channels: int


_REGISTRY: dict[str, _Control] = {}


def register_control(name: str, fn: Callable, *, rotates: bool, num_channels: int):
    _REGISTRY[name] = _Control(fn, rotates, num_channels)


register_control("translation", _extract_translation, rotates=False, num_channels=2)
register_control("raw", _extract_raw, rotates=True, num_channels=4)
register_control("quad", _extract_quad, rotates=True, num_channels=4)
register_control("quad_ch2", _extract_quad_ch2, rotates=True, num_channels=2)
register_control("supersample", _extract_supersample, rotates=True, num_channels=4)
register_control("dominant", _extract_dominant, rotates=True, num_channels=4)
register_control("randomised", _extract_randomised, rotates=True, num_channels=4)
register_control("matched", _extract_matched, rotates=True, num_channels=4)


def available_controls() -> list[str]:
    return sorted(_REGISTRY)


# CLI/checkpoint option name (+ default) for each extractor kwarg a control takes.
# The single source of truth shared by training (live argparse args) and eval
# (the ``args`` dict stored in a checkpoint), so the two can never drift.
_CONTROL_KWARG_SPECS: dict[str, dict[str, tuple[str, float]]] = {
    "supersample": {"factor": ("supersample", 4)},
    "dominant": {"sigma": ("sigma_lp", 1.0)},
    "randomised": {"sigma_max": ("sigma_max", 0.8)},
}


def control_kwargs(name: str, opts) -> dict:
    """Extractor kwargs for the named control, read from argparse-style options.

    ``opts`` is any mapping (e.g. ``vars(args)`` at train time, or the ``args``
    dict recovered from a checkpoint at eval time); missing keys fall back to the
    CLI defaults so old checkpoints reconstruct the extractor they trained with.
    """
    spec = _CONTROL_KWARG_SPECS.get(name, {})
    return {kw: opts.get(opt, default) for kw, (opt, default) in spec.items()}


def build_extractor(name: str, **kwargs) -> tuple[Extractor, int, bool]:
    """Return ``(extractor, num_channels, rotates)`` for the named control.

    ``num_channels`` is the relative-head width the control implies (2 for the
    translation baseline, 4 once rotation adds ``(cos Δφ, sin Δφ)``); ``rotates``
    says whether the dataset must sample per-patch angles. The extractor is a
    picklable ``functools.partial``.
    """
    if name not in _REGISTRY:
        raise ValueError(f"unknown control {name!r}; available: {available_controls()}")
    ctrl = _REGISTRY[name]
    extractor = functools.partial(ctrl.fn, **kwargs) if kwargs else ctrl.fn
    return extractor, ctrl.num_channels, ctrl.rotates
