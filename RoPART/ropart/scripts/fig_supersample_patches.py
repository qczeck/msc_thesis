"""Figure: what the supersample control does to a patch (report \\cref{fig:ss-patches}).

The three panels are the same patch of the dog's face, read through the three paths the
report distinguishes in \\cref{sec:interpolation-controls}:

1. ``phi = 0``, raw --- the axis-aligned read. Sample points land exactly on pixel
   centres, so \\cref{eq:bilinear} degenerates to a copy and the raw path costs
   nothing: the control has no work to do here.
2. ``phi = 40``, raw --- one bilinear inverse warp (\\cref{eq:inverse-warp}) at an
   off-axis angle. The fractional offsets are ``O(1)``, so the read costs
   high-frequency energy, and the loss is a deterministic function of the angle. That
   is the confound.
3. ``phi = 40``, supersample ``x4`` --- the same warp after a bicubic ``x4`` upsample of
   the source, so the fractional offsets shrink as ``O(1/k)`` (\\cref{eq:supersample})
   and the read is near-lossless.

Panels 2 and 3 differ *only* in the control, so the visible sharpness gap between them
is the cue the model could otherwise have read instead of the orientation.

Patches are shown at their true ``P = 32`` resolution with nearest-neighbour magnification
--- one patch pixel is one visible block --- because the whole effect lives at the
pixel scale and any smooth upscaling would hide it. Sharpness is \\cref{eq:sharpness} on the patch shown, divided by the same quantity for
the supersampled read of the same region at the same angle --- a rotated window frames
slightly different pixels, and dividing by its own near-lossless read cancels that
content difference and leaves the resampling loss.

Run from the ``RoPART`` package root::

    python ropart/scripts/fig_supersample_patches.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

from helpers import crop_patches_rotated, load_image

# --- the sampling configuration this figure depicts ------------------------------- #
IMG_SIZE = 224           # the ImageNet-100 pretraining frame - the true source scale
PATCH_SIZE = 32          # ViT-B/32
BOX = (112, 62, 144, 94)  # the dog's face; the reference patch of \cref{fig:ropart-teaser}
PHI_DEG = 40.0           # an off-axis angle: worst case for the bilinear read
SUPERSAMPLE = 4          # the control that all reported arms use

OUT = Path(__file__).resolve().parents[3] / "latex" / "final_report" / "figures"
SRC = OUT / "dog_1344.png"

# --- palette (shared with fig_interpolation_confound.py) -------------------------- #
C_RAW = "#2a78d6"      # slot 1, blue
C_CONFOUND = "#eb6834"  # slot 2, orange
C_SUPER = "#eda100"    # slot 4, yellow
INK = "#0b0b0b"
MUTED = "#8a8983"


def hf_energy(patches: torch.Tensor) -> float:
    """Mean absolute pixel gradient --- the high-frequency-energy sharpness proxy.

    The same estimator as ``fig_interpolation_confound.py`` and \\cref{eq:sharpness}.
    """
    dx = (patches[..., 1:] - patches[..., :-1]).abs().mean(dim=(1, 2, 3))
    dy = (patches[..., 1:, :] - patches[..., :-1, :]).abs().mean(dim=(1, 2, 3))
    return float((dx + dy).mean())


def main() -> None:
    # The true pretraining scale: a 224 frame, so the read really is P = 32 pixels.
    image = load_image(str(SRC), img_size=IMG_SIZE)
    box = torch.tensor([[BOX[0]], [BOX[1]], [BOX[2]], [BOX[3]]], dtype=torch.long)

    def read(phi_deg: float, supersample: int = 1) -> torch.Tensor:
        angles = torch.tensor([math.radians(phi_deg)], dtype=torch.float32)
        return crop_patches_rotated(
            image, box, angles, fill_corners=True, supersample=supersample
        )

    p_axis = read(0.0)
    p_raw = read(PHI_DEG)
    p_super = read(PHI_DEG, supersample=SUPERSAMPLE)

    # Sharpness is content-dependent, and a rotated window frames slightly different
    # pixels, so a panel is only comparable with the near-lossless read of the SAME
    # region at the SAME angle. Dividing by that cancels the content and leaves the
    # resampling loss; panel 3 is its own reference, hence exactly 1 by construction.
    ratios = [
        hf_energy(p_axis) / hf_energy(read(0.0, supersample=SUPERSAMPLE)),
        hf_energy(p_raw) / hf_energy(p_super),
        1.0,
    ]

    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "mathtext.fontset": "cm",
    })

    panels = (
        (p_axis, C_RAW, r"$\varphi = 0^\circ$, raw",
         "sample points on pixel centres", ratios[0]),
        (p_raw, C_CONFOUND, rf"$\varphi = {PHI_DEG:.0f}^\circ$, raw",
         "one bilinear warp: the cue", ratios[1]),
        (p_super, C_SUPER, rf"$\varphi = {PHI_DEG:.0f}^\circ$, supersample $\times {SUPERSAMPLE}$",
         "the control the arms use", ratios[2]),
    )

    fig_w, fig_h = 5.4, 2.35
    fig = plt.figure(figsize=(fig_w, fig_h))
    pw = 0.29                      # panel width in figure coords
    ph = pw * fig_w / fig_h        # square in inches
    py = 0.30
    for k, (patch, colour, title, sub, ratio) in enumerate(panels):
        ax = fig.add_axes((0.025 + k * (pw + 0.043), py, pw, ph))
        # nearest: one patch pixel = one visible block, so the blur is not hidden
        ax.imshow(patch[0].permute(1, 2, 0).numpy(), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(colour)
            sp.set_linewidth(1.8)
        ax.set_title(title, fontsize=8.2, color=colour, pad=4)
        ax.text(0.5, -0.045, sub, transform=ax.transAxes, ha="center", va="top",
                fontsize=7.0, color=MUTED, style="italic")
        rel = (rf"$E / E_{{\times {SUPERSAMPLE}}} \equiv 1$" if k == 2
               else rf"$E / E_{{\times {SUPERSAMPLE}}} = {ratio:.2f}$")
        ax.text(0.5, -0.175, rel, transform=ax.transAxes,
                ha="center", va="top", fontsize=8.2, color=INK)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "supersample_patches.pdf", bbox_inches="tight")
    fig.savefig(OUT / "supersample_patches.png", dpi=220, bbox_inches="tight")

    print(f"patch {BOX} at P = {PATCH_SIZE} in a {IMG_SIZE} frame")
    print("E of each panel over the supersampled read of the same region and angle:")
    print(f"phi   0 raw        : {ratios[0]:.3f}")
    print(f"phi {PHI_DEG:3.0f} raw        : {ratios[1]:.3f}")
    print(f"phi {PHI_DEG:3.0f} supersample: {ratios[2]:.3f} (its own reference)")
    print("wrote", OUT / "supersample_patches.pdf")


if __name__ == "__main__":
    main()
