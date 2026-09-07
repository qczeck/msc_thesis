"""Figure: the RoPART pretext task in one picture (report \\cref{fig:ropart-teaser}).

Left --- one image, two off-grid patches, each sampled at a continuous position *and*
a continuous orientation. The footprints are the exact regions
:func:`helpers.sampling.crop_patches_rotated` reads; the tick on each footprint points
along that patch's local $+x$ axis, so its orientation is unambiguous.

Right --- the two patches as the encoder receives them: no positional embeddings, no
absolute coordinates, nothing but pixels. The supervision is the relative rigid motion
between them, which is what the double arrow asks for.

The geometry is defined in the $224 \\times 224$ frame the ImageNet-100 runs use, with
$P = 32$ and the $\\pm 30^\\circ$ arm's rotation margin, so the numbers printed by this
script are the actual targets of \\cref{eq:target-ropart}. Pixels are read from a
$6\\times$ copy of the same photograph (1344 px) purely so the crops survive print;
scaling every coordinate by six leaves $(\\Delta x, \\Delta y)$ --- measured in patch
widths --- and $\\Delta\\varphi$ unchanged.

Run from the ``RoPART`` package root::

    python ropart/scripts/fig_ropart_pair.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, Polygon

from helpers import (
    crop_patches_rotated,
    load_image,
    relative_orientation,
    relative_translation,
    rotation_margin,
)

# --- the sampling configuration this figure depicts ------------------------------- #
IMG_SIZE = 224           # the ImageNet-100 pretraining frame
PATCH_SIZE = 32          # ViT-B/32
SCALE = 6                # 224 * 6 = 1344: the print-resolution copy of the same photo
DISPLAY = 672            # downsample for the overlay panel (keeps the PDF small)

# Two hand-picked patches, in 224-frame (x_s, y_s, x_e, y_e). Both carry recoverable
# orientation: the head (eyes, muzzle, open mouth) and the outstretched front paw.
BOX_HEAD = (112, 62, 144, 94)
BOX_PAW = (144, 166, 176, 198)

# Per-patch orientations, in degrees, both inside the +-30 arm's range. Positive is
# clockwise on screen: the local +x axis is R(phi) . (1, 0) and the image y axis
# points down (see helpers.sampling.crop_patches_rotated).
PHI_HEAD_DEG = -25.0
PHI_PAW_DEG = +20.0

OUT = Path(__file__).resolve().parents[3] / "latex" / "final_report" / "figures"
SRC = OUT / "dog_1344.png"

# --- palette (shared with fig_interpolation_confound.py) -------------------------- #
C_REF = "#2a78d6"    # reference patch i, blue
C_TGT = "#eb6834"    # target patch j, orange
INK = "#0b0b0b"
MUTED = "#8a8983"


def _to_hwc(patch: torch.Tensor) -> np.ndarray:
    """``(C, H, W)`` float tensor -> ``(H, W, C)`` array for ``imshow``."""
    return patch.detach().cpu().permute(1, 2, 0).numpy()


def _footprint(ax, box: tuple[int, int, int, int], phi: float, colour: str) -> None:
    """Draw one rotated patch footprint plus its orientation tick, in 224-frame coords."""
    x_s, y_s, x_e, y_e = box
    cx, cy, h = (x_s + x_e) / 2.0, (y_s + y_e) / 2.0, (x_e - x_s) / 2.0
    cos, sin = math.cos(phi), math.sin(phi)

    def rot(lx: float, ly: float) -> tuple[float, float]:
        return (cx + cos * lx - sin * ly, cy + sin * lx + cos * ly)

    corners = [rot(-h, -h), rot(h, -h), rot(h, h), rot(-h, h)]
    # a white underlay so the outline reads against both grass and concrete
    ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor="white",
                         linewidth=3.0, joinstyle="round", alpha=0.85, zorder=3))
    ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=colour,
                         linewidth=1.8, joinstyle="round", zorder=4))
    tx, ty = rot(h, 0.0)  # tick along the patch's local +x
    ax.plot([cx, tx], [cy, ty], color="white", lw=2.8, alpha=0.85, zorder=3,
            solid_capstyle="round")
    ax.plot([cx, tx], [cy, ty], color=colour, lw=1.6, zorder=4, solid_capstyle="round")
    ax.plot([cx], [cy], marker="o", ms=2.6, mfc=colour, mec="white", mew=0.7, zorder=5)


def main() -> None:
    margin = rotation_margin(PATCH_SIZE)
    for name, box in (("head", BOX_HEAD), ("paw", BOX_PAW)):
        lo, hi = margin, IMG_SIZE - PATCH_SIZE - margin
        assert lo <= box[0] <= hi and lo <= box[1] <= hi, f"{name} box violates margin {margin}"

    boxes224 = torch.tensor(
        [[BOX_HEAD[k], BOX_PAW[k]] for k in range(4)], dtype=torch.long
    )  # (4, 2), rows (x_s, y_s, x_e, y_e)
    angles = torch.tensor(
        [math.radians(PHI_HEAD_DEG), math.radians(PHI_PAW_DEG)], dtype=torch.float32
    )

    # --- the targets, straight from the target builders --- #
    delta = relative_translation(boxes224, normalize_by=PATCH_SIZE)  # (2, N, N)
    rot = relative_orientation(angles)                               # (2, N, N)
    dx, dy = float(delta[0, 0, 1]), float(delta[1, 0, 1])
    cos_d, sin_d = float(rot[0, 0, 1]), float(rot[1, 0, 1])
    dphi_deg = math.degrees(math.atan2(sin_d, cos_d))

    # --- the pixels, read at 6x from the same photograph --- #
    image = load_image(str(SRC))
    patches = crop_patches_rotated(image, boxes224 * SCALE, angles, fill_corners=True)

    display = load_image(str(SRC), img_size=DISPLAY)

    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "mathtext.fontset": "cm",
    })

    fig_w, fig_h = 7.0, 2.6
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax_img = fig.add_axes((0.004, 0.10, 0.345, 0.88))
    # square patch axes: equal width and height in inches, so the coloured spines
    # hug the patch instead of letterboxing it
    pw = 0.228
    ph = pw * fig_w / fig_h
    py = 0.20
    ax_ref = fig.add_axes((0.392, py, pw, ph))
    ax_tgt = fig.add_axes((0.767, py, pw, ph))

    # --- (a) the image, with both footprints --- #
    ax_img.imshow(display, extent=(0, IMG_SIZE, IMG_SIZE, 0), interpolation="antialiased")
    ax_img.set_xlim(0, IMG_SIZE)
    ax_img.set_ylim(IMG_SIZE, 0)
    ax_img.set_axis_off()

    ci = ((BOX_HEAD[0] + BOX_HEAD[2]) / 2, (BOX_HEAD[1] + BOX_HEAD[3]) / 2)
    cj = ((BOX_PAW[0] + BOX_PAW[2]) / 2, (BOX_PAW[1] + BOX_PAW[3]) / 2)
    ax_img.annotate(
        "", xy=cj, xytext=ci,
        arrowprops=dict(arrowstyle="-|>,head_width=0.16,head_length=0.4", color="white",
                        lw=2.4, alpha=0.8, shrinkA=6, shrinkB=6),
    )
    ax_img.annotate(
        "", xy=cj, xytext=ci,
        arrowprops=dict(arrowstyle="-|>,head_width=0.16,head_length=0.4", color=INK,
                        lw=1.0, ls=(0, (3, 2)), shrinkA=6, shrinkB=6),
    )
    _footprint(ax_img, BOX_HEAD, float(angles[0]), C_REF)
    _footprint(ax_img, BOX_PAW, float(angles[1]), C_TGT)

    fig.text(0.1765, 0.075, f"one image, two off-grid patches ($P = {PATCH_SIZE}$ px)",
             ha="center", va="top", fontsize=7.4, color=MUTED, style="italic")

    # --- (b) the two patches, as the encoder receives them --- #
    for ax, patch, colour, label, phi in (
        (ax_ref, patches[0], C_REF, r"reference $i$", PHI_HEAD_DEG),
        (ax_tgt, patches[1], C_TGT, r"target $j$", PHI_PAW_DEG),
    ):
        ax.imshow(_to_hwc(patch), interpolation="antialiased")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(colour)
            sp.set_linewidth(1.8)
        ax.set_title(label, fontsize=8.6, color=colour, pad=4)

    # --- the question between them --- #
    y_mid = py + ph / 2.0
    arrow = FancyArrowPatch(
        (0.645, y_mid), (0.742, y_mid),
        transform=fig.transFigure, figure=fig,
        arrowstyle="<|-|>,head_width=0.2,head_length=0.45",
        mutation_scale=11, lw=1.2, color=INK, shrinkA=0, shrinkB=0,
    )
    fig.add_artist(arrow)
    fig.text(0.6935, y_mid + 0.05,
             "\n".join((r"$\Delta x = {?}$", r"$\Delta y = {?}$", r"$\Delta\varphi = {?}$")),
             ha="center", va="bottom", fontsize=9.2, color=INK, linespacing=1.5)
    fig.text(0.6935, 0.075, "what the pretext task asks for", ha="center", va="top",
             fontsize=7.4, color=MUTED, style="italic")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "ropart_pair.pdf", bbox_inches="tight")
    fig.savefig(OUT / "ropart_pair.png", dpi=220, bbox_inches="tight")

    print(f"reference i: box {BOX_HEAD}  phi {PHI_HEAD_DEG:+.0f} deg")
    print(f"target    j: box {BOX_PAW}  phi {PHI_PAW_DEG:+.0f} deg")
    print(f"(dx, dy) = ({dx:+.3f}, {dy:+.3f}) patch widths "
          f"= ({dx * PATCH_SIZE:+.0f}, {dy * PATCH_SIZE:+.0f}) px")
    print(f"dphi = {dphi_deg:+.1f} deg -> (cos, sin) = ({cos_d:+.3f}, {sin_d:+.3f})")
    print("wrote", OUT / "ropart_pair.pdf")


if __name__ == "__main__":
    main()
