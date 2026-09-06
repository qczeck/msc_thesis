"""Figure: the interpolation confound and the four controls that break it.

Reproduces the sharpness-vs-angle sweep of ``02_rotation.ipynb`` (cells 4c/4e) as a
publication figure for the report's \\cref{sec:interpolation-controls}.

Left panel  --- the cue: bilinear resampling loses high-frequency energy as a
deterministic function of the rotation angle. The loss is a near-step function: zero at
exact multiples of 90 degrees (where the sample points land on pixel centres) and
saturated at roughly 21 per cent everywhere else. ``nearest`` is the mechanism check ---
no interpolation, no loss. The inset resolves the residual ripple, whose maxima sit at
45/135 degrees, *not* minima.

Right panel --- the four controls of the section, on the same axis, so the reader can
see *which cue each one breaks* and what it costs in signal.

Both panels plot high-frequency energy relative to the raw path at 0 degrees, so the
curves can be read directly as the sharpness ratios quoted in the text.

Run from the ``RoPART`` package root::

    python ropart/scripts/fig_interpolation_confound.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

from helpers import (
    crop_patches_rotated,
    gaussian_blur_patches,
    load_image,
    rotation_margin,
    sample_offgrid_patches,
)

# --- the run's sampling configuration (notebook 02) ------------------------------- #
IMG_SIZE = 512
PATCH_SIZE = 64
NUM_PATCHES = 64
NUM_PROBE = 16
SEED = 0

SIGMA_LP = 1.0        # (d) dominant isotropic low-pass
SIGMA_RAND_MAX = 0.8  # (b) randomised blur, sigma ~ U[0, SIGMA_RAND_MAX]
SUPERSAMPLE = 4       # (c) supersample factor

OUT = Path(__file__).resolve().parents[3] / "latex" / "final_report" / "figures"

# --- palette (dataviz reference instance, validated) ------------------------------ #
C_RAW = "#2a78d6"      # slot 1, blue
C_MATCHED = "#eb6834"  # slot 2, orange
C_RANDOM = "#1baf7a"   # slot 3, aqua
C_SUPER = "#eda100"    # slot 4, yellow
C_DOM = "#4a3aa7"      # slot 7, violet
INK = "#0b0b0b"
MUTED = "#8a8983"


def hf_energy(patches: torch.Tensor) -> float:
    """Mean absolute pixel gradient --- the high-frequency-energy sharpness proxy."""
    dx = (patches[..., 1:] - patches[..., :-1]).abs().mean(dim=(1, 2, 3))
    dy = (patches[..., 1:, :] - patches[..., :-1, :]).abs().mean(dim=(1, 2, 3))
    return float((dx + dy).mean())


def match_sigma(patches: torch.Tensor, target: float, hi: float = 3.0, iters: int = 20) -> float:
    """Bisect for the isotropic sigma that blurs ``patches`` down to ``target`` energy."""
    if hf_energy(patches) <= target:
        return 0.0
    lo = 0.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if hf_energy(gaussian_blur_patches(patches, mid)) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main() -> None:
    image = load_image("example.jpeg", img_size=IMG_SIZE)
    gen = torch.Generator().manual_seed(SEED)
    boxes = sample_offgrid_patches(
        IMG_SIZE, PATCH_SIZE, NUM_PATCHES, margin=rotation_margin(PATCH_SIZE), generator=gen
    )
    probe = boxes[:, :NUM_PROBE]
    n = probe.shape[1]

    deg = torch.arange(0, 181, 2.5)
    rad = deg * math.pi / 180

    def rot_at(a: float, **kw) -> torch.Tensor:
        return crop_patches_rotated(image, probe, torch.full((n,), float(a)), **kw)

    base = [rot_at(a) for a in rad]
    e_raw = torch.tensor([hf_energy(p) for p in base])
    e_near = torch.tensor([hf_energy(rot_at(a, interpolation="nearest")) for a in rad])

    # (a) matched blur: per-angle sigma flattening every angle to the 45-degree floor.
    target = float(e_raw.min())
    e_matched = torch.tensor(
        [hf_energy(gaussian_blur_patches(p, match_sigma(p, target))) for p in base]
    )

    # (b) randomised blur: sigma drawn independently of the angle.
    g_rand = torch.Generator().manual_seed(1)
    e_random = torch.tensor(
        [
            hf_energy(gaussian_blur_patches(p, torch.rand(n, generator=g_rand) * SIGMA_RAND_MAX))
            for p in base
        ]
    )

    # (c) supersample, (d) dominant isotropic low-pass.
    e_super = torch.tensor([hf_energy(rot_at(a, supersample=SUPERSAMPLE)) for a in rad])
    e_dom = torch.tensor([hf_energy(gaussian_blur_patches(p, SIGMA_LP)) for p in base])

    ref = float(e_raw[0])  # normalise everything by the raw path at 0 degrees
    x = deg.numpy()

    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "text.color": INK,
        "axes.labelcolor": INK,
        "pdf.fonttype": 42,
    })

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)

    for ax in (axl, axr):
        for d in (45, 135):
            ax.axvline(d, color=MUTED, lw=0.5, ls=":", zorder=0)
        ax.set_xlim(0, 180)
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_xlabel(r"patch orientation $\varphi$ (deg)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=MUTED, lw=0.4, alpha=0.25)
        ax.set_axisbelow(True)

    # --- left: the cue --- #
    axl.plot(x, (e_near / ref).numpy(), lw=1.3, color=MUTED, ls=(0, (4, 2)))
    axl.plot(x, (e_raw / ref).numpy(), lw=1.6, color=C_RAW, marker="o", ms=2.0,
             markevery=2, mec="none")
    axl.set_ylabel("high-frequency energy\n(relative to raw at $0^\\circ$)")
    axl.annotate("sharp only at exact\nmultiples of $90^\\circ$", xy=(90, 1.005),
                 xytext=(101, 1.16), color=C_RAW, fontsize=7, va="center",
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=C_RAW, shrinkA=0, shrinkB=3))
    axl.text(30, 1.32, "nearest (no interpolation)", color=MUTED, fontsize=7)
    axl.text(112, 0.86, "raw (bilinear)", color=C_RAW, fontsize=7.5)
    axl.set_title("(a) the cue", fontsize=8, color=INK, loc="left", pad=6)

    # inset: the residual ripple, magnified
    ins = axl.inset_axes([0.28, 0.12, 0.46, 0.30])
    ripple = (e_raw / ref).clone()
    off_axis = ((deg % 90.0) > 2.5) & ((deg % 90.0) < 87.5)
    ripple[~off_axis] = float("nan")
    ins.plot(deg.numpy(), ripple.numpy(), lw=1.0, color=C_RAW)
    vals = ripple[off_axis]
    lo, hi = float(vals.min()), float(vals.max())
    pad = 0.15 * (hi - lo)
    ins.set_ylim(lo - pad, hi + pad)
    ins.set_xlim(0, 180)
    ins.set_xticks([45, 135])
    ins.set_yticks([])
    for d in (45, 135):
        ins.axvline(d, color=MUTED, lw=0.5, ls=":")
    ins.tick_params(labelsize=5.5, length=2, pad=1)
    for sp in ins.spines.values():
        sp.set_color(MUTED)
        sp.set_linewidth(0.5)
    ins.set_title(r"off-axis ripple, $\times 30$ zoom --- maxima at $45^\circ$",
                  fontsize=5.6, color=MUTED, pad=2)

    # --- right: the controls --- #
    series = [
        ("raw (confounded)", e_raw, C_RAW, "-"),
        (f"(c) supersample $\\times{SUPERSAMPLE}$", e_super, C_SUPER, (0, (1, 1.2))),
        ("(b) randomised blur", e_random, C_RANDOM, (0, (5, 1.5))),
        ("(a) matched blur", e_matched, C_MATCHED, (0, (3, 1.3, 1, 1.3))),
        ("(d) dominant low-pass", e_dom, C_DOM, (0, (6, 1.4, 1, 1.4, 1, 1.4))),
    ]
    for label, e, colour, style in series:
        axr.plot(x, (e / ref).numpy(), lw=1.4, color=colour, ls=style, label=label)
    axr.set_title("(b) the four controls", fontsize=8, color=INK, loc="left", pad=6)
    axr.legend(loc="lower left", bbox_to_anchor=(0.0, -0.02), fontsize=6.4, frameon=False,
               handlelength=2.6, labelspacing=0.3, borderaxespad=0.2, ncol=2,
               columnspacing=1.0)

    axl.set_ylim(0, 1.45)
    fig.tight_layout(pad=0.4, w_pad=1.4)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "interpolation_confound.pdf", bbox_inches="tight")
    fig.savefig(OUT / "interpolation_confound.png", dpi=200, bbox_inches="tight")

    print(f"raw     : 0deg {float(e_raw[0]/ref):.3f}  45deg {float(e_raw[18]/ref):.3f}")
    print(f"super x{SUPERSAMPLE}: 0deg {float(e_super[0]/ref):.3f}  45deg {float(e_super[18]/ref):.3f}")
    print(f"matched : 0deg {float(e_matched[0]/ref):.3f}  45deg {float(e_matched[18]/ref):.3f}")
    print(f"random  : 0deg {float(e_random[0]/ref):.3f}  45deg {float(e_random[18]/ref):.3f}")
    print(f"dom lp  : 0deg {float(e_dom[0]/ref):.3f}  45deg {float(e_dom[18]/ref):.3f}")
    print(f"nearest : 0deg {float(e_near[0]/ref):.3f}  45deg {float(e_near[18]/ref):.3f}")
    print("wrote", OUT / "interpolation_confound.pdf")


if __name__ == "__main__":
    main()
