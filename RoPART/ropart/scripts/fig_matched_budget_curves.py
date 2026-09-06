"""Figure: the four matched-budget arms over 250 epochs, and what the extra budget bought.

Companion to ``fig_convergence_confound.py``, which showed the same comparison cut short
at 100 epochs. Here the vertical rule marks where that earlier budget ended, so the
reader can see directly what happened beyond it.

Upper panel --- ``train_loss`` against epoch for all four arms, with the final twenty
epochs shaded. The inset zooms that tail on a linear axis: the curves are close to flat
but still creeping, falling 1.9--2.7 per cent over the final ten epochs against the
5.3 per cent that the 100-epoch set was still falling.

Lower panel --- the cosine learning rate, for the same reason as in the earlier figure:
convergence is the loss having stopped moving *while* the schedule is at its floor.

Data: ``data/matched_budget_250ep_curves.csv``, extracted from the SLURM stdout of jobs
269805 (baseline), 269807 (rotated pixels, unsupervised), 269806 (RoPART +-30 deg) and
269808 (RoPART +-90 deg).

Run from the ``RoPART`` package root::

    python ropart/scripts/fig_matched_budget_curves.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "matched_budget_250ep_curves.csv"
OUT = HERE.parents[2] / "latex" / "final_report" / "figures"

OLD_BUDGET = 100  # where the stood-down comparison ended
TAIL = 20         # shaded "has it converged?" band
ZOOM = 40         # epochs shown in the inset

# --- palette (dataviz reference instance, validated) ------------------------------ #
C_BASE = "#4a3aa7"   # slot 7, violet
C_UNSUP = "#1baf7a"  # slot 3, aqua
C_R30 = "#2a78d6"    # slot 1, blue  (as in fig_convergence_confound)
C_R90 = "#eb6834"    # slot 2, orange
INK = "#0b0b0b"
MUTED = "#8a8983"

ARMS = [
    ("baseline", "baseline", C_BASE),
    ("rot_unsup", "rotated pixels, unsupervised", C_UNSUP),
    ("ropart30", r"RoPART $\pm 30^\circ$", C_R30),
    ("ropart90", r"RoPART $\pm 90^\circ$", C_R90),
]


def load() -> dict[str, list[tuple[int, float, float]]]:
    """Read the curve file as ``{arm: [(epoch, lr, train_loss), ...]}``."""
    series: dict[str, list[tuple[int, float, float]]] = {}
    with DATA.open() as fh:
        for row in csv.DictReader(r for r in fh if not r.startswith("#")):
            series.setdefault(row["arm"], []).append(
                (int(row["epoch"]), float(row["lr"]), float(row["train_loss"]))
            )
    for arm in series.values():
        arm.sort()
    return series


def main() -> None:
    series = load()
    last = max(e for e, _, _ in series["baseline"])

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

    fig, (axl, axlr) = plt.subplots(
        2, 1, figsize=(6.6, 3.6), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.12},
    )

    axl.axvspan(last - TAIL, last, color=MUTED, alpha=0.10, lw=0)
    axl.axvline(OLD_BUDGET, color=MUTED, lw=0.7, ls=(0, (4, 2)), zorder=1)
    axl.text(OLD_BUDGET - 3, 1.15, "the 100-epoch budget\nof the stood-down comparison",
             fontsize=6.4, color=MUTED, ha="right", va="top")

    for key, label, colour in ARMS:
        curve = series[key]
        axl.plot([e for e, _, _ in curve], [v for _, _, v in curve],
                 lw=1.3, color=colour, label=label, zorder=3)

    axl.set_yscale("log")
    axl.set_ylim(0.04, 1.35)
    axl.set_yticks([0.05, 0.1, 0.2, 0.5, 1.0])
    axl.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1.0"])
    axl.minorticks_off()
    axl.set_ylabel("train loss")
    axl.legend(loc="lower left", fontsize=7, frameon=False, handlelength=2.0,
               labelspacing=0.25, borderaxespad=0.6)
    axl.grid(axis="y", color=MUTED, alpha=0.16, lw=0.5)
    axl.set_axisbelow(True)

    # --- inset: the tail, linear, so "flat" can be judged --- #
    ins = axl.inset_axes((0.545, 0.50, 0.43, 0.45))
    ins.axvspan(last - TAIL, last, color=MUTED, alpha=0.10, lw=0)
    for key, _, colour in ARMS:
        curve = [p for p in series[key] if p[0] >= last - ZOOM]
        ins.plot([e for e, _, _ in curve], [v for _, _, v in curve], lw=1.1, color=colour)
    ins.set_xlim(last - ZOOM, last)
    ins.set_ylim(0.03, 0.135)
    ins.set_yticks([0.05, 0.10])
    ins.tick_params(labelsize=5.6, length=2, pad=1.5)
    for side in ins.spines.values():
        side.set_color(MUTED)
        side.set_linewidth(0.5)
    ins.set_title(f"final {ZOOM} epochs, linear axis --- still falling 1.9-2.7%",
                  fontsize=5.8, color=MUTED, pad=2)

    # --- lower: the schedule --- #
    axlr.axvspan(last - TAIL, last, color=MUTED, alpha=0.10, lw=0)
    axlr.axvline(OLD_BUDGET, color=MUTED, lw=0.7, ls=(0, (4, 2)), zorder=1)
    axlr.plot([e for e, _, _ in series["baseline"]],
              [lr for _, lr, _ in series["baseline"]], lw=1.2, color=MUTED)
    axlr.set_yscale("log")
    axlr.set_xlabel("epoch")
    axlr.set_ylabel("lr")
    axlr.set_xlim(0, last)
    axlr.set_ylim(5e-7, 1.5e-3)
    axlr.set_yticks([1e-6, 1e-5, 1e-4])
    axlr.minorticks_off()
    axlr.text(150, 6.0e-6, "cosine annealed to min_lr", fontsize=6.4, color=MUTED,
              ha="center", va="center")
    axlr.grid(axis="y", color=MUTED, alpha=0.16, lw=0.5)
    axlr.set_axisbelow(True)

    fig.tight_layout(pad=0.4)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "matched_budget_curves.pdf", bbox_inches="tight")
    fig.savefig(OUT / "matched_budget_curves.png", dpi=200, bbox_inches="tight")

    for key, label, _ in ARMS:
        curve = series[key]
        drop = 100 * (curve[-11][2] - curve[-1][2]) / curve[-11][2]
        print(f"{key:10s} ep{OLD_BUDGET} {curve[OLD_BUDGET - 1][2]:.4f}  "
              f"final {curve[-1][2]:.4f}  final-10 drop {drop:.2f}%")
    print("wrote", OUT / "matched_budget_curves.pdf")


if __name__ == "__main__":
    main()
