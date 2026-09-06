"""Figure: the convergence confound behind the 100-epoch rotation-range comparison.

Plots the per-epoch training curves of the three bounded-rotation arms at P=32 that
were compared at 100 epochs, and shows why their endpoint ordering could not be read
as a cost of rotation range.

Upper panel --- ``train_loss`` against epoch. The horizontal rule marks the +-90 deg
arm's *final* value, 0.2259; the +-30 deg arm passes through it at **epoch 56**, so the
wider arm ends where the narrower one stood at 56 per cent of the budget. The shaded
band is the final ten epochs, over which +-30 deg still falls 5.3 per cent.

Lower panel --- the cosine learning rate on the same axis, annealed to ``min_lr`` well
before the loss stops moving. Together the panels say the runs hit the end of their
*schedule*, not convergence, so "wider range costs structure" and "the wider arm had
half the effective budget" are indistinguishable in this data.

Data: ``data/matched_budget_100ep_curves.csv``, extracted from the SLURM stdout of jobs
264032 (+-30 deg), 264914 (+-60 deg) and 264915 (+-90 deg).

Run from the ``RoPART`` package root::

    python ropart/scripts/fig_convergence_confound.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "matched_budget_100ep_curves.csv"
OUT = HERE.parents[2] / "latex" / "final_report" / "figures"

TAIL = 10  # epochs in the "still falling at min_lr" band

# --- palette (dataviz reference instance, validated) ------------------------------ #
C_B30 = "#2a78d6"   # slot 1, blue
C_B60 = "#1baf7a"   # slot 3, aqua
C_B90 = "#eb6834"   # slot 2, orange
INK = "#0b0b0b"
MUTED = "#8a8983"


def load() -> dict[str, list[tuple[int, float, float]]]:
    """Read the curve file as ``{range_deg: [(epoch, lr, train_loss), ...]}``."""
    series: dict[str, list[tuple[int, float, float]]] = {}
    with DATA.open() as fh:
        rows = csv.DictReader(r for r in fh if not r.startswith("#"))
        for row in rows:
            series.setdefault(row["range_deg"], []).append(
                (int(row["epoch"]), float(row["lr"]), float(row["train_loss"]))
            )
    for arm in series.values():
        arm.sort()
    return series


def crossing(curve: list[tuple[int, float, float]], level: float) -> int:
    """First epoch at which ``curve``'s loss has fallen to ``level``."""
    return next(e for e, _, loss in curve if loss <= level)


def main() -> None:
    series = load()
    b30, b60, b90 = series["30"], series["60"], series["90"]

    target = b90[-1][2]
    cross = crossing(b30, target)
    tail_drop = 100 * (b30[-TAIL - 1][2] - b30[-1][2]) / b30[-TAIL - 1][2]

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
        2, 1, figsize=(6.6, 3.4), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.12},
    )

    # --- upper: the loss curves --- #
    axl.axvspan(99 - TAIL, 99, color=MUTED, alpha=0.10, lw=0)
    axl.axhline(target, color=C_B90, lw=0.7, ls=(0, (4, 2)), zorder=1)

    for label, curve, colour in (
        (r"$\pm 30^\circ$", b30, C_B30),
        (r"$\pm 60^\circ$", b60, C_B60),
        (r"$\pm 90^\circ$", b90, C_B90),
    ):
        axl.plot([e for e, _, _ in curve], [v for _, _, v in curve],
                 lw=1.4, color=colour, label=label, zorder=3)

    axl.plot([cross], [target], "o", ms=4.5, mfc="white", mec=C_B30, mew=1.2, zorder=4)
    axl.annotate(
        f"$\\pm 30^\\circ$ is already here at epoch {cross}\n"
        f"--- {cross}% of the budget",
        xy=(cross, target), xytext=(cross + 4, 0.95),
        fontsize=6.8, color=INK, ha="left", va="top",
        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": MUTED,
                    "shrinkA": 1, "shrinkB": 4},
    )
    axl.text(1.5, target * 1.09, f"$\\pm 90^\\circ$ final, {target:.4f}",
             fontsize=6.6, color=C_B90, ha="left", va="bottom")
    axl.text(97, 0.42, f"final {TAIL} epochs:\n$\\pm 30^\\circ$ still\nfalling {tail_drop:.1f}%",
             fontsize=6.4, color=MUTED, ha="right", va="center")

    axl.set_yscale("log")
    axl.set_ylim(0.10, 1.35)
    axl.set_yticks([0.1, 0.2, 0.3, 0.5, 1.0])
    axl.set_yticklabels(["0.1", "0.2", "0.3", "0.5", "1.0"])
    axl.minorticks_off()
    axl.set_ylabel("train loss")
    axl.legend(loc="lower left", fontsize=7, frameon=False, handlelength=2.0,
               labelspacing=0.25, borderaxespad=0.6)
    axl.grid(axis="y", color=MUTED, alpha=0.16, lw=0.5)
    axl.set_axisbelow(True)

    # --- lower: the schedule that actually ended --- #
    axlr.axvspan(99 - TAIL, 99, color=MUTED, alpha=0.10, lw=0)
    axlr.plot([e for e, _, _ in b30], [lr for _, lr, _ in b30], lw=1.2, color=MUTED)
    axlr.set_yscale("log")
    axlr.set_xlabel("epoch")
    axlr.set_ylabel("lr")
    axlr.set_xlim(0, 99)
    axlr.set_ylim(5e-7, 1.5e-3)
    axlr.set_yticks([1e-6, 1e-5, 1e-4])
    axlr.minorticks_off()
    axlr.text(56, 6.0e-6, "cosine annealed to min_lr", fontsize=6.4, color=MUTED,
              ha="center", va="center")
    axlr.grid(axis="y", color=MUTED, alpha=0.16, lw=0.5)
    axlr.set_axisbelow(True)

    fig.tight_layout(pad=0.4)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "convergence_confound.pdf", bbox_inches="tight")
    fig.savefig(OUT / "convergence_confound.png", dpi=200, bbox_inches="tight")

    print(f"+-30 final {b30[-1][2]:.4f}  +-60 final {b60[-1][2]:.4f}  "
          f"+-90 final {b90[-1][2]:.4f}")
    print(f"crossing epoch {cross}; final-{TAIL} drop {tail_drop:.2f}%")
    print("wrote", OUT / "convergence_confound.pdf")


if __name__ == "__main__":
    main()
