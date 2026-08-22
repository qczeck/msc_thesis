"""Paired comparison of two HLW arms on their per-image errors.

Why paired. The arms are scored on **the same images**, so most of the variation in an
error is a property of the image (a textureless sky is hard for every arm) rather than of
the arm. Comparing two aggregate MAEs throws that away and leaves an *unpaired* standard
error dominated by image difficulty — on the 525-image v2 val split that is ~0.058 deg,
which is wider than every between-arm gap in the study except ``ch2``'s. Differencing per
image cancels the shared component, so the test sees only what actually differs between
the arms.

Two instruments, both distribution-free, because the error distribution is non-negative
and heavy-tailed and a t-interval on it would be optimistic:

* a **paired bootstrap** over images — resample images with replacement, recompute the
  mean difference, report the percentile CI. This is the headline: it puts an interval on
  the same quantity the results tables report (a difference of MAEs).
* the **Wilcoxon signed-rank** test on the per-image differences, as a p-value that does
  not depend on the mean being the right summary. Uses the normal approximation with tie
  and zero corrections, which at n in the thousands is standard.

Reads the CSVs written by :mod:`ropart.scripts.score_horizon`. Files are matched **by
filename**, not by row order, and a mismatch is fatal — two arms scored on different
splits would otherwise difference cleanly and produce a confident wrong answer.

Usage::

    python -m ropart.scripts.paired_horizon_test \\
        --baseline out/hlw_ft_base_p32_v2/scores_test.csv \\
        --arm      out/hlw_ft_w30_p32_v2/scores_test.csv \\
        --column theta_err_deg
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def read_scores(path: str, column: str) -> dict[str, float]:
    """``{filename: error}`` from a ``score_horizon`` CSV."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path} is empty")
    if column not in rows[0]:
        raise SystemExit(f"{path} has no column {column!r} (has {list(rows[0])})")
    return {r["filename"]: float(r[column]) for r in rows}


def paired_bootstrap(diff: np.ndarray, n_boot: int, alpha: float,
                     rng: np.random.Generator) -> tuple[float, float]:
    """Percentile CI for the mean of ``diff``, resampling images with replacement."""
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def wilcoxon_signed_rank(diff: np.ndarray) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank on ``diff``; returns ``(z, p)``.

    Normal approximation with tie and zero corrections. Implemented here rather than
    pulled from scipy: scipy is not a project dependency and this is the only test that
    would need it. At n in the thousands
    the approximation is what scipy itself defaults to.
    """
    d = diff[diff != 0.0]
    n = len(d)
    if n == 0:
        return 0.0, 1.0
    order = np.argsort(np.abs(d), kind="stable")
    ranks = np.empty(n, dtype=float)
    a = np.abs(d)[order]
    i = 0
    while i < n:                                   # average ranks within ties
        j = i
        while j + 1 < n and a[j + 1] == a[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    signed = np.sign(d[order]) * ranks
    w = signed[signed > 0].sum()
    mean_w = n * (n + 1) / 4.0
    # tie correction on the variance
    _, counts = np.unique(a, return_counts=True)
    tie_term = ((counts ** 3 - counts).sum()) / 48.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term
    if var_w <= 0:
        return 0.0, 1.0
    z = (w - mean_w) / math.sqrt(var_w)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return float(z), float(p)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", required=True, help="reference arm's scores CSV")
    p.add_argument("--arm", required=True, help="arm under test")
    p.add_argument("--column", default="theta_err_deg",
                   choices=["theta_err_deg", "rho_err", "horizon_err"],
                   help="theta_err_deg is the headline for the orientation claim; the "
                        "AUC's horizon_err is structurally rho-dominated (~26x)")
    p.add_argument("--n-boot", default=20000, type=int)
    p.add_argument("--alpha", default=0.05, type=float)
    p.add_argument("--seed", default=0, type=int)
    args = p.parse_args()

    base = read_scores(args.baseline, args.column)
    arm = read_scores(args.arm, args.column)
    if set(base) != set(arm):
        raise SystemExit(
            f"the two CSVs cover different images ({len(base)} vs {len(arm)}, "
            f"{len(set(base) ^ set(arm))} not shared) — refusing to pair them")

    names = sorted(base)
    b = np.array([base[n] for n in names])
    a = np.array([arm[n] for n in names])
    diff = a - b                                   # negative => the arm is better

    lo, hi = paired_bootstrap(diff, args.n_boot, args.alpha, np.random.default_rng(args.seed))
    z, pval = wilcoxon_signed_rank(diff)
    better = int((diff < 0).sum())

    conf = int(round((1 - args.alpha) * 100))
    print(f"column      : {args.column}   n = {len(names)} paired images")
    print(f"baseline    : {Path(args.baseline).parent.name}   mean {b.mean():.4f}")
    print(f"arm         : {Path(args.arm).parent.name}   mean {a.mean():.4f}")
    print(f"mean diff   : {diff.mean():+.4f}   (negative = arm better)")
    print(f"{conf}% CI      : [{lo:+.4f}, {hi:+.4f}]   paired bootstrap, {args.n_boot} resamples")
    print(f"median diff : {np.median(diff):+.4f}")
    print(f"arm better  : {better}/{len(names)} images ({100 * better / len(names):.1f}%)")
    print(f"Wilcoxon    : z = {z:+.3f}   p = {pval:.4g}  (two-sided, normal approx)")
    verdict = "excludes 0" if lo > 0 or hi < 0 else "includes 0 — no detectable difference"
    print(f"verdict     : the {conf}% interval {verdict}")


if __name__ == "__main__":
    main()
