"""Between-seed analysis of the HLW horizon arms — the pre-registered primary test.

**What this answers, and why the per-seed paired test does not.**
:mod:`ropart.scripts.paired_horizon_test` compares two arms on the same images and so
removes *image* sampling noise. It says nothing about **finetune stochasticity** — data
ordering, augmentation draws and head initialisation all move with ``--seed``, and a single
seed cannot distinguish "this pretext objective transfers better" from "this run drew a
lucky shuffle". That is the question this file exists for, and it is the one that decides
whether the RoPART result is reportable: on seed 0 alone the headline comparison sits at
Wilcoxon p = 0.027 with a bootstrap CI of [-0.0626, +0.0006], i.e. exactly where one extra
seed could go either way.

**The pre-registered primary statistic** (pre-registered 2026-08-22, before any
replicate finished): the per-image error difference ``arm - baseline`` is averaged within
each seed, giving one number per seed; the report is the mean of those across seeds with a
Student-t interval on ``n_seeds - 1`` degrees of freedom. The seed is the unit of
replication because the seed is what was randomised.

**Decision rule, fixed in advance:** the interval excluding zero is the claim; the interval
including zero is "not established", *regardless of the per-seed p-values*. Per-seed
statistics are reported as a consistency check and are secondary — a 4/5 sign split with one
significant seed is not a result, and pre-committing to that is the point.

A hierarchical bootstrap (resample seeds, then images within seed) is reported alongside as
a robustness check. It is **not** the primary: with five seeds the seed-level variance has
4 df either way, and a bootstrap over five units is not more trustworthy than the t-interval
— it is shown so that a large disagreement between the two is visible rather than hidden.

Usage::

    python -m ropart.scripts.seed_replication_test \\
        --out-root /vol/gpudata/msk123-ropart-in100/out \\
        --baseline base --arm w30 --seeds 0 1 2 3 4 --tag v2 --column theta_err_deg
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from ropart.scripts.paired_horizon_test import (
    paired_bootstrap,
    read_scores,
    wilcoxon_signed_rank,
)

# Two-sided 97.5% Student-t quantiles by degrees of freedom. Tabulated rather than computed:
# scipy is not a project dependency and an inverse
# incomplete beta for this one number would be more code than table. df > 30 falls back to
# the normal quantile, where the difference is under 5%.
_T_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
    23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
    30: 2.042,
}


def t_critical_975(df: int) -> float:
    """Two-sided 95% critical value on ``df`` degrees of freedom."""
    if df < 1:
        raise ValueError("need at least 2 seeds for an interval")
    return _T_975.get(df, 1.960)


def run_dir(out_root: str, arm: str, seed: int, tag: str) -> Path:
    """Output directory for one arm/seed, matching ``run_horizon_doc.sh``'s naming.

    Seed 0 keeps the original unsuffixed name so the 2026-08-20 runs stay addressable;
    every other seed gets ``_s<N>``. **This rule is duplicated from the shell script**, so
    if that naming ever changes both must change — ``tests/test_hlw.py`` pins it.
    """
    name = f"hlw_ft_{arm}_p32" if seed == 0 else f"hlw_ft_{arm}_p32_s{seed}"
    return Path(out_root) / (f"{name}_{tag}" if tag else name)


def per_seed_differences(out_root: str, baseline: str, arm: str, seeds: list[int],
                         tag: str, split: str, column: str) -> tuple[np.ndarray, list[np.ndarray]]:
    """``(per_seed_mean_diff, per_seed_image_diffs)``, paired on filenames within each seed."""
    means, diffs = [], []
    for s in seeds:
        b_csv = run_dir(out_root, baseline, s, tag) / f"scores_{split}.csv"
        a_csv = run_dir(out_root, arm, s, tag) / f"scores_{split}.csv"
        for p in (b_csv, a_csv):
            if not p.exists():
                raise SystemExit(f"missing {p} — has that arm been scored on '{split}'?")
        b, a = read_scores(str(b_csv), column), read_scores(str(a_csv), column)
        if set(b) != set(a):
            raise SystemExit(f"seed {s}: the two CSVs cover different images — refusing to pair")
        names = sorted(b)
        d = np.array([a[n] for n in names]) - np.array([b[n] for n in names])
        diffs.append(d)
        means.append(d.mean())
    return np.array(means), diffs


def hierarchical_bootstrap(diffs: list[np.ndarray], n_boot: int,
                           rng: np.random.Generator) -> tuple[float, float]:
    """Percentile CI resampling **seeds** with replacement, then images within each seed."""
    n_seeds = len(diffs)
    out = np.empty(n_boot)
    for i in range(n_boot):
        picked = rng.integers(0, n_seeds, size=n_seeds)
        vals = [diffs[k][rng.integers(0, len(diffs[k]), size=len(diffs[k]))].mean()
                for k in picked]
        out[i] = float(np.mean(vals))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", required=True)
    p.add_argument("--baseline", default="base")
    p.add_argument("--arm", required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--tag", default="v2")
    p.add_argument("--split", default="test")
    p.add_argument("--column", default="theta_err_deg",
                   choices=["theta_err_deg", "rho_err", "horizon_err"])
    p.add_argument("--n-boot", default=20000, type=int)
    p.add_argument("--seed", default=0, type=int, help="RNG seed for the bootstraps")
    args = p.parse_args()

    means, diffs = per_seed_differences(args.out_root, args.baseline, args.arm,
                                        args.seeds, args.tag, args.split, args.column)
    n = len(means)
    if n < 2:
        raise SystemExit("need at least 2 seeds")

    mean = float(means.mean())
    sd = float(means.std(ddof=1))
    se = sd / math.sqrt(n)
    tcrit = t_critical_975(n - 1)
    lo, hi = mean - tcrit * se, mean + tcrit * se

    print(f"comparison  : {args.arm} - {args.baseline}   column={args.column} "
          f"split={args.split}   {n} seeds {args.seeds}")
    print()
    print("--- per seed (secondary: a consistency check, not the test) ---")
    print(f"{'seed':>5} {'mean diff':>11} {'CI low':>9} {'CI high':>9} {'Wilcoxon p':>11} {'n img':>7}")
    rng = np.random.default_rng(args.seed)
    for s, d in zip(args.seeds, diffs):
        blo, bhi = paired_bootstrap(d, args.n_boot, 0.05, np.random.default_rng(args.seed))
        _, pv = wilcoxon_signed_rank(d)
        print(f"{s:>5} {d.mean():>+11.4f} {blo:>+9.4f} {bhi:>+9.4f} {pv:>11.4g} {len(d):>7}")

    print()
    print("--- PRIMARY: between-seed (the pre-registered test) ---")
    print(f"per-seed means : {', '.join(f'{m:+.4f}' for m in means)}")
    print(f"mean           : {mean:+.4f}   (negative = {args.arm} better)")
    print(f"between-seed SD: {sd:.4f}   SE {se:.4f}   t crit ({n - 1} df) {tcrit:.3f}")
    print(f"95% CI         : [{lo:+.4f}, {hi:+.4f}]")
    same_sign = int((means < 0).sum()) if mean < 0 else int((means > 0).sum())
    print(f"sign agreement : {same_sign}/{n} seeds agree with the mean's direction")

    hlo, hhi = hierarchical_bootstrap(diffs, max(2000, args.n_boot // 10),
                                      np.random.default_rng(args.seed + 1))
    print(f"hierarchical   : [{hlo:+.4f}, {hhi:+.4f}]   (robustness check, seeds+images)")

    print()
    excludes = lo > 0 or hi < 0
    print(f"VERDICT        : the 95% between-seed interval "
          f"{'EXCLUDES zero' if excludes else 'includes zero'} -> "
          f"{'effect established at this n' if excludes else 'NOT established'}")
    print("                 (Bonferroni over 3 arms: alpha 0.0167, i.e. a wider interval; "
          "per the 2026-08-22 pre-registration)")


if __name__ == "__main__":
    main()
