"""Verify that :mod:`ropart.scripts.preprocess_hlw` is label-preserving.

Why this exists. The preprocessing pass resizes every image and rescales every label.
If the two fall out of step the result is not a crash but a *plausible-looking* one:
the model trains, the loss falls, and the AUC is meaningless. The crop/resize label
transform is the single most likely source of such a silent bug, so it gets an explicit
check rather than a visual one.

**The invariant.** A preprocessed image plus its rescaled label must yield the *same*
``(theta, rho)`` target as the original image plus the original label. Both roots go
through the identical :class:`ropart.hlw.HLWDataset` code path, so any disagreement is
attributable to the offline resize, not to the loader.

**Expected behaviour, measured on v1 (2026-08-13, 159 images).** ``theta`` agrees
*exactly* — to all printed digits — because a uniform scale and a centre crop are both
direction-preserving, and ``theta`` depends only on the direction of the endpoint
difference. All the rounding lands in ``rho``, which is an offset: observed mean 1.5e-4,
max 1.4e-3 image heights, i.e. **0.31 px at 224** — consistent with ``preprocess_hlw``
rounding the resized dimensions to whole pixels while reporting the *short-side* scale
(≤0.5 px on the long side), plus ``HLWDataset`` rounding its own crop box.

So the tolerances below are set from that mechanism, not picked round: ``theta`` should be
essentially exact, and ``rho`` should stay well inside a pixel or two. For scale, the AUC
threshold is **0.25 image heights**, so a 1.4e-3 discrepancy is 0.56% of it — negligible.
A failure here means something structural (a dropped offset, a wrong axis, an anisotropic
resize), not accumulated rounding.

Usage::

    python -m ropart.scripts.check_hlw_preprocess \\
        --raw $WS/hlw --pre $WS/hlw224_smoke --split train --n 200
"""

from __future__ import annotations

import argparse
import math

import torch

from ropart.hlw import HLWDataset


def main() -> None:
    p = argparse.ArgumentParser("Check HLW preprocessing is label-preserving")
    p.add_argument("--raw", required=True, help="original HLW root")
    p.add_argument("--pre", required=True, help="preprocessed root")
    p.add_argument("--split", default="train")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--n", type=int, default=200, help="max images to compare")
    # Set from the rounding mechanism, not picked round — see the module docstring.
    # 0.005 image heights is ~1.1 px at 224 and 2% of the 0.25 AUC threshold.
    p.add_argument("--theta-tol-deg", type=float, default=0.01)
    p.add_argument("--rho-tol", type=float, default=5e-3)
    a = p.parse_args()

    # train=False: the random horizontal flip would otherwise desynchronise the two
    # datasets and turn this into a test of torch's RNG.
    raw = HLWDataset(a.raw, a.split, a.size, train=False)
    pre = HLWDataset(a.pre, a.split, a.size, train=False)

    raw_idx = {n: i for i, n in enumerate(raw.names)}
    shared = [n for n in pre.names if n in raw_idx][: a.n]
    print(f"raw: {len(raw.names):,} images   preprocessed: {len(pre.names):,}   "
          f"comparing: {len(shared):,}")
    if not shared:
        raise SystemExit("no overlapping filenames between the two roots")

    d_theta: list[float] = []
    d_rho: list[float] = []
    worst = (0.0, "")
    for j, name in enumerate(shared):
        _, t_raw = raw[raw_idx[name]]
        _, t_pre = pre[pre.names.index(name)]
        dt = abs(float(t_raw[0] - t_pre[0]))
        dr = abs(float(t_raw[1] - t_pre[1]))
        d_theta.append(dt)
        d_rho.append(dr)
        if math.degrees(dt) > worst[0]:
            worst = (math.degrees(dt), name)

    theta_deg = torch.tensor(d_theta) * 180.0 / math.pi
    rho = torch.tensor(d_rho)
    print(f"\n|dtheta| deg : mean {theta_deg.mean():.5f}  "
          f"p99 {theta_deg.quantile(0.99):.5f}  max {theta_deg.max():.5f}")
    print(f"|drho|       : mean {rho.mean():.6f}  "
          f"p99 {rho.quantile(0.99):.6f}  max {rho.max():.6f}")
    print(f"worst theta  : {worst[0]:.5f} deg on {worst[1]}")

    ok = bool(theta_deg.max() < a.theta_tol_deg and rho.max() < a.rho_tol)
    print(f"\n{'PASS' if ok else 'FAIL'} "
          f"(tolerances: theta < {a.theta_tol_deg} deg, rho < {a.rho_tol})")
    if not ok:
        raise SystemExit("preprocessing is NOT label-preserving — do not train on it")


if __name__ == "__main__":
    main()
