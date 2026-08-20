"""Characterise the HLW ``(theta, rho)`` target distribution and the constant-predictor floor.

Two questions this answers, both of which must be settled *before* four arms are launched.

**1. Is the shared regression loss balanced across the two channels?** ``theta`` is in
radians and ``rho`` in image heights, and there is no reason for them to have comparable
spread. The finetune uses a single ``smooth_l1(beta=0.05)`` over both, so if one channel's
typical magnitude is an order of magnitude smaller it is effectively down-weighted — and
``theta`` is precisely the channel the task was chosen to probe, since it *is* in-plane
orientation while ``rho`` is translation-like. Losing it to a scale mismatch would silently
destroy the specificity control.

**2. What does "no better than nothing" score?** An AUC of 0.4 is uninterpretable on its
own. Predicting the training mean gives the floor, exactly as the mean-predictor floor does
for the pretext metrics: any arm near it has not learned the task.

Targets are computed from image dimensions and metadata only — no pixel decode — via
:meth:`ropart.hlw.HLWDataset.target_for`, so this runs over the training set in seconds
rather than minutes. See that method for why it is shared with ``__getitem__`` rather than
reimplemented.

Usage::

    python -m ropart.scripts.hlw_target_stats --root $WS/hlw --fit train --eval val
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
from PIL import Image

from ropart.hlw import HLWDataset, horizon_auc, horizon_error


def _dims_from_metadata(root) -> dict[str, tuple[int, int]]:
    """``{key: (w, h)}`` from a **v2** ``metadata.csv``, which stores the dimensions.

    HLW v2 rows are ``filename, width, height, x1, y1, x2, y2``, so the image sizes the
    targets need are already in the CSV and no file has to be touched at all. On this NFS
    a header read costs ~13 ms, so the header path spends ~3 h on the 96,617-image v2
    train split; this reads one file in seconds.

    Only v2 (7-column) rows carry dimensions — a v1 root returns an empty mapping and the
    caller must fall back to headers.
    """
    from ropart.hlw import _key           # local: keeps the import surface of this script small
    out: dict[str, tuple[int, int]] = {}
    with open(Path(root) / "metadata.csv", newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 7:
                continue
            try:
                w, h = int(float(row[1])), int(float(row[2]))
            except ValueError:             # header row
                continue
            out[_key(row[0])] = (w, h)
    return out


def _targets(ds: HLWDataset, limit: int = 0, dims: dict | None = None) -> torch.Tensor:
    """``[N, 2]`` of ``(theta, rho)`` for every image in ``ds``, without decoding pixels.

    ``dims`` optionally supplies ``(w, h)`` per image so no file is opened at all — see
    :func:`_dims_from_metadata`. ``theta`` and ``rho`` are invariant to a uniform resize
    (``rho`` is in image heights), so the statistics are the same whether they are taken
    on the raw root with raw dimensions or on the preprocessed root with preprocessed
    ones, up to the sub-pixel rounding the round-trip check bounds at 1.4e-3 image heights.
    """
    from ropart.hlw import _key
    names = ds.names[:limit] if limit else ds.names
    out = []
    for k, name in enumerate(names):
        if dims is not None:
            w, h = dims[_key(name)]
        else:
            with Image.open(ds.root / "images" / name) as im:
                w, h = im.size      # header only; no decode
        out.append(ds.target_for(name, w, h))
        if (k + 1) % 5000 == 0:
            print(f"  {k + 1:,}/{len(names):,}", flush=True)
    return torch.tensor(out, dtype=torch.float64)


def _describe(t: torch.Tensor, name: str, scale: float = 1.0, unit: str = "") -> None:
    v = t * scale
    q = torch.quantile(v, torch.tensor([0.01, 0.25, 0.5, 0.75, 0.99], dtype=v.dtype))
    print(f"{name:>6}{unit:>16} : mean {v.mean():+.4f}  sd {v.std():.4f}  "
          f"min {v.min():+.4f}  max {v.max():+.4f}")
    print(f"{'':>6}{'':>16}   p01 {q[0]:+.4f}  p25 {q[1]:+.4f}  p50 {q[2]:+.4f}  "
          f"p75 {q[3]:+.4f}  p99 {q[4]:+.4f}")


def main() -> None:
    p = argparse.ArgumentParser("HLW target statistics and constant-predictor floor")
    p.add_argument("--root", required=True)
    p.add_argument("--fit", default="train", help="split the constant predictor is fit on")
    p.add_argument("--eval", default="val", help="split the floor is evaluated on")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--limit", type=int, default=0, help="cap images per split (smoke test)")
    p.add_argument("--dims-from-metadata", action="store_true",
                   help="take (w, h) from a v2 metadata.csv instead of image headers "
                        "(instant; requires the 7-column v2 layout, so point --root at "
                        "the raw v2 root)")
    a = p.parse_args()

    fit_ds = HLWDataset(a.root, a.fit, a.size, train=False)
    ev_ds = HLWDataset(a.root, a.eval, a.size, train=False)
    print(f"fit on {a.fit}: {len(fit_ds):,} images   eval on {a.eval}: {len(ev_ds):,}\n")

    dims = None
    if a.dims_from_metadata:
        dims = _dims_from_metadata(a.root)
        if not dims:
            raise SystemExit(f"--dims-from-metadata: {a.root}/metadata.csv has no "
                             f"7-column (v2) rows, so it carries no image dimensions")
        print(f"dimensions from metadata.csv: {len(dims):,} rows (no image headers read)\n")

    fit = _targets(fit_ds, a.limit, dims)
    ev = _targets(ev_ds, a.limit, dims)

    print(f"--- target distribution ({a.fit}) ---")
    _describe(fit[:, 0], "theta", 180.0 / math.pi, "(degrees)")
    _describe(fit[:, 1], "rho", 1.0, "(image heights)")

    # The loss-balance question, stated as the ratio the shared smooth_l1 actually sees.
    s_theta, s_rho = float(fit[:, 0].std()), float(fit[:, 1].std())
    print(f"\nsd(theta) = {s_theta:.5f} rad   sd(rho) = {s_rho:.5f} image heights"
          f"   ratio rho/theta = {s_rho / max(s_theta, 1e-12):.2f}x")
    print("A ratio far from 1 means a shared smooth_l1 would not weight the two channels "
          "equally; theta is the channel the orientation claim rests on.")

    # Paste-ready, because these are module constants in ropart.hlw and a different
    # training set (HLWv2) needs its own. Getting them stale is silent: the model would
    # still train, just against a slightly wrong centre and scale.
    mu_fit = fit.mean(dim=0)
    print(f"\n--- constants for ropart/hlw.py (fit split = {a.fit}) ---")
    print(f"TARGET_MEAN: tuple[float, float] = ({float(mu_fit[0]):.4e}, {float(mu_fit[1]):.6f})")
    print(f"TARGET_STD: tuple[float, float] = ({s_theta:.4e}, {s_rho:.6f})")
    # Same numbers in the form ropart.hlw reads from the environment, so a second training
    # set needs no source edit and no hand-transcription (mis-typing one digit here is
    # exactly the silent fault the override exists to avoid).
    print("\n--- or, without editing the source (HLWv2 etc.) ---")
    print(f"export HLW_TARGET_STATS={float(mu_fit[0]):.4e},{float(mu_fit[1]):.6f},"
          f"{s_theta:.4e},{s_rho:.6f}")

    # Constant predictor: the training mean, which is the best constant under L2.
    mu = fit.mean(dim=0)
    print(f"\n--- constant-predictor floor (predict train mean) on {a.eval} ---")
    print(f"predicting theta = {math.degrees(float(mu[0])):+.4f} deg, "
          f"rho = {float(mu[1]):+.4f}")
    err = horizon_error(
        torch.full((len(ev),), float(mu[0])), torch.full((len(ev),), float(mu[1])),
        ev[:, 0].float(), ev[:, 1].float(), 1.0)
    print(f"AUC @ {0.25} = {horizon_auc(err) * 100:.2f}%   "
          f"(mean err {err.mean():.4f}, median {err.median():.4f})")
    print(f"theta MAE = {math.degrees(float((ev[:, 0] - mu[0]).abs().mean())):.3f} deg   "
          f"rho MAE = {float((ev[:, 1] - mu[1]).abs().mean()):.4f}")
    print("\nAny arm scoring near this AUC has not learned the task.")


if __name__ == "__main__":
    main()
