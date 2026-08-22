"""HLW (Horizon Lines in the Wild) — the Phase-3 orientation-sensitive downstream task.

Why this task. RoPART's pretext supervises **relative in-plane orientation** between
patch pairs. The horizon line's tilt is in-plane image orientation by construction, so
horizon estimation is the natural real-world test of whether that pretext transfers.

The task also carries a **built-in specificity control**, which is the main reason it
was chosen over a single-scalar alternative. Following Workman et al. (BMVC 2016,
arXiv:1604.02129) §3, the horizon is parameterised two ways:

* ``(theta, rho)`` — ``theta`` is the angle the horizon makes with the image x-axis and
  ``rho`` the perpendicular distance from the image centre to the line. **``theta`` is
  orientation; ``rho`` is translation-like.** RoPART should improve ``theta`` and leave
  ``rho`` roughly where the translation-only baseline puts it. A gain concentrated in
  ``theta`` is much stronger evidence than a gain in a single lumped scalar.
* ``(l, r)`` — the vertical offsets at which the horizon meets the left and right image
  edges. This is the form the **standard metric** is defined on.

``rho``, ``l`` and ``r`` are all in units of **image heights** (the paper's convention).

The metric is the maximum distance between the detected and ground-truth horizon within
the image, normalised by image height ("horizon detection error"), reported as the area
under the cumulative error histogram, cut at **0.25**. Because both are straight lines,
that maximum is always attained at an image edge, so it reduces exactly to
``max(|l_pred - l_gt|, |r_pred - r_gt|)`` — which is why ``(l, r)`` is the metric's
natural parameterisation even though the network predicts ``(theta, rho)``.

Reference difficulty: Lezama et al. reach **52.59%** AUC on HLW, against 94.07% on YUD
and 89.57% on ECD. HLW is the hard one.

.. note::
   **The y-axis convention was settled on 2026-08-13: metadata y increases UPWARDS**
   (maths convention), so :data:`Y_AXIS_DOWN` is ``False``. Two independent strands
   agree:

   1. The NG-DSAC reference loader negates y before applying the centre offset
      (``vislearn/ngdsac_horizon``, ``hlw_dataset.py`` lines 77-84: ``gt[1] *= -1``
      then ``gt[1] += yOffset``, likewise for ``gt[3]``).
   2. ``python -m ropart.hlw --verify <hlw_root>`` renders agree on 8/8 test images
      under ``False`` and fail under ``True``. Judge only on images where the line
      sits **far from the image centre** — a sign flip is nearly invisible near the
      centre, which is what made the original ``True`` renders look acceptable.

   A wrong setting here trains and converges perfectly well while producing a
   meaningless AUC, so re-run ``--verify`` if this constant is ever touched.

Dataset layout (v1, 13 GB — same test set as v2, and the version the NG-DSAC baseline
uses)::

    <root>/images/...            image files
    <root>/split/{train,val,test}.txt   plain-text image lists, one path per line
    <root>/metadata.csv          rows: filename, x1, y1, x2, y2

**v2 puts the image dimensions first** — ``filename, width, height, x1, y1, x2, y2``
(7 columns, not 5). :func:`_endpoints_from_row` discriminates on the column count;
see there for why the naive slice is a silent fault rather than a loud one.

Splits (v1, counted from the archive on 2026-08-12 — 100,553 is the paper's full
collection, not what v1 ships): **19,809 images total — 16,906 train, 885 val,
2,018 test**. ``test.txt`` further decomposes into ``test_seen.txt`` (1,300) and
``test_heldout.txt`` (718), a free generalisation control.
Licence: CC BY-NC 4.0, research use only.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ropart.data import MEAN, STD
from ropart.engine import _amp_ctx, _Meters

#: Whether ``metadata.csv`` y-coordinates increase **downwards** (image convention).
#: ``False`` — metadata y is maths-convention (up). Settled 2026-08-13 against the
#: NG-DSAC reference loader and 8/8 ``--verify`` renders; see the module note.
Y_AXIS_DOWN = False

#: Error threshold for the AUC, per the standard protocol.
AUC_THRESHOLD = 0.25

#: Per-channel ``(theta, rho)`` mean and standard deviation of the **HLW v1 train split**,
#: theta in radians and rho in image heights, measured 2026-08-13 over all 16,906 images
#: with ``python -m ropart.scripts.hlw_target_stats --root $WS/hlw224 --fit train``.
#:
#: The regression head predicts **standardised** targets, and these convert back. Three
#: reasons, in order of importance:
#:
#: 1. ``sd(rho) / sd(theta) = 9.70``, so a single ``smooth_l1(beta=0.05)`` over raw targets
#:    treats the two channels very differently: rho residuals sit in the loss's *linear*
#:    regime (gradient +/-1) while theta residuals straddle ``beta`` and spend much of their
#:    mass in the *quadratic* regime (gradient ``x/beta < 1``) — and sink further into it as
#:    training improves theta. Standardising makes ``beta`` mean "0.05 sd" for both.
#: 2. Both outputs are then O(1), which is far better conditioned for the randomly
#:    initialised linear head than asking one unit to emit ~0.06 and another ~0.6.
#: 3. Predicting zero becomes exactly the constant-predictor floor, so an untrained head
#:    starts at the floor rather than somewhere arbitrary.
#:
#: Applied identically to every arm, so it cannot confound the between-arm comparison.
#: **Recompute these for a different training set** (e.g. HLWv2) — ``hlw_target_stats``
#: prints them in paste-ready form.
#:
#: The values below are **HLW v1 train**. They are overridable from the environment via
#: ``HLW_TARGET_STATS`` so that a different training set (HLWv2) can supply its own
#: without editing this file — see :func:`_load_target_stats`.
_TARGET_STATS_V1: tuple[float, float, float, float] = (2.077e-4, 0.237300, 5.8653e-2, 0.569070)


def _load_target_stats() -> tuple[tuple[float, float], tuple[float, float]]:
    """Return ``(TARGET_MEAN, TARGET_STD)``, optionally overridden from the environment.

    ``HLW_TARGET_STATS`` takes four comma-separated floats in the order
    ``mean_theta,mean_rho,std_theta,std_rho``, exactly as ``hlw_target_stats`` prints
    them. It exists because these constants are **training-set specific**: v1 and v2 have
    different target distributions, and training v2 against v1's centre and scale is a
    *silent* fault — the run converges perfectly well against a slightly wrong
    normalisation, so nothing in the logs reveals it.

    Defaulting to the v1 values keeps the completed v1 runs reproducible byte-for-byte.
    """
    raw = os.environ.get("HLW_TARGET_STATS", "").strip()
    if not raw:
        m_theta, m_rho, s_theta, s_rho = _TARGET_STATS_V1
    else:
        parts = [p for p in raw.replace(" ", "").split(",") if p]
        if len(parts) != 4:
            raise ValueError(
                "HLW_TARGET_STATS needs 4 comma-separated floats "
                f"(mean_theta,mean_rho,std_theta,std_rho), got {raw!r}"
            )
        m_theta, m_rho, s_theta, s_rho = (float(p) for p in parts)
    if s_theta <= 0.0 or s_rho <= 0.0:
        raise ValueError(
            f"HLW_TARGET_STATS standard deviations must be > 0, got {s_theta}, {s_rho}"
        )
    return (m_theta, m_rho), (s_theta, s_rho)


TARGET_MEAN, TARGET_STD = _load_target_stats()


# --------------------------------------------------------------------------- #
# Horizon geometry
#
# Everything below works on endpoints in a **centred pixel frame** (origin at the
# image centre, x rightwards, y downwards). Endpoints — not (theta, rho) — are the
# canonical internal representation, because a crop or resize is then a plain affine
# map on two points; transforming (theta, rho) directly is far easier to get subtly
# wrong, and a silent error there would look like a plausible-but-wrong result.
# --------------------------------------------------------------------------- #


def endpoints_to_theta_rho(x1: float, y1: float, x2: float, y2: float,
                           height: float) -> tuple[float, float]:
    """Convert centred-frame endpoints to ``(theta, rho)``.

    Args:
        x1, y1, x2, y2: horizon endpoints in centred pixel coordinates.
        height: image height in pixels, used to put ``rho`` in image-height units.

    Returns:
        ``(theta, rho)`` with ``theta`` in radians wrapped to ``(-pi/2, pi/2]`` (a line
        is undirected, so the two endpoint orderings must give the same answer) and
        ``rho`` the signed perpendicular offset from the centre, in image heights.
    """
    theta = math.atan2(y2 - y1, x2 - x1)
    # Undirected: fold the direction onto a half-turn so swapping the endpoints is a
    # no-op. Without this the target flips sign for half the dataset.
    if theta > math.pi / 2:
        theta -= math.pi
    elif theta <= -math.pi / 2:
        theta += math.pi
    # Signed distance from origin along the line normal (-sin, cos).
    rho = (-x1 * math.sin(theta) + y1 * math.cos(theta)) / height
    return theta, rho


def theta_rho_to_lr(theta, rho, width_over_height):
    """Convert ``(theta, rho)`` to left/right edge intercepts ``(l, r)``.

    The horizon satisfies ``-x sin(theta) + y cos(theta) = rho``, so at the image edges
    ``x = -+ W/2`` the height is ``y = (rho +- (W/2) sin(theta)) / cos(theta)``.

    Args:
        theta: horizon angle in radians (tensor or float).
        rho: perpendicular offset in image heights (tensor or float).
        width_over_height: image aspect ratio ``W / H``; the half-width in image-height
            units is therefore ``width_over_height / 2``.

    Returns:
        ``(l, r)`` in image heights.
    """
    half_w = width_over_height / 2.0
    if isinstance(theta, torch.Tensor):
        cos_t = torch.cos(theta).clamp(min=1e-6)
        sin_t = torch.sin(theta)
    else:
        cos_t = max(math.cos(theta), 1e-6)
        sin_t = math.sin(theta)
    return (rho - half_w * sin_t) / cos_t, (rho + half_w * sin_t) / cos_t


def transform_endpoints(pts, crop_box, scale):
    """Map endpoints through a crop then a uniform resize.

    Args:
        pts: ``(x1, y1, x2, y2)`` in **top-left origin** pixel coordinates.
        crop_box: ``(left, top, w, h)`` of the crop, same frame.
        scale: factor applied after cropping (resize).

    Returns:
        Endpoints in the output image's **centred** frame, plus the output ``(W, H)``.

    A crop that moves the image centre moves the horizon relative to it, so this must
    run per sample — inheriting the original ``(theta, rho)`` after cropping is the
    single most likely silent bug in this pipeline.
    """
    left, top, cw, ch = crop_box
    out_w, out_h = cw * scale, ch * scale
    x1, y1, x2, y2 = pts
    # Into crop-local top-left coords, then scale, then re-centre.
    tx1 = (x1 - left) * scale - out_w / 2.0
    ty1 = (y1 - top) * scale - out_h / 2.0
    tx2 = (x2 - left) * scale - out_w / 2.0
    ty2 = (y2 - top) * scale - out_h / 2.0
    return (tx1, ty1, tx2, ty2), (out_w, out_h)


# --------------------------------------------------------------------------- #
# Metric
# --------------------------------------------------------------------------- #


def horizon_error(theta_p, rho_p, theta_t, rho_t, width_over_height) -> torch.Tensor:
    """Horizon detection error: max edge-to-edge gap, in image heights.

    Both horizons are straight lines, so their maximum separation across the image width
    is attained at an image edge — the max over the two edge intercepts is therefore
    exact, not an approximation.
    """
    lp, rp = theta_rho_to_lr(theta_p, rho_p, width_over_height)
    lt, rt = theta_rho_to_lr(theta_t, rho_t, width_over_height)
    return torch.maximum((lp - lt).abs(), (rp - rt).abs())


def horizon_auc(errors: torch.Tensor, threshold: float = AUC_THRESHOLD) -> float:
    """Area under the cumulative histogram of horizon errors, cut at ``threshold``.

    Normalised so a perfect predictor scores 1.0 and one that never falls below the
    threshold scores 0.0. Computed exactly rather than on a grid: the cumulative
    histogram is a step function, so the area is a closed-form sum over sorted errors.
    """
    if errors.numel() == 0:
        return 0.0
    e = torch.sort(errors.flatten().float()).values
    e = e[e < threshold]
    n = errors.numel()
    if e.numel() == 0:
        return 0.0
    # Each error e_i contributes (threshold - e_i) of width at height 1/n.
    return float((threshold - e).sum() / (threshold * n))


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


class HLWDataset(torch.utils.data.Dataset):
    """HLW images with ``(theta, rho)`` targets.

    Preprocessing follows the paper's finding that **aspect ratio matters** for this
    geometric task: they take a maximal square centre crop and report that reshaping the
    image to square was "far less accurate". We therefore resize the short side and
    centre-crop to ``img_size`` — never a squash-resize — and transform the endpoints
    through exactly the same crop so the label stays consistent with the pixels.
    """

    def __init__(self, root: str | Path, split: str, img_size: int = 224,
                 train: bool = False):
        self.root = Path(root)
        self.img_size = img_size
        self.train = train
        self.meta = _read_metadata(self.root / "metadata.csv")
        listing = self.root / "split" / f"{split}.txt"
        names = [ln.strip() for ln in listing.read_text().splitlines() if ln.strip()]
        # Keep only entries we have a horizon for; report rather than silently drop.
        self.names = [n for n in names if _key(n) in self.meta]
        missing = len(names) - len(self.names)
        if missing:
            print(f"hlw[{split}]: {missing}/{len(names)} images have no metadata row — skipped")
        self.normalize = transforms.Normalize(MEAN, STD)

    def __len__(self) -> int:
        return len(self.names)

    def _geometry(self, w: int, h: int) -> tuple[float, float, float, float, float]:
        """Resize-and-centre-crop geometry for a ``w x h`` image.

        Returns ``(scale, sw, sh, left, top)``: the short-side scale, the resized
        dimensions, and the top-left corner of the crop box within them.
        """
        scale = self.img_size / min(w, h)
        sw, sh = w * scale, h * scale
        return scale, sw, sh, (sw - self.img_size) / 2.0, (sh - self.img_size) / 2.0

    def target_for(self, name: str, w: int, h: int, flip: bool = False) -> tuple[float, float]:
        """``(theta, rho)`` for one image, from its **dimensions and metadata alone**.

        Split out of :meth:`__getitem__` so the label distribution can be characterised
        without decoding pixels — a full decode of the training set costs minutes, a
        header read costs seconds — while keeping exactly one copy of the label maths.
        Decoding and this must never disagree, so they share the code rather than
        mirroring it.

        Args:
            name: listing entry, used to look up the metadata row.
            w, h: dimensions of the **original** image, in pixels.
            flip: whether the horizontal train-time flip was applied to the pixels.
        """
        x1, y1, x2, y2 = self.meta[_key(name)]
        # metadata is centred; move to top-left origin so the crop maths is uniform.
        sign = 1.0 if Y_AXIS_DOWN else -1.0
        pts = (x1 + w / 2.0, sign * y1 + h / 2.0, x2 + w / 2.0, sign * y2 + h / 2.0)
        scale, _, _, left, top = self._geometry(w, h)
        # Endpoints through the same resize-then-crop, in the output centre frame.
        cpts, (_, oh) = transform_endpoints(
            (pts[0] * scale, pts[1] * scale, pts[2] * scale, pts[3] * scale),
            (left, top, self.img_size, self.img_size), 1.0)
        if flip:                                     # horizontal flip: x -> -x
            cpts = (-cpts[0], cpts[1], -cpts[2], cpts[3])
        return endpoints_to_theta_rho(*cpts, height=oh)

    def __getitem__(self, i: int):
        name = self.names[i]
        img = Image.open(self.root / "images" / name).convert("RGB")
        w, h = img.size
        _, sw, sh, left, top = self._geometry(w, h)
        img = img.resize((max(1, round(sw)), max(1, round(sh))), Image.BILINEAR)
        img = img.crop((round(left), round(top),
                        round(left) + self.img_size, round(top) + self.img_size))

        flip = bool(self.train and torch.rand(()) < 0.5)
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        theta, rho = self.target_for(name, w, h, flip=flip)
        x = self.normalize(transforms.functional.to_tensor(img))
        return x, torch.tensor([theta, rho], dtype=torch.float32)


def _key(name: str) -> str:
    """Normalise a listing entry to its metadata key (paths vs bare names vary)."""
    return str(name).replace("\\", "/").lstrip("./")


def _endpoints_from_row(row: list[str]) -> tuple[float, float, float, float] | None:
    """The four endpoint columns of one ``metadata.csv`` row, for **both** HLW layouts.

    HLW **v1** rows are ``filename, x1, y1, x2, y2`` (5 columns); HLW **v2** rows carry
    the image dimensions first — ``filename, width, height, x1, y1, x2, y2`` (7 columns).
    Slicing ``row[1:5]`` unconditionally therefore reads ``(width, height, x1, y1)`` on v2
    and drops ``y2`` entirely. That is a **silent** fault of exactly the kind this task is
    full of: the run converges perfectly well against a meaningless target. Measured on
    v2 train, the bad parse gives ``theta`` mean **-40.8 deg** against v1's **0.01 deg** —
    an average 40-degree horizon tilt, which is how it was caught.

    The column count is the only thing that distinguishes the two layouts, so discriminate
    on it. Returns ``None`` for a header row, which the callers skip.
    """
    cols = row[3:7] if len(row) >= 7 else row[1:5]
    try:
        return tuple(float(v) for v in cols)  # type: ignore[return-value]
    except ValueError:      # header row
        return None


def _read_metadata(path: Path) -> dict[str, tuple[float, float, float, float]]:
    out: dict[str, tuple[float, float, float, float]] = {}
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 5:
                continue
            vals = _endpoints_from_row(row)
            if vals is None:
                continue
            out[_key(row[0])] = vals
    return out


def build_hlw_datasets(root: str, *, img_size: int = 224, val_split: str = "val"):
    """Train/val datasets for the finetune loop.

    Per-epoch validation uses HLW's **``val``** split (885 images — the earlier "525"
    figure was wrong; counted from the archive 2026-08-12). ``test`` is deliberately
    *not* tracked per epoch: reporting "best epoch on test" over 100 evaluations is
    test-set peeking. Score ``test`` once, at the end, with the selected checkpoint.

    Args:
        root: dataset root containing ``images/``, ``split/`` and ``metadata.csv``.
        img_size: side length of the square centre crop.
        val_split: split to validate on per epoch. Pass ``"test"`` only for the
            single final scoring run.
    """
    return (HLWDataset(root, "train", img_size, train=True),
            HLWDataset(root, val_split, img_size, train=False))


# --------------------------------------------------------------------------- #
# Train / eval epoch
# --------------------------------------------------------------------------- #


def horizon_run_epoch(model, loader, device, *, optimizer=None, scaler=None,
                      amp_dtype: torch.dtype = torch.float16,
                      max_steps: int | None = None,
                      per_image: bool = False) -> dict[str, float | torch.Tensor]:
    """One horizon-regression epoch (train if ``optimizer`` given, else eval).

    Mirrors :func:`ropart.engine.cls_run_epoch` so the finetune loop is identical
    across the two Phase-3 tasks.

    **The model predicts standardised targets** (see :data:`TARGET_MEAN` /
    :data:`TARGET_STD`); predictions are converted back to raw units before any metric, so
    everything reported here — and the AUC in particular — stays in the protocol's units
    and is unaffected by the standardisation.

    Reports ``auc`` plus ``theta_mae`` in **degrees** and ``rho_mae`` separately. **That
    split is not a nicety — it is the result.** The AUC is structurally dominated by rho:
    with a square crop (``W/H == 1``) the edge offsets are ``l, r = rho -/+ tan(theta)/2``,
    so a theta error enters the AUC at *half* weight while a rho error enters at full
    weight. At the constant-predictor floor that is 0.0127 vs 0.3269 image heights — rho
    outweighs theta by ~26x. Since theta is the in-plane orientation channel the whole
    task was chosen to probe, **``theta_mae`` is the headline for the orientation claim**
    and ``auc`` is reported for comparability with the published protocol.

    Args:
        per_image: also retain the **per-sample** errors, returned under the keys
            ``theta_err`` (degrees), ``rho_err`` and ``horizon_err`` as 1-D tensors in
            loader order. Off by default so the training loop is unchanged. Needed for
            a *paired* comparison between arms: the aggregate MAEs alone only support an
            unpaired test, which on this task is far weaker — the arms are scored on the
            same images and their errors are strongly correlated, so the shared component
            cancels once the comparison is done per image. ``theta_err`` and ``rho_err``
            average exactly to the reported ``theta_mae`` / ``rho_mae``; ``horizon_err``
            is the quantity :func:`horizon_auc` consumes.
    """
    train = optimizer is not None
    model.train(train)
    meters = _Meters()
    errs: list[torch.Tensor] = []
    per_theta: list[torch.Tensor] = []
    per_rho: list[torch.Tensor] = []
    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    t_mean = torch.tensor(TARGET_MEAN, device=device, dtype=torch.float32)
    t_std = torch.tensor(TARGET_STD, device=device, dtype=torch.float32)

    with grad_ctx:
        for step, (images, target) in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            bs = images.shape[0]
            with _amp_ctx(device, scaler is not None, amp_dtype):
                pred = model.forward_classify(images)      # (b, 2), standardised
                loss = F.smooth_l1_loss(pred, (target - t_mean) / t_std, beta=0.05)
            if train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            with torch.no_grad():
                # Back to raw units before any metric — in float32, since under AMP `pred`
                # may be float16 and rho spans ~[-6, 12], which is fine in fp16 but the
                # errors we care about are ~1e-3 and would be quantised away.
                p = pred.float() * t_std + t_mean
                t = target.float()
                # Square crop, so W/H == 1 in the network's frame.
                e = horizon_error(p[:, 0], p[:, 1], t[:, 0], t[:, 1], 1.0)
                errs.append(e.detach().cpu())
                d_theta = (p[:, 0] - t[:, 0]).abs()
                d_rho = (p[:, 1] - t[:, 1]).abs()
                if per_image:
                    per_theta.append(d_theta.detach().cpu() * 180.0 / math.pi)
                    per_rho.append(d_rho.detach().cpu())
                meters.update(bs, loss=loss.item(),
                              theta_mae=float(d_theta.mean()) * 180.0 / math.pi,
                              rho_mae=float(d_rho.mean()))

    out = meters.summary()
    out["auc"] = horizon_auc(torch.cat(errs)) if errs else 0.0
    if per_image:
        out["theta_err"] = torch.cat(per_theta) if per_theta else torch.empty(0)
        out["rho_err"] = torch.cat(per_rho) if per_rho else torch.empty(0)
        out["horizon_err"] = torch.cat(errs) if errs else torch.empty(0)
    return out


# --------------------------------------------------------------------------- #
# Convention check (run this once, before trusting any number)
# --------------------------------------------------------------------------- #


def _verify(root: str, n: int = 12, out_dir: str = "./hlw_verify"):
    """Render ``n`` samples with the parsed horizon drawn on, to settle the y-axis
    convention. If the drawn line does not sit on the visible horizon, flip
    :data:`Y_AXIS_DOWN` and re-run."""
    from PIL import ImageDraw

    ds = HLWDataset(root, "test", img_size=224, train=False)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    inv_mean = torch.tensor(MEAN).view(3, 1, 1)
    inv_std = torch.tensor(STD).view(3, 1, 1)
    for i in range(min(n, len(ds))):
        x, tgt = ds[i]
        img = transforms.functional.to_pil_image((x * inv_std + inv_mean).clamp(0, 1))
        theta, rho = float(tgt[0]), float(tgt[1])
        l, r = theta_rho_to_lr(theta, rho, 1.0)
        s = ds.img_size
        draw = ImageDraw.Draw(img)
        draw.line([(0, l * s + s / 2), (s, r * s + s / 2)], fill=(255, 0, 0), width=3)
        img.save(Path(out_dir) / f"{i:03d}_{Path(ds.names[i]).stem}.png")
    print(f"wrote {min(n, len(ds))} annotated samples to {out_dir}/ — "
          f"check the red line lies on the horizon (Y_AXIS_DOWN={Y_AXIS_DOWN})")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser("HLW helpers")
    p.add_argument("--verify", metavar="HLW_ROOT",
                   help="render samples with the parsed horizon drawn, to check the "
                        "coordinate convention before any training run")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--out", default="./hlw_verify")
    a = p.parse_args()
    if a.verify:
        _verify(a.verify, a.n, a.out)
    else:
        p.print_help()
