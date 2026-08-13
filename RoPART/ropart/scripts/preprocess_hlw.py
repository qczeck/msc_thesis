"""Pre-resize HLW to the training resolution so finetune epochs are not NFS-bound.

Why this exists. HLWv2 is ~69 GB of full-resolution Flickr photographs. The lab boxes
have 62 GB of RAM, so a 69 GB training set cannot sit in the page cache: every epoch
would re-read it from NFS at ~10 MB/s and the run would be I/O-bound rather than
GPU-bound. (IN-100 was ~13 GB and *did* cache — which is exactly why its epoch 0 took
~370 s and every later epoch 142 s.) Resizing once, offline, to the training resolution
takes the training set to ~10 GB, which caches comfortably.

**The transform here is deliberately only a uniform resize — no crop.** That is what
makes this script safe to run *before* the ``Y_AXIS_DOWN`` question is settled:

* A uniform scale is **sign-agnostic**. Endpoints map ``(x, y) -> (s·x, s·y)`` whether y
  increases up or down, so rescaled metadata is correct under either convention.
* A **crop is not**: it adds ``h/2`` offsets, whose sign depends on the convention. Bake
  a crop in now and a later flip of ``Y_AXIS_DOWN`` silently invalidates every label,
  with 500k images to redo.

So the centre crop stays where it is, in :class:`ropart.hlw.HLWDataset`, applied at load
time to an already-small image (cheap). The dataset needs **no changes**: for a
preprocessed image ``min(w, h) == img_size``, so its internal ``scale`` is 1.0 and the
crop maths passes through unaltered. Point it at the preprocessed root and it works.

Two deliberate non-transformations, for consistency with the load-time path:

* **No EXIF auto-rotation.** ``PIL.Image.open`` does not apply EXIF orientation and
  neither does ``HLWDataset``; applying it here would rotate pixels out from under
  labels that were annotated in the stored frame. Doing it in *both* places would also
  be defensible, but doing it in only one is silently wrong.
* **No colour conversion beyond RGB**, matching the dataset.

Usage::

    python -m ropart.scripts.preprocess_hlw --src $WS/hlw --dst $WS/hlw224 --workers 8

Output mirrors the input layout — ``images/`` (same relative paths), ``metadata.csv``
(endpoints rescaled), ``split/`` (copied verbatim) — so the two roots are interchangeable.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image


def _resize_one(args: tuple[Path, Path, Path, int, int]) -> tuple[str, float | None]:
    """Resize one image so its short side is ``size``. Returns ``(rel_path, scale)``.

    ``scale`` is ``None`` if the image could not be read, so the caller can drop the row
    rather than emit a label with no pixels behind it.
    """
    src_root, dst_root, rel, size, quality = args
    src, dst = src_root / rel, dst_root / rel
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = size / min(w, h)
            # Round to whole pixels, but never below `size` on the short side — a 1px
            # shortfall would make HLWDataset's crop box run off the image.
            nw, nh = max(size, round(w * scale)), max(size, round(h * scale))
            im = im.resize((nw, nh), Image.BILINEAR)
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, "JPEG", quality=quality)
        # Report the scale actually realised, not the requested one: the rounding above
        # can shift it slightly, and the labels must follow the pixels.
        return str(rel), nw / w
    except Exception as exc:                      # noqa: BLE001 - report, do not abort
        print(f"  skip {rel}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return str(rel), None


def _read_metadata(path: Path) -> dict[str, tuple[float, float, float, float]]:
    out: dict[str, tuple[float, float, float, float]] = {}
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 5:
                continue
            try:
                out[row[0]] = tuple(float(v) for v in row[1:5])  # type: ignore[assignment]
            except ValueError:                     # header row
                continue
    return out


def main() -> None:
    p = argparse.ArgumentParser("Pre-resize HLW to the training resolution")
    p.add_argument("--src", required=True, help="HLW root (images/, split/, metadata.csv)")
    p.add_argument("--dst", required=True, help="output root; created if absent")
    p.add_argument("--size", type=int, default=224, help="target short side")
    p.add_argument("--quality", type=int, default=95, help="output JPEG quality")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="process only N images (smoke test)")
    a = p.parse_args()

    src_root, dst_root = Path(a.src), Path(a.dst)
    meta = _read_metadata(src_root / "metadata.csv")
    print(f"metadata rows: {len(meta):,}")

    rels = [r for r in meta if (src_root / "images" / r).exists()]
    missing = len(meta) - len(rels)
    if missing:
        print(f"WARNING: {missing:,} metadata rows have no image on disk — dropped")
    if a.limit:
        rels = rels[: a.limit]
    print(f"resizing {len(rels):,} images -> short side {a.size}, {a.workers} workers")

    (dst_root / "images").mkdir(parents=True, exist_ok=True)
    jobs = [(src_root / "images", dst_root / "images", Path(r), a.size, a.quality)
            for r in rels]

    scales: dict[str, float] = {}
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(_resize_one, j) for j in jobs]
        for fut in as_completed(futures):
            rel, scale = fut.result()
            if scale is not None:
                scales[rel] = scale
            done += 1
            if done % 5000 == 0:
                print(f"  {done:,}/{len(jobs):,}", flush=True)

    # Rescale the labels by the *realised* scale. Uniform scaling is sign-agnostic, so
    # this is correct under either Y_AXIS_DOWN convention — see the module docstring.
    out_csv = dst_root / "metadata.csv"
    with open(out_csv, "w", newline="") as fh:
        wr = csv.writer(fh)
        for rel, s in sorted(scales.items()):
            x1, y1, x2, y2 = meta[rel]
            wr.writerow([rel, f"{x1 * s:.6f}", f"{y1 * s:.6f}",
                         f"{x2 * s:.6f}", f"{y2 * s:.6f}"])
    print(f"wrote {out_csv} with {len(scales):,} rows")

    src_split = src_root / "split"
    if src_split.is_dir():
        shutil.copytree(src_split, dst_root / "split", dirs_exist_ok=True)
        print(f"copied {src_split} -> {dst_root / 'split'}")

    failed = len(jobs) - len(scales)
    if failed:
        print(f"WARNING: {failed:,} images failed to convert and were dropped from "
              f"metadata.csv — the split files still list them, and HLWDataset will "
              f"report them as missing metadata at load time")


if __name__ == "__main__":
    main()
