"""Materialise the standard ImageNet-100 subset as train/ val/ ImageFolders.

Pulls the CMC/Tian 100-class ImageNet subset from a HuggingFace mirror (default
``clane9/imagenet-100``) and writes it to ``<out>/train/<wnid>/*.jpg`` and
``<out>/val/<wnid>/*.jpg`` — the layout :class:`ropart.data.RoPARTImageFolder`
expects. Resumable: a class dir already holding the expected number of images is
skipped, so re-running after an interruption is cheap.

Run on a lab PC with network (e.g. gpu30) via ``setup_imagenet_cluster.sh`` — *not*
on the cluster head nodes (no pip/network there). Point ``HF_HOME`` at the workspace
so the ~15 GB HF cache does not land on the NFS home quota.

    python ropart/scripts/_download_imagenet100.py <out_dir> [hf_dataset_id]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def _safe_class_dir(label_idx: int, label_name: str) -> str:
    """Filesystem-safe, sortable class-dir name, identical across train/val.

    The HF mirror exposes human-readable class names (``'bonnet, poke bonnet'``),
    not wnids, so spaces/commas would land in paths. Zero-pad the label index so the
    ImageFolder sort order matches the original label order, and append a slugged
    name for readability.
    """
    slug = re.sub(r"[^0-9A-Za-z]+", "_", label_name).strip("_").lower()
    return f"{label_idx:03d}_{slug}"


def _val_split_name(splits) -> str:
    for cand in ("validation", "val", "test"):
        if cand in splits:
            return cand
    raise SystemExit(f"no validation split found among {list(splits)}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    out = Path(sys.argv[1])
    dataset_id = sys.argv[2] if len(sys.argv) > 2 else "clane9/imagenet-100"

    from datasets import load_dataset  # heavy import; only needed for prep

    print(f"[in100] loading {dataset_id} (this downloads ~15 GB on first run) ...")
    ds = load_dataset(dataset_id)
    splits = set(ds.keys())
    split_map = {"train": "train", _val_split_name(splits): "val"}

    for src_split, dst_split in split_map.items():
        d = ds[src_split]
        label_feat = d.features["label"]  # ClassLabel; .int2str -> wnid
        n = len(d)
        print(f"[in100] {src_split} -> {dst_split}: {n} images, "
              f"{label_feat.num_classes} classes")
        # bucket counts so we can skip already-complete class dirs on resume
        written = 0
        for i, ex in enumerate(d):
            label_idx = int(ex["label"])
            cls_dir = out / dst_split / _safe_class_dir(label_idx, label_feat.int2str(label_idx))
            cls_dir.mkdir(parents=True, exist_ok=True)
            dst = cls_dir / f"{i:08d}.jpg"
            if dst.exists():
                continue
            img = ex["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(dst, "JPEG", quality=95)
            written += 1
            if written % 5000 == 0:
                print(f"[in100]   {dst_split}: {written} new / {i + 1} seen")
        print(f"[in100] {dst_split} done: {written} newly written")

    print(f"[in100] materialised at {out}")


if __name__ == "__main__":
    main()
