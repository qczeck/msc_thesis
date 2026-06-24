"""Datasets for RoPART pretraining and the linear-probe.

``RoPARTCIFAR`` is the focused, off-grid, constant-patch-size pretraining dataset
(the equivalent of upstream ``VectorizedCIFAR`` ``v3``, minus the grid variants
and the candidate-box bank). It is **control-agnostic**: it samples boxes (and
angles, when rotating) and delegates pixel extraction to an injected control
extractor (see ``ropart.controls``), then normalises and row-major packs the
patches into the pseudo-image the ViT ingests.

``build_cls_dataset`` returns plain CIFAR-100 with standard transforms for the
linear-probe / finetune.

Image-level augmentation is deliberately minimal (normalise only) so the
baseline-vs-rotation comparison is clean — upstream applies the full DeiT
augmentation during pretraining; we skip it here on purpose.
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision import datasets, transforms

from helpers.sampling import (
    rotation_margin,
    sample_offgrid_patches,
    sample_rotation_angles,
    retile_patches,
)
from ropart.controls import Extractor

# ImageNet normalisation (matches upstream's build_transform).
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _normalise(packed: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN, dtype=packed.dtype).view(3, 1, 1)
    std = torch.tensor(STD, dtype=packed.dtype).view(3, 1, 1)
    return (packed - mean) / std


class RoPARTCIFAR(datasets.CIFAR100):
    """Off-grid, constant-patch-size CIFAR-100 for the relative pretext task.

    ``__getitem__`` returns ``(packed [3, img, img], boxes [4, N], angles [N])``.
    ``angles`` is zeros when the active control does not rotate.
    """

    def __init__(
        self,
        root: str,
        *,
        extractor: Extractor,
        rotates: bool,
        img_size: int = 32,
        patch_size: int = 4,
        train: bool = True,
        download: bool = True,
    ):
        super().__init__(root, train=train, download=download)
        self.extractor = extractor
        self.rotates = rotates
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.margin = rotation_margin(patch_size) if rotates else 0

    def __getitem__(self, index: int):
        image = self.data[index]  # (H, W, C) uint8

        boxes = sample_offgrid_patches(
            self.img_size, self.patch_size, self.num_patches, margin=self.margin
        )
        if self.rotates:
            angles = sample_rotation_angles(self.num_patches)
        else:
            angles = torch.zeros(self.num_patches)

        patches = self.extractor(image, boxes, angles)  # (N, C, P, P) in [0, 1]
        packed = retile_patches(patches, self.img_size)  # (C, img, img)
        packed = _normalise(packed)
        return packed, boxes, angles


def build_cls_dataset(root: str, *, img_size: int = 32, download: bool = True):
    """Plain CIFAR-100 train/val with standard transforms (for the linear-probe)."""
    train_tf = transforms.Compose([
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train = datasets.CIFAR100(root, train=True, transform=train_tf, download=download)
    val = datasets.CIFAR100(root, train=False, transform=val_tf, download=download)
    return train, val
