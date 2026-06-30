"""Datasets for RoPART pretraining and the linear-probe.

``RoPARTCIFAR`` and ``RoPARTImageFolder`` are the off-grid, constant-patch-size
pretraining datasets (the equivalent of upstream ``VectorizedCIFAR`` ``v3``, minus
the grid variants and the candidate-box bank). They are **control-agnostic**: they
sample boxes (and angles, when rotating) and delegate pixel extraction to an
injected control extractor (see ``ropart.controls``), then normalise and row-major
pack the patches into the pseudo-image the ViT ingests. The only difference between
them is the image *source* — CIFAR's in-memory array vs ImageNet JPEGs on disk —
so the per-image packing is shared in :func:`pack_patches_from_image`.

``build_cls_dataset`` / ``build_cls_dataset_imagenet`` return the plain
classification datasets with standard transforms for the linear-probe / finetune.

Image-level augmentation is deliberately minimal (normalise only) during pretraining
so the baseline-vs-rotation comparison is clean — upstream applies the full DeiT
augmentation; we skip it here on purpose.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms

from helpers.sampling import (
    rotation_margin,
    sample_offgrid_patches,
    sample_quad_rotations,
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


def pack_patches_from_image(
    image: np.ndarray,
    *,
    extractor: Extractor,
    rotates: bool,
    img_size: int,
    patch_size: int,
    num_patches: int,
    margin: int,
    max_angle: float | None = None,
    angle_set: str = "continuous",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample patches from one image and pack them into the ViT pseudo-image.

    The dataset-agnostic core shared by every RoPART pretraining dataset: sample
    off-grid boxes (and per-patch angles when ``rotates``), extract pixels via the
    active control ``extractor``, then normalise and row-major pack.

    Args:
        image: ``(H, W, C)`` uint8 array (already resized to ``img_size`` square).
        extractor: control extractor from :mod:`ropart.controls`.
        rotates: whether the active control samples per-patch orientations.
        img_size: side of the (square) source/packed image, in pixels.
        patch_size: side of every patch, in pixels.
        num_patches: number of patches (must be a perfect square tiling ``img_size``).
        margin: edge margin for box sampling (``rotation_margin`` when ``rotates``).
        max_angle: half-range for per-patch orientation, in **radians**. ``None``
            (default) samples the full circle ``[0, 2π)``; a value ``a`` samples
            symmetrically in ``[-a, +a]`` (the bounded-rotation experiment).
            Ignored when ``angle_set == "quad"``.
        angle_set: ``"continuous"`` (default) samples real-valued angles; ``"quad"``
            samples discrete right angles ``{0, 90, 180, 270}°`` (pairs with the
            ``quad`` control's exact ``rot90`` extraction).

    Returns:
        ``(packed [3, img_size, img_size], boxes [4, N], angles [N])``; ``angles``
        is zeros when the control does not rotate.
    """
    boxes = sample_offgrid_patches(img_size, patch_size, num_patches, margin=margin)
    if not rotates:
        angles = torch.zeros(num_patches)
    elif angle_set == "quad":
        angles = sample_quad_rotations(num_patches)
    elif max_angle is None:
        angles = sample_rotation_angles(num_patches)
    else:
        angles = sample_rotation_angles(num_patches, low=-max_angle, high=max_angle)
    patches = extractor(image, boxes, angles)  # (N, C, P, P) in [0, 1]
    packed = retile_patches(patches, img_size)  # (C, img, img)
    return _normalise(packed), boxes, angles


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
        max_angle: float | None = None,
        angle_set: str = "continuous",
    ):
        super().__init__(root, train=train, download=download)
        self.extractor = extractor
        self.rotates = rotates
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        # quad reads axis-aligned P×P (exact rot90), so it needs no diagonal margin.
        self.margin = rotation_margin(patch_size) if (rotates and angle_set != "quad") else 0
        self.max_angle = max_angle
        self.angle_set = angle_set

    def __getitem__(self, index: int):
        image = self.data[index]  # (H, W, C) uint8
        return pack_patches_from_image(
            image, extractor=self.extractor, rotates=self.rotates,
            img_size=self.img_size, patch_size=self.patch_size,
            num_patches=self.num_patches, margin=self.margin,
            max_angle=self.max_angle, angle_set=self.angle_set,
        )


class RoPARTImageFolder(datasets.ImageFolder):
    """Off-grid, constant-patch-size ImageNet (or any ImageFolder) for the pretext task.

    The disk-backed twin of :class:`RoPARTCIFAR`: each JPEG is loaded, resized to a
    square ``img_size`` (BILINEAR, mirroring ``helpers.sampling.load_image`` — minimal
    augmentation on purpose), then handed to the shared
    :func:`pack_patches_from_image`. ``__getitem__`` returns
    ``(packed [3, img, img], boxes [4, N], angles [N])``; the class label is dropped
    (the pretext task is label-free).
    """

    def __init__(
        self,
        root: str,
        *,
        extractor: Extractor,
        rotates: bool,
        img_size: int = 224,
        patch_size: int = 16,
        train: bool = True,
        max_angle: float | None = None,
        angle_set: str = "continuous",
    ):
        super().__init__(self._split_dir(root, train))
        self.extractor = extractor
        self.rotates = rotates
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        # quad reads axis-aligned P×P (exact rot90), so it needs no diagonal margin.
        self.margin = rotation_margin(patch_size) if (rotates and angle_set != "quad") else 0
        self.max_angle = max_angle
        self.angle_set = angle_set

    @staticmethod
    def _split_dir(root: str, train: bool) -> str:
        import os

        return os.path.join(root, "train" if train else "val")

    def __getitem__(self, index: int):
        path, _ = self.samples[index]
        image = Image.open(path).convert("RGB").resize(
            (self.img_size, self.img_size), Image.BILINEAR
        )
        image = np.asarray(image)  # (H, W, C) uint8
        return pack_patches_from_image(
            image, extractor=self.extractor, rotates=self.rotates,
            img_size=self.img_size, patch_size=self.patch_size,
            num_patches=self.num_patches, margin=self.margin,
            max_angle=self.max_angle, angle_set=self.angle_set,
        )


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


def build_cls_dataset_imagenet(root: str, *, img_size: int = 224):
    """Plain ImageNet-100 train/val ImageFolders with standard transforms (probe).

    Standard ImageNet evaluation transforms: RandomResizedCrop + flip for train,
    Resize(short side) + CenterCrop for val. ``root`` holds ``train/`` and ``val/``
    ImageFolders.
    """
    import os

    resize = int(round(img_size * 256 / 224))  # standard 256->224 ratio
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train = datasets.ImageFolder(os.path.join(root, "train"), transform=train_tf)
    val = datasets.ImageFolder(os.path.join(root, "val"), transform=val_tf)
    return train, val
