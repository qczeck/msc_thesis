"""Sanity checks for the RoPART training package. Run from RoPART/:

    .venv/bin/python tests/test_ropart.py
"""

import math

import torch

from helpers.sampling import sample_offgrid_patches, sample_rotation_angles, rotation_margin
from helpers.targets import (
    relative_translation,
    relative_orientation,
    check_translation_invariants,
    check_orientation_invariants,
)
from ropart.targets import build_targets, gather_pairs
from ropart.losses import RelativeMSE
from ropart.model import RoPARTViT
from ropart.controls import build_extractor, available_controls


def test_targets_match_helpers_oracle():
    ps, n = 4, 64
    g = torch.Generator().manual_seed(0)
    boxes = sample_offgrid_patches(32, ps, n, margin=rotation_margin(ps), generator=g)
    angles = sample_rotation_angles(n, generator=g)

    # translation channels match the single-image helper
    t_batch = build_targets(boxes[None], rotates=False, centered=True, normalize_by=ps)
    t_oracle = relative_translation(boxes, normalize_by=ps)
    assert torch.allclose(t_batch[0], t_oracle, atol=1e-5)

    # rotation: translation + (cos, sin) channels both match
    r_batch = build_targets(boxes[None], angles[None], rotates=True, centered=True, normalize_by=ps)
    assert r_batch.shape == (1, 4, n, n)
    assert torch.allclose(r_batch[0, :2], t_oracle, atol=1e-5)
    assert torch.allclose(r_batch[0, 2:], relative_orientation(angles), atol=1e-5)
    print("OK targets match helpers oracle (translation + rotation)")


def test_invariants_on_batch():
    ps, n = 4, 64
    boxes = torch.stack([sample_offgrid_patches(32, ps, n) for _ in range(3)])  # (3,4,N)
    angles = torch.stack([sample_rotation_angles(n) for _ in range(3)])
    r = build_targets(boxes, angles, rotates=True, normalize_by=ps)
    for b in range(3):
        check_translation_invariants(r[b, :2])
        check_orientation_invariants(r[b, 2:])
    print("OK invariants hold across a batch")


def test_gather_shapes():
    ps, n = 4, 64
    boxes = torch.stack([sample_offgrid_patches(32, ps, n) for _ in range(2)])
    angles = torch.stack([sample_rotation_angles(n) for _ in range(2)])
    targets = build_targets(boxes, angles, rotates=True, normalize_by=ps)
    indices = torch.randint(0, n, (2, 17))
    gathered = gather_pairs(targets, indices)
    assert gathered.shape == (2, 4, 17)
    print("OK gather_pairs shape")


def test_model_forward():
    for nc in (2, 4):
        model = RoPARTViT(num_channels=nc, num_pairs=20, num_classes=100)
        x = torch.randn(2, 3, 32, 32)
        out, idx = model.forward_pretrain(x)
        assert out.shape == (2, nc, 20), out.shape
        assert idx.shape == (2, 20), idx.shape
        logits = model.forward_classify(x)
        assert logits.shape == (2, 100), logits.shape
    print("OK model forward (pretrain + classify), num_channels 2 and 4")


def test_loss():
    crit = RelativeMSE()
    for nc in (2, 4):
        loss = crit(torch.randn(2, nc, 8), torch.randn(2, nc, 8))
        assert loss.ndim == 0 and torch.isfinite(loss)
    print("OK RelativeMSE")


def test_controls_registry():
    assert {"translation", "raw", "supersample", "dominant", "randomised", "matched"} <= set(available_controls())
    _, nc_t, rot_t = build_extractor("translation")
    _, nc_r, rot_r = build_extractor("raw")
    assert (nc_t, rot_t) == (2, False)
    assert (nc_r, rot_r) == (4, True)
    print("OK controls registry")


def test_extractors_run_on_image():
    import numpy as np
    img = (np.random.rand(32, 32, 3) * 255).astype("uint8")
    ps, n = 4, 64
    boxes = sample_offgrid_patches(32, ps, n, margin=rotation_margin(ps))
    angles = sample_rotation_angles(n)
    for name in ("translation", "raw", "supersample", "dominant", "randomised"):
        extract, nc, rot = build_extractor(name)
        patches = extract(img, boxes, angles if rot else None)
        assert patches.shape == (n, 3, ps, ps), (name, patches.shape)
    print("OK all (implemented) control extractors run end-to-end")


if __name__ == "__main__":
    test_targets_match_helpers_oracle()
    test_invariants_on_batch()
    test_gather_shapes()
    test_model_forward()
    test_loss()
    test_controls_registry()
    test_extractors_run_on_image()
    print("\nALL TESTS PASSED")
