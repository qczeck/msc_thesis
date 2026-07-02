"""Sanity checks for the RoPART training package. Run from RoPART/:

    .venv/bin/python tests/test_ropart.py
"""

import math

import numpy as np
import torch

from helpers.sampling import (
    sample_offgrid_patches,
    sample_rotation_angles,
    sample_quad_rotations,
    rotation_margin,
    crop_patches,
    crop_patches_quad,
)
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
from ropart.eval import _SqAcc, _accumulate, predict_table, report


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


def test_model_registry():
    from ropart.model import model_config
    from ropart.train import parse_model_name

    # ViT-S/4 dims unchanged (the CIFAR baseline must not move)
    assert model_config("deit_small_patch4_32") == dict(
        img_size=32, patch_size=4, embed_dim=384, depth=12, num_heads=6)
    # ViT-B/16 for ImageNet
    base = model_config("deit_base_patch16_224")
    assert (base["embed_dim"], base["depth"], base["num_heads"]) == (768, 12, 12)
    # two-digit patch size parses correctly (the old string-slice gave 6, not 16)
    assert parse_model_name("deit_base_patch16_224") == (224, 16)
    assert parse_model_name("deit_small_patch8_32") == (32, 8)
    try:
        model_config("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown model should raise")
    print("OK model registry (dims + two-digit patch parse)")


def test_loss():
    crit = RelativeMSE()
    for nc in (2, 4):
        loss = crit(torch.randn(2, nc, 8), torch.randn(2, nc, 8))
        assert loss.ndim == 0 and torch.isfinite(loss)
    print("OK RelativeMSE")


def test_loss_balance_default_unchanged():
    """balance='none' (default) must be byte-identical to the raw grouped MSE."""
    torch.manual_seed(0)
    out, tgt = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
    se = (out - tgt) ** 2
    expected = se[:, 0:2].mean() + se[:, 2:4].mean()
    assert torch.allclose(RelativeMSE()(out, tgt), expected)
    print("OK RelativeMSE balance=none unchanged")


def test_loss_balance_variance_rescales_groups():
    """balance='variance' lifts a tiny-scale rotation group to fraction-of-floor.

    With a pixel-scale translation group and a unit-circle rotation group, the raw
    loss is dominated by translation; the variance-balanced loss measures each group
    as a fraction of its own floor, so a mean-predictor on both gives ~1 + ~1 = ~2.
    """
    torch.manual_seed(0)
    n = 4096
    tgt = torch.empty(1, 4, n)
    tgt[:, 0:2] = torch.randn(1, 2, n) * 5.0            # translation, Var ~ 25
    phi = (torch.rand(1, n) - 0.5) * (math.pi / 3)      # Δφ in [-30°, 30°]
    tgt[:, 2] = torch.cos(phi)
    tgt[:, 3] = torch.sin(phi)
    # mean predictor for both groups
    out = tgt.mean(dim=2, keepdim=True).expand_as(tgt).clone()

    raw = RelativeMSE(balance="none")(out, tgt)
    bal = RelativeMSE(balance="variance")(out, tgt)
    # raw is swamped by the translation scale; balanced sits near 2 (two groups,
    # each ~1× its floor under a mean predictor).
    assert raw > 5.0
    assert 1.5 < bal.item() < 2.5
    print(f"OK RelativeMSE balance=variance (raw={raw:.1f}, bal={bal:.2f})")


def test_bounded_rotation_sampling():
    """max_angle bounds the sampled per-patch orientation to [-a, +a]."""
    a = math.radians(30)
    ang = sample_rotation_angles(100000, low=-a, high=a)
    assert ang.min() >= -a - 1e-6 and ang.max() <= a + 1e-6
    full = sample_rotation_angles(100000)
    assert full.max() > a  # full circle reaches well beyond ±30°
    print("OK bounded rotation sampling")


def test_quad_rotation_sampling():
    """quad sampling yields only the four right angles {0, 90, 180, 270}°."""
    ang = sample_quad_rotations(100000)
    ks = torch.round(ang / (math.pi / 2)).long()
    assert set(ks.tolist()) == {0, 1, 2, 3}                  # all four appear
    assert torch.allclose(ang, ks.float() * (math.pi / 2))   # exact multiples
    print("OK quad rotation sampling")


def test_crop_patches_quad_is_exact_rot90():
    """quad extraction is an exact rot90 of the axis-aligned crop — no resampling."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    n, ps = 4, 8
    boxes = sample_offgrid_patches(32, ps, n, margin=0, generator=torch.Generator().manual_seed(1))
    base = crop_patches(img, boxes)  # (n, C, P, P), axis-aligned reference
    for k in range(4):
        angles = torch.full((n,), k * math.pi / 2)
        out = crop_patches_quad(img, boxes, angles)
        expect = torch.rot90(base, k, dims=(-2, -1))
        assert torch.equal(out, expect), f"k={k} not an exact rot90"
    print("OK crop_patches_quad exact rot90")


def test_quad_control_registered():
    """The 'quad' control is registered, rotates, and emits 4 channels."""
    assert "quad" in available_controls()
    extractor, num_channels, rotates = build_extractor("quad")
    assert rotates and num_channels == 4
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    boxes = sample_offgrid_patches(32, 8, 4, margin=0)
    patches = extractor(img, boxes, sample_quad_rotations(4))
    assert patches.shape == (4, 3, 8, 8)
    print("OK quad control registered + runs")


def test_quad_ch2_control_rotates_but_unsupervised():
    """`quad_ch2` rotates pixels exactly like `quad` but emits only 2 channels.

    The extractor must be pixel-identical to `quad` (exact rot90), yet its registered
    num_channels=2 so the target builder drops the rotation channels — the run that
    isolates whether rotated *pixels* or the ch=4 *objective* collapses translation.
    """
    assert "quad_ch2" in available_controls()
    ext2, nc2, rot2 = build_extractor("quad_ch2")
    ext4, nc4, rot4 = build_extractor("quad")
    assert (nc2, rot2) == (2, True)
    assert (nc4, rot4) == (4, True)
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    boxes = sample_offgrid_patches(32, 8, 4, margin=0)
    angles = sample_quad_rotations(4)
    assert torch.equal(ext2(img, boxes, angles), ext4(img, boxes, angles))  # same pixels
    print("OK quad_ch2 control: same rot90 pixels as quad, num_channels=2")


def test_deit_base_patch32_224_config():
    """Large-patch ViT-B/32 gives an exact 7x7=49-patch tiling and forwards cleanly."""
    from ropart.model import model_config
    from ropart.train import parse_model_name

    cfg = model_config("deit_base_patch32_224")
    assert cfg == dict(img_size=224, patch_size=32, embed_dim=768, depth=12, num_heads=12)
    assert parse_model_name("deit_base_patch32_224") == (224, 32)
    model = RoPARTViT(**cfg, num_channels=4, num_pairs=20, num_classes=100)
    assert model.num_patches == 49  # (224 // 32) ** 2, exact
    out, idx = model.forward_pretrain(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 4, 20) and idx.shape == (2, 20)
    print("OK deit_base_patch32_224 (49 patches, forward)")


def test_loss_eps_guards_small_variance_blowup():
    """A near-constant (small-angle) rotation group must not explode the balanced loss.

    Mimics ±5°: cos Δφ ~ 1 with variance ~1e-5. With the default eps=1e-3 the divisor
    is clamped, so the balanced loss stays O(1); with a tiny eps it blows up ~100x.
    """
    torch.manual_seed(0)
    n = 4096
    tgt = torch.empty(1, 4, n)
    tgt[:, 0:2] = torch.randn(1, 2, n) * 5.0
    phi = (torch.rand(1, n) - 0.5) * (2 * math.radians(5))   # Δφ in ~[-5°, 5°]
    tgt[:, 2], tgt[:, 3] = torch.cos(phi), torch.sin(phi)
    out = tgt.clone()
    out[:, 2:] += 0.01  # small rotation error against a near-zero-variance target
    guarded = RelativeMSE(balance="variance", eps=1e-3)(out, tgt)
    unguarded = RelativeMSE(balance="variance", eps=1e-9)(out, tgt)
    assert torch.isfinite(guarded) and guarded < 1.0
    assert unguarded > 10 * guarded  # the clamp is what prevents the blow-up
    print(f"OK loss-eps guards small-variance blow-up (eps=1e-3:{guarded:.3f} vs 1e-9:{unguarded:.1f})")


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


class _IdxHead:
    """Stub relative head: echoes the queried pair indices ``(i, j)`` as the two
    output channels, so :func:`predict_table` placement can be checked."""

    query_type = "oracle"

    def __call__(self, feats, pairs):  # pairs (b, P, 2) -> (b, P, 2)
        return pairs.float()


class _OracleModel:
    """Minimal stand-in exposing the surface :func:`predict_table` touches."""

    num_channels = 2

    def __init__(self, n: int, embed: int = 4):
        self.n = n
        self.embed = embed
        self.head = _IdxHead()

    def _encode(self, x, *, add_pos, mask):  # noqa: D401 - matches model signature
        b = x.shape[0]
        return torch.zeros(b, self.n + 1, self.embed, device=x.device)  # cls slot dropped by caller


def test_eval_predict_table_orientation():
    """``predict_table`` must place pair ``(i, j)`` at ``[:, :, i, j]`` — the same
    (ref i, tgt j) orientation as ``build_targets`` — or every symmetry/composition
    metric is silently transposed."""
    n = 9
    model = _OracleModel(n)
    table = predict_table(model, torch.zeros(2, 3, 32, 32))  # (b, 2, n, n)
    assert table.shape == (2, 2, n, n), table.shape
    exp_i = torch.arange(n)[:, None].expand(n, n).float()  # row = ref i
    exp_j = torch.arange(n)[None, :].expand(n, n).float()  # col = tgt j
    assert torch.allclose(table[0, 0], exp_i), "channel 0 not indexed by ref i"
    assert torch.allclose(table[0, 1], exp_j), "channel 1 not indexed by tgt j"
    print("OK eval.predict_table orientation matches target convention")


def test_eval_metrics_oracle():
    """A perfect predictor (pred == target) must drive every 'want ~0' metric to ~0
    and the MSE-vs-floor ratios to ~0 — guards the eval metric math itself."""
    torch.manual_seed(0)
    ps, n = 4, 64
    boxes = torch.stack([sample_offgrid_patches(32, ps, n) for _ in range(3)])
    angles = torch.stack([sample_rotation_angles(n) for _ in range(3)])
    tgt = build_targets(boxes, angles, rotates=True, centered=True, normalize_by=ps)

    acc = _SqAcc()
    _accumulate(tgt.clone(), tgt, rotates=True, acc=acc, n_triples=2048)
    m = report(acc, rotates=True, patch_size=ps)

    assert m["mse_x_vs_floor"] < 1e-6 and m["mse_y_vs_floor"] < 1e-6
    assert m["trans_rmse_px"] < 1e-3
    assert m["identity_rmse_px"] < 1e-3 and m["identity_rot_rmse"] < 1e-4
    assert m["neg_sym_xy_ratio"] < 1e-4 and m["neg_sym_rot_ratio"] < 1e-4
    assert m["comp_xy_ratio"] < 1e-4 and m["comp_rot_mae_deg"] < 1e-2
    print("OK eval metrics ~0 on a perfect predictor")


if __name__ == "__main__":
    test_targets_match_helpers_oracle()
    test_invariants_on_batch()
    test_gather_shapes()
    test_model_forward()
    test_loss()
    test_loss_balance_default_unchanged()
    test_loss_balance_variance_rescales_groups()
    test_bounded_rotation_sampling()
    test_quad_rotation_sampling()
    test_crop_patches_quad_is_exact_rot90()
    test_quad_control_registered()
    test_quad_ch2_control_rotates_but_unsupervised()
    test_deit_base_patch32_224_config()
    test_loss_eps_guards_small_variance_blowup()
    test_controls_registry()
    test_extractors_run_on_image()
    test_eval_predict_table_orientation()
    test_eval_metrics_oracle()
    print("\nALL TESTS PASSED")
