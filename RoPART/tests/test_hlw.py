"""Geometry and metric checks for the HLW horizon task. Run from RoPART/:

    .venv/bin/python tests/test_hlw.py

These exist because the crop/resize label transform is the single most likely source
of a silent bug in the Phase-3 orientation pipeline: a wrong label transform still
trains, still converges, and produces a plausible-but-meaningless AUC.
"""

import math

import torch

from ropart.hlw import (
    AUC_THRESHOLD,
    _endpoints_from_row,
    endpoints_to_theta_rho,
    horizon_auc,
    horizon_error,
    theta_rho_to_lr,
    transform_endpoints,
)


def test_horizontal_line_through_centre():
    """A flat horizon through the image centre is theta = 0, rho = 0, l = r = 0."""
    theta, rho = endpoints_to_theta_rho(-100.0, 0.0, 100.0, 0.0, height=200.0)
    assert abs(theta) < 1e-9, theta
    assert abs(rho) < 1e-9, rho
    l, r = theta_rho_to_lr(theta, rho, 1.0)
    assert abs(l) < 1e-9 and abs(r) < 1e-9
    print("OK flat centred horizon -> (0, 0)")


def test_offset_line():
    """A flat horizon a quarter-image below centre has rho = 0.25 image heights."""
    theta, rho = endpoints_to_theta_rho(-100.0, 50.0, 100.0, 50.0, height=200.0)
    assert abs(theta) < 1e-9
    assert abs(rho - 0.25) < 1e-9, rho
    l, r = theta_rho_to_lr(theta, rho, 1.0)
    assert abs(l - 0.25) < 1e-9 and abs(r - 0.25) < 1e-9
    print("OK offset horizon -> rho = 0.25")


def test_endpoint_order_is_irrelevant():
    """A line is undirected: swapping the endpoints must not change the target.

    Without the half-turn fold in endpoints_to_theta_rho this flips sign on roughly
    half the dataset, which trains to a confidently wrong model.
    """
    a = endpoints_to_theta_rho(-80.0, -20.0, 90.0, 35.0, height=200.0)
    b = endpoints_to_theta_rho(90.0, 35.0, -80.0, -20.0, height=200.0)
    assert abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9, (a, b)
    print("OK endpoint order is irrelevant")


def test_theta_rho_lr_roundtrip():
    """(theta, rho) -> (l, r) must agree with evaluating the line at the edges."""
    for theta_deg, rho in [(0.0, 0.0), (10.0, 0.1), (-25.0, -0.3), (5.0, 0.42)]:
        theta = math.radians(theta_deg)
        l, r = theta_rho_to_lr(theta, rho, 1.0)
        # The line is -x sin(t) + y cos(t) = rho; check both edge points satisfy it.
        for x, y in [(-0.5, l), (0.5, r)]:
            resid = -x * math.sin(theta) + y * math.cos(theta) - rho
            assert abs(resid) < 1e-9, (theta_deg, rho, resid)
    print("OK (theta, rho) <-> (l, r) round-trip")


def test_crop_moves_the_horizon():
    """Cropping off-centre must move the horizon relative to the new centre.

    This is the check that catches 'inherited the original label after cropping'.
    """
    # Flat horizon at y = 100 in a 400x400 image (top-left origin), i.e. dead centre.
    pts = (0.0, 100.0, 400.0, 100.0)
    # Centre crop of the top-left 200x200 quadrant: its centre is (100, 100), so the
    # horizon now passes exactly through the crop centre -> rho = 0.
    cpts, (ow, oh) = transform_endpoints(pts, (0.0, 0.0, 200.0, 200.0), 1.0)
    _, rho_centred = endpoints_to_theta_rho(*cpts, height=oh)
    assert abs(rho_centred) < 1e-9, rho_centred
    # Crop y in [50, 250], so its centre is y = 150 — i.e. 50px *below* the horizon at
    # y = 100. In a y-down centred frame the horizon is then at -50, rho = -50/200.
    cpts2, (_, oh2) = transform_endpoints(pts, (0.0, 50.0, 200.0, 200.0), 1.0)
    _, rho_shift = endpoints_to_theta_rho(*cpts2, height=oh2)
    assert abs(rho_shift + 0.25) < 1e-9, rho_shift
    print("OK crop transforms the label (rho tracks the crop centre)")


def test_resize_is_scale_invariant():
    """rho is in image heights, so a uniform resize must not change it."""
    pts = (0.0, 150.0, 400.0, 50.0)
    a_pts, (_, ah) = transform_endpoints(pts, (0.0, 0.0, 400.0, 400.0), 1.0)
    b_pts, (_, bh) = transform_endpoints(
        tuple(v * 0.5 for v in pts), (0.0, 0.0, 200.0, 200.0), 1.0)
    ta, ra = endpoints_to_theta_rho(*a_pts, height=ah)
    tb, rb = endpoints_to_theta_rho(*b_pts, height=bh)
    assert abs(ta - tb) < 1e-9 and abs(ra - rb) < 1e-9, ((ta, ra), (tb, rb))
    print("OK rho is resize-invariant (units of image heights)")


def test_horizon_error_is_zero_on_exact():
    t = torch.tensor([0.0, 0.1, -0.2])
    r = torch.tensor([0.0, 0.3, -0.1])
    e = horizon_error(t, r, t, r, 1.0)
    assert torch.allclose(e, torch.zeros_like(e), atol=1e-6), e
    print("OK horizon error is 0 for an exact prediction")


def test_horizon_error_pure_offset():
    """A pure rho error of d shifts both edges by d, so the error is exactly d."""
    z = torch.zeros(1)
    e = horizon_error(z, z + 0.1, z, z, 1.0)
    assert abs(float(e) - 0.1) < 1e-6, e
    print("OK pure offset error == |d rho|")


def test_auc_bounds():
    """AUC is 1.0 for a perfect predictor and 0.0 when every error exceeds the cut."""
    assert abs(horizon_auc(torch.zeros(100)) - 1.0) < 1e-6
    assert horizon_auc(torch.full((100,), 1.0)) == 0.0
    # Half the errors at 0 and half above threshold -> exactly 0.5.
    mixed = torch.cat([torch.zeros(50), torch.full((50,), 10.0)])
    assert abs(horizon_auc(mixed) - 0.5) < 1e-6, horizon_auc(mixed)
    # A constant error at half the threshold -> 1 - 0.5 = 0.5 of the area.
    half = torch.full((64,), AUC_THRESHOLD / 2)
    assert abs(horizon_auc(half) - 0.5) < 1e-6, horizon_auc(half)
    print("OK AUC bounds and hand-computable cases")


def test_horizon_epoch_runs_on_a_model():
    """horizon_run_epoch must train and eval a real RoPARTViT with a 2-wide head.

    Runs on synthetic tensors so the wiring (forward_classify -> (b, 2), loss, metric
    accumulation) is checked now rather than on the day the dataset lands.
    """
    from ropart.hlw import horizon_run_epoch
    from ropart.model import RoPARTViT, model_config

    cfg = model_config("deit_small_patch4_32")
    model = RoPARTViT(**cfg, num_classes=2, num_channels=2, mask_prob=0.0)
    model.head = torch.nn.Identity()
    x = torch.randn(8, 3, 32, 32)
    y = torch.stack([torch.empty(8).uniform_(-0.3, 0.3),
                     torch.empty(8).uniform_(-0.5, 0.5)], dim=1)
    loader = [(x[:4], y[:4]), (x[4:], y[4:])]
    dev = torch.device("cpu")

    ev = horizon_run_epoch(model, loader, dev)
    for k in ("loss", "auc", "theta_mae", "rho_mae"):
        assert k in ev, (k, ev)
    assert 0.0 <= ev["auc"] <= 1.0, ev["auc"]

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tr = horizon_run_epoch(model, loader, dev, optimizer=opt)
    assert tr["loss"] > 0.0
    print(f"OK horizon_run_epoch trains and evals (auc={ev['auc']:.3f})")


def test_metadata_row_layouts_v1_and_v2():
    """Both HLW ``metadata.csv`` layouts must yield the same four endpoint columns.

    Regression test for 2026-08-20: v2 rows carry ``width, height`` before the endpoints,
    so the unconditional ``row[1:5]`` slice read ``(width, height, x1, y1)`` and dropped
    ``y2``. It produced a target distribution with a mean horizon tilt of -40.8 degrees
    that trained and converged without complaint — nothing but the constants revealed it.
    """
    v1 = ["Alamo/a.jpg", "-5000", "445.311034", "5000", "-62.602260"]
    v2 = ["0006/b.jpg", "1800", "2400", "-5000", "445.311034", "5000", "-62.602260"]
    expected = (-5000.0, 445.311034, 5000.0, -62.602260)
    assert _endpoints_from_row(v1) == expected
    assert _endpoints_from_row(v2) == expected
    assert _endpoints_from_row(["filename", "x1", "y1", "x2", "y2"]) is None


def test_per_image_errors_reproduce_the_aggregates():
    """``per_image=True`` must average exactly to the reported theta_mae / rho_mae.

    The paired test in ``ropart.scripts.paired_horizon_test`` differences these arrays
    per image, so if they were a *different* quantity from the headline MAE the paired
    interval would be an interval on something nobody reports. This ties the two
    together. ``horizon_err`` must likewise be what the AUC is computed from.
    """
    from ropart.hlw import horizon_auc, horizon_run_epoch
    from ropart.model import RoPARTViT, model_config

    torch.manual_seed(0)
    cfg = model_config("deit_small_patch4_32")
    model = RoPARTViT(**cfg, num_classes=2, num_channels=2, mask_prob=0.0)
    model.head = torch.nn.Identity()
    x = torch.randn(10, 3, 32, 32)
    y = torch.stack([torch.empty(10).uniform_(-0.3, 0.3),
                     torch.empty(10).uniform_(-0.5, 0.5)], dim=1)
    # Deliberately ragged batches: a per-sample array must not depend on batching, while
    # the batch-weighted meter would only agree with it if the weighting is right.
    loader = [(x[:4], y[:4]), (x[4:], y[4:])]
    dev = torch.device("cpu")

    ev = horizon_run_epoch(model, loader, dev, per_image=True)
    assert len(ev["theta_err"]) == 10 and len(ev["rho_err"]) == 10
    assert len(ev["horizon_err"]) == 10
    assert abs(float(ev["theta_err"].mean()) - ev["theta_mae"]) < 1e-4
    assert abs(float(ev["rho_err"].mean()) - ev["rho_mae"]) < 1e-6
    assert abs(horizon_auc(ev["horizon_err"]) - ev["auc"]) < 1e-9
    assert (ev["theta_err"] >= 0).all() and (ev["rho_err"] >= 0).all()

    # The default path must be untouched: no extra keys, same aggregates.
    plain = horizon_run_epoch(model, loader, dev)
    assert "theta_err" not in plain and "horizon_err" not in plain
    assert abs(plain["theta_mae"] - ev["theta_mae"]) < 1e-6
    print("OK per-image errors average to the reported MAEs")


def test_wilcoxon_against_hand_computable_cases():
    """The signed-rank statistic, on cases small enough to check by hand."""
    import numpy as np

    from ropart.scripts.paired_horizon_test import wilcoxon_signed_rank

    # Perfectly symmetric differences: W sits exactly at its null mean, so z = 0.
    z, p = wilcoxon_signed_rank(np.array([1.0, -1.0, 2.0, -2.0]))
    assert abs(z) < 1e-12, z
    assert p > 0.99, p

    # All differences positive: W takes its maximum, n(n+1)/2 = 55 against a mean of
    # 27.5 and variance 96.25, so z = 27.5 / sqrt(96.25) = 2.803.
    z, p = wilcoxon_signed_rank(np.arange(1.0, 11.0))
    assert abs(z - 27.5 / math.sqrt(96.25)) < 1e-9, z
    assert p < 0.01, p

    # Zeros are dropped, not ranked — otherwise they would dilute the statistic.
    z_with, _ = wilcoxon_signed_rank(np.array([0.0, 0.0, 1.0, 2.0, 3.0]))
    z_without, _ = wilcoxon_signed_rank(np.array([1.0, 2.0, 3.0]))
    assert abs(z_with - z_without) < 1e-12

    # An empty (or all-zero) input is a no-difference result, not a crash.
    assert wilcoxon_signed_rank(np.array([0.0, 0.0])) == (0.0, 1.0)
    print("OK Wilcoxon signed-rank matches hand-computed cases")


def test_paired_bootstrap_recovers_a_known_shift():
    """The bootstrap CI must cover a planted shift and exclude zero when it should."""
    import numpy as np

    from ropart.scripts.paired_horizon_test import paired_bootstrap

    rng = np.random.default_rng(0)
    diff = rng.normal(-0.5, 1.0, size=2000)
    lo, hi = paired_bootstrap(diff, 2000, 0.05, np.random.default_rng(1))
    assert lo < diff.mean() < hi, (lo, diff.mean(), hi)
    assert hi < 0.0, (lo, hi)          # a real shift: the interval excludes zero

    # No shift at all: the interval must straddle zero.
    lo, hi = paired_bootstrap(rng.normal(0.0, 1.0, size=2000), 2000, 0.05,
                              np.random.default_rng(2))
    assert lo < 0.0 < hi, (lo, hi)
    print("OK paired bootstrap covers a planted shift and rejects a null one")


if __name__ == "__main__":
    test_horizontal_line_through_centre()
    test_offset_line()
    test_endpoint_order_is_irrelevant()
    test_theta_rho_lr_roundtrip()
    test_crop_moves_the_horizon()
    test_resize_is_scale_invariant()
    test_horizon_error_is_zero_on_exact()
    test_horizon_error_pure_offset()
    test_auc_bounds()
    test_horizon_epoch_runs_on_a_model()
    test_metadata_row_layouts_v1_and_v2()
    test_per_image_errors_reproduce_the_aggregates()
    test_wilcoxon_against_hand_computable_cases()
    test_paired_bootstrap_recovers_a_known_shift()
    print("\nALL HLW TESTS PASSED")
