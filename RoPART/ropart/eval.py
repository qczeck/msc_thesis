"""Intrinsic correctness evaluation for a pretrained RoPART checkpoint.

This is the Phase-0 **correctness gate**: a trained relative head that
genuinely learned geometry must satisfy structural invariants that a
mean-predictor or a shortcut cannot. We check four, on the held-out val split,
without any retraining:

1. **Floor comparison** — per-channel ``mse_x/mse_y`` vs the mean-predictor floors
   ``Var(Δx)/Var(Δy)``, plus the translation RMSE in **pixels**. Beating the floor
   is necessary but not sufficient.
2. **Identity / diagonal** — the predicted relative transform for a patch with
   *itself* ``(i, i)`` must be ``(Δx, Δy) = 0`` (and ``(cos, sin) = (1, 0)`` when
   rotating). Tests that the head reads the queried slots rather than averaging.
3. **Negative symmetry** (source paper §5.1(iv), Fig. 6) — ``Δ(i→j) = −Δ(j→i)``.
   Translation channels are antisymmetric; for rotation ``cos`` is symmetric and
   ``sin`` antisymmetric. Reported as the residual RMS relative to the signal RMS
   (a ratio → 0 if perfectly antisymmetric).
4. **Compositional consistency** — relative motion composes: ``Δ(i→k) =
   Δ(i→j) + Δ(j→k)`` (additive for translation; angle-additive for ``Δφ``). The
   strongest of the four — pairwise symmetry can hold while composition fails.

Unlike :func:`forward_pretrain` (which samples random pairs), this queries the
relative head on **chosen** pairs/triples via ``model.head(feats, pairs)``, so the
invariants are directly testable.

Run from the ``RoPART/`` directory::

    python -m ropart.eval --resume out/baseline_translation/checkpoint.pth \
        --data-path ./data --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from ropart.controls import build_extractor
from ropart.data import RoPARTCIFAR
from ropart.model import RoPARTViT
from ropart.targets import build_targets
from ropart.train import parse_model_name, pick_device, set_seed


# --------------------------------------------------------------------------- #
# Running accumulators (true element-weighted RMS, not a mean-of-means)
# --------------------------------------------------------------------------- #


class _SqAcc:
    """Accumulate sums of squares and counts for element-weighted RMS / ratios."""

    def __init__(self):
        self._sq: dict[str, float] = {}
        self._n: dict[str, float] = {}

    def add(self, key: str, sq_sum: float, n: int):
        self._sq[key] = self._sq.get(key, 0.0) + float(sq_sum)
        self._n[key] = self._n.get(key, 0.0) + n

    def rms(self, key: str) -> float:
        return math.sqrt(self._sq[key] / self._n[key])

    def mean(self, key: str) -> float:
        return self._sq[key] / self._n[key]

    def ratio_rms(self, num: str, den: str) -> float:
        return math.sqrt(self._sq[num] / self._sq[den])


# --------------------------------------------------------------------------- #
# Model / data reconstruction from a checkpoint
# --------------------------------------------------------------------------- #


def load_model(ckpt: dict, device: torch.device) -> tuple[RoPARTViT, int, int]:
    """Rebuild the encoder + relative head exactly as the checkpoint expects.

    Returns ``(model, img_size, patch_size)``. ``num_channels`` is read from the
    saved head weight (authoritative even if ``args`` is incomplete).
    """
    ckpt_args = ckpt.get("args", {})
    img_size, patch_size = parse_model_name(ckpt_args.get("model", "deit_small_patch4_32"))
    head_w = ckpt["model"].get("head.output_project.weight")
    num_channels = head_w.shape[0] if head_w is not None else ckpt_args.get("num_channels", 2)
    query_type = ckpt_args.get("query_type", "patch_cat")

    model = RoPARTViT(
        img_size=img_size, patch_size=patch_size, num_classes=100,
        num_channels=num_channels, num_pairs=ckpt_args.get("num_pairs", 64),
        mask_prob=0.0, cross_attention_query_type=query_type,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, img_size, patch_size


def build_val_loader(ckpt: dict, args, device) -> tuple[torch.utils.data.DataLoader, bool]:
    """Val loader matching the checkpoint's control (so the targets are built the
    same way it trained). Returns ``(loader, rotates)``."""
    ckpt_args = ckpt.get("args", {})
    control = ckpt_args.get("control") or ("raw" if ckpt_args.get("rotation") else "translation")
    extractor, _, rotates = build_extractor(control)
    img_size, patch_size = parse_model_name(ckpt_args.get("model", "deit_small_patch4_32"))
    ds = RoPARTCIFAR(args.data_path, extractor=extractor, rotates=rotates,
                     img_size=img_size, patch_size=patch_size, train=False)
    kw = {"num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
    if args.num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 4
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, **kw)
    return loader, rotates


# --------------------------------------------------------------------------- #
# Dense prediction table
# --------------------------------------------------------------------------- #


@torch.no_grad()
def predict_table(model: RoPARTViT, x: torch.Tensor) -> torch.Tensor:
    """Predict the relative transform for **every** ordered pair ``(i, j)``.

    Encodes once (no position embeddings, no masking — deterministic) and runs the
    relative head over the full ``N*N`` pair grid.

    Returns ``(b, C, N, N)`` where ``[:, :, i, j]`` is the predicted transform from
    reference ``i`` to target ``j`` — same orientation as
    :func:`ropart.targets.build_targets`.
    """
    feats = model._encode(x, add_pos=False, mask=False)[:, :-1, :]  # drop cls
    b, n, _ = feats.shape
    ii, jj = torch.meshgrid(torch.arange(n, device=x.device),
                            torch.arange(n, device=x.device), indexing="ij")
    pairs = torch.stack((ii.reshape(-1), jj.reshape(-1)), dim=1)  # (N*N, 2) = (ref i, tgt j)
    pairs = pairs.unsqueeze(0).expand(b, -1, -1)  # (b, N*N, 2)
    out = model.head(feats, pairs)  # (b, N*N, C)
    return out.permute(0, 2, 1).reshape(b, -1, n, n)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def _accumulate(pred: torch.Tensor, tgt: torch.Tensor, rotates: bool,
                acc: _SqAcc, *, n_triples: int):
    """Fold one batch's dense ``(b, C, N, N)`` pred/target tables into ``acc``."""
    b, c, n, _ = pred.shape
    eye = torch.eye(n, dtype=torch.bool, device=pred.device)
    off = ~eye  # (N, N) off-diagonal mask

    # --- (1) floor comparison + RMSE, off-diagonal pairs only ---------------- #
    for ci, name in ((0, "x"), (1, "y")):
        e = pred[:, ci][:, off]
        t = tgt[:, ci][:, off]
        acc.add(f"se_{name}", ((e - t) ** 2).sum().item(), e.numel())
        # mean-predictor floor = Var(target); accumulate sum and sum-of-squares
        acc.add(f"tsum_{name}", t.sum().item(), t.numel())
        acc.add(f"tsq_{name}", (t ** 2).sum().item(), t.numel())
    # squared translation error per pair (for euclidean RMSE in pixels)
    se_xy = ((pred[:, 0:2] - tgt[:, 0:2]) ** 2).sum(dim=1)[:, off]  # (b, n_off)
    acc.add("se_xy", se_xy.sum().item(), se_xy.numel())

    # --- (2) identity / diagonal: pred at (i, i) ----------------------------- #
    diag_xy = pred[:, 0:2][:, :, eye]  # (b, 2, N)
    acc.add("diag_xy", (diag_xy ** 2).sum().item(), diag_xy[:, 0].numel())
    if rotates and c >= 4:
        # target on the diagonal is (cos, sin) = (1, 0)
        dcos = pred[:, 2][:, eye] - 1.0
        dsin = pred[:, 3][:, eye]
        acc.add("diag_rot", (dcos ** 2 + dsin ** 2).sum().item(), dcos.numel())

    # --- (3) negative symmetry ---------------------------------------------- #
    predT = pred.transpose(-1, -2)  # [:, :, j, i]
    # translation: antisymmetric -> pred + predT == 0
    res_xy = (pred[:, 0:2] + predT[:, 0:2])[:, :, off]
    sig_xy = pred[:, 0:2][:, :, off]
    acc.add("sym_xy_res", (res_xy ** 2).sum().item(), res_xy.numel())
    acc.add("sym_xy_sig", (sig_xy ** 2).sum().item(), sig_xy.numel())
    if rotates and c >= 4:
        res_cos = (pred[:, 2] - predT[:, 2])[:, off]   # cos symmetric
        res_sin = (pred[:, 3] + predT[:, 3])[:, off]   # sin antisymmetric
        sig_rot = pred[:, 2:4][:, :, off]
        acc.add("sym_rot_res", (res_cos ** 2 + res_sin ** 2).sum().item(), res_cos.numel())
        acc.add("sym_rot_sig", (sig_rot ** 2).sum().item(), sig_rot.numel())

    # --- (4) compositional consistency over random distinct triples ---------- #
    i = torch.randint(n, (n_triples,), device=pred.device)
    j = torch.randint(n, (n_triples,), device=pred.device)
    k = torch.randint(n, (n_triples,), device=pred.device)
    ok = (i != j) & (j != k) & (i != k)
    i, j, k = i[ok], j[ok], k[ok]
    if i.numel() > 0:
        ik = pred[:, :, i, k]            # (b, C, T)
        ij = pred[:, :, i, j]
        jk = pred[:, :, j, k]
        res_t = ik[:, 0:2] - (ij[:, 0:2] + jk[:, 0:2])
        acc.add("comp_xy_res", (res_t ** 2).sum().item(), res_t.numel())
        acc.add("comp_xy_sig", (ik[:, 0:2] ** 2).sum().item(), ik[:, 0:2].numel())
        if rotates and c >= 4:
            # angle-additive: φ(i→k) ?= φ(i→j) + φ(j→k)
            a_ik = torch.atan2(ik[:, 3], ik[:, 2])
            a_sum = torch.atan2(ij[:, 3], ij[:, 2]) + torch.atan2(jk[:, 3], jk[:, 2])
            d = (a_ik - a_sum + math.pi) % (2 * math.pi) - math.pi
            acc.add("comp_rot_absdeg", (d.abs() * 180.0 / math.pi).sum().item(), d.numel())


def report(acc: _SqAcc, rotates: bool, patch_size: int) -> dict:
    """Turn the accumulators into the final metrics dict (+ pretty print)."""
    floor_x = acc.mean("tsq_x") - (acc._sq["tsum_x"] / acc._n["tsum_x"]) ** 2
    floor_y = acc.mean("tsq_y") - (acc._sq["tsum_y"] / acc._n["tsum_y"]) ** 2
    mse_x, mse_y = acc.mean("se_x"), acc.mean("se_y")
    rmse_px = math.sqrt(acc.mean("se_xy")) * patch_size

    m = {
        "mse_x": mse_x, "floor_x": floor_x, "mse_x_vs_floor": mse_x / floor_x,
        "mse_y": mse_y, "floor_y": floor_y, "mse_y_vs_floor": mse_y / floor_y,
        "trans_rmse_px": rmse_px,
        "identity_rmse_px": acc.rms("diag_xy") * patch_size,
        "neg_sym_xy_ratio": acc.ratio_rms("sym_xy_res", "sym_xy_sig"),
        "comp_xy_ratio": acc.ratio_rms("comp_xy_res", "comp_xy_sig"),
    }
    if rotates:
        m["identity_rot_rmse"] = acc.rms("diag_rot")
        m["neg_sym_rot_ratio"] = acc.ratio_rms("sym_rot_res", "sym_rot_sig")
        m["comp_rot_mae_deg"] = acc.mean("comp_rot_absdeg")

    print("\n=== RoPART intrinsic correctness (val, all ordered pairs) ===")
    print(f"(1) floor    mse_x={mse_x:.4f} / floor_x={floor_x:.4f}  ({m['mse_x_vs_floor']:.2%} of floor)")
    print(f"             mse_y={mse_y:.4f} / floor_y={floor_y:.4f}  ({m['mse_y_vs_floor']:.2%} of floor)")
    print(f"             translation RMSE = {rmse_px:.3f} px (image is {patch_size * (32 // patch_size)} px)")
    print(f"(2) identity translation RMSE at (i,i) = {m['identity_rmse_px']:.3f} px  (want ~0)")
    if rotates:
        print(f"             rotation (cos,sin) RMSE at (i,i) = {m['identity_rot_rmse']:.4f}  (want ~0)")
    print(f"(3) neg-sym  xy residual/signal = {m['neg_sym_xy_ratio']:.3%}  (want ~0)")
    if rotates:
        print(f"             rot residual/signal = {m['neg_sym_rot_ratio']:.3%}  (want ~0)")
    print(f"(4) compose  xy residual/signal = {m['comp_xy_ratio']:.3%}  (want ~0)")
    if rotates:
        print(f"             rot angle MAE = {m['comp_rot_mae_deg']:.2f}°  (want ~0)")
    print("=============================================================\n")
    return m


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


@torch.no_grad()
def run(args, device):
    ckpt = torch.load(args.resume, map_location="cpu")
    model, img_size, patch_size = load_model(ckpt, device)
    loader, rotates = build_val_loader(ckpt, args, device)
    print(f"checkpoint epoch={ckpt.get('epoch')}  rotates={rotates}  "
          f"num_channels={model.num_channels}  query={model.head.query_type}")

    acc = _SqAcc()
    for step, (packed, boxes, angles) in enumerate(loader):
        if args.max_batches is not None and step >= args.max_batches:
            break
        packed = packed.to(device, non_blocking=True)
        boxes = boxes.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        pred = predict_table(model, packed)
        tgt = build_targets(boxes, angles if rotates else None,
                            rotates=rotates, centered=True, normalize_by=patch_size)
        _accumulate(pred, tgt, rotates, acc, n_triples=args.num_triples)

    metrics = report(acc, rotates, patch_size)
    out_path = Path(args.resume).with_name("eval_intrinsic.json")
    out_path.write_text(json.dumps({"epoch": ckpt.get("epoch"), "rotates": rotates, **metrics}, indent=2))
    print(f"wrote {out_path}")


def get_args():
    p = argparse.ArgumentParser("RoPART intrinsic correctness eval")
    p.add_argument("--resume", required=True, help="path to a pretrain checkpoint.pth")
    p.add_argument("--data-path", default="./data")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--batch-size", default=128, type=int)
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--max-batches", default=None, type=int, help="cap val batches (smoke)")
    p.add_argument("--num-triples", default=4096, type=int, help="random triples/batch for composition")
    return p.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"device: {device}")
    run(args, device)


if __name__ == "__main__":
    main()
