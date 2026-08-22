"""Score a **finished** HLW horizon finetune on one split, without retraining.

Why this exists as its own entry point. ``ropart.train --finetune`` always trains: it
has no path that loads finished weights and merely evaluates them. The two flags that
look like they might are ``--init-from`` (seed a finetune from a *pretrain* checkpoint)
and ``--resume`` (continue *this* finetune) — neither means "score these weights once".
Adding a third mode to ``run_finetune`` would put the scoring path inside the function
the completed runs depend on; this file follows the ``run_horizon_doc.sh`` precedent and
keeps them apart.

**The load must be strict, and it must not go through the finetune loader.**
:func:`ropart.train.load_encoder_for_finetune` deliberately drops ``clf.*`` and
re-initialises ``pos_embed`` — correct when seeding a finetune from a pretrain
checkpoint, destructive here, since it would discard the trained head and position
embeddings and then happily score a half-initialised model to a plausible-looking
number. So the architecture is rebuilt by that same function (guaranteeing it matches
how the model was constructed during training) and the finetuned weights are then
loaded over it with ``strict=True``. Every key must land; anything else raises.

**Gate before trusting a test number.** Run this on ``--split val`` first. It must
reproduce the arm's final ``val_theta_mae`` from ``run.log`` to ~1e-3. The test split has
no such reference, so it cannot be the first thing scored — a silently mangled load has
no other tell.

Per-image errors are written to CSV so arms can be compared **paired** rather than by
their aggregate MAEs; see :mod:`ropart.scripts.paired_horizon_test`.

Usage::

    HLW_TARGET_STATS=-4.1879e-04,0.291614,7.0494e-02,0.549827 \\
    python -m ropart.scripts.score_horizon \\
        --checkpoint $WS/out/hlw_ft_base_p32_v2/checkpoint.pth \\
        --data-path /vol/bitbucket/msk123/hlwv2/hlw224 --split val
"""

from __future__ import annotations

import argparse
import csv
from argparse import Namespace
from pathlib import Path

import torch

from ropart.hlw import TARGET_MEAN, TARGET_STD, HLWDataset, horizon_run_epoch
from ropart.train import load_encoder_for_finetune, loader_kwargs


def load_finetuned(checkpoint: str, device: torch.device) -> tuple[torch.nn.Module, int, dict]:
    """Rebuild a finetuned horizon model and load its weights strictly.

    Returns ``(model, img_size, finetune_args)``.

    The architecture is reconstructed by :func:`load_encoder_for_finetune` from the
    **pretrain** checkpoint this finetune recorded in its own ``args["init_from"]``, so
    the module tree is built by exactly the code path that built it during training —
    there is no second copy of the construction logic that could drift. Its weights are
    then overwritten by the finetune checkpoint's ``strict=True``, which is the actual
    guarantee: a shape or naming mismatch raises here rather than producing a number.
    """
    ckpt = torch.load(checkpoint, map_location="cpu")
    ft_args = ckpt.get("args", {})
    init_from = ft_args.get("init_from", "")
    if not init_from:
        raise SystemExit(
            f"{checkpoint} records no 'init_from' — it does not look like a finetune "
            "checkpoint, and the architecture cannot be reconstructed from it alone.")
    if not Path(init_from).exists():
        raise SystemExit(
            f"the pretrain checkpoint this finetune was seeded from is missing:\n"
            f"  {init_from}\n"
            "It is needed to rebuild the architecture (num_channels, num_pairs and the "
            "cross-attention query type are properties of the pretrain run).")

    ns = Namespace(**ft_args)
    ns.num_classes = 2                      # (theta, rho)
    model, img_size = load_encoder_for_finetune(init_from, ns, device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    print(f"score: loaded finetuned weights from {checkpoint} "
          f"(epoch {ckpt.get('epoch')}, strict=True)")
    return model, img_size, ft_args


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="a finished horizon finetune's checkpoint.pth")
    p.add_argument("--data-path", required=True, help="HLW root (images/, split/, metadata.csv)")
    p.add_argument("--split", default="val", choices=["val", "test", "test_seen", "test_heldout"],
                   help="split to score. Run 'val' first as the gate (see module docstring)")
    p.add_argument("--batch-size", default=192, type=int)
    p.add_argument("--num-workers", default=6, type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="",
                   help="CSV path for the per-image errors "
                        "(default: scores_<split>.csv beside the checkpoint)")
    args = p.parse_args()

    device = torch.device(args.device)
    model, img_size, ft_args = load_finetuned(args.checkpoint, device)

    # Provenance, for the same reason run_finetune prints it: a wrong centre and scale is
    # silent, and HLW_TARGET_STATS is an env var that is easy to forget on a fresh shell.
    print(f"hlw data: {args.data_path} split: {args.split}")
    print(f"hlw target_mean: {TARGET_MEAN} target_std: {TARGET_STD}")
    trained_on = ft_args.get("data_path", "")
    if trained_on and Path(trained_on).name != Path(args.data_path).name:
        print(f"score: WARNING this arm trained on {trained_on}, scoring against "
              f"{args.data_path}")

    ds = HLWDataset(args.data_path, args.split, img_size, train=False)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        **loader_kwargs(Namespace(num_workers=args.num_workers), device))

    # No scaler and no autocast — this mirrors run_finetune's *eval* call exactly, which
    # passes neither. Scoring under AMP when training validated in fp32 would move the
    # number for reasons that have nothing to do with the arm.
    stats = horizon_run_epoch(model, loader, device, per_image=True)

    theta_err = stats["theta_err"]
    if len(theta_err) != len(ds):
        raise SystemExit(f"per-image count {len(theta_err)} != dataset size {len(ds)}")

    out = Path(args.out) if args.out else Path(args.checkpoint).parent / f"scores_{args.split}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "theta_err_deg", "rho_err", "horizon_err"])
        for name, dt, dr, he in zip(ds.names, theta_err.tolist(),
                                    stats["rho_err"].tolist(),
                                    stats["horizon_err"].tolist()):
            w.writerow([name, f"{dt:.8f}", f"{dr:.8f}", f"{he:.8f}"])

    print(f"[score] {Path(args.checkpoint).parent.name} split={args.split} n={len(ds)} "
          f"theta_mae={stats['theta_mae']:.4f} rho_mae={stats['rho_mae']:.4f} "
          f"auc={stats['auc']:.4f}")
    print(f"[score] per-image errors -> {out}")


if __name__ == "__main__":
    main()
