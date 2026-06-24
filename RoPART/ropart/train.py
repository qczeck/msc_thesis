"""RoPART training entry point — pretraining and linear-probe.

Run as a module from the ``RoPART/`` directory so both ``ropart`` and ``helpers``
are importable:

    python -m ropart.train --control translation --epochs 100 ...
    python -m ropart.train --control raw         --epochs 100 ...     # rotation
    python -m ropart.train --linear-probe --resume out/.../checkpoint.pth

Device is auto-selected (cuda -> mps -> cpu); AMP is used only on CUDA. wandb runs
online by default (live remote monitoring for unattended SSH runs) with a graceful
offline fallback and resumable run ids.
"""

from __future__ import annotations

import argparse
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import torch

from ropart.controls import available_controls, build_extractor
from ropart.data import RoPARTCIFAR, build_cls_dataset
from ropart.engine import cls_run_epoch, pretrain_run_epoch
from ropart.losses import RelativeMSE
from ropart.model import RoPARTViT


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #


def pick_device(name: str) -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_model_name(name: str) -> tuple[int, int]:
    """'deit_small_patch4_32' -> (img_size=32, patch_size=4)."""
    img_size = int(name.split("_")[-1])
    patch_size = int(name.split("_")[-2][-1])
    return img_size, patch_size


def lr_at(epoch: int, args) -> float:
    if epoch < args.warmup_epochs:
        return args.warmup_lr + (args.lr - args.warmup_lr) * epoch / max(1, args.warmup_epochs)
    progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
    return args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def param_groups(model, weight_decay):
    no_wd = model.no_weight_decay() if hasattr(model, "no_weight_decay") else set()
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias") or name in no_wd:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _slug(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z._-]+", "-", s).strip("-")


class WandbRun:
    """Thin wandb wrapper: chosen mode, resumable id, graceful offline fallback."""

    def __init__(self, args, config: dict):
        self.enabled = args.wandb_mode != "disabled"
        self.run = None
        if not self.enabled:
            return
        import wandb

        run_id = args.wandb_id or _slug(args.wandb_name)
        common = dict(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_name,
            id=run_id,
            resume="allow",
            config=config,
        )
        try:
            self.run = wandb.init(mode=args.wandb_mode, **common)
        except Exception as e:  # not logged in / no network
            print(f"[wandb] init failed ({e}); retrying offline")
            try:
                self.run = wandb.init(mode="offline", **common)
            except Exception as e2:
                print(f"[wandb] offline failed ({e2}); disabling wandb")
                self.enabled = False

    def log(self, data: dict, step: int | None = None):
        if self.enabled:
            import wandb

            wandb.log(data, step=step)

    def finish(self):
        if self.enabled:
            import wandb

            wandb.finish()


def save_checkpoint(path: Path, model, optimizer, epoch, args, extra: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            **extra,
        },
        path,
    )


# --------------------------------------------------------------------------- #
# Pretraining
# --------------------------------------------------------------------------- #


def run_pretrain(args, device):
    img_size, patch_size = parse_model_name(args.model)

    control = args.control or ("raw" if args.rotation else "translation")
    control_kw = {}
    if control == "supersample":
        control_kw["factor"] = args.supersample
    elif control == "dominant":
        control_kw["sigma"] = args.sigma_lp
    elif control == "randomised":
        control_kw["sigma_max"] = args.sigma_max
    extractor, num_channels, rotates = build_extractor(control, **control_kw)
    print(f"control={control}  num_channels={num_channels}  rotates={rotates}")

    train_ds = RoPARTCIFAR(args.data_path, extractor=extractor, rotates=rotates,
                           img_size=img_size, patch_size=patch_size, train=True)
    val_ds = RoPARTCIFAR(args.data_path, extractor=extractor, rotates=rotates,
                         img_size=img_size, patch_size=patch_size, train=False)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = RoPARTViT(
        img_size=img_size, patch_size=patch_size, num_classes=100,
        num_channels=num_channels, num_pairs=args.num_pairs, mask_prob=args.mask_prob,
        drop_path_rate=args.drop_path,
    ).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = RelativeMSE(w_xy=args.w_xy, w_phi=args.w_phi)
    optimizer = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    wb = WandbRun(args, vars(args))
    out_dir = Path(args.output_dir)

    for epoch in range(start_epoch, args.epochs):
        lr = lr_at(epoch, args)
        for g in optimizer.param_groups:
            g["lr"] = lr
        t0 = time.time()
        train_stats = pretrain_run_epoch(
            model, train_loader, device, criterion,
            patch_size=patch_size, rotates=rotates, optimizer=optimizer, scaler=scaler,
            max_norm=args.clip_grad, max_steps=args.max_steps,
        )
        val_stats = pretrain_run_epoch(
            model, val_loader, device, criterion,
            patch_size=patch_size, rotates=rotates, max_steps=args.max_steps,
        )
        dt = time.time() - t0
        log = {"epoch": epoch, "lr": lr, "epoch_time_s": dt,
               **{f"train/{k}": v for k, v in train_stats.items()},
               **{f"val/{k}": v for k, v in val_stats.items()}}
        wb.log(log, step=epoch)
        print(f"[{epoch}] lr={lr:.2e} train_loss={train_stats['loss']:.4f} "
              f"val_loss={val_stats['loss']:.4f} "
              + (f"val_ang_err={val_stats.get('ang_err_deg', float('nan')):.1f}° " if rotates else "")
              + f"({dt:.0f}s)")
        save_checkpoint(out_dir / "checkpoint.pth", model, optimizer, epoch, args,
                        extra={"wandb_id": args.wandb_id or _slug(args.wandb_name)})

    wb.finish()


# --------------------------------------------------------------------------- #
# Linear probe
# --------------------------------------------------------------------------- #


def run_linear_probe(args, device):
    assert args.resume, "--linear-probe needs --resume <pretrain checkpoint>"
    ckpt = torch.load(args.resume, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    img_size, patch_size = parse_model_name(ckpt_args.get("model", args.model))
    num_channels = ckpt_args.get("num_channels", 2)
    # num_channels may not be stored directly; infer from head weight if needed
    head_w = ckpt["model"].get("head.output_project.weight")
    if head_w is not None:
        num_channels = head_w.shape[0]

    model = RoPARTViT(img_size=img_size, patch_size=patch_size, num_classes=100,
                      num_channels=num_channels, mask_prob=0.0).to(device)
    missing = model.load_state_dict(ckpt["model"], strict=False)
    print(f"probe: loaded encoder (missing={len(missing.missing_keys)}, "
          f"unexpected={len(missing.unexpected_keys)})")

    # freeze everything, re-init + train the classifier only
    for p in model.parameters():
        p.requires_grad = False
    model.clf = torch.nn.Linear(model.embed_dim, 100).to(device)
    for p in model.clf.parameters():
        p.requires_grad = True

    train_ds, val_ds = build_cls_dataset(args.data_path, img_size=img_size)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"))

    optimizer = torch.optim.AdamW(model.clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    wb = WandbRun(args, vars(args))

    best = 0.0
    for epoch in range(args.epochs):
        lr = lr_at(epoch, args)
        for g in optimizer.param_groups:
            g["lr"] = lr
        train_stats = cls_run_epoch(model, train_loader, device, optimizer=optimizer,
                                    scaler=scaler, eval_features=True, max_steps=args.max_steps)
        val_stats = cls_run_epoch(model, val_loader, device, max_steps=args.max_steps)
        best = max(best, val_stats["acc1"])
        wb.log({"epoch": epoch, "lr": lr,
                **{f"probe_train/{k}": v for k, v in train_stats.items()},
                **{f"probe_val/{k}": v for k, v in val_stats.items()},
                "probe_val/best_acc1": best}, step=epoch)
        print(f"[probe {epoch}] train_acc1={train_stats['acc1']:.2f} "
              f"val_acc1={val_stats['acc1']:.2f} best={best:.2f}")
    wb.finish()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def get_args():
    p = argparse.ArgumentParser("RoPART training")
    p.add_argument("--model", default="deit_small_patch4_32")
    p.add_argument("--data-path", default="./data")
    p.add_argument("--output_dir", default="./out")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--epochs", default=100, type=int)
    p.add_argument("--batch-size", default=128, type=int)
    p.add_argument("--max-steps", default=None, type=int, help="cap steps/epoch (smoke/diagnostics)")
    # optim / schedule
    p.add_argument("--lr", default=5e-4, type=float)
    p.add_argument("--warmup-lr", default=1e-6, type=float)
    p.add_argument("--min-lr", default=1e-5, type=float)
    p.add_argument("--warmup-epochs", default=5, type=int)
    p.add_argument("--weight-decay", default=0.05, type=float)
    p.add_argument("--clip-grad", default=None, type=float)
    p.add_argument("--drop-path", default=0.0, type=float)
    # pretext
    p.add_argument("--control", default=None, choices=available_controls(),
                   help="patch-extraction control; default translation (or raw if --rotation)")
    p.add_argument("--rotation", action="store_true", help="shorthand: use control 'raw' if --control unset")
    p.add_argument("--num_pairs", default=64, type=int)
    p.add_argument("--mask-prob", default=0.0, type=float)
    p.add_argument("--w-xy", default=1.0, type=float)
    p.add_argument("--w-phi", default=1.0, type=float)
    # control knobs
    p.add_argument("--supersample", default=4, type=int)
    p.add_argument("--sigma-lp", default=1.0, type=float)
    p.add_argument("--sigma-max", default=0.8, type=float)
    # mode
    p.add_argument("--linear-probe", action="store_true")
    p.add_argument("--resume", default="")
    # wandb
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-project", default="ropart")
    p.add_argument("--wandb-entity", default="")
    p.add_argument("--wandb-name", default="ropart-run")
    p.add_argument("--wandb-id", default="")
    return p.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"device: {device}")
    if args.linear_probe:
        run_linear_probe(args, device)
    else:
        run_pretrain(args, device)


if __name__ == "__main__":
    main()
