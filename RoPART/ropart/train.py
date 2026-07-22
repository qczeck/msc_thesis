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

from ropart.controls import available_controls, build_extractor, control_kwargs
from ropart.data import (
    RoPARTCIFAR,
    RoPARTImageFolder,
    build_cls_dataset,
    build_cls_dataset_imagenet,
)
from ropart.engine import cls_run_epoch, pretrain_run_epoch
from ropart.losses import RelativeMSE
from ropart.model import RoPARTViT, model_config
from ropart.targets import build_targets, gather_pairs


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
    """``'deit_base_patch16_224'`` -> ``(img_size, patch_size)`` via the registry.

    Thin compat shim over :func:`ropart.model.model_config` (used by ``ropart.eval``);
    unlike the old string-slicing version it parses two-digit patch sizes correctly.
    """
    cfg = model_config(name)
    return cfg["img_size"], cfg["patch_size"]


# Controls that read exact rot90 pixels and therefore require quad angle sampling.
# ``quad`` supervises rotation (num_channels=4); ``quad_ch2`` rotates pixels but
# leaves rotation unsupervised (num_channels=2).
_QUAD_CONTROLS = {"quad", "quad_ch2"}


def resolve_control(args) -> str:
    """Resolve the control name, binding the quad controls <-> quad sampler pair.

    The quad controls (exact ``rot90``) and quad angle sampling are a matched pair:
    passing either ``--control quad``/``--control quad_ch2`` or ``--rotation-set quad``
    implies the other, and a conflicting combination (e.g. ``--control raw
    --rotation-set quad``) is rejected — quad angles through a bilinear extractor, or
    continuous angles through ``rot90``, would desync the target from the rotated
    pixels. May mutate ``args.rotation_set`` so it is recorded consistently in the
    checkpoint.
    """
    control = args.control or ("raw" if args.rotation else "translation")
    if args.rotation_set == "quad" and args.control is None and not args.rotation:
        control = "quad"
    is_quad = control in _QUAD_CONTROLS
    if is_quad:
        args.rotation_set = "quad"
    if is_quad != (args.rotation_set == "quad"):
        raise SystemExit(
            f"--rotation-set quad pairs only with a quad control {sorted(_QUAD_CONTROLS)} "
            f"(got control={control!r}, rotation-set={args.rotation_set!r})"
        )
    return control


def build_pretrain_dataset(args, cfg: dict, extractor, rotates: bool, *, train: bool):
    """CIFAR-100 or ImageNet-100 pretraining dataset, per ``--data-set``.

    Both share the off-grid sampling + control extraction + retile packing; they
    differ only in the image source (CIFAR's in-memory array vs ImageFolder JPEGs).
    """
    max_angle = (
        math.radians(args.rotation_max_deg)
        if args.rotation_max_deg is not None else None
    )
    common = dict(
        extractor=extractor, rotates=rotates, max_angle=max_angle,
        angle_set=args.rotation_set,
        img_size=cfg["img_size"], patch_size=cfg["patch_size"], train=train,
    )
    if args.data_set == "IMAGENET":
        return RoPARTImageFolder(args.data_path, **common)
    return RoPARTCIFAR(args.data_path, **common)


def lr_at(epoch: int, args) -> float:
    if epoch < args.warmup_epochs:
        return args.warmup_lr + (args.lr - args.warmup_lr) * epoch / max(1, args.warmup_epochs)
    progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
    return args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def loader_kwargs(args, device) -> dict:
    """DataLoader options shared by every loader.

    ``persistent_workers`` keeps worker processes alive across epochs (otherwise
    they are torn down and respawned each epoch — the GPU-utilisation sawtooth that
    drops to 0 at every epoch boundary); ``prefetch_factor`` deepens the per-worker
    queue so the GPU is not starved while batches are sampled (off-grid sampling and
    rotated cropping are CPU-bound). Both require ``num_workers > 0``.
    """
    kw: dict = {"num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
    if args.num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 4
    return kw


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


def resolve_run_id(args, resumed_id: str | None) -> str:
    """Pick the wandb run id (also used as the display name).

    Priority: an explicit ``--wandb-id`` wins; otherwise a ``resumed_id`` recovered
    from a checkpoint continues logging on the *same* run; otherwise a fresh
    timestamped id (``<name>-YYYYmmdd-HHMMSS``) so each launch is its own run and
    sorts chronologically in the dashboard (no more id collisions on relaunch).
    """
    if args.wandb_id:
        return args.wandb_id
    if resumed_id:
        return resumed_id
    return f"{_slug(args.wandb_name)}-{time.strftime('%Y%m%d-%H%M%S')}"


class WandbRun:
    """Thin wandb wrapper: chosen mode, resumable id, graceful offline fallback."""

    def __init__(self, args, config: dict):
        self.enabled = args.wandb_mode != "disabled"
        self.run = None
        if not self.enabled:
            return
        import wandb

        run_id = resolve_run_id(args, None)
        common = dict(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=run_id,
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
        # A wandb hiccup (dropped socket, dead internal process, sync error) must
        # never kill a multi-hour pretraining run: on failure, warn once and disable
        # wandb for the rest of the run — training and checkpointing carry on.
        if not self.enabled:
            return
        import wandb

        try:
            wandb.log(data, step=step)
        except Exception as e:
            print(f"[wandb] log failed ({e}); disabling wandb for the rest of the run")
            self.enabled = False

    def finish(self):
        if self.enabled:
            import wandb

            try:
                wandb.finish()
            except Exception as e:
                print(f"[wandb] finish failed ({e}); ignoring")


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
    cfg = model_config(args.model)
    patch_size = cfg["patch_size"]

    control = resolve_control(args)
    extractor, num_channels, rotates = build_extractor(
        control, **control_kwargs(control, vars(args)))
    # `rotates` drives pixel rotation + angle sampling in the dataset; whether
    # rotation is *supervised* (targets carry (cos Δφ, sin Δφ)) is decided by the
    # head width. They coincide for every control except the ch2 pair `quad_ch2` /
    # `ss_ch2` (rotated pixels, translation-only target), which decouple the two to
    # isolate the cause of the translation collapse in the ch=4 runs.
    supervise_rot = num_channels >= 4
    print(f"control={control}  num_channels={num_channels}  rotates_pixels={rotates}  "
          f"supervise_rot={supervise_rot}  rotation_set={args.rotation_set}  "
          f"data={args.data_set}  model={args.model}")

    train_ds = build_pretrain_dataset(args, cfg, extractor, rotates, train=True)
    val_ds = build_pretrain_dataset(args, cfg, extractor, rotates, train=False)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        **loader_kwargs(args, device),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        **loader_kwargs(args, device),
    )

    model = RoPARTViT(
        **cfg, num_classes=args.num_classes,
        num_channels=num_channels, num_pairs=args.num_pairs, mask_prob=args.mask_prob,
        drop_path_rate=args.drop_path, cross_attention_query_type=args.query_type,
    ).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = RelativeMSE(w_xy=args.w_xy, w_phi=args.w_phi, balance=args.loss_balance,
                            eps=args.loss_eps)
    optimizer = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16

    start_epoch = 0
    resumed_id = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        resumed_id = ckpt.get("wandb_id")
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    # pin the run id once: fresh timestamped id for new runs, recovered id on resume,
    # so checkpoints carry it and a resume continues the same wandb run.
    args.wandb_id = resolve_run_id(args, resumed_id)
    wb = WandbRun(args, vars(args))
    out_dir = Path(args.output_dir)

    for epoch in range(start_epoch, args.epochs):
        lr = lr_at(epoch, args)
        for g in optimizer.param_groups:
            g["lr"] = lr
        t0 = time.time()
        train_stats = pretrain_run_epoch(
            model, train_loader, device, criterion,
            patch_size=patch_size, rotates=supervise_rot, optimizer=optimizer, scaler=scaler,
            amp_dtype=amp_dtype, max_norm=args.clip_grad, max_steps=args.max_steps,
        )
        # Validation is a full 10k-image sweep; running it every epoch is the main
        # source of the epoch-boundary GPU-utilisation dip. Gate it behind
        # --eval-every (always evaluate the final epoch).
        do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs - 1)
        val_stats = (
            pretrain_run_epoch(model, val_loader, device, criterion,
                               patch_size=patch_size, rotates=supervise_rot,
                               amp_dtype=amp_dtype, max_steps=args.max_steps)
            if do_eval else None
        )
        dt = time.time() - t0
        log = {"epoch": epoch, "lr": lr, "epoch_time_s": dt,
               **{f"train/{k}": v for k, v in train_stats.items()}}
        if val_stats is not None:
            log.update({f"val/{k}": v for k, v in val_stats.items()})
        wb.log(log, step=epoch)
        print(f"[{epoch}] lr={lr:.2e} train_loss={train_stats['loss']:.4f} "
              + (f"val_loss={val_stats['loss']:.4f} "
                 + (f"val_ang_err={val_stats.get('ang_err_deg', float('nan')):.1f}° " if supervise_rot else "")
                 if val_stats is not None else "")
              + f"({dt:.0f}s)")
        save_checkpoint(out_dir / "checkpoint.pth", model, optimizer, epoch, args,
                        extra={"wandb_id": args.wandb_id})

    wb.finish()


# --------------------------------------------------------------------------- #
# Single-batch overfit diagnostic
# --------------------------------------------------------------------------- #


def run_overfit(args, device):
    """Fit one fixed batch for ``--overfit-steps`` steps; loss should fall well
    below the mean-predictor floor if the pretext path is correctly wired.

    This is the decisive wiring-vs-task test: the dataset resamples patches every
    ``__getitem__``, so we pull a single batch *once* and loop the optimiser on it
    (fp32, no AMP, no LR schedule — to remove those as variables). If loss stays
    pinned at the floor here, the bug is in the model/loss/gradient path, not the
    data scale or the schedule.
    """
    cfg = model_config(args.model)
    patch_size = cfg["patch_size"]
    control = resolve_control(args)
    extractor, num_channels, rotates = build_extractor(
        control, **control_kwargs(control, vars(args)))
    supervise_rot = num_channels >= 4  # see run_pretrain: pixels rotate iff `rotates`
    print(f"[overfit] control={control} num_channels={num_channels} rotates_pixels={rotates} "
          f"supervise_rot={supervise_rot} rotation_set={args.rotation_set} "
          f"data={args.data_set} model={args.model}")

    ds = build_pretrain_dataset(args, cfg, extractor, rotates, train=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                         num_workers=0, drop_last=True)
    packed, boxes, angles = next(iter(loader))
    packed = packed.to(device)
    boxes = boxes.to(device)
    angles = angles.to(device)

    model = RoPARTViT(**cfg, num_classes=args.num_classes,
                      num_channels=num_channels, num_pairs=args.num_pairs,
                      mask_prob=args.mask_prob,
                      cross_attention_query_type=args.query_type).to(device)
    criterion = RelativeMSE(w_xy=args.w_xy, w_phi=args.w_phi, balance=args.loss_balance,
                            eps=args.loss_eps)
    optimizer = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr)

    targets_full = build_targets(boxes, angles if supervise_rot else None, rotates=supervise_rot,
                                 centered=True, normalize_by=patch_size)
    floor = (targets_full[:, 0].var(unbiased=False) + targets_full[:, 1].var(unbiased=False)).item()
    print(f"[overfit] xy floor (target to beat) ~ {floor:.3f}")

    model.train()
    for step in range(args.overfit_steps):
        outputs, indices = model.forward_pretrain(packed)
        targets = gather_pairs(targets_full, indices)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
        optimizer.step()
        if step % 50 == 0 or step == args.overfit_steps - 1:
            print(f"[overfit {step:4d}] loss={loss.item():.4f} grad_norm={float(gnorm):.3f}")
    print("[overfit] done. loss << floor => wiring sound (cause is scale/schedule); "
          "loss ~ floor => wiring/gradient bug.")


# --------------------------------------------------------------------------- #
# Linear probe
# --------------------------------------------------------------------------- #


def run_linear_probe(args, device):
    assert args.resume, "--linear-probe needs --resume <pretrain checkpoint>"
    ckpt = torch.load(args.resume, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    cfg = model_config(ckpt_args.get("model", args.model))
    img_size = cfg["img_size"]
    num_channels = ckpt_args.get("num_channels", 2)
    # num_channels may not be stored directly; infer from head weight if needed
    head_w = ckpt["model"].get("head.output_project.weight")
    if head_w is not None:
        num_channels = head_w.shape[0]

    model = RoPARTViT(**cfg, num_classes=args.num_classes,
                      num_channels=num_channels, mask_prob=0.0).to(device)
    missing = model.load_state_dict(ckpt["model"], strict=False)
    print(f"probe: loaded encoder (missing={len(missing.missing_keys)}, "
          f"unexpected={len(missing.unexpected_keys)})")

    # freeze everything, re-init + train the classifier only
    for p in model.parameters():
        p.requires_grad = False
    model.clf = torch.nn.Linear(model.embed_dim, args.num_classes).to(device)
    for p in model.clf.parameters():
        p.requires_grad = True

    if args.data_set == "IMAGENET":
        train_ds, val_ds = build_cls_dataset_imagenet(args.data_path, img_size=img_size)
    else:
        train_ds, val_ds = build_cls_dataset(args.data_path, img_size=img_size)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        **loader_kwargs(args, device))
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        **loader_kwargs(args, device))

    optimizer = torch.optim.AdamW(model.clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    wb = WandbRun(args, vars(args))

    best = 0.0
    for epoch in range(args.epochs):
        lr = lr_at(epoch, args)
        for g in optimizer.param_groups:
            g["lr"] = lr
        train_stats = cls_run_epoch(model, train_loader, device, optimizer=optimizer,
                                    scaler=scaler, amp_dtype=amp_dtype, eval_features=True,
                                    max_steps=args.max_steps)
        val_stats = cls_run_epoch(model, val_loader, device, amp_dtype=amp_dtype,
                                  max_steps=args.max_steps)
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
    p.add_argument("--data-set", default="CIFAR", choices=["CIFAR", "IMAGENET"],
                   help="CIFAR-100 (in-memory) or ImageNet-100 (ImageFolder at --data-path)")
    p.add_argument("--num-classes", default=100, type=int,
                   help="classifier head width (probe/finetune); pretext is label-free")
    p.add_argument("--data-path", default="./data")
    p.add_argument("--output_dir", default="./out")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--epochs", default=100, type=int)
    p.add_argument("--batch-size", default=128, type=int)
    p.add_argument("--max-steps", default=None, type=int, help="cap steps/epoch (smoke/diagnostics)")
    p.add_argument("--eval-every", default=1, type=int,
                   help="run validation every N epochs (and the last); higher = less epoch-boundary GPU idle")
    # optim / schedule
    p.add_argument("--lr", default=5e-4, type=float)
    p.add_argument("--warmup-lr", default=1e-6, type=float)
    p.add_argument("--min-lr", default=1e-5, type=float)
    p.add_argument("--warmup-epochs", default=5, type=int)
    p.add_argument("--weight-decay", default=0.05, type=float)
    p.add_argument("--clip-grad", default=None, type=float)
    p.add_argument("--drop-path", default=0.0, type=float)
    p.add_argument("--amp-dtype", default="float16", choices=["float16", "bfloat16"],
                   help="CUDA autocast dtype. 'float16' (default) keeps existing runs "
                        "byte-identical; 'bfloat16' has fp32 exponent range so ViT-B "
                        "attention logits do not overflow to inf/NaN (fixes the quad "
                        "P=32 crash). No-op off CUDA.")
    # pretext
    p.add_argument("--control", default=None, choices=available_controls(),
                   help="patch-extraction control; default translation (or raw if --rotation)")
    p.add_argument("--rotation", action="store_true", help="shorthand: use control 'raw' if --control unset")
    p.add_argument("--num_pairs", default=64, type=int)
    p.add_argument("--query-type", default="patch_cat", choices=["positional", "patch_cat"],
                   help="relative-head query: the two patch features (patch_cat, default — learns) "
                        "or a fixed pair code (positional, upstream default — does not learn on this config)")
    p.add_argument("--mask-prob", default=0.0, type=float)
    p.add_argument("--w-xy", default=1.0, type=float)
    p.add_argument("--w-phi", default=1.0, type=float)
    p.add_argument("--loss-balance", default="none", choices=["none", "variance"],
                   help="'none' (default, baseline-unchanged): raw grouped MSE; "
                        "'variance': normalise each channel by its batch target "
                        "variance so the translation and rotation groups are on a "
                        "comparable scale (proper Δφ weighting)")
    p.add_argument("--loss-eps", default=1e-3, type=float,
                   help="lower clamp on the per-channel variance divisor (variance "
                        "balance only). Guards against the cos-Δφ-variance -> 0 blow-up "
                        "at small rotation ranges that NaNs training; 1e-3 caps "
                        "amplification at ~1000x and is inert for moderate/quad ranges")
    p.add_argument("--rotation-max-deg", default=None, type=float,
                   help="half-range for per-patch orientation in degrees; unset = "
                        "full circle [0,360). e.g. 30 samples φ in [-30°, +30°] "
                        "(bounded-rotation experiment)")
    p.add_argument("--rotation-set", default="continuous", choices=["continuous", "quad"],
                   help="'continuous' (default): real-valued angles; 'quad': discrete "
                        "{0,90,180,270}° via exact rot90 (RotNet-style, interpolation-"
                        "free). Pairs with --control quad (either flag implies the other)")
    # control knobs
    p.add_argument("--supersample", default=4, type=int)
    p.add_argument("--sigma-lp", default=1.0, type=float)
    p.add_argument("--sigma-max", default=0.8, type=float)
    # mode
    p.add_argument("--linear-probe", action="store_true")
    p.add_argument("--overfit-steps", default=0, type=int,
                   help="diagnostic: fit one fixed batch for N steps (0=off)")
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
    if args.overfit_steps > 0:
        run_overfit(args, device)
    elif args.linear_probe:
        run_linear_probe(args, device)
    else:
        run_pretrain(args, device)


if __name__ == "__main__":
    main()
