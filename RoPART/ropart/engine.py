"""Train / eval loops and intrinsic metrics.

Pretraining logs per-channel MSE against the **mean-predictor floors**: for ~uniform targets, predicting the mean gives ``mse_x/mse_y ==
Var(Δx/Δy)`` and the ``(cos Δφ, sin Δφ)`` pair MSE collapses to ``1.0``. Comparing
the live MSE to these floors is the quickest read on whether the channels are
actually learning. For rotation we also report the mean **angular error** in
degrees (``atan2`` of the predicted ``(cos, sin)`` vs the target), whose
chance level is ~90° for uniform ``Δφ``.
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from ropart.targets import build_targets, gather_pairs


class _Meters:
    """Running batch-weighted means."""

    def __init__(self):
        self._sum: dict[str, float] = {}
        self._n: dict[str, float] = {}

    def update(self, n: int, **values: float):
        for k, v in values.items():
            self._sum[k] = self._sum.get(k, 0.0) + float(v) * n
            self._n[k] = self._n.get(k, 0.0) + n

    def summary(self) -> dict[str, float]:
        return {k: self._sum[k] / self._n[k] for k in self._sum}


def _amp_ctx(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda")
    return nullcontext()


def pretrain_run_epoch(
    model,
    loader,
    device,
    criterion,
    *,
    patch_size: int,
    rotates: bool,
    optimizer=None,
    scaler=None,
    max_norm: float | None = None,
    max_steps: int | None = None,
) -> dict[str, float]:
    """One pretraining epoch (train if ``optimizer`` given, else eval)."""
    train = optimizer is not None
    model.train(train)
    meters = _Meters()
    grad_ctx = torch.enable_grad() if train else torch.no_grad()

    with grad_ctx:
        for step, (packed, boxes, angles) in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break
            packed = packed.to(device, non_blocking=True)
            boxes = boxes.to(device, non_blocking=True)
            angles = angles.to(device, non_blocking=True)
            bs = packed.shape[0]

            with _amp_ctx(device, scaler is not None):
                outputs, indices = model.forward_pretrain(packed)
                targets = build_targets(
                    boxes, angles if rotates else None,
                    rotates=rotates, centered=True, normalize_by=patch_size,
                )
                targets = gather_pairs(targets, indices)
                loss = criterion(outputs, targets)

            if train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if max_norm:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if max_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    optimizer.step()

            meters.update(bs, **_pretrain_metrics(outputs.detach(), targets.detach(), rotates))
            meters.update(bs, loss=loss.item())

    return meters.summary()


def _pretrain_metrics(outputs, targets, rotates) -> dict[str, float]:
    se = (outputs - targets) ** 2  # (b, C, P)
    out = {
        "mse_x": se[:, 0].mean().item(),
        "mse_y": se[:, 1].mean().item(),
        # mean-predictor floors = Var(Δx/Δy)
        "floor_x": targets[:, 0].var(unbiased=False).item(),
        "floor_y": targets[:, 1].var(unbiased=False).item(),
        "euclid": torch.sqrt(se[:, 0] + se[:, 1]).mean().item(),
    }
    if rotates and outputs.shape[1] >= 4:
        # per-pair (cos,sin) MSE summed over the 2 channels -> mean-predictor floor 1.0
        out["rot_mse"] = (se[:, 2] + se[:, 3]).mean().item()
        out["rot_floor"] = 1.0
        pred = torch.atan2(outputs[:, 3], outputs[:, 2])
        true = torch.atan2(targets[:, 3], targets[:, 2])
        d = (pred - true + math.pi) % (2 * math.pi) - math.pi
        out["ang_err_deg"] = d.abs().mean().item() * 180.0 / math.pi
    return out


def cls_run_epoch(
    model,
    loader,
    device,
    *,
    optimizer=None,
    scaler=None,
    eval_features: bool = False,
    max_steps: int | None = None,
) -> dict[str, float]:
    """One classification epoch (train if ``optimizer`` given, else eval).

    ``eval_features=True`` keeps the model in eval mode even while training the
    classifier — used for the linear-probe so the frozen backbone is deterministic.
    """
    train = optimizer is not None
    model.train(train and not eval_features)
    meters = _Meters()
    grad_ctx = torch.enable_grad() if train else torch.no_grad()

    with grad_ctx:
        for step, (images, labels) in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            bs = images.shape[0]
            with _amp_ctx(device, scaler is not None):
                logits = model.forward_classify(images)
                loss = F.cross_entropy(logits, labels)
            if train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            acc1 = (logits.argmax(dim=1) == labels).float().mean().item() * 100.0
            meters.update(bs, loss=loss.item(), acc1=acc1)

    return meters.summary()
