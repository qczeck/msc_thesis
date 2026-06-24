#!/usr/bin/env bash
# Linear-probe a pretrained checkpoint: freeze the encoder, train a CIFAR-100
# classifier head, report top-1. The no-degradation check (baseline vs rotation).
#   usage: ropart/scripts/probe.sh <checkpoint.pth> [extra args]
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> RoPART/
ckpt="${1:?usage: probe.sh <checkpoint.pth> [extra args]}"; shift || true
# shellcheck disable=SC1091
source ropart/scripts/_doc_env.sh   # load CUDA on DoC; no-op off /vol/cuda
.venv/bin/python -m ropart.train --linear-probe --resume "$ckpt" \
  --model deit_small_patch4_32 --data-path ./data \
  --epochs 50 --batch-size 256 --lr 1e-3 --warmup-epochs 2 \
  --output_dir ./out/probe \
  --wandb-name ropart-probe \
  "$@"
