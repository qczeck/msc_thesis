#!/usr/bin/env bash
# Translation-only baseline pretraining (num_channels=2). The control everything
# is measured against. Extra args are forwarded, e.g. --epochs 50 --wandb-mode offline.
#
# Leaving it running unattended over SSH: launch inside tmux (or with nohup) and
# keep --wandb-mode online to watch it from the wandb dashboard, e.g.
#   tmux new -s ropart 'ropart/scripts/pretrain_baseline.sh'
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> RoPART/
# shellcheck disable=SC1091
source ropart/scripts/_doc_env.sh   # load CUDA on DoC; no-op off /vol/cuda
.venv/bin/python -m ropart.train \
  --control translation \
  --query-type patch_cat \
  --model deit_small_patch4_32 --data-path ./data \
  --epochs 100 --batch-size 128 --lr 5e-4 --warmup-epochs 5 \
  --output_dir ./out/baseline_translation \
  --wandb-name ropart-baseline-translation \
  "$@"
