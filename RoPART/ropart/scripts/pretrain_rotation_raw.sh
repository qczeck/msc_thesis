#!/usr/bin/env bash
# Translation + raw rotation pretraining (num_channels=4). Uses the 'raw' control
# (plain bilinear rotation) — deliberately confounded by interpolation blur; this
# is the comparison point for the artefact remedies (supersample/dominant/...).
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> RoPART/
# shellcheck disable=SC1091
source ropart/scripts/_doc_env.sh   # load CUDA on DoC; no-op off /vol/cuda
.venv/bin/python -m ropart.train \
  --control raw \
  --query-type patch_cat \
  --model deit_small_patch4_32 --data-path ./data \
  --epochs 100 --batch-size 128 --lr 5e-4 --warmup-epochs 5 \
  --output_dir ./out/rotation_raw \
  --wandb-name ropart-rotation-raw \
  "$@"
