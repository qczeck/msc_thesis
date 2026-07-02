#!/bin/bash
# ImageNet-100 SMALL ±5° rotation pretext (num_channels=4), ViT-B/16 @224.
# Same 'raw' bilinear rotation control as the bounded run, but a much narrower range:
#   --rotation-max-deg 5 : φ ∈ [-5°, +5°]. A rotation this small barely perturbs the
#                          cross-patch content continuity the jigsaw relies on, so the
#                          hypothesis is that TRANSLATION survives (euclid -> ~1.6, not
#                          the ~6.5 collapse of the larger-angle runs). Whether Δφ
#                          itself learns is secondary AND interpolation-confounded at
#                          this scale (tiny blur ~ tiny angle), so treat any positive
#                          Δφ with suspicion.
#   --loss-balance variance + --loss-eps 1e-3 : at ±5° the cos Δφ variance ~1e-5, so
#                          the balance divisor MUST be clamped (eps=1e-3) or it
#                          amplifies that channel ~1e5x and NaNs training (as ±30° did
#                          at epoch 21). eps=1e-3 caps amplification at ~1000x.
# Score against the ±5° floors (ang_err ~3.3°, rot_mse ~0.005), NOT 90° / 1.0.
# WATCH the first ~20 epochs for NaN; fallback is --loss-balance none (resubmit).
# Distinct OUT_NAME + wandb name so it does NOT resume/clobber other rotation runs.
# Submit from gpucluster2/3:  sbatch ropart/scripts/cluster/pretrain_imagenet_rotation_small.sbatch
#SBATCH --job-name=in100-rotb5
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
# SLURM spools the batch script, so $0 is not the repo path — locate via REPO.
REPO="${REPO:-${HOME}/msc_thesis/RoPART}"
source "${REPO}/ropart/scripts/cluster/_imagenet_env.sh"
export OUT_NAME=in100_rot_bounded5
run_imagenet raw ropart-in100-rotation-bounded5 \
  --rotation-max-deg 5 --loss-balance variance --loss-eps 1e-3 --clip-grad 1.0 "$@"
