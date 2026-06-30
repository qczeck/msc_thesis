#!/bin/bash
# ImageNet-100 QUAD-rotation pretext (num_channels=4), ViT-B/16 @224. Per-patch
# rotation restricted to {0,90,180,270}° via exact torch.rot90 (the 'quad' control)
# — an exact pixel permutation, so there is NO interpolation and the interpolation
# confound vanishes entirely; no blur/supersample controls are needed. RotNet-style
# discrete rotation, but kept relative between patch pairs.
#   --control quad --rotation-set quad : exact rot90 extractor + discrete sampler.
#   --loss-balance variance            : keep Δφ from being swamped by the
#                                        pixel-scale translation group.
# Distinct OUT_NAME + wandb name so it does NOT resume/clobber other rotation runs.
# Submit from gpucluster2/3:  sbatch ropart/scripts/cluster/pretrain_imagenet_quad.sbatch
#SBATCH --job-name=in100-quad
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
# SLURM spools the batch script, so $0 is not the repo path — locate via REPO.
REPO="${REPO:-${HOME}/msc_thesis/RoPART}"
source "${REPO}/ropart/scripts/cluster/_imagenet_env.sh"
export OUT_NAME=in100_quad
run_imagenet quad ropart-in100-quad \
  --rotation-set quad --loss-balance variance --clip-grad 1.0 "$@"
