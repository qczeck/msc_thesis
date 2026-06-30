#!/bin/bash
# ImageNet-100 BOUNDED-rotation pretext (num_channels=4), ViT-B/16 @224.
# Same 'raw' pixel-rotation control as pretrain_imagenet_rotation.sbatch, but two
# changes that test the two leading hypotheses for the flat Δφ (see
# josiah-meeting-06-30.md / docs):
#   --rotation-max-deg 30  : φ ∈ [-30°, +30°] instead of the full circle, so the
#                            per-patch rotation no longer destroys the cross-patch
#                            content continuity the jigsaw relies on (translation
#                            should survive) and Δφ becomes a smaller, plausibly
#                            learnable signal.
#   --loss-balance variance: normalise each channel by its batch target variance so
#                            the (cos,sin) group is no longer swamped ~26:1 by the
#                            pixel-scale translation group — proper Δφ weighting.
# Distinct OUT_NAME + wandb name so it does NOT resume/clobber the in100_raw run.
# Submit from gpucluster2/3:  sbatch ropart/scripts/cluster/pretrain_imagenet_rotation_bounded.sbatch
#SBATCH --job-name=in100-rotb30
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
# SLURM spools the batch script, so $0 is not the repo path — locate via REPO.
REPO="${REPO:-${HOME}/msc_thesis/RoPART}"
source "${REPO}/ropart/scripts/cluster/_imagenet_env.sh"
export OUT_NAME=in100_rot_bounded30
run_imagenet raw ropart-in100-rotation-bounded30 \
  --rotation-max-deg 30 --loss-balance variance --clip-grad 1.0 "$@"
