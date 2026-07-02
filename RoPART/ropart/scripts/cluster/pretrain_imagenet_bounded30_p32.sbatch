#!/bin/bash
# ImageNet-100 BOUNDED ±30° rotation pretext at LARGE PATCH (num_channels=4),
# ViT-B/32 @224. Identical to pretrain_imagenet_rotation_bounded.sbatch except the
# patch size: P=32 -> 7x7 = 49 patches (vs 196 at P=16). The continuous twin of the
# quad_p32 recoverability probe: does more orientable content per patch let the
# bounded (±30°) continuous Δφ learn where P=16 sat at its floor? Directly comparable
# to in100_rot_bounded30 (P=16) — only the patch size changes.
# NB continuous rotation is interpolation-confounded;
# score any positive Δφ against that, and against the *bounded* floor (~20° / rot_mse
# ~0.17), not 90° / 1.0. Variance balance uses the eps=1e-3 divisor clamp (default).
# Distinct OUT_NAME + wandb name so it does NOT resume/clobber other rotation runs.
# Submit from gpucluster2/3:  sbatch ropart/scripts/cluster/pretrain_imagenet_bounded30_p32.sbatch
#SBATCH --job-name=in100-rotb30-p32
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
# SLURM spools the batch script, so $0 is not the repo path — locate via REPO.
REPO="${REPO:-${HOME}/msc_thesis/RoPART}"
source "${REPO}/ropart/scripts/cluster/_imagenet_env.sh"
export OUT_NAME=in100_rot_bounded30_p32
# --model overrides the P=16 default set inside run_imagenet (argparse: last wins).
run_imagenet raw ropart-in100-rotation-bounded30-p32 \
  --model deit_base_patch32_224 \
  --rotation-max-deg 30 --loss-balance variance --clip-grad 1.0 "$@"
