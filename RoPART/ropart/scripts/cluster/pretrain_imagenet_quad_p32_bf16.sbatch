#!/bin/bash
# ImageNet-100 QUAD-rotation pretext at LARGE PATCH, bf16 rerun (num_channels=4),
# ViT-B/32 @224. Identical to pretrain_imagenet_quad_p32.sbatch except:
#   (1) --amp-dtype bfloat16 — the fp16 default overflowed ViT-B attention logits to
#       inf/NaN at epoch 19 (grad_norm -> NaN, LR near peak); bf16 has fp32 exponent
#       range so it does not overflow. Root-cause fix, keeps LR/loss-balance identical
#       so the result stays comparable to the bounded30_p32 run.
#   (2) OUT_NAME=in100_quad_p32_bf16 — a fresh checkpoint dir so it does NOT resume the
#       NaN-poisoned in100_quad_p32/checkpoint.pth (auto-resume would reload NaN weights).
# Confound-free (exact rot90): the decisive test of whether relative orientation is
# learnable at P=32 without the interpolation confound. Compare to in100_quad (P=16,
# flat at floor) and in100_rot_bounded30_p32 (continuous ±30°, learned but confounded).
# Submit from gpucluster2/3:  sbatch --partition=a100 ropart/scripts/cluster/pretrain_imagenet_quad_p32_bf16.sbatch
#SBATCH --job-name=in100-quad-p32-bf16
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
# SLURM spools the batch script, so $0 is not the repo path — locate via REPO.
REPO="${REPO:-${HOME}/msc_thesis/RoPART}"
source "${REPO}/ropart/scripts/cluster/_imagenet_env.sh"
export OUT_NAME=in100_quad_p32_bf16
# --model overrides the P=16 default set inside run_imagenet (argparse: last wins).
run_imagenet quad ropart-in100-quad-p32-bf16 \
  --model deit_base_patch32_224 \
  --rotation-set quad --loss-balance variance --clip-grad 1.0 \
  --amp-dtype bfloat16 "$@"
