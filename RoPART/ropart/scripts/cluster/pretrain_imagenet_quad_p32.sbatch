#!/bin/bash
# ImageNet-100 QUAD-rotation pretext at LARGE PATCH (num_channels=4), ViT-B/32 @224.
# Identical to pretrain_imagenet_quad.sbatch except the patch size: P=32 -> 7x7 = 49
# patches (vs 196 at P=16). Bigger patches carry more orientable content, so this is
# the orientation-*recoverability* probe (Validity threat #2): if the flat quad P=16
# result is because a 16px patch is too small/ambiguous to orient, P=32 should show
# ang_err dropping below the 90° floor. Confound-free (exact rot90). Directly
# comparable to the in100_quad (P=16) run — only the patch size changes.
# Fewer patches => LESS memory/faster than P=16, so this is cheaper, not harder.
# Distinct OUT_NAME + wandb name so it does NOT resume/clobber other rotation runs.
# Submit from gpucluster2/3:  sbatch ropart/scripts/cluster/pretrain_imagenet_quad_p32.sbatch
#SBATCH --job-name=in100-quad-p32
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
# SLURM spools the batch script, so $0 is not the repo path — locate via REPO.
REPO="${REPO:-${HOME}/msc_thesis/RoPART}"
source "${REPO}/ropart/scripts/cluster/_imagenet_env.sh"
export OUT_NAME=in100_quad_p32
# --model overrides the P=16 default set inside run_imagenet (argparse: last wins).
run_imagenet quad ropart-in100-quad-p32 \
  --model deit_base_patch32_224 \
  --rotation-set quad --loss-balance variance --clip-grad 1.0 "$@"
