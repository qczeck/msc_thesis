#!/bin/bash
# ImageNet-100 translation-only baseline (num_channels=2), ViT-B/16 @224.
# Submit from gpucluster2/3:  sbatch ropart/scripts/cluster/pretrain_imagenet_baseline.sbatch
# Extra args forward to ropart.train, e.g.  sbatch ... --epochs 50
#SBATCH --job-name=in100-base
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
# SLURM spools the batch script, so $0 is not the repo path — locate via REPO.
REPO="${REPO:-${HOME}/msc_thesis/RoPART}"
source "${REPO}/ropart/scripts/cluster/_imagenet_env.sh"
run_imagenet translation ropart-in100-baseline "$@"
