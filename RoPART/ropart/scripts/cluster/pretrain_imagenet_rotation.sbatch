#!/bin/bash
# ImageNet-100 raw-rotation pretext (num_channels=4), ViT-B/16 @224. The 'raw'
# control is the interpolation-confounded comparison point; artefact remedies are
# separate --control runs.
# Submit from gpucluster2/3:  sbatch ropart/scripts/cluster/pretrain_imagenet_rotation.sbatch
#SBATCH --job-name=in100-rot
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
# SLURM spools the batch script, so $0 is not the repo path — locate via REPO.
REPO="${REPO:-${HOME}/msc_thesis/RoPART}"
source "${REPO}/ropart/scripts/cluster/_imagenet_env.sh"
run_imagenet raw ropart-in100-rotation-raw "$@"
