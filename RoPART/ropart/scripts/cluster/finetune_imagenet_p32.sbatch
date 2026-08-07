#!/bin/bash
# Phase-3 downstream finetune of a P=32 pretrained encoder, ViT-B/32 @224, IN-100.
#
# The source paper's protocol (§4 "Training setup"): the relative encoder is
# discarded and replaced by a linear head on [CLS], randomly initialised learnable
# position embeddings are introduced, and the whole network is tuned end-to-end.
# PART never linear-probes — every downstream number in the paper is a finetune —
# so this, not ropart.train's --linear-probe path, is the comparable measurement.
#
# Parametrised rather than one file per arm, because the same script serves all four
# encoders and (via --data-path / --finetune-task) the HLW horizon task too. Submit
# through submit_finetune_p32.sh, or by hand:
#
#   sbatch -J in100-ft-base \
#     --export=ALL,OUT_NAME=in100_ft_translation_p32,\
#INIT_FROM=/vol/gpudata/msk123-ropart-in100/out/in100_translation_p32_e250/checkpoint.pth,\
#WANDB_NAME=ropart-in100-ft-baseline-p32 \
#     ropart/scripts/cluster/finetune_imagenet_p32.sbatch
#
# OUT_NAME must be FRESH. _imagenet_env.sh auto-resumes from
# $OUT_ROOT/$OUT_NAME/checkpoint.pth, so reusing a pretrain dir would both warm-start
# from the wrong state and overwrite the encoder this run was seeded from; the helper
# refuses the exact-collision case but cannot catch every variant.
#SBATCH --job-name=in100-ft
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
# SLURM spools the batch script, so $0 is not the repo path — locate via REPO.
REPO="${REPO:-${HOME}/msc_thesis/RoPART}"
source "${REPO}/ropart/scripts/cluster/_imagenet_env.sh"
: "${OUT_NAME:?set OUT_NAME via --export}"
: "${INIT_FROM:?set INIT_FROM via --export}"
finetune_imagenet "${WANDB_NAME:-ropart-${OUT_NAME}}" "$@"
