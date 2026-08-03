#!/bin/bash
# ImageNet-100 BOUNDED ±30° at P=32, supersample, --w-phi 0.1 — the 250-epoch rerun of
# 264032 (the project's first defensible positive Δφ result).
#
# Why rerun rather than resume 264032: at 100 epochs nothing had converged. train_loss
# was still falling 5.3% over the final ten epochs with the cosine already annealed to
# min_lr, and the ±90° twin ended at 0.2259 — the value ±30° passed through around
# epoch 56. So the 2026-07-24 "graded cost with rotation range" reading is confounded
# with degree of convergence. Resuming 264032 with a longer --epochs would warm-restart
# the cosine near peak LR (lr_at is a pure function of epoch and args.epochs), which is
# a different experiment; from scratch at 250 keeps one clean schedule.
#
# Distinct OUT_NAME (_e250): _imagenet_env.sh auto-resumes from
# ${OUT_ROOT}/${OUT_NAME}/checkpoint.pth, so reusing the old name would both warm-start
# from epoch 100 and overwrite 264032, which current results depend on.
# Submit from gpucluster2/3:  sbatch ropart/scripts/cluster/pretrain_imagenet_bounded30_p32_ss_wphi01_e250.sbatch
#SBATCH --job-name=in100-rotb30-p32-ss-wphi01-e250
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=%x_%j.out
# SLURM spools the batch script, so $0 is not the repo path — locate via REPO.
REPO="${REPO:-${HOME}/msc_thesis/RoPART}"
source "${REPO}/ropart/scripts/cluster/_imagenet_env.sh"
export OUT_NAME=in100_rot_bounded30_p32_ss_wphi01_e250
# --model overrides the P=16 default set inside run_imagenet (argparse: last wins).
run_imagenet supersample ropart-in100-rotation-bounded30-p32-ss-wphi01-e250 \
  --model deit_base_patch32_224 \
  --epochs 250 --save-every 50 \
  --rotation-max-deg 30 --supersample 4 --w-phi 0.1 \
  --loss-balance variance --clip-grad 1.0 \
  --amp-dtype bfloat16 --num_workers 16 "$@"
