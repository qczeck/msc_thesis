#!/bin/bash
# ImageNet-100 TRANSLATION-ONLY baseline (num_channels=2) at P=32, ViT-B/32 @224,
# 250 epochs — the missing control for the whole P=32 rotation programme. Every P=32
# rotation result so far is quoted against the mean-predictor floor, never against a
# matched translation-only run at the same patch size (the shipped
# pretrain_imagenet_baseline.sbatch is ViT-B/16).
#
# --loss-balance variance is load-bearing, NOT cosmetic. On a ch=2 run it switches the
# objective from raw MSE (order tens) to mse/var, a fraction of floor (order one) —
# see losses.py::_group. Since w_xy=1 in every RoPART arm, this makes the baseline's
# loss term-for-term identical to the translation group inside RoPART's, which is what
# makes it a control; without it the baseline trains at a ~20x different effective
# gradient scale and adjudicates nothing.
#
# a40 rather than a100: partition choice is scientifically neutral (the schedule is
# epoch-indexed, not time-indexed) and a100 is heavily contended. Expect ~55 h here,
# inside the 3-day partition limit.
# Submit from gpucluster2/3:  sbatch ropart/scripts/cluster/pretrain_imagenet_baseline_p32_e250.sbatch
#SBATCH --job-name=in100-base-p32-e250
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=%x_%j.out
# SLURM spools the batch script, so $0 is not the repo path — locate via REPO.
REPO="${REPO:-${HOME}/msc_thesis/RoPART}"
source "${REPO}/ropart/scripts/cluster/_imagenet_env.sh"
export OUT_NAME=in100_translation_p32_e250
# --model overrides the P=16 default set inside run_imagenet (argparse: last wins).
run_imagenet translation ropart-in100-baseline-p32-e250 \
  --model deit_base_patch32_224 \
  --epochs 250 --save-every 50 \
  --loss-balance variance --clip-grad 1.0 \
  --amp-dtype bfloat16 --num_workers 16 "$@"
