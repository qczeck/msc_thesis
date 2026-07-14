#!/bin/bash
# ImageNet-100 BOUNDED ±60° rotation pretext at LARGE PATCH (num_channels=4),
# ViT-B/32 @224. Identical to pretrain_imagenet_bounded30_p32.sbatch except the angle
# half-range: ±30° -> ±60°. Control stays `raw` (NO blur removal) so this compares
# DIRECTLY to the in100_rot_bounded30_p32 positive — only the rotation range moves.
# Question: does the P=32 benefit survive a 2x harder target, or was ±30° a sweet spot?
# NB floors change with the range: Δφ = φj − φi is triangular on [−120°, 120°], so the
# mean-predictor floors are ang_err (E|Δφ|) = 40° and rot_mse = 1 − (E cos Δφ)² ≈ 0.53
# (same formula gives ±30°'s documented 0.169) — score against
# those, NOT ±30°'s 20° / 0.169. Still interpolation-confounded (continuous rotation);
# the ..._ss run adjudicates that axis.
# bf16 per the standing directive for all rotation runs (fp16 overflows ViT-B); NB the
# fp16-vs-bf16 dtype is the one deviation from the (fp16) positive being compared to.
# Distinct OUT_NAME + wandb name so it does NOT resume/clobber other rotation runs.
# Submit from gpucluster2/3:  sbatch ropart/scripts/cluster/pretrain_imagenet_bounded60_p32.sbatch
#SBATCH --job-name=in100-rotb60-p32
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
# SLURM spools the batch script, so $0 is not the repo path — locate via REPO.
REPO="${REPO:-${HOME}/msc_thesis/RoPART}"
source "${REPO}/ropart/scripts/cluster/_imagenet_env.sh"
export OUT_NAME=in100_rot_bounded60_p32
# --model overrides the P=16 default set inside run_imagenet (argparse: last wins).
run_imagenet raw ropart-in100-rotation-bounded60-p32 \
  --model deit_base_patch32_224 \
  --rotation-max-deg 60 --loss-balance variance --clip-grad 1.0 \
  --amp-dtype bfloat16 "$@"
