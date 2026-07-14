#!/bin/bash
# ImageNet-100 BOUNDED ±30° rotation at LARGE PATCH with BLUR REMOVAL (supersample),
# ViT-B/32 @224. The interpolation-confound control for the in100_rot_bounded30_p32
# positive (ang_err 7.0° vs 20° floor): identical task, angle range, model and schedule;
# the only change is --control supersample — the source image is bicubically upsampled
# 4x before the rotated read, so the resampling blur is near-lossless and (near)
# angle-independent (the (c) remedy for the interpolation confound). If ang_err still
# clears the ~20° floor here, the bounded30_p32 signal is genuine orientation, not
# sharpness leakage; if it collapses to the floor, the positive was the confound.
# THIS IS THE ADJUDICATING RUN for the P=32 positive.
# Cost of the control (measured, 1 CPU thread, 224/P=32): extraction 17.5 ms/image vs
# 2.5 ms raw — CPU-bound, hence cpus-per-task/num_workers 16 (extraction ~140 s/epoch
# at 16 workers; GPU pace was ~100 s/epoch on the quad P=32 run, so the loader roughly
# keeps up and the 2-day wallclock has huge headroom either way).
# bf16 per the standing directive for all rotation runs (fp16 overflows ViT-B); NB the
# fp16-vs-bf16 dtype is the one deviation from the (fp16) positive being controlled.
# Distinct OUT_NAME + wandb name so it does NOT resume/clobber other rotation runs.
# Submit from gpucluster2/3:  sbatch ropart/scripts/cluster/pretrain_imagenet_bounded30_p32_ss.sbatch
#SBATCH --job-name=in100-rotb30-p32-ss
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
# SLURM spools the batch script, so $0 is not the repo path — locate via REPO.
REPO="${REPO:-${HOME}/msc_thesis/RoPART}"
source "${REPO}/ropart/scripts/cluster/_imagenet_env.sh"
export OUT_NAME=in100_rot_bounded30_p32_ss
# --model overrides the P=16 default set inside run_imagenet (argparse: last wins).
run_imagenet supersample ropart-in100-rotation-bounded30-p32-ss \
  --model deit_base_patch32_224 \
  --rotation-max-deg 30 --supersample 4 --loss-balance variance --clip-grad 1.0 \
  --amp-dtype bfloat16 --num_workers 16 "$@"
