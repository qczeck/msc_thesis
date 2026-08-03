#!/bin/bash
# ImageNet-100 BOUNDED ±30° at P=32 with SUPERSAMPLED ROTATED PIXELS but rotation
# UNSUPERVISED (--control ss_ch2, num_channels=2), 250 epochs — the 250-epoch rerun of
# 264031, now serving a new purpose.
#
# Its original job (mechanism probe: is 260603's translation collapse the drag, or do
# ss pixels break translation?) is done and settled. Here it is the AUGMENTATION
# CONTROL for the downstream comparison. The three arms differ by exactly one thing
# each:
#   translation baseline  axis-aligned pixels, (dx, dy)
#   ss_ch2 (this)         ROTATED pixels,      (dx, dy)
#   ss_wphi01             rotated pixels,      (dx, dy, cos, sin)
# baseline -> ss_ch2 isolates the rotated pixels; ss_ch2 -> ss_wphi01 isolates the
# orientation supervision. Without this arm, a downstream win for RoPART cannot be
# separated from "feeding a ViT randomly rotated patches is decent augmentation",
# which is not this project's claim.
#
# Must share the 250-epoch schedule with the other two: comparing encoders trained on
# different budgets would reintroduce the exact confound the rerun exists to remove.
# Distinct OUT_NAME (_e250) so the auto-resume in _imagenet_env.sh cannot warm-start
# from, or overwrite, 264031.
# Submit from gpucluster2/3:  sbatch ropart/scripts/cluster/pretrain_imagenet_bounded30_p32_ss_ch2_e250.sbatch
#SBATCH --job-name=in100-rotb30-p32-ss-ch2-e250
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=%x_%j.out
# SLURM spools the batch script, so $0 is not the repo path — locate via REPO.
REPO="${REPO:-${HOME}/msc_thesis/RoPART}"
source "${REPO}/ropart/scripts/cluster/_imagenet_env.sh"
export OUT_NAME=in100_rot_bounded30_p32_ss_ch2_e250
# --model overrides the P=16 default set inside run_imagenet (argparse: last wins).
run_imagenet ss_ch2 ropart-in100-rotation-bounded30-p32-ss-ch2-e250 \
  --model deit_base_patch32_224 \
  --epochs 250 --save-every 50 \
  --rotation-max-deg 30 --supersample 4 \
  --loss-balance variance --clip-grad 1.0 \
  --amp-dtype bfloat16 --num_workers 16 "$@"
