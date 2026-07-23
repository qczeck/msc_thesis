#!/bin/bash
# ImageNet-100 BOUNDED ±60° at P=32, SUPERSAMPLE control, rotation DOWN-WEIGHTED
# (--w-phi 0.1) — pushing the P=32 ss-wphi01 result (the ±30° top run, 264032:
# dense angular MAE 10.28° vs 19.96° floor) to a wider rotation range. Same config
# as pretrain_imagenet_bounded30_p32_ss_wphi01 in every respect EXCEPT the angle
# range, so the pixels are produced by the identical (bit-verified) supersample
# extractor and the two runs stay directly comparable. The dense angular MAE floor
# scales with range: ~40° here (uniform ±60°); a genuine result breaks below it.
# NB: no raw twin at this range — the signature-ruled-out (deconfound) claim rests
# on the ±30° raw/ss pair; this is a "how far can we push it" probe of ss learning.
# Submit from gpucluster2/3:  sbatch ropart/scripts/cluster/pretrain_imagenet_bounded60_p32_ss_wphi01.sbatch
#SBATCH --job-name=in100-rotb60-p32-ss-wphi01
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
# SLURM spools the batch script, so $0 is not the repo path — locate via REPO.
REPO="${REPO:-${HOME}/msc_thesis/RoPART}"
source "${REPO}/ropart/scripts/cluster/_imagenet_env.sh"
export OUT_NAME=in100_rot_bounded60_p32_ss_wphi01
# --model overrides the P=16 default set inside run_imagenet (argparse: last wins).
run_imagenet supersample ropart-in100-rotation-bounded60-p32-ss-wphi01 \
  --model deit_base_patch32_224 \
  --rotation-max-deg 60 --supersample 4 --w-phi 0.1 \
  --loss-balance variance --clip-grad 1.0 \
  --amp-dtype bfloat16 --num_workers 16 "$@"
