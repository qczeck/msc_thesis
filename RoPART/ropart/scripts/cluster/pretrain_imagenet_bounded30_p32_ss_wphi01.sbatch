#!/bin/bash
# ImageNet-100 BOUNDED ±30° at P=32, SUPERSAMPLE control, rotation DOWN-WEIGHTED
# (--w-phi 0.1) — the drag-controlled rerun of the collapsed ss adjudicator (260603).
# Under variance balancing w_phi is a genuine importance knob (losses.py); 0.1 cuts
# the rotation group's gradient 10x so an unlearnable rotation target cannot drag
# translation down through the shared encoder. Two informative outcomes:
#   - translation recovers (~0.1x floor) + rotation at the 19.96° floor -> the clean
#     negative: blur removal kills rotation while translation stays learnable, sealing
#     the signature verdict in a setting where "both channels broke" cannot confuse it;
#   - rotation ALSO learns -> the 260603 collapse was optimisation pathology (its loss
#     spikes / grad_norm 387), and genuine orientation learning survives blur removal.
# Interpret against the raw --w-phi 0.1 twin (pretrain_imagenet_bounded30_p32_raw_wphi01),
# which proves 0.1 suffices to learn rotation when the cue is present.
# Submit from gpucluster2/3:  sbatch ropart/scripts/cluster/pretrain_imagenet_bounded30_p32_ss_wphi01.sbatch
#SBATCH --job-name=in100-rotb30-p32-ss-wphi01
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
# SLURM spools the batch script, so $0 is not the repo path — locate via REPO.
REPO="${REPO:-${HOME}/msc_thesis/RoPART}"
source "${REPO}/ropart/scripts/cluster/_imagenet_env.sh"
export OUT_NAME=in100_rot_bounded30_p32_ss_wphi01
# --model overrides the P=16 default set inside run_imagenet (argparse: last wins).
run_imagenet supersample ropart-in100-rotation-bounded30-p32-ss-wphi01 \
  --model deit_base_patch32_224 \
  --rotation-max-deg 30 --supersample 4 --w-phi 0.1 \
  --loss-balance variance --clip-grad 1.0 \
  --amp-dtype bfloat16 --num_workers 16 "$@"
