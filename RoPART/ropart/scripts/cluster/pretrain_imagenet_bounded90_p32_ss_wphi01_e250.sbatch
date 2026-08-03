#!/bin/bash
# ImageNet-100 BOUNDED ±90° at P=32, supersample, --w-phi 0.1 — the 250-epoch rerun of
# 264915, and the arm that settles the rotation-range question at a MATCHED budget.
#
# The 2026-07-24 reading was that the structural gates loosen monotonically with range
# (identity 0.028 -> 0.202, neg-symmetry 1.2% -> 8.5%, composition 0.63° -> 3.90°,
# translation 12.5% -> 24.7% of floor from ±30° to ±90°) and that this is rotation's
# intrinsic cost. But at 100 epochs ±90° ended at train_loss 0.2259, which is where
# ±30° stood around epoch 56 under the identical schedule — so "wider range costs
# structure" and "wider range is a harder regression that got half the effective
# budget" are indistinguishable in that data. Run to 250 against the ±30° twin, the two
# hypotheses separate: gates that close are convergence, gates that persist are cost.
#
# ±60° is deliberately not rerun — ±30° vs ±90° is the widest contrast and carries the
# question; the middle point is corroborating detail and can be added later if wanted.
# Distinct OUT_NAME (_e250) so the auto-resume in _imagenet_env.sh cannot warm-start
# from, or overwrite, 264915.
# Submit from gpucluster2/3:  sbatch ropart/scripts/cluster/pretrain_imagenet_bounded90_p32_ss_wphi01_e250.sbatch
#SBATCH --job-name=in100-rotb90-p32-ss-wphi01-e250
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=%x_%j.out
# SLURM spools the batch script, so $0 is not the repo path — locate via REPO.
REPO="${REPO:-${HOME}/msc_thesis/RoPART}"
source "${REPO}/ropart/scripts/cluster/_imagenet_env.sh"
export OUT_NAME=in100_rot_bounded90_p32_ss_wphi01_e250
# --model overrides the P=16 default set inside run_imagenet (argparse: last wins).
run_imagenet supersample ropart-in100-rotation-bounded90-p32-ss-wphi01-e250 \
  --model deit_base_patch32_224 \
  --epochs 250 --save-every 50 \
  --rotation-max-deg 90 --supersample 4 --w-phi 0.1 \
  --loss-balance variance --clip-grad 1.0 \
  --amp-dtype bfloat16 --num_workers 16 "$@"
