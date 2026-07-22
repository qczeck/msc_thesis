#!/bin/bash
# ImageNet-100 BOUNDED ±30° at P=32 with SUPERSAMPLED PIXELS but rotation UNSUPERVISED
# (--control ss_ch2, num_channels=2) — the mechanism probe for the ss adjudication
# adjudication. The ss run (260603) collapsed on
# EVERY channel; blur removal has no business breaking translation, so two hypotheses
# compete: (i) drag — the unlearnable supervised rotation group injects O(1) noise
# gradient through the shared encoder; (ii) the ss extraction itself damages the
# translation task. This arm keeps the exact ss pixels but drops the rotation target:
# translation recovering to ~0.1x floor confirms (i) and exonerates the ss pixels;
# translation collapsing again confirms (ii) and invalidates the 260603 adjudication.
# Continuous analogue of the quad_ch2 control (2026-07-02).
# Same extraction cost as ss (17.5 ms/img) -> 16 CPUs/workers. bf16 per the standing
# rotation-run directive (moot here for overflow — ch2 — but kept config-identical).
# Distinct OUT_NAME + wandb name so it does NOT resume/clobber other rotation runs.
# Submit from gpucluster2/3:  sbatch ropart/scripts/cluster/pretrain_imagenet_bounded30_p32_ss_ch2.sbatch
#SBATCH --job-name=in100-rotb30-p32-ss-ch2
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
# SLURM spools the batch script, so $0 is not the repo path — locate via REPO.
REPO="${REPO:-${HOME}/msc_thesis/RoPART}"
source "${REPO}/ropart/scripts/cluster/_imagenet_env.sh"
export OUT_NAME=in100_rot_bounded30_p32_ss_ch2
# --model overrides the P=16 default set inside run_imagenet (argparse: last wins).
run_imagenet ss_ch2 ropart-in100-rotation-bounded30-p32-ss-ch2 \
  --model deit_base_patch32_224 \
  --rotation-max-deg 30 --supersample 4 --loss-balance variance --clip-grad 1.0 \
  --amp-dtype bfloat16 --num_workers 16 "$@"
