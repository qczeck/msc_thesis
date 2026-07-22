#!/bin/bash
# ImageNet-100 BOUNDED ±30° at P=32, RAW control, rotation DOWN-WEIGHTED (--w-phi 0.1)
# — the reference arm for the ss --w-phi 0.1 twin
# (pretrain_imagenet_bounded30_p32_ss_wphi01). The interpolation signature is present
# here (raw bilinear read), so if w_phi 0.1 supplies enough gradient to learn at all,
# this run should reproduce the ±30° rotation learning (7.17° vs 19.96° floor at full
# weight) — removing the "you starved the rotation channel" objection when the ss twin
# floors. It also checks that down-weighting keeps translation at ~0.1x floor on the
# raw pipeline. The pair differs ONLY in the extractor (raw vs supersample): rotation
# learning iff the signature is present, with translation healthy in both, is the
# both-channels-intact adjudication the full-weight ss run could not deliver.
# Raw extraction is cheap (2.5 ms/img) -> standard 8 CPUs/workers.
# Submit from gpucluster2/3:  sbatch ropart/scripts/cluster/pretrain_imagenet_bounded30_p32_raw_wphi01.sbatch
#SBATCH --job-name=in100-rotb30-p32-raw-wphi01
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
# SLURM spools the batch script, so $0 is not the repo path — locate via REPO.
REPO="${REPO:-${HOME}/msc_thesis/RoPART}"
source "${REPO}/ropart/scripts/cluster/_imagenet_env.sh"
export OUT_NAME=in100_rot_bounded30_p32_wphi01
# --model overrides the P=16 default set inside run_imagenet (argparse: last wins).
run_imagenet raw ropart-in100-rotation-bounded30-p32-raw-wphi01 \
  --model deit_base_patch32_224 \
  --rotation-max-deg 30 --w-phi 0.1 \
  --loss-balance variance --clip-grad 1.0 \
  --amp-dtype bfloat16 "$@"
