#!/usr/bin/env bash
# Submit the four Phase-3 IN-100 classification finetunes — one per pretrained arm of
# the 250-epoch matched set (269805-269808).
#
#   usage: ropart/scripts/cluster/submit_finetune_p32.sh [--dry-run] [extra train args]
#
# The four arms differ ONLY by --init-from. Everything else (epochs, lr, schedule,
# batch, transforms) is fixed inside finetune_imagenet(), which is the point: a
# downstream difference is then attributable to the pretext objective rather than to
# the finetuning recipe.
#
# The triple baseline -> ss_ch2 -> ss_wphi01 differs by exactly one factor per step
# (axis-aligned vs rotated pixels; then unsupervised vs supervised rotation), so a win
# for ss_wphi01 over ss_ch2 is attributable to the Δφ *target* and not to rotated
# patches being decent augmentation.
set -euo pipefail

DRY=0
[ "${1:-}" = "--dry-run" ] && { DRY=1; shift; }

REPO="${REPO:-${HOME}/msc_thesis/RoPART}"
WS="${WS:-/vol/gpudata/${USER}-ropart-in100}"
SBATCH_SCRIPT="${REPO}/ropart/scripts/cluster/finetune_imagenet_p32.sbatch"

# job-suffix | pretrain OUT_NAME (the encoder) | wandb name
ARMS=(
  "base|in100_translation_p32_e250|ropart-in100-ft-baseline-p32"
  "ch2|in100_rot_bounded30_p32_ss_ch2_e250|ropart-in100-ft-ss-ch2-30-p32"
  "w30|in100_rot_bounded30_p32_ss_wphi01_e250|ropart-in100-ft-ss-wphi01-30-p32"
  "w90|in100_rot_bounded90_p32_ss_wphi01_e250|ropart-in100-ft-ss-wphi01-90-p32"
)

for arm in "${ARMS[@]}"; do
  IFS='|' read -r tag pre wname <<< "${arm}"
  init="${WS}/out/${pre}/checkpoint.pth"
  out_name="in100_ft_${tag}_p32"
  if [ ! -f "${init}" ]; then
    echo "SKIP ${tag}: no checkpoint at ${init}" >&2
    continue
  fi
  cmd=(sbatch -J "in100-ft-${tag}"
       --export="ALL,OUT_NAME=${out_name},INIT_FROM=${init},WANDB_NAME=${wname}"
       "${SBATCH_SCRIPT}" "$@")
  if [ "${DRY}" = "1" ]; then
    printf '%q ' "${cmd[@]}"; echo
  else
    "${cmd[@]}"
  fi
done
