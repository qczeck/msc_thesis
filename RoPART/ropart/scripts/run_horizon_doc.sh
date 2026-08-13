#!/usr/bin/env bash
# Phase-3 HLW horizon-line finetune on a **standalone DoC lab GPU box** (gpu01-32).
# The orientation-sensitive half of Phase 3; the classification half is run_finetune_doc.sh.
#
#   usage: run_horizon_doc.sh <arm> [--seed N] [--dry-run] [extra ropart.train args]
#          arm in {base, ch2, w30, w90}
#
# >>> THE RECIPE HERE MIRRORS run_finetune_doc.sh AND MUST BE KEPT IN SYNC WITH IT. <<<
# Same encoders, same 100 epochs / warmup 5 / lr 1e-4 / cosine to 1e-6 / batch 192 / fp16 /
# workers 6 / --eval-every 1. Only the task differs (--finetune-task horizon), which swaps
# the 100-class head for a 2-output (theta, rho) head and the metric for horizon AUC.
# Kept as a separate file rather than a --task flag on run_finetune_doc.sh so that editing
# the horizon path cannot perturb the completed 2026-08-07 classification runs; the cost is
# that a recipe change must be made twice, hence this banner.
#
# WHAT IS BEING MEASURED. Predict (theta, rho); report AUC via (l, r). theta *is* in-plane
# orientation and rho is translation-like, so the task carries a built-in specificity
# control: RoPART should improve theta and leave rho near the translation-only baseline.
# horizon_run_epoch reports theta_mae (degrees) and rho_mae separately for exactly this.
#
# ⚠ TWO THINGS TO SETTLE BEFORE TRUSTING ANY NUMBER FROM THIS SCRIPT:
#   1. Y_AXIS_DOWN in ropart/hlw.py. metadata.csv endpoints are documented as zero-centred
#      but the y-sign is not stated anywhere reachable. A wrong guess trains and converges
#      perfectly well while producing a meaningless AUC. Settle it first:
#        python -m ropart.hlw --verify $WS/hlw --n 12 --out ~/hlw_verify
#      then look at the PNGs: the red line must lie on the visible horizon.
#   2. The constant-predictor floor. Without it an AUC of 0.4 is uninterpretable. theta is
#      ~radians (small) and rho ~image heights, so the shared smooth_l1 may also under-weight
#      theta relative to rho — which would blunt the specificity control. Check the epoch-1
#      theta_mae/rho_mae against the floor before committing to a full 4-arm set.
#
# Data layout expected at ${HLW_ROOT}: images/, split/{train,val,test}.txt, metadata.csv.
# Train is the ~98k-image train split; "val" during finetune is HLW's held-out 2,018-image
# **test** split (the 525-image val split is too small to track per epoch).
set -euo pipefail

ARM="${1:?usage: run_horizon_doc.sh <base|ch2|w30|w90> [--seed N] [--dry-run] [extra args]}"
shift
DRY=0
SEED=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --seed)    SEED="${2:?--seed needs a value}"; shift 2 ;;
    *)         break ;;      # everything else is passed through to ropart.train
  esac
done

REPO="${REPO:-${HOME}/msc_thesis/RoPART}"
WS="${WS:-/vol/gpudata/${USER}-ropart-in100}"
HLW_ROOT="${HLW_ROOT:-${WS}/hlw}"
OUT_ROOT="${WS}/out"

# arm -> pretrained encoder OUT_NAME | wandb run name. Identical mapping to
# run_finetune_doc.sh: the same four encoders are scored on both Phase-3 tasks.
case "${ARM}" in
  base) PRE=in100_translation_p32_e250;              WNAME=ropart-hlw-ft-baseline-p32 ;;
  ch2)  PRE=in100_rot_bounded30_p32_ss_ch2_e250;     WNAME=ropart-hlw-ft-ss-ch2-30-p32 ;;
  w30)  PRE=in100_rot_bounded30_p32_ss_wphi01_e250;  WNAME=ropart-hlw-ft-ss-wphi01-30-p32 ;;
  w90)  PRE=in100_rot_bounded90_p32_ss_wphi01_e250;  WNAME=ropart-hlw-ft-ss-wphi01-90-p32 ;;
  *) echo "ERROR: unknown arm '${ARM}' (want base|ch2|w30|w90)" >&2; exit 2 ;;
esac

INIT_FROM="${OUT_ROOT}/${PRE}/checkpoint.pth"
if [ "${SEED}" = "0" ]; then
  OUT_NAME="hlw_ft_${ARM}_p32"
else
  OUT_NAME="hlw_ft_${ARM}_p32_s${SEED}"
  WNAME="${WNAME}-s${SEED}"
fi
OUT="${OUT_ROOT}/${OUT_NAME}"
LOG="${OUT}/run.log"

[ -f "${INIT_FROM}" ] || { echo "ERROR: no pretrain checkpoint at ${INIT_FROM}" >&2; exit 1; }
[ -d "${HLW_ROOT}" ] || { echo "ERROR: no HLW dataset at ${HLW_ROOT}" >&2; exit 1; }
[ -f "${HLW_ROOT}/metadata.csv" ] || {
  echo "ERROR: ${HLW_ROOT}/metadata.csv missing — is the archive extracted one level deeper?" >&2
  exit 1; }
# Never let a finetune overwrite the encoder it was seeded from.
[ "${OUT}/checkpoint.pth" = "${INIT_FROM}" ] && {
  echo "ERROR: OUT resolves onto INIT_FROM — pick a different OUT_NAME" >&2; exit 1; }

# Resume THIS finetune if it already has a checkpoint (a box can be lost to another user
# mid-run; train.py saves every epoch). --init-from stays separate from --resume.
RESUME=()
if [ -f "${OUT}/checkpoint.pth" ]; then
  RESUME=(--resume "${OUT}/checkpoint.pth")
  echo "[doc] resuming existing finetune at ${OUT}/checkpoint.pth"
fi

# python -u: without it stdout is block-buffered and run.log looks frozen for many minutes.
CMD=("${REPO}/.venv/bin/python" -u -m ropart.train --finetune --finetune-task horizon
     --init-from "${INIT_FROM}"
     --data-path "${HLW_ROOT}"
     --epochs 100 --warmup-epochs 5 --lr 1e-4 --min-lr 1e-6 --eval-every 1
     --batch-size 192 --num_workers 6 --amp-dtype float16
     --seed "${SEED}"
     --output_dir "${OUT}"
     --wandb-mode "${WANDB_MODE:-offline}" --wandb-project "${WANDB_PROJECT:-ropart}"
     --wandb-name "${WNAME}"
     ${RESUME[@]+"${RESUME[@]}"} "$@")   # ${x[@]+…}: empty-array expansion under `set -u`
                                         # is an error in bash 3.2 (macOS), so this keeps
                                         # --dry-run testable off-cluster.

echo "[doc] host=$(hostname -s) arm=${ARM} seed=${SEED} task=horizon"
echo "[doc] init_from=${INIT_FROM}"
echo "[doc] data=${HLW_ROOT}"
echo "[doc] out=${OUT}"
echo "[doc] log=${LOG}"

if [ "${DRY}" = "1" ]; then
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

# Only after the dry-run exit: --dry-run must have no side effects.
mkdir -p "${OUT}"

# Refuse to double-launch on this box: two ~6 GB jobs do not fit in 11 GB.
if pgrep -f "ropart.train.*${OUT_NAME}" >/dev/null 2>&1; then
  echo "ERROR: a run for ${OUT_NAME} is already alive on $(hostname -s)" >&2; exit 1
fi

cd "${REPO}"
# shellcheck disable=SC1091
source ropart/scripts/_doc_env.sh   # WARN about /vol/cuda is benign: the torch wheel
                                    # bundles its own CUDA runtime, only the driver matters.
nohup setsid "${CMD[@]}" > "${LOG}" 2>&1 < /dev/null &
echo "[doc] launched pid=$! -> ${LOG}"
