#!/usr/bin/env bash
# Phase-3 HLW horizon-line finetune on a **standalone DoC lab GPU box** (gpu01-32).
# The orientation-sensitive half of Phase 3; the classification half is run_finetune_doc.sh.
#
#   usage: run_horizon_doc.sh <arm> [--seed N] [--tag NAME] [--dry-run] [extra args]
#          arm in {base, ch2, w30, w90}
#
#   Use --tag for anything throwaway, e.g. a smoke:
#     run_horizon_doc.sh base --tag smoke --epochs 2
#   It gives the run its own OUT_NAME so it cannot be resumed into by the real run.
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
#
# ⚠ READ theta_mae, NOT auc, FOR THE ORIENTATION CLAIM. The AUC is structurally dominated
# by rho: with a square crop the edge offsets are l, r = rho -/+ tan(theta)/2, so a theta
# error enters at *half* weight and a rho error at full weight. At the constant-predictor
# floor that is 0.0127 vs 0.3269 image heights — rho outweighs theta ~26x. A real
# orientation advantage would be nearly invisible in the AUC. horizon_run_epoch reports
# theta_mae (degrees) and rho_mae separately for exactly this reason; auc is kept for
# comparability with the published protocol, not as the headline.
#
# BASELINES to compare epoch-1 numbers against (HLW v1 train-mean constant predictor,
# measured 2026-08-13 on val): auc 27.71%, theta_mae 1.459 deg, rho_mae 0.3269. An arm
# near those has learned nothing. Because the head predicts standardised targets, an
# untrained head starts exactly at this floor.
#
# SETTLED 2026-08-13, no longer a pre-flight check: Y_AXIS_DOWN = False (metadata y is
# maths-convention/up), confirmed against the NG-DSAC reference loader and 8/8 --verify
# renders. See ropart/hlw.py. Re-run --verify only if that constant is ever touched.
#
# Data layout expected at ${HLW_ROOT}: images/, split/{train,val,test}.txt, metadata.csv.
# HLW v1 train is 16,906 images (not the ~98k the paper's full collection suggests), and
# per-epoch validation uses the 885-image **val** split — test is scored once, at the end,
# so that "best epoch" is not a selection over 100 evaluations on the test set.
#
# HLW_ROOT defaults to the **preprocessed** root ($WS/hlw224). The raw root also works and
# is label-identical, but every epoch re-reads full-resolution originals over NFS, which
# makes the run I/O-bound rather than GPU-bound. Override with HLW_ROOT=... if needed.
set -euo pipefail

ARM="${1:?usage: run_horizon_doc.sh <base|ch2|w30|w90> [--seed N] [--dry-run] [extra args]}"
shift
DRY=0
SEED=0
TAG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --seed)    SEED="${2:?--seed needs a value}"; shift 2 ;;
    --tag)     TAG="${2:?--tag needs a value}"; shift 2 ;;
    *)         break ;;      # everything else is passed through to ropart.train
  esac
done

REPO="${REPO:-${HOME}/msc_thesis/RoPART}"
WS="${WS:-/vol/gpudata/${USER}-ropart-in100}"
HLW_ROOT="${HLW_ROOT:-${WS}/hlw224}"
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
# --tag isolates throwaway runs (smokes, config trials) into their own OUT_NAME. Without
# it a 2-epoch smoke writes checkpoint.pth into the *real* run's directory, and the real
# run then silently auto-resumes from it — having already spent epochs 1-2 on a cosine
# schedule compressed to 2 epochs. That failure is invisible: the run completes, the
# curves look plausible, and the LR schedule is quietly wrong for the whole arm.
if [ -n "${TAG}" ]; then
  OUT_NAME="${OUT_NAME}_${TAG}"
  WNAME="${WNAME}-${TAG}"
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
