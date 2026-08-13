#!/usr/bin/env bash
# Phase-3 IN-100 classification finetune on a **standalone DoC lab GPU box** (gpu01-32),
# as opposed to the SLURM path in ropart/scripts/cluster/.
#
#   usage: run_finetune_doc.sh <arm> [--seed N] [--dry-run] [extra ropart.train args]
#          arm in {base, ch2, w30, w90}
#
# SEED REPLICATION. --seed varies the finetune's stochasticity only — data order,
# augmentation draws, and the fresh pos_embed/clf init (train.py::set_seed). The encoder
# is fixed by --init-from, so this bounds run-to-run variance *given* a pretrain, not
# pretraining variance (that would cost ~30 h/arm to re-pretrain and is out of budget).
# Seed 0 keeps the original OUT_NAME so the 2026-08-07 runs stay resumable; any other
# seed gets its own directory, because sharing one would make the auto-resume path below
# silently continue the seed-0 run instead of starting a new one.
#
# Why this exists rather than reusing cluster/_imagenet_env.sh::finetune_imagenet():
# that helper hardcodes --amp-dtype bfloat16 and --num_workers 16, both wrong here, and
# assumes SLURM auto-resume semantics. Keeping the two paths separate means launching on a
# lab box cannot perturb the (working) SLURM path.
#
# TWO HARDWARE-FORCED DEVIATIONS FROM THE SLURM RECIPE — read before mixing results:
#
#   1. --amp-dtype float16, not bfloat16. The 2080 Ti / TITAN Xp boxes are Turing/Pascal.
#      torch.cuda.is_bf16_supported() returns True on Turing but bf16 is *emulated* with no
#      tensor-core acceleration, so bf16 there is silently slow rather than a clean error.
#   2. --num_workers 6, not 16. These boxes have 8 CPUs (the SLURM nodes allocate 12-16).
#
# >>> ALL FOUR ARMS MUST USE THE SAME RECIPE AND THE SAME HARDWARE CLASS. <<<
# Running two arms here on fp16 and two on SLURM under bf16 makes the arms incomparable and
# destroys the one-factor-per-step attribution that baseline -> ss_ch2 -> ss_wphi01 exists
# to provide. Batch size stays 192 everywhere for the same reason; if it ever has to shrink,
# shrink it for all four.
#
# Everything else matches finetune_imagenet(): 100 epochs, warmup 5, lr 1e-4, cosine to
# 1e-6, batch 192, --eval-every 1. lr 1e-4 is a judgement call (not 5e-4: without layer-wise
# decay that risks washing out the pretrained features). Epoch-1 top-1 is the early signal.
#
# Measured on gpu22 (RTX 2080 Ti, 2026-08-07): 6290/11264 MiB peak; **~142 s/epoch** steady
# state (659 steps), so ~4 h for 100 epochs.
#
# Epoch 0 is an outlier at ~340-370 s and every later epoch is ~142 s: the first pass is
# NFS-cold, and once the ~13 GB of JPEGs are in the page cache (62 GB RAM) the run is
# GPU-bound. Do not extrapolate a schedule from epoch 0 or from a short --max-steps probe;
# a 30-step probe measured ~1.07 s/step against a true steady state of ~0.21 s/step, i.e.
# 5x too pessimistic.
set -euo pipefail

ARM="${1:?usage: run_finetune_doc.sh <base|ch2|w30|w90> [--seed N] [--dry-run] [extra args]}"
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
DATA="${WS}/imagenet100"
OUT_ROOT="${WS}/out"

# arm -> pretrained encoder OUT_NAME | wandb run name
case "${ARM}" in
  base) PRE=in100_translation_p32_e250;              WNAME=ropart-in100-ft-baseline-p32 ;;
  ch2)  PRE=in100_rot_bounded30_p32_ss_ch2_e250;     WNAME=ropart-in100-ft-ss-ch2-30-p32 ;;
  w30)  PRE=in100_rot_bounded30_p32_ss_wphi01_e250;  WNAME=ropart-in100-ft-ss-wphi01-30-p32 ;;
  w90)  PRE=in100_rot_bounded90_p32_ss_wphi01_e250;  WNAME=ropart-in100-ft-ss-wphi01-90-p32 ;;
  *) echo "ERROR: unknown arm '${ARM}' (want base|ch2|w30|w90)" >&2; exit 2 ;;
esac

INIT_FROM="${OUT_ROOT}/${PRE}/checkpoint.pth"
# Seed 0 keeps the un-suffixed name the 2026-08-07 runs used, so they remain resumable.
if [ "${SEED}" = "0" ]; then
  OUT_NAME="in100_ft_${ARM}_p32"
else
  OUT_NAME="in100_ft_${ARM}_p32_s${SEED}"
  WNAME="${WNAME}-s${SEED}"
fi
OUT="${OUT_ROOT}/${OUT_NAME}"
LOG="${OUT}/run.log"

[ -f "${INIT_FROM}" ] || { echo "ERROR: no pretrain checkpoint at ${INIT_FROM}" >&2; exit 1; }
# Same clobber guard as the SLURM helper: never let a finetune overwrite the encoder it was
# seeded from.
[ "${OUT}/checkpoint.pth" = "${INIT_FROM}" ] && {
  echo "ERROR: OUT resolves onto INIT_FROM — pick a different OUT_NAME" >&2; exit 1; }

# Resume THIS finetune if it already has a checkpoint (a box can be lost to another user
# mid-run; train.py saves every epoch, so at most one epoch is lost). --init-from stays
# separate from --resume deliberately.
RESUME=()
if [ -f "${OUT}/checkpoint.pth" ]; then
  RESUME=(--resume "${OUT}/checkpoint.pth")
  echo "[doc] resuming existing finetune at ${OUT}/checkpoint.pth"
fi

# python -u: without it stdout is block-buffered and run.log looks frozen for many minutes.
CMD=("${REPO}/.venv/bin/python" -u -m ropart.train --finetune
     --init-from "${INIT_FROM}"
     --data-set IMAGENET --data-path "${DATA}"
     --epochs 100 --warmup-epochs 5 --lr 1e-4 --min-lr 1e-6 --eval-every 1
     --batch-size 192 --num_workers 6 --amp-dtype float16
     --seed "${SEED}"
     --output_dir "${OUT}"
     --wandb-mode "${WANDB_MODE:-offline}" --wandb-project "${WANDB_PROJECT:-ropart}"
     --wandb-name "${WNAME}"
     ${RESUME[@]+"${RESUME[@]}"} "$@")   # ${x[@]+…}: empty-array expansion under
                                         # `set -u` is an error in bash 3.2 (macOS), so
                                         # this keeps --dry-run testable off-cluster.

echo "[doc] host=$(hostname -s) arm=${ARM} seed=${SEED}"
echo "[doc] init_from=${INIT_FROM}"
echo "[doc] out=${OUT}"
echo "[doc] log=${LOG}"

if [ "${DRY}" = "1" ]; then
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

# Only after the dry-run exit: --dry-run must have no side effects (an earlier version
# created ${OUT} before this check, so merely inspecting a command made directories).
mkdir -p "${OUT}"

# Refuse to double-launch on this box: two 6.3 GB jobs do not fit in 11 GB.
if pgrep -f "ropart.train.*${OUT_NAME}" >/dev/null 2>&1; then
  echo "ERROR: a run for ${OUT_NAME} is already alive on $(hostname -s)" >&2; exit 1
fi

cd "${REPO}"
# shellcheck disable=SC1091
source ropart/scripts/_doc_env.sh   # WARN about /vol/cuda is benign: the torch wheel
                                    # bundles its own CUDA runtime, only the driver matters.
# setsid so the run survives the SSH connection dropping — there is no SLURM .out here.
nohup setsid "${CMD[@]}" > "${LOG}" 2>&1 < /dev/null &
echo "[doc] launched pid=$! -> ${LOG}"
