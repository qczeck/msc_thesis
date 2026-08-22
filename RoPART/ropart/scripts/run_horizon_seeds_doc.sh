#!/usr/bin/env bash
# Seed replication for the HLW horizon arms: run <arms> x <seeds> on ONE box, supervised,
# scoring each arm as it finishes.
#
#   usage: run_horizon_seeds_doc.sh "<arms>" "<seeds>" [-- extra args to ropart.train]
#   e.g.:  run_horizon_seeds_doc.sh "base ch2 w30" "1 2 3 4"  --  --tag v2
#
# WHY THIS EXISTS RATHER THAN A LOOP OVER run_horizon_all_doc.sh.
# run_horizon_all_doc.sh launches an arm, waits for the ropart.train process to disappear,
# and then launches the next one. It never checks that the arm *finished*. An arm killed at
# epoch 40 (OOM, reboot, a co-tenant claiming the GPU) vanishes from the process table just
# like a completed one, so the chain moves on, leaves that arm half-trained, and still logs
# ALL ARMS COMPLETE. Over ~5 h x 12 arms that is the failure mode to expect, and it is
# silent — which is the dangerous part. This script instead checks the arm's own run.log for
# the final epoch and relaunches (auto-resuming) until it gets there.
#
# ORDERING IS BY SEED BLOCK, NOT BY ARM: [s1: base ch2 w30] [s2: ...] ...
# So an interruption at any point still leaves a set of *complete, matched* seed blocks
# rather than three seeds of `base` and none of `w30`. The arm comparison is what the study
# rests on, so it is the thing that must never be left half-collected.
#
# ONE BOX, SERIAL. All seed-0 arms ran on gpu29; keeping every replicate there too means no
# run in the study differs by hardware or batch size. Three idle 2080 Tis are the same GPU
# *model*, which is not the same thing as the same box.
set -euo pipefail

ARMS_STR="${1:?usage: run_horizon_seeds_doc.sh \"<arms>\" \"<seeds>\" [-- extra]}"
SEEDS_STR="${2:?need a seed list, e.g. \"1 2 3 4\"}"
shift 2
EXTRA=()
if [ "${1:-}" = "--" ]; then shift; EXTRA=("$@"); fi

read -r -a ARMS <<< "$ARMS_STR"
read -r -a SEEDS <<< "$SEEDS_STR"

REPO="${REPO:-${HOME}/msc_thesis/RoPART}"
WS="${WS:-/vol/gpudata/${USER}-ropart-in100}"
OUT_ROOT="${WS}/out"
TAG="${TAG:-v2}"
EPOCHS="${EPOCHS:-100}"
POLL="${POLL:-60}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-6}"
# Ceiling on a single attempt. A v2 arm is ~5.1 h; 8 h leaves room for a contended box
# without waiting forever on a wedged one.
MAX_ATTEMPT_SECONDS="${MAX_ATTEMPT_SECONDS:-28800}"
SCORE="${SCORE:-1}"

: "${HLW_ROOT:?export HLW_ROOT=... (the preprocessed v2 root)}"
: "${HLW_TARGET_STATS:?export HLW_TARGET_STATS=... — without it hlw.py silently uses the v1 constants}"

log() { echo "[seeds] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

cd "$REPO" || { log "FATAL: no repo at $REPO"; exit 2; }

# Is this arm finished? The last epoch line in its own run.log is the authority — not the
# process table (a dead arm and a finished arm look identical there) and not the checkpoint
# (which exists from epoch 0 onward).
arm_done() {
  local out="$1" last
  [ -f "${out}/run.log" ] || return 1
  last=$(grep -oE '^\[ft [0-9]+\]' "${out}/run.log" 2>/dev/null | tail -1 | tr -dc '0-9') || true
  [ -n "${last}" ] && [ "${last}" -ge "$((EPOCHS - 1))" ]
}

wait_quiet() {
  local waited=0
  while pgrep -f '[r]opart[.]train' >/dev/null 2>&1; do
    sleep "$POLL"
    waited=$((waited + POLL))
    if [ "$waited" -ge "$MAX_ATTEMPT_SECONDS" ]; then
      log "WARN: a ropart.train has been alive ${MAX_ATTEMPT_SECONDS}s — treating the attempt as wedged"
      return 1
    fi
  done
  return 0
}

TOTAL=$(( ${#ARMS[@]} * ${#SEEDS[@]} )); DONE=0
log "host=$(hostname -s) arms=${ARMS[*]} seeds=${SEEDS[*]} tag=${TAG} total=${TOTAL} epochs=${EPOCHS}"
log "hlw_root=${HLW_ROOT}"
log "HLW_TARGET_STATS=${HLW_TARGET_STATS}"

for seed in "${SEEDS[@]}"; do
  log "===== seed block ${seed} ====="
  for arm in "${ARMS[@]}"; do
    OUT_NAME="hlw_ft_${arm}_p32_s${seed}${TAG:+_${TAG}}"
    OUT="${OUT_ROOT}/${OUT_NAME}"

    if arm_done "$OUT"; then
      log "skip ${OUT_NAME}: already at epoch >= $((EPOCHS - 1))"
    else
      attempt=0
      until arm_done "$OUT"; do
        attempt=$((attempt + 1))
        if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
          log "FATAL: ${OUT_NAME} did not finish in ${MAX_ATTEMPTS} attempts — stopping"
          exit 1
        fi
        # Relaunching is what resumes: run_horizon_doc.sh passes --resume when a
        # checkpoint.pth is present, and train.py restores model, optimizer and the epoch
        # index (the cosine LR is recomputed from that index, so the schedule continues).
        log "launch ${OUT_NAME} attempt ${attempt}/${MAX_ATTEMPTS}"
        wait_quiet || true
        if ! bash ropart/scripts/run_horizon_doc.sh "$arm" --seed "$seed" \
               ${TAG:+--tag "$TAG"} ${EXTRA[@]+"${EXTRA[@]}"}; then
          log "WARN: launcher returned non-zero for ${OUT_NAME}; retrying"
          sleep 30; continue
        fi
        sleep 120          # let the job appear in the process table before we poll for it
        wait_quiet || true
        if arm_done "$OUT"; then
          log "done ${OUT_NAME} after attempt ${attempt}"
        else
          last=$(grep -oE '^\[ft [0-9]+\]' "${OUT}/run.log" 2>/dev/null | tail -1 | tr -dc '0-9' || true)
          log "WARN: ${OUT_NAME} stopped at epoch ${last:-<none>} of $((EPOCHS - 1)) — resuming"
        fi
      done
    fi

    DONE=$((DONE + 1))
    log "progress ${DONE}/${TOTAL}"

    if [ "$SCORE" = "1" ]; then
      # Gate on val first: the arm must reproduce its own final run.log numbers before its
      # test scores mean anything. score_horizon writes scores_<split>.csv beside the ckpt.
      for split in val test; do
        if ! PYTHONPATH=. .venv/bin/python -u -m ropart.scripts.score_horizon \
               --checkpoint "${OUT}/checkpoint.pth" --data-path "${HLW_ROOT}" --split "$split"; then
          log "WARN: scoring ${OUT_NAME} on ${split} failed — continuing; rescore by hand later"
          break
        fi
      done
    fi
  done
done

log "ALL SEED BLOCKS COMPLETE: arms=${ARMS[*]} seeds=${SEEDS[*]}"
