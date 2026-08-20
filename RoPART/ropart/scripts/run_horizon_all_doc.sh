#!/usr/bin/env bash
# Run the remaining HLW horizon arms back-to-back on ONE box, unattended.
#
#   usage: run_horizon_all_doc.sh [arm ...] [-- <extra args for run_horizon_doc.sh>]
#          default arms: ch2 w30 w90
#
# Everything after `--` is forwarded verbatim to each run_horizon_doc.sh invocation, which
# is what makes a second dataset runnable without editing this file. For HLWv2:
#
#   HLW_ROOT=/vol/bitbucket/msk123/hlwv2/hlw224 MAX_WAIT=28800 \
#     bash ropart/scripts/run_horizon_all_doc.sh base ch2 w30 w90 -- --tag v2
#
# ⚠ The `--tag` is not optional there. Without it OUT_NAME resolves to the *completed v1*
# directory hlw_ft_<arm>_p32, whose checkpoint run_horizon_doc.sh would then auto-resume —
# yielding a "finished" run that never saw the v2 data, with nothing in the log to say so.
#
# Why serially on one box rather than in parallel across three. All four arms must share
# hardware *and* batch size, or a between-arm gap could be numerics or throughput rather
# than the pretext objective — which destroys the one-factor-per-step attribution the
# baseline -> ch2 -> wphi01 triple exists to provide. The IN-100 write-up already had to
# record a box-assignment confound discovered after the fact. Three idle 2080 Tis are the
# *same model*, which is not the same thing as the same box, so this trades ~1.2 h of
# wall-clock for an attribution that cannot be questioned.
#
# It also cannot double-book the GPU: two ~6 GB jobs do not fit in 11 GB, and
# run_horizon_doc.sh refuses to start a second run for the same OUT_NAME anyway.
#
# Launch under nohup setsid so it survives the SSH connection dropping:
#
#   ssh -A shell1 'ssh -o BatchMode=yes gpu29 bash -s' <<'EOF'
#   cd $HOME/msc_thesis/RoPART
#   nohup setsid bash ropart/scripts/run_horizon_all_doc.sh \
#     > $HOME/horizon_all.log 2>&1 < /dev/null &
#   EOF
#
# Watch:  grep '^\[all\]' $HOME/horizon_all.log
#         grep '^\[ft ' $WS/out/hlw_ft_<arm>_p32/run.log | tail -3

set -u

# Split "arms -- extra args" into the two lists.
ARMS=()
EXTRA=()
seen_sep=0
for a in "$@"; do
  if [ "$seen_sep" = "0" ] && [ "$a" = "--" ]; then seen_sep=1; continue; fi
  if [ "$seen_sep" = "1" ]; then EXTRA+=("$a"); else ARMS+=("$a"); fi
done
[ ${#ARMS[@]} -eq 0 ] && ARMS=(ch2 w30 w90)

REPO="${REPO:-${HOME}/msc_thesis/RoPART}"
POLL=60
# Stuck-job cap. A v1 arm is ~35 min, so 4 h is a generous default — but a v2 arm is
# ~3.3 h (5.7x the training images), which would leave almost no headroom, so raise it
# via the environment for v2: MAX_WAIT=28800.
MAX_WAIT="${MAX_WAIT:-14400}"

log() { echo "[all] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

wait_idle() {
  local waited=0
  # Bracket pattern: a plain `pgrep -f ropart.train` matches the very shell running the
  # check, and would report a phantom job forever on an idle box.
  while pgrep -f '[r]opart[.]train' >/dev/null 2>&1; do
    sleep "$POLL"
    waited=$((waited + POLL))
    if [ "$waited" -ge "$MAX_WAIT" ]; then
      log "FATAL: a ropart.train has been alive ${MAX_WAIT}s — refusing to queue behind it"
      return 1
    fi
  done
  return 0
}

cd "$REPO" || { log "FATAL: no repo at $REPO"; exit 2; }
log "host=$(hostname -s) arms=${ARMS[*]} extra=${EXTRA[*]-} hlw_root=${HLW_ROOT:-<default>} max_wait=${MAX_WAIT}"

for arm in "${ARMS[@]}"; do
  log "waiting for the GPU to go idle before launching '$arm'"
  wait_idle || exit 1
  log "launching '$arm'"
  if ! bash ropart/scripts/run_horizon_doc.sh "$arm" ${EXTRA[@]+"${EXTRA[@]}"}; then
    log "FATAL: launcher failed for '$arm' — stopping rather than skipping"
    exit 1
  fi
  # run_horizon_doc.sh backgrounds the job and returns immediately, so give it time to
  # appear in the process table before wait_idle looks for it. Without this the next
  # iteration sees an idle box and launches two arms onto one 11 GB GPU.
  sleep 120
done

log "waiting for the final arm to finish"
wait_idle || exit 1
log "ALL ARMS COMPLETE: ${ARMS[*]}"
