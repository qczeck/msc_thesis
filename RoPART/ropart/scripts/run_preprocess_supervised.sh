#!/usr/bin/env bash
# Run preprocess_hlw to completion, restarting it if it dies.
#
# Why this exists. The HLWv2 preprocessing pass is a multi-hour to ~29-hour job (v1
# measured 290 images/min, I/O-bound at the NFS ceiling), run unattended on a lab GPU box
# that has NO SCHEDULER: another user can claim the machine, or the process can be OOM-
# killed, at any moment. Without supervision that silently costs the whole weekend, and
# the failure is invisible until someone looks.
#
# preprocess_hlw --skip-existing is idempotent — it reuses existing outputs via two header
# reads and converts only the remainder — and writes each image atomically, so a killed run
# never leaves a truncated file for the next attempt to accept. That makes plain restarting
# both safe and cheap, which is all this wrapper does.
#
# Usage (always under nohup setsid, so it survives the SSH connection dropping):
#
#   cd $HOME/msc_thesis/RoPART
#   nohup setsid bash ropart/scripts/run_preprocess_supervised.sh \
#       /vol/bitbucket/msk123/hlwv2/raw /vol/bitbucket/msk123/hlwv2/hlw224 6 \
#       > $HOME/preprocess_v2.log 2>&1 < /dev/null &
#
# Watch it with:  grep '^\[supervisor\]' $HOME/preprocess_v2.log
#
# NB the destination should be on /vol/bitbucket, not $WS — gpudata is quota'd at ~52 GB
# and cannot hold a preprocessed v2 alongside everything else.

set -u

SRC=${1:?usage: $0 <src-root> <dst-root> [workers] [max-attempts]}
DST=${2:?usage: $0 <src-root> <dst-root> [workers] [max-attempts]}
WORKERS=${3:-6}
MAX_ATTEMPTS=${4:-50}
BACKOFF=60

# `date -Is` is GNU-only and silently prints nothing on macOS, which would leave every
# line of a 29-hour log untimestamped. This spelling works on both.
log() { echo "[supervisor] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

log "src=$SRC dst=$DST workers=$WORKERS host=$(hostname)"
[ -d "$SRC" ] || { log "FATAL: src root does not exist"; exit 2; }

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  log "attempt $attempt/$MAX_ATTEMPTS starting"
  # --skip-existing is passed on every attempt including the first: on a fresh destination
  # nothing exists so it is a no-op, and passing it unconditionally means there is no
  # "first run" special case to get wrong on a restart.
  if PYTHONPATH=. .venv/bin/python -u -m ropart.scripts.preprocess_hlw \
        --src "$SRC" --dst "$DST" --workers "$WORKERS" --skip-existing; then
    log "COMPLETED on attempt $attempt"
    log "images written: $(find "$DST/images" -type f 2>/dev/null | wc -l)"
    exit 0
  fi
  rc=$?
  log "attempt $attempt exited $rc — retrying in ${BACKOFF}s"
  sleep "$BACKOFF"
done

log "GAVE UP after $MAX_ATTEMPTS attempts"
exit 1
