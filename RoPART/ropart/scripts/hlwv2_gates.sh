#!/usr/bin/env bash
# Post-preprocessing gates for HLWv2, run unattended on the box.
#
#   usage: hlwv2_gates.sh          # env-tunable: SRC PRE PRE_LOG POLL MAX_WAIT N_CHECK
#
# Runs the three things that must hold before the four v2 horizon arms may start, in
# order, and **stops without launching**. Launching is a separate, human-approved step:
# the arms are ~13 h of compute keyed to a target-standardisation constant computed here,
# and a wrong-but-plausible constant is invisible at run time (the model trains happily
# against a slightly wrong centre and scale). So the numbers get read before they are used.
#
#   gate 0  preprocessing actually finished  — metadata.csv and split/ are written only
#           at the very end of the pass, so their absence means "not finished" no matter
#           how many images are on disk
#   gate 1  preprocessing is label-preserving — same filename through HLWDataset on the
#           raw and preprocessed roots must give the same (theta, rho)
#   gate 2  HLWv2's own target statistics    — v1's constants are wrong for v2
#
# Every gate is fail-closed: it exits non-zero and launches nothing.
set -u

SRC="${SRC:-/vol/bitbucket/msk123/hlwv2/raw/hlw}"
PRE="${PRE:-/vol/bitbucket/msk123/hlwv2/hlw224}"
PRE_LOG="${PRE_LOG:-$HOME/preprocess_v2.log}"
REPO="${REPO:-$HOME/msc_thesis/RoPART}"
POLL="${POLL:-300}"
MAX_WAIT="${MAX_WAIT:-36000}"        # 10 h, well past the ~5 h the pass should need
N_CHECK="${N_CHECK:-500}"
EXPECT_ROWS="${EXPECT_ROWS:-100553}"

log() { echo "[gate] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

cd "$REPO" || { log "FATAL: no repo at $REPO"; exit 2; }
log "host=$(hostname -s) src=$SRC pre=$PRE"

# ---- gate 0: wait for the preprocessing pass to finish ---------------------------
log "gate 0: waiting for preprocessing to complete (log=$PRE_LOG, poll=${POLL}s)"
waited=0
while :; do
  if grep -q '^\[supervisor\].*COMPLETED on attempt' "$PRE_LOG" 2>/dev/null; then
    log "gate 0: supervisor reports COMPLETED"
    break
  fi
  if grep -q '^\[supervisor\].*GAVE UP' "$PRE_LOG" 2>/dev/null; then
    log "FATAL: supervisor GAVE UP — preprocessing did not finish"
    exit 1
  fi
  if [ "$waited" -ge "$MAX_WAIT" ]; then
    log "FATAL: preprocessing still unfinished after ${MAX_WAIT}s — investigate, do not launch"
    exit 1
  fi
  sleep "$POLL"
  waited=$((waited + POLL))
done

for f in "$PRE/metadata.csv" "$PRE/split"; do
  [ -e "$f" ] || { log "FATAL: $f missing — the pass did not complete"; exit 1; }
done
rows=$(wc -l < "$PRE/metadata.csv" | tr -d ' ')
log "gate 0 PASS: metadata.csv has ${rows} rows"
if [ "$rows" -ne "$EXPECT_ROWS" ]; then
  log "WARN: expected ${EXPECT_ROWS} rows — images that failed to convert are dropped from"
  log "WARN: metadata.csv while the split files still list them. Check before launching."
fi

# ---- gate 1: the preprocessing must be label-preserving --------------------------
log "gate 1: label round-trip on ${N_CHECK} train images"
if ! PYTHONPATH=. .venv/bin/python -u -m ropart.scripts.check_hlw_preprocess \
      --raw "$SRC" --pre "$PRE" --split train --n "$N_CHECK"; then
  log "FATAL: round-trip FAILED — the labels do not survive preprocessing, do not train"
  exit 1
fi
log "gate 1 PASS"

# ---- gate 2: HLWv2's own target statistics ---------------------------------------
log "gate 2: computing HLWv2 target statistics (fit=train, floor on val)"
STATS_OUT="$HOME/hlwv2_target_stats.txt"
if ! PYTHONPATH=. .venv/bin/python -u -m ropart.scripts.hlw_target_stats \
      --root "$PRE" --fit train --eval val > "$STATS_OUT" 2>&1; then
  log "FATAL: hlw_target_stats failed:"
  cat "$STATS_OUT"
  exit 1
fi
cat "$STATS_OUT"
EXPORT_LINE=$(grep -m1 '^export HLW_TARGET_STATS=' "$STATS_OUT" || true)
[ -n "$EXPORT_LINE" ] || { log "FATAL: no export line in stats output"; exit 1; }
log "gate 2 PASS"

# ---- stop. Launching is deliberately a separate, approved step. ------------------
cat <<EOF

=========================================================================
ALL GATES PASSED — STOPPING BEFORE LAUNCH, BY DESIGN.

v1 constants, for comparison:
  export HLW_TARGET_STATS=2.0770e-04,0.237300,5.8653e-02,0.569070
v2 constants, computed above:
  ${EXPORT_LINE}

Sanity: theta/rho means should stay near zero/small, and the sds should be the
same order as v1's (5.87e-2, 0.569). A wildly different sd means something is
wrong with the tree, not with the dataset — investigate before launching.

To launch the four arms once the numbers look right:

  cd \$HOME/msc_thesis/RoPART
  HLW_ROOT=${PRE} \\
  ${EXPORT_LINE#export } \\
  MAX_WAIT=28800 \\
    nohup setsid bash ropart/scripts/run_horizon_all_doc.sh base ch2 w30 w90 -- --tag v2 \\
      > \$HOME/horizon_v2_all.log 2>&1 < /dev/null &

(as one line, HLW_TARGET_STATS=... — the '--tag v2' is NOT optional: without it
 OUT_NAME resolves onto the completed v1 runs and they get auto-resumed.)
=========================================================================
EOF
log "done — gates passed, awaiting approval to launch"
