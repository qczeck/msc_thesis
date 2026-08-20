#!/usr/bin/env bash
# Wait for the HLWv2 metadata regeneration, verify it, then launch the four v2 horizon arms.
#
#   usage: hlwv2_verify_and_launch.sh        # env-tunable: SRC PRE PRE_LOG REPO POLL MAX_WAIT
#
# This is the 2026-08-20 successor to `hlwv2_gates.sh`, and the difference matters.
# `hlwv2_gates.sh` *computed* the target-standardisation constants and then stopped, because
# a wrong-but-plausible constant is invisible at run time and thirteen hours of compute
# should not start on a number no human has read. That reasoning was vindicated: the
# constants it printed on 2026-08-18 were garbage (theta mean -40.8 deg) from the 7-column
# metadata bug, and reading them is what caught it.
#
# Here the constants are already computed and already read — they are pinned in EXPECT_*
# below. So the gate is no longer a discovery step but a *verification against a
# pre-committed value*, which is safe to run unattended: any disagreement is a hard stop,
# and the value that gets trained against is the one a human approved, not one derived
# moments earlier by the same script that launches.
#
# Every gate is fail-closed: it exits non-zero and launches nothing.
#
#   gate 0  the regeneration actually finished  (metadata.csv and split/ are written only
#           at the very end of the pass, so their absence means "not finished" no matter
#           how many images are on disk)
#   gate 1  preprocessing is label-preserving   (raw vs preprocessed round-trip)
#   gate 2  the preprocessed tree reproduces the raw tree's targets on the same sample,
#           AND both land inside sanity bounds a horizon dataset must satisfy
#   gate 3  after launch, the first arm's run.log shows the pinned constants and the v2
#           data root — otherwise the arm is void, so kill the chain rather than let it run
set -u

SRC="${SRC:-/vol/bitbucket/msk123/hlwv2/raw/hlw}"
PRE="${PRE:-/vol/bitbucket/msk123/hlwv2/hlw224}"
PRE_LOG="${PRE_LOG:-$HOME/preprocess_v2.log}"
REPO="${REPO:-$HOME/msc_thesis/RoPART}"
OUT_ROOT="${OUT_ROOT:-/vol/gpudata/msk123-ropart-in100/out}"
POLL="${POLL:-60}"
MAX_WAIT="${MAX_WAIT:-10800}"        # 3 h; the regeneration should need well under 1 h
N_CHECK="${N_CHECK:-500}"
N_STATS="${N_STATS:-3000}"
EXPECT_ROWS="${EXPECT_ROWS:-100553}"

# The constants computed and human-read on 2026-08-20 (fit = v2 train, 96,617 images).
# mean_theta,mean_rho,std_theta,std_rho — theta in radians, rho in image heights.
EXPECT_STATS="${EXPECT_STATS:--4.1879e-04,0.291614,7.0494e-02,0.549827}"

log() { echo "[launch] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

cd "$REPO" || { log "FATAL: no repo at $REPO"; exit 2; }
log "host=$(hostname -s) src=$SRC pre=$PRE"
log "pinned HLW_TARGET_STATS=$EXPECT_STATS"

# ---- gate 0: wait for the regeneration to finish ---------------------------------
log "gate 0: waiting for the metadata regeneration (log=$PRE_LOG, poll=${POLL}s)"
waited=0
while :; do
  if grep -q '^\[supervisor\].*COMPLETED on attempt' "$PRE_LOG" 2>/dev/null; then
    log "gate 0: supervisor reports COMPLETED"; break
  fi
  if grep -q '^\[supervisor\].*GAVE UP' "$PRE_LOG" 2>/dev/null; then
    log "FATAL: supervisor GAVE UP — regeneration did not finish"; exit 1
  fi
  if [ "$waited" -ge "$MAX_WAIT" ]; then
    log "FATAL: still unfinished after ${MAX_WAIT}s — investigate, do not launch"; exit 1
  fi
  sleep "$POLL"; waited=$((waited + POLL))
done

for f in "$PRE/metadata.csv" "$PRE/split"; do
  [ -e "$f" ] || { log "FATAL: $f missing — the pass did not complete"; exit 1; }
done
rows=$(wc -l < "$PRE/metadata.csv" | tr -d ' ')
log "gate 0 PASS: metadata.csv has ${rows} rows"
[ "$rows" -eq "$EXPECT_ROWS" ] || { log "FATAL: expected ${EXPECT_ROWS} rows, got ${rows}"; exit 1; }

# The regenerated file must differ from the corrupt one, or the fixed parser never ran.
BAD="$HOME/hlwv2_metadata_BAD_7col_bug.csv"
if [ -f "$BAD" ] && cmp -s "$BAD" "$PRE/metadata.csv"; then
  log "FATAL: metadata.csv is byte-identical to the known-corrupt backup — the fixed"
  log "FATAL: parser did not run. Check the md5 of ropart/scripts/preprocess_hlw.py."
  exit 1
fi
log "gate 0 PASS: metadata.csv differs from the known-corrupt backup"

# ---- gate 1: label round-trip -----------------------------------------------------
log "gate 1: label round-trip on ${N_CHECK} train images"
if ! PYTHONPATH=. .venv/bin/python -u -m ropart.scripts.check_hlw_preprocess \
      --raw "$SRC" --pre "$PRE" --split train --n "$N_CHECK"; then
  log "FATAL: round-trip FAILED — labels do not survive preprocessing, do not train"; exit 1
fi
log "gate 1 PASS"

# ---- gate 2: the two roots must agree, and both must be sane ----------------------
# Same first-N names on both roots (they share split/), so these are directly comparable.
# The raw side reads no images at all (v2 metadata carries the dimensions).
log "gate 2: target statistics on ${N_STATS} train images, preprocessed vs raw"
PRE_OUT="$HOME/hlwv2_stats_pre.txt"; RAW_OUT="$HOME/hlwv2_stats_raw.txt"
PYTHONPATH=. .venv/bin/python -u -m ropart.scripts.hlw_target_stats \
  --root "$PRE" --fit train --eval val --limit "$N_STATS" > "$PRE_OUT" 2>&1 || {
    log "FATAL: stats failed on the preprocessed root:"; cat "$PRE_OUT"; exit 1; }
PYTHONPATH=. .venv/bin/python -u -m ropart.scripts.hlw_target_stats \
  --root "$SRC" --fit train --eval val --limit "$N_STATS" --dims-from-metadata > "$RAW_OUT" 2>&1 || {
    log "FATAL: stats failed on the raw root:"; cat "$RAW_OUT"; exit 1; }

PYTHONPATH=. .venv/bin/python - "$PRE_OUT" "$RAW_OUT" <<'PY' || { log "FATAL: gate 2 FAILED — see above"; exit 1; }
import re, sys

def stats(path):
    for line in open(path):
        if line.startswith("export HLW_TARGET_STATS="):
            return [float(v) for v in line.split("=", 1)[1].strip().split(",")]
    raise SystemExit(f"no export line in {path}")

pre, raw = stats(sys.argv[1]), stats(sys.argv[2])
names = ("mean_theta", "mean_rho", "std_theta", "std_rho")
# theta is exact through a uniform resize; rho carries sub-pixel rounding the round-trip
# check bounds at 1.4e-3 image heights. These tolerances are far inside that.
tol = (1e-3, 5e-3, 1e-3, 5e-3)
ok = True
print("  channel      preprocessed            raw        |diff|      tol")
for n, a, b, t in zip(names, pre, raw, tol):
    d = abs(a - b)
    flag = "" if d <= t else "   <-- EXCEEDS"
    print(f"  {n:<11} {a:>14.6e} {b:>14.6e} {d:>10.2e} {t:>8.0e}{flag}")
    ok &= d <= t
if not ok:
    print("\nFAIL: the preprocessed tree does not reproduce the raw tree's targets.")
    raise SystemExit(1)

# Sanity bounds a horizon dataset must satisfy regardless of version. These are what the
# 7-column bug violated: it produced mean_theta = -0.712 rad (-40.8 deg).
mt, mr, st, sr = pre
checks = [("|mean_theta| < 0.02 rad", abs(mt) < 0.02),
          ("0.03 < std_theta < 0.15 rad", 0.03 < st < 0.15),
          ("|mean_rho| < 1.0", abs(mr) < 1.0),
          ("0.2 < std_rho < 1.5", 0.2 < sr < 1.5)]
print()
for label, good in checks:
    print(f"  {'PASS' if good else 'FAIL'}  {label}")
if not all(g for _, g in checks):
    print("\nFAIL: targets are outside the bounds a horizon dataset must satisfy.")
    raise SystemExit(1)
print("\ngate 2 checks OK")
PY
log "gate 2 PASS"

# ---- launch -----------------------------------------------------------------------
log "launching the four v2 arms: base ch2 w30 w90 (serial, this box, --tag v2)"
cd "$REPO"
HLW_ROOT="$PRE" \
HLW_TARGET_STATS="$EXPECT_STATS" \
MAX_WAIT=28800 \
  nohup setsid bash ropart/scripts/run_horizon_all_doc.sh base ch2 w30 w90 -- --tag v2 \
    > "$HOME/horizon_v2_all.log" 2>&1 < /dev/null &
log "launched run_horizon_all_doc.sh -> $HOME/horizon_v2_all.log"

# ---- gate 3: the first arm must actually be using the v2 root and the pinned stats --
# HLW_TARGET_STATS is exported once into run_horizon_all_doc.sh's environment and inherited
# by all four arms, so verifying the first proves the plumbing for all of them.
FIRST_LOG="$OUT_ROOT/hlw_ft_base_p32_v2/run.log"
log "gate 3: waiting up to 600s for ${FIRST_LOG}"
waited=0
while [ ! -s "$FIRST_LOG" ] && [ "$waited" -lt 600 ]; do sleep 15; waited=$((waited + 15)); done
if [ ! -s "$FIRST_LOG" ]; then
  log "FATAL: first arm produced no run.log in 600s — killing the chain"
  pkill -f '[r]un_horizon_all_doc'; pkill -f '[r]opart[.]train'; exit 1
fi
waited=0
while ! grep -q 'hlw target_mean' "$FIRST_LOG" 2>/dev/null && [ "$waited" -lt 600 ]; do
  sleep 15; waited=$((waited + 15))
done
grep -E 'hlw target_mean|hlw data:' "$FIRST_LOG" | head -5 | sed 's/^/[launch]   /'
if ! grep -q 'target_mean: (-0.00041879' "$FIRST_LOG" 2>/dev/null; then
  log "FATAL: the first arm is NOT using the pinned target stats — the arm is void."
  log "FATAL: killing the chain rather than burning 13 h on a wrong normalisation."
  pkill -f '[r]un_horizon_all_doc'; pkill -f '[r]opart[.]train'; exit 1
fi
if ! grep -q 'hlwv2' "$FIRST_LOG" 2>/dev/null; then
  log "FATAL: the first arm is NOT reading the hlwv2 data root — killing the chain"
  pkill -f '[r]un_horizon_all_doc'; pkill -f '[r]opart[.]train'; exit 1
fi
log "gate 3 PASS: first arm is on the v2 root with the pinned constants"
log "ALL GATES PASSED — four arms running. Watch: grep '^\[all\]' \$HOME/horizon_v2_all.log"
