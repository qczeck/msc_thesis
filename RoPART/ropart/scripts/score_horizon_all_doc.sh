#!/usr/bin/env bash
# Score all four finished HLW horizon arms on one split, on a standalone DoC lab box.
#
#   usage: score_horizon_all_doc.sh <split> [arms...]
#          split in {val, test, test_seen, test_heldout};  arms default to base ch2 w30 w90
#
# ⚠ RUN `val` FIRST. It is the gate: each arm must reproduce its final val_theta_mae from
# run.log to ~1e-3. The test split has no such reference, so scoring it first would leave a
# silently mangled weight load with no tell — and this project has already been bitten twice
# by faults that converge perfectly well while computing the wrong thing. Only run `test`
# once `val` has matched for all four.
#
# HLW_TARGET_STATS is NOT optional: without it ropart/hlw.py falls back to the v1 constants
# and every score is computed against a wrong centre and scale. Silent. Same trap as the
# training runs.
set -euo pipefail

SPLIT="${1:?usage: score_horizon_all_doc.sh <val|test|test_seen|test_heldout> [arms...]}"
shift
ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=(base ch2 w30 w90)

REPO="${REPO:-${HOME}/msc_thesis/RoPART}"
WS="${WS:-/vol/gpudata/${USER}-ropart-in100}"
HLW_ROOT="${HLW_ROOT:-${WS}/hlw224}"
TAG="${TAG:-_v2}"

: "${HLW_TARGET_STATS:?export HLW_TARGET_STATS=mean_theta,mean_rho,std_theta,std_rho first}"

cd "$REPO"
echo "[score-all] $(date -u +%FT%TZ) host=$(hostname -s) split=${SPLIT} root=${HLW_ROOT}"
echo "[score-all] HLW_TARGET_STATS=${HLW_TARGET_STATS}"

for arm in "${ARMS[@]}"; do
  CKPT="${WS}/out/hlw_ft_${arm}_p32${TAG}/checkpoint.pth"
  if [ ! -f "$CKPT" ]; then
    echo "[score-all] FATAL: no checkpoint at ${CKPT}" >&2
    exit 1
  fi
  echo "[score-all] --- ${arm} ---"
  PYTHONPATH=. .venv/bin/python -u -m ropart.scripts.score_horizon \
    --checkpoint "$CKPT" --data-path "$HLW_ROOT" --split "$SPLIT"
done

echo "[score-all] $(date -u +%FT%TZ) done — CSVs are scores_${SPLIT}.csv beside each checkpoint"
