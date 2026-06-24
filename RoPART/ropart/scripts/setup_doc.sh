#!/usr/bin/env bash
# One-time environment bootstrap for an Imperial DoC GPU machine (e.g. 2080ti).
# Builds a fresh Linux/CUDA virtualenv for RoPART. Safe to re-run (idempotent).
#
#   ls /vol/cuda                                  # pick a CUDA version
#   CUDA_VERSION=12.2.2 ropart/scripts/setup_doc.sh
#
# The .venv and data/ dirs are gitignored, so they are NOT synced from the Mac —
# this rebuilds them on the GPU box. CIFAR-100 downloads on first run.
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> RoPART/

# shellcheck disable=SC1091
source ropart/scripts/_doc_env.sh

PY="${PYTHON:-python3.12}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "ERROR: '$PY' not found. Try a DoC python, e.g.:" >&2
  echo "  ls /vol/linux/bin/python3* ; or set PYTHON=python3" >&2
  exit 1
fi
echo "[doc] using $($PY --version) at $(command -v "$PY")"

if [ ! -d .venv ]; then
  "$PY" -m venv .venv
  echo "[doc] created .venv"
fi
.venv/bin/python -m pip install --upgrade pip
# requirements.txt pins torch>=2.2; on Linux the default PyPI wheel is a CUDA
# build (cu12x) — fine for the 2080ti (Turing, sm_75).
.venv/bin/python -m pip install -r requirements.txt

echo "--- GPU sanity ---"
.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "| bundled cuda", torch.version.cuda)
ok = torch.cuda.is_available()
print("cuda available:", ok)
if ok:
    print("device:", torch.cuda.get_device_name(0))
else:
    print("WARN: no CUDA device visible — check nvidia-smi and the NVIDIA driver.")
PY
echo "[doc] setup done. Next: launch training in tmux (see ropart/scripts/pretrain_baseline.sh)."
