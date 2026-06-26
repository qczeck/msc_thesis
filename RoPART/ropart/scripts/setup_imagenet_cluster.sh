#!/usr/bin/env bash
# Stage ImageNet-100 for the GPU-cluster pretraining runs. Run this on a DoC lab
# machine with network + pip (e.g. gpu30) — NOT on gpucluster2/3 head nodes (the
# guide forbids pip/git there). Idempotent: re-running resumes the download.
#
#   ssh gpu30viashell1
#   cd ~/msc_thesis/RoPART && ropart/scripts/setup_imagenet_cluster.sh
#
# Result: $WS/imagenet100/{train,val}/<wnid>/*.jpg on a /vol/gpudata CephFS
# workspace (because /vol/bitbucket is ~full). Prints the data path to use as
# --data-path in the sbatch scripts.
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> RoPART/

WS_NAME="${WS_NAME:-ropart-in100}"
HF_DATASET="${IN100_HF_DATASET:-clane9/imagenet-100}"

# 1. Allocate (or reuse) a CephFS workspace.  ws_allocate is idempotent-ish: it
#    prints the path and extends the lifetime if it already exists.
if command -v ws_allocate >/dev/null 2>&1; then
  WS="$(ws_allocate "${WS_NAME}" 365)"
else
  echo "ERROR: ws_allocate not found — are you on a DoC machine that mounts /vol/gpudata?" >&2
  exit 1
fi
echo "[in100] workspace: ${WS}"

DATA="${WS}/imagenet100"
export HF_HOME="${WS}/hf_cache"   # keep the ~15 GB HF cache OFF the NFS home quota
mkdir -p "${DATA}" "${HF_HOME}"

# 2. Ensure the prep-only deps are in the venv (datasets is not a training dep, so
#    it is intentionally kept out of requirements.txt — installed here on the lab PC).
# shellcheck disable=SC1091
source ropart/scripts/_doc_env.sh
.venv/bin/python -m pip install -q --upgrade "datasets>=2.18" "pillow>=10.2"

# 3. Materialise the ImageFolders.
.venv/bin/python ropart/scripts/_download_imagenet100.py "${DATA}" "${HF_DATASET}"

# 4. Verify layout: 100 class dirs per split, non-trivial counts.
echo "--- verify ---"
for split in train val; do
  ncls=$(find "${DATA}/${split}" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
  nimg=$(find "${DATA}/${split}" -type f | wc -l | tr -d ' ')
  echo "[in100] ${split}: ${ncls} classes, ${nimg} images"
  [ "${ncls}" -eq 100 ] || echo "[in100] WARN: expected 100 classes in ${split}, got ${ncls}" >&2
done

# 5. Optionally reclaim the HF cache now that ImageFolders are written.
if [ "${KEEP_HF_CACHE:-0}" != "1" ]; then
  echo "[in100] clearing HF cache at ${HF_HOME} (set KEEP_HF_CACHE=1 to keep)"
  rm -rf "${HF_HOME}"
fi

echo "[in100] DONE. Use this as --data-path:"
echo "        ${DATA}"
