# Sourced by the ImageNet sbatch scripts: resolve paths + load the environment on a
# DoC GPU-cluster compute node. Keeps the two sbatch scripts to just their #SBATCH
# headers and the one train command that differs between them.
#
# Overridable via env: WS_NAME, REPO, CUDA_VERSION, WANDB_MODE, WANDB_PROJECT.

set -euo pipefail

WS_NAME="${WS_NAME:-ropart-in100}"
REPO="${REPO:-${HOME}/msc_thesis/RoPART}"

# Resolve the CephFS workspace holding the data + outputs.
if command -v ws_find >/dev/null 2>&1 && ws_find "${WS_NAME}" >/dev/null 2>&1; then
  WS="$(ws_find "${WS_NAME}")"
else
  WS="/vol/gpudata/${USER}-${WS_NAME}"   # ws_allocate's default naming
fi
DATA="${WS}/imagenet100"
OUT_ROOT="${WS}/out"
[ -d "${DATA}/train" ] || { echo "ERROR: ${DATA}/train not found — run setup_imagenet_cluster.sh on gpu30 first" >&2; exit 1; }

# CUDA (PyTorch wheels bundle their own runtime; this is the DoC convention).
# shellcheck disable=SC1091
source "${REPO}/ropart/scripts/_doc_env.sh"

# Use the repo venv (built on a lab PC; cu12 wheel covers A40/A100 sm_80/86).
export PATH="${REPO}/.venv/bin:${PATH}"

# wandb: default OFFLINE on compute nodes (likely no egress). Put the run dir on NFS
# home so a `wandb sync` loop from gpu30 gives a near-live dashboard. The 1-epoch
# smoke decides whether online works here; pass WANDB_MODE=online to use it.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${REPO}/wandb"
mkdir -p "${WANDB_DIR}"

cd "${REPO}"
echo "[slurm] node=$(hostname) repo=${REPO}"
echo "[slurm] data=${DATA} out_root=${OUT_ROOT} wandb_mode=${WANDB_MODE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# run_imagenet <control> <wandb-name> [extra train args...]
run_imagenet() {
  local control="$1"; shift
  local wname="$1"; shift
  # OUT_NAME lets a variant run (e.g. bounded rotation, same 'raw' control) use its
  # own checkpoint dir instead of clobbering/resuming the plain in100_<control> run.
  local out="${OUT_ROOT}/${OUT_NAME:-in100_${control}}"
  local resume=()
  [ -f "${out}/checkpoint.pth" ] && resume=(--resume "${out}/checkpoint.pth") && \
    echo "[slurm] resuming from ${out}/checkpoint.pth"
  python -m ropart.train \
    --data-set IMAGENET --data-path "${DATA}" \
    --model deit_base_patch16_224 \
    --control "${control}" --query-type patch_cat \
    --num_pairs 2048 --batch-size 192 \
    --epochs 100 --warmup-epochs 5 --lr 5e-4 --eval-every 5 \
    --num_workers 8 \
    --output_dir "${out}" \
    --wandb-mode "${WANDB_MODE}" --wandb-project "${WANDB_PROJECT:-ropart}" \
    --wandb-name "${wname}" \
    "${resume[@]}" "$@"
}
