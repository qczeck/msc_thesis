# Sourced helper: load CUDA on the Imperial DoC machines.
#
# Per DoC, CUDA is not on the local disk — versions live under /vol/cuda and are
# activated by sourcing their setup.sh. This file is a no-op on machines without
# /vol/cuda (e.g. the Mac dev box), so the same run scripts work in both places.
#
# Override the version:  CUDA_VERSION=11.8.0 source ropart/scripts/_doc_env.sh
# List what's available:  ls /vol/cuda
#
# NB: PyTorch's pip wheels bundle their own CUDA runtime, so a GPU run only needs
# the NVIDIA *driver* (nvidia-smi), which is installed system-wide. Loading
# /vol/cuda is the department convention and is required to *compile* CUDA code;
# we source it for completeness and correct LD_LIBRARY_PATH.

if [ -d /vol/cuda ]; then
  CUDA_VERSION="${CUDA_VERSION:-12.2.2}"
  _cuda_setup="/vol/cuda/${CUDA_VERSION}/setup.sh"
  if [ -f "${_cuda_setup}" ]; then
    # setup.sh references unset vars; relax nounset just for the source.
    _had_u=0; case "$-" in *u*) _had_u=1;; esac
    set +u
    # shellcheck disable=SC1090
    . "${_cuda_setup}"
    [ "${_had_u}" = 1 ] && set -u
    echo "[doc] loaded CUDA ${CUDA_VERSION}"
  else
    echo "[doc] WARN: ${_cuda_setup} not found. Run 'ls /vol/cuda' and set CUDA_VERSION." >&2
  fi
fi
