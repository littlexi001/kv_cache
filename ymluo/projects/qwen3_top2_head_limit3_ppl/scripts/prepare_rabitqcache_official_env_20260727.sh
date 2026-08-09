#!/usr/bin/env bash
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-/home/fdong/miniconda3}"
ENV_ROOT="${ENV_ROOT:-$CONDA_ROOT/envs/rabitqcache}"
RABITQ_ROOT="${RABITQ_ROOT:-/home/fdong/ymluo/external/RaBitQCache}"

# The target server is RTX 3090 (SM86). Limiting architectures avoids
# compiling unused SM80/SM90 kernels, and parallel jobs keep setup bounded.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export MAX_JOBS="${MAX_JOBS:-8}"
export NVCC_THREADS="${NVCC_THREADS:-4}"

if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
  "$CONDA_ROOT/bin/conda" create -y -p "$ENV_ROOT" python=3.10 pip
fi

"$ENV_ROOT/bin/python" -m pip install \
  "pip<26" \
  "setuptools>=70.1" \
  wheel \
  ninja \
  packaging

# Both CUDA extensions import torch while preparing their wheels. Install the
# runtime first and disable isolated builds so setup.py can see it.
"$ENV_ROOT/bin/python" -m pip install "torch==2.5.0"
"$ENV_ROOT/bin/python" -m pip install \
  "transformers==4.45.2" \
  "accelerate==1.0.1" \
  "datasets==3.0.1"
"$ENV_ROOT/bin/python" -m pip install \
  --no-build-isolation \
  "flashinfer-python==0.2.0.post1"
"$ENV_ROOT/bin/python" -m pip install \
  --no-build-isolation \
  "flash-attn==2.6.3"
"$ENV_ROOT/bin/python" -m pip install --no-deps -e "$RABITQ_ROOT"
"$ENV_ROOT/bin/python" -m pip check
"$ENV_ROOT/bin/python" - <<'PY'
import torch
import transformers

print(
    {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
)
PY
