#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
"${PYTHON_BIN}" -m pip install --upgrade pip
# The PyTorch binary in common Alibaba Cloud images was compiled against the
# NumPy 1.x C API. Install this constraint first so later dependency resolution
# cannot leave a pre-existing NumPy 2.x in the environment.
"${PYTHON_BIN}" -m pip install --upgrade --force-reinstall "numpy>=1.26,<2"
"${PYTHON_BIN}" -m pip install -r "${PROJECT_ROOT}/requirements.txt"
"${PYTHON_BIN}" - <<'PY'
import numpy, torch, transformers, datasets, tensorboard
if int(numpy.__version__.split(".", 1)[0]) >= 2:
    raise RuntimeError(f"NumPy 1.x is required by this PyTorch build, found {numpy.__version__}")
if transformers.__version__ != "4.51.3":
    raise RuntimeError(
        f"This package pins transformers 4.51.3 for Qwen3 and trusted local Trainer resume; "
        f"found {transformers.__version__}"
    )
if datasets.__version__ != "3.6.0":
    raise RuntimeError(
        f"LongBench still uses a Hugging Face loading script and requires datasets 3.6.0; "
        f"found {datasets.__version__}"
    )
print({"numpy": numpy.__version__, "torch": torch.__version__, "transformers": transformers.__version__, "datasets": datasets.__version__, "tensorboard": tensorboard.__version__})
print({"cuda_available": torch.cuda.is_available(), "gpu_count": torch.cuda.device_count()})
PY
