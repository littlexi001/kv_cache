#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
GPU="${QKSIEVE_GPU:-0}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/20260728_qksieve_qfused_correctness}"

if [[ ! "$GPU" =~ ^[0-5]$ ]]; then
  echo "QKSIEVE_GPU must be one of physical GPUs 0-5" >&2
  exit 2
fi

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
mkdir -p "$OUT_ROOT"
cd "$ROOT"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
  src/validate_qksieve_qfused_matrix_20260728.py \
  --lengths "${QKSIEVE_LENGTHS:-4096,32768}" \
  --group_counts "${QKSIEVE_GROUP_COUNTS:-4,8}" \
  --dtypes "${QKSIEVE_DTYPES:-float16,bfloat16}" \
  --trials "${QKSIEVE_TRIALS:-3}" \
  --warmup "${QKSIEVE_WARMUP:-10}" \
  --iterations "${QKSIEVE_ITERATIONS:-100}" \
  --min_query_prepare_speedup "${QKSIEVE_MIN_PREP_SPEEDUP:-1.05}" \
  --min_selection_speedup "${QKSIEVE_MIN_SELECTION_SPEEDUP:-1.00}" \
  --output "$OUT_ROOT/validation_matrix.json" \
  2>&1 | tee "$OUT_ROOT/run.log"
