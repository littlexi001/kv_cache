#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
PYTHON_DIR="${PYTHON_DIR:-/home/fdong/miniconda3/envs/moe/bin}"
OUTPUT="${OUTPUT:-${ROOT}/outputs/dynamic_kv_multisample30_v1}"

mapfile -t FREE_GPUS < <(
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F',' '{gsub(/ /, "", $0); if (($2 + 0) < 1000 && ($3 + 0) < 10) print $1}'
)
if ((${#FREE_GPUS[@]} == 0)); then
  echo "No idle GPU is available" >&2
  exit 1
fi
GPU_LIST="$(IFS=,; echo "${FREE_GPUS[*]}")"
WORLD_SIZE="${#FREE_GPUS[@]}"

cd "${ROOT}"
rm -rf "${OUTPUT}"
echo "Using idle GPUs: ${GPU_LIST}"
CUDA_VISIBLE_DEVICES="${GPU_LIST}" "${PYTHON_DIR}/torchrun" \
  --standalone --nproc_per_node="${WORLD_SIZE}" \
  src/run_dynamic_kv_multisample.py \
  --corpus_dir data/real_longbench_docqa_10m_clean_record64 \
  --output_dir "${OUTPUT}" \
  --queries_per_dataset 10 \
  --max_new_tokens 128
