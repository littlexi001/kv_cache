#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
TEMPLATE="${TEMPLATE:-${ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
PROJECTION="${PROJECTION:-${ROOT}/data/public_baselines/binarypc/llama3-1-8b-ins-projection-mixlen-mixdata.pt}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260801_binarypc_exactrerank4x_native128k_gpu01}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"
HISTORY_TOKENS="${HISTORY_TOKENS:-131040}"
EVAL_TOKENS="${EVAL_TOKENS:-16}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
mkdir -p "${RUN_ROOT}/logs"
cd "${ROOT}"

for item in "medicine|20260812" "sports_both|20260811"; do
  IFS="|" read -r topic seed <<<"${item}"
  output_dir="${RUN_ROOT}/${topic}_seed${seed}"
  mkdir -p "${output_dir}"
  if [[ -f "${output_dir}/ALL_COMPLETE" ]]; then
    echo "SKIP ${topic}/seed${seed}"
    continue
  fi
  CUDA_VISIBLE_DEVICES=0,1 "${PYTHON}" -u \
    src/run_qksieve_coldskip_longcontext_quality_20260730.py \
    --model_name_or_path "${MODEL}" \
    --template "${TEMPLATE}" \
    --binarypc_projection_path "${PROJECTION}" \
    --output_dir "${output_dir}" \
    --history_tokens "${HISTORY_TOKENS}" \
    --eval_tokens "${EVAL_TOKENS}" \
    --topic "${topic}" \
    --seed "${seed}" \
    --repeat_topic_stream_if_short \
    --prefill_chunk_tokens 1024 \
    --dataset_cache_dir "${DATASET_CACHE_DIR}" \
    --dtype float16 \
    --device cuda \
    --device_map balanced \
    --max_memory_per_gpu_gib 22 \
    --variants binarypc_exactrerank4x_k1280 \
    >"${output_dir}/run.log" 2>&1
done

touch "${RUN_ROOT}/ALL_COMPLETE"
