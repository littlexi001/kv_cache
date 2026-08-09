#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
TEMPLATE="${TEMPLATE:-${ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
PROJECTION="${PROJECTION:-${ROOT}/data/public_baselines/binarypc/llama3-1-8b-ins-projection-mixlen-mixdata.pt}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260801_binarypc_exactrerank8x_native128k_gpu01_v2}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
mkdir -p "${RUN_ROOT}/logs"
cd "${ROOT}"

run_case() {
  local topic="$1"
  local seed="$2"
  local out="${RUN_ROOT}/${topic}_seed${seed}"
  mkdir -p "${out}"
  if [[ -f "${out}/ALL_COMPLETE" ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES=0,1 \
    "${PYTHON}" -u src/run_qksieve_coldskip_longcontext_quality_20260730.py \
    --model_name_or_path "${MODEL}" \
    --template "${TEMPLATE}" \
    --binarypc_projection_path "${PROJECTION}" \
    --output_dir "${out}" \
    --history_tokens 131040 \
    --eval_tokens 16 \
    --topic "${topic}" \
    --repeat_topic_stream_if_short \
    --prefill_chunk_tokens 1024 \
    --dataset_cache_dir "${DATASET_CACHE_DIR}" \
    --seed "${seed}" \
    --dtype float16 \
    --device cuda \
    --device_map balanced \
    --max_memory_per_gpu_gib 22 \
    --variants binarypc_exactrerank8x_k1280 \
    >"${out}/run.log" 2>&1
  "${PYTHON}" - "${out}/summary.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
row = next(item for item in payload["rows"] if item["variant"] != "full_attention")
assert row["candidate_overfetch"] == 8.0, row["candidate_overfetch"]
assert "overfetch8x" in row["selector"], row["selector"]
PY
  touch "${out}/ALL_COMPLETE"
}

run_case medicine 20260812
run_case sports_both 20260811
touch "${RUN_ROOT}/ALL_COMPLETE"
