#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/20260726_countcap_theory_closure}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}/logs"

run_probe() {
  local gpu="$1"
  local budget="$2"
  local tag="$3"
  local output="${OUTPUT_ROOT}/budget_probe_32k/${tag}"
  if [[ -s "${output}/case_summary.json" ]]; then
    return
  fi
  local fraction
  fraction="$("${PYTHON}" -c "print(${budget} / 32000)")"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" \
    src/run_direct_countcap_denseprompt_ppl_20260725.py \
    --model_name_or_path "${MODEL}" \
    --output_dir "${output}" \
    --topics mixed_a,mixed_b \
    --window_indices 0,1 \
    --methods direct_countcap \
    --history_tokens 32000 \
    --eval_tokens 128 \
    --target_anchor_tokens 128000 \
    --direct_fraction "${fraction}" \
    --direct_min_tokens 1 \
    --direct_max_tokens "${budget}" \
    --projection_dim 48 \
    --sample_count 256 \
    --prefill_chunk_tokens 2048 \
    --cache_mode auto \
    --device cuda \
    --device_map auto
}

(
  run_probe 2 640 cross_gpu2_budget640
  run_probe 2 1280 cross_gpu2_budget1280
) >"${OUTPUT_ROOT}/logs/budget_cross_gpu2.log" 2>&1 &
gpu2_pid=$!

(
  run_probe 3 320 cross_gpu3_budget320
  run_probe 3 960 cross_gpu3_budget960
) >"${OUTPUT_ROOT}/logs/budget_cross_gpu3.log" 2>&1 &
gpu3_pid=$!

status=0
if ! wait "${gpu2_pid}"; then
  status=1
fi
if ! wait "${gpu3_pid}"; then
  status=1
fi
exit "${status}"
