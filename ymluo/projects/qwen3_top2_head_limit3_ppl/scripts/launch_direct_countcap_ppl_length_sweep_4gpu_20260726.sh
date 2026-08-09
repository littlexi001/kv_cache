#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON_BIN="${PYTHON_BIN:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/results/20260726_direct_countcap_ppl_length_sweep_4gpu}"
RUNNER="${PROJECT_DIR}/src/run_direct_countcap_denseprompt_ppl_20260725.py"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${RUN_ROOT}/logs"

run_length() {
  local visible_gpus="$1"
  local length="$2"
  local topics="$3"
  local name="$4"
  local device_map="$5"
  CUDA_VISIBLE_DEVICES="${visible_gpus}" "${PYTHON_BIN}" "${RUNNER}" \
    --model_name_or_path "${MODEL_PATH}" \
    --output_dir "${RUN_ROOT}/${name}" \
    --topics "${topics}" \
    --window_indices 0,1,2 \
    --methods full_attention,direct_countcap \
    --history_tokens "${length}" \
    --eval_tokens 256 \
    --window_stride_tokens 128512 \
    --target_anchor_tokens 128000 \
    --direct_fraction 0.06 \
    --direct_min_tokens 256 \
    --direct_max_tokens 1280 \
    --projection_dim 48 \
    --sample_count 256 \
    --prefill_chunk_tokens 2048 \
    --dataset_cache_dir "${DATASET_CACHE_DIR}" \
    --device_map "${device_map}" \
    >"${RUN_ROOT}/logs/${name}.log" 2>&1
}

worker0() {
  run_length 0 2048 mixed_a,mixed_b length2048 auto
  run_length 0 24000 mixed_a,mixed_b length24000 auto
}

worker1() {
  run_length 1 4096 mixed_a,mixed_b length4096 auto
  run_length 1 32000 mixed_a,mixed_b length32000 auto
}

worker2() {
  run_length 2 8192 mixed_a,mixed_b length8192 auto
}

worker3() {
  run_length 3 16000 mixed_a,mixed_b length16000 auto
}

worker0 &
pid0=$!
worker1 &
pid1=$!
worker2 &
pid2=$!
worker3 &
pid3=$!
wait "${pid0}" "${pid1}" "${pid2}" "${pid3}"
touch "${RUN_ROOT}/SMALL_COMPLETE"

run_length 0,1 64000 mixed_a length64000_mixed_a balanced &
pid64a=$!
run_length 2,3 64000 mixed_b length64000_mixed_b balanced &
pid64b=$!
wait "${pid64a}" "${pid64b}"
touch "${RUN_ROOT}/LENGTH64K_COMPLETE"

run_length 0,1,2,3 128000 mixed_a,mixed_b length128000 balanced
touch "${RUN_ROOT}/ALL_COMPLETE"
