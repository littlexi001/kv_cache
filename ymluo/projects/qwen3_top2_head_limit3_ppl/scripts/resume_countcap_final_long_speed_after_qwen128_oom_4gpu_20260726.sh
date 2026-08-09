#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
LLAMA_MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
QWEN_MODEL=/home/fdong/models/Qwen3-4B-Instruct
DATASET_CACHE=/home/fdong/ymluo/datasets/sklearn
RUN_ROOT=$ROOT/results/20260726_countcap_final_long_speed_multimodel_4gpu
RUNNER=$ROOT/src/run_direct_countcap_denseprompt_ppl_20260725.py

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

mkdir -p "$RUN_ROOT/logs"
cd "$ROOT"

log() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$RUN_ROOT/logs/resume.log"
}

run_case() {
  local gpus="$1"
  local model="$2"
  local model_label="$3"
  local length="$4"
  local topic="$5"
  local device_map="$6"
  local output="$RUN_ROOT/$model_label/length${length}_${topic}"
  local log_path="$RUN_ROOT/logs/${model_label}_${length}_${topic}.log"

  if [[ -s "$output/case_summary.json" ]] &&
    [[ -s "$output/token_results.csv" ]]; then
    log "skip completed $model_label $length $topic"
    return
  fi

  mkdir -p "$output"
  log "run $model_label $length $topic on GPU $gpus"
  CUDA_VISIBLE_DEVICES="$gpus" "$PYTHON" -u "$RUNNER" \
    --model_name_or_path "$model" \
    --output_dir "$output" \
    --topics "$topic" \
    --window_indices 0,1 \
    --methods full_attention,direct_countcap \
    --history_tokens "$length" \
    --eval_tokens 256 \
    --window_stride_tokens 128512 \
    --target_anchor_tokens 128000 \
    --direct_fraction 0.06 \
    --direct_min_tokens 256 \
    --direct_max_tokens 1280 \
    --projection_dim 48 \
    --sample_count 256 \
    --prefill_chunk_tokens 2048 \
    --cache_mode auto \
    --preallocated_cache_min_tokens 14000 \
    --dataset_cache_dir "$DATASET_CACHE" \
    --device_map "$device_map" \
    >"$log_path" 2>&1
}

run_parallel_pair() {
  local model="$1"
  local label="$2"
  local length="$3"
  local gpu_a="$4"
  local gpu_b="$5"
  local map="$6"
  run_case "$gpu_a" "$model" "$label" "$length" mixed_a "$map" &
  local pid_a=$!
  run_case "$gpu_b" "$model" "$label" "$length" mixed_b "$map" &
  local pid_b=$!
  wait "$pid_a" "$pid_b"
}

# A single 3090 cannot hold Qwen3-4B plus a 128K FP16 KV cache. Split
# layers and their caches over two cards, and run the two topics serially.
run_case 0,1 "$QWEN_MODEL" qwen3_4b 128000 mixed_a balanced
run_case 0,1 "$QWEN_MODEL" qwen3_4b 128000 mixed_b balanced

run_parallel_pair "$LLAMA_MODEL" llama31_8b 64000 0,1 2,3 balanced
run_case 0,1,2,3 "$LLAMA_MODEL" llama31_8b 128000 mixed_a balanced
run_case 0,1,2,3 "$LLAMA_MODEL" llama31_8b 128000 mixed_b balanced

"$PYTHON" -u \
  src/summarize_countcap_multimodel_long_speed_20260726.py \
  --run_root "$RUN_ROOT" \
  --output_dir "$RUN_ROOT/analysis" \
  >"$RUN_ROOT/logs/analysis.log" 2>&1

touch "$RUN_ROOT/ALL_COMPLETE"
log "complete: $RUN_ROOT"
