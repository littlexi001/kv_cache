#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
OUT=$ROOT/results/20260716_128k_multitopic_windows_w3
LOG_ROOT=$ROOT/outputs/logs/20260716_128k_multitopic_windows_w3

export PATH=/home/fdong/miniconda3/envs/moe/bin:$PATH
mkdir -p "$OUT" "$LOG_ROOT"
cd "$ROOT"

run_case() {
  local topic=$1
  local window=$2
  local devices=$3
  local stem="${topic}_w${window}"
  if [[ ! -s "$OUT/${stem}_full.json" ]]; then
    CUDA_VISIBLE_DEVICES="$devices" "$PYTHON" \
      src/run_full_cache_ppl_baseline_20260715.py \
      --model_name_or_path "$MODEL" \
      --output "$OUT/${stem}_full.json" \
      --topic "$topic" \
      --history_tokens 128000 \
      --query_tokens 256 \
      --eval_tokens 256 \
      --window_index "$window" \
      --window_stride_tokens 128512 \
      --prefill_chunk_tokens 2048 \
      --dtype float16 \
      --device cuda \
      --device_map auto \
      > "$LOG_ROOT/${stem}_full.log" 2>&1
  fi
  if [[ ! -s "$OUT/${stem}_sparse.json" ]]; then
    CUDA_VISIBLE_DEVICES="$devices" "$PYTHON" \
      src/run_hierarchical_physical_cache_ppl_20260715.py \
      --model_name_or_path "$MODEL" \
      --output "$OUT/${stem}_sparse.json" \
      --topic "$topic" \
      --history_tokens 128000 \
      --query_tokens 256 \
      --eval_tokens 256 \
      --window_index "$window" \
      --window_stride_tokens 128512 \
      --projection_dim 64 \
      --index_bits 4 \
      --candidate_fraction 0.015 \
      --attention_fraction 0.015 \
      --candidate_selection_mode per_head_stream \
      --stream_group_size 2 \
      --candidate_refresh_interval 1 \
      --exact_cache_fraction 0.032 \
      --directory_backend fused \
      --prefill_cache_mode dynamic \
      --prefill_chunk_tokens 2048 \
      --dtype float16 \
      --device cuda \
      --device_map auto \
      > "$LOG_ROOT/${stem}_sparse.log" 2>&1
  fi
}

run_queue() {
  local devices=$1
  local parity=$2
  local cpus
  if [[ "$parity" -eq 0 ]]; then cpus=0-23,48-71; else cpus=24-47,72-95; fi
  taskset -pc "$cpus" "$BASHPID" >/dev/null
  local index=0
  local topic window
  for topic in computer sports medicine space politics religion; do
    for window in 0 1 2; do
      if [[ $((index % 2)) -eq "$parity" ]]; then
        echo "starting topic=$topic window=$window devices=$devices"
        run_case "$topic" "$window" "$devices"
      fi
      index=$((index + 1))
    done
  done
}

run_queue 0,1,2,3 0 > "$LOG_ROOT/queue0.log" 2>&1 &
pid0=$!
run_queue 4,5,6,7 1 > "$LOG_ROOT/queue1.log" 2>&1 &
pid1=$!
echo "queue0_pid=$pid0 queue1_pid=$pid1"
wait "$pid0" "$pid1"
touch "$OUT/COMPLETE"
