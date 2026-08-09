#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
OUT=$ROOT/results/20260716_128k_candidate_overlap_trace
LOG_ROOT=$ROOT/outputs/logs/20260716_128k_candidate_overlap_trace

export PATH=/home/fdong/miniconda3/envs/moe/bin:$PATH
mkdir -p "$OUT" "$LOG_ROOT"
cd "$ROOT"

run_case() {
  local devices=$1
  local cpus=$2
  local topic=$3
  local reference_ppl=$4
  local output="$OUT/${topic}.json"
  [[ -s "$output" ]] && return 0
  CUDA_VISIBLE_DEVICES="$devices" taskset -c "$cpus" "$PYTHON" \
    src/run_hierarchical_physical_cache_ppl_20260715.py \
    --model_name_or_path "$MODEL" \
    --output "$output" \
    --topic "$topic" \
    --history_tokens 128000 \
    --query_tokens 256 \
    --eval_tokens 8 \
    --window_index 0 \
    --projection_dim 64 \
    --index_bits 4 \
    --candidate_fraction 0.01 \
    --attention_fraction 0.01 \
    --candidate_selection_mode per_head_stream \
    --stream_group_size 2 \
    --candidate_refresh_interval 1 \
    --exact_cache_fraction 0.032 \
    --directory_backend fused \
    --record_candidate_overlap \
    --known_reference_ppl "$reference_ppl" \
    --prefill_chunk_tokens 2048 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    > "$LOG_ROOT/${topic}.log" 2>&1
}

run_case 0,1,2,3 0-23,48-71 religion 15.16051075145507 &
left=$!
run_case 4,5,6,7 24-47,72-95 computer 60.43246449071301 &
right=$!
wait "$left" "$right"

"$PYTHON" src/summarize_128k_candidate_overlap_20260716.py \
  --input_dir "$OUT" \
  --output "$OUT/summary.json"
