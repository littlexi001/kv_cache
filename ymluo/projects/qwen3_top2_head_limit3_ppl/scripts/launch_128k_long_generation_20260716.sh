#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
OUT=results/20260716_128k_long_generation_religion_w3_m2048

export PATH=/home/fdong/miniconda3/envs/moe/bin:$PATH
cd "$ROOT"
mkdir -p "$OUT" outputs/logs

full_output="$OUT/full_kv.json"
sparse_output="$OUT/pca64_int4_top1p5_stream2_cache3p2.json"

if [[ ! -s "$full_output" ]]; then
  CUDA_VISIBLE_DEVICES=0,1,2,3 taskset -c 0-23,48-71 "$PYTHON" \
    src/run_full_cache_ppl_baseline_20260715.py \
    --model_name_or_path "$MODEL" \
    --output "$full_output" \
    --topic religion \
    --history_tokens 128000 \
    --query_tokens 256 \
    --eval_tokens 2048 \
    --window_index 3 \
    --window_stride_tokens 128512 \
    --prefill_chunk_tokens 2048 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    > outputs/logs/20260716_128k_long_generation_full.log 2>&1 &
  full_pid=$!
else
  full_pid=""
fi

if [[ ! -s "$sparse_output" ]]; then
  CUDA_VISIBLE_DEVICES=4,5,6,7 taskset -c 24-47,72-95 "$PYTHON" \
    src/run_hierarchical_physical_cache_ppl_20260715.py \
    --model_name_or_path "$MODEL" \
    --output "$sparse_output" \
    --topic religion \
    --history_tokens 128000 \
    --query_tokens 256 \
    --eval_tokens 2048 \
    --window_index 3 \
    --window_stride_tokens 128512 \
    --projection_dim 64 \
    --index_bits 4 \
    --candidate_fraction 0.015 \
    --attention_fraction 0.015 \
    --candidate_selection_mode per_head_stream \
    --rerank_selection_mode shared_sum \
    --exact_cache_fraction 0.032 \
    --stream_group_size 2 \
    --candidate_refresh_interval 1 \
    --host_append_mode async \
    --conversion_mode async \
    --directory_backend fused \
    --known_reference_ppl 1.0 \
    --prefill_chunk_tokens 2048 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    > outputs/logs/20260716_128k_long_generation_sparse.log 2>&1 &
  sparse_pid=$!
else
  sparse_pid=""
fi

[[ -n "$full_pid" ]] && wait "$full_pid"
[[ -n "$sparse_pid" ]] && wait "$sparse_pid"

"$PYTHON" src/summarize_128k_long_generation_20260716.py \
  --input_dir "$OUT" \
  --output "$OUT/summary.json"
