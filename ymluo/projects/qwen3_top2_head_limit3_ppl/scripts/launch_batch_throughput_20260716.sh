#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
OUT=$ROOT/results/20260716_batch_throughput
LOG_ROOT=$ROOT/outputs/logs/20260716_batch_throughput

export PATH=/home/fdong/miniconda3/envs/moe/bin:$PATH
mkdir -p "$OUT" "$LOG_ROOT"
cd "$ROOT"

for length in 32000 64000; do
  for batch in 1 2 4; do
    output="$OUT/l${length}_b${batch}.json"
    if [[ -s "$output" ]]; then continue; fi
    CUDA_VISIBLE_DEVICES=0,1,2,3 taskset -c 0-23,48-71 "$PYTHON" \
      src/run_hierarchical_batch_throughput_20260716.py \
      --model_name_or_path "$MODEL" \
      --output "$output" \
      --batch_size "$batch" \
      --history_tokens "$length" \
      --query_tokens 256 \
      --eval_tokens 256 \
      --projection_dim 64 \
      --index_bits 4 \
      --candidate_fraction 0.015 \
      --exact_cache_fraction 0.032 \
      --stream_group_size 2 \
      --prefill_chunk_tokens 2048 \
      --dtype float16 \
      --device cuda \
      --device_map auto \
      > "$LOG_ROOT/l${length}_b${batch}.log" 2>&1
  done
done

"$PYTHON" src/summarize_batch_throughput_20260716.py \
  --input_dir "$OUT" \
  --output_dir "$OUT/summary" \
  --expected_cases 6 \
  > "$LOG_ROOT/summary.log" 2>&1

touch "$OUT/COMPLETE"
