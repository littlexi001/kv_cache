#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
OUT="$ROOT/outputs/20260716_128k_attention_bottleneck"

export PATH=/home/fdong/miniconda3/envs/moe/bin:$PATH
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
cd "$ROOT"
mkdir -p "$OUT"

run_case() {
  local name=$1
  local attention_fraction=$2
  local hit_rate=$3
  local index_order=${4:-random}
  taskset -c 0-23,48-71 "$PYTHON" \
    src/benchmark_mapped_host_gather_20260715.py \
    --history_count 131072 \
    --selected_fraction "$attention_fraction" \
    --attention_fraction "$attention_fraction" \
    --cache_fraction 0.032 \
    --cache_hit_rate "$hit_rate" \
    --index_order "$index_order" \
    --warmup 5 \
    --repeats 50 \
    --output "$OUT/${name}.json" \
    > "$OUT/${name}.log" 2>&1
}

run_case attn010_hit050 0.010 0.50
run_case attn010_hit079 0.010 0.79
run_case attn010_hit095 0.010 0.95
run_case attn015_hit079 0.015 0.79
run_case attn015_hit079_token_sorted 0.015 0.79 token_sorted
run_case attn020_hit079 0.020 0.79

"$PYTHON" src/summarize_128k_attention_bottleneck_20260716.py \
  --input_dir "$OUT" \
  --output_dir "$OUT/summary"

bash scripts/launch_128k_candidate_overlap_trace_20260716.sh
