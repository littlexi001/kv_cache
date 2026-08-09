#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
TRACE_ROOT=$ROOT/results/20260715_real_qk_traces_32k
OUT=$ROOT/results/20260716_svd_int2_grouped_recall_32k

export PATH=/home/fdong/miniconda3/envs/moe/bin:$PATH
cd "$ROOT"
mkdir -p "$OUT/sports" "$OUT/medicine" "$OUT/combined" outputs/logs

run_topic() {
  local gpu=$1
  local topic=$2
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
    src/analyze_svd_index_recall_20260716.py \
    --trace_path "$TRACE_ROOT/${topic}.pt" \
    --output_dir "$OUT/$topic" \
    --topic "$topic" \
    --device cuda \
    --top_fraction 0.02 \
    --sample_stride 32 \
    --ranks 32,48,64 \
    > "outputs/logs/20260716_svd_int2_grouped_recall_${topic}.log" 2>&1
}

run_topic 0 sports &
left=$!
run_topic 4 medicine &
right=$!
wait "$left" "$right"

"$PYTHON" src/summarize_svd_index_recall_20260716.py \
  --input_glob "$OUT/*/per_head_query.csv" \
  --output_dir "$OUT/combined" \
  > outputs/logs/20260716_svd_int2_grouped_recall_summary.log 2>&1

echo "complete: $OUT/combined/summary.json"
