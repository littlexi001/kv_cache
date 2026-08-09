#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
OUTPUT_ROOT="$PROJECT/results/20260717_progressive_retrieval_offline"
LOG_ROOT="$PROJECT/logs"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$PROJECT"

CUDA_VISIBLE_DEVICES=1 "$PYTHON" \
  src/analyze_progressive_attention_stability_20260717.py \
  --trace_path results/20260715_real_qk_traces_32k/sports.pt \
  --topic sports \
  --output_dir "$OUTPUT_ROOT/stability_sports" \
  > "$LOG_ROOT/progressive_stability_sports.log" 2>&1 &
p1=$!

CUDA_VISIBLE_DEVICES=2 "$PYTHON" \
  src/analyze_progressive_attention_stability_20260717.py \
  --trace_path results/20260715_real_qk_traces_32k/medicine.pt \
  --topic medicine \
  --output_dir "$OUTPUT_ROOT/stability_medicine" \
  > "$LOG_ROOT/progressive_stability_medicine.log" 2>&1 &
p2=$!

CUDA_VISIBLE_DEVICES=3 "$PYTHON" \
  src/analyze_progressive_dimension_cascade_20260717.py \
  --trace_path results/20260715_real_qk_traces_32k/sports.pt \
  --topic sports \
  --output_dir "$OUTPUT_ROOT/cascade_sports" \
  > "$LOG_ROOT/progressive_cascade_sports.log" 2>&1 &
p3=$!

CUDA_VISIBLE_DEVICES=4 "$PYTHON" \
  src/analyze_progressive_dimension_cascade_20260717.py \
  --trace_path results/20260715_real_qk_traces_32k/medicine.pt \
  --topic medicine \
  --output_dir "$OUTPUT_ROOT/cascade_medicine" \
  > "$LOG_ROOT/progressive_cascade_medicine.log" 2>&1 &
p4=$!

status=0
for pid in "$p1" "$p2" "$p3" "$p4"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
