#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
TRACE=results/20260801_real_qkv_trace_computer128k/computer.pt

cd "$PROJECT"

pids=()
for gpu_rank in "0:16" "1:32" "2:64" "3:96" "4:128"; do
  gpu="${gpu_rank%%:*}"
  rank="${gpu_rank##*:}"
  output="results/20260803_riskrank${rank}_trace128k_v5"
  mkdir -p "$output"
  env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
    src/analyze_qksieve_tail_partition_calibration_20260803.py \
    --traces "$TRACE" \
    --output_dir "$output" \
    --sample_counts 256 \
    --block_sizes 128 \
    --value_rank "$rank" \
    > "$output/run.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
