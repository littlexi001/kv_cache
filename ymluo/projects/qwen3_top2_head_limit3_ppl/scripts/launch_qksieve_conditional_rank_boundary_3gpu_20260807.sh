#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
TRACE_ROOT="${TRACE_ROOT:-$ROOT/results/tail_shrinkage_realqkv_longbench_20260807}"
OUTPUT="${OUTPUT:-$ROOT/results/conditional_rank_boundary_realqkv_20260807}"
MODEL="${MODEL:-/home/fdong/qksieve_iclr2027/models/Qwen3-4B-Instruct-2507}"

export PATH="/home/fdong/miniconda3/envs/nanogpt/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTPUT/logs"
cd "$ROOT"

run_condition() {
  local gpu="$1"
  local rank="$2"
  local top_k="$3"
  local condition="r${rank}_k${top_k}"
  for case_name in narrative32k narrative64k narrative128k lcc64k qmsum64k; do
    local output_dir="$OUTPUT/$condition/$case_name"
    mkdir -p "$output_dir"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
      src/analyze_qksieve_tail_partition_calibration_20260803.py \
      --traces "$TRACE_ROOT/traces/${case_name}.pt" \
      --output_dir "$output_dir" \
      --model_name_or_path "$MODEL" \
      --device cuda \
      --top_k "$top_k" \
      --sample_counts 256 \
      --block_sizes 256 \
      --conditional_dims 8 \
      --conditional_fit_stride 32 \
      --tail_sampling systematic \
      --key_sample_stride 32 \
      --value_sample_stride 32 \
      --query_shrinkage 0.75 \
      --key_rate_budget 15 \
      --value_rank "$rank" \
      --value_bits 4 \
      --value_scale_block 256 \
      --value_metric wo_group \
      --max_records_per_trace 3 \
      >"$OUTPUT/logs/${condition}_${case_name}.log" 2>&1
    touch "$output_dir/COMPLETE"
  done
  touch "$OUTPUT/${condition}_COMPLETE"
}

run_condition 0 8 1120 & pid0=$!
run_condition 1 8 1280 & pid1=$!
run_condition 2 4 1280 & pid2=$!

status=0
for pid in "$pid0" "$pid1" "$pid2"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi
touch "$OUTPUT/ALL_COMPLETE"
