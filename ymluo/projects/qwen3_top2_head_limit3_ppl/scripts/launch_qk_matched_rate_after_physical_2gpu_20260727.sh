#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
PREREQUISITE=$ROOT/results/20260727_qk_progressive_refinement_32k
TRACE_ROOT=$ROOT/results/20260727_qkv_value_sensitive_32k/traces
RUN_ROOT=$ROOT/results/20260727_qk_matched_rate_all_dims_32k
LOG_ROOT=$RUN_ROOT/logs

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin
export PYTHONPATH=$ROOT/src
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT"

while [[ ! -e "$PREREQUISITE/ALL_COMPLETE" ]]; do
  if ! pgrep -f '^bash scripts/launch_qk_progressive_refinement_after_physical_2gpu_20260727.sh$' >/dev/null; then
    echo "progressive-refinement prerequisite exited without ALL_COMPLETE" >&2
    exit 1
  fi
  sleep 60
done

topics=(sports medicine)
for topic in "${topics[@]}"; do
  trace=$TRACE_ROOT/qwen3_4b_${topic}_qkv.pt
  output=$RUN_ROOT/$topic
  if [[ ! -s "$trace" ]]; then
    echo "missing trace: $trace" >&2
    exit 2
  fi
  if [[ -s "$output/summary.json" ]]; then
    continue
  fi
  CUDA_VISIBLE_DEVICES=7 "$PYTHON" -u \
    src/analyze_qk_matched_rate_all_dims_20260727.py \
    --trace_path "$trace" \
    --output_dir "$output" \
    --label "qwen3_${topic}32k" \
    --device cuda \
    --sample_stride 32 \
    --calibration_steps 8 \
    --query_shrinkage 0.75 \
    --top_fraction 0.01 \
    --selected_fractions 0.01,0.04 \
    >"$LOG_ROOT/${topic}.log" 2>&1
done
touch "$RUN_ROOT/ALL_COMPLETE"
