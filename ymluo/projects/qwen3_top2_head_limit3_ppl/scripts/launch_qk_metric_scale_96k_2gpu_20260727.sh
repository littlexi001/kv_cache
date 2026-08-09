#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
OUTPUT="${OUTPUT:-$ROOT/results/20260727_qk_metric_scale_96k}"
TRACE_ROOT="$ROOT/results/20260727_qk_balanced_96k_independent/traces"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTPUT/logs"
cd "$ROOT"

run_case() {
  local gpu="$1"
  local label="$2"
  local trace="$3"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
    src/analyze_qk_balanced_spectral_rate_20260727.py \
    --trace_path "$trace" \
    --output_dir "$OUTPUT/$label" \
    --label "$label" \
    --device cuda \
    --sample_stride 32 \
    --calibration_steps 8 \
    --total_rate_budget 15 \
    --query_shrinkage 0.75 \
    --selected_fractions 0.01 \
    --top_fraction 0.01 \
    >"$OUTPUT/logs/$label.log" 2>&1
}

run_case 6 qwen3_sports96k \
  "$TRACE_ROOT/qwen3_4b_sports96k_32steps.pt" &
pid6=$!
run_case 7 qwen3_medicine96k \
  "$TRACE_ROOT/qwen3_4b_medicine96k_32steps.pt" &
pid7=$!

wait "$pid6"
wait "$pid7"
touch "$OUTPUT/ALL_COMPLETE"
