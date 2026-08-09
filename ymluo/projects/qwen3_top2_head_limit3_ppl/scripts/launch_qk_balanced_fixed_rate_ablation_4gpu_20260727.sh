#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
OUTPUT="${OUTPUT:-$ROOT/results/20260727_qk_balanced_fixed_rate_ablation}"
LOGS="$OUTPUT/logs"
TRACE32="$ROOT/results/20260726_qk_matrix_spectrum_multimodel_32k/traces"
TRACEQ25="$ROOT/results/20260726_qwen25_7b_qk_matrix_spectrum_32k/traces"
FIXED="444=4-4-4-0-0-0-0-0,822=8-2-2-0-0-0-0-0,840=8-4-0-0-0-0-0-0,844=8-4-4-0-0-0-0-0"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$LOGS"
cd "$ROOT"

run_trace() {
  local gpu="$1"
  local label="$2"
  local trace="$3"
  local output="$OUTPUT/$label"
  if [[ -s "$output/summary.json" ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
    src/analyze_qk_balanced_spectral_rate_20260727.py \
    --trace_path "$trace" \
    --output_dir "$output" \
    --label "$label" \
    --device cuda \
    --sample_stride 32 \
    --calibration_steps 8 \
    --total_rate_budget 15 \
    --query_shrinkage 0.5 \
    --selected_fractions 0.01 \
    --top_fraction 0.01 \
    --qk_fixed_allocations "$FIXED" \
    >"$LOGS/$label.log" 2>&1
}

(
  run_trace 4 qwen3_sports32k "$TRACE32/qwen3_4b_sports.pt"
  run_trace 4 qwen3_medicine32k "$TRACE32/qwen3_4b_medicine.pt"
) &
pid0=$!
(
  run_trace 5 llama31_sports32k "$TRACE32/llama31_8b_sports.pt"
  run_trace 5 llama31_medicine32k "$TRACE32/llama31_8b_medicine.pt"
) &
pid1=$!
run_trace 6 qwen25_sports32k "$TRACEQ25/qwen25_7b_sports.pt" &
pid2=$!
run_trace 7 qwen25_medicine32k "$TRACEQ25/qwen25_7b_medicine.pt" &
pid3=$!

wait "$pid0"
wait "$pid1"
wait "$pid2"
wait "$pid3"
touch "$OUTPUT/ALL_COMPLETE"
echo "ALL_COMPLETE"
