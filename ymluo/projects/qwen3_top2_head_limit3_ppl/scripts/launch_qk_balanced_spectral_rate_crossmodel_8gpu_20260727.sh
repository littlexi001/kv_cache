#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
OUTPUT="${OUTPUT:-$ROOT/results/20260727_qk_balanced_spectral_rate_crossmodel}"
LOGS="$OUTPUT/logs"
mkdir -p "$LOGS"

run_trace() {
  local gpu="$1"
  local label="$2"
  local trace="$3"
  local output="$OUTPUT/$label"
  local log="$LOGS/$label.log"

  if [[ -s "$output/summary.json" ]]; then
    echo "SKIP ${label}"
    return
  fi

  echo "START ${label} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" -u "$ROOT/src/analyze_qk_balanced_spectral_rate_20260727.py" \
    --trace_path "$trace" \
    --output_dir "$output" \
    --label "$label" \
    --device cuda \
    --sample_stride 32 \
    --calibration_steps 8 \
    --total_rate_budget 15 \
    --query_shrinkage 0.5 \
    --selected_fractions 0.01,0.02,0.06 \
    --top_fraction 0.01 \
    >"$log" 2>&1
  echo "DONE ${label}"
}

TRACE32="$ROOT/results/20260726_qk_matrix_spectrum_multimodel_32k/traces"
TRACEQ25="$ROOT/results/20260726_qwen25_7b_qk_matrix_spectrum_32k/traces"
TRACE96="$ROOT/results/20260727_hierarchical_spectral_quantization_128k_traces/traces"

run_trace 0 qwen3_sports32k "$TRACE32/qwen3_4b_sports.pt" &
pid0=$!
run_trace 1 qwen3_medicine32k "$TRACE32/qwen3_4b_medicine.pt" &
pid1=$!
run_trace 2 llama31_sports32k "$TRACE32/llama31_8b_sports.pt" &
pid2=$!
run_trace 3 llama31_medicine32k "$TRACE32/llama31_8b_medicine.pt" &
pid3=$!
run_trace 4 qwen25_sports32k "$TRACEQ25/qwen25_7b_sports.pt" &
pid4=$!
run_trace 5 qwen25_medicine32k "$TRACEQ25/qwen25_7b_medicine.pt" &
pid5=$!
run_trace 6 qwen3_sports96k "$TRACE96/qwen3_4b_sports96k.pt" &
pid6=$!
run_trace 7 qwen3_medicine96k "$TRACE96/qwen3_4b_medicine96k.pt" &
pid7=$!

wait "$pid0"
wait "$pid1"
wait "$pid2"
wait "$pid3"
wait "$pid4"
wait "$pid5"
wait "$pid6"
wait "$pid7"
echo "ALL_COMPLETE"
