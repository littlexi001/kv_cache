#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/20260727_fused_threshold_quality}"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

run_one() {
  local gpu="$1"
  local trace="$2"
  local label="$3"
  local output="$OUTPUT_ROOT/$label"
  local log="$LOG_DIR/$label.log"
  if [[ -s "$output/summary.json" ]]; then
    echo "skip complete: $label"
    return
  fi
  echo "start gpu=$gpu label=$label"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
    "$ROOT/src/analyze_fused_sampled_threshold_quality_20260727.py" \
    --trace_path "$ROOT/$trace" \
    --output_dir "$output" \
    --label "$label" \
    --candidate_fractions 0.2,0.3,0.4 \
    --selected_fractions 0.02,0.04,0.06 \
    --calibration_samples 256 \
    --device cuda \
    >"$log" 2>&1
  echo "complete gpu=$gpu label=$label"
}

run_one 0 \
  results/20260726_qk_matrix_spectrum_multimodel_32k/traces/qwen3_4b_sports.pt \
  qwen3_sports32k &
run_one 1 \
  results/20260726_qk_matrix_spectrum_multimodel_32k/traces/qwen3_4b_medicine.pt \
  qwen3_medicine32k &
run_one 2 \
  results/20260726_qk_matrix_spectrum_multimodel_32k/traces/llama31_8b_sports.pt \
  llama31_sports32k &
run_one 3 \
  results/20260726_qk_matrix_spectrum_multimodel_32k/traces/llama31_8b_medicine.pt \
  llama31_medicine32k &
run_one 4 \
  results/20260726_qwen25_7b_qk_matrix_spectrum_32k/traces/qwen25_7b_sports.pt \
  qwen25_sports32k &
run_one 5 \
  results/20260726_qwen25_7b_qk_matrix_spectrum_32k/traces/qwen25_7b_medicine.pt \
  qwen25_medicine32k &
run_one 6 \
  results/20260727_hierarchical_spectral_quantization_128k_traces/traces/qwen3_4b_sports96k.pt \
  qwen3_sports96k &
run_one 7 \
  results/20260727_hierarchical_spectral_quantization_128k_traces/traces/qwen3_4b_medicine96k.pt \
  qwen3_medicine96k &

wait
echo "ALL_COMPLETE"
