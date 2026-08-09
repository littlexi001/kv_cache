#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/20260727_automatic_spectral_rate_allocation}"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

traces=(
  "$ROOT/results/20260726_qk_matrix_spectrum_multimodel_32k/traces/qwen3_4b_sports.pt"
  "$ROOT/results/20260726_qk_matrix_spectrum_multimodel_32k/traces/qwen3_4b_medicine.pt"
  "$ROOT/results/20260726_qk_matrix_spectrum_multimodel_32k/traces/llama31_8b_sports.pt"
  "$ROOT/results/20260726_qk_matrix_spectrum_multimodel_32k/traces/llama31_8b_medicine.pt"
  "$ROOT/results/20260726_qwen25_7b_qk_matrix_spectrum_32k/traces/qwen25_7b_sports.pt"
  "$ROOT/results/20260726_qwen25_7b_qk_matrix_spectrum_32k/traces/qwen25_7b_medicine.pt"
  "$ROOT/results/20260727_hierarchical_spectral_quantization_128k_traces/traces/qwen3_4b_sports96k.pt"
  "$ROOT/results/20260727_hierarchical_spectral_quantization_128k_traces/traces/qwen3_4b_medicine96k.pt"
)

labels=(
  qwen3_sports32k
  qwen3_medicine32k
  llama31_sports32k
  llama31_medicine32k
  qwen25_sports32k
  qwen25_medicine32k
  qwen3_sports96k
  qwen3_medicine96k
)

calibration_steps=(8 8 8 8 8 8 0 0)

for gpu in "${!labels[@]}"; do
  label="${labels[$gpu]}"
  output="$OUTPUT_ROOT/$label"
  log="$LOG_DIR/$label.log"
  if [[ -s "$output/summary.json" ]]; then
    echo "skip complete: $label"
    continue
  fi
  nohup env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
    "$ROOT/src/analyze_automatic_spectral_rate_allocation_20260727.py" \
    --trace_path "${traces[$gpu]}" \
    --output_dir "$output" \
    --label "$label" \
    --calibration_steps "${calibration_steps[$gpu]}" \
    --bit_budgets 21,26 \
    --selected_fractions 0.02,0.03,0.04,0.06 \
    >"$log" 2>&1 &
  echo "launched gpu=$gpu pid=$! label=$label log=$log"
done
