#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/20260727_hierarchical_mass_budget}"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

traces=(
  "$ROOT/results/20260726_qk_matrix_spectrum_multimodel_32k/traces/qwen3_4b_sports.pt"
  "$ROOT/results/20260726_qk_matrix_spectrum_multimodel_32k/traces/qwen3_4b_medicine.pt"
  "$ROOT/results/20260726_qk_matrix_spectrum_multimodel_32k/traces/llama31_8b_sports.pt"
  "$ROOT/results/20260726_qk_matrix_spectrum_multimodel_32k/traces/llama31_8b_medicine.pt"
  "$ROOT/results/20260726_qwen25_7b_qk_matrix_spectrum_32k/traces/qwen25_7b_sports.pt"
  "$ROOT/results/20260726_qwen25_7b_qk_matrix_spectrum_32k/traces/qwen25_7b_medicine.pt"
  "$ROOT/results/20260726_qk_matrix_spectrum_multimodel_32k/traces/qwen3_4b_sports.pt"
  "$ROOT/results/20260726_qk_matrix_spectrum_multimodel_32k/traces/qwen3_4b_medicine.pt"
)
labels=(
  qwen3_sports_fullbasis
  qwen3_medicine_fullbasis
  llama31_sports_fullbasis
  llama31_medicine_fullbasis
  qwen25_sports_fullbasis
  qwen25_medicine_fullbasis
  qwen3_sports_first2k
  qwen3_medicine_first2k
)
basis_tokens=(0 0 0 0 0 0 2048 2048)

for gpu in "${!labels[@]}"; do
  label="${labels[$gpu]}"
  output="$OUTPUT_ROOT/$label"
  log="$LOG_DIR/$label.log"
  if [[ -s "$output/summary.json" ]]; then
    echo "skip complete: $label"
    continue
  fi
  nohup env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
    "$ROOT/src/analyze_hierarchical_mass_budget_20260727.py" \
    --trace_path "${traces[$gpu]}" \
    --output_dir "$output" \
    --label "$label" \
    --basis_tokens "${basis_tokens[$gpu]}" \
    >"$log" 2>&1 &
  echo "launched gpu=$gpu pid=$! label=$label"
done
