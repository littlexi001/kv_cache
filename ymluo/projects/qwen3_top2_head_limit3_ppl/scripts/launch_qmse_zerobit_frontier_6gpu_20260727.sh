#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/20260727_qmse_zerobit_frontier_32k}"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

traces=(
  "$ROOT/results/20260726_qk_matrix_spectrum_multimodel_32k/traces/qwen3_4b_sports.pt"
  "$ROOT/results/20260726_qk_matrix_spectrum_multimodel_32k/traces/qwen3_4b_medicine.pt"
  "$ROOT/results/20260726_qk_matrix_spectrum_multimodel_32k/traces/llama31_8b_sports.pt"
  "$ROOT/results/20260726_qk_matrix_spectrum_multimodel_32k/traces/llama31_8b_medicine.pt"
  "$ROOT/results/20260726_qwen25_7b_qk_matrix_spectrum_32k/traces/qwen25_7b_sports.pt"
  "$ROOT/results/20260726_qwen25_7b_qk_matrix_spectrum_32k/traces/qwen25_7b_medicine.pt"
)
labels=(
  qwen3_sports32k
  qwen3_medicine32k
  llama31_sports32k
  llama31_medicine32k
  qwen25_sports32k
  qwen25_medicine32k
)

for gpu in {0..5}; do
  output="$OUTPUT_ROOT/${labels[$gpu]}"
  log="$LOG_DIR/${labels[$gpu]}.log"
  if [[ -s "$output/summary.json" ]]; then
    echo "skip complete: ${labels[$gpu]}"
    continue
  fi
  nohup env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
    "$ROOT/src/analyze_automatic_spectral_rate_allocation_20260727.py" \
    --trace_path "${traces[$gpu]}" \
    --output_dir "$output" \
    --label "${labels[$gpu]}" \
    --sample_stride 32 \
    --basis_tokens 0 \
    --calibration_steps 8 \
    --bit_budgets 10,12,14,16,18 \
    --selected_fractions 0.02,0.04,0.06 \
    --top_fraction 0.02 \
    --allow_zero_bits \
    --device cuda \
    >"$log" 2>&1 &
  echo "launched gpu=$gpu pid=$! label=${labels[$gpu]}"
done
