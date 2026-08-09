#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
SCRIPT="$ROOT/src/analyze_automatic_spectral_rate_allocation_20260727.py"
OUTPUT="$ROOT/results/20260727_metadata_aware_qmse_frontier"
TRACE32="$ROOT/results/20260726_qk_matrix_spectrum_multimodel_32k/traces"
TRACE25="$ROOT/results/20260726_qwen25_7b_qk_matrix_spectrum_32k/traces"
mkdir -p "$OUTPUT/logs"

run_one() {
  local gpu="$1"
  local label="$2"
  local trace="$3"
  local output="$OUTPUT/$label"
  local log="$OUTPUT/logs/$label.log"
  if [[ -s "$output/summary.json" ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$SCRIPT" \
    --trace_path "$trace" \
    --output_dir "$output" \
    --label "$label" \
    --device cuda \
    --sample_stride 32 \
    --calibration_steps 8 \
    --bit_budgets 10 \
    --total_rate_budgets 12,13,14,15,16 \
    --selected_fractions 0.02,0.04,0.06 \
    --top_fraction 0.02 \
    --allow_zero_bits \
    >"$log" 2>&1
}

(
  run_one 6 qwen3_sports32k "$TRACE32/qwen3_4b_sports.pt"
  run_one 6 llama31_sports32k "$TRACE32/llama31_8b_sports.pt"
  run_one 6 qwen25_sports32k "$TRACE25/qwen25_7b_sports.pt"
) &

(
  run_one 7 qwen3_medicine32k "$TRACE32/qwen3_4b_medicine.pt"
  run_one 7 llama31_medicine32k "$TRACE32/llama31_8b_medicine.pt"
  run_one 7 qwen25_medicine32k "$TRACE25/qwen25_7b_medicine.pt"
) &

wait
echo "ALL_COMPLETE"
