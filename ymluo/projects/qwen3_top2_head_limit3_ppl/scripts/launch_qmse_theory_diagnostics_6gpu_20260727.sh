#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
SCRIPT="$ROOT/src/analyze_qmse_theory_diagnostics_20260727.py"
OUTPUT="${OUTPUT:-$ROOT/results/20260727_qmse_theory_diagnostics}"
TRACE32="$ROOT/results/20260726_qk_matrix_spectrum_multimodel_32k/traces"
TRACE25="$ROOT/results/20260726_qwen25_7b_qk_matrix_spectrum_32k/traces"
GPU_OFFSET="${GPU_OFFSET:-0}"
mkdir -p "$OUTPUT/logs"

labels=(
  qwen3_sports32k
  qwen3_medicine32k
  llama31_sports32k
  llama31_medicine32k
  qwen25_sports32k
  qwen25_medicine32k
)
traces=(
  "$TRACE32/qwen3_4b_sports.pt"
  "$TRACE32/qwen3_4b_medicine.pt"
  "$TRACE32/llama31_8b_sports.pt"
  "$TRACE32/llama31_8b_medicine.pt"
  "$TRACE25/qwen25_7b_sports.pt"
  "$TRACE25/qwen25_7b_medicine.pt"
)

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
    --selected_fraction 0.06 \
    --top_fraction 0.02 \
    >"$log" 2>&1
}

for index in "${!labels[@]}"; do
  gpu=$((GPU_OFFSET + index))
  run_one "$gpu" "${labels[$index]}" "${traces[$index]}" &
done

wait
echo "ALL_COMPLETE"
