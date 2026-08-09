#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/20260727_fused_threshold_samplecount_frontier}"
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

run_one() {
  local gpu="$1"
  local samples="$2"
  local trace="$3"
  local label="$4"
  local output="$OUTPUT_ROOT/n${samples}_${label}"
  local log="$LOG_DIR/n${samples}_${label}.log"
  if [[ -s "$output/summary.json" ]]; then
    echo "skip complete: n${samples}_${label}"
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
    "$ROOT/src/analyze_fused_sampled_threshold_quality_20260727.py" \
    --trace_path "$trace" \
    --output_dir "$output" \
    --label "$label" \
    --sample_stride 32 \
    --calibration_samples "$samples" \
    --candidate_fractions 0.2 \
    --selected_fractions 0.02,0.04,0.06 \
    --top_fraction 0.02 \
    --capacity_multiplier 2 \
    --minimum_capacity_fraction 0.04 \
    --device cuda \
    >"$log" 2>&1
}

for gpu in {0..7}; do
  (
    run_one "$gpu" 512 "${traces[$gpu]}" "${labels[$gpu]}"
    run_one "$gpu" 1024 "${traces[$gpu]}" "${labels[$gpu]}"
  ) &
done

wait
echo "ALL_COMPLETE"
