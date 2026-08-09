#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/20260727_spectral_tail_countsketch}"
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
  "qwen3_sports32k"
  "qwen3_medicine32k"
  "llama31_sports32k"
  "llama31_medicine32k"
  "qwen25_sports32k"
  "qwen25_medicine32k"
  "qwen3_sports96k"
  "qwen3_medicine96k"
)

for gpu in "${!traces[@]}"; do
  trace="${traces[$gpu]}"
  label="${labels[$gpu]}"
  output="$OUTPUT_ROOT/$label"
  log="$LOG_DIR/$label.log"
  if [[ -s "$output/summary.json" ]]; then
    echo "skip complete: $label"
    continue
  fi

  extra_args=()
  if (( gpu >= 6 )); then
    extra_args=(
      --bucket_counts 16,32
      --sketch_bits 2,4
      --seeds 20260727
    )
  fi

  nohup env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
    "$ROOT/src/analyze_spectral_tail_countsketch_20260727.py" \
    --trace_path "$trace" \
    --output_dir "$output" \
    --label "$label" \
    "${extra_args[@]}" \
    >"$log" 2>&1 &
  echo "launched gpu=$gpu pid=$! label=$label log=$log"
done
