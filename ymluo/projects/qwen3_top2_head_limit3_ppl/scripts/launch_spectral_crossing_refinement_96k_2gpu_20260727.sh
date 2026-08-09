#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
TRACE_ROOT="${TRACE_ROOT:-$ROOT/results/20260727_hierarchical_spectral_quantization_128k_traces/traces}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/20260727_spectral_crossing_refinement_96k}"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

traces=(
  "$TRACE_ROOT/qwen3_4b_sports96k.pt"
  "$TRACE_ROOT/qwen3_4b_medicine96k.pt"
)
labels=(qwen3_sports96k qwen3_medicine96k)

for local_gpu in 0 1; do
  gpu=$((local_gpu + 6))
  label="${labels[$local_gpu]}"
  output="$OUTPUT_ROOT/$label"
  log="$LOG_DIR/$label.log"
  if [[ -s "$output/summary.json" ]]; then
    echo "skip complete: $label"
    continue
  fi
  nohup env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
    "$ROOT/src/analyze_spectral_crossing_refinement_20260727.py" \
    --trace_path "${traces[$local_gpu]}" \
    --output_dir "$output" \
    --label "$label" \
    --sample_stride 32 \
    --calibration_samples 256 \
    --alphas 0.05,0.1,0.2 \
    --top_fraction 0.02 \
    --device cuda \
    >"$log" 2>&1 &
  echo "launched gpu=$gpu pid=$! label=$label"
done
