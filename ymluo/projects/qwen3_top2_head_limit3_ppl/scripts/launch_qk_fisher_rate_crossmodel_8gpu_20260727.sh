#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
OUTPUT="${OUTPUT:-$ROOT/results/20260727_qk_fisher_rate_crossmodel}"
TRACE32="$ROOT/results/20260726_qk_matrix_spectrum_multimodel_32k/traces"
TRACEQ25="$ROOT/results/20260726_qwen25_7b_qk_matrix_spectrum_32k/traces"
TRACE96="$ROOT/results/20260727_qk_balanced_96k_independent/traces"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTPUT/logs"
cd "$ROOT"

run_case() {
  local gpu="$1"
  local label="$2"
  local trace="$3"
  local output="$OUTPUT/$label"
  if [[ -s "$output/summary.json" ]]; then
    echo "SKIP $label"
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
    src/analyze_qk_balanced_spectral_rate_20260727.py \
    --trace_path "$trace" \
    --output_dir "$output" \
    --label "$label" \
    --device cuda \
    --sample_stride 32 \
    --calibration_steps 8 \
    --total_rate_budget 15 \
    --query_shrinkage 0.75 \
    --selected_fractions 0.01 \
    --top_fraction 0.01 \
    >"$OUTPUT/logs/$label.log" 2>&1
}

run_case 0 qwen3_sports32k "$TRACE32/qwen3_4b_sports.pt" &
pid0=$!
run_case 1 qwen3_medicine32k "$TRACE32/qwen3_4b_medicine.pt" &
pid1=$!
run_case 2 llama31_sports32k "$TRACE32/llama31_8b_sports.pt" &
pid2=$!
run_case 3 llama31_medicine32k "$TRACE32/llama31_8b_medicine.pt" &
pid3=$!
run_case 4 qwen25_sports32k "$TRACEQ25/qwen25_7b_sports.pt" &
pid4=$!
run_case 5 qwen25_medicine32k "$TRACEQ25/qwen25_7b_medicine.pt" &
pid5=$!
run_case 6 qwen3_sports96k "$TRACE96/qwen3_4b_sports96k_32steps.pt" &
pid6=$!
run_case 7 qwen3_medicine96k "$TRACE96/qwen3_4b_medicine96k_32steps.pt" &
pid7=$!

wait "$pid0"
wait "$pid1"
wait "$pid2"
wait "$pid3"
wait "$pid4"
wait "$pid5"
wait "$pid6"
wait "$pid7"
touch "$OUTPUT/ALL_COMPLETE"
echo "ALL_COMPLETE"
