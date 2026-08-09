#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
OUTPUT="${OUTPUT:-$ROOT/results/20260727_qk_balanced_shrinkage_sensitivity}"
LOGS="$OUTPUT/logs"
TRACE32="$ROOT/results/20260726_qk_matrix_spectrum_multimodel_32k/traces"
TRACEQ25="$ROOT/results/20260726_qwen25_7b_qk_matrix_spectrum_32k/traces"
TRACE96="$ROOT/results/20260727_qk_balanced_96k_independent/traces"
SHRINKAGES=(0.10 0.25 0.50 0.75 0.90)

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$LOGS"
cd "$ROOT"

run_grid() {
  local gpu="$1"
  local label="$2"
  local trace="$3"
  local calibration_steps="$4"
  for shrinkage in "${SHRINKAGES[@]}"; do
    local tag="${shrinkage/./p}"
    local output="$OUTPUT/${label}_s${tag}"
    if [[ -s "$output/summary.json" ]]; then
      continue
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
      src/analyze_qk_balanced_spectral_rate_20260727.py \
      --trace_path "$trace" \
      --output_dir "$output" \
      --label "$label" \
      --device cuda \
      --sample_stride 32 \
      --calibration_steps "$calibration_steps" \
      --total_rate_budget 15 \
      --query_shrinkage "$shrinkage" \
      --selected_fractions 0.01 \
      --top_fraction 0.01 \
      >"$LOGS/${label}_s${tag}.log" 2>&1
  done
}

run_grid 0 qwen3_sports32k "$TRACE32/qwen3_4b_sports.pt" 8 &
pid0=$!
run_grid 1 qwen3_medicine32k "$TRACE32/qwen3_4b_medicine.pt" 8 &
pid1=$!
run_grid 2 llama31_sports32k "$TRACE32/llama31_8b_sports.pt" 8 &
pid2=$!
run_grid 3 llama31_medicine32k "$TRACE32/llama31_8b_medicine.pt" 8 &
pid3=$!
run_grid 4 qwen25_sports32k "$TRACEQ25/qwen25_7b_sports.pt" 8 &
pid4=$!
run_grid 5 qwen25_medicine32k "$TRACEQ25/qwen25_7b_medicine.pt" 8 &
pid5=$!
run_grid 6 qwen3_sports96k "$TRACE96/qwen3_4b_sports96k_32steps.pt" 8 &
pid6=$!
run_grid 7 qwen3_medicine96k "$TRACE96/qwen3_4b_medicine96k_32steps.pt" 8 &
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
