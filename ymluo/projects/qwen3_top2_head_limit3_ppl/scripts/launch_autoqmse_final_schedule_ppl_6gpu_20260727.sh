#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/20260727_autoqmse_final_schedule_ppl}"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

run_full() {
  local gpu="$1"
  local topic="$2"
  local length="$3"
  local output="$OUTPUT_ROOT/full_l${length}_${topic}"
  local log="$LOG_DIR/full_l${length}_${topic}.log"
  if [[ -s "$output/summary.json" ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" "$ROOT/src/run_critical_position_budget_probe_20260715.py" \
    --model_name_or_path "$MODEL" \
    --output_dir "$output" \
    --topics "$topic" \
    --window_indices 0 \
    --history_tokens "$length" \
    --query_tokens 64 \
    --eval_tokens 64 \
    --window_stride_tokens "$((length + 512))" \
    --only_full \
    --prefill_chunk_tokens 1024 \
    --dataset_cache_dir /home/fdong/ymluo/datasets/sklearn \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    >"$log" 2>&1
}

run_sparse() {
  local gpu="$1"
  local topic="$2"
  local length="$3"
  local fraction="$4"
  local budget="$5"
  local mode="pca_hierarchical_autoqmse${budget}z"
  local label="autoqmse${budget}z_l${length}_${topic}"
  local output="$OUTPUT_ROOT/$label"
  local log="$LOG_DIR/$label.log"
  if [[ -s "$output/summary.json" ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" "$ROOT/src/run_adaptive_mass_budget_ppl_20260715.py" \
    --model_name_or_path "$MODEL" \
    --output_dir "$output" \
    --topics "$topic" \
    --window_indices 0 \
    --history_tokens "$length" \
    --query_tokens 64 \
    --eval_tokens 64 \
    --window_stride_tokens "$((length + 512))" \
    --mass_thresholds 0.95 \
    --mass_estimator qabs_sampled_tail \
    --budget_fractions "$fraction" \
    --sample_fraction 0.0025 \
    --qabs_dim_count 8 \
    --candidate_fraction "$fraction" \
    --qabs_use_cuda_kernels \
    --qabs_skip_candidate_rerank \
    --qabs_score_mode "$mode" \
    --qabs_projection_dim 128 \
    --qabs_gqa_candidate_mode independent \
    --prefill_chunk_tokens 1024 \
    --dataset_cache_dir /home/fdong/ymluo/datasets/sklearn \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    >"$log" 2>&1
}

topics=(sports medicine computer space politics religion)
lengths=(4096 8192 16384 32000)
fractions=(0.0625 0.06 0.06 0.04)

for gpu in {0..5}; do
  topic="${topics[$gpu]}"
  (
    for index in "${!lengths[@]}"; do
      length="${lengths[$index]}"
      fraction="${fractions[$index]}"
      run_full "$gpu" "$topic" "$length"
      run_sparse "$gpu" "$topic" "$length" "$fraction" 10
      run_sparse "$gpu" "$topic" "$length" "$fraction" 12
    done
  ) &
done

wait
echo "ALL_COMPLETE"
