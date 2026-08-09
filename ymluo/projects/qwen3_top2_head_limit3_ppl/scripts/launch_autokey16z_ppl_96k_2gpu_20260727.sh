#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/20260727_autokey16z_ppl_96k}"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

run_full() {
  local gpu="$1"
  local topic="$2"
  local output="$OUTPUT_ROOT/full_${topic}"
  local log="$LOG_DIR/full_${topic}.log"
  if [[ -s "$output/summary.json" ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" \
    "$ROOT/src/run_adaptive_mass_budget_ppl_20260715.py" \
    --model_name_or_path "$MODEL" \
    --output_dir "$output" \
    --topics "$topic" \
    --window_indices 0 \
    --history_tokens 96000 \
    --query_tokens 64 \
    --eval_tokens 32 \
    --window_stride_tokens 96512 \
    --mass_thresholds 1.0 \
    --mass_estimator exact \
    --budget_fractions 1.0 \
    --chunked_gqa_sdpa \
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
  local output="$OUTPUT_ROOT/autokey16z_${topic}"
  local log="$LOG_DIR/autokey16z_${topic}.log"
  if [[ -s "$output/summary.json" ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" \
    "$ROOT/src/run_adaptive_mass_budget_ppl_20260715.py" \
    --model_name_or_path "$MODEL" \
    --output_dir "$output" \
    --topics "$topic" \
    --window_indices 0 \
    --history_tokens 96000 \
    --query_tokens 64 \
    --eval_tokens 32 \
    --window_stride_tokens 96512 \
    --mass_thresholds 0.95 \
    --mass_estimator qabs_sampled_tail \
    --budget_fractions 0.02 \
    --sample_fraction 0.0025 \
    --qabs_dim_count 8 \
    --candidate_fraction 0.02 \
    --qabs_use_cuda_kernels \
    --qabs_skip_candidate_rerank \
    --qabs_score_mode pca_hierarchical_autokey16z \
    --qabs_projection_dim 128 \
    --qabs_gqa_candidate_mode independent \
    --chunked_gqa_sdpa \
    --prefill_chunk_tokens 1024 \
    --dataset_cache_dir /home/fdong/ymluo/datasets/sklearn \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    >"$log" 2>&1
}

(
  run_full 6 sports
  run_sparse 6 sports
) &
(
  run_full 7 medicine
  run_sparse 7 medicine
) &

wait
echo "ALL_COMPLETE"
