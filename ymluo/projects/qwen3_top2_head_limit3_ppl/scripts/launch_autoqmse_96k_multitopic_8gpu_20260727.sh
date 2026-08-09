#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
OUTPUT="${OUTPUT:-$ROOT/results/20260727_autoqmse_96k_smoke}"
LOGS="$OUTPUT/logs"
mkdir -p "$LOGS"

run_full() {
  local devices="$1"
  local topic="$2"
  local output="$OUTPUT/full_l96000_${topic}"
  if [[ -s "$output/summary.json" ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES="$devices" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" "$ROOT/src/run_critical_position_budget_probe_20260715.py" \
    --model_name_or_path "$MODEL" \
    --output_dir "$output" \
    --topics "$topic" \
    --window_indices 0 \
    --history_tokens 96000 \
    --query_tokens 64 \
    --eval_tokens 64 \
    --window_stride_tokens 96512 \
    --only_full \
    --prefill_chunk_tokens 512 \
    --dataset_cache_dir /home/fdong/ymluo/datasets/sklearn \
    --dtype float16 \
    --device cuda \
    --device_map balanced \
    >"$LOGS/full_l96000_${topic}.log" 2>&1
}

run_sparse() {
  local devices="$1"
  local topic="$2"
  local label="$3"
  local score_mode="$4"
  local output="$OUTPUT/${label}_l96000_${topic}"
  if [[ -s "$output/summary.json" ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES="$devices" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" "$ROOT/src/run_adaptive_mass_budget_ppl_20260715.py" \
    --model_name_or_path "$MODEL" \
    --output_dir "$output" \
    --topics "$topic" \
    --window_indices 0 \
    --history_tokens 96000 \
    --query_tokens 64 \
    --eval_tokens 64 \
    --window_stride_tokens 96512 \
    --mass_thresholds 0.95 \
    --mass_estimator qabs_sampled_tail \
    --budget_fractions 0.013333333333333334 \
    --sample_fraction 0.0025 \
    --qabs_dim_count 8 \
    --candidate_fraction 0.013333333333333334 \
    --qabs_use_cuda_kernels \
    --qabs_skip_candidate_rerank \
    --qabs_score_mode "$score_mode" \
    --qabs_projection_dim 128 \
    --qabs_gqa_candidate_mode independent \
    --prefill_chunk_tokens 512 \
    --dataset_cache_dir /home/fdong/ymluo/datasets/sklearn \
    --dtype float16 \
    --device cuda \
    --device_map balanced \
    --chunked_gqa_sdpa \
    >"$LOGS/${label}_l96000_${topic}.log" 2>&1
}

run_topic() {
  local devices="$1"
  local topic="$2"
  run_full "$devices" "$topic"
  run_sparse \
    "$devices" "$topic" autoqmse12z \
    pca_hierarchical_autoqmse12z
  run_sparse \
    "$devices" "$topic" autoqmsetotal15z \
    pca_hierarchical_autoqmsetotal15z
}

(
  run_topic 0,1 sports
  run_topic 0,1 politics
) &
(
  run_topic 2,3 medicine
  run_topic 2,3 religion
) &
run_topic 4,5 computer &
run_topic 6,7 space &

wait
echo "ALL_COMPLETE"
