#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/20260727_progressive_lowrate_ppl}"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

run_one() {
  local gpu="$1"
  local score_mode="$2"
  local topic="$3"
  local label="$4"
  local output="$OUTPUT_ROOT/$label"
  local log="$LOG_DIR/$label.log"
  if [[ -s "$output/summary.json" ]]; then
    echo "skip complete: $label"
    return
  fi
  echo "start gpu=$gpu label=$label"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
    "$ROOT/src/run_adaptive_mass_budget_ppl_20260715.py" \
    --model_name_or_path "$MODEL" \
    --output_dir "$output" \
    --topics "$topic" \
    --window_indices 0 \
    --history_tokens 32000 \
    --query_tokens 64 \
    --eval_tokens 64 \
    --window_stride_tokens 32512 \
    --mass_thresholds 0.95 \
    --mass_estimator qabs_sampled_tail \
    --budget_fractions 0.06 \
    --sample_fraction 0.0025 \
    --qabs_dim_count 8 \
    --candidate_fraction 0.06 \
    --qabs_score_mode "$score_mode" \
    --qabs_projection_dim 128 \
    --qabs_gqa_candidate_mode independent \
    --qabs_skip_candidate_rerank \
    --prefill_chunk_tokens 1024 \
    --dataset_cache_dir /home/fdong/ymluo/datasets/sklearn \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    >"$log" 2>&1
  echo "complete gpu=$gpu label=$label"
}

(
  run_one 0 pca_hierarchical_841prog30 sports prog30_f06_sports
  run_one 0 pca_hierarchical_autokey16 politics autokey16_f06_politics
  run_one 0 pca_hierarchical_autokey18 computer autokey18_f06_computer
) &
(
  run_one 1 pca_hierarchical_841prog30 medicine prog30_f06_medicine
  run_one 1 pca_hierarchical_autokey16 religion autokey16_f06_religion
  run_one 1 pca_hierarchical_autokey18 space autokey18_f06_space
) &
(
  run_one 2 pca_hierarchical_841prog30 computer prog30_f06_computer
  run_one 2 pca_hierarchical_autokey16 sports autokey16_f06_sports
) &
(
  run_one 3 pca_hierarchical_841prog30 space prog30_f06_space
  run_one 3 pca_hierarchical_autokey16 medicine autokey16_f06_medicine
) &
(
  run_one 4 pca_hierarchical_841prog30 politics prog30_f06_politics
  run_one 4 pca_hierarchical_autokey18 sports autokey18_f06_sports
) &
(
  run_one 5 pca_hierarchical_841prog30 religion prog30_f06_religion
  run_one 5 pca_hierarchical_autokey18 medicine autokey18_f06_medicine
) &
(
  run_one 6 pca_hierarchical_autokey16 computer autokey16_f06_computer
  run_one 6 pca_hierarchical_autokey18 politics autokey18_f06_politics
) &
(
  run_one 7 pca_hierarchical_autokey16 space autokey16_f06_space
  run_one 7 pca_hierarchical_autokey18 religion autokey18_f06_religion
) &

wait
echo "ALL_COMPLETE"
