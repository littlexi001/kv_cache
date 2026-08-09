#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/20260727_proxy_and_zerobit_ppl}"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

run_one() {
  local gpu="$1"
  local score_mode="$2"
  local topic="$3"
  local label="$4"
  local budget="$5"
  local output="$OUTPUT_ROOT/$label"
  local log="$LOG_DIR/$label.log"
  if [[ -s "$output/summary.json" ]]; then
    echo "skip complete: $label"
    return
  fi
  local rerank_args=()
  if [[ "$score_mode" != "pca_hierarchical_841sample20_proxy" ]]; then
    rerank_args+=(--qabs_skip_candidate_rerank)
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
    --budget_fractions "$budget" \
    --sample_fraction 0.0025 \
    --qabs_dim_count 8 \
    --candidate_fraction "$budget" \
    --qabs_use_cuda_kernels \
    --qabs_score_mode "$score_mode" \
    --qabs_projection_dim 128 \
    --qabs_gqa_candidate_mode independent \
    "${rerank_args[@]}" \
    --prefill_chunk_tokens 1024 \
    --dataset_cache_dir /home/fdong/ymluo/datasets/sklearn \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    >"$log" 2>&1
  echo "complete gpu=$gpu label=$label"
}

topics=(sports medicine computer space politics religion)
for gpu in {0..5}; do
  topic="${topics[$gpu]}"
  (
    run_one "$gpu" pca_hierarchical_841sample20_proxy "$topic" \
      "sample20_proxy_${topic}" 0.12
    run_one "$gpu" pca_hierarchical_autokey14z "$topic" \
      "autokey14z_${topic}" 0.06
    run_one "$gpu" pca_hierarchical_autokey16z "$topic" \
      "autokey16z_${topic}" 0.06
  ) &
done

(
  run_one 6 pca_hierarchical_autokey18z sports autokey18z_sports 0.06
  run_one 6 pca_hierarchical_autokey18z medicine autokey18z_medicine 0.06
  run_one 6 pca_hierarchical_autokey18z computer autokey18z_computer 0.06
) &
(
  run_one 7 pca_hierarchical_autokey18z space autokey18z_space 0.06
  run_one 7 pca_hierarchical_autokey18z politics autokey18z_politics 0.06
  run_one 7 pca_hierarchical_autokey18z religion autokey18z_religion 0.06
) &

wait
echo "ALL_COMPLETE"
