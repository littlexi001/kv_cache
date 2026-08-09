#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/20260727_autoqmse_zerobit_ppl}"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

run_one() {
  local gpu="$1"
  local mode="$2"
  local topic="$3"
  local label="$4"
  local output="$OUTPUT_ROOT/$label"
  local log="$LOG_DIR/$label.log"
  if [[ -s "$output/summary.json" ]]; then
    return
  fi
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
for gpu in {0..5}; do
  topic="${topics[$gpu]}"
  (
    run_one "$gpu" pca_hierarchical_autoqmse14z "$topic" \
      "autoqmse14z_${topic}"
    run_one "$gpu" pca_hierarchical_autoqmse16z "$topic" \
      "autoqmse16z_${topic}"
  ) &
done

wait
echo "ALL_COMPLETE"
