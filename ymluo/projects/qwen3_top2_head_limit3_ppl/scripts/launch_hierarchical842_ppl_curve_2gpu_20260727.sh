#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/20260727_hierarchical842_ppl_curve}"
FRACTION="${FRACTION:-0.06}"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

topics=(sports medicine)
for gpu in 6 7; do
  topic="${topics[$((gpu - 6))]}"
  label="${topic}_f${FRACTION/./p}"
  output="$OUTPUT_ROOT/$label"
  log="$LOG_DIR/$label.log"
  if [[ -s "$output/summary.json" ]]; then
    echo "skip complete: $label"
    continue
  fi
  nohup env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
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
    --budget_fractions "$FRACTION" \
    --mass_estimator qabs_sampled_tail \
    --sample_fraction 0.0025 \
    --qabs_dim_count 8 \
    --candidate_fraction "$FRACTION" \
    --qabs_skip_candidate_rerank \
    --qabs_score_mode pca_hierarchical_842 \
    --qabs_projection_dim 128 \
    --qabs_gqa_candidate_mode independent \
    --prefill_chunk_tokens 1024 \
    --dataset_cache_dir /home/fdong/ymluo/datasets/sklearn \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    >"$log" 2>&1 &
  echo "launched gpu=$gpu pid=$! label=$label log=$log"
done
