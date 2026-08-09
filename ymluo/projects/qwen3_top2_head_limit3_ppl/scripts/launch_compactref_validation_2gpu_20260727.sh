#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/20260727_compactref_validation}"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

run_one() {
  local gpu="$1"
  local topic="$2"
  local length="$3"
  local eval_tokens="$4"
  local fraction="$5"
  local label="$6"
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
    --eval_tokens "$eval_tokens" \
    --window_stride_tokens "$((length + 512))" \
    --mass_thresholds 0.95 \
    --mass_estimator qabs_sampled_tail \
    --budget_fractions "$fraction" \
    --sample_fraction 0.0025 \
    --qabs_dim_count 8 \
    --candidate_fraction "$fraction" \
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

topics=(sports medicine)
for gpu in 6 7; do
  topic="${topics[$((gpu - 6))]}"
  (
    run_one "$gpu" "$topic" 32000 64 0.06 \
      "compact32k_${topic}"
    run_one "$gpu" "$topic" 96000 32 0.02 \
      "compact96k_${topic}"
  ) &
done

wait
echo "ALL_COMPLETE"
