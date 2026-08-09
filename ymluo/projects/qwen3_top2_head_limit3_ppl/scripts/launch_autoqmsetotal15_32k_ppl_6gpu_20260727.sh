#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
OUTPUT="${OUTPUT:-$ROOT/results/20260727_autoqmsetotal15_32k_ppl}"
LOGS="$OUTPUT/logs"
mkdir -p "$LOGS"

topics=(sports medicine computer space politics religion)
for gpu in {0..5}; do
  topic="${topics[$gpu]}"
  (
    output="$OUTPUT/autoqmsetotal15z_l32000_${topic}"
    if [[ ! -s "$output/summary.json" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PYTHON" "$ROOT/src/run_adaptive_mass_budget_ppl_20260715.py" \
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
        --budget_fractions 0.04 \
        --sample_fraction 0.0025 \
        --qabs_dim_count 8 \
        --candidate_fraction 0.04 \
        --qabs_use_cuda_kernels \
        --qabs_skip_candidate_rerank \
        --qabs_score_mode pca_hierarchical_autoqmsetotal15z \
        --qabs_projection_dim 128 \
        --qabs_gqa_candidate_mode independent \
        --prefill_chunk_tokens 1024 \
        --dataset_cache_dir /home/fdong/ymluo/datasets/sklearn \
        --dtype float16 \
        --device cuda \
        --device_map auto \
        >"$LOGS/autoqmsetotal15z_l32000_${topic}.log" 2>&1
    fi
  ) &
done

wait
echo "ALL_COMPLETE"
