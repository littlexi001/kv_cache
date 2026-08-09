#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
OUTPUT_ROOT="$PROJECT/results/20260717_pca64_overfetch_frontier_32k"
LOG_ROOT="$PROJECT/logs"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$PROJECT"
export PATH=/home/fdong/miniconda3/envs/moe/bin:$PATH

pids=()
gpu=1
for topic in sports medicine; do
  for candidate in 0.04 0.06 0.08; do
    tag="${topic}_c${candidate/0./}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
      src/run_adaptive_mass_budget_ppl_20260715.py \
      --model_name_or_path "$MODEL" \
      --output_dir "$OUTPUT_ROOT/$tag" \
      --topics "$topic" \
      --window_indices 0 \
      --history_tokens 32000 \
      --query_tokens 256 \
      --eval_tokens 256 \
      --mass_thresholds 0.000001 \
      --budget_fractions 0.02 \
      --mass_estimator qabs_sampled_tail \
      --sample_fraction 0.0025 \
      --qabs_dim_count 8 \
      --candidate_fraction "$candidate" \
      --qabs_use_cuda_kernels \
      --qabs_score_mode pca_int4 \
      --qabs_projection_dim 64 \
      --prefill_chunk_tokens 2048 \
      --dataset_cache_dir /home/fdong/ymluo/datasets/sklearn \
      --seed 20260714 \
      --dtype float16 \
      --device cuda \
      --device_map auto \
      > "$LOG_ROOT/pca64_overfetch_frontier_${tag}.log" 2>&1 &
    pids+=("$!")
    gpu=$((gpu + 1))
  done
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
