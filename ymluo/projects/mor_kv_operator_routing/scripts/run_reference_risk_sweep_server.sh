#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-6}"
BASE="${BASE:-/home/fdong/ymluo/projects/mor_kv_operator_routing}"
MODEL="${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}"
CORPUS="${CORPUS:-/home/fdong/ymluo/projects/parallel_block_retrieval/data/real_longbench_docqa_10m_holdout64_v1}"
BUNDLE="${BUNDLE:-$BASE/outputs/head_distortion_router_64_gqa_q90_v2/deployment_bundle.json}"

for threshold in 0.01 0.02 0.03; do
  tag="$(printf '%03d' "$(awk "BEGIN { print 100 * $threshold }")")"
  output="$BASE/outputs/sparse_reference_nll_sweep_t${tag}_v2"
  mkdir -p "$output"
  for shard in 0 1 2 3 4 5 6 7; do
    shard_output="$output/shard$shard"
    if [[ -s "$shard_output/reference_nll_rows.csv" ]]; then
      continue
    fi
    start=$((shard * 8))
    CUDA_VISIBLE_DEVICES="$GPU" /home/fdong/miniconda3/envs/moe/bin/python \
      "$BASE/src/evaluate_sparse_attention_reference_nll.py" \
      --model_name_or_path "$MODEL" \
      --corpus_dir "$CORPUS" \
      --output_dir "$shard_output" \
      --query_start "$start" \
      --max_queries 8 \
      --max_context_tokens 4096 \
      --risk_threshold "$threshold" \
      --router_error_threshold "$threshold" \
      --actions full,learned_conformal,risk_oracle \
      --router_bundle "$BUNDLE" \
      --router_test_only \
      > "$output/shard$shard.log" 2>&1
  done
done
