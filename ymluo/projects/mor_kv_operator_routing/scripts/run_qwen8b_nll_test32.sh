#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/home/fdong/ymluo/projects/mor_kv_operator_routing}"
MODEL="${MODEL:-/home/fdong/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
CORPUS="${CORPUS:-/home/fdong/ymluo/projects/parallel_block_retrieval/data/real_longbench_docqa_10m_holdout64_v2}"
BUNDLE="${BUNDLE:-$BASE/outputs/qwen8b_head_router_64q_2k_q90_v4/deployment_bundle.json}"
OUTPUT="${OUTPUT:-$BASE/outputs/qwen8b_sparse_reference_nll_test32_q90_v2}"

mkdir -p "$OUTPUT"
for gpu in 0 1 2 3 4 5 6; do
  start=$((gpu * 10))
  shard="$OUTPUT/shard$gpu"
  mkdir -p "$shard"
  CUDA_VISIBLE_DEVICES="$gpu" nohup \
    /home/fdong/miniconda3/envs/moe/bin/python \
    "$BASE/src/evaluate_sparse_attention_reference_nll.py" \
    --model_name_or_path "$MODEL" \
    --corpus_dir "$CORPUS" \
    --output_dir "$shard" \
    --query_start "$start" \
    --max_queries 10 \
    --max_context_tokens 2048 \
    --answer_tokens 1 \
    --block_tokens 256 \
    --budget_blocks 4 \
    --sink_blocks 1 \
    --recent_blocks 1 \
    --risk_threshold 0.05 \
    --actions full,learned_conformal \
    --router_bundle "$BUNDLE" \
    --router_test_only \
    >"$shard/run.log" 2>&1 < /dev/null &
done
