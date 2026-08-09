#!/usr/bin/env bash
set -euo pipefail

: "${GPU:?set GPU}"
: "${SHARD:?set SHARD in [0,7]}"
BASE="${BASE:-/home/fdong/ymluo/projects/mor_kv_operator_routing}"
MODEL="${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}"
CORPUS="${CORPUS:-/home/fdong/ymluo/projects/parallel_block_retrieval/data/real_longbench_docqa_10m_holdout64_v2}"
BUNDLE="${BUNDLE:-$BASE/outputs/head_distortion_router_64_gqa_q90_v2/deployment_bundle.json}"
ROOT="${ROOT:-$BASE/outputs/external_holdout64_v2_layer_alloc_v1}"
start=$((SHARD * 8))

for policy in uniform_t001 uniform_t002 allocated_t002; do
  case "$policy" in
    uniform_t001)
      threshold=0.01
      layers=all
      ;;
    uniform_t002)
      threshold=0.02
      layers=all
      ;;
    allocated_t002)
      threshold=0.02
      layers=0-13,21-27
      ;;
  esac
  output="$ROOT/$policy/shard$SHARD"
  mkdir -p "$ROOT/$policy"
  if [[ -s "$output/reference_nll_rows.csv" ]]; then
    continue
  fi
  CUDA_VISIBLE_DEVICES="$GPU" /home/fdong/miniconda3/envs/moe/bin/python \
    "$BASE/src/evaluate_sparse_attention_reference_nll.py" \
    --model_name_or_path "$MODEL" \
    --corpus_dir "$CORPUS" \
    --output_dir "$output" \
    --query_start "$start" \
    --max_queries 8 \
    --max_context_tokens 4096 \
    --risk_threshold "$threshold" \
    --router_error_threshold "$threshold" \
    --actions full,learned_conformal \
    --router_bundle "$BUNDLE" \
    --sparse_layers "$layers" \
    > "$ROOT/$policy/shard$SHARD.log" 2>&1
done
