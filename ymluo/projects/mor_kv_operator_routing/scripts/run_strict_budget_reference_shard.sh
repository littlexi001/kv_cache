#!/usr/bin/env bash
set -euo pipefail

: "${GPU:?set GPU to one idle GPU index}"
: "${QUERY_START:?set QUERY_START}"

BASE="${BASE:-/home/fdong/ymluo/projects/mor_kv_operator_routing}"
MODEL="${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}"
CORPUS="${CORPUS:-/home/fdong/ymluo/projects/parallel_block_retrieval/data/real_longbench_docqa_10m_holdout64_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$BASE/outputs/external_holdout64_v2_strict_budget3_v1}"
QUERY_COUNT="${QUERY_COUNT:-8}"
BUDGET_BLOCKS="${BUDGET_BLOCKS:-3}"
BLOCK_TOKENS="${BLOCK_TOKENS:-256}"
ANSWER_TOKENS="${ANSWER_TOKENS:-0}"
ACTIONS="${ACTIONS:-full,qk_top_blocks,mass_oracle_blocks,layer_shared_mass_oracle_blocks}"
OUTPUT_DIR="$OUTPUT_ROOT/shard_${QUERY_START}_${QUERY_COUNT}"
EXTRA_ARGS=()
if [[ -n "${POSTROPE_BASIS:-}" ]]; then
  EXTRA_ARGS+=(--postrope_basis "$POSTROPE_BASIS")
  EXTRA_ARGS+=(--proposal_multiplier "${PROPOSAL_MULTIPLIER:-4}")
fi

mkdir -p "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES="$GPU" /home/fdong/miniconda3/envs/moe/bin/python \
  "$BASE/src/evaluate_sparse_attention_reference_nll.py" \
  --model_name_or_path "$MODEL" \
  --corpus_dir "$CORPUS" \
  --output_dir "$OUTPUT_DIR" \
  --query_start "$QUERY_START" \
  --max_queries "$QUERY_COUNT" \
  --max_context_tokens 4096 \
  --block_tokens "$BLOCK_TOKENS" \
  --budget_blocks "$BUDGET_BLOCKS" \
  --sink_blocks 1 \
  --recent_blocks 1 \
  --answer_tokens "$ANSWER_TOKENS" \
  --actions "$ACTIONS" \
  "${EXTRA_ARGS[@]}"
