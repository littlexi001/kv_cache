#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/fdong/ymluo/projects/qwen3_per_head_hierarchical_memory}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}"
TEXT="${TEXT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/data/war_and_peace_pg2600.txt}"
ATLAS="${ATLAS:-/home/fdong/ymluo/projects/qwen3_head_function_atlas/outputs/head_function_atlas.csv}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/outputs/sparse_ppl_war16k_${STAMP}}"

cd "$PROJECT_DIR"
mkdir -p "$OUT_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

"$PY" -u src/run_sparse_memory_ppl.py \
  --model_name_or_path "$MODEL" \
  --text_path "$TEXT" \
  --atlas_csv "$ATLAS" \
  --output_dir "$OUT_DIR" \
  --prefill_tokens "${PREFILL_TOKENS:-16384}" \
  --train_queries "${TRAIN_QUERIES:-64}" \
  --test_queries "${TEST_QUERIES:-64}" \
  --chunk_size "${CHUNK_SIZE:-8}" \
  --prefill_chunk_size "${PREFILL_CHUNK_SIZE:-128}" \
  --l0_capacity "${L0_CAPACITY:-500}" \
  --l0_recent_tokens "${L0_RECENT_TOKENS:-448}" \
  --promotion_policy "${PROMOTION_POLICY:-uniform}" \
  --medium_promotion_slots "${MEDIUM_PROMOTION_SLOTS:-20}" \
  --promotion_categories "${PROMOTION_CATEGORIES:-semantic_evidence,lexical_copy,structural_anchor}" \
  --l1_capacity "${L1_CAPACITY:-4096}" \
  --l2_block_size "${L2_BLOCK_SIZE:-64}" \
  --l2_block_budget "${L2_BLOCK_BUDGET:-64}" \
  --sink_tokens "${SINK_TOKENS:-4}" \
  --dtype "${DTYPE:-float16}" \
  --device cuda \
  --device_map auto \
  --attn_implementation eager \
  --policies "${POLICIES:-full_attention,sink_recent_500,flat_function_500,hier_function_500}" \
  --log_every "${LOG_EVERY:-1}" \
  2>&1 | tee "$OUT_DIR/run.log"

echo "[per-head-hierarchical-memory:ppl] done: $OUT_DIR"
