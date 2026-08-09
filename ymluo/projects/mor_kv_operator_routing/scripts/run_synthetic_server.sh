#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/fdong/ymluo/projects/mor_kv_operator_routing}"
PARALLEL="${PARALLEL:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
HEAD_PROFILE="${HEAD_PROFILE:-/home/fdong/ymluo/projects/qwen3_head_function_stability/outputs/full_v3/head_profiles.csv}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-${PROJECT}/outputs/synthetic_${STAMP}}"

CORPUS="${CORPUS:-${PARALLEL}/data/synthetic_controlled_100k_500_v1}"
ALLHEAD="${ALLHEAD:-${PARALLEL}/outputs/synthetic_controlled_100k_500_allhead_consensus_v1/per_head_topk.npz}"
BM25="${BM25:-${PARALLEL}/outputs/synthetic_controlled_100k_500_bm25_v1/block_scores.npy}"
MODEL="${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}"

cd "$PROJECT"
mkdir -p "$OUT"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

"$PY" -u src/run_mor_kv_offline.py \
  --allhead_topk_npz "$ALLHEAD" \
  --queries_jsonl "$CORPUS/queries.jsonl" \
  --bm25_scores "$BM25" \
  --head_profiles "$HEAD_PROFILE" \
  --output_dir "$OUT" \
  --budgets "${BUDGETS:-1,4,8,16,39}" \
  --negative_penalty "${NEGATIVE_PENALTY:-0.25}" \
  --gqa_group_size 2 \
  --gqa_deduplicate true \
  2>&1 | tee "$OUT/run.log"

if [[ "${RUN_NLL:-true}" == "true" ]]; then
  mkdir -p "$OUT/answer_nll_b4"
  GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
  IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PY" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "${#GPU_ARRAY[@]}" \
    src/evaluate_mor_answer_nll.py \
      --model_name_or_path "$MODEL" \
      --corpus_dir "$CORPUS" \
      --retrieval_results "$OUT/query_results.csv" \
      --output_dir "$OUT/answer_nll_b4" \
      --methods bm25_b4,single_hybrid_b4,mor_kv_b4,wrong_router_mor_b4 \
      --split test \
      --dtype float16 \
      --attn_implementation sdpa \
    2>&1 | tee "$OUT/answer_nll_b4.log"

  mkdir -p "$OUT/answer_nll_b4_dev"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PY" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "${#GPU_ARRAY[@]}" \
    src/evaluate_mor_answer_nll.py \
      --model_name_or_path "$MODEL" \
      --corpus_dir "$CORPUS" \
      --retrieval_results "$OUT/query_results.csv" \
      --output_dir "$OUT/answer_nll_b4_dev" \
      --methods bm25_b4,single_hybrid_b4,mor_kv_b4,wrong_router_mor_b4 \
      --split dev \
      --dtype float16 \
      --attn_implementation sdpa \
    2>&1 | tee "$OUT/answer_nll_b4_dev.log"

  "$PY" src/compile_nll_routed_policy.py \
    --dev_nll_rows "$OUT/answer_nll_b4_dev/answer_nll_rows.csv" \
    --test_nll_rows "$OUT/answer_nll_b4/answer_nll_rows.csv" \
    --router_predictions "$OUT/router_predictions.csv" \
    --retrieval_results "$OUT/query_results.csv" \
    --output_dir "$OUT/nll_routed_b4" \
    2>&1 | tee "$OUT/nll_routed_b4.log"
fi

echo "[mor-kv] done: $OUT"
