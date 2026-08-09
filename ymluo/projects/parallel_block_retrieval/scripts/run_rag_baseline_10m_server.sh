#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
TORCHRUN="${TORCHRUN:-/home/fdong/miniconda3/envs/moe/bin/torchrun}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-/home/fdong/models/e5-base-v2}"
READER_MODEL="${READER_MODEL:-Qwen/Qwen3-8B}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,4}"
NPROC="${NPROC:-4}"

CORPUS="data/musique_official_10m_aligned_2000_v3"
STEPS="${CORPUS}/step_queries.jsonl"
INDEX="outputs/rag_e5_base_index_v1"
GOLD_RETRIEVAL="outputs/rag_e5_base_goldstate_retrieval_v1"
BRIDGE_BRANCHES="outputs/rag_e5_hybrid_bridge3_branches_v1"
BRIDGE_GENERATION="outputs/rag_e5_hybrid_bridge8b_test500_v1"
STRICT_CHAIN="outputs/rag_e5_hybrid_strict_chain500_v1"
SECOND_RETRIEVAL="outputs/rag_e5_hybrid_strict_second_retrieval_v1"
ANSWER_BRANCHES="outputs/rag_e5_hybrid_answer16_branches_v1"
ANSWER_GENERATION="outputs/rag_e5_hybrid_answer_direct_extract16_test500_v1"
SUPPORT_SCORES="outputs/rag_e5_hybrid_answer_direct_extract16_support_scores_v1"

cd "${PROJECT_DIR}"

if [[ ! -f "${GOLD_RETRIEVAL}/summary.json" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES%%,*}" "${PYTHON}" \
    src/run_external_embedding_retrieval.py \
    --embedding_model_name_or_path "${EMBEDDING_MODEL}" \
    --corpus_dir "${CORPUS}" \
    --step_queries_path "${STEPS}" \
    --index_dir "${INDEX}" \
    --output_dir "${GOLD_RETRIEVAL}" \
    --splits dev,test \
    --step_types resolve_bridge,resolve_answer_from_bridge \
    --candidate_blocks 512 \
    --batch_size 64 \
    --max_length 512 \
    --pooling mean \
    --query_prefix "query: " \
    --passage_prefix "passage: " \
    --dtype float16 \
    --device cuda
fi

if [[ ! -f "${BRIDGE_BRANCHES}/summary.json" ]]; then
  "${PYTHON}" src/prepare_allhead_block_branches.py \
    --step_queries_path "${STEPS}" \
    --allhead_rows_path "${GOLD_RETRIEVAL}/rows.jsonl" \
    --output_dir "${BRIDGE_BRANCHES}" \
    --ranking_field hybrid_rrf_candidates \
    --branch_blocks 3
fi

if [[ ! -f "${BRIDGE_GENERATION}/summary.json" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${TORCHRUN}" \
    --standalone --nproc_per_node="${NPROC}" \
    src/evaluate_global_step_branch_generation.py \
    --model_name_or_path "${READER_MODEL}" \
    --corpus_dir "${CORPUS}" \
    --step_queries_path "${STEPS}" \
    --retrieval_rows_path "${BRIDGE_BRANCHES}/rows.jsonl" \
    --output_dir "${BRIDGE_GENERATION}" \
    --split test \
    --step_types resolve_bridge \
    --max_new_tokens 24 \
    --max_retrieval_branches 3 \
    --branch_mode concat \
    --prompt_mode adaptive \
    --device cuda \
    --dtype float16
fi

if [[ ! -f "${STRICT_CHAIN}/summary.json" ]]; then
  "${PYTHON}" src/prepare_verified_chained_answer_steps.py \
    --step_queries_path "${STEPS}" \
    --bridge_generation_rows_path "${BRIDGE_GENERATION}/rows.jsonl" \
    --output_dir "${STRICT_CHAIN}" \
    --split test
fi

if [[ ! -f "${SECOND_RETRIEVAL}/summary.json" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES%%,*}" "${PYTHON}" \
    src/run_external_embedding_retrieval.py \
    --embedding_model_name_or_path "${EMBEDDING_MODEL}" \
    --corpus_dir "${CORPUS}" \
    --step_queries_path "${STRICT_CHAIN}/answer_steps.jsonl" \
    --index_dir "${INDEX}" \
    --output_dir "${SECOND_RETRIEVAL}" \
    --splits test \
    --step_types resolve_answer_from_bridge \
    --candidate_blocks 512 \
    --batch_size 64 \
    --max_length 512 \
    --pooling mean \
    --query_prefix "query: " \
    --passage_prefix "passage: " \
    --dtype float16 \
    --device cuda
fi

if [[ ! -f "${ANSWER_BRANCHES}/summary.json" ]]; then
  "${PYTHON}" src/prepare_allhead_block_branches.py \
    --step_queries_path "${STRICT_CHAIN}/answer_steps.jsonl" \
    --allhead_rows_path "${SECOND_RETRIEVAL}/rows.jsonl" \
    --output_dir "${ANSWER_BRANCHES}" \
    --ranking_field hybrid_rrf_candidates \
    --branch_blocks 16
fi

if [[ ! -f "${ANSWER_GENERATION}/summary.json" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${TORCHRUN}" \
    --standalone --nproc_per_node="${NPROC}" \
    src/evaluate_global_step_branch_generation.py \
    --model_name_or_path "${READER_MODEL}" \
    --corpus_dir "${CORPUS}" \
    --step_queries_path "${STRICT_CHAIN}/answer_steps.jsonl" \
    --retrieval_rows_path "${ANSWER_BRANCHES}/rows.jsonl" \
    --output_dir "${ANSWER_GENERATION}" \
    --split test \
    --step_types resolve_answer_from_bridge \
    --max_new_tokens 24 \
    --max_retrieval_branches 16 \
    --branch_mode independent \
    --prompt_mode atomic \
    --device cuda \
    --dtype float16
fi

if [[ ! -f "${SUPPORT_SCORES}/summary.json" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${TORCHRUN}" \
    --standalone --nproc_per_node="${NPROC}" \
    src/score_candidate_support_distributed.py \
    --model_name_or_path "${READER_MODEL}" \
    --step_queries_path "${STRICT_CHAIN}/answer_steps.jsonl" \
    --generation_rows_path "${ANSWER_GENERATION}/rows.jsonl" \
    --output_dir "${SUPPORT_SCORES}" \
    --split test \
    --batch_size 8 \
    --dtype float16 \
    --device cuda
fi

"${PYTHON}" src/summarize_rag_baseline.py \
  --first_retrieval_rows_path "${GOLD_RETRIEVAL}/rows.jsonl" \
  --bridge_generation_rows_path "${BRIDGE_GENERATION}/rows.jsonl" \
  --second_retrieval_rows_path "${SECOND_RETRIEVAL}/rows.jsonl" \
  --support_rows_path "${SUPPORT_SCORES}/rows.jsonl" \
  --baseline_support_rows_path \
    outputs/musique_official_aligned_2000_8b_answer_direct_extract16_support_scores_v15/rows.jsonl \
  --ranking_prefix hybrid_rrf \
  --split test \
  --output_path outputs/rag_e5_hybrid_strict_chain500_summary_v1.json
