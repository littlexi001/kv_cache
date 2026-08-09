#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
LOG="${LOG:-/tmp/musique_official_pipeline.log}"

cd "${ROOT}"

while true; do
  mapfile -t IDLE_GPUS < <(
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
      awk -F, '{gsub(/ /, "", $0); if (($2 + 0) < 100 && ($3 + 0) < 5) print $1}'
  )
  if (( ${#IDLE_GPUS[@]} >= 1 )); then
    break
  fi
  printf '%s waiting for an idle GPU\n' "$(date -Iseconds)" >> "${LOG}"
  sleep 60
done

if (( ${#IDLE_GPUS[@]} > 4 )); then
  IDLE_GPUS=("${IDLE_GPUS[@]:0:4}")
fi
GPU_LIST="$(IFS=,; echo "${IDLE_GPUS[*]}")"
NPROC="${#IDLE_GPUS[@]}"
FIRST_GPU="${IDLE_GPUS[0]}"
printf '%s using GPUs %s\n' "$(date -Iseconds)" "${GPU_LIST}" >> "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_LIST}" "${PYTHON}" -m torch.distributed.run \
  --standalone --nproc_per_node="${NPROC}" \
  src/evaluate_global_step_branch_generation.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --corpus_dir data/musique_official_10m_400_v1 \
  --step_queries_path data/musique_official_10m_396_audited_v1/step_queries.jsonl \
  --retrieval_rows_path outputs/musique_official_10m_bm25_block3_branches_v2/rows.jsonl \
  --output_dir outputs/musique_official_10m_bm25_block3_generation_test_v2 \
  --split test --max_new_tokens 24 --max_retrieval_branches 3 --device cuda \
  >> "${LOG}" 2>&1

CUDA_VISIBLE_DEVICES="${FIRST_GPU}" "${PYTHON}" \
  src/profile_sparse_candidate_k.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --corpus_dir data/musique_official_10m_400_v1 \
  --candidate_rows_path outputs/musique_official_10m_lexical_anchor_v3/rows.jsonl \
  --candidate_field lexical_candidates --candidate_limit 3 \
  --profile_dir outputs/musique_official_10m_sparse_k_top3_v2 \
  --pairs 3:10,21:8,6:7,16:14 --batch_blocks 8 --dtype float16 --device cuda \
  >> "${LOG}" 2>&1

"${PYTHON}" src/build_sparse_sentence_sidecar.py \
  --corpus_dir data/musique_official_10m_400_v1 \
  --candidate_rows_path outputs/musique_official_10m_lexical_anchor_v3/rows.jsonl \
  --candidate_field lexical_candidates --candidate_limit 3 \
  --output_dir outputs/musique_official_10m_sparse_sentence_top3_v2 \
  --model_name_or_path Qwen/Qwen3-0.6B \
  >> "${LOG}" 2>&1

CUDA_VISIBLE_DEVICES="${FIRST_GPU}" "${PYTHON}" \
  src/run_global_candidate_sentence_kv_rerank.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --corpus_dir data/musique_official_10m_400_v1 \
  --profile_dir outputs/musique_official_10m_sparse_k_top3_v2 \
  --sidecar_dir outputs/musique_official_10m_sparse_sentence_top3_v2 \
  --step_queries_path data/musique_official_10m_396_audited_v1/step_queries.jsonl \
  --candidate_rows_path outputs/musique_official_10m_lexical_anchor_v3/rows.jsonl \
  --candidate_field lexical_candidates --candidate_limit 3 \
  --output_dir outputs/musique_official_10m_localspan_top3_v2 \
  --splits train,dev,test --task_types multihop \
  --query_tokens 16 --branch_blocks 3 --spans_per_block 3 \
  --branch_block_order candidate --index_backend sparse_cpu --device cuda \
  --resolve_bridge_profiles 0,1,2,3 --resolve_answer_profiles 0,1,2,3 \
  >> "${LOG}" 2>&1

"${PYTHON}" src/group_sentence_block_branches.py \
  --rows_path outputs/musique_official_10m_localspan_top3_v2/rows.jsonl \
  --output_dir outputs/musique_official_10m_localspan_grouped3_v2 \
  --branch_blocks 3 --spans_per_block 3 \
  >> "${LOG}" 2>&1

CUDA_VISIBLE_DEVICES="${GPU_LIST}" "${PYTHON}" -m torch.distributed.run \
  --standalone --nproc_per_node="${NPROC}" \
  src/evaluate_global_step_branch_generation.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --corpus_dir data/musique_official_10m_400_v1 \
  --step_queries_path data/musique_official_10m_396_audited_v1/step_queries.jsonl \
  --retrieval_rows_path outputs/musique_official_10m_localspan_grouped3_v2/rows.jsonl \
  --output_dir outputs/musique_official_10m_localspan_grouped3_generation_test_v2 \
  --split test --max_new_tokens 24 --max_retrieval_branches 3 --device cuda \
  >> "${LOG}" 2>&1

printf '%s MuSiQue official pipeline complete\n' "$(date -Iseconds)" >> "${LOG}"
