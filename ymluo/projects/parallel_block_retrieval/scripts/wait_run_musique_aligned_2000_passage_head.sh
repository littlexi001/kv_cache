#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
LOG="${LOG:-${ROOT}/logs/musique_aligned_2000_passage_head_v3.log}"
CORPUS="data/musique_official_10m_aligned_2000_v3"
CANDIDATES="outputs/musique_official_aligned_2000_lexical_anchor_v3/rows.jsonl"
PROFILE="outputs/musique_official_aligned_2000_sparse_k_top16_v3"
RERANK="outputs/musique_official_aligned_2000_qk_rerank_v3"

cd "${ROOT}"
mkdir -p "$(dirname "${LOG}")"

while true; do
  GPU="$({
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits |
      awk -F, '{gsub(/ /, "", $0); if (($2 + 0) < 100 && ($3 + 0) < 5) {print $1; exit}}'
  } || true)"
  if [[ -n "${GPU}" ]]; then
    break
  fi
  printf '%s waiting for one idle GPU\n' "$(date -Iseconds)" >> "${LOG}"
  sleep 60
done
printf '%s using GPU %s\n' "$(date -Iseconds)" "${GPU}" >> "${LOG}"

if [[ ! -f "${PROFILE}/summary.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" src/profile_sparse_candidate_k.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --corpus_dir "${CORPUS}" \
    --candidate_rows_path "${CANDIDATES}" \
    --candidate_field lexical_candidates --candidate_limit 16 \
    --profile_dir "${PROFILE}" \
    --pairs 3:10,21:8,6:7,16:14 \
    --batch_blocks 8 --dtype float16 --device cuda \
    >> "${LOG}" 2>&1
fi

if [[ ! -f "${PROFILE}/svd32_summary.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" src/build_sparse_k_svd_sidecar.py \
    --profile_dir "${PROFILE}" \
    --candidate_rows_path "${CANDIDATES}" \
    --candidate_field lexical_candidates --candidate_limit 16 \
    --basis_splits train --calibration_blocks 512 \
    --svd_rank 32 --batch_blocks 32 --device cuda \
    >> "${LOG}" 2>&1
fi

if [[ ! -f "${RERANK}/summary.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" src/rerank_sparse_candidate_blocks_svd.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --profile_dir "${PROFILE}" \
    --step_queries_path "${CORPUS}/step_queries.jsonl" \
    --candidate_rows_path "${CANDIDATES}" \
    --candidate_field lexical_candidates --candidate_limit 16 \
    --output_dir "${RERANK}" \
    --splits train,dev,test --task_types multihop \
    --query_tokens 16 --svd_rank 32 --dtype float16 --device cuda \
    >> "${LOG}" 2>&1
fi

"${PYTHON}" src/train_pairwise_qk_passage_head.py \
  --rows_path "${RERANK}/rows.jsonl" \
  --train_splits train \
  --output_path "${RERANK}/passage_head_train_only.json" \
  >> "${LOG}" 2>&1

"${PYTHON}" src/train_pairwise_qk_passage_head.py \
  --rows_path "${RERANK}/rows.jsonl" \
  --train_splits train,dev \
  --output_path "${RERANK}/passage_head_train_dev.json" \
  >> "${LOG}" 2>&1

printf '%s aligned-2000 passage-head experiment complete\n' "$(date -Iseconds)" >> "${LOG}"
