#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
LOG="${LOG:-${ROOT}/logs/musique_2000_reader_paired_v3.log}"
CORPUS="data/musique_official_10m_aligned_2000_v3"
BM25_BRANCHES="outputs/musique_official_aligned_2000_bm25_block3_branches_v3/rows.jsonl"
SVD_BRANCHES="outputs/musique_official_aligned_2000_passage_head_svd_block3_branches_v3/rows.jsonl"
BM25_OUT="outputs/musique_official_aligned_2000_bm25_reader_test200_v3"
SVD_OUT="outputs/musique_official_aligned_2000_svd_passage_reader_test200_v3"

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

CUDA_VISIBLE_DEVICES="${GPU}" /usr/bin/time -f 'BM25_WALL=%e' -a -o "${LOG}" \
  "${PYTHON}" src/evaluate_global_step_branch_generation.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --corpus_dir "${CORPUS}" \
  --step_queries_path "${CORPUS}/step_queries.jsonl" \
  --retrieval_rows_path "${BM25_BRANCHES}" \
  --output_dir "${BM25_OUT}" --split test \
  --max_new_tokens 24 --max_retrieval_branches 3 --max_steps 200 \
  --prompt_mode adaptive --device cuda --dtype float16 \
  >> "${LOG}" 2>&1

CUDA_VISIBLE_DEVICES="${GPU}" /usr/bin/time -f 'SVD_WALL=%e' -a -o "${LOG}" \
  "${PYTHON}" src/evaluate_global_step_branch_generation.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --corpus_dir "${CORPUS}" \
  --step_queries_path "${CORPUS}/step_queries.jsonl" \
  --retrieval_rows_path "${SVD_BRANCHES}" \
  --output_dir "${SVD_OUT}" --split test \
  --max_new_tokens 24 --max_retrieval_branches 3 --max_steps 200 \
  --prompt_mode adaptive --device cuda --dtype float16 \
  >> "${LOG}" 2>&1

"${PYTHON}" src/analyze_branch_transition_verifier.py \
  --rows_path "${BM25_OUT}/rows.jsonl" \
  --step_queries_path "${CORPUS}/step_queries.jsonl" \
  --output_path "${BM25_OUT}/verifier.json" >> "${LOG}" 2>&1

"${PYTHON}" src/analyze_branch_transition_verifier.py \
  --rows_path "${SVD_OUT}/rows.jsonl" \
  --step_queries_path "${CORPUS}/step_queries.jsonl" \
  --output_path "${SVD_OUT}/verifier.json" >> "${LOG}" 2>&1

printf '%s paired reader complete\n' "$(date -Iseconds)" >> "${LOG}"
