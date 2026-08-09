#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
LOG="${LOG:-${ROOT}/logs/musique_qk_channel_diagnostics_v2.log}"
PROFILE="outputs/musique_official_aligned_10m_sparse_k_top16_v1"
RERANK="outputs/musique_official_aligned_10m_bm25_top16_qk_rerank_diag_v2"

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

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" src/rerank_sparse_candidate_blocks_svd.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --profile_dir "${PROFILE}" \
  --step_queries_path data/musique_official_10m_aligned_400_v2/step_queries.jsonl \
  --candidate_rows_path outputs/musique_official_aligned_10m_lexical_anchor_v1/rows.jsonl \
  --candidate_field lexical_candidates --candidate_limit 16 \
  --output_dir "${RERANK}" \
  --splits train,dev,test --task_types multihop \
  --query_tokens 16 --svd_rank 32 --dtype float16 --device cuda \
  >> "${LOG}" 2>&1

"${PYTHON}" src/analyze_qk_channel_token_diagnostics.py \
  --profile_dir "${PROFILE}" \
  --rows_path "${RERANK}/rows.jsonl" \
  --output_path "${RERANK}/channel_token_diagnostics.json" \
  >> "${LOG}" 2>&1

printf '%s channel diagnostics complete\n' "$(date -Iseconds)" >> "${LOG}"
