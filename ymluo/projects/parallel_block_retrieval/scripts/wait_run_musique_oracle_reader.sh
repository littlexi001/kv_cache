#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
PIPELINE_SUMMARY="${ROOT}/outputs/musique_official_10m_localspan_grouped3_generation_test_v2/summary.json"
LOG="${LOG:-/tmp/musique_official_oracle_reader.log}"

while [[ ! -f "${PIPELINE_SUMMARY}" ]]; do
  sleep 60
done

while true; do
  GPU="$({
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
      awk -F, '{gsub(/ /, "", $0); if (($2 + 0) < 100 && ($3 + 0) < 5) {print $1; exit}}'
  })"
  if [[ -n "${GPU}" ]]; then
    break
  fi
  sleep 60
done

cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
  src/evaluate_global_step_branch_generation.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --corpus_dir data/musique_official_10m_400_v1 \
  --step_queries_path data/musique_official_10m_396_audited_v1/step_queries.jsonl \
  --retrieval_rows_path outputs/musique_official_10m_oracle_block_branches_v2/rows.jsonl \
  --output_dir outputs/musique_official_10m_oracle_block_generation_test_v2 \
  --split test --max_new_tokens 24 --max_retrieval_branches 1 --device cuda \
  >> "${LOG}" 2>&1
