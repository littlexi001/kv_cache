#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
LOG="${LOG:-${ROOT}/logs/musique_oracle_prompt_ablation_v3.log}"
CORPUS="data/musique_official_10m_aligned_2000_v3"
BRANCHES="outputs/musique_official_aligned_2000_oracle_block_branches_v3/rows.jsonl"

cd "${ROOT}"
mkdir -p "$(dirname "${LOG}")"
while true; do
  GPU="$({
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits |
      awk -F, '{gsub(/ /, "", $0); if (($2 + 0) < 100 && ($3 + 0) < 5) {print $1; exit}}'
  } || true)"
  [[ -n "${GPU}" ]] && break
  printf '%s waiting for one idle GPU\n' "$(date -Iseconds)" >> "${LOG}"
  sleep 60
done
printf '%s using GPU %s\n' "$(date -Iseconds)" "${GPU}" >> "${LOG}"

for MODE in legacy adaptive_extract; do
  OUTPUT="outputs/musique_official_aligned_2000_oracle_answer_${MODE}_test100_v3"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    src/evaluate_global_step_branch_generation.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --corpus_dir "${CORPUS}" \
    --step_queries_path "${CORPUS}/step_queries.jsonl" \
    --retrieval_rows_path "${BRANCHES}" \
    --output_dir "${OUTPUT}" --split test \
    --step_types resolve_answer_from_bridge --max_steps 100 \
    --max_new_tokens 24 --max_retrieval_branches 1 \
    --prompt_mode "${MODE}" --device cuda --dtype float16 \
    >> "${LOG}" 2>&1
done

"${PYTHON}" src/compare_paired_reader_runs.py \
  --baseline_rows_path outputs/musique_official_aligned_2000_oracle_answer_legacy_test100_v3/rows.jsonl \
  --candidate_rows_path outputs/musique_official_aligned_2000_oracle_answer_adaptive_extract_test100_v3/rows.jsonl \
  --output_path outputs/musique_official_aligned_2000_oracle_answer_adaptive_extract_test100_v3/paired_vs_legacy.json \
  >> "${LOG}" 2>&1

printf '%s oracle prompt ablation complete\n' "$(date -Iseconds)" >> "${LOG}"
