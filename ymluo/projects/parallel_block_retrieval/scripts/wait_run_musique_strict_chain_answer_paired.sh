#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
LOG="${LOG:-${ROOT}/logs/musique_strict_chain_answer_paired_v3.log}"
CORPUS="data/musique_official_10m_aligned_2000_v3"

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

for METHOD in bm25 svd; do
  STEPS="outputs/musique_official_aligned_2000_${METHOD}_strict_chain100_v3/answer_steps.jsonl"
  BRANCHES="outputs/musique_official_aligned_2000_${METHOD}_strict_chain100_bm25_branches_v3/rows.jsonl"
  OUTPUT="outputs/musique_official_aligned_2000_${METHOD}_strict_chain100_answer_reader_v3"
  CUDA_VISIBLE_DEVICES="${GPU}" /usr/bin/time \
    -f "${METHOD^^}_WALL=%e" -a -o "${LOG}" \
    "${PYTHON}" src/evaluate_global_step_branch_generation.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --corpus_dir "${CORPUS}" \
    --step_queries_path "${STEPS}" \
    --retrieval_rows_path "${BRANCHES}" \
    --output_dir "${OUTPUT}" --split test \
    --max_new_tokens 24 --max_retrieval_branches 3 \
    --prompt_mode adaptive --device cuda --dtype float16 \
    >> "${LOG}" 2>&1
done

printf '%s strict-chain paired answer complete\n' "$(date -Iseconds)" >> "${LOG}"
