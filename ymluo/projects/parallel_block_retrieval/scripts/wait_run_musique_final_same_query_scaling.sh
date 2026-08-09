#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
LOG="${LOG:-${ROOT}/logs/musique_final_same_query_scaling_v3.log}"
OUTPUT="outputs/musique_official_aligned_2000_svd_passage_same_query_scaling_v3"

cd "${ROOT}"
mkdir -p "${OUTPUT}" "$(dirname "${LOG}")"

idle_gpus() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits |
    awk -F, '{gsub(/ /, "", $0); if (($2 + 0) < 100 && ($3 + 0) < 5) print $1}'
}

for WORLD in 1 2 3; do
  while true; do
    mapfile -t GPUS < <(idle_gpus)
    (( ${#GPUS[@]} >= WORLD )) && break
    printf '%s waiting for %s idle GPUs\n' "$(date -Iseconds)" "${WORLD}" >> "${LOG}"
    sleep 60
  done
  GPUS=("${GPUS[@]:0:${WORLD}}")
  GPU_LIST="$(IFS=,; echo "${GPUS[*]}")"
  printf '%s world=%s GPUs=%s\n' "$(date -Iseconds)" "${WORLD}" "${GPU_LIST}" >> "${LOG}"
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" "${PYTHON}" -m torch.distributed.run \
    --standalone --nproc_per_node="${WORLD}" \
    src/benchmark_same_query_branch_parallel.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --corpus_dir data/musique_official_10m_aligned_2000_v3 \
    --step_queries_path data/musique_official_10m_aligned_2000_v3/step_queries.jsonl \
    --retrieval_rows_path outputs/musique_official_aligned_2000_passage_head_svd_block3_branches_v3/rows.jsonl \
    --output_dir "${OUTPUT}/world${WORLD}" --split test \
    --step_types resolve_bridge,resolve_answer_from_bridge \
    --exclude_query_ids '' --max_queries 12 --branches 3 \
    --max_new_tokens 24 --dtype float16 --device cuda \
    >> "${LOG}" 2>&1
done

"${PYTHON}" src/analyze_same_query_branch_scaling.py \
  --summary_paths "${OUTPUT}/world1/summary.json,${OUTPUT}/world2/summary.json,${OUTPUT}/world3/summary.json" \
  --output_path "${OUTPUT}/scaling.json" >> "${LOG}" 2>&1

printf '%s final same-query scaling complete\n' "$(date -Iseconds)" >> "${LOG}"
