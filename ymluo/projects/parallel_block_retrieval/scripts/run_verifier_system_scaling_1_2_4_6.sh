#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
TORCHRUN="${TORCHRUN:-/home/fdong/miniconda3/envs/moe/bin/torchrun}"
OUTPUT="${OUTPUT:-${ROOT}/outputs/musique_verifier_system_scaling_30q_v1}"
LOG="${LOG:-${ROOT}/logs/musique_verifier_system_scaling_30q_v1.log}"
MAX_QUERIES="${MAX_QUERIES:-30}"
WARMUP_QUERIES="${WARMUP_QUERIES:-2}"

cd "${ROOT}"
mkdir -p "${OUTPUT}" "$(dirname "${LOG}")"

idle_gpus() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits |
    awk -F, '{gsub(/ /, "", $0); if (($2 + 0) < 100 && ($3 + 0) < 5) print $1}'
}

for WORLD in 1 2 4 6; do
  while true; do
    mapfile -t GPUS < <(idle_gpus)
    (( ${#GPUS[@]} >= WORLD )) && break
    printf '%s waiting for %s idle GPUs\n' "$(date -Iseconds)" "${WORLD}" >> "${LOG}"
    sleep 60
  done
  GPUS=("${GPUS[@]:0:${WORLD}}")
  GPU_LIST="$(IFS=,; echo "${GPUS[*]}")"
  printf '%s world=%s gpus=%s\n' "$(date -Iseconds)" "${WORLD}" "${GPU_LIST}" >> "${LOG}"
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" "${TORCHRUN}" --standalone \
    --nproc_per_node="${WORLD}" \
    src/benchmark_verifier_system_same_query_scaling.py \
    --corpus_dir data/musique_official_10m_aligned_2000_v3 \
    --bridge_steps_path data/musique_official_10m_aligned_2000_v3/step_queries.jsonl \
    --bridge_retrieval_rows_path outputs/musique_official_aligned_2000_passage_head_svd_block3_branches_v3/rows.jsonl \
    --answer_steps_path outputs/musique_official_aligned_2000_qwen8b_concat3_strict_chain500_v10/answer_steps.jsonl \
    --answer_generation_rows_path outputs/musique_official_aligned_2000_8b_answer_direct_extract16_test500_v15/rows.jsonl \
    --output_dir "${OUTPUT}/world${WORLD}" \
    --warmup_queries "${WARMUP_QUERIES}" --max_queries "${MAX_QUERIES}" \
    --branches 16 --max_new_tokens 24 --verifier_batch_size 8 \
    --retrieval_seconds 0.0425 --dtype float16 --device cuda \
    >> "${LOG}" 2>&1
done

"${PYTHON}" src/analyze_verifier_system_scaling.py \
  --summary_paths "${OUTPUT}/world1/summary.json,${OUTPUT}/world2/summary.json,${OUTPUT}/world4/summary.json,${OUTPUT}/world6/summary.json" \
  --output_path "${OUTPUT}/scaling.json" >> "${LOG}" 2>&1

printf '%s verifier system scaling complete\n' "$(date -Iseconds)" >> "${LOG}"
