#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
TORCHRUN="${TORCHRUN:-/home/fdong/miniconda3/envs/moe/bin/torchrun}"
LOG="${LOG:-${ROOT}/logs/musique_confidence_extension_r4_6_v7.log}"
EXTENSION="outputs/musique_official_aligned_2000_confidence_extend_r4_6_v7"
GENERATION="outputs/musique_official_aligned_2000_confidence_extend_r4_6_generation_v7"
MERGED="outputs/musique_official_aligned_2000_confidence_extend_r4_6_merged_v7"
CHAIN="outputs/musique_official_aligned_2000_strict_chain500_v6"

cd "${ROOT}"
mkdir -p "$(dirname "${LOG}")"

while true; do
  mapfile -t IDLE_GPUS < <(
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits |
      awk -F, '{gsub(/ /, "", $0); if (($2 + 0) < 100 && ($3 + 0) < 5) print $1}' |
      head -6
  )
  if (( ${#IDLE_GPUS[@]} >= 1 )); then
    break
  fi
  printf '%s waiting for at least one idle GPU\n' "$(date -Iseconds)" >> "${LOG}"
  sleep 60
done

GPU_LIST="$(IFS=,; echo "${IDLE_GPUS[*]}")"
NPROC="${#IDLE_GPUS[@]}"
EXCLUDED="$(cat "${EXTENSION}/excluded_query_ids.txt")"
printf '%s using GPUs %s, nproc=%s\n' "$(date -Iseconds)" "${GPU_LIST}" "${NPROC}" >> "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_LIST}" "${TORCHRUN}" --standalone \
  --nproc_per_node="${NPROC}" src/evaluate_global_step_branch_generation.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --corpus_dir data/musique_official_10m_aligned_2000_v3 \
  --step_queries_path "${CHAIN}/answer_steps.jsonl" \
  --retrieval_rows_path "${EXTENSION}/rows.jsonl" \
  --output_dir "${GENERATION}" --split test \
  --step_types resolve_answer_from_bridge \
  --exclude_query_ids "${EXCLUDED}" \
  --max_new_tokens 24 --max_retrieval_branches 3 \
  --prompt_mode adaptive --device cuda --dtype float16 \
  >> "${LOG}" 2>&1

"${PYTHON}" src/merge_gated_extension_generations.py \
  --base_rows_path outputs/musique_official_aligned_2000_strict_chain500_answer_reader_v6/rows.jsonl \
  --extension_rows_path "${GENERATION}/rows.jsonl" \
  --output_dir "${MERGED}" >> "${LOG}" 2>&1

"${PYTHON}" src/apply_transition_support_head.py \
  --step_queries_path "${CHAIN}/answer_steps.jsonl" \
  --generation_rows_path "${MERGED}/rows.jsonl" \
  --head_path outputs/musique_official_aligned_2000_transition_head_final_support100_v5/train_answer500_transition_head.json \
  --output_path "${MERGED}/answer_selections.jsonl" \
  --method heuristic_structured --split test \
  --step_type resolve_answer_from_bridge >> "${LOG}" 2>&1

"${PYTHON}" src/compare_adaptive_extension.py \
  --base_generation_rows_path outputs/musique_official_aligned_2000_strict_chain500_answer_reader_v6/rows.jsonl \
  --base_selection_rows_path "${CHAIN}/answer_selections.jsonl" \
  --adaptive_generation_rows_path "${MERGED}/rows.jsonl" \
  --adaptive_selection_rows_path "${MERGED}/answer_selections.jsonl" \
  --output_path "${MERGED}/paired_vs_top3.json" \
  --method heuristic_structured >> "${LOG}" 2>&1

printf '%s confidence-gated extension complete\n' "$(date -Iseconds)" >> "${LOG}"
