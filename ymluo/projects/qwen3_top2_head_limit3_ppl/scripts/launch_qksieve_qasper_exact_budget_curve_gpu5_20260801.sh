#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
DATA="${DATA:-/home/fdong/ymluo/external/KVCache-Factory/data/LongBench}"
FULL_ROOT="${FULL_ROOT:-${ROOT}/results/20260727_qkbalanced_longbench_official_middle_5way_5gpu}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260801_qksieve_qasper_exact_budget_curve_gpu5}"
FRACTIONS=(0.20 0.30)

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${RUN_ROOT}/logs"
cd "${ROOT}"

for fraction in "${FRACTIONS[@]}"; do
  tag="${fraction/./p}"
  method_root="${RUN_ROOT}/fraction_${tag}"
  output_dir="${method_root}/shard0"
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES=5 "${PYTHON}" -u \
    src/run_sample_calibrated_longbench_20260717.py \
    --model_name_or_path "${MODEL}" \
    --longbench_data_dir "${DATA}" \
    --tasks qasper \
    --methods exact_top2_fullprompt \
    --max_samples_per_task 50 \
    --max_prompt_tokens 7500 \
    --prompt_truncation_mode official_middle \
    --official_query_tail_tokens 8 \
    --max_context_tokens 0 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper llama3 \
    --countcap_direct_fraction_override "${fraction}" \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    --max_new_tokens_override 0 \
    --output_dir "${output_dir}" \
    >"${RUN_ROOT}/logs/fraction_${tag}.log" 2>&1
  "${PYTHON}" src/analyze_targeted_sparse_longbench_20260801.py \
    --full_root "${FULL_ROOT}" \
    --sparse_root "${method_root}" \
    --method exact_top2_fullprompt \
    --score_mode exact_qk_top2 \
    --tasks qasper \
    --expected_per_task 50 \
    --output "${method_root}/targeted_summary.json" \
    >"${RUN_ROOT}/logs/fraction_${tag}_summary.log" 2>&1
done

touch "${RUN_ROOT}/ALL_COMPLETE"
