#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
DATA="${DATA:-${ROOT}/data/ruler_generated/llama31_8b_64k128k_m5_seed42.jsonl}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260801_qksieve_general_ruler_hard64k128k_m2_gpu1234}"
METHODS="full_kv,qksieve_requestlocal_qkbalanced_keymse_wmma_valuesketch16"
TASKS="niah_multivalue,vt,cwe,fwe,qa_hotpot"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=0.5
export QKSIEVE_VALUE_SKETCH_MIN_HISTORY=65536

mkdir -p "${RUN_ROOT}/logs"
cd "${ROOT}"

CUDA_VISIBLE_DEVICES=1,2,3,4 "${PYTHON}" -u \
  src/run_sample_calibrated_ruler_20260717.py \
  --model_name_or_path "${MODEL}" \
  --examples_jsonl "${DATA}" \
  --output_dir "${RUN_ROOT}" \
  --methods "${METHODS}" \
  --ruler_tasks "${TASKS}" \
  --ruler_lengths 65536,131072 \
  --max_samples_per_task 2 \
  --max_new_tokens_override 0 \
  --prefill_chunk_tokens 2048 \
  --prompt_wrapper llama3 \
  --qk_metric_query_shrinkage 0.75 \
  --sampled_quantile_sample_count 256 \
  --sampled_quantile_target_tail_count 32 \
  --dtype float16 \
  --device cuda \
  --device_map balanced \
  >"${RUN_ROOT}/logs/run.log" 2>&1

"${PYTHON}" src/summarize_countcap_benchmark_20260722.py \
  --kind ruler \
  --input_glob "${RUN_ROOT}/sample_results.csv" \
  --output_dir "${RUN_ROOT}/merged" \
  >"${RUN_ROOT}/logs/merge.log" 2>&1

"${PYTHON}" src/summarize_qksieve_ruler_20260728.py \
  --input_csv "${RUN_ROOT}/merged/sample_results.csv" \
  --project_root "${ROOT}" \
  --output "${RUN_ROOT}/targeted_summary.json" \
  --expected_length_samples 65536:2,131072:2 \
  >"${RUN_ROOT}/logs/summary.log" 2>&1

touch "${RUN_ROOT}/ALL_COMPLETE"
