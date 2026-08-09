#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
MODEL="${MODEL:-${ROOT}/models/Meta-Llama-3.1-8B-Instruct-ms}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260809_valuesketch_removal_ablation_32k_v1}"
SEED="${SEED:-20260809}"
VARIANTS="qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64_novalue_k1280,qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64_k1280"

export PATH="/home/fdong/miniconda3/envs/nanogpt/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
export QKSIEVE_PRELOAD_EXTENSIONS=1
export QKSIEVE_PRELOAD_QMSE_RATE_TABLES=1
export QKSIEVE_MAX_QUANTILE_SAMPLE_COUNT=512

names=(war_and_peace monte_cristo qksieve_code)
texts=(
  "${ROOT}/data/ablation/war_and_peace_pg2600.txt"
  "${ROOT}/data/ablation/count_monte_cristo_pg1184.txt"
  "${ROOT}/src/run_head_top2_targeted_ppl_20260714.py"
)
histories=(32768 32768 32768 98304 98304 98304)
eval_counts=(64 64 64 32 32 32)

mkdir -p "${RUN_ROOT}/logs"
rm -f "${RUN_ROOT}/ALL_COMPLETE" "${RUN_ROOT}/FAILED"
cd "${ROOT}"

pids=()
for gpu in 0 1 2 3 4 5; do
  stream_index=$((gpu % 3))
  name="${names[$stream_index]}"
  history_tokens="${histories[$gpu]}"
  eval_tokens="${eval_counts[$gpu]}"
  output="${RUN_ROOT}/h${history_tokens}_${name}"
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    src/run_qksieve_coldskip_longcontext_quality_20260730.py \
    --model_name_or_path "${MODEL}" \
    --template "${ROOT}/data/unused_requestlocal_template.pt" \
    --output_dir "${output}" \
    --history_tokens "${history_tokens}" \
    --eval_tokens "${eval_tokens}" \
    --text_file "${texts[$stream_index]}" \
    --repeat_topic_stream_if_short \
    --prefill_chunk_tokens 1024 \
    --protect_recent_tokens 0 \
    --dataset_cache_dir "${ROOT}/data/sklearn" \
    --seed "${SEED}" \
    --dtype float16 \
    --device cuda \
    --device_map balanced \
    --max_memory_per_gpu_gib 22 \
    --variants "${VARIANTS}" \
    >"${RUN_ROOT}/logs/h${history_tokens}_${name}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
if [[ "${status}" -ne 0 ]]; then
  touch "${RUN_ROOT}/FAILED"
  exit "${status}"
fi
touch "${RUN_ROOT}/ALL_COMPLETE"
