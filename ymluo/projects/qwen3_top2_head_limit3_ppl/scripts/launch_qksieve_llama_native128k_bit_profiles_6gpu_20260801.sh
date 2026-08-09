#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
TEMPLATE="${TEMPLATE:-${ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260801_qksieve_llama31_native128k_bit_profiles_6gpu}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"
HISTORY_TOKENS="${HISTORY_TOKENS:-131040}"
EVAL_TOKENS="${EVAL_TOKENS:-16}"
VARIANTS="${VARIANTS:-exact_qk_oracle_k1280,qksieve_keymse_requestlocal_fixedalloc_i112_41_fulltopk_k1280,qksieve_keymse_requestlocal_fixedalloc_b13_4221_fulltopk_k1280,qksieve_keymse_requestlocal_fixedalloc_b15_4421_fulltopk_k1280,qksieve_keymse_requestlocal_fulltopk_k1280}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export QKSIEVE_EXACT_SELECTION_DIAGNOSTICS=1
mkdir -p "${RUN_ROOT}/logs"
cd "${ROOT}"

workers=(
  "0,1|sports_both|20260811|politics|20260814"
  "2,3|medicine|20260812|religion|20260815"
  "4,5|computer|20260813|mixed_b|20260816"
)
pids=()
for worker in "${workers[@]}"; do
  IFS="|" read -r gpus topic_a seed_a topic_b seed_b <<<"${worker}"
  (
    for item in "${topic_a}|${seed_a}" "${topic_b}|${seed_b}"; do
      IFS="|" read -r topic seed <<<"${item}"
      output_dir="${RUN_ROOT}/${topic}_seed${seed}"
      mkdir -p "${output_dir}"
      if [[ -f "${output_dir}/ALL_COMPLETE" ]]; then
        echo "SKIP ${topic}/seed${seed}"
        continue
      fi
      CUDA_VISIBLE_DEVICES="${gpus}" "${PYTHON}" -u \
        src/run_qksieve_coldskip_longcontext_quality_20260730.py \
        --model_name_or_path "${MODEL}" \
        --template "${TEMPLATE}" \
        --output_dir "${output_dir}" \
        --history_tokens "${HISTORY_TOKENS}" \
        --eval_tokens "${EVAL_TOKENS}" \
        --topic "${topic}" \
        --seed "${seed}" \
        --repeat_topic_stream_if_short \
        --prefill_chunk_tokens 1024 \
        --dataset_cache_dir "${DATASET_CACHE_DIR}" \
        --dtype float16 \
        --device cuda \
        --device_map balanced \
        --max_memory_per_gpu_gib 22 \
        --variants "${VARIANTS}" \
        >"${output_dir}/run.log" 2>&1
    done
  ) >"${RUN_ROOT}/logs/${topic_a}_${topic_b}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi
touch "${RUN_ROOT}/ALL_COMPLETE"

