#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
TEMPLATE="${TEMPLATE:-${ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
OUTPUT="${OUTPUT:-${ROOT}/results/20260803_qksieve_massladder_32k_heldout3_gpu023_v1}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
export QKSIEVE_MASS_LADDER_FLOOR_K=1280
export QKSIEVE_MASS_LADDER_GROWTH=1.5
export QKSIEVE_MASS_LADDER_MAX_FRACTION=0.25

mkdir -p "${OUTPUT}/logs"
cd "${ROOT}"

topics=(computer space politics)
gpus=(0 2 3)
seeds=(20260871 20260872 20260873)
pids=()

for index in "${!topics[@]}"; do
  topic="${topics[$index]}"
  gpu="${gpus[$index]}"
  seed="${seeds[$index]}"
  topic_output="${OUTPUT}/${topic}"
  mkdir -p "${topic_output}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    src/run_qksieve_coldskip_longcontext_quality_20260730.py \
    --model_name_or_path "${MODEL}" \
    --template "${TEMPLATE}" \
    --output_dir "${topic_output}" \
    --history_tokens 32000 \
    --eval_tokens 64 \
    --topic "${topic}" \
    --seed "${seed}" \
    --repeat_topic_stream_if_short \
    --prefill_chunk_tokens 1024 \
    --dataset_cache_dir /home/fdong/ymluo/datasets/sklearn \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    --max_memory_per_gpu_gib 22 \
    --variants \
      qksieve_keymse_requestlocal_valuesketch16i4_sampled_k1280,qksieve_keymse_requestlocal_valuesketch16i4_massladder90 \
    >"${OUTPUT}/logs/${topic}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi
touch "${OUTPUT}/ALL_COMPLETE"
