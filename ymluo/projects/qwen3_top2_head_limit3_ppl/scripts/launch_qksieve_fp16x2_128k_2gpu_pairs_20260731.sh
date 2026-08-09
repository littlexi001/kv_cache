#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
TEMPLATE="${TEMPLATE:-${PROJECT_ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"
PROTOCOL="${PROTOCOL:-synthetic}"
HISTORY_TOKENS="${HISTORY_TOKENS:-131072}"
PREFILL_CHUNK_TOKENS="${PREFILL_CHUNK_TOKENS:-1024}"
ALLOW_CONTEXT_EXTRAPOLATION="${ALLOW_CONTEXT_EXTRAPOLATION:-1}"

case "${PROTOCOL}" in
  synthetic)
    RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/results/20260731_qksieve_fp16x2_128k_synthetic_8seed}"
    EVAL_TOKENS=2
    VARIANTS="exact_qk_oracle_k1280,qksieve_keymse_requestlocal_fixedalloc_i112_41_fulltopk_k1280,qksieve_requestlocal_fp16x2_fulltopk_k1280,exact_qk_oracle_k2560,qksieve_keymse_requestlocal_fixedalloc_i112_41_fulltopk_k2560,qksieve_requestlocal_fp16x2_fulltopk_k2560"
    seed_base="${SYNTHETIC_SEED_BASE:-256}"
    workers=(
      "0,1|seed${seed_base}|${seed_base}|seed$((seed_base + 4))|$((seed_base + 4))"
      "2,3|seed$((seed_base + 1))|$((seed_base + 1))|seed$((seed_base + 5))|$((seed_base + 5))"
      "4,5|seed$((seed_base + 2))|$((seed_base + 2))|seed$((seed_base + 6))|$((seed_base + 6))"
      "6,7|seed$((seed_base + 3))|$((seed_base + 3))|seed$((seed_base + 7))|$((seed_base + 7))"
    )
    export QKSIEVE_EXACT_SELECTION_DIAGNOSTICS=1
    ;;
  natural)
    RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/results/20260731_qksieve_fp16x2_128k_natural_8topic}"
    EVAL_TOKENS="${EVAL_TOKENS:-16}"
    VARIANTS="qksieve_keymse_requestlocal_fixedalloc_i112_41_fulltopk_k1280,qksieve_requestlocal_fp16x2_fulltopk_k1280,qksieve_keymse_requestlocal_fixedalloc_i112_41_fulltopk_k2560,qksieve_requestlocal_fp16x2_fulltopk_k2560"
    workers=(
      "0,1|sports_both|20260741|politics|20260745"
      "2,3|medicine|20260742|religion|20260746"
      "4,5|computer|20260743|mixed_a|20260747"
      "6,7|space|20260744|mixed_b|20260748"
    )
    unset QKSIEVE_EXACT_SELECTION_DIAGNOSTICS
    export QKSIEVE_PROFILE_STAGES=1
    ;;
  *)
    echo "Unsupported PROTOCOL=${PROTOCOL}; expected synthetic or natural" >&2
    exit 2
    ;;
esac

mkdir -p "${RUN_ROOT}/logs"
cd "${PROJECT_ROOT}"
export PATH="/home/fdong/miniconda3/envs/moe/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

pids=()
for worker in "${workers[@]}"; do
  IFS="|" read -r gpus name_a seed_a name_b seed_b <<<"${worker}"
  (
    export CUDA_VISIBLE_DEVICES="${gpus}"
    for item in "${name_a}|${seed_a}" "${name_b}|${seed_b}"; do
      IFS="|" read -r name seed <<<"${item}"
      output_dir="${RUN_ROOT}/${name}"
      mkdir -p "${output_dir}"
      if [[ -f "${output_dir}/ALL_COMPLETE" ]]; then
        echo "SKIP completed: ${name}"
        continue
      fi
      extra_args=()
      if [[ "${PROTOCOL}" == "synthetic" ]]; then
        extra_args+=(--synthetic_rope_seed "${seed}")
      else
        extra_args+=(--topic "${name}" --seed "${seed}" --repeat_topic_stream_if_short)
      fi
      if [[ "${ALLOW_CONTEXT_EXTRAPOLATION}" == "1" ]]; then
        extra_args+=(--allow_context_extrapolation)
      fi
      "${PYTHON}" -u src/run_qksieve_coldskip_longcontext_quality_20260730.py \
        --model_name_or_path "${MODEL}" \
        --template "${TEMPLATE}" \
        --output_dir "${output_dir}" \
        --history_tokens "${HISTORY_TOKENS}" \
        --eval_tokens "${EVAL_TOKENS}" \
        --prefill_chunk_tokens "${PREFILL_CHUNK_TOKENS}" \
        --dataset_cache_dir "${DATASET_CACHE_DIR}" \
        --dtype float16 \
        --device cuda \
        --device_map balanced \
        --max_memory_per_gpu_gib 22 \
        --variants "${VARIANTS}" \
        "${extra_args[@]}" \
        >"${output_dir}/run.log" 2>&1
    done
  ) >"${RUN_ROOT}/logs/${name_a}_${name_b}.log" 2>&1 &
  pids+=("$!")
  echo "${name_a},${name_b}: GPUs ${gpus}, PID $!"
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi
touch "${RUN_ROOT}/ALL_COMPLETE"
