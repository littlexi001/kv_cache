#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
QWEN_MODEL="${QWEN_MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
LLAMA_MODEL="${LLAMA_MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
QWEN_TEMPLATE="${QWEN_TEMPLATE:-${ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
LLAMA_TEMPLATE="${LLAMA_TEMPLATE:-${ROOT}/results/20260801_llama31_qksieve_keymse_template_gpu23/templates/llama31_global32_3domain_keymse_runtime.pt}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260804_qksieve_condres_query_closedloop_8gpu_v1}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/home/fdong/miniconda3/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=1.0
export QKSIEVE_CONDRES_FIT_STRIDE=32
export QKSIEVE_PROFILE_STAGES=1

mkdir -p "${RUN_ROOT}/logs"
cd "${ROOT}"

run_case() {
  local gpu="$1"
  local name="$2"
  local model="$3"
  local template="$4"
  local topic="$5"
  local history="$6"
  local budget="$7"
  local eval_tokens="$8"
  local seed="$9"
  local variants
  case "${METHOD_SET:-conditional}" in
    conditional)
      variants="qksieve_qmse_oas_requestlocal_valuesketch16_k${budget},qksieve_qmse_oas_requestlocal_condres8_k${budget},qksieve_qmse_oas_requestlocal_condres8query_k${budget},qksieve_qmse_oas_requestlocal_condres8safequery_k${budget}"
      ;;
    robust)
      variants="qksieve_qmse_oas_requestlocal_dualmass975_k${budget}"
      ;;
    robust_diag)
      variants="qksieve_qmse_oas_requestlocal_dualmass975_diag_k${budget}"
      ;;
    robust_compare)
      variants="qksieve_qmse_oas_requestlocal_dualmass975_k${budget},qksieve_qmse_oas_requestlocal_dualmass975_diag_k${budget}"
      ;;
    valuesketch8)
      variants="qksieve_qmse_oas_requestlocal_valuesketch8_k${budget}"
      ;;
    valuesketch16)
      variants="qksieve_qmse_oas_requestlocal_valuesketch16_k${budget}"
      ;;
    valuesketch12)
      variants="qksieve_qmse_oas_requestlocal_valuesketch12_k${budget}"
      ;;
    valuesketch_compare)
      variants="qksieve_qmse_oas_requestlocal_valuesketch16_k${budget},qksieve_qmse_oas_requestlocal_valuesketch12_k${budget},qksieve_qmse_oas_requestlocal_valuesketch8_k${budget}"
      ;;
    *)
      echo "unsupported METHOD_SET=${METHOD_SET}" >&2
      return 2
      ;;
  esac

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    src/run_qksieve_coldskip_longcontext_quality_20260730.py \
    --model_name_or_path "${model}" \
    --template "${template}" \
    --output_dir "${RUN_ROOT}/${name}" \
    --history_tokens "${history}" \
    --stream_reference_history_tokens "${history}" \
    --eval_tokens "${eval_tokens}" \
    --topic "${topic}" \
    --seed "${seed}" \
    --repeat_topic_stream_if_short \
    --prefill_chunk_tokens 1024 \
    --protect_recent_tokens 0 \
    --dataset_cache_dir "${DATASET_CACHE_DIR}" \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    --max_memory_per_gpu_gib 22 \
    --variants "${variants}" \
    >"${RUN_ROOT}/logs/${name}.log" 2>&1
  touch "${RUN_ROOT}/${name}_COMPLETE"
}

eval_tokens="${EVAL_TOKENS:-4}"
run_case 0 qwen_medicine4k "${QWEN_MODEL}" "${QWEN_TEMPLATE}" medicine 4096 256 "${eval_tokens}" 20261401 & p0=$!
run_case 1 qwen_sports4k "${QWEN_MODEL}" "${QWEN_TEMPLATE}" sports 4096 256 "${eval_tokens}" 20261402 & p1=$!
run_case 2 qwen_medicine32k "${QWEN_MODEL}" "${QWEN_TEMPLATE}" medicine 32768 1280 "${eval_tokens}" 20261403 & p2=$!
run_case 3 qwen_sports32k "${QWEN_MODEL}" "${QWEN_TEMPLATE}" sports 32768 1280 "${eval_tokens}" 20261404 & p3=$!
run_case 4 llama_religion4k "${LLAMA_MODEL}" "${LLAMA_TEMPLATE}" religion 4096 256 "${eval_tokens}" 20261405 & p4=$!
run_case 5 llama_computer32k "${LLAMA_MODEL}" "${LLAMA_TEMPLATE}" computer 32768 1280 "${eval_tokens}" 20261406 & p5=$!
run_case 6 qwen_religion32k "${QWEN_MODEL}" "${QWEN_TEMPLATE}" religion 32768 1280 "${eval_tokens}" 20261407 & p6=$!
run_case 7 qwen_computer32k "${QWEN_MODEL}" "${QWEN_TEMPLATE}" computer 32768 1280 "${eval_tokens}" 20261408 & p7=$!

failed=0
for pid in "${p0}" "${p1}" "${p2}" "${p3}" "${p4}" "${p5}" "${p6}" "${p7}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  touch "${RUN_ROOT}/FAILED"
  exit 1
fi
touch "${RUN_ROOT}/ALL_COMPLETE"
