#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
TEMPLATE="${TEMPLATE:-${ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260804_qksieve_condres_vs_robust_8gpu_v1}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
# Request-local QK bases solve many batched 128x128 CPU matrices.  Letting
# every GPU worker spawn a full BLAS thread pool causes severe oversubscription
# when all eight GPUs run concurrently.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=1.0
export QKSIEVE_CONDRES_FIT_STRIDE=32
export QKSIEVE_PROFILE_STAGES=1

mkdir -p "${RUN_ROOT}/logs"
cd "${ROOT}"

run_case() {
  local gpus="$1"
  local topic="$2"
  local history="$3"
  local budget="$4"
  local eval_tokens="$5"
  local seed="$6"
  local name="${topic}_${history}"
  local variants
  case "${METHOD_SET:-all}" in
    all)
      variants="qksieve_qmse_oas_requestlocal_dualmass975_k${budget},qksieve_qmse_oas_requestlocal_valuesketch16_k${budget},qksieve_qmse_oas_requestlocal_condres8_k${budget},qksieve_qmse_oas_requestlocal_condres8wiener_k${budget}"
      ;;
    condres)
      variants="qksieve_qmse_oas_requestlocal_valuesketch16_k${budget},qksieve_qmse_oas_requestlocal_condres8_k${budget},qksieve_qmse_oas_requestlocal_condres8wiener_k${budget}"
      ;;
    robust)
      variants="qksieve_qmse_oas_requestlocal_dualmass975_k${budget}"
      ;;
    *)
      echo "unsupported METHOD_SET=${METHOD_SET}" >&2
      return 2
      ;;
  esac
  if [[ "${SKIP_EXACT_ORACLE:-0}" != "1" ]]; then
    variants="exact_qk_oracle_k${budget},${variants}"
  fi

  CUDA_VISIBLE_DEVICES="${gpus}" "${PYTHON}" -u \
    src/run_qksieve_coldskip_longcontext_quality_20260730.py \
    --model_name_or_path "${MODEL}" \
    --template "${TEMPLATE}" \
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

run_case 0 medicine 4096 256 "${EVAL_4K:-16}" 20261201 & p0=$!
run_case 1 sports 4096 256 "${EVAL_4K:-16}" 20261202 & p1=$!
run_case 2 medicine 32768 1280 "${EVAL_32K:-12}" 20261203 & p2=$!
run_case 3 sports 32768 1280 "${EVAL_32K:-12}" 20261204 & p3=$!
run_case 4,5 medicine 96000 1280 "${EVAL_96K:-8}" 20261205 & p4=$!
run_case 6,7 sports 96000 1280 "${EVAL_96K:-8}" 20261206 & p5=$!

failed=0
for pid in "${p0}" "${p1}" "${p2}" "${p3}" "${p4}" "${p5}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  touch "${RUN_ROOT}/FAILED"
  exit 1
fi
touch "${RUN_ROOT}/ALL_COMPLETE"
