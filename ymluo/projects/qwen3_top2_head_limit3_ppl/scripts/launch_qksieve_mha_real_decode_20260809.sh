#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027}"
SRC_ROOT="${SRC_ROOT:-${ROOT}/src}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260809_qksieve_mha_real_decode}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
MODEL="${MODEL:-${ROOT}/models/longchat-7b-v1.5-32k}"
TEXT_FILE="${TEXT_FILE:-${ROOT}/src/run_head_top2_targeted_ppl_pre_indexopt_20260807.py}"
METHOD="${1:-${METHOD:-full}}"
HISTORY_TOKENS="${HISTORY_TOKENS:-8192}"
GENERATION_STEPS="${GENERATION_STEPS:-64}"
STEADY_START="${STEADY_START:-16}"
SEED="${SEED:-20260809}"
GPU_TAG="${GPU_TAG:-${CUDA_VISIBLE_DEVICES:-0}}"
MAX_MEMORY_PER_GPU_GIB="${MAX_MEMORY_PER_GPU_GIB:-22}"
GLOBAL_MAX_POSITION="${GLOBAL_MAX_POSITION:-131072}"
DTYPE="${DTYPE:-float16}"

export PATH="/home/fdong/miniconda3/envs/nanogpt/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${SRC_ROOT}:${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export QKSIEVE_QK_FACTOR_SOLVER=legacy
export QKSIEVE_QMSE_RATE_ALLOCATOR=torch
export QKSIEVE_PRELOAD_EXTENSIONS=0
export QKSIEVE_PRELOAD_QMSE_RATE_TABLES=1
export QKSIEVE_PARALLEL_QK_WORKERS="${QKSIEVE_PARALLEL_QK_WORKERS:-12}"
export QKSIEVE_MAX_QUANTILE_SAMPLE_COUNT=512
export QKSIEVE_TRUST_REMOTE_CODE="${QKSIEVE_TRUST_REMOTE_CODE:-1}"
unset QKSIEVE_PROFILE_STAGES || true

if [[ "${METHOD}" == "qksieve_valuesketch_top1280" ]]; then
  export QKSIEVE_BUILD_RESIDENT_VALUE_SKETCH=1
  export QKSIEVE_DEBUG_DISABLE_VALUE_SKETCH=0
  export QKSIEVE_VALUE_SKETCH_TAIL_ALPHA="${QKSIEVE_VALUE_SKETCH_TAIL_ALPHA:-0.5}"
  export QKSIEVE_PARALLEL_VALUE_WORKERS="${QKSIEVE_PARALLEL_VALUE_WORKERS:-12}"
  export QKSIEVE_ATTENTION_SPLIT_OVERRIDE=0
else
  export QKSIEVE_BUILD_RESIDENT_VALUE_SKETCH=0
  export QKSIEVE_DEBUG_DISABLE_VALUE_SKETCH=1
  export QKSIEVE_PARALLEL_VALUE_WORKERS=0
  export QKSIEVE_ATTENTION_SPLIT_OVERRIDE=8
fi

OUT_DIR="${RUN_ROOT}/n${HISTORY_TOKENS}/seed${SEED}"
OUT_JSON="${OUT_DIR}/${METHOD}.json"
TMP_JSON="${OUT_JSON}.tmp.$$"
LOG="${RUN_ROOT}/logs/n${HISTORY_TOKENS}_seed${SEED}_${METHOD}_gpu${GPU_TAG//,/-}.log"
mkdir -p "${OUT_DIR}" "${RUN_ROOT}/logs"

if [[ -s "${OUT_JSON}" ]]; then
  echo "Already complete: ${OUT_JSON}"
  exit 0
fi

"${PYTHON}" -u "${SRC_ROOT}/run_qksieve_fier_autoregressive_speed_20260808.py" \
  --model_name_or_path "${MODEL}" \
  --text_file "${TEXT_FILE}" \
  --output "${TMP_JSON}" \
  --method "${METHOD}" \
  --history_tokens "${HISTORY_TOKENS}" \
  --generation_steps "${GENERATION_STEPS}" \
  --steady_start "${STEADY_START}" \
  --prefill_chunk_tokens 1024 \
  --dtype "${DTYPE}" \
  --device cuda \
  --device_map auto \
  --max_memory_per_gpu_gib "${MAX_MEMORY_PER_GPU_GIB}" \
  --original_max_position_embeddings 4096 \
  --global_max_position "${GLOBAL_MAX_POSITION}" \
  --seed "${SEED}" \
  >"${LOG}" 2>&1

mv "${TMP_JSON}" "${OUT_JSON}"
echo "Completed: ${OUT_JSON}"
