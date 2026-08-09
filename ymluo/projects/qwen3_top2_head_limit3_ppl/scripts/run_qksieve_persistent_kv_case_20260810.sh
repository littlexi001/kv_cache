#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027}"
SRC_ROOT="${SRC_ROOT:-${ROOT}/src}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260810_qksieve_persistent_kv_v1}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
MODEL="${MODEL:-${ROOT}/models/Yarn-Llama-2-7b-128k}"
TEXT_FILE="${TEXT_FILE:-${ROOT}/src/run_head_top2_targeted_ppl_pre_indexopt_20260807.py}"
METHOD="${METHOD:-qksieve_robust}"
HISTORY_TOKENS="${HISTORY_TOKENS:-32768}"
BRANCH_COUNT="${BRANCH_COUNT:-4}"
BRANCH_STEPS="${BRANCH_STEPS:-32}"
APPEND_STEPS="${APPEND_STEPS:-128}"
SEED="${SEED:-20260810}"
GPU_TAG="${GPU_TAG:-${CUDA_VISIBLE_DEVICES:-0}}"

export PATH="$(dirname "${PYTHON}"):/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${SRC_ROOT}:${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export QKSIEVE_QK_FACTOR_SOLVER=legacy
export QKSIEVE_QMSE_RATE_ALLOCATOR=torch
export QKSIEVE_PRELOAD_QMSE_RATE_TABLES=1
export QKSIEVE_PRELOAD_EXTENSIONS=1
export QKSIEVE_PARALLEL_QK_WORKERS="${QKSIEVE_PARALLEL_QK_WORKERS:-12}"
export QKSIEVE_PARALLEL_VALUE_WORKERS="${QKSIEVE_PARALLEL_VALUE_WORKERS:-12}"
export QKSIEVE_MAX_QUANTILE_SAMPLE_COUNT=512
export QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=0.5
export QKSIEVE_TRUST_REMOTE_CODE=0
unset QKSIEVE_PROFILE_STAGES || true

OUT_DIR="${RUN_ROOT}/n${HISTORY_TOKENS}/seed${SEED}"
OUT_JSON="${OUT_DIR}/${METHOD}.json"
TMP_JSON="${OUT_JSON}.tmp.$$"
LOG="${RUN_ROOT}/logs/n${HISTORY_TOKENS}_seed${SEED}_${METHOD}_gpu${GPU_TAG//,/-}.log"
mkdir -p "${OUT_DIR}" "${RUN_ROOT}/logs"

if [[ -s "${OUT_JSON}" ]]; then
  echo "Already complete: ${OUT_JSON}"
  exit 0
fi

"${PYTHON}" -u "${SRC_ROOT}/benchmark_qksieve_persistent_kv_20260810.py" \
  --model_name_or_path "${MODEL}" \
  --text_file "${TEXT_FILE}" \
  --output "${TMP_JSON}" \
  --method "${METHOD}" \
  --history_tokens "${HISTORY_TOKENS}" \
  --branch_count "${BRANCH_COUNT}" \
  --branch_steps "${BRANCH_STEPS}" \
  --append_steps "${APPEND_STEPS}" \
  --prefill_chunk_tokens 1024 \
  --dtype float16 \
  --device cuda \
  --device_map auto \
  --max_memory_per_gpu_gib 22 \
  --original_max_position_embeddings 4096 \
  --global_max_position 131072 \
  --seed "${SEED}" \
  >"${LOG}" 2>&1

mv "${TMP_JSON}" "${OUT_JSON}"
echo "Completed: ${OUT_JSON}"
