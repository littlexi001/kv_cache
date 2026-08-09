#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027/experiments/frozen_c64_20260807}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
MODEL="${MODEL:-/home/fdong/qksieve_iclr2027/models/Qwen3-4B-Instruct-2507}"
HISTORY="${HISTORY:-4096}"
EVAL_TOKENS="${EVAL_TOKENS:-2}"
GPU="${GPU:-0}"
SEED="${SEED:-20260844}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/blockcondres_model_ppl_smoke_20260807}"
OUTPUT="${RUN_ROOT}/n${HISTORY}_seed${SEED}"
PREFILL_CHUNK="${PREFILL_CHUNK:-512}"
VARIANTS="${VARIANTS:-qksieve_qmse_oas_requestlocal_blockcondres8_r8_m8_k1120}"

export PATH="/home/fdong/miniconda3/envs/nanogpt/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export QKSIEVE_QK_FACTOR_SOLVER=legacy
export QKSIEVE_QMSE_RATE_ALLOCATOR=torch
export QKSIEVE_PRELOAD_EXTENSIONS=1
export QKSIEVE_PRELOAD_QMSE_RATE_TABLES=1
export QKSIEVE_BATCH_QMSE_ALLOCATION=1
export QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=1.0
export QKSIEVE_BLOCK_CONDRES_FIT_STRIDE=32
export QKSIEVE_BLOCK_CONDRES_CHUNK=4096
export QKSIEVE_DEBUG_FINITE=1
unset QKSIEVE_BUILD_RESIDENT_KEY_FACTORS || true
unset QKSIEVE_BUILD_RESIDENT_VALUE_SKETCH || true
unset QKSIEVE_DEBUG_DISABLE_VALUE_SKETCH || true
unset QKSIEVE_PROFILE_STAGES || true

mkdir -p "${RUN_ROOT}/logs" "${OUTPUT}"
rm -f "${OUTPUT}/ALL_COMPLETE" "${OUTPUT}/FAILED"

{
  echo "purpose=model-level quality gate for rank8/top1120 INT8 block conditional residual"
  echo "history=${HISTORY}"
  echo "eval_tokens=${EVAL_TOKENS}"
  echo "gpu=${GPU}"
  echo "seed=${SEED}"
  echo "variants=${VARIANTS}"
  sha256sum \
    "${ROOT}/src/run_head_top2_targeted_ppl_20260714.py" \
    "${ROOT}/src/run_direct_countcap_denseprompt_ppl_20260725.py" \
    "${ROOT}/src/run_qksieve_coldskip_longcontext_quality_20260730.py"
} >"${OUTPUT}/manifest.txt"

set +e
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u \
  "${ROOT}/src/run_qksieve_coldskip_longcontext_quality_20260730.py" \
  --model_name_or_path "${MODEL}" \
  --template "${ROOT}/nonexistent_requestlocal_template.pt" \
  --output_dir "${OUTPUT}" \
  --history_tokens "${HISTORY}" \
  --stream_reference_history_tokens "${HISTORY}" \
  --eval_tokens "${EVAL_TOKENS}" \
  --text_file "${ROOT}/src/run_head_top2_targeted_ppl_20260714.py" \
  --repeat_topic_stream_if_short \
  --prefill_chunk_tokens "${PREFILL_CHUNK}" \
  --protect_recent_tokens 0 \
  --dataset_cache_dir "${ROOT}/datasets" \
  --seed "${SEED}" \
  --dtype float16 \
  --device cuda \
  --device_map auto \
  --max_memory_per_gpu_gib 22 \
  --variants "${VARIANTS}" \
  >"${RUN_ROOT}/logs/n${HISTORY}_seed${SEED}.log" 2>&1
status=$?
set -e

if [[ "${status}" -ne 0 ]]; then
  touch "${OUTPUT}/FAILED"
  exit "${status}"
fi
touch "${OUTPUT}/ALL_COMPLETE"
