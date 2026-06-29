#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/../../.." && pwd)"

export TOKENIZERS_PARALLELISM=false

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/mnt/workspace/Qwen3-0.6B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs}"
MODES="${MODES:-baseline,qabs8cand3reuse,sparqfast8cand3}"

PREFILL_TOKENS="${PREFILL_TOKENS:-4096}"
EVAL_TOKENS="${EVAL_TOKENS:-512}"
CHUNK_SIZE="${CHUNK_SIZE:-8}"
EVAL_CHUNK_SIZE="${EVAL_CHUNK_SIZE:-1}"
MAX_NEEDLE_CASES="${MAX_NEEDLE_CASES:-12}"
NEEDLE_PREFILL_TOKENS="${NEEDLE_PREFILL_TOKENS:-0}"
NEEDLE_EVAL_TOKENS="${NEEDLE_EVAL_TOKENS:-0}"

TOP_FRACTION="${TOP_FRACTION:-0.02}"
PROTECT_SINK_TOKENS="${PROTECT_SINK_TOKENS:-10}"
PROTECT_RECENT_TOKENS="${PROTECT_RECENT_TOKENS:-10}"

DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
QABS_CUDA_FINAL_KERNEL="${QABS_CUDA_FINAL_KERNEL:-true}"
QABS_CUDA_CANDIDATE_KERNEL="${QABS_CUDA_CANDIDATE_KERNEL:-false}"
QABS_CUDA_REUSE_SELECT_KERNEL="${QABS_CUDA_REUSE_SELECT_KERNEL:-false}"

RUN_NEEDLE_GENERATION="${RUN_NEEDLE_GENERATION:-false}"
NEEDLE_GENERATION_MAX_CASES="${NEEDLE_GENERATION_MAX_CASES:-12}"
NEEDLE_GENERATION_LENGTHS="${NEEDLE_GENERATION_LENGTHS:-1000,2000,4000,8000}"
NEEDLE_GENERATION_DEPTHS="${NEEDLE_GENERATION_DEPTHS:-0,25,50,75,100}"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" "${PROJECT_DIR}/src/build_topic_texts.py"

"${PYTHON_BIN}" "${PROJECT_DIR}/src/run_quality_suite.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --output_root "${OUTPUT_ROOT}" \
  --modes "${MODES}" \
  --prefill_tokens "${PREFILL_TOKENS}" \
  --eval_tokens "${EVAL_TOKENS}" \
  --chunk_size "${CHUNK_SIZE}" \
  --eval_chunk_size "${EVAL_CHUNK_SIZE}" \
  --max_needle_cases "${MAX_NEEDLE_CASES}" \
  --needle_prefill_tokens "${NEEDLE_PREFILL_TOKENS}" \
  --needle_eval_tokens "${NEEDLE_EVAL_TOKENS}" \
  --dtype "${DTYPE}" \
  --device "${DEVICE}" \
  --device_map "${DEVICE_MAP}" \
  --attn_implementation "${ATTN_IMPLEMENTATION}" \
  --top_fraction "${TOP_FRACTION}" \
  --protect_sink_tokens "${PROTECT_SINK_TOKENS}" \
  --protect_recent_tokens "${PROTECT_RECENT_TOKENS}" \
  --qabs_cuda_final_kernel "${QABS_CUDA_FINAL_KERNEL}" \
  --qabs_cuda_candidate_kernel "${QABS_CUDA_CANDIDATE_KERNEL}" \
  --qabs_cuda_reuse_select_kernel "${QABS_CUDA_REUSE_SELECT_KERNEL}" \
  --make_plots false

if [[ "${RUN_NEEDLE_GENERATION}" == "true" ]]; then
  "${PYTHON_BIN}" "${PROJECT_DIR}/src/evaluate_needle_generation.py" \
    --model_name_or_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_ROOT}/needle_generation" \
    --modes "${MODES}" \
    --lengths "${NEEDLE_GENERATION_LENGTHS}" \
    --depths "${NEEDLE_GENERATION_DEPTHS}" \
    --max_cases "${NEEDLE_GENERATION_MAX_CASES}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --device_map "${DEVICE_MAP}" \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    --top_fraction "${TOP_FRACTION}" \
    --protect_sink_tokens "${PROTECT_SINK_TOKENS}" \
    --protect_recent_tokens "${PROTECT_RECENT_TOKENS}" \
    --qabs_cuda_final_kernel "${QABS_CUDA_FINAL_KERNEL}" \
    --qabs_cuda_candidate_kernel "${QABS_CUDA_CANDIDATE_KERNEL}" \
    --qabs_cuda_reuse_select_kernel "${QABS_CUDA_REUSE_SELECT_KERNEL}"
fi

echo "quality suite complete: ${OUTPUT_ROOT}"
