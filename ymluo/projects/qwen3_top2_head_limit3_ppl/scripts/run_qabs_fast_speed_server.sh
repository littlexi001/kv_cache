#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_VISION="${TRANSFORMERS_NO_VISION:-1}"

PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_DIR="$(cd "$(dirname "${PYTHON_BIN}")" 2>/dev/null && pwd || true)"
if [[ -n "${PYTHON_DIR}" && -x "${PYTHON_DIR}/ninja" ]]; then
  export PATH="${PYTHON_DIR}:${PATH}"
fi
MODEL_PATH="${MODEL_PATH:-/mnt/workspace/Qwen3-0.6B}"
TEXT_PATH="${TEXT_PATH:-/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/qabs_fast_speed_server}"

PREFILL_LENGTHS="${PREFILL_LENGTHS:-10000 20000 40000 60000}"
EVAL_TOKENS="${EVAL_TOKENS:-1000}"
CHUNK_SIZE="${CHUNK_SIZE:-8}"
EVAL_CHUNK_SIZE="${EVAL_CHUNK_SIZE:-1}"
MODES="${MODES:-qabs8cand3attn,baseline}"

TOP_FRACTION="${TOP_FRACTION:-0.02}"
PROTECT_SINK_TOKENS="${PROTECT_SINK_TOKENS:-10}"
PROTECT_RECENT_TOKENS="${PROTECT_RECENT_TOKENS:-10}"

MAX_CHARS="${MAX_CHARS:-80000000}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
LOG_EVERY="${LOG_EVERY:-100}"
QABS_CUDA_FINAL_KERNEL="${QABS_CUDA_FINAL_KERNEL:-false}"
QABS_CUDA_CANDIDATE_KERNEL="${QABS_CUDA_CANDIDATE_KERNEL:-true}"
QABS_PROFILE="${QABS_PROFILE:-false}"
REUSE_PREFILL_CACHE="${REUSE_PREFILL_CACHE:-true}"
BASELINE_LAST="${BASELINE_LAST:-true}"

mkdir -p "${OUTPUT_ROOT}"

if [[ ! -f "${TEXT_PATH}" ]]; then
  echo "Eval text not found: ${TEXT_PATH}" >&2
  echo "Set TEXT_PATH=/path/to/a/long/text/file." >&2
  exit 1
fi

if [[ "${QABS_CUDA_FINAL_KERNEL}" == "true" ]]; then
  if ! "${PYTHON_BIN}" -c 'import torch.utils.cpp_extension as ce; raise SystemExit(0 if ce.is_ninja_available() else 1)' >/dev/null 2>&1; then
    echo "warning: QABS_CUDA_FINAL_KERNEL=true but ninja is unavailable; eval will fall back to the PyTorch qabs path." >&2
  fi
fi

for prefill_tokens in ${PREFILL_LENGTHS}; do
  out_dir="${OUTPUT_ROOT}/prefill${prefill_tokens}_eval${EVAL_TOKENS}"
  echo "=== qabs fast speed prefill=${prefill_tokens} eval=${EVAL_TOKENS} prefill_chunk=${CHUNK_SIZE} eval_chunk=${EVAL_CHUNK_SIZE} ==="
  "${PYTHON_BIN}" "${PROJECT_DIR}/src/evaluate_qwen3_top2_head_limit3_ppl.py" \
    --model_name_or_path "${MODEL_PATH}" \
    --text_path "${TEXT_PATH}" \
    --output_dir "${out_dir}" \
    --prefill_tokens "${prefill_tokens}" \
    --eval_tokens "${EVAL_TOKENS}" \
    --chunk_size "${CHUNK_SIZE}" \
    --eval_chunk_size "${EVAL_CHUNK_SIZE}" \
    --max_chars "${MAX_CHARS}" \
    --add_special_tokens false \
    --append_eos false \
    --require_total_tokens true \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --device_map "${DEVICE_MAP}" \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    --top_fraction "${TOP_FRACTION}" \
    --protect_sink_tokens "${PROTECT_SINK_TOKENS}" \
    --protect_recent_tokens "${PROTECT_RECENT_TOKENS}" \
    --always_keep_self true \
    --modes "${MODES}" \
    --qabs_fast_path true \
    --qabs_cuda_final_kernel "${QABS_CUDA_FINAL_KERNEL}" \
    --qabs_cuda_candidate_kernel "${QABS_CUDA_CANDIDATE_KERNEL}" \
    --qabs_profile "${QABS_PROFILE}" \
    --reuse_prefill_cache "${REUSE_PREFILL_CACHE}" \
    --baseline_last "${BASELINE_LAST}" \
    --disable_sparse_stats true \
    --log_every "${LOG_EVERY}" \
    --make_plots false
done
