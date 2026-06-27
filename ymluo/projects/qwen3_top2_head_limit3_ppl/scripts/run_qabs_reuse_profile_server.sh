#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TOKENIZERS_PARALLELISM=false

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/mnt/workspace/Qwen3-0.6B}"
TEXT_PATH="${TEXT_PATH:-/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/qabs_reuse_profile_server}"

PREFILL_TOKENS="${PREFILL_TOKENS:-10000}"
EVAL_TOKENS="${EVAL_TOKENS:-128}"
CHUNK_SIZE="${CHUNK_SIZE:-8}"
EVAL_CHUNK_SIZE="${EVAL_CHUNK_SIZE:-1}"
MODES="${MODES:-qabs8cand3reuse,baseline}"
CANDIDATE_SELECTIONS="${CANDIDATE_SELECTIONS:-topk sample_quantile previous_threshold}"
QABS_THRESHOLD_SAMPLE_SIZE="${QABS_THRESHOLD_SAMPLE_SIZE:-256}"

TOP_FRACTION="${TOP_FRACTION:-0.02}"
PROTECT_SINK_TOKENS="${PROTECT_SINK_TOKENS:-10}"
PROTECT_RECENT_TOKENS="${PROTECT_RECENT_TOKENS:-10}"

MAX_CHARS="${MAX_CHARS:-80000000}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
LOG_EVERY="${LOG_EVERY:-16}"
QABS_CUDA_FINAL_KERNEL="${QABS_CUDA_FINAL_KERNEL:-true}"
QABS_CUDA_CANDIDATE_KERNEL="${QABS_CUDA_CANDIDATE_KERNEL:-true}"
QABS_CUDA_REUSE_SELECT_KERNEL="${QABS_CUDA_REUSE_SELECT_KERNEL:-false}"
REUSE_PREFILL_CACHE="${REUSE_PREFILL_CACHE:-true}"
BASELINE_LAST="${BASELINE_LAST:-true}"

mkdir -p "${OUTPUT_ROOT}"

if [[ ! -f "${TEXT_PATH}" ]]; then
  echo "Eval text not found: ${TEXT_PATH}" >&2
  exit 1
fi

for selection in ${CANDIDATE_SELECTIONS}; do
  out_dir="${OUTPUT_ROOT}/prefill${PREFILL_TOKENS}_eval${EVAL_TOKENS}_${selection}"
  echo "=== qabs reuse profile selection=${selection} prefill=${PREFILL_TOKENS} eval=${EVAL_TOKENS} ==="
  "${PYTHON_BIN}" "${PROJECT_DIR}/src/evaluate_qwen3_top2_head_limit3_ppl.py"     --model_name_or_path "${MODEL_PATH}"     --text_path "${TEXT_PATH}"     --output_dir "${out_dir}"     --prefill_tokens "${PREFILL_TOKENS}"     --eval_tokens "${EVAL_TOKENS}"     --chunk_size "${CHUNK_SIZE}"     --eval_chunk_size "${EVAL_CHUNK_SIZE}"     --max_chars "${MAX_CHARS}"     --add_special_tokens false     --append_eos false     --require_total_tokens true     --dtype "${DTYPE}"     --device "${DEVICE}"     --device_map "${DEVICE_MAP}"     --attn_implementation "${ATTN_IMPLEMENTATION}"     --top_fraction "${TOP_FRACTION}"     --protect_sink_tokens "${PROTECT_SINK_TOKENS}"     --protect_recent_tokens "${PROTECT_RECENT_TOKENS}"     --always_keep_self true     --modes "${MODES}"     --qabs_fast_path true     --qabs_cuda_final_kernel "${QABS_CUDA_FINAL_KERNEL}"     --qabs_cuda_candidate_kernel "${QABS_CUDA_CANDIDATE_KERNEL}"     --qabs_cuda_reuse_select_kernel "${QABS_CUDA_REUSE_SELECT_KERNEL}"     --qabs_profile true     --qabs_candidate_selection "${selection}"     --qabs_threshold_sample_size "${QABS_THRESHOLD_SAMPLE_SIZE}"     --reuse_prefill_cache "${REUSE_PREFILL_CACHE}"     --baseline_last "${BASELINE_LAST}"     --disable_sparse_stats true     --log_every "${LOG_EVERY}"     --make_plots false

done
