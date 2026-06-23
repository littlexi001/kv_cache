#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/../../.." && pwd)"

export TOKENIZERS_PARALLELISM=false

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/mnt/workspace/Qwen3-0.6B}"
HELDOUT_DATASET="${HELDOUT_DATASET:-wikitext103_validation}"
HELDOUT_TEXT_DIR="${HELDOUT_TEXT_DIR:-${REPO_DIR}/ymluo/projects/qwen3_routed_top4_mha_pretrain/output/heldout_text}"
TEXT_PATH="${TEXT_PATH:-/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs}"

PREFILL_LENGTHS="${PREFILL_LENGTHS:-10000 20000 40000 60000}"
QUALITY_MODES="${QUALITY_MODES:-baseline,top2union,top2,obstop2fullnonmasst80kn2mn1}"
SPEED_MODES="${SPEED_MODES:-baseline,top2union,top2,obstop2fullnonmasst80kn2mn1}"

TOP_FRACTION="${TOP_FRACTION:-0.02}"
OBS_WINDOW_TOKENS="${OBS_WINDOW_TOKENS:-1000}"
PROTECT_SINK_TOKENS="${PROTECT_SINK_TOKENS:-1000}"
PROTECT_RECENT_TOKENS="${PROTECT_RECENT_TOKENS:-1000}"
OBS_RECENT_TOKENS="${OBS_RECENT_TOKENS:-${PROTECT_RECENT_TOKENS}}"

QUALITY_CHUNK_SIZE="${QUALITY_CHUNK_SIZE:-8}"
QUALITY_EVAL_TOKENS="${QUALITY_EVAL_TOKENS:-1000}"
SPEED_EVAL_TOKENS="${SPEED_EVAL_TOKENS:-64}"
SPEED_CHUNK_SIZE="${SPEED_CHUNK_SIZE:-1}"

MAX_CHARS="${MAX_CHARS:-80000000}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
MAKE_PLOTS="${MAKE_PLOTS:-false}"
RUN_QUALITY="${RUN_QUALITY:-true}"
RUN_SPEED="${RUN_SPEED:-true}"

mkdir -p "${OUTPUT_ROOT}"

if [[ ! -f "${TEXT_PATH}" ]]; then
  echo "Long-context eval text not found: ${TEXT_PATH}" >&2
  echo "Prepare it first, for example:" >&2
  echo "  cd ${REPO_DIR}/ymluo/projects/qwen3_routed_top4_mha_pretrain" >&2
  echo "  HELDOUT_MAX_CHARS=${MAX_CHARS} bash scripts/prepare_heldout_ppl_text.sh" >&2
  echo "Or set TEXT_PATH=/path/to/a/long/text/file." >&2
  exit 1
fi

run_eval() {
  local total_tokens="$1"
  local prefill_tokens="$2"
  local eval_tokens="$3"
  local chunk_size="$4"
  local modes="$5"
  local output_dir="$6"

  "${PYTHON_BIN}" "${PROJECT_DIR}/src/evaluate_qwen3_top2_head_limit3_ppl.py" \
    --model_name_or_path "${MODEL_PATH}" \
    --text_path "${TEXT_PATH}" \
    --output_dir "${output_dir}" \
    --prefill_tokens "${prefill_tokens}" \
    --eval_tokens "${eval_tokens}" \
    --chunk_size "${chunk_size}" \
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
    --obs_window_tokens "${OBS_WINDOW_TOKENS}" \
    --obs_recent_tokens "${OBS_RECENT_TOKENS}" \
    --obs_fallback_all true \
    --always_keep_self true \
    --modes "${modes}" \
    --make_plots "${MAKE_PLOTS}"
}

for prefill_tokens in ${PREFILL_LENGTHS}; do
  eval_tokens="${QUALITY_EVAL_TOKENS}"
  total_tokens=$((prefill_tokens + eval_tokens))

  if [[ "${RUN_QUALITY}" == "true" ]]; then
    out_dir="${OUTPUT_ROOT}/quality_prefill${prefill_tokens}_eval${eval_tokens}"
    echo "=== quality total=${total_tokens} prefill=${prefill_tokens} eval=${eval_tokens} chunk=${QUALITY_CHUNK_SIZE} ==="
    run_eval "${total_tokens}" "${prefill_tokens}" "${eval_tokens}" "${QUALITY_CHUNK_SIZE}" "${QUALITY_MODES}" "${out_dir}"
  fi

  if [[ "${RUN_SPEED}" == "true" ]]; then
    speed_eval="${SPEED_EVAL_TOKENS}"
    speed_total=$((prefill_tokens + speed_eval))
    out_dir="${OUTPUT_ROOT}/speed_prefill${prefill_tokens}_eval${speed_eval}_chunk1"
    echo "=== speed total=${speed_total} prefill=${prefill_tokens} eval=${speed_eval} chunk=${SPEED_CHUNK_SIZE} ==="
    run_eval "${speed_total}" "${prefill_tokens}" "${speed_eval}" "${SPEED_CHUNK_SIZE}" "${SPEED_MODES}" "${out_dir}"
  fi
done
