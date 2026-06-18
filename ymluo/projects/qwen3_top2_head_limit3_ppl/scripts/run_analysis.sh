#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/../../.." && pwd)"

export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-${REPO_DIR}/ymluo/models/Qwen3-0.6B}"
TEXT_PATH="${TEXT_PATH:-${REPO_DIR}/external/needle-in-a-haystack/needlehaystack/PaulGrahamEssays/worked.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/top2_head_limit3_ppl}"

python "${PROJECT_DIR}/src/evaluate_qwen3_top2_head_limit3_ppl.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --text_path "${TEXT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --prefill_tokens "${PREFILL_TOKENS:-1024}" \
  --eval_tokens "${EVAL_TOKENS:-512}" \
  --chunk_size "${CHUNK_SIZE:-128}" \
  --max_chars "${MAX_CHARS:-8000000}" \
  --add_special_tokens "${ADD_SPECIAL_TOKENS:-false}" \
  --append_eos "${APPEND_EOS:-false}" \
  --require_total_tokens "${REQUIRE_TOTAL_TOKENS:-true}" \
  --dtype "${DTYPE:-bfloat16}" \
  --device "${DEVICE:-cuda}" \
  --device_map "${DEVICE_MAP:-auto}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-eager}" \
  --top_fraction "${TOP_FRACTION:-0.02}" \
  --max_heads_per_token "${MAX_HEADS_PER_TOKEN:-3}" \
  --always_keep_self "${ALWAYS_KEEP_SELF:-true}" \
  --seed "${SEED:-1234}" \
  --modes "${MODES:-baseline,top2,top2limit3score}" \
  --make_plots "${MAKE_PLOTS:-true}" \
  --plot_dpi "${PLOT_DPI:-180}"
