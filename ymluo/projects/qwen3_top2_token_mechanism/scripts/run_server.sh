#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PY="${PY:-python}"
MODEL="${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}"
TEXT="${TEXT:-${PROJECT_DIR}/../qwen3_top2_head_limit3_ppl/data/war_and_peace_pg2600.txt}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/outputs/top2_token_mechanism_${STAMP}}"

mkdir -p "$OUT_DIR"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$PY" -u "$PROJECT_DIR/src/run_selector_ppl.py" \
  --model_name_or_path "$MODEL" \
  --text_path "$TEXT" \
  --output_dir "$OUT_DIR" \
  --prefill_tokens "${PREFILL_TOKENS:-4096}" \
  --eval_tokens "${EVAL_TOKENS:-512}" \
  --chunk_size "${CHUNK_SIZE:-64}" \
  --dtype "${DTYPE:-float16}" \
  --device cuda \
  --device_map "${DEVICE_MAP:-auto}" \
  --attn_implementation eager \
  --ratio_grid "${RATIO_GRID:-0.001,0.005,0.01,0.02,0.04,0.08,0.16,0.32,1.0}" \
  --target_ratio "${TARGET_RATIO:-0.02}" \
  --control_selectors "${CONTROL_SELECTORS:-sink_recent_s0,sink_recent_s1,sink_recent_s2,sink_recent_s4,sink_recent_s8,sink_recent_s16,recent,sink,random,bottom_attention,top_attention_drop_sink,top_attention_drop_recent,top_attention_drop_remote}" \
  --diagnostic_sink_sweep "${DIAGNOSTIC_SINK_SWEEP:-0,1,2,4,8,16,32}" \
  --role_sink_tokens "${ROLE_SINK_TOKENS:-4}" \
  --role_recent_tokens "${ROLE_RECENT_TOKENS:-256}" \
  --always_keep_self true \
  --collect_diagnostics true \
  --write_token_nll true \
  --make_plots true \
  2>&1 | tee "$OUT_DIR/run.log"

"$PY" "$PROJECT_DIR/src/analyze_diagnostics.py" \
  --run_dir "$OUT_DIR"

"$PY" "$PROJECT_DIR/src/compare_selectors.py" \
  --run_dir "$OUT_DIR" \
  --target_ratio "${TARGET_RATIO:-0.02}" \
  --equivalence_nll_margin "${EQUIVALENCE_NLL_MARGIN:-0.01}" \
  --equivalence_ppl_relative_margin "${EQUIVALENCE_PPL_MARGIN:-0.01}" \
  --block_size "${BOOTSTRAP_BLOCK_SIZE:-64}" \
  --bootstrap_repetitions "${BOOTSTRAP_REPETITIONS:-2000}" \
  --make_plot

echo "[top2-mechanism] done: $OUT_DIR"

