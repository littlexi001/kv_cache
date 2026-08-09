#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PY="${PY:-python}"
MODEL="${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_DIR}/outputs/top2_coselection_${STAMP}}"

WAR_TEXT="${WAR_TEXT:-${PROJECT_DIR}/../qwen3_top2_head_limit3_ppl/data/war_and_peace_pg2600.txt}"
MONTE_TEXT="${MONTE_TEXT:-${PROJECT_DIR}/../qwen3_top2_head_limit3_ppl/data/count_monte_cristo_pg1184.txt}"
DATASETS="${DATASETS:-war,monte}"

mkdir -p "$OUT_ROOT"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

run_one() {
  local label="$1"
  local text_path="$2"
  local out_dir="${OUT_ROOT}/${label}"
  mkdir -p "$out_dir"
  "$PY" -u "$PROJECT_DIR/src/run_top2_coselection.py" \
    --model_name_or_path "$MODEL" \
    --text_path "$text_path" \
    --output_dir "$out_dir" \
    --context_tokens "${CONTEXT_TOKENS:-1024}" \
    --eval_tokens "${EVAL_TOKENS:-512}" \
    --text_token_offset "${TEXT_TOKEN_OFFSET:-0}" \
    --chunk_size "${CHUNK_SIZE:-64}" \
    --query_stride "${QUERY_STRIDE:-1}" \
    --max_query_samples "${MAX_QUERY_SAMPLES:-0}" \
    --dtype "${DTYPE:-float16}" \
    --device cuda \
    --device_map "${DEVICE_MAP:-auto}" \
    --analysis_device "${ANALYSIS_DEVICE:-cuda}" \
    --attn_implementation eager \
    --top_fraction "${TOP_FRACTION:-0.02}" \
    --layers "${LAYERS:-all}" \
    --heads "${HEADS:-all}" \
    --min_token_count "${MIN_TOKEN_COUNT:-8}" \
    --min_pair_count "${MIN_PAIR_COUNT:-4}" \
    --fdr_alpha "${FDR_ALPHA:-0.01}" \
    --top_pairs_per_head "${TOP_PAIRS_PER_HEAD:-30}" \
    --representative_heads "${REPRESENTATIVE_HEADS:-6}" \
    --heatmap_tokens "${HEATMAP_TOKENS:-128}" \
    --save_selection_indices true \
    --make_plots true \
    2>&1 | tee "$out_dir/run.log"
}

IFS=',' read -r -a dataset_list <<< "$DATASETS"
for dataset in "${dataset_list[@]}"; do
  case "$dataset" in
    war) run_one war "$WAR_TEXT" ;;
    monte) run_one monte "$MONTE_TEXT" ;;
    *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
  esac
done

echo "[top2-coselection] done: $OUT_ROOT"
