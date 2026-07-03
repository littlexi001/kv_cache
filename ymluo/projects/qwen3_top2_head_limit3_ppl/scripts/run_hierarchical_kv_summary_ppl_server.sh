#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/bin/activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/src:${PYTHONPATH:-}"

OUT="${OUT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/hierarchical_kv_summary_ppl_20k512_20260703}"
MODEL="${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}"
TEXT="${TEXT:-data/war_and_peace_pg2600.txt}"

mkdir -p "$OUT"

python src/run_hierarchical_kv_summary_ppl.py \
  --model_name_or_path "$MODEL" \
  --text_path "$TEXT" \
  --output_dir "$OUT" \
  --prefill_tokens "${PREFILL_TOKENS:-20000}" \
  --eval_tokens "${EVAL_TOKENS:-512}" \
  --chunk_size "${CHUNK_SIZE:-512}" \
  --eval_chunk_size 1 \
  --modes "${MODES:-baseline,recent,hierkv}" \
  --sink_tokens "${SINK_TOKENS:-64}" \
  --recent_tokens "${RECENT_TOKENS:-512}" \
  --block_tokens "${BLOCK_TOKENS:-10000}" \
  --mid_tokens "${MID_TOKENS:-1000}" \
  --leaf_tokens "${LEAF_TOKENS:-100}" \
  --top_blocks "${TOP_BLOCKS:-1}" \
  --mids_per_block "${MIDS_PER_BLOCK:-2}" \
  --leafs_per_mid "${LEAFS_PER_MID:-2}" \
  --seed_leafs "${SEED_LEAFS:-0}" \
  --route_refresh_tokens "${ROUTE_REFRESH_TOKENS:-16}" \
  --dtype "${DTYPE:-float16}" \
  --device "${DEVICE:-cuda}" \
  --device_map "${DEVICE_MAP:-auto}" \
  --attn_implementation eager \
  --log_every "${LOG_EVERY:-64}" \
  --seed "${SEED:-2026070301}"
