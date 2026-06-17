#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

RUN_NAME="${RUN_NAME:-qk-feature-svd-qwen3-0.6b-t5000-n100}"
MODEL_DIR="${MODEL_DIR:-../../../Qwen3-0.6B}"
DATA_DIR="${DATA_DIR:-../../../dclm/global-shard_01_of_10}"
INPUT_SOURCE="${INPUT_SOURCE:-synthetic_long_qa}"  # synthetic_long_qa | dclm | text_file
PROMPT_FILE="${PROMPT_FILE:-}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-../outputs/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-../logs}"

DEVICE="${DEVICE:-auto}"                # auto | cuda | mps | cpu
DTYPE="${DTYPE:-auto}"                  # auto | float32 | float16 | bfloat16
SVD_DEVICE="${SVD_DEVICE:-cpu}"         # cpu | model

TOKEN_START="${TOKEN_START:-5000}"
NUM_QUERY_TOKENS="${NUM_QUERY_TOKENS:-100}"
TOP_RATIO="${TOP_RATIO:-0.02}"
LAYERS="${LAYERS:-all}"                 # all | comma-separated layer ids, e.g. 0,13,27
HEADS="${HEADS:-all}"                   # all | comma-separated head ids, e.g. 0,1,2,3

BAND_MODE="${BAND_MODE:-equal_energy}"  # equal_energy | fixed
NUM_ENERGY_BANDS="${NUM_ENERGY_BANDS:-8}"
SVD_RANK_LIMIT="${SVD_RANK_LIMIT:-256}" # <=0 means full rank; 256 is faster for overnight all-layer/head runs
FIXED_ENERGY_EDGES="${FIXED_ENERGY_EDGES:-0,0.01,0.05,0.1,0.2,0.4,0.7,1.0}"

SVD_KEYS_PER_QUERY="${SVD_KEYS_PER_QUERY:-4}"       # <=0 means all top-ratio keys
TAIL_SVD_KEYS_PER_QUERY="${TAIL_SVD_KEYS_PER_QUERY:-4}" # <=0 means match selected top key count
TAIL_SAMPLE_MODE="${TAIL_SAMPLE_MODE:-low_score}" # low_score | random_tail
RANDOM_NEGATIVES_PER_QUERY="${RANDOM_NEGATIVES_PER_QUERY:-8}"
DISTANCE_NEGATIVES_PER_QUERY="${DISTANCE_NEGATIVES_PER_QUERY:-8}"
MAX_EXAMPLES="${MAX_EXAMPLES:-256}"
EXAMPLE_TOP_PAIRS="${EXAMPLE_TOP_PAIRS:-16}"
EXAMPLE_RANK_LIMIT="${EXAMPLE_RANK_LIMIT:-256}"
MAX_FILES="${MAX_FILES:-0}"
SEED="${SEED:-0}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

nohup python3 analyze_qk_feature_svd.py \
  --model_dir "$MODEL_DIR" \
  --data_dir "$DATA_DIR" \
  --input_source "$INPUT_SOURCE" \
  --prompt_file "$PROMPT_FILE" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --svd_device "$SVD_DEVICE" \
  --token_start "$TOKEN_START" \
  --num_query_tokens "$NUM_QUERY_TOKENS" \
  --top_ratio "$TOP_RATIO" \
  --layers "$LAYERS" \
  --heads "$HEADS" \
  --band_mode "$BAND_MODE" \
  --num_energy_bands "$NUM_ENERGY_BANDS" \
  --svd_rank_limit "$SVD_RANK_LIMIT" \
  --fixed_energy_edges "$FIXED_ENERGY_EDGES" \
  --svd_keys_per_query "$SVD_KEYS_PER_QUERY" \
  --tail_svd_keys_per_query "$TAIL_SVD_KEYS_PER_QUERY" \
  --tail_sample_mode "$TAIL_SAMPLE_MODE" \
  --random_negatives_per_query "$RANDOM_NEGATIVES_PER_QUERY" \
  --distance_negatives_per_query "$DISTANCE_NEGATIVES_PER_QUERY" \
  --max_examples "$MAX_EXAMPLES" \
  --example_top_pairs "$EXAMPLE_TOP_PAIRS" \
  --example_rank_limit "$EXAMPLE_RANK_LIMIT" \
  --max_files "$MAX_FILES" \
  --seed "$SEED" \
  >"$LOG_DIR/${RUN_NAME}.log" 2>&1 &

echo "Started PID $!"
echo "Console log: $LOG_DIR/${RUN_NAME}.log"
echo "Output dir:  $OUTPUT_DIR"
echo "Monitor:     tail -f $LOG_DIR/${RUN_NAME}.log"
