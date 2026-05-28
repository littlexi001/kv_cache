#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-qwen3_geometry_mps_long_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-fdong_seq_compress/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-fdong_seq_compress/outputs/$RUN_NAME}"
LOG_PATH="$LOG_DIR/$RUN_NAME.log"
PID_PATH="$LOG_DIR/$RUN_NAME.pid"
ENV_PATH="$LOG_DIR/$RUN_NAME.env"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

cat > "$ENV_PATH" <<EOF
RUN_NAME=$RUN_NAME
MODEL_PATH=${MODEL_PATH:-fdong/Qwen3-0.6B}
TEXT_PATH=${TEXT_PATH:-fdong_seq_compress/data/synthetic_texts/long_english_12000_words.txt}
OUTPUT_DIR=$OUTPUT_DIR
DEVICE=${DEVICE:-mps}
DTYPE=${DTYPE:-float16}
MAX_TOKENS=${MAX_TOKENS:-12000}
PREFIX_LENGTHS=${PREFIX_LENGTHS:-512,1024,2048,4096,8192,12000}
LAYERS=${LAYERS:-all}
HEADS=${HEADS:-all}
KINDS=${KINDS:-K,V}
BLOCK_SIZES=${BLOCK_SIZES:-4,8,16,32,64,128}
SUBSPACE_RANK=${SUBSPACE_RANK:-16}
ENERGY_THRESHOLDS=${ENERGY_THRESHOLDS:-0.90,0.95,0.99}
LOG_DIR=$LOG_DIR
LOG_PATH=$LOG_PATH
PID_PATH=$PID_PATH
EOF

(
  export MODEL_PATH="${MODEL_PATH:-fdong/Qwen3-0.6B}"
  export TEXT_PATH="${TEXT_PATH:-fdong_seq_compress/data/synthetic_texts/long_english_12000_words.txt}"
  export OUTPUT_DIR="$OUTPUT_DIR"
  export DEVICE="${DEVICE:-mps}"
  export DTYPE="${DTYPE:-float16}"
  export MAX_TOKENS="${MAX_TOKENS:-12000}"
  export PREFIX_LENGTHS="${PREFIX_LENGTHS:-512,1024,2048,4096,8192,12000}"
  export LAYERS="${LAYERS:-all}"
  export HEADS="${HEADS:-all}"
  export KINDS="${KINDS:-K,V}"
  export BLOCK_SIZES="${BLOCK_SIZES:-4,8,16,32,64,128}"
  export SUBSPACE_RANK="${SUBSPACE_RANK:-16}"
  export ENERGY_THRESHOLDS="${ENERGY_THRESHOLDS:-0.90,0.95,0.99}"
  nohup bash fdong_seq_compress/scripts/run_mps_geometry_long.sh > "$LOG_PATH" 2>&1 &
  echo $! > "$PID_PATH"
)

echo "Started background run."
echo "  run name:   $RUN_NAME"
echo "  pid file:   $PID_PATH"
echo "  log file:   $LOG_PATH"
echo "  env file:   $ENV_PATH"
echo "  output dir: $OUTPUT_DIR"
echo
echo "Track progress:"
echo "  tail -f $LOG_PATH"
echo
echo "Check process:"
echo "  ps -p \$(cat $PID_PATH)"
echo
echo "Stop process if needed:"
echo "  kill \$(cat $PID_PATH)"

