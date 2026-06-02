#!/usr/bin/env bash
set -euo pipefail

timestamp="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-round2_geometry_sweeps_${timestamp}}"
LOG_DIR="${LOG_DIR:-fdong_seq_compress/logs}"
RUN_ROOT="${RUN_ROOT:-fdong_seq_compress/outputs/$RUN_NAME}"
LOG_PATH="$LOG_DIR/$RUN_NAME.log"
PID_PATH="$LOG_DIR/$RUN_NAME.pid"
ENV_PATH="$LOG_DIR/$RUN_NAME.env"

mkdir -p "$LOG_DIR" "$RUN_ROOT"

cat > "$ENV_PATH" <<EOF
RUN_NAME=$RUN_NAME
RUN_ROOT=$RUN_ROOT
LOG_PATH=$LOG_PATH
PID_PATH=$PID_PATH
MODEL_PATH=${MODEL_PATH:-fdong/Qwen3-0.6B}
TEXT_PATH=${TEXT_PATH:-fdong_seq_compress/data/synthetic_texts/long_english_12000_words.txt}
TEXT_PATHS=${TEXT_PATHS:-fdong_seq_compress/data/synthetic_texts/long_english_12000_words.txt fdong_seq_compress/data/synthetic_texts/long_codebase_query_engine.txt fdong_seq_compress/data/synthetic_texts/long_textbook_distributed_systems.txt fdong_seq_compress/data/synthetic_texts/long_news_supply_chain_dossier.txt fdong_seq_compress/data/synthetic_texts/long_dialogue_tool_transcript.txt}
DEVICE=${DEVICE:-mps}
DTYPE=${DTYPE:-float16}
L2_MAX_TOKENS=${L2_MAX_TOKENS:-1000}
L2_TOP_K_VALUES=${L2_TOP_K_VALUES:-10 20 50}
L2_ANALYSIS_LEVELS=${L2_ANALYSIS_LEVELS:-token head}
SEQ_MAX_TOKENS_VALUES=${SEQ_MAX_TOKENS_VALUES:-1000 2000 4000 8000}
SEQ_TOP_K_VALUES=${SEQ_TOP_K_VALUES:-10}
SEQ_SIMILARITIES=${SEQ_SIMILARITIES:-cos l2}
SEQ_LAYERS=${SEQ_LAYERS:-0,6,11,15,21,27}
SEQ_HEADS=${SEQ_HEADS:-0,1}
HEAD_MAX_TOKENS=${HEAD_MAX_TOKENS:-1000}
HEAD_TOP_K_VALUES=${HEAD_TOP_K_VALUES:-10 20 50}
HEAD_SIMILARITIES=${HEAD_SIMILARITIES:-cos l2}
EOF

(
  export RUN_ROOT="$RUN_ROOT"
  export MODEL_PATH="${MODEL_PATH:-fdong/Qwen3-0.6B}"
  export TEXT_PATHS="${TEXT_PATHS:-fdong_seq_compress/data/synthetic_texts/long_english_12000_words.txt fdong_seq_compress/data/synthetic_texts/long_codebase_query_engine.txt fdong_seq_compress/data/synthetic_texts/long_textbook_distributed_systems.txt fdong_seq_compress/data/synthetic_texts/long_news_supply_chain_dossier.txt fdong_seq_compress/data/synthetic_texts/long_dialogue_tool_transcript.txt}"
  export DEVICE="${DEVICE:-mps}"
  export DTYPE="${DTYPE:-float16}"
  export L2_MAX_TOKENS="${L2_MAX_TOKENS:-1000}"
  export L2_TOP_K_VALUES="${L2_TOP_K_VALUES:-10 20 50}"
  export L2_ANALYSIS_LEVELS="${L2_ANALYSIS_LEVELS:-token head}"
  export L2_LAYERS="${L2_LAYERS:-all}"
  export L2_HEADS="${L2_HEADS:-all}"
  export SEQ_MAX_TOKENS_VALUES="${SEQ_MAX_TOKENS_VALUES:-1000 2000 4000 8000}"
  export SEQ_TOP_K_VALUES="${SEQ_TOP_K_VALUES:-10}"
  export SEQ_SIMILARITIES="${SEQ_SIMILARITIES:-cos l2}"
  export SEQ_ANALYSIS_LEVEL="${SEQ_ANALYSIS_LEVEL:-head}"
  export SEQ_LAYERS="${SEQ_LAYERS:-0,6,11,15,21,27}"
  export SEQ_HEADS="${SEQ_HEADS:-0,1}"
  export HEAD_MAX_TOKENS="${HEAD_MAX_TOKENS:-1000}"
  export HEAD_TOP_K_VALUES="${HEAD_TOP_K_VALUES:-10 20 50}"
  export HEAD_SIMILARITIES="${HEAD_SIMILARITIES:-cos l2}"
  export HEAD_ANALYSIS_LEVEL="${HEAD_ANALYSIS_LEVEL:-head}"
  export HEAD_LAYERS="${HEAD_LAYERS:-all}"
  export HEAD_HEADS="${HEAD_HEADS:-all}"
  nohup bash fdong_seq_compress/scripts/run_round2_geometry_sweeps.sh > "$LOG_PATH" 2>&1 &
  echo $! > "$PID_PATH"
)

echo "Started Round2 geometry sweeps."
echo "  run name:   $RUN_NAME"
echo "  run root:   $RUN_ROOT"
echo "  pid file:   $PID_PATH"
echo "  log file:   $LOG_PATH"
echo "  env file:   $ENV_PATH"
echo
echo "Track progress:"
echo "  tail -f $LOG_PATH"
echo
echo "Check process:"
echo "  ps -p \$(cat $PID_PATH)"
