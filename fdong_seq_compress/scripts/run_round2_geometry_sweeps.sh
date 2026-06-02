#!/usr/bin/env bash
set -euo pipefail

timestamp="$(date +%Y%m%d_%H%M%S)"

MODEL_PATH="${MODEL_PATH:-fdong/Qwen3-0.6B}"
TEXT_PATHS="${TEXT_PATHS:-fdong_seq_compress/data/synthetic_texts/long_english_12000_words.txt fdong_seq_compress/data/synthetic_texts/long_codebase_query_engine.txt fdong_seq_compress/data/synthetic_texts/long_textbook_distributed_systems.txt fdong_seq_compress/data/synthetic_texts/long_news_supply_chain_dossier.txt fdong_seq_compress/data/synthetic_texts/long_dialogue_tool_transcript.txt}"
DEVICE="${DEVICE:-mps}"
DTYPE="${DTYPE:-float16}"

RUN_ROOT="${RUN_ROOT:-fdong_seq_compress/outputs/round2_geometry_sweeps_${timestamp}}"
mkdir -p "$RUN_ROOT"

# Stage 1: Round2's central missing metric. This is intentionally run at 1k
# tokens across all layers/heads so it is broad but still cheap enough.
L2_MAX_TOKENS="${L2_MAX_TOKENS:-1000}"
L2_TOP_K_VALUES="${L2_TOP_K_VALUES:-10 20 50}"
L2_ANALYSIS_LEVELS="${L2_ANALYSIS_LEVELS:-token head}"
L2_LAYERS="${L2_LAYERS:-all}"
L2_HEADS="${L2_HEADS:-all}"

# Stage 2: seq-len scaling. Exact K-K graph is O(N^2), so the default uses
# representative layers/heads. Override these env vars for broader sweeps.
SEQ_MAX_TOKENS_VALUES="${SEQ_MAX_TOKENS_VALUES:-1000 2000 4000 8000}"
SEQ_TOP_K_VALUES="${SEQ_TOP_K_VALUES:-10}"
SEQ_SIMILARITIES="${SEQ_SIMILARITIES:-cos l2}"
SEQ_ANALYSIS_LEVEL="${SEQ_ANALYSIS_LEVEL:-head}"
SEQ_LAYERS="${SEQ_LAYERS:-0,6,11,15,21,27}"
SEQ_HEADS="${SEQ_HEADS:-0,1}"

# Stage 3: layer/head selection at a fixed manageable length.
HEAD_MAX_TOKENS="${HEAD_MAX_TOKENS:-1000}"
HEAD_TOP_K_VALUES="${HEAD_TOP_K_VALUES:-10 20 50}"
HEAD_SIMILARITIES="${HEAD_SIMILARITIES:-cos l2}"
HEAD_ANALYSIS_LEVEL="${HEAD_ANALYSIS_LEVEL:-head}"
HEAD_LAYERS="${HEAD_LAYERS:-all}"
HEAD_HEADS="${HEAD_HEADS:-all}"

manifest="$RUN_ROOT/round2_manifest.txt"
latest_path_file="fdong_seq_compress/outputs/round2_geometry_sweeps_latest_path.txt"
printf '%s\n' "$RUN_ROOT" > "$latest_path_file"

cat > "$RUN_ROOT/run_config.json" <<EOF
{
  "model_path": "$MODEL_PATH",
  "text_paths": "$TEXT_PATHS",
  "device": "$DEVICE",
  "dtype": "$DTYPE",
  "run_root": "$RUN_ROOT",
  "stage1_l2": {
    "max_tokens": "$L2_MAX_TOKENS",
    "top_k_values": "$L2_TOP_K_VALUES",
    "analysis_levels": "$L2_ANALYSIS_LEVELS",
    "layers": "$L2_LAYERS",
    "heads": "$L2_HEADS"
  },
  "stage2_seq_len": {
    "max_tokens_values": "$SEQ_MAX_TOKENS_VALUES",
    "top_k_values": "$SEQ_TOP_K_VALUES",
    "similarities": "$SEQ_SIMILARITIES",
    "analysis_level": "$SEQ_ANALYSIS_LEVEL",
    "layers": "$SEQ_LAYERS",
    "heads": "$SEQ_HEADS"
  },
  "stage3_head_selection": {
    "max_tokens": "$HEAD_MAX_TOKENS",
    "top_k_values": "$HEAD_TOP_K_VALUES",
    "similarities": "$HEAD_SIMILARITIES",
    "analysis_level": "$HEAD_ANALYSIS_LEVEL",
    "layers": "$HEAD_LAYERS",
    "heads": "$HEAD_HEADS"
  }
}
EOF

{
  echo "Round2 geometry sweeps started at $(date)"
  echo "RUN_ROOT=$RUN_ROOT"
  echo "MODEL_PATH=$MODEL_PATH"
  echo "TEXT_PATHS=$TEXT_PATHS"
  echo "DEVICE=$DEVICE DTYPE=$DTYPE"
  echo

  dataset_id=0
  for text_path in $TEXT_PATHS; do
    dataset_id=$((dataset_id + 1))
    dataset_name="$(basename "$text_path" .txt)"
    dataset_dir="$RUN_ROOT/d$(printf '%02d' "$dataset_id")_${dataset_name}"
    mkdir -p "$dataset_dir"
    echo "============================================================"
    echo "Dataset ${dataset_id}: $text_path"
    echo "Dataset output: $dataset_dir"
    echo "============================================================"

    echo "== Dataset ${dataset_id} Stage 1/3: L2 metric sweep =="
    stage1_dir="$dataset_dir/01_l2_metric_sweep"
    SWEEP_OUTPUT_DIR="$stage1_dir" \
    MODEL_PATH="$MODEL_PATH" \
    TEXT_PATH="$text_path" \
    DEVICE="$DEVICE" \
    DTYPE="$DTYPE" \
    MAX_TOKENS="$L2_MAX_TOKENS" \
    LAYERS="$L2_LAYERS" \
    HEADS="$L2_HEADS" \
    TOP_K_VALUES="$L2_TOP_K_VALUES" \
    SIMILARITIES="l2" \
    ANALYSIS_LEVELS="$L2_ANALYSIS_LEVELS" \
    SAVE_NEIGHBORS=0 \
    bash fdong_seq_compress/scripts/run_k_graph_metric_sweep.sh
    echo "dataset=${dataset_name},stage1_l2_metric_sweep=$stage1_dir" >> "$manifest"
    echo

    echo "== Dataset ${dataset_id} Stage 2/3: seq-len scaling sweep =="
    stage2_dir="$dataset_dir/02_seq_len_scaling"
    SWEEP_OUTPUT_DIR="$stage2_dir" \
    MODEL_PATH="$MODEL_PATH" \
    TEXT_PATH="$text_path" \
    DEVICE="$DEVICE" \
    DTYPE="$DTYPE" \
    MAX_TOKENS_VALUES="$SEQ_MAX_TOKENS_VALUES" \
    LAYERS="$SEQ_LAYERS" \
    HEADS="$SEQ_HEADS" \
    ANALYSIS_LEVEL="$SEQ_ANALYSIS_LEVEL" \
    TOP_K_VALUES="$SEQ_TOP_K_VALUES" \
    SIMILARITIES="$SEQ_SIMILARITIES" \
    KEY_TRANSFORM=center \
    SAVE_NEIGHBORS=0 \
    bash fdong_seq_compress/scripts/run_k_seq_len_scaling_sweep.sh
    echo "dataset=${dataset_name},stage2_seq_len_scaling=$stage2_dir" >> "$manifest"
    echo

    echo "== Dataset ${dataset_id} Stage 3/3: layer/head selection sweep =="
    stage3_dir="$dataset_dir/03_layer_head_selection"
    SWEEP_OUTPUT_DIR="$stage3_dir" \
    MODEL_PATH="$MODEL_PATH" \
    TEXT_PATH="$text_path" \
    DEVICE="$DEVICE" \
    DTYPE="$DTYPE" \
    MAX_TOKENS_VALUES="$HEAD_MAX_TOKENS" \
    LAYERS="$HEAD_LAYERS" \
    HEADS="$HEAD_HEADS" \
    ANALYSIS_LEVEL="$HEAD_ANALYSIS_LEVEL" \
    TOP_K_VALUES="$HEAD_TOP_K_VALUES" \
    SIMILARITIES="$HEAD_SIMILARITIES" \
    KEY_TRANSFORM=center \
    SAVE_NEIGHBORS=0 \
    bash fdong_seq_compress/scripts/run_k_seq_len_scaling_sweep.sh
    echo "dataset=${dataset_name},stage3_layer_head_selection=$stage3_dir" >> "$manifest"
    echo
  done

  echo "Round2 geometry sweeps finished at $(date)"
  echo "Manifest: $manifest"
} 2>&1 | tee "$RUN_ROOT/run.log"

echo "Run complete: $RUN_ROOT"
