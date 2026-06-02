#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-fdong/Qwen3-0.6B}"
TEXT_PATH="${TEXT_PATH:-fdong_seq_compress/data/synthetic_texts/long_english_12000_words.txt}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
MAX_TOKENS="${MAX_TOKENS:-1000}"
LAYERS="${LAYERS:-all}"
HEADS="${HEADS:-all}"
ANALYSIS_LEVEL="${ANALYSIS_LEVEL:-head}"
SIMILARITY="${SIMILARITY:-cos}"
KEY_TRANSFORM="${KEY_TRANSFORM:-center}"
PC_REMOVE_COUNT="${PC_REMOVE_COUNT:-0}"
TOP_K_VALUES="${TOP_K_VALUES:-10 20 50}"
RADIUS_THRESHOLDS="${RADIUS_THRESHOLDS:-}"
MAX_RADIUS_NEIGHBORS="${MAX_RADIUS_NEIGHBORS:-200}"
HIST_BINS="${HIST_BINS:-auto}"
SAVE_NEIGHBORS="${SAVE_NEIGHBORS:-0}"

timestamp="$(date +%Y%m%d_%H%M%S)"
SWEEP_OUTPUT_DIR="${SWEEP_OUTPUT_DIR:-fdong_seq_compress/outputs/k_graph_construction_sweep_${timestamp}}"
mkdir -p "$SWEEP_OUTPUT_DIR"

manifest="$SWEEP_OUTPUT_DIR/manifest.csv"
printf 'experiment_id,graph_mode,top_k,radius_threshold,similarity,key_transform,analysis_level,max_tokens,output_dir,summary,graph_summary\n' > "$manifest"

experiment_id=0
for top_k in $TOP_K_VALUES; do
  experiment_id=$((experiment_id + 1))
  output_dir="$SWEEP_OUTPUT_DIR/exp$(printf '%02d' "$experiment_id")_topk${top_k}_${SIMILARITY}_${KEY_TRANSFORM}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] exp=${experiment_id} graph=topk top_k=${top_k}"

  MODEL_PATH="$MODEL_PATH" TEXT_PATH="$TEXT_PATH" DEVICE="$DEVICE" DTYPE="$DTYPE" \
  MAX_TOKENS="$MAX_TOKENS" LAYERS="$LAYERS" HEADS="$HEADS" ANALYSIS_LEVEL="$ANALYSIS_LEVEL" \
  GRAPH_MODE=topk TOP_K="$top_k" SIMILARITY="$SIMILARITY" KEY_TRANSFORM="$KEY_TRANSFORM" \
  PC_REMOVE_COUNT="$PC_REMOVE_COUNT" HIST_BINS="$HIST_BINS" SAVE_NEIGHBORS="$SAVE_NEIGHBORS" \
  OUTPUT_DIR="$output_dir" bash fdong_seq_compress/scripts/run_k_similarity_graph_probe.sh

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$experiment_id" "topk" "$top_k" "" "$SIMILARITY" "$KEY_TRANSFORM" "$ANALYSIS_LEVEL" "$MAX_TOKENS" \
    "$output_dir" "$output_dir/summary.json" "$output_dir/graph_structure_summary_by_layer.csv" >> "$manifest"
done

for radius_threshold in $RADIUS_THRESHOLDS; do
  experiment_id=$((experiment_id + 1))
  output_dir="$SWEEP_OUTPUT_DIR/exp$(printf '%02d' "$experiment_id")_radius${radius_threshold}_${SIMILARITY}_${KEY_TRANSFORM}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] exp=${experiment_id} graph=radius threshold=${radius_threshold}"

  MODEL_PATH="$MODEL_PATH" TEXT_PATH="$TEXT_PATH" DEVICE="$DEVICE" DTYPE="$DTYPE" \
  MAX_TOKENS="$MAX_TOKENS" LAYERS="$LAYERS" HEADS="$HEADS" ANALYSIS_LEVEL="$ANALYSIS_LEVEL" \
  GRAPH_MODE=radius RADIUS_THRESHOLD="$radius_threshold" MAX_RADIUS_NEIGHBORS="$MAX_RADIUS_NEIGHBORS" \
  TOP_K=1 SIMILARITY="$SIMILARITY" KEY_TRANSFORM="$KEY_TRANSFORM" PC_REMOVE_COUNT="$PC_REMOVE_COUNT" \
  HIST_BINS="$HIST_BINS" SAVE_NEIGHBORS="$SAVE_NEIGHBORS" OUTPUT_DIR="$output_dir" \
  bash fdong_seq_compress/scripts/run_k_similarity_graph_probe.sh

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$experiment_id" "radius" "" "$radius_threshold" "$SIMILARITY" "$KEY_TRANSFORM" "$ANALYSIS_LEVEL" "$MAX_TOKENS" \
    "$output_dir" "$output_dir/summary.json" "$output_dir/graph_structure_summary_by_layer.csv" >> "$manifest"
done

echo "Graph construction sweep complete: $SWEEP_OUTPUT_DIR"
