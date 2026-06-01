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
TOP_K_VALUES="${TOP_K_VALUES:-10 20}"
SIMILARITIES="${SIMILARITIES:-cos l2}"
TRANSFORM_SPECS="${TRANSFORM_SPECS:-raw:0 center:0 remove_pc:1 remove_pc:4 remove_pc:8 whiten:0}"
HIST_BINS="${HIST_BINS:-auto}"
SAVE_NEIGHBORS="${SAVE_NEIGHBORS:-0}"

timestamp="$(date +%Y%m%d_%H%M%S)"
SWEEP_OUTPUT_DIR="${SWEEP_OUTPUT_DIR:-fdong_seq_compress/outputs/k_transform_sweep_${timestamp}}"
mkdir -p "$SWEEP_OUTPUT_DIR"

manifest="$SWEEP_OUTPUT_DIR/manifest.csv"
printf 'experiment_id,key_transform,pc_remove_count,top_k,similarity,analysis_level,max_tokens,output_dir,summary,graph_summary\n' > "$manifest"

experiment_id=0
for spec in $TRANSFORM_SPECS; do
  key_transform="${spec%%:*}"
  pc_remove_count="${spec##*:}"
  center_tokens_flag=1
  if [[ "$key_transform" == "raw" ]]; then
    center_tokens_flag=0
  fi
  for similarity in $SIMILARITIES; do
    for top_k in $TOP_K_VALUES; do
      experiment_id=$((experiment_id + 1))
      output_dir="$SWEEP_OUTPUT_DIR/exp$(printf '%02d' "$experiment_id")_${ANALYSIS_LEVEL}_${similarity}_top${top_k}_${key_transform}${pc_remove_count}"
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] exp=${experiment_id} transform=${key_transform}:${pc_remove_count} similarity=${similarity} top_k=${top_k}"

      MODEL_PATH="$MODEL_PATH" \
      TEXT_PATH="$TEXT_PATH" \
      DEVICE="$DEVICE" \
      DTYPE="$DTYPE" \
      MAX_TOKENS="$MAX_TOKENS" \
      LAYERS="$LAYERS" \
      HEADS="$HEADS" \
      ANALYSIS_LEVEL="$ANALYSIS_LEVEL" \
      GRAPH_MODE=topk \
      TOP_K="$top_k" \
      SIMILARITY="$similarity" \
      CENTER_TOKENS="$center_tokens_flag" \
      KEY_TRANSFORM="$key_transform" \
      PC_REMOVE_COUNT="$pc_remove_count" \
      HIST_BINS="$HIST_BINS" \
      SAVE_NEIGHBORS="$SAVE_NEIGHBORS" \
      OUTPUT_DIR="$output_dir" \
      bash fdong_seq_compress/scripts/run_k_similarity_graph_probe.sh

      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$experiment_id" "$key_transform" "$pc_remove_count" "$top_k" "$similarity" "$ANALYSIS_LEVEL" "$MAX_TOKENS" \
        "$output_dir" "$output_dir/summary.json" "$output_dir/graph_structure_summary_by_layer.csv" >> "$manifest"
    done
  done
done

echo "Transform sweep complete: $SWEEP_OUTPUT_DIR"
