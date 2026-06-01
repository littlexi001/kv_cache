#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-fdong/Qwen3-0.6B}"
TEXT_PATH="${TEXT_PATH:-fdong_seq_compress/data/synthetic_texts/long_english_12000_words.txt}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
MAX_TOKENS_VALUES="${MAX_TOKENS_VALUES:-1000 2000 4000 8000 12000}"
LAYERS="${LAYERS:-all}"
HEADS="${HEADS:-all}"
ANALYSIS_LEVEL="${ANALYSIS_LEVEL:-head}"
TOP_K_VALUES="${TOP_K_VALUES:-10 20}"
SIMILARITIES="${SIMILARITIES:-cos l2}"
KEY_TRANSFORM="${KEY_TRANSFORM:-center}"
PC_REMOVE_COUNT="${PC_REMOVE_COUNT:-0}"
HIST_BINS="${HIST_BINS:-auto}"
SAVE_NEIGHBORS="${SAVE_NEIGHBORS:-0}"

timestamp="$(date +%Y%m%d_%H%M%S)"
SWEEP_OUTPUT_DIR="${SWEEP_OUTPUT_DIR:-fdong_seq_compress/outputs/k_seq_len_scaling_sweep_${timestamp}}"
mkdir -p "$SWEEP_OUTPUT_DIR"

manifest="$SWEEP_OUTPUT_DIR/manifest.csv"
printf 'experiment_id,max_tokens,top_k,similarity,analysis_level,key_transform,output_dir,summary,graph_summary\n' > "$manifest"

experiment_id=0
for max_tokens in $MAX_TOKENS_VALUES; do
  for similarity in $SIMILARITIES; do
    for top_k in $TOP_K_VALUES; do
      experiment_id=$((experiment_id + 1))
      output_dir="$SWEEP_OUTPUT_DIR/exp$(printf '%02d' "$experiment_id")_n${max_tokens}_${ANALYSIS_LEVEL}_${similarity}_top${top_k}_${KEY_TRANSFORM}"
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] exp=${experiment_id} max_tokens=${max_tokens} similarity=${similarity} top_k=${top_k}"

      MODEL_PATH="$MODEL_PATH" \
      TEXT_PATH="$TEXT_PATH" \
      DEVICE="$DEVICE" \
      DTYPE="$DTYPE" \
      MAX_TOKENS="$max_tokens" \
      LAYERS="$LAYERS" \
      HEADS="$HEADS" \
      ANALYSIS_LEVEL="$ANALYSIS_LEVEL" \
      GRAPH_MODE=topk \
      TOP_K="$top_k" \
      SIMILARITY="$similarity" \
      KEY_TRANSFORM="$KEY_TRANSFORM" \
      PC_REMOVE_COUNT="$PC_REMOVE_COUNT" \
      HIST_BINS="$HIST_BINS" \
      SAVE_NEIGHBORS="$SAVE_NEIGHBORS" \
      OUTPUT_DIR="$output_dir" \
      bash fdong_seq_compress/scripts/run_k_similarity_graph_probe.sh

      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$experiment_id" "$max_tokens" "$top_k" "$similarity" "$ANALYSIS_LEVEL" "$KEY_TRANSFORM" \
        "$output_dir" "$output_dir/summary.json" "$output_dir/graph_structure_summary_by_layer.csv" >> "$manifest"
    done
  done
done

echo "Seq-len scaling sweep complete: $SWEEP_OUTPUT_DIR"

