#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-fdong/Qwen3-0.6B}"
TEXT_PATH="${TEXT_PATH:-fdong_seq_compress/data/synthetic_texts/long_english_12000_words.txt}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
MAX_TOKENS="${MAX_TOKENS:-1000}"
LAYERS="${LAYERS:-all}"
HEADS="${HEADS:-all}"
ANALYSIS_LEVEL="${ANALYSIS_LEVEL:-token}"
SIMILARITY="${SIMILARITY:-cos}"
HIST_BINS="${HIST_BINS:-auto}"
TOP_K_VALUES="${TOP_K_VALUES:-5 10 20}"
CENTER_VALUES="${CENTER_VALUES:-1}"
SAVE_NEIGHBORS="${SAVE_NEIGHBORS:-0}"

timestamp="$(date +%Y%m%d_%H%M%S)"
BATCH_OUTPUT_DIR="${BATCH_OUTPUT_DIR:-fdong_seq_compress/outputs/k_similarity_batch_1000_${timestamp}}"
mkdir -p "$BATCH_OUTPUT_DIR"

manifest="$BATCH_OUTPUT_DIR/manifest.csv"
printf 'top_k,center_tokens,output_dir,summary,global_svg,plots_dir\n' > "$manifest"

for center_tokens in $CENTER_VALUES; do
  for top_k in $TOP_K_VALUES; do
    output_dir="$BATCH_OUTPUT_DIR/top${top_k}_centered${center_tokens}"
    echo "Running top_k=${top_k} center_tokens=${center_tokens} -> ${output_dir}"

    MODEL_PATH="$MODEL_PATH" \
    TEXT_PATH="$TEXT_PATH" \
    DEVICE="$DEVICE" \
    DTYPE="$DTYPE" \
    MAX_TOKENS="$MAX_TOKENS" \
    LAYERS="$LAYERS" \
    HEADS="$HEADS" \
    ANALYSIS_LEVEL="$ANALYSIS_LEVEL" \
    TOP_K="$top_k" \
    SIMILARITY="$SIMILARITY" \
    HIST_BINS="$HIST_BINS" \
    CENTER_TOKENS="$center_tokens" \
    SAVE_NEIGHBORS="$SAVE_NEIGHBORS" \
    OUTPUT_DIR="$output_dir" \
    bash fdong_seq_compress/scripts/run_k_similarity_graph_probe.sh

    printf '%s,%s,%s,%s,%s,%s\n' \
      "$top_k" \
      "$center_tokens" \
      "$output_dir" \
      "$output_dir/summary.json" \
      "$output_dir/histogram_global.svg" \
      "$output_dir/plots" >> "$manifest"
  done
done

index="$BATCH_OUTPUT_DIR/index.html"
{
  printf '<!doctype html>\n<html><head><meta charset="utf-8">\n'
  printf '<title>K similarity batch</title>\n'
  printf '<style>body{font-family:Arial,sans-serif;margin:24px;background:#fbfaf7;color:#222}section{margin-bottom:36px}img{max-width:100%%;border:1px solid #ddd;background:white}code{background:#eee;padding:2px 4px}</style>\n'
  printf '</head><body>\n'
  printf '<h1>K-cache similarity batch</h1>\n'
  printf '<p><code>MAX_TOKENS=%s</code> <code>LAYERS=%s</code> <code>ANALYSIS_LEVEL=%s</code> <code>SIMILARITY=%s</code></p>\n' "$MAX_TOKENS" "$LAYERS" "$ANALYSIS_LEVEL" "$SIMILARITY"
  printf '<p>Manifest: <a href="manifest.csv">manifest.csv</a></p>\n'

  for center_tokens in $CENTER_VALUES; do
    for top_k in $TOP_K_VALUES; do
      output_dir="top${top_k}_centered${center_tokens}"
      printf '<section>\n'
      printf '<h2>top-%s, centered=%s</h2>\n' "$top_k" "$center_tokens"
      printf '<p><a href="%s/summary.json">summary.json</a> | <a href="%s/summary_by_layer.csv">summary_by_layer.csv</a> | <a href="%s/distance_summary_by_layer.csv">distance_summary_by_layer.csv</a> | <a href="%s/indegree_summary_by_layer.csv">indegree_summary_by_layer.csv</a> | <a href="%s/plots/">per-layer plots</a></p>\n' "$output_dir" "$output_dir" "$output_dir" "$output_dir" "$output_dir"
      printf '<img src="%s/histogram_global.svg" alt="top-%s centered=%s global histogram">\n' "$output_dir" "$top_k" "$center_tokens"
      printf '<img src="%s/distance_histogram_global.svg" alt="top-%s centered=%s distance histogram">\n' "$output_dir" "$top_k" "$center_tokens"
      printf '<img src="%s/indegree_histogram_global.svg" alt="top-%s centered=%s indegree histogram">\n' "$output_dir" "$top_k" "$center_tokens"
      printf '</section>\n'
    done
  done

  printf '</body></html>\n'
} > "$index"

echo "Wrote batch outputs to $BATCH_OUTPUT_DIR"
echo "Open $index to preview the global SVG plots."
