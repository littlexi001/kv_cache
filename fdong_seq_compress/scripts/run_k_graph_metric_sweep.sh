#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-fdong/Qwen3-0.6B}"
TEXT_PATH="${TEXT_PATH:-fdong_seq_compress/data/synthetic_texts/long_english_12000_words.txt}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
MAX_TOKENS="${MAX_TOKENS:-1000}"
LAYERS="${LAYERS:-all}"
HEADS="${HEADS:-all}"
TOP_K_VALUES="${TOP_K_VALUES:-10 20 50}"
SIMILARITIES="${SIMILARITIES:-cos dot}"
ANALYSIS_LEVELS="${ANALYSIS_LEVELS:-token head}"
HIST_BINS="${HIST_BINS:--1.0:1.0:0.05}"
SAVE_NEIGHBORS="${SAVE_NEIGHBORS:-0}"
CENTER_TOKENS="${CENTER_TOKENS:-1}"

if [[ "$CENTER_TOKENS" != "1" ]]; then
  echo "This sweep is intended to run centered K only. Set CENTER_TOKENS=1." >&2
  exit 2
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
SWEEP_OUTPUT_DIR="${SWEEP_OUTPUT_DIR:-fdong_seq_compress/outputs/k_graph_metric_sweep_${timestamp}}"
mkdir -p "$SWEEP_OUTPUT_DIR"

manifest="$SWEEP_OUTPUT_DIR/manifest.csv"
printf 'experiment_id,top_k,similarity,analysis_level,center_tokens,max_tokens,output_dir,summary,similarity_svg,distance_svg,indegree_svg\n' > "$manifest"

config_json="$SWEEP_OUTPUT_DIR/sweep_config.json"
cat > "$config_json" <<EOF
{
  "model_path": "$MODEL_PATH",
  "text_path": "$TEXT_PATH",
  "device": "$DEVICE",
  "dtype": "$DTYPE",
  "max_tokens": $MAX_TOKENS,
  "layers": "$LAYERS",
  "heads": "$HEADS",
  "top_k_values": "$TOP_K_VALUES",
  "similarities": "$SIMILARITIES",
  "analysis_levels": "$ANALYSIS_LEVELS",
  "center_tokens": true,
  "save_neighbors": "$SAVE_NEIGHBORS"
}
EOF

experiment_id=0
for analysis_level in $ANALYSIS_LEVELS; do
  for similarity in $SIMILARITIES; do
    for top_k in $TOP_K_VALUES; do
      experiment_id=$((experiment_id + 1))
      output_dir="$SWEEP_OUTPUT_DIR/exp$(printf '%02d' "$experiment_id")_${analysis_level}_${similarity}_top${top_k}_centered1"
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] experiment=${experiment_id} analysis_level=${analysis_level} similarity=${similarity} top_k=${top_k}"
      echo "  output_dir=${output_dir}"

      MODEL_PATH="$MODEL_PATH" \
      TEXT_PATH="$TEXT_PATH" \
      DEVICE="$DEVICE" \
      DTYPE="$DTYPE" \
      MAX_TOKENS="$MAX_TOKENS" \
      LAYERS="$LAYERS" \
      HEADS="$HEADS" \
      ANALYSIS_LEVEL="$analysis_level" \
      TOP_K="$top_k" \
      SIMILARITY="$similarity" \
      HIST_BINS="$HIST_BINS" \
      CENTER_TOKENS=1 \
      SAVE_NEIGHBORS="$SAVE_NEIGHBORS" \
      OUTPUT_DIR="$output_dir" \
      bash fdong_seq_compress/scripts/run_k_similarity_graph_probe.sh

      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$experiment_id" \
        "$top_k" \
        "$similarity" \
        "$analysis_level" \
        "1" \
        "$MAX_TOKENS" \
        "$output_dir" \
        "$output_dir/summary.json" \
        "$output_dir/histogram_global.svg" \
        "$output_dir/distance_histogram_global.svg" \
        "$output_dir/indegree_histogram_global.svg" >> "$manifest"
    done
  done
done

index="$SWEEP_OUTPUT_DIR/index.html"
{
  printf '<!doctype html>\n<html><head><meta charset="utf-8">\n'
  printf '<title>K graph metric sweep</title>\n'
  printf '<style>body{font-family:Arial,sans-serif;margin:24px;background:#fbfaf7;color:#222}section{margin-bottom:42px}img{max-width:100%%;border:1px solid #ddd;background:white;margin:8px 0}code{background:#eee;padding:2px 4px}</style>\n'
  printf '</head><body>\n'
  printf '<h1>K graph metric sweep</h1>\n'
  printf '<p><code>centered=true</code> <code>MAX_TOKENS=%s</code> <code>LAYERS=%s</code> <code>HEADS=%s</code></p>\n' "$MAX_TOKENS" "$LAYERS" "$HEADS"
  printf '<p><a href="manifest.csv">manifest.csv</a> | <a href="sweep_config.json">sweep_config.json</a></p>\n'

  tail -n +2 "$manifest" | while IFS=',' read -r exp_id top_k similarity analysis_level centered max_tokens output_dir summary sim_svg dist_svg indeg_svg; do
    rel_dir="$(basename "$output_dir")"
    printf '<section>\n'
    printf '<h2>Experiment %s: %s, %s, top-%s</h2>\n' "$exp_id" "$analysis_level" "$similarity" "$top_k"
    printf '<p><a href="%s/summary.json">summary.json</a> | <a href="%s/summary_by_layer.csv">similarity layer summary</a> | <a href="%s/distance_summary_by_layer.csv">distance layer summary</a> | <a href="%s/indegree_summary_by_layer.csv">in-degree layer summary</a></p>\n' "$rel_dir" "$rel_dir" "$rel_dir" "$rel_dir"
    printf '<img src="%s/histogram_global.svg" alt="similarity histogram">\n' "$rel_dir"
    printf '<img src="%s/distance_histogram_global.svg" alt="distance histogram">\n' "$rel_dir"
    printf '<img src="%s/indegree_histogram_global.svg" alt="in-degree histogram">\n' "$rel_dir"
    printf '</section>\n'
  done

  printf '</body></html>\n'
} > "$index"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Sweep complete."
echo "Output: $SWEEP_OUTPUT_DIR"
echo "Index:  $index"
