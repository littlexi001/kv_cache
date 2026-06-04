#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

MODEL_PATH="${MODEL_PATH:-fdong/Qwen3-0.6B}"
DEVICE="${DEVICE:-mps}"
DTYPE="${DTYPE:-float16}"
MAX_TOKENS="${MAX_TOKENS:-2000}"
DECODE_START="${DECODE_START:-1000}"
LOCAL_WINDOW="${LOCAL_WINDOW:-128}"
SEED_COUNT="${SEED_COUNT:-8}"
MAX_CANDIDATES="${MAX_CANDIDATES:-256}"
GRAPH_TOP_K="${GRAPH_TOP_K:-10}"
GRAPH_HOPS="${GRAPH_HOPS:-1}"
GRAPH_DIRECTION="${GRAPH_DIRECTION:-both}"
SIMILARITY="${SIMILARITY:-l2}"
KMEANS_STEPS="${KMEANS_STEPS:-8}"
RANDOM_SEED="${RANDOM_SEED:-0}"
METHODS="${METHODS:-local,local_topq_graph,cluster_topn,local_cluster_topn}"
ALWAYS_INCLUDE_POSITIONS="${ALWAYS_INCLUDE_POSITIONS:-0:10}"
SINK_POSITIONS="${SINK_POSITIONS:-0}"
TOP_ATTENTION_KS="${TOP_ATTENTION_KS:-1,5,10}"

TEXT_PATHS="${TEXT_PATHS:-fdong_seq_compress/data/synthetic_texts/long_textbook_distributed_systems.txt fdong_seq_compress/data/synthetic_texts/long_news_supply_chain_dossier.txt fdong_seq_compress/data/synthetic_texts/long_codebase_query_engine.txt}"
NUM_CLUSTERS_VALUES="${NUM_CLUSTERS_VALUES:-5 10 20}"
TOP_CLUSTERS_VALUES="${TOP_CLUSTERS_VALUES:-1 2 4}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-fdong_seq_compress/outputs/sparse_ppl_cluster_sweep_${STAMP}}"
LOG_DIR="${LOG_DIR:-fdong_seq_compress/logs}"
mkdir -p "${RUN_ROOT}" "${LOG_DIR}"

echo "run_root=${RUN_ROOT}"
echo "log_dir=${LOG_DIR}"
echo "model=${MODEL_PATH} device=${DEVICE} dtype=${DTYPE}"
echo "max_tokens=${MAX_TOKENS} decode_start=${DECODE_START}"
echo "methods=${METHODS}"
echo "texts=${TEXT_PATHS}"
echo "num_clusters=${NUM_CLUSTERS_VALUES}"
echo "top_clusters=${TOP_CLUSTERS_VALUES}"

for text_path in ${TEXT_PATHS}; do
  text_name="$(basename "${text_path}" .txt)"
  for num_clusters in ${NUM_CLUSTERS_VALUES}; do
    for top_clusters in ${TOP_CLUSTERS_VALUES}; do
      output_dir="${RUN_ROOT}/${text_name}_c${num_clusters}_top${top_clusters}"
      echo "START text=${text_path} clusters=${num_clusters} top_clusters=${top_clusters} output=${output_dir}"
      python3 fdong_seq_compress/src/run_sparse_perplexity.py \
        --model-path "${MODEL_PATH}" \
        --text-path "${text_path}" \
        --output-dir "${output_dir}" \
        --device "${DEVICE}" \
        --dtype "${DTYPE}" \
        --attn-implementation eager \
        --max-tokens "${MAX_TOKENS}" \
        --decode-start "${DECODE_START}" \
        --methods "${METHODS}" \
        --local-window "${LOCAL_WINDOW}" \
        --seed-count "${SEED_COUNT}" \
        --max-candidates "${MAX_CANDIDATES}" \
        --always-include-positions "${ALWAYS_INCLUDE_POSITIONS}" \
        --sink-positions "${SINK_POSITIONS}" \
        --top-attention-ks "${TOP_ATTENTION_KS}" \
        --graph-top-k "${GRAPH_TOP_K}" \
        --graph-hops "${GRAPH_HOPS}" \
        --graph-direction "${GRAPH_DIRECTION}" \
        --similarity "${SIMILARITY}" \
        --num-clusters "${num_clusters}" \
        --top-clusters "${top_clusters}" \
        --kmeans-steps "${KMEANS_STEPS}" \
        --random-seed "${RANDOM_SEED}"
      echo "DONE output=${output_dir}"
    done
  done
done

python3 - <<'PY' "${RUN_ROOT}"
import csv
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
rows = []
timing_rows = []
for path in sorted(run_root.glob("*/perplexity_by_method.csv")):
    summary = json.loads((path.parent / "summary.json").read_text())
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["case"] = path.parent.name
            row["text_path"] = summary["text_path"]
            row["seq_len"] = summary["seq_len"]
            rows.append(row)
    timing_path = path.parent / "timing_by_method_layer.csv"
    if timing_path.exists():
        with timing_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["case"] = path.parent.name
                row["text_path"] = summary["text_path"]
                row["seq_len"] = summary["seq_len"]
                timing_rows.append(row)

if rows:
    out = run_root / "aggregate_perplexity_by_method.csv"
    fieldnames = sorted({k for row in rows for k in row})
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote aggregate: {out}")
else:
    print("No result rows found.")

if timing_rows:
    out = run_root / "aggregate_timing_by_method_layer.csv"
    fieldnames = sorted({k for row in timing_rows for k in row})
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(timing_rows)
    print(f"Wrote aggregate timing: {out}")
PY

echo "ALL_DONE run_root=${RUN_ROOT}"
