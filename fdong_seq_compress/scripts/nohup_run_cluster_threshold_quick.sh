#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

MODEL_PATH="${MODEL_PATH:-fdong/Qwen3-0.6B}"
TEXT_PATH="${TEXT_PATH:-fdong_seq_compress/data/synthetic_texts/biomed_long_range_facts_hard_compact.txt}"
DEVICE="${DEVICE:-mps}"
DTYPE="${DTYPE:-float16}"
MAX_TOKENS="${MAX_TOKENS:-3000}"
DECODE_START="${DECODE_START:-2500}"
METHODS="${METHODS:-cluster_threshold,local_cluster_threshold}"
NUM_CLUSTERS="${NUM_CLUSTERS:-20}"
CLUSTER_THRESHOLDS="${CLUSTER_THRESHOLDS:-0.50 0.75 1.00}"
KMEANS_STEPS="${KMEANS_STEPS:-3}"
CLUSTER_SCALE_SAMPLE_PAIRS="${CLUSTER_SCALE_SAMPLE_PAIRS:-20000}"
MIN_SELECTED_CLUSTERS="${MIN_SELECTED_CLUSTERS:-1}"
MAX_CANDIDATES="${MAX_CANDIDATES:-0}"
LOCAL_WINDOW="${LOCAL_WINDOW:-128}"
ALWAYS_INCLUDE_POSITIONS="${ALWAYS_INCLUDE_POSITIONS:-0:10}"
SINK_POSITIONS="${SINK_POSITIONS:-0}"
TOP_ATTENTION_KS="${TOP_ATTENTION_KS:-1,5,10}"
RANDOM_SEED="${RANDOM_SEED:-0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-fdong_seq_compress/outputs/cluster_threshold_quick_${STAMP}}"
LOG_DIR="${LOG_DIR:-fdong_seq_compress/logs}"
mkdir -p "${RUN_ROOT}" "${LOG_DIR}"

echo "run_root=${RUN_ROOT}"
echo "model=${MODEL_PATH} text=${TEXT_PATH}"
echo "device=${DEVICE} dtype=${DTYPE}"
echo "max_tokens=${MAX_TOKENS} decode_start=${DECODE_START}"
echo "methods=${METHODS}"
echo "num_clusters=${NUM_CLUSTERS} thresholds=${CLUSTER_THRESHOLDS}"
echo "max_candidates=${MAX_CANDIDATES} kmeans_steps=${KMEANS_STEPS}"

for threshold in ${CLUSTER_THRESHOLDS}; do
  threshold_name="${threshold/./p}"
  output_dir="${RUN_ROOT}/threshold_${threshold_name}"
  echo "START threshold=${threshold} output=${output_dir}"
  python3 fdong_seq_compress/src/run_sparse_perplexity.py \
    --model-path "${MODEL_PATH}" \
    --text-path "${TEXT_PATH}" \
    --output-dir "${output_dir}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --attn-implementation eager \
    --max-tokens "${MAX_TOKENS}" \
    --decode-start "${DECODE_START}" \
    --methods "${METHODS}" \
    --local-window "${LOCAL_WINDOW}" \
    --max-candidates "${MAX_CANDIDATES}" \
    --always-include-positions "${ALWAYS_INCLUDE_POSITIONS}" \
    --sink-positions "${SINK_POSITIONS}" \
    --top-attention-ks "${TOP_ATTENTION_KS}" \
    --num-clusters "${NUM_CLUSTERS}" \
    --cluster-threshold "${threshold}" \
    --cluster-scale-sample-pairs "${CLUSTER_SCALE_SAMPLE_PAIRS}" \
    --min-selected-clusters "${MIN_SELECTED_CLUSTERS}" \
    --kmeans-steps "${KMEANS_STEPS}" \
    --random-seed "${RANDOM_SEED}"
  echo "DONE threshold=${threshold} output=${output_dir}"
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
    case = path.parent.name
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["case"] = case
            row["text_path"] = summary["text_path"]
            row["seq_len"] = summary["seq_len"]
            row["cluster_threshold"] = summary["args"]["cluster_threshold"]
            rows.append(row)
    timing_path = path.parent / "timing_by_method_layer.csv"
    if timing_path.exists():
        with timing_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["case"] = case
                row["text_path"] = summary["text_path"]
                row["seq_len"] = summary["seq_len"]
                row["cluster_threshold"] = summary["args"]["cluster_threshold"]
                timing_rows.append(row)

if rows:
    out = run_root / "aggregate_perplexity_by_method.csv"
    fieldnames = sorted({k for row in rows for k in row})
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote aggregate: {out}")

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
