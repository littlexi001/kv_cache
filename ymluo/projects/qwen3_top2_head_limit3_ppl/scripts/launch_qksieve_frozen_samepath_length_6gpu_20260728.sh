#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/20260728_qksieve_frozen_samepath_length_6gpu}"
SCORE_MODE=pca_hierarchical_autoqmsetotal15z_qkmetric_packed_fulltopk

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH=$ROOT/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
mkdir -p "$RUN_ROOT/logs"
cd "$ROOT"

run_case() {
  local length="$1"
  local horizon="$2"
  local devices="$3"
  local device_map="$4"
  local output="$RUN_ROOT/n${length}_g${horizon}"
  local log="$RUN_ROOT/logs/n${length}_g${horizon}.log"
  if [[ -s "$output/case_summary.json" ]]; then
    echo "[skip] length=$length horizon=$horizon"
    return
  fi
  CUDA_VISIBLE_DEVICES="$devices" "$PYTHON" -u \
    src/run_direct_countcap_denseprompt_ppl_20260725.py \
    --model_name_or_path "$MODEL" \
    --output_dir "$output" \
    --topics sports,medicine \
    --window_indices 0 \
    --methods full_attention,direct_countcap \
    --history_tokens "$length" \
    --eval_tokens "$horizon" \
    --direct_fraction 0.06 \
    --direct_min_tokens 256 \
    --direct_max_tokens 1280 \
    --projection_dim 128 \
    --sample_count 256 \
    --candidate_overfetch 1.0 \
    --protect_recent_tokens 0 \
    --direct_score_mode "$SCORE_MODE" \
    --qk_metric_query_shrinkage 0.75 \
    --prefill_chunk_tokens 2048 \
    --cache_mode preallocated \
    --dtype float16 \
    --device cuda \
    --device_map "$device_map" \
    >"$log" 2>&1
}

wait_phase() {
  local failed=0
  for pid in "$@"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "one or more same-path length cases failed; valid outputs remain" >&2
    exit 1
  fi
}

# Phase 1: six single-GPU jobs.
pids=()
for spec in \
  "16000 64 0" "16000 256 1" "16000 1024 2" \
  "32000 64 3" "32000 256 4" "32000 1024 5"; do
  read -r length horizon gpu <<<"$spec"
  run_case "$length" "$horizon" "$gpu" auto &
  pids+=("$!")
done
wait_phase "${pids[@]}"

# Phase 2: three disjoint two-GPU jobs.
pids=()
for spec in "64000 64 0,1" "64000 256 2,3" "64000 1024 4,5"; do
  read -r length horizon devices <<<"$spec"
  run_case "$length" "$horizon" "$devices" balanced &
  pids+=("$!")
done
wait_phase "${pids[@]}"

# Phase 3: reuse the same disjoint pairs for 128K.
pids=()
for spec in "128000 64 0,1" "128000 256 2,3" "128000 1024 4,5"; do
  read -r length horizon devices <<<"$spec"
  run_case "$length" "$horizon" "$devices" balanced &
  pids+=("$!")
done
wait_phase "${pids[@]}"

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u \
  src/benchmark_qksieve_fulltopk_breakdown_20260728.py \
  --lengths 16000,32000,64000,128000 \
  --warmup 8 \
  --iterations 40 \
  --output "$RUN_ROOT/attention_breakdown.json" \
  >"$RUN_ROOT/logs/attention_breakdown.log" 2>&1

"$PYTHON" - "$RUN_ROOT" "$MODEL" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
model_path = sys.argv[2]
project_root = root.parents[1]
expected_lengths = (16_000, 32_000, 64_000, 128_000)
expected_horizons = (64, 256, 1024)
score_mode = (
    "pca_hierarchical_autoqmsetotal15z_"
    "qkmetric_packed_fulltopk"
)

cells = {}
for length in expected_lengths:
    for horizon in expected_horizons:
        case_root = root / f"n{length}_g{horizon}"
        config = json.loads(
            (case_root / "config.json").read_text(encoding="utf-8")
        )
        rows = json.loads(
            (case_root / "case_summary.json").read_text(encoding="utf-8")
        )
        if config["direct_score_mode"] != score_mode:
            raise SystemExit(f"wrong score mode in {case_root}")
        if float(config["qk_metric_query_shrinkage"]) != 0.75:
            raise SystemExit(f"wrong shrinkage in {case_root}")
        if int(config["direct_min_tokens"]) != 256:
            raise SystemExit(f"wrong minimum budget in {case_root}")
        if int(config["direct_max_tokens"]) != 1280:
            raise SystemExit(f"wrong maximum budget in {case_root}")
        if float(config["candidate_overfetch"]) != 1.0:
            raise SystemExit(f"rerank/overfetch enabled in {case_root}")
        if int(config["protect_recent_tokens"]) != 0:
            raise SystemExit(f"recent reservation enabled in {case_root}")
        if int(config["projection_dim"]) != 128:
            raise SystemExit(f"wrong projection dimension in {case_root}")

        by_method = {}
        for row in rows:
            by_method.setdefault(row["method"], []).append(row)
        if set(by_method) != {"full_attention", "direct_countcap"}:
            raise SystemExit(f"method mismatch in {case_root}")
        if any(len(values) != 2 for values in by_method.values()):
            raise SystemExit(f"sports/medicine pair missing in {case_root}")
        for sparse_row in by_method["direct_countcap"]:
            if int(sparse_row["projection_dim"]) != 128:
                raise SystemExit(f"runtime projection drift in {case_root}")
            if int(sparse_row["packed_prefill_query_tokens"]) != 8:
                raise SystemExit(f"wrong Query calibration count in {case_root}")
            if not bool(sparse_row["packed_allocation_frozen"]):
                raise SystemExit(f"allocation is not frozen in {case_root}")
            if int(sparse_row["packed_index_rebuild_count"]) != 1:
                raise SystemExit(f"index rebuild count drift in {case_root}")
            if sparse_row["packed_transform"] != "qk_metric":
                raise SystemExit(f"wrong transform in {case_root}")
            if abs(
                float(sparse_row["packed_index_ratio_of_full_kv"])
                - 240.0 / 4096.0
            ) > 1.0e-6:
                raise SystemExit(f"physical index rate drift in {case_root}")

        def aggregate(method):
            values = by_method[method]
            mean_nll = sum(float(row["nll"]) for row in values) / len(values)
            return {
                "ppl": math.exp(min(20.0, mean_nll)),
                "steady_seconds_per_step": sum(
                    float(row["steady_sparse_seconds_per_step"])
                    for row in values
                )
                / len(values),
                "decode_seconds": sum(
                    float(row["sparse_decode_seconds"]) for row in values
                )
                / len(values),
                "dense_prompt_seconds": sum(
                    float(row["dense_prompt_seconds"]) for row in values
                )
                / len(values),
                "fixed_overhead_seconds": sum(
                    float(row["fixed_sparse_overhead_seconds"])
                    for row in values
                )
                / len(values),
                "configured_attention_tokens": sum(
                    float(row["configured_attention_tokens_mean"])
                    for row in values
                )
                / len(values),
                "actual_attention_tokens": sum(
                    float(row["actual_attention_tokens_mean"])
                    for row in values
                )
                / len(values),
                "attention_token_ratio": sum(
                    float(row["actual_attention_tokens_mean"])
                    / float(row["history_tokens"])
                    for row in values
                )
                / len(values),
                "index_ratio_of_full_kv": sum(
                    float(row["packed_index_ratio_of_full_kv"])
                    for row in values
                )
                / len(values),
                "peak_allocated_bytes_total": max(
                    int(row["peak_gpu_allocated_bytes_total"])
                    for row in values
                ),
                "peak_reserved_bytes_total": max(
                    int(row["peak_gpu_reserved_bytes_total"])
                    for row in values
                ),
            }

        full = aggregate("full_attention")
        sparse = aggregate("direct_countcap")
        per_step_saving = (
            full["steady_seconds_per_step"]
            - sparse["steady_seconds_per_step"]
        )
        fixed_extra = (
            sparse["fixed_overhead_seconds"]
            - full["fixed_overhead_seconds"]
        )
        cells[f"{length}@{horizon}"] = {
            "history_tokens": length,
            "decode_steps": horizon,
            "full": full,
            "qksieve": sparse,
            "quality_retention": full["ppl"] / sparse["ppl"],
            "steady_decode_speedup": (
                full["steady_seconds_per_step"]
                / sparse["steady_seconds_per_step"]
            ),
            "request_speedup_including_prefill_and_index": (
                (full["dense_prompt_seconds"] + full["decode_seconds"])
                / (
                    sparse["dense_prompt_seconds"]
                    + sparse["decode_seconds"]
                )
            ),
            "break_even_decode_steps": (
                max(0.0, fixed_extra) / per_step_saving
                if per_step_saving > 0
                else None
            ),
        }

source_paths = [
    project_root / "src/run_direct_countcap_denseprompt_ppl_20260725.py",
    project_root / "src/run_head_top2_targeted_ppl_20260714.py",
    project_root / "src/variablebit_spectral_cuda_20260727.py",
    project_root / "src/qabs_cuda_kernels.py",
    project_root / "src/qksieve_query_cuda_20260728.py",
    project_root / "src/benchmark_qksieve_fulltopk_breakdown_20260728.py",
    project_root / "src/preallocated_dynamic_cache_20260724.py",
]

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

summary = {
    "model_path": model_path,
    "frozen_method": {
        "method": "qksieve_fullprompt_auto_plain_fulltopk",
        "score_mode": score_mode,
        "query_shrinkage": 0.75,
        "query_tail_tokens": 8,
        "index_bits_per_token_per_kv_head": 240,
        "budget": "min(N, 1280, max(256, ceil(0.06*N)))",
        "proxy_topk_dtype": "float32",
        "exact_kv_dtype": "float16",
        "rerank": False,
        "fallback": False,
        "recent_or_sink_reservation": False,
    },
    "source_sha256": {
        str(path.relative_to(project_root)): sha256(path)
        for path in source_paths
    },
    "cells": cells,
    "attention_breakdown": json.loads(
        (root / "attention_breakdown.json").read_text(encoding="utf-8")
    ),
}
(root / "samepath_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

touch "$RUN_ROOT/ALL_COMPLETE"
