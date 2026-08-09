#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/results/20260730_qksieve_qmse_keymse_deploy_compare_3gpu}"
QMSE_TEMPLATE="${QMSE_TEMPLATE:-${PROJECT_ROOT}/results/20260729_qksieve_frozen_template_frontier/templates/global32_3domain_runtime.pt}"
KEYMSE_TEMPLATE="${KEYMSE_TEMPLATE:-${PROJECT_ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
QMSE_SCORE_MODE="pca_hierarchical_autoqmsetotal15z_qkmetric_qfused_gqa4_wmma_kappend_unbiased_packed_direct"
KEYMSE_SCORE_MODE="pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_kappend_unbiased_packed_direct"

mkdir -p "${RUN_ROOT}/logs"
cd "${PROJECT_ROOT}"
export PATH="/home/fdong/miniconda3/envs/moe/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_case() {
  local gpu="$1"
  local name="$2"
  local method="$3"
  local template="${4:-}"
  local score_mode="${5:-${QMSE_SCORE_MODE}}"
  local extra=()
  if [[ -n "${template}" ]]; then
    extra+=(--packed_qmse_template_in "${template}")
  fi
  mkdir -p "${RUN_ROOT}/${name}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON}" -u src/run_direct_countcap_denseprompt_ppl_20260725.py \
      --model_name_or_path "${MODEL}" \
      --output_dir "${RUN_ROOT}/${name}" \
      --topics mixed_b \
      --window_indices 0 \
      --methods "${method}" \
      --history_tokens 32000 \
      --eval_tokens 256 \
      --direct_fraction 0.06 \
      --direct_min_tokens 256 \
      --direct_max_tokens 1280 \
      --projection_dim 128 \
      --sample_count 256 \
      --direct_score_mode "${score_mode}" \
      --qk_metric_query_shrinkage 0.75 \
      --prefill_chunk_tokens 2048 \
      --cache_mode preallocated \
      --dataset_cache_dir "${DATASET_CACHE_DIR}" \
      --dtype float16 \
      --device cuda \
      --device_map balanced \
      "${extra[@]}" \
      >"${RUN_ROOT}/logs/${name}.log" 2>&1
}

run_case 0 full full_attention "" "${QMSE_SCORE_MODE}" &
pid0="$!"
run_case 1 qmse direct_countcap "${QMSE_TEMPLATE}" "${QMSE_SCORE_MODE}" &
pid1="$!"
run_case 2 keymse direct_countcap "${KEYMSE_TEMPLATE}" "${KEYMSE_SCORE_MODE}" &
pid2="$!"
wait "${pid0}" "${pid1}" "${pid2}"

"${PYTHON}" - <<'PY' "${RUN_ROOT}"
import csv
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = {}
for name in ("full", "qmse", "keymse"):
    path = root / name / "case_summary.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        values = list(csv.DictReader(handle))
    if len(values) != 1:
        raise RuntimeError(f"{name} expected one row, got {len(values)}")
    row = values[0]
    rows[name] = {
        key: float(row[key])
        for key in (
            "nll",
            "ppl",
            "steady_sparse_seconds_per_step",
            "packed_index_ratio_of_full_kv",
            "actual_attention_tokens_mean",
        )
        if row.get(key, "") != ""
    }
full_nll = rows["full"]["nll"]
full_ms = 1000.0 * rows["full"]["steady_sparse_seconds_per_step"]
for name, row in rows.items():
    row["steady_ms_per_token"] = (
        1000.0 * row["steady_sparse_seconds_per_step"]
    )
    row["quality_retention_vs_full"] = math.exp(full_nll - row["nll"])
    row["decode_speedup_vs_full"] = full_ms / row["steady_ms_per_token"]
payload = {
    "schema": "qksieve_qmse_keymse_deploy_compare_v1",
    "history_tokens": 32000,
    "eval_tokens": 256,
    "topic": "mixed_b",
    "score_mode": (
        "pca_hierarchical_autoqmsetotal15z_qkmetric_qfused_gqa4_"
        "wmma_kappend_unbiased_packed_direct"
    ),
    "rows": rows,
}
(root / "summary.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

touch "${RUN_ROOT}/ALL_COMPLETE"
