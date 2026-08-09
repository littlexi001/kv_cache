#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027/experiments/frozen_c64_20260807}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
MODEL="${MODEL:-/home/fdong/qksieve_iclr2027/models/Qwen3-4B-Instruct-2507}"
TEXT_FILE="${TEXT_FILE:-${ROOT}/src/run_head_top2_targeted_ppl_20260714.py}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/dynamic_short_recheck_20260807}"
GPU="${GPU:-0}"
REPEATS="${REPEATS:-3}"
EVAL_TOKENS="${EVAL_TOKENS:-32}"
VARIANT="qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64_k1280"

export PATH="/home/fdong/miniconda3/envs/nanogpt/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export QKSIEVE_QK_FACTOR_SOLVER=legacy
export QKSIEVE_QMSE_RATE_ALLOCATOR=torch
export QKSIEVE_PRELOAD_EXTENSIONS=1
export QKSIEVE_PRELOAD_QMSE_RATE_TABLES=1
export QKSIEVE_BUILD_RESIDENT_VALUE_SKETCH=1
export QKSIEVE_RESIDENT_VALUE_WORKERS=12
export QKSIEVE_PARALLEL_QK_WORKERS=36
export QKSIEVE_PARALLEL_VALUE_WORKERS=0
export QKSIEVE_FUSED_WOMETRIC_VALUE_APPEND=1
export QKSIEVE_BATCH_QMSE_ALLOCATION=1
export QKSIEVE_RESIDENT_VALUE_ATTENTION_WORKSPACE=1
export QKSIEVE_TILED_VALUE_ATTENTION=0
export QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=1.0
unset QKSIEVE_PROFILE_STAGES || true
unset QKSIEVE_EXACT_SELECTION_DIAGNOSTICS || true

if [[ ! -f "${MODEL}/config.json" ]]; then
  echo "Model is incomplete: ${MODEL}" >&2
  exit 1
fi
if [[ ! -f "${TEXT_FILE}" ]]; then
  echo "Frozen speed corpus is missing: ${TEXT_FILE}" >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}/logs"

write_manifest() {
  {
    echo "root=${ROOT}"
    echo "model=${MODEL}"
    echo "text_file=${TEXT_FILE}"
    echo "gpu=${GPU}"
    echo "repeats=${REPEATS}"
    echo "eval_tokens=${EVAL_TOKENS}"
    echo "variant=${VARIANT}"
    echo "expected_budget_8192=492"
    echo "expected_budget_16384=984"
    echo "expected_c64_samples_8192=1280"
    echo "expected_c64_samples_16384=1280"
    sha256sum \
      "${ROOT}/src/run_head_top2_targeted_ppl_20260714.py" \
      "${ROOT}/src/run_direct_countcap_denseprompt_ppl_20260725.py" \
      "${ROOT}/src/run_qksieve_coldskip_longcontext_quality_20260730.py" \
      "${ROOT}/src/profile_qksieve_realmodel_index_build_20260807.py"
  } >"${RUN_ROOT}/manifest.txt"
}

run_one() {
  local history_tokens="$1"
  local repeat="$2"
  local output="${RUN_ROOT}/n${history_tokens}/r${repeat}/legacy"
  local log="${RUN_ROOT}/logs/n${history_tokens}_r${repeat}.log"
  if [[ -f "${output}/ALL_COMPLETE" ]]; then
    return
  fi
  mkdir -p "${output}/quality"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  QKSIEVE_PROFILE_OUTPUT="${output}/index_profile.json" \
    "${PYTHON}" -u \
      "${ROOT}/src/profile_qksieve_realmodel_index_build_20260807.py" \
      --model_name_or_path "${MODEL}" \
      --template "${ROOT}/nonexistent_requestlocal_template.pt" \
      --output_dir "${output}/quality" \
      --history_tokens "${history_tokens}" \
      --stream_reference_history_tokens "${history_tokens}" \
      --eval_tokens "${EVAL_TOKENS}" \
      --text_file "${TEXT_FILE}" \
      --repeat_topic_stream_if_short \
      --prefill_chunk_tokens 1024 \
      --protect_recent_tokens 0 \
      --dataset_cache_dir "${ROOT}/datasets" \
      --seed "$((20260807 + history_tokens / 1024))" \
      --dtype float16 \
      --device cuda \
      --device_map auto \
      --max_memory_per_gpu_gib 22 \
      --variants "${VARIANT}" \
      >"${log}" 2>&1
  touch "${output}/ALL_COMPLETE"
}

verify_run() {
  local history_tokens="$1"
  local repeat="$2"
  local expected_budget="$3"
  local expected_samples="$4"
  local summary="${RUN_ROOT}/n${history_tokens}/r${repeat}/legacy/quality/summary.json"
  "${PYTHON}" - "${summary}" "${expected_budget}" "${expected_samples}" <<'PY'
import json
import sys

path, expected_budget, expected_samples = sys.argv[1:]
payload = json.load(open(path, encoding="utf-8"))
rows = [row for row in payload["rows"] if row["method"] != "full_attention"]
assert len(rows) == 1, len(rows)
row = rows[0]
assert int(row["max_exact_tokens_per_head"]) == int(expected_budget), row
assert int(row["requested_quantile_sample_count_per_head"]) == int(expected_samples), row
assert abs(float(row["configured_attention_tokens_mean"]) - int(expected_budget)) < 2.0, row
print(
    f"VERIFIED history={payload['history_tokens']} "
    f"budget={row['max_exact_tokens_per_head']} "
    f"samples={row['requested_quantile_sample_count_per_head']}"
)
PY
}

write_manifest
for history_tokens in 8192 16384; do
  if [[ "${history_tokens}" -eq 8192 ]]; then
    expected_budget=492
  else
    expected_budget=984
  fi
  for ((repeat=1; repeat<=REPEATS; repeat++)); do
    run_one "${history_tokens}" "${repeat}"
    verify_run "${history_tokens}" "${repeat}" "${expected_budget}" 1280 \
      >>"${RUN_ROOT}/logs/verification.log"
  done
done

"${PYTHON}" \
  "${ROOT}/src/summarize_qksieve_optimized_length_sweep_20260807.py" \
  "${RUN_ROOT}" >"${RUN_ROOT}/logs/summarize.log" 2>&1
touch "${RUN_ROOT}/ALL_COMPLETE"
