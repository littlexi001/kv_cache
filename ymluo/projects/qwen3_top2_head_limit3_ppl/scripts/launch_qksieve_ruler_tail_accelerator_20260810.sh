#!/usr/bin/env bash
set -uo pipefail

# Run only the strict-pair RULER examples absent from a completed-row snapshot.
# Reversing the missing list lets this auxiliary host complement a primary run
# that is still advancing in the original frozen-data order.
ROOT="${ROOT:-/data/u21307130306/qksieve_ruler_accel_20260810}"
PYTHON="${PYTHON:-/data/u21307130306/qksieve_iclr2027/venv/bin/python}"
MODEL="${MODEL:-${ROOT}/models/Meta-Llama-3.1-8B-Instruct-ms}"
FULL_DATA="${FULL_DATA:-${ROOT}/data/llama31_8b_ruler13_64k128k_m5_seed42.jsonl}"
COMPLETED_SNAPSHOT="${COMPLETED_SNAPSHOT:-${ROOT}/data/completed_snapshot.csv}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260810_qksieve_robust_ruler_tail_accel}"
RUNNER="${ROOT}/src/run_sample_calibrated_ruler_20260717.py"
FROZEN_CONFIG="${ROOT}/configs/qksieve_robust_iclr2027_frozen_20260810.json"
METHOD="qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64"
TASKS="niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_squad,qa_hotpot"
LENGTHS="65536,131072"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
FILTERED_DATA="${RUN_ROOT}/missing_reversed.jsonl"

export PATH="$(dirname "${PYTHON}"):/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export QKSIEVE_TRUST_REMOTE_CODE=0
export QKSIEVE_QK_FACTOR_SOLVER=legacy
export QKSIEVE_QMSE_RATE_ALLOCATOR=torch
export QKSIEVE_PRELOAD_EXTENSIONS=1
export QKSIEVE_PRELOAD_QMSE_RATE_TABLES=1
export QKSIEVE_PARALLEL_QK_WORKERS="${QKSIEVE_PARALLEL_QK_WORKERS:-8}"
export QKSIEVE_FUSED_WOMETRIC_VALUE_APPEND=1
export QKSIEVE_BATCH_QMSE_ALLOCATION=1
export QKSIEVE_RESIDENT_VALUE_ATTENTION_WORKSPACE=1
export QKSIEVE_TILED_VALUE_ATTENTION=0
export QKSIEVE_MAX_QUANTILE_SAMPLE_COUNT=512
export QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=0.5
export QKSIEVE_BUILD_RESIDENT_VALUE_SKETCH=1
export QKSIEVE_DEBUG_DISABLE_VALUE_SKETCH=0
unset QKSIEVE_PROFILE_STAGES QKSIEVE_EXACT_SELECTION_DIAGNOSTICS || true

fail() {
  touch "${RUN_ROOT}/FAILED"
  rm -f "${RUN_ROOT}/RUNNING"
  exit 1
}

mkdir -p "${RUN_ROOT}/logs"
touch "${RUN_ROOT}/RUNNING"
rm -f "${RUN_ROOT}/ALL_COMPLETE" "${RUN_ROOT}/FAILED"

for path in "${PYTHON}" "${MODEL}/config.json" "${FULL_DATA}" \
  "${COMPLETED_SNAPSHOT}" "${RUNNER}" "${FROZEN_CONFIG}"; do
  if [[ ! -s "${path}" ]]; then
    echo "Missing required artifact: ${path}" >&2
    fail
  fi
done

"${PYTHON}" - <<'PY' || fail
import sys

import numpy
import torch
import transformers

observed = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "numpy": numpy.__version__,
}
expected = {
    "torch": "2.7.1+cu126",
    "cuda": "12.6",
    "transformers": "4.53.1",
    "numpy": "2.2.6",
}
for name, value in expected.items():
    if observed[name] != value:
        raise AssertionError(f"software stack drifted: {name}={observed[name]}")
if sys.version_info[:2] != (3, 10):
    raise AssertionError(f"software stack drifted: python={observed['python']}")
if not torch.cuda.is_available():
    raise AssertionError("CUDA is unavailable")
print(observed)
PY

"${PYTHON}" - "${FULL_DATA}" "${COMPLETED_SNAPSHOT}" \
  "${FILTERED_DATA}" "${RUN_ROOT}/filter_audit.json" "${METHOD}" <<'PY' || fail
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

full_path, snapshot_path, output_path, audit_path = map(Path, sys.argv[1:5])
method = sys.argv[5]
expected_methods = {"full_kv", method}
with full_path.open(encoding="utf-8") as handle:
    examples = [json.loads(line) for line in handle if line.strip()]
completed = defaultdict(set)
with snapshot_path.open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        completed[(row["task"], row["sample_id"])].add(row["method"])
missing = [
    row
    for row in examples
    if completed[(row["task"], row["sample_id"])] != expected_methods
]
missing.reverse()
if not missing:
    raise AssertionError("snapshot already contains all long RULER pairs")
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as handle:
    for row in missing:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
audit = {
    "schema": "qksieve_ruler_tail_filter_v1",
    "source_examples": len(examples),
    "snapshot_rows": sum(len(value) for value in completed.values()),
    "snapshot_strict_pairs": sum(value == expected_methods for value in completed.values()),
    "missing_examples": len(missing),
    "order": "reverse_of_frozen_jsonl",
    "source_sha256": digest(full_path),
    "snapshot_sha256": digest(snapshot_path),
    "filtered_sha256": digest(output_path),
}
audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
print(json.dumps(audit, indent=2))
PY

{
  echo "schema=qksieve_ruler_tail_accelerator_protocol_v1"
  "${PYTHON}" - "${FROZEN_CONFIG}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"numerical_freeze_commit_sha={payload['numerical_freeze_commit_sha']}")
print(
    "audited_implementation_commit_sha="
    f"{payload['audited_implementation_commit_sha']}"
)
PY
  echo "model=${MODEL}"
  echo "method=${METHOD}"
  echo "gpus=${GPUS}"
  echo "order=reverse_of_frozen_jsonl"
  "${PYTHON}" - <<'PY'
import sys

import numpy
import torch
import transformers

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"transformers={transformers.__version__}")
print(f"numpy={numpy.__version__}")
PY
  sha256sum "${FROZEN_CONFIG}" "${RUNNER}" "${FULL_DATA}" \
    "${COMPLETED_SNAPSHOT}" "${FILTERED_DATA}" "$0"
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
    --format=csv,noheader,nounits
} >"${RUN_ROOT}/manifest.txt" || fail

IFS=',' read -r -a gpu_list <<<"${GPUS}"
if [[ "${#gpu_list[@]}" -ne 8 ]]; then
  echo "Tail acceleration requires exactly eight GPUs." >&2
  fail
fi

run_shard() {
  local shard="$1" devices="$2"
  local output="${RUN_ROOT}/shard${shard}"
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${devices}" "${PYTHON}" -u "${RUNNER}" \
    --model_name_or_path "${MODEL}" \
    --examples_jsonl "${FILTERED_DATA}" \
    --output_dir "${output}" \
    --methods "full_kv,${METHOD}" \
    --ruler_tasks "${TASKS}" \
    --ruler_lengths "${LENGTHS}" \
    --max_samples_per_task 5 \
    --num_shards 2 \
    --shard_index "${shard}" \
    --max_new_tokens_override 0 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper llama3 \
    --minimum_sparse_prefix_tokens 0 \
    --collect_attention_stats \
    --dtype float16 \
    --device cuda \
    --device_map balanced \
    >"${RUN_ROOT}/logs/shard${shard}.log" 2>&1
}

devices_a="${gpu_list[0]},${gpu_list[1]},${gpu_list[2]},${gpu_list[3]}"
devices_b="${gpu_list[4]},${gpu_list[5]},${gpu_list[6]},${gpu_list[7]}"
run_shard 0 "${devices_a}" & p0=$!
run_shard 1 "${devices_b}" & p1=$!
status=0
wait "${p0}" || status=1
wait "${p1}" || status=1
[[ "${status}" -eq 0 ]] || fail

"${PYTHON}" - "${RUN_ROOT}" "${FILTERED_DATA}" "${METHOD}" <<'PY' || fail
import csv
import json
import sys
from collections import Counter
from pathlib import Path

root, filtered_path = map(Path, sys.argv[1:3])
method = sys.argv[3]
with filtered_path.open(encoding="utf-8") as handle:
    examples = [json.loads(line) for line in handle if line.strip()]
expected = {
    (row["task"], row["sample_id"], candidate)
    for row in examples
    for candidate in ("full_kv", method)
}
rows = []
for path in sorted(root.glob("shard[01]/sample_results.csv")):
    with path.open(newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))
keys = [(row["task"], row["sample_id"], row["method"]) for row in rows]
counts = Counter(keys)
if set(keys) != expected:
    raise AssertionError(
        f"tail key mismatch: missing={len(expected - set(keys))}, "
        f"extra={len(set(keys) - expected)}"
    )
if any(value != 1 for value in counts.values()):
    raise AssertionError("tail output contains duplicate method rows")
if any(row.get("diagnostics_enabled") != "True" for row in rows):
    raise AssertionError("tail output lacks attention diagnostics")
audit = {
    "schema": "qksieve_ruler_tail_output_audit_v1",
    "examples": len(examples),
    "rows": len(rows),
    "strict_pairs": len(examples),
    "methods": sorted({row["method"] for row in rows}),
    "diagnostics_enabled": True,
}
(root / "output_audit.json").write_text(
    json.dumps(audit, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(audit, indent=2))
PY

rm -f "${RUN_ROOT}/RUNNING" "${RUN_ROOT}/FAILED"
touch "${RUN_ROOT}/ALL_COMPLETE"
echo "ALL_COMPLETE: ${RUN_ROOT}"
