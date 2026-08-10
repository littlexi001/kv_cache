#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027}"
PYTHON="${PYTHON:-/home/fdong/qksieve_env_torch271_py310/bin/python}"
MODEL="${MODEL:-${ROOT}/models/Yarn-Llama-2-7b-128k}"
CASE_RUNNER="${ROOT}/scripts/run_qksieve_persistent_kv_case_20260810.sh"
SUMMARY_RUNNER="${ROOT}/src/summarize_qksieve_persistent_kv_20260810.py"
SOURCE_MANIFEST="${ROOT}/configs/qksieve_robust_source_manifest_20260810.json"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260810_qksieve_persistent_kv_v3_multiseed}"
SEEDS="${SEEDS:-20260810,20260811,20260812}"
GPUS_32K="${GPUS_32K:-4,5}"
GPUS_64K="${GPUS_64K:-4,5,6}"

export PATH="$(dirname "${PYTHON}"):/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

mkdir -p "${RUN_ROOT}/logs"
touch "${RUN_ROOT}/RUNNING"
rm -f "${RUN_ROOT}/ALL_COMPLETE" "${RUN_ROOT}/FAILED"

fail() {
  touch "${RUN_ROOT}/FAILED"
  rm -f "${RUN_ROOT}/RUNNING"
  exit 1
}

for path in "${PYTHON}" "${MODEL}/config.json" "${CASE_RUNNER}" \
  "${SUMMARY_RUNNER}" "${SOURCE_MANIFEST}"; do
  if [[ ! -s "${path}" ]]; then
    echo "Missing required artifact: ${path}" >&2
    fail
  fi
done

"${PYTHON}" - "${ROOT}" "${SOURCE_MANIFEST}" <<'PY' || fail
import hashlib
import json
import sys
from pathlib import Path

import numpy
import torch
import transformers

root, manifest_path = map(Path, sys.argv[1:3])
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
if sys.version_info[:2] != (3, 10):
    raise AssertionError(f"software stack drifted: python={observed['python']}")
for name, value in expected.items():
    if observed[name] != value:
        raise AssertionError(f"software stack drifted: {name}={observed[name]}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("schema") != "qksieve_frozen_source_manifest_v1":
    raise AssertionError("frozen source manifest schema drifted")
for relative, expected_hash in manifest["files"].items():
    path = root / relative
    if not path.is_file():
        raise AssertionError(f"frozen source is missing: {relative}")
    observed_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed_hash != expected_hash:
        raise AssertionError(f"frozen source drifted: {relative}")
print(observed)
PY

declare -A MODEL_HASHES=(
  [config.json]=bf8239b8842439a1149effb9af58e5eba5db867d414abaf4c071b3ba48a6a215
  [pytorch_model-00001-of-00002.bin]=e15eff64c7ef2159ecd7228424d4d3ba813e9bcda2f6cb543accbe5028bd0ae0
  [pytorch_model-00002-of-00002.bin]=0f85245cab4358e94a5cadce299ddb16964a22d86eece081caa5e05616f3828a
  [pytorch_model.bin.index.json]=e572e08c4d4e81c7916197f6fcd2956a2f05e5919f28d72c9ba4f351efae1e29
  [tokenizer.model]=9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347
)
for name in "${!MODEL_HASHES[@]}"; do
  observed=$(sha256sum "${MODEL}/${name}" | awk '{print $1}')
  if [[ "${observed}" != "${MODEL_HASHES[${name}]}" ]]; then
    echo "Model artifact drifted: ${name}" >&2
    fail
  fi
done

{
  echo "schema=qksieve_persistent_multiseed_protocol_v1"
  echo "seeds=${SEEDS}"
  echo "gpus_32k=${GPUS_32K}"
  echo "gpus_64k=${GPUS_64K}"
  "${PYTHON}" - "${SOURCE_MANIFEST}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"audited_implementation_commit_sha={payload['audited_implementation_commit_sha']}")
PY
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
  sha256sum "${SOURCE_MANIFEST}" "${CASE_RUNNER}" "${SUMMARY_RUNNER}" "$0"
  for name in "${!MODEL_HASHES[@]}"; do
    sha256sum "${MODEL}/${name}"
  done
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
    --format=csv,noheader,nounits
} >"${RUN_ROOT}/manifest.txt" || fail

run_pair() {
  local devices="$1" length="$2" seed="$3" method
  for method in full qksieve_robust; do
    CUDA_VISIBLE_DEVICES="${devices}" \
    GPU_TAG="${devices//,/-}" \
    ROOT="${ROOT}" \
    RUN_ROOT="${RUN_ROOT}" \
    PYTHON="${PYTHON}" \
    MODEL="${MODEL}" \
    HISTORY_TOKENS="${length}" \
    METHOD="${method}" \
    SEED="${seed}" \
    BRANCH_COUNT=4 \
    BRANCH_STEPS=32 \
    APPEND_STEPS=128 \
    bash "${CASE_RUNNER}" || return 1
  done
}

IFS=',' read -r -a seeds <<<"${SEEDS}"
if [[ "${#seeds[@]}" -lt 3 ]]; then
  echo "Persistent evidence requires at least three seeds." >&2
  fail
fi
for seed in "${seeds[@]}"; do
  run_pair "${GPUS_32K}" 32768 "${seed}" || fail
  run_pair "${GPUS_64K}" 65536 "${seed}" || fail
done

"${PYTHON}" "${SUMMARY_RUNNER}" \
  --run_root "${RUN_ROOT}" \
  --output "${RUN_ROOT}/summary.json" \
  >"${RUN_ROOT}/logs/summary.log" 2>&1 || fail

rm -f "${RUN_ROOT}/RUNNING" "${RUN_ROOT}/FAILED"
touch "${RUN_ROOT}/ALL_COMPLETE"
echo "ALL_COMPLETE: ${RUN_ROOT}"
