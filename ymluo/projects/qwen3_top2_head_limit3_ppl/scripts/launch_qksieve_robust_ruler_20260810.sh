#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
MODEL="${MODEL:-${ROOT}/models/Meta-Llama-3.1-8B-Instruct-ms}"
LM_EVAL="${LM_EVAL:-${ROOT}/third_party/lm-evaluation-harness}"
LM_EVAL_REPO="https://github.com/EleutherAI/lm-evaluation-harness.git"
LM_EVAL_COMMIT="8c05cfe04fafcdd41dd64019f2b3797ef54dcd81"
HOT_POT="${ROOT}/data/ruler_sources/hotpotqa/distractor/validation-00000-of-00001.parquet"
DATA_ROOT="${ROOT}/data/qksieve_robust_ruler_20260810"
SHORT_DATA="${DATA_ROOT}/llama31_8b_ruler13_4k32k_m10_seed42.jsonl"
LONG_DATA="${DATA_ROOT}/llama31_8b_ruler13_64k128k_m5_seed42.jsonl"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260810_qksieve_robust_ruler_v1}"
RUNNER="${ROOT}/src/run_sample_calibrated_ruler_20260717.py"
PREPARE="${ROOT}/src/prepare_hierarchical_ruler_data_20260716.py"
SUMMARIZER="${ROOT}/src/summarize_qksieve_robust_ruler_20260810.py"
METHOD="qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64"
TASKS="niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_squad,qa_hotpot"
SHORT_LENGTHS="4096,8192,16384,32768"
LONG_LENGTHS="65536,131072"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

export PATH="$(dirname "${PYTHON}"):/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
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

mkdir -p "${RUN_ROOT}/logs" "${DATA_ROOT}" "$(dirname "${HOT_POT}")"
touch "${RUN_ROOT}/RUNNING"
rm -f "${RUN_ROOT}/ALL_COMPLETE" "${RUN_ROOT}/FAILED"

fail() {
  touch "${RUN_ROOT}/FAILED"
  rm -f "${RUN_ROOT}/RUNNING"
  exit 1
}

if [[ ! -f "${MODEL}/config.json" ]]; then
  echo "missing model: ${MODEL}" >&2
  fail
fi

if [[ ! -d "${LM_EVAL}/.git" ]]; then
  mkdir -p "$(dirname "${LM_EVAL}")"
  git clone "${LM_EVAL_REPO}" "${LM_EVAL}" \
    >"${RUN_ROOT}/logs/lm_eval_clone.log" 2>&1 || fail
fi
git -C "${LM_EVAL}" fetch origin "${LM_EVAL_COMMIT}" \
  >"${RUN_ROOT}/logs/lm_eval_fetch.log" 2>&1 || fail
git -C "${LM_EVAL}" checkout --detach "${LM_EVAL_COMMIT}" \
  >"${RUN_ROOT}/logs/lm_eval_checkout.log" 2>&1 || fail

if [[ ! -s "${HOT_POT}" ]]; then
  "${PYTHON}" - "${HOT_POT}" <<'PY' \
    >"${RUN_ROOT}/logs/hotpot_download.log" 2>&1 || fail
import sys
import os
from pathlib import Path

import pyarrow.parquet as pq
import requests

target = Path(sys.argv[1])
target.parent.mkdir(parents=True, exist_ok=True)
endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
urls = [
    f"{endpoint}/datasets/hotpotqa/hotpot_qa/resolve/main/"
    "distractor/validation-00000-of-00001.parquet",
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
    "distractor/validation-00000-of-00001.parquet",
]
partial = target.with_suffix(target.suffix + ".part")
last_error = None
for url in dict.fromkeys(urls):
    try:
        with requests.get(
            url,
            stream=True,
            timeout=(20, 300),
            allow_redirects=True,
        ) as response:
            response.raise_for_status()
            with partial.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=8 << 20):
                    if chunk:
                        stream.write(chunk)
        if partial.stat().st_size < 1_000_000:
            raise RuntimeError("downloaded HotpotQA parquet is unexpectedly small")
        metadata = pq.ParquetFile(partial).metadata
        if metadata.num_rows <= 0:
            raise RuntimeError("downloaded HotpotQA parquet has no rows")
        partial.replace(target)
        print(
            f"downloaded {target} ({target.stat().st_size} bytes, "
            f"{metadata.num_rows} rows)"
        )
        break
    except Exception as error:
        last_error = error
        partial.unlink(missing_ok=True)
else:
    raise RuntimeError("all HotpotQA direct downloads failed") from last_error
PY
fi

prepare_data() {
  local output="$1" lengths="$2" samples="$3"
  "${PYTHON}" "${PREPARE}" \
    --model_name_or_path "${MODEL}" \
    --lm_eval_path "${LM_EVAL}" \
    --output "${output}" \
    --ruler_tasks "${TASKS}" \
    --ruler_lengths "${lengths}" \
    --max_samples_per_task "${samples}" \
    --max_new_tokens_override 0 \
    --seed 42 \
    --ruler_hotpot_parquet "${HOT_POT}"
}

prepare_data "${SHORT_DATA}" "${SHORT_LENGTHS}" 10 \
  >"${RUN_ROOT}/logs/prepare_short.log" 2>&1 || fail
prepare_data "${LONG_DATA}" "${LONG_LENGTHS}" 5 \
  >"${RUN_ROOT}/logs/prepare_long.log" 2>&1 || fail

{
  echo "schema=qksieve_robust_ruler_protocol_v1"
  echo "git_commit=$(git -C "${ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "model=${MODEL}"
  echo "lm_eval_commit=$(git -C "${LM_EVAL}" rev-parse HEAD)"
  echo "tasks=${TASKS}"
  echo "short_lengths=${SHORT_LENGTHS}; samples=10"
  echo "long_lengths=${LONG_LENGTHS}; samples=5"
  echo "method=${METHOD}"
  echo "max_quantile_samples=${QKSIEVE_MAX_QUANTILE_SAMPLE_COUNT}"
  echo "value_tail_alpha=${QKSIEVE_VALUE_SKETCH_TAIL_ALPHA}"
  sha256sum \
    "${ROOT}/configs/qksieve_robust_iclr2027_frozen_20260810.json" \
    "${ROOT}/src/run_head_top2_targeted_ppl_20260714.py" \
    "${ROOT}/src/run_sample_calibrated_longbench_20260717.py" \
    "${RUNNER}" "${SUMMARIZER}" \
    "${SHORT_DATA}" "${SHORT_DATA}.manifest.json" \
    "${LONG_DATA}" "${LONG_DATA}.manifest.json"
} >"${RUN_ROOT}/manifest.txt" || fail

run_shard() {
  local output="$1" devices="$2" data="$3" lengths="$4"
  local samples="$5" shards="$6" shard="$7" device_map="$8"
  if [[ -f "${output}/ALL_COMPLETE" ]]; then
    return
  fi
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${devices}" "${PYTHON}" -u "${RUNNER}" \
    --model_name_or_path "${MODEL}" \
    --examples_jsonl "${data}" \
    --output_dir "${output}" \
    --methods "full_kv,${METHOD}" \
    --ruler_tasks "${TASKS}" \
    --ruler_lengths "${lengths}" \
    --max_samples_per_task "${samples}" \
    --num_shards "${shards}" \
    --shard_index "${shard}" \
    --max_new_tokens_override 0 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper llama3 \
    --minimum_sparse_prefix_tokens 0 \
    --dtype float16 \
    --device cuda \
    --device_map "${device_map}" \
    >"${RUN_ROOT}/logs/$(basename "${output}").log" 2>&1 || return 1
  touch "${output}/ALL_COMPLETE"
}

IFS=',' read -r -a gpu_list <<<"${GPUS}"
if [[ "${#gpu_list[@]}" -ne 8 ]]; then
  echo "formal protocol requires exactly eight GPUs" >&2
  fail
fi

if [[ ! -f "${RUN_ROOT}/smoke/ALL_COMPLETE" ]]; then
  run_shard "${RUN_ROOT}/smoke" "${gpu_list[0]}" "${SHORT_DATA}" \
    4096 1 1 0 auto || fail
fi

pids=()
for shard in 0 1 2 3 4 5 6 7; do
  run_shard "${RUN_ROOT}/shard${shard}" "${gpu_list[$shard]}" \
    "${SHORT_DATA}" "${SHORT_LENGTHS}" 10 8 "${shard}" auto &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
[[ "${status}" -eq 0 ]] || fail

devices_a="${gpu_list[0]},${gpu_list[1]},${gpu_list[2]},${gpu_list[3]}"
devices_b="${gpu_list[4]},${gpu_list[5]},${gpu_list[6]},${gpu_list[7]}"
run_shard "${RUN_ROOT}/shard8" "${devices_a}" "${LONG_DATA}" \
  "${LONG_LENGTHS}" 5 2 0 balanced & p0=$!
run_shard "${RUN_ROOT}/shard9" "${devices_b}" "${LONG_DATA}" \
  "${LONG_LENGTHS}" 5 2 1 balanced & p1=$!
status=0
wait "${p0}" || status=1
wait "${p1}" || status=1
[[ "${status}" -eq 0 ]] || fail

"${PYTHON}" "${SUMMARIZER}" \
  --run_root "${RUN_ROOT}" \
  --expected_tasks "${TASKS}" \
  --expected_length_samples \
    "4096:10,8192:10,16384:10,32768:10,65536:5,131072:5" \
  --bootstrap_resamples 10000 \
  --seed 20260810 \
  >"${RUN_ROOT}/logs/summarize.log" 2>&1 || fail

rm -f "${RUN_ROOT}/RUNNING" "${RUN_ROOT}/FAILED"
touch "${RUN_ROOT}/ALL_COMPLETE"
