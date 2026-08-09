#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
DATA_DIR="${DATA_DIR:-${ROOT}/experiments/frozen_c64_20260807/data/LongBench}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260810_qksieve_robust_multimodel_m10_v1}"
RUNNER="${ROOT}/src/run_sample_calibrated_longbench_20260717.py"
SUMMARIZER="${ROOT}/src/summarize_qksieve_robust_longbench_20260810.py"
COMBINER="${ROOT}/src/summarize_qksieve_robust_multimodel_20260810.py"
FROZEN_CONFIG="${ROOT}/configs/qksieve_robust_iclr2027_frozen_20260810.json"
DIRECT_DOWNLOADER="${ROOT}/src/download_hf_snapshot_direct_20260810.py"
METHOD="qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64"
TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p"
MAX_SAMPLES_PER_TASK="${MAX_SAMPLES_PER_TASK:-10}"
SAMPLE_OFFSET_PER_TASK="${SAMPLE_OFFSET_PER_TASK:-40}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

LLAMA_MODEL="${LLAMA_MODEL:-${ROOT}/models/Meta-Llama-3.1-8B-Instruct-ms}"
QWEN_MODEL="${QWEN_MODEL:-${ROOT}/models/Qwen3-4B-Instruct-2507}"
MISTRAL_MODEL="${MISTRAL_MODEL:-${ROOT}/models/Mistral-7B-Instruct-v0.3}"
MODEL_TAGS="${MODEL_TAGS:-llama31_8b,qwen3_4b,mistral_7b}"

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

mkdir -p "${RUN_ROOT}/logs"
touch "${RUN_ROOT}/RUNNING"
rm -f "${RUN_ROOT}/ALL_COMPLETE" "${RUN_ROOT}/FAILED"

fail() {
  touch "${RUN_ROOT}/FAILED"
  rm -f "${RUN_ROOT}/RUNNING"
  exit 1
}

model_selected() {
  case ",${MODEL_TAGS}," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

model_complete() {
  "${PYTHON}" - "$1" <<'PY' >/dev/null
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
if not (root / "config.json").is_file():
    raise SystemExit(1)
indices = list(root.glob("*.safetensors.index.json"))
if indices:
    index = json.loads(indices[0].read_text(encoding="utf-8"))
    files = {root / name for name in index["weight_map"].values()}
else:
    files = set(root.glob("*.safetensors")) | set(root.glob("pytorch_model*.bin"))
if not files or any(not path.is_file() or path.stat().st_size == 0 for path in files):
    raise SystemExit(1)
PY
}

ensure_model() {
  local path="$1" repo="$2"
  if model_complete "${path}"; then return; fi
  if "${PYTHON}" - "${repo}" "${path}" <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=sys.argv[1],
    local_dir=sys.argv[2],
    token=os.environ.get("HF_TOKEN"),
)
PY
  then
    model_complete "${path}" && return
  fi
  "${PYTHON}" "${DIRECT_DOWNLOADER}" \
    --repo_id "${repo}" \
    --local_dir "${path}" \
    --endpoint "${HF_ENDPOINT}" \
    --endpoint https://huggingface.co || return 1
  model_complete "${path}"
}

[[ -f "${DATA_DIR}/manifest.json" ]] || { echo "missing LongBench data"; fail; }
IFS=',' read -r -a selected_models <<<"${MODEL_TAGS}"
[[ "${#selected_models[@]}" -gt 0 ]] || { echo "MODEL_TAGS is empty" >&2; fail; }
for tag in "${selected_models[@]}"; do
  case "${tag}" in
    llama31_8b|qwen3_4b|mistral_7b) ;;
    *) echo "unsupported MODEL_TAGS entry: ${tag}" >&2; fail ;;
  esac
done
if model_selected llama31_8b; then
  ensure_model "${LLAMA_MODEL}" "meta-llama/Meta-Llama-3.1-8B-Instruct" || fail
fi
if model_selected qwen3_4b; then
  ensure_model "${QWEN_MODEL}" "Qwen/Qwen3-4B-Instruct-2507" || fail
fi
if model_selected mistral_7b; then
  ensure_model "${MISTRAL_MODEL}" "mistralai/Mistral-7B-Instruct-v0.3" || fail
fi

IFS=',' read -r -a gpu_list <<<"${GPUS}"
if [[ "${#gpu_list[@]}" -ne 8 ]]; then
  echo "formal protocol requires exactly eight GPUs" >&2
  fail
fi

run_model() {
  local tag="$1" model="$2" wrapper="$3"
  local output="${RUN_ROOT}/${tag}"
  mkdir -p "${output}/logs"
  {
    echo "schema=qksieve_robust_multimodel_longbench_v1"
    echo "source_tree_commit=$(git -C "${ROOT}" rev-parse HEAD 2>/dev/null || echo unavailable)"
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
    echo "model=${model}"
    echo "prompt_wrapper=${wrapper}"
    echo "sample_offset_per_task=${SAMPLE_OFFSET_PER_TASK}"
    echo "max_samples_per_task=${MAX_SAMPLES_PER_TASK}"
    echo "method=${METHOD}"
    echo "max_quantile_samples=${QKSIEVE_MAX_QUANTILE_SAMPLE_COUNT}"
    echo "value_tail_alpha=${QKSIEVE_VALUE_SKETCH_TAIL_ALPHA}"
    sha256sum \
      "${model}/config.json" \
      "${DATA_DIR}/manifest.json" \
      "${FROZEN_CONFIG}" \
      "${ROOT}/src/run_head_top2_targeted_ppl_20260714.py" \
      "${RUNNER}" "${SUMMARIZER}" "${DIRECT_DOWNLOADER}"
  } >"${output}/manifest.txt" || return 1

  if [[ ! -f "${output}/smoke/ALL_COMPLETE" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu_list[0]}" "${PYTHON}" -u "${RUNNER}" \
      --model_name_or_path "${model}" \
      --longbench_data_dir "${DATA_DIR}" \
      --output_dir "${output}/smoke" \
      --tasks narrativeqa \
      --methods "full_kv,${METHOD}" \
      --max_samples_per_task 1 \
      --sample_offset_per_task "${SAMPLE_OFFSET_PER_TASK}" \
      --num_shards 1 --shard_index 0 \
      --max_prompt_tokens 7500 \
      --prompt_truncation_mode official_middle \
      --official_query_tail_tokens 8 \
      --max_new_tokens_override 8 \
      --prefill_chunk_tokens 2048 \
      --prompt_wrapper "${wrapper}" \
      --minimum_sparse_prefix_tokens 0 \
      --dtype float16 --device cuda --device_map auto \
      --max_memory_per_gpu_gib 22 \
      >"${output}/logs/smoke.log" 2>&1 || return 1
    touch "${output}/smoke/ALL_COMPLETE"
  fi

  local pids=() status=0 shard
  for shard in 0 1 2 3 4 5 6 7; do
    if [[ -f "${output}/shard${shard}/ALL_COMPLETE" ]]; then continue; fi
    mkdir -p "${output}/shard${shard}"
    CUDA_VISIBLE_DEVICES="${gpu_list[$shard]}" "${PYTHON}" -u "${RUNNER}" \
      --model_name_or_path "${model}" \
      --longbench_data_dir "${DATA_DIR}" \
      --output_dir "${output}/shard${shard}" \
      --tasks "${TASKS}" \
      --methods "full_kv,${METHOD}" \
      --max_samples_per_task "${MAX_SAMPLES_PER_TASK}" \
      --sample_offset_per_task "${SAMPLE_OFFSET_PER_TASK}" \
      --num_shards 8 --shard_index "${shard}" \
      --max_prompt_tokens 7500 \
      --prompt_truncation_mode official_middle \
      --official_query_tail_tokens 8 \
      --max_new_tokens_override 0 \
      --prefill_chunk_tokens 2048 \
      --prompt_wrapper "${wrapper}" \
      --minimum_sparse_prefix_tokens 0 \
      --dtype float16 --device cuda --device_map auto \
      --max_memory_per_gpu_gib 22 \
      >"${output}/logs/shard${shard}.log" 2>&1 && \
      touch "${output}/shard${shard}/ALL_COMPLETE" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
  [[ "${status}" -eq 0 ]] || return 1

  local expected_pairs=$((16 * MAX_SAMPLES_PER_TASK))
  "${PYTHON}" "${SUMMARIZER}" \
    --run_root "${output}" \
    --expected_pairs "${expected_pairs}" \
    --expected_tasks 16 \
    --bootstrap_resamples 10000 \
    --seed 20260810 \
    >"${output}/logs/summarize.log" 2>&1 || return 1
  touch "${output}/ALL_COMPLETE"
}

if model_selected llama31_8b; then
  run_model llama31_8b "${LLAMA_MODEL}" llama3 || fail
fi
if model_selected qwen3_4b; then
  run_model qwen3_4b "${QWEN_MODEL}" qwen3 || fail
fi
if model_selected mistral_7b; then
  run_model mistral_7b "${MISTRAL_MODEL}" tokenizer_chat || fail
fi

"${PYTHON}" "${COMBINER}" \
  --run_root "${RUN_ROOT}" \
  --models "${MODEL_TAGS}" \
  --expected_pairs $((16 * MAX_SAMPLES_PER_TASK)) \
  --expected_tasks 16 \
  >"${RUN_ROOT}/logs/combine.log" 2>&1 || fail

rm -f "${RUN_ROOT}/RUNNING" "${RUN_ROOT}/FAILED"
touch "${RUN_ROOT}/ALL_COMPLETE"
