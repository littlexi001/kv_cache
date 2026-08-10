#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260810_qksieve_shrinkage_sensitivity_v1}"
GPUS="${GPUS:-0,1,2,3}"
LLAMA_MODEL="${LLAMA_MODEL:-${ROOT}/models/Meta-Llama-3.1-8B-Instruct-ms}"
QWEN_MODEL="${QWEN_MODEL:-${ROOT}/models/Qwen3-4B-Instruct-2507}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"
COLLECTOR="${ROOT}/src/collect_real_qk_trace_20260715.py"
ANALYZER="${ROOT}/src/analyze_qk_balanced_spectral_rate_20260727.py"
SUMMARIZER="${ROOT}/src/summarize_qksieve_shrinkage_sensitivity_20260810.py"
LABELS="qwen3_4b_sports32k,qwen3_4b_medicine32k,llama31_8b_sports32k,llama31_8b_medicine32k"
SHRINKAGES=(0.00 0.25 0.50 0.75 0.90)

export PATH="$(dirname "${PYTHON}"):/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

IFS=',' read -r -a gpu_list <<<"${GPUS}"
if (( ${#gpu_list[@]} < 4 )); then
  echo "Need four GPUs for the registered four-trace grid." >&2
  exit 2
fi
for path in "${LLAMA_MODEL}" "${QWEN_MODEL}" "${COLLECTOR}" "${ANALYZER}" "${SUMMARIZER}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 2
  fi
done
if [[ -f "${RUN_ROOT}/ALL_COMPLETE" ]]; then
  echo "Already complete: ${RUN_ROOT}"
  exit 0
fi

mkdir -p "${RUN_ROOT}/traces" "${RUN_ROOT}/analysis" "${RUN_ROOT}/logs"
rm -f "${RUN_ROOT}/FAILED"
{
  echo "schema=qksieve_shrinkage_sensitivity_protocol_v1"
  echo "timestamp=$(date -Is)"
  echo "root=${ROOT}"
  echo "python=${PYTHON}"
  echo "gpus=${GPUS}"
  echo "llama_model=${LLAMA_MODEL}"
  echo "qwen_model=${QWEN_MODEL}"
  echo "calibration_source=prefill_tail"
  echo "calibration_steps=8"
  echo "query_shrinkages=0,0.25,0.5,0.75,0.9"
  echo "selected_fractions=0.01,0.02,0.04"
  git -C "${ROOT}" rev-parse HEAD 2>/dev/null | sed 's/^/git_commit=/' || true
  sha256sum "${COLLECTOR}" "${ANALYZER}" "${SUMMARIZER}"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
  "${PYTHON}" - <<'PY'
import platform
import torch
import transformers
print("python_version=" + platform.python_version())
print("pytorch_version=" + torch.__version__)
print("transformers_version=" + transformers.__version__)
print("cuda_runtime=" + str(torch.version.cuda))
PY
} >"${RUN_ROOT}/manifest.txt"

labels=(
  qwen3_4b_sports32k
  qwen3_4b_medicine32k
  llama31_8b_sports32k
  llama31_8b_medicine32k
)
models=("${QWEN_MODEL}" "${QWEN_MODEL}" "${LLAMA_MODEL}" "${LLAMA_MODEL}")
topics=(sports medicine sports medicine)
layers=("0,8,17,26,35" "0,8,17,26,35" "0,8,16,24,31" "0,8,16,24,31")

trace_pids=()
for index in 0 1 2 3; do
  label="${labels[$index]}"
  trace="${RUN_ROOT}/traces/${label}.pt"
  if [[ -s "${trace}" ]]; then
    continue
  fi
  CUDA_VISIBLE_DEVICES="${gpu_list[$index]}" "${PYTHON}" -u "${COLLECTOR}" \
    --model_name_or_path "${models[$index]}" \
    --output_path "${trace}" \
    --topic "${topics[$index]}" \
    --history_tokens 32000 \
    --steps 64 \
    --layers "${layers[$index]}" \
    --prefill_chunk_tokens 2048 \
    --prefill_query_tail_tokens 8 \
    --dataset_cache_dir "${DATASET_CACHE_DIR}" \
    --omit_values \
    --dtype float16 --device cuda --device_map auto \
    >"${RUN_ROOT}/logs/trace_${label}.log" 2>&1 &
  trace_pids+=("$!")
done

failed=0
for pid in "${trace_pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if (( failed )); then
  touch "${RUN_ROOT}/FAILED"
  exit 1
fi

analysis_pids=()
for index in 0 1 2 3; do
  label="${labels[$index]}"
  (
    for shrinkage in "${SHRINKAGES[@]}"; do
      tag="lambda_${shrinkage/./p}"
      output="${RUN_ROOT}/analysis/${label}/${tag}"
      if [[ -s "${output}/summary.json" && -s "${output}/per_head.csv" ]]; then
        continue
      fi
      CUDA_VISIBLE_DEVICES="${gpu_list[$index]}" "${PYTHON}" -u "${ANALYZER}" \
        --trace_path "${RUN_ROOT}/traces/${label}.pt" \
        --output_dir "${output}" \
        --label "${label}" \
        --device cuda \
        --sample_stride 32 \
        --calibration_steps 8 \
        --calibration_source prefill_tail \
        --total_rate_budget 15 \
        --query_shrinkage "${shrinkage}" \
        --selected_fractions 0.01,0.02,0.04 \
        --top_fraction 0.02 \
        >"${RUN_ROOT}/logs/${label}_${tag}.log" 2>&1 || exit 1
    done
  ) &
  analysis_pids+=("$!")
done

failed=0
for pid in "${analysis_pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if (( failed )); then
  touch "${RUN_ROOT}/FAILED"
  exit 1
fi

"${PYTHON}" "${SUMMARIZER}" \
  --run_root "${RUN_ROOT}" \
  --labels "${LABELS}" \
  --shrinkages 0,0.25,0.5,0.75,0.9 \
  --fractions 0.01,0.02,0.04 \
  --bootstrap_samples 10000 \
  --output "${RUN_ROOT}/summary.json" \
  >"${RUN_ROOT}/logs/summarize.log" 2>&1 || {
    touch "${RUN_ROOT}/FAILED"
    exit 1
  }

touch "${RUN_ROOT}/ALL_COMPLETE"
echo "ALL_COMPLETE: ${RUN_ROOT}"
