#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
TEMPLATE="${TEMPLATE:-${ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
OUTPUT="${OUTPUT:-${ROOT}/results/20260803_qksieve_affineres_closedloop_small_3gpu_v1}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"
VARIANTS="qksieve_keymse_requestlocal_valuesketch16i4_wometric_massfloor950_fulltopk_k1280,qksieve_keymse_requestlocal_valuesketch16i4_wometric_massfloor950_condres8_fulltopk_k1280,qksieve_keymse_requestlocal_valuesketch16i4_wometric_massfloor950_affineres_fulltopk_k1280"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
export QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=1.0
export QKSIEVE_PROFILE_STAGES=1

mkdir -p "${OUTPUT}/logs"
cd "${ROOT}"

run_case() {
  local gpu="$1"
  local topic="$2"
  local history="$3"
  local seed="$4"
  local name="${topic}${history}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    src/run_qksieve_coldskip_longcontext_quality_20260730.py \
    --model_name_or_path "${MODEL}" \
    --template "${TEMPLATE}" \
    --output_dir "${OUTPUT}/${name}" \
    --history_tokens "${history}" \
    --stream_reference_history_tokens "${history}" \
    --eval_tokens 4 \
    --topic "${topic}" \
    --seed "${seed}" \
    --repeat_topic_stream_if_short \
    --prefill_chunk_tokens 1024 \
    --protect_recent_tokens 0 \
    --dataset_cache_dir "${DATASET_CACHE_DIR}" \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    --max_memory_per_gpu_gib 22 \
    --variants "${VARIANTS}" \
    >"${OUTPUT}/logs/${name}.log" 2>&1
  touch "${OUTPUT}/${name}_COMPLETE"
}

run_case 0 religion 4096 20260901 & pid0=$!
run_case 1 sports 32000 20260902 & pid1=$!
run_case 2 medicine 32000 20260903 & pid2=$!

failed=0
for pid in "${pid0}" "${pid1}" "${pid2}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  touch "${OUTPUT}/FAILED"
  exit 1
fi
touch "${OUTPUT}/ALL_COMPLETE"
