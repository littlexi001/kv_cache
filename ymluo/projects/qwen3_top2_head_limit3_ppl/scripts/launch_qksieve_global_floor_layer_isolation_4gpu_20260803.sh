#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
TEMPLATE="${TEMPLATE:-${ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
OUTPUT="${OUTPUT:-${ROOT}/results/20260803_qksieve_global_floor_layer_isolation_4gpu_v1}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"
VARIANT="qksieve_keymse_requestlocal_valuesketch16i4_wometric_residualrisk4_globalfloorrss25e4_safety1_fulltopk_k1280"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
export QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=1.0
export QKSIEVE_PROFILE_STAGES=1
export QKSIEVE_RECORD_LAYER_CANDIDATE_STATS=1

mkdir -p "${OUTPUT}/logs"
cd "${ROOT}"

run_case() {
  local gpu="$1"
  local name="$2"
  local layers="$3"
  local output_dir="${OUTPUT}/${name}"
  mkdir -p "${output_dir}"
  QKSIEVE_GLOBAL_FLOOR_RSS_LAYERS="${layers}" \
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    src/run_qksieve_coldskip_longcontext_quality_20260730.py \
    --model_name_or_path "${MODEL}" \
    --template "${TEMPLATE}" \
    --output_dir "${output_dir}" \
    --history_tokens 4096 \
    --stream_reference_history_tokens 4096 \
    --eval_tokens 2 \
    --topic religion \
    --seed 20260835 \
    --repeat_topic_stream_if_short \
    --prefill_chunk_tokens 1024 \
    --protect_recent_tokens 0 \
    --dataset_cache_dir "${DATASET_CACHE_DIR}" \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    --max_memory_per_gpu_gib 22 \
    --capture_layer_hidden_drift \
    --variants "${VARIANT}" \
    >"${OUTPUT}/logs/${name}.log" 2>&1
  touch "${OUTPUT}/${name}_COMPLETE"
}

run_case 0 layer0 0 & pid0=$!
run_case 1 layer1 1 & pid1=$!
run_case 2 layer2 2 & pid2=$!
run_case 3 layers0_1 0,1 & pid3=$!

failed=0
for pid in "${pid0}" "${pid1}" "${pid2}" "${pid3}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  touch "${OUTPUT}/FAILED"
  exit 1
fi
touch "${OUTPUT}/ALL_COMPLETE"
