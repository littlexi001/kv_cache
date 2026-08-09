#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
TEMPLATE="${TEMPLATE:-${PROJECT_ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
ROPE_PROJECT="${ROPE_PROJECT:-/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/results/20260731_qksieve_rope_retrieval_discovery_8gpu}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"
VARIANTS="${VARIANTS:-qksieve_keymse_requestlocal_fixedalloc_b8_fulltopk_k1280,qksieve_keymse_requestlocal_fixedalloc_post2xprererank_b8_fulltopk_k1280,qksieve_keymse_requestlocal_fixedalloc_post2xboundary75prererank_b8_fulltopk_k1280}"
SEED_BASE="${SEED_BASE:-256}"

mkdir -p "${RUN_ROOT}/logs"
cd "${PROJECT_ROOT}"
export PATH="/home/fdong/miniconda3/envs/moe/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  seed=$((SEED_BASE + gpu))
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    for history_tokens in 32768 65536; do
      output_dir="${RUN_ROOT}/${history_tokens}/seed${seed}"
      mkdir -p "${output_dir}"
      if [[ -f "${output_dir}/ALL_COMPLETE" ]]; then
        echo "SKIP completed: ${history_tokens}/seed${seed}"
        continue
      fi
      "${PYTHON}" -u src/run_qksieve_coldskip_longcontext_quality_20260730.py \
        --model_name_or_path "${MODEL}" \
        --template "${TEMPLATE}" \
        --output_dir "${output_dir}" \
        --history_tokens "${history_tokens}" \
        --eval_tokens 2 \
        --topic synthetic_rope \
        --synthetic_rope_seed "${seed}" \
        --synthetic_rope_source_root "${ROPE_PROJECT}" \
        --prefill_chunk_tokens 1024 \
        --dataset_cache_dir "${DATASET_CACHE_DIR}" \
        --seed "${seed}" \
        --dtype float16 \
        --device cuda \
        --device_map balanced \
        --max_memory_per_gpu_gib 22 \
        --variants "${VARIANTS}" \
        >"${output_dir}/run.log" 2>&1
    done
  ) >"${RUN_ROOT}/logs/seed${seed}.log" 2>&1 &
  pids+=("$!")
  echo "seed ${seed}: GPU ${gpu}, PID $!"
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi
touch "${RUN_ROOT}/ALL_COMPLETE"
