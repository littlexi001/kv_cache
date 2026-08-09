#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
TEMPLATE="${TEMPLATE:-${PROJECT_ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
SOURCE_ROOT="${SOURCE_ROOT:-/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/results/20260731_qksieve_qwen8_rope_128k_2gpu_pairs}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"
HISTORY_TOKENS="${HISTORY_TOKENS:-131072}"
SEED_BASE="${SEED_BASE:-296}"
GPU_PAIRS="${GPU_PAIRS:-0,1;2,3;4,5;6,7}"
ORIGINAL_MAX_POSITION="${ORIGINAL_MAX_POSITION:-40960}"
GLOBAL_MAX_POSITION="${GLOBAL_MAX_POSITION:-163840}"
VARIANTS="${VARIANTS:-qksieve_keymse_requestlocal_fixedalloc_i112_41_fulltopk_k1280,qksieve_keymse_requestlocal_fixedalloc_post2xprererank_i112_41_l00to08_fulltopk_k1280}"

mkdir -p "${RUN_ROOT}/logs"
cd "${PROJECT_ROOT}"
export PATH="/home/fdong/miniconda3/envs/moe/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

IFS=";" read -r -a pairs <<<"${GPU_PAIRS}"
pids=()
for worker_index in "${!pairs[@]}"; do
  pair="${pairs[$worker_index]}"
  seed=$((SEED_BASE + worker_index))
  output="${RUN_ROOT}/${HISTORY_TOKENS}/seed${seed}"
  mkdir -p "${output}"
  if [[ -f "${output}/ALL_COMPLETE" ]]; then
    echo "SKIP seed=${seed}"
    continue
  fi
  (
    export CUDA_VISIBLE_DEVICES="${pair}"
    "${PYTHON}" -u src/run_qksieve_coldskip_longcontext_quality_20260730.py \
      --model_name_or_path "${MODEL}" \
      --template "${TEMPLATE}" \
      --output_dir "${output}" \
      --history_tokens "${HISTORY_TOKENS}" \
      --eval_tokens 2 \
      --topic synthetic_rope \
      --synthetic_rope_seed "${seed}" \
      --synthetic_rope_source_root "${SOURCE_ROOT}" \
      --prefill_chunk_tokens 1024 \
      --dataset_cache_dir "${DATASET_CACHE_DIR}" \
      --seed "${seed}" \
      --dtype float16 \
      --device cuda \
      --device_map balanced \
      --max_memory_per_gpu_gib 22 \
      --load_in_4bit \
      --original_max_position_embeddings "${ORIGINAL_MAX_POSITION}" \
      --global_max_position "${GLOBAL_MAX_POSITION}" \
      --variants "${VARIANTS}" \
      >"${output}/run.log" 2>&1
  ) >"${RUN_ROOT}/logs/seed${seed}.log" 2>&1 &
  pids+=("$!")
  echo "seed ${seed}: GPUs ${pair}, PID $!"
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi
touch "${RUN_ROOT}/ALL_COMPLETE"
