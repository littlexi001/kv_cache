#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/results/20260730_qksieve_global_keymse_template_3gpu}"
SCORE_MODE="pca_hierarchical_autokeytotal15z_qkmetric_packed_fulltopk"
GPU_SET="${GPU_SET:-2 5 6}"
TOPICS="${TOPICS:-sports medicine mixed_a}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/templates"
cd "${PROJECT_ROOT}"
export PATH="/home/fdong/miniconda3/envs/moe/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

read -r -a gpus <<<"${GPU_SET}"
read -r -a topics <<<"${TOPICS}"
if [[ "${#gpus[@]}" -ne "${#topics[@]}" ]]; then
  echo "GPU_SET and TOPICS must have the same number of entries" >&2
  exit 2
fi

pids=()
for index in "${!topics[@]}"; do
  topic="${topics[$index]}"
  gpu="${gpus[$index]}"
  output_dir="${RUN_ROOT}/${topic}"
  template_out="${RUN_ROOT}/templates/${topic}.pt"
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON}" -u src/run_direct_countcap_denseprompt_ppl_20260725.py \
      --model_name_or_path "${MODEL}" \
      --output_dir "${output_dir}" \
      --topics "${topic}" \
      --window_indices 0 \
      --methods direct_countcap \
      --history_tokens 32000 \
      --eval_tokens 16 \
      --direct_fraction 0.06 \
      --direct_min_tokens 256 \
      --direct_max_tokens 1280 \
      --projection_dim 128 \
      --sample_count 256 \
      --direct_score_mode "${SCORE_MODE}" \
      --qk_metric_query_shrinkage 0.75 \
      --packed_qmse_template_out "${template_out}" \
      --prefill_chunk_tokens 2048 \
      --cache_mode preallocated \
      --dataset_cache_dir "${DATASET_CACHE_DIR}" \
      --dtype float16 \
      --device cuda \
      --device_map balanced \
      >"${RUN_ROOT}/logs/${topic}.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

"${PYTHON}" -u src/build_global_qksieve_template_20260729.py \
  --inputs \
    "${RUN_ROOT}/templates/sports.pt" \
    "${RUN_ROOT}/templates/medicine.pt" \
    "${RUN_ROOT}/templates/mixed_a.pt" \
  --output "${RUN_ROOT}/templates/global32_3domain_keymse_runtime.pt" \
  --query_shrinkage 0.75 \
  | tee "${RUN_ROOT}/logs/build_global.log"

"${PYTHON}" - <<'PY' "${RUN_ROOT}/templates/global32_3domain_keymse_runtime.pt" "${RUN_ROOT}/summary.json"
import json
import sys
from pathlib import Path

import torch

template_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
template = torch.load(template_path, map_location="cpu", weights_only=False)
allocations = torch.cat(
    [template[layer]["allocation"].reshape(-1, 8) for layer in sorted(template)]
)
active = (allocations > 0).sum(dim=-1).float()
bits = allocations.sum(dim=-1).float()
index_bytes = 2.0 * bits + 2.0 * active
payload = {
    "schema": "qksieve_global_keymse_template_v1",
    "template": str(template_path),
    "layers": len(template),
    "head_allocations": int(allocations.shape[0]),
    "allocation_objective": sorted(
        {str(template[layer]["allocation_objective"]) for layer in template}
    ),
    "mean_index_bytes_per_token_kv_head": float(index_bytes.mean().item()),
    "index_ratio_of_full_fp16_kv": float(
        index_bytes.mean().item() / 512.0
    ),
    "mean_bits_by_band": allocations.float().mean(dim=0).tolist(),
}
output_path.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

touch "${RUN_ROOT}/ALL_COMPLETE"
