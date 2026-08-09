#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"
TEMPLATE_ROOT="${TEMPLATE_ROOT:-${ROOT}/results/20260801_llama31_qksieve_keymse_template_gpu23}"
DEPLOYMENT_ROOT="${DEPLOYMENT_ROOT:-${ROOT}/results/20260801_qksieve_deployment_matched_longbench_m10_gpu23}"
SCORE_MODE=pca_hierarchical_autokeytotal15z_qkmetric_packed_fulltopk

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
mkdir -p "${TEMPLATE_ROOT}/logs" "${TEMPLATE_ROOT}/templates"
cd "${ROOT}"

calibrate() {
  local gpu="$1"
  local topic="$2"
  local output_dir="${TEMPLATE_ROOT}/${topic}"
  local template_out="${TEMPLATE_ROOT}/templates/${topic}.pt"
  if [[ -f "${template_out}" ]]; then
    echo "SKIP template ${topic}"
    return
  fi
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    src/run_direct_countcap_denseprompt_ppl_20260725.py \
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
    >"${TEMPLATE_ROOT}/logs/${topic}.log" 2>&1
}

calibrate 2 sports &
pid_sports=$!
calibrate 3 medicine &
pid_medicine=$!
wait "${pid_sports}"
wait "${pid_medicine}"
calibrate 2 mixed_a

global_template="${TEMPLATE_ROOT}/templates/llama31_global32_3domain_keymse_runtime.pt"
"${PYTHON}" -u src/build_global_qksieve_template_20260729.py \
  --inputs \
  "${TEMPLATE_ROOT}/templates/sports.pt" \
  "${TEMPLATE_ROOT}/templates/medicine.pt" \
  "${TEMPLATE_ROOT}/templates/mixed_a.pt" \
  --output "${global_template}" \
  --query_shrinkage 0.75 \
  >"${TEMPLATE_ROOT}/logs/build_global.log" 2>&1

"${PYTHON}" - "${global_template}" "${TEMPLATE_ROOT}/summary.json" <<'PY'
import json
import sys
from pathlib import Path

import torch

template_path = Path(sys.argv[1])
template = torch.load(template_path, map_location="cpu", weights_only=False)
layers = sorted(int(layer) for layer in template)
if layers != list(range(32)):
    raise ValueError(f"expected Llama layers 0..31, got {layers}")
allocations = torch.cat(
    [template[layer]["allocation"].reshape(-1, 8) for layer in layers]
)
active = (allocations > 0).sum(dim=-1).float()
bits = allocations.sum(dim=-1).float()
index_bytes = 2.0 * bits + 2.0 * active
payload = {
    "schema": "llama31_qksieve_global_keymse_template_v1",
    "template": str(template_path),
    "layers": len(layers),
    "mean_index_bytes_per_token_kv_head": float(index_bytes.mean()),
    "index_ratio_of_full_fp16_kv": float(index_bytes.mean() / 512.0),
    "mean_bits_by_band": allocations.float().mean(dim=0).tolist(),
}
Path(sys.argv[2]).write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
touch "${TEMPLATE_ROOT}/ALL_COMPLETE"

QKSIEVE_GPUS=2,3 \
TEMPLATE="${global_template}" \
RUN_ROOT="${DEPLOYMENT_ROOT}" \
bash scripts/launch_qksieve_deployment_matched_longbench_m10_2gpu_20260801.sh
