#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
RUN_ROOT="${RUN_ROOT:-${BASE}/outputs/20260731_prerope_pca64k_discovery_8gpu}"
VARIANTS="${VARIANTS:-full_rope,local_global_postscore,prerope_pca32_int4_postscore,prerope_pca64_int2_postscore,prerope_pca64_int4_postscore}"
SEED_BASE="${SEED_BASE:-104}"

mkdir -p "${RUN_ROOT}/logs"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${BASE}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${BASE}"

pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  seed=$((SEED_BASE + gpu))
  output="${RUN_ROOT}/gpu${gpu}_seed${seed}"
  mkdir -p "${output}"
  if [[ -f "${output}/done.txt" ]]; then
    echo "SKIP completed GPU ${gpu}, seed ${seed}"
    continue
  fi
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${PYTHON}" -u src/run_local_global_rope_probe_8b.py \
      --model-name-or-path "${MODEL}" \
      --output-dir "${output}" \
      --lengths 65536 \
      --seed-start "${seed}" \
      --num-seeds 1 \
      --variants "${VARIANTS}" \
      --ratio 0.06 \
      --minimum-keep-tokens 256 \
      --maximum-keep-tokens 1280 \
      --local-window 128 \
      --sink-tokens 16 \
      --prefill-chunk-size 128 \
      --dtype bfloat16 \
      --load-in-4bit \
      --attn-implementation sdpa \
      --original-max-position-embeddings 40960 \
      --global-max-position 70000 \
      >"${output}/run.log" 2>&1
  ) >"${RUN_ROOT}/logs/gpu${gpu}.log" 2>&1 &
  pids+=("$!")
  echo "GPU ${gpu}: seed ${seed}, PID $!"
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi
touch "${RUN_ROOT}/ALL_COMPLETE"
