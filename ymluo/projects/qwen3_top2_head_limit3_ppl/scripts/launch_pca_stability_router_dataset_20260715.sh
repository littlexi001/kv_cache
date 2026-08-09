#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON_BIN="${PYTHON_BIN:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
SOURCE_ROOT="${ROOT}/results/20260715_pca_int8_scan_router_dataset_32k_w012"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260715_pca_stability_router_dataset_32k_w012}"
mkdir -p "${RUN_ROOT}"

for topic in sports medicine; do
  for window in 0 1 2; do
    for fraction in 0.01 0.02; do
      source="${SOURCE_ROOT}/${topic}_w${window}_f${fraction}"
      target="${RUN_ROOT}/${topic}_w${window}_f${fraction}"
      if [[ ! -d "${target}" ]]; then
        cp -a "${source}" "${target}"
      fi
    done
  done
done

topics=(sports medicine sports medicine sports medicine)
windows=(0 0 1 1 2 2)
for gpu in {0..5}; do
  topic="${topics[$gpu]}"
  window="${windows[$gpu]}"
  name="${topic}_w${window}_f0.005"
  nohup env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${ROOT}/src" \
    PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin" \
    "${PYTHON_BIN}" "${ROOT}/src/run_adaptive_mass_budget_ppl_20260715.py" \
      --model_name_or_path "${MODEL}" \
      --output_dir "${RUN_ROOT}/${name}" \
      --topics "${topic}" \
      --window_indices "${window}" \
      --history_tokens 32000 \
      --query_tokens 256 \
      --eval_tokens 256 \
      --window_stride_tokens 32512 \
      --mass_thresholds 0.000001 \
      --budget_fractions 0.005 \
      --mass_estimator qabs_sampled_tail \
      --sample_fraction 0.0025 \
      --qabs_dim_count 16 \
      --candidate_fraction 0.005 \
      --qabs_use_cuda_kernels \
      --qabs_skip_candidate_rerank \
      --qabs_score_mode pca_int8 \
      --qabs_projection_dim 32 \
      --prefill_chunk_tokens 2048 \
      >"${RUN_ROOT}/${name}.log" 2>&1 </dev/null &
  echo "launched pid=$! gpu=${gpu} topic=${topic} window=${window}"
done
