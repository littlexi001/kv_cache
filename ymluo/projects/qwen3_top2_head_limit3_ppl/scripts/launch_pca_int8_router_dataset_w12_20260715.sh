#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON_BIN="${PYTHON_BIN:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
SOURCE_ROOT="${ROOT}/results/20260715_pca_int8_budget_probe_32k_w0"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260715_pca_int8_scan_router_dataset_32k_w012}"

mkdir -p "${RUN_ROOT}"
for topic in sports medicine; do
  for fraction in 0.005 0.01 0.02; do
    source="${SOURCE_ROOT}/${topic}_w0_m32_f${fraction}"
    target="${RUN_ROOT}/${topic}_w0_f${fraction}"
    if [[ -d "${source}" && ! -d "${target}" ]]; then
      cp -a "${source}" "${target}"
    fi
  done
done

topics=(
  sports medicine sports medicine sports medicine
  sports medicine sports medicine sports medicine
)
windows=(1 1 1 1 1 1 2 2 2 2 2 2)
fractions=(0.005 0.005 0.01 0.01 0.02 0.02 0.005 0.005 0.01 0.01 0.02 0.02)

for gpu in {0..7}; do
  (
    for index in "${gpu}" "$((gpu + 8))"; do
      if (( index >= ${#topics[@]} )); then
        continue
      fi
      topic="${topics[$index]}"
      window="${windows[$index]}"
      fraction="${fractions[$index]}"
      name="${topic}_w${window}_f${fraction}"
      if [[ -f "${RUN_ROOT}/${name}/summary.json" ]]; then
        continue
      fi
      env \
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
          --budget_fractions "${fraction}" \
          --mass_estimator qabs_sampled_tail \
          --sample_fraction 0.0025 \
          --qabs_dim_count 16 \
          --candidate_fraction "${fraction}" \
          --qabs_use_cuda_kernels \
          --qabs_skip_candidate_rerank \
          --qabs_score_mode pca_int8 \
          --qabs_projection_dim 32 \
          --prefill_chunk_tokens 2048 \
          >"${RUN_ROOT}/${name}.log" 2>&1
    done
  ) </dev/null >"${RUN_ROOT}/gpu${gpu}_queue.log" 2>&1 &
  echo "launched queue pid=$! gpu=${gpu}"
done
