#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON_BIN="${PYTHON_BIN:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260715_scanonly_router_dataset_32k_w012}"

mkdir -p "${RUN_ROOT}"
topics=(sports sports sports medicine medicine medicine)
windows=(0 1 2 0 1 2)
fractions=(0.005 0.01 0.02)
pids=()

for gpu in 0 1 2 3 4 5 6 7; do
  (
    for ((job=gpu; job<18; job+=8)); do
      case_index=$((job / 3))
      fraction_index=$((job % 3))
      topic="${topics[$case_index]}"
      window="${windows[$case_index]}"
      fraction="${fractions[$fraction_index]}"
      name="${topic}_w${window}_f${fraction}"
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
          --prefill_chunk_tokens 2048 \
          >"${RUN_ROOT}/${name}.log" 2>&1
    done
  ) >"${RUN_ROOT}/gpu${gpu}_queue.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done
echo "all scan-only router dataset jobs completed"
