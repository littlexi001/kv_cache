#!/usr/bin/env bash
set -euo pipefail

root="/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl"
python_bin="/home/fdong/miniconda3/envs/moe/bin/python"
model="/home/fdong/models/Qwen3-4B-Instruct"
data="/home/fdong/ymluo/datasets/sklearn"
template="${root}/results/20260729_qksieve_frozen_template_frontier/templates/global32_3domain_runtime.pt"
run_root="${root}/results/20260730_qksieve_per_head_coldskip_six_topic_ppl_32k_v2"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${root}/src"
export TORCH_CUDA_ARCH_LIST=8.6
export TOKENIZERS_PARALLELISM=false

mkdir -p "${run_root}/logs"
cd "${root}"

topics=(sports medicine space computer politics religion)
modes=(
  "baseline:pca_hierarchical_autoqmsetotal15z_qkmetric_packed_fulltopk"
  "skip50:pca_hierarchical_autoqmsetotal15z_qkmetric_freqskip50shard4recent256carry_packed_fulltopk"
  "skip60:pca_hierarchical_autoqmsetotal15z_qkmetric_freqskip60shard4recent256carry_packed_fulltopk"
)

run_topic() {
  local gpu="$1"
  local topic="$2"
  for specification in "${modes[@]}"; do
    local name="${specification%%:*}"
    local mode="${specification#*:}"
    local output="${run_root}/${topic}_${name}"
    mkdir -p "${output}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
      "${python_bin}" -u src/run_direct_countcap_denseprompt_ppl_20260725.py \
      --model_name_or_path "${model}" \
      --output_dir "${output}" \
      --topics "${topic}" \
      --window_indices 0 \
      --methods full_attention,direct_countcap \
      --history_tokens 32000 \
      --eval_tokens 64 \
      --window_stride_tokens 32512 \
      --direct_fraction 0.06 \
      --direct_min_tokens 256 \
      --direct_max_tokens 1280 \
      --projection_dim 128 \
      --sample_count 256 \
      --candidate_overfetch 1.0 \
      --protect_recent_tokens 0 \
      --direct_score_mode "${mode}" \
      --qk_metric_query_shrinkage 0.75 \
      --prefill_chunk_tokens 2048 \
      --cache_mode preallocated \
      --dataset_cache_dir "${data}" \
      --dtype float16 \
      --device cuda \
      --device_map balanced \
      --packed_qmse_template_in "${template}" \
      --collect_logit_stability \
      >"${run_root}/logs/${topic}_${name}.log" 2>&1
  done
  touch "${run_root}/${topic}.ALL_COMPLETE"
}

pids=()
for gpu in 0 1 2 3 4 5; do
  run_topic "${gpu}" "${topics[${gpu}]}" &
  pids+=("$!")
done
wait "${pids[@]}"
touch "${run_root}/ALL_COMPLETE"
