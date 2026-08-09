#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/20260730_qksieve_allocation_structure_ppl_3gpu}"
GPUS="${QKSIEVE_GPUS:-0,1,2}"
QMSE_TEMPLATE="${QMSE_TEMPLATE:-$ROOT/results/20260729_qksieve_frozen_template_frontier/templates/global32_3domain_runtime.pt}"
KEYMSE_TEMPLATE="${KEYMSE_TEMPLATE:-$ROOT/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
FIXED_TEMPLATE="${FIXED_TEMPLATE:-$ROOT/results/20260730_qksieve_fixed411111_template/global32_3domain_fixed411111_runtime.pt}"

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

IFS=',' read -r -a gpu_ids <<< "$GPUS"
if [[ "${#gpu_ids[@]}" -ne 3 ]]; then
  echo "QKSIEVE_GPUS must contain exactly three GPU ids" >&2
  exit 2
fi
for gpu in "${gpu_ids[@]}"; do
  if [[ ! "$gpu" =~ ^[0-6]$ ]]; then
    echo "GPU ids are restricted to 0-6; got $gpu" >&2
    exit 2
  fi
done

mkdir -p "$RUN_ROOT/logs"
cd "$ROOT"
test -f "$QMSE_TEMPLATE"
test -f "$KEYMSE_TEMPLATE"
test -f "$FIXED_TEMPLATE"

tags=(qmse keymse fixed411111)
templates=("$QMSE_TEMPLATE" "$KEYMSE_TEMPLATE" "$FIXED_TEMPLATE")
score_modes=(
  pca_hierarchical_autoqmsetotal15z_qkmetric_qfused_gqa4_wmma_kappend_unbiased_packed_direct
  pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_kappend_unbiased_packed_direct
  pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_kappend_unbiased_packed_direct
)

pids=()
for index in 0 1 2; do
  tag="${tags[$index]}"
  output_dir="$RUN_ROOT/$tag"
  if [[ -f "$output_dir/case_summary.json" ]]; then
    echo "[skip] $tag"
    continue
  fi
  CUDA_VISIBLE_DEVICES="${gpu_ids[$index]}" "$PYTHON" -u \
    src/run_direct_countcap_denseprompt_ppl_20260725.py \
    --model_name_or_path "$MODEL" \
    --output_dir "$output_dir" \
    --topics sports,medicine,computer,space,politics,religion \
    --window_indices 0,1 \
    --methods full_attention,direct_countcap \
    --history_tokens 32000 \
    --eval_tokens 128 \
    --window_stride_tokens 32512 \
    --direct_fraction 0.06 \
    --direct_min_tokens 256 \
    --direct_max_tokens 1280 \
    --projection_dim 128 \
    --sample_count 256 \
    --candidate_overfetch 1.0 \
    --protect_recent_tokens 0 \
    --direct_score_mode "${score_modes[$index]}" \
    --qk_metric_query_shrinkage 0.75 \
    --packed_qmse_template_in "${templates[$index]}" \
    --prefill_chunk_tokens 2048 \
    --cache_mode preallocated \
    --dtype float16 \
    --device cuda \
    --device_map balanced \
    --collect_logit_stability \
    >"$RUN_ROOT/logs/$tag.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "one or more allocation workers failed; outputs were preserved" >&2
  exit 1
fi

"$PYTHON" src/summarize_qksieve_allocation_structure_ppl_20260730.py \
  --run_root "$RUN_ROOT" \
  --expected_pairs 12 \
  --bootstrap_iterations 20000 \
  --output "$RUN_ROOT/summary.json" \
  >"$RUN_ROOT/logs/summary.log" 2>&1

touch "$RUN_ROOT/ALL_COMPLETE"
