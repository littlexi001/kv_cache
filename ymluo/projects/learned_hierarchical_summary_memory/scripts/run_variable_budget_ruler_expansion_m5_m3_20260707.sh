#!/usr/bin/env bash
set -u

ROOT=/home/fdong/ymluo/projects/learned_hierarchical_summary_memory
PY=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
LONGDIR=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
RULERDIR=/home/fdong/ymluo/external/KVCache-Factory/data/RULER
BESTCAL=$ROOT/outputs/variable_budget_planner_qwen8b_m4_plus_longbench12_k1k2k3k4k6k8_best_calibrated_ls005_cp001_20260707/variable_budget_planner.pt
CONFORMAL=$ROOT/outputs/conformal_tailrisk_multiseed_qwen8b_m4_plus_longbench12_alpha005_addone_20260707/conformal_seed_2026070811/conformal_tailrisk_planner.pt
LOGDIR=$ROOT/outputs/variable_budget_ruler_expansion_m5_m3_20260707_logs
TASKS=(niah_single_1 niah_single_2 niah_multikey_1 niah_multiquery niah_multivalue vt cwe fwe)
GPUS=(0 5 6 7)

mkdir -p "$LOGDIR"
cd "$ROOT" || exit 1

run_ruler() {
  local gpu="$1"
  local ctx="$2"
  local m="$3"
  local planner="$4"
  local threshold="$5"
  local tag="$6"
  local name="ruler${ctx}_m${m}_${tag}"
  local out="$ROOT/outputs/variable_budget_runtime_qwen8b_ruler${ctx}_m${m}_${tag}_20260707"
  local log="$LOGDIR/${name}.log"

  if [ -f "$out/summary.csv" ]; then
    echo "$(date -Is) skip $name: summary.csv exists" | tee -a "$log"
    return 0
  fi
  if [ ! -f "$planner" ]; then
    echo "$(date -Is) missing planner for $name: $planner" | tee -a "$log"
    return 2
  fi

  echo "$(date -Is) launch $name on gpu $gpu" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  TOKENIZERS_PARALLELISM=false \
  "$PY" src/run_rope_aware_kv_repack_benchmark.py \
    --output_dir "$out" \
    --model_name_or_path "$MODEL" \
    --longbench_data_dir "$LONGDIR" \
    --ruler_data_dir "$RULERDIR" \
    --longbench_tasks "" \
    --ruler_context_lengths "$ctx" \
    --max_examples_per_task "$m" \
    --max_context_tokens "$ctx" \
    --page_tokens 512 \
    --top_k 2 \
    --max_new_tokens_exact 48 \
    --max_new_tokens_summary 120 \
    --dtype float16 \
    --attn_implementation sdpa \
    --runtime_methods full_kv_cache,variable_budget_kv_planner \
    --variable_budget_planner_path "$planner" \
    --variable_budget_policy tail_risk \
    --variable_budget_tail_threshold "$threshold" \
    --variable_budget_temperature 1.0 \
    --seed 2026070701 \
    >>"$log" 2>&1
  local rc=$?
  echo "$(date -Is) done $name rc=$rc" | tee -a "$log"
  return "$rc"
}

run_16k_case() {
  local gpu="$1"
  local task="$2"
  local case_id="$3"
  local planner="$4"
  local threshold="$5"
  local tag="$6"
  local name="ruler16k_${task}_case${case_id}_${tag}"
  local out="$ROOT/outputs/variable_budget_runtime_qwen8b_ruler_16k_${task}_case${case_id}_${tag}_20260707"
  local log="$LOGDIR/${name}.log"

  if [ -f "$out/summary.csv" ]; then
    echo "$(date -Is) skip $name: summary.csv exists" | tee -a "$log"
    return 0
  fi
  if [ ! -f "$planner" ]; then
    echo "$(date -Is) missing planner for $name: $planner" | tee -a "$log"
    return 2
  fi

  echo "$(date -Is) launch $name on gpu $gpu" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  TOKENIZERS_PARALLELISM=false \
  "$PY" src/run_rope_aware_kv_repack_benchmark.py \
    --output_dir "$out" \
    --model_name_or_path "$MODEL" \
    --longbench_data_dir "$LONGDIR" \
    --ruler_data_dir "$RULERDIR" \
    --longbench_tasks "" \
    --ruler_tasks "$task" \
    --ruler_context_lengths 16384 \
    --max_examples_per_task 3 \
    --case_start "$case_id" \
    --case_limit 1 \
    --max_context_tokens 16384 \
    --page_tokens 512 \
    --top_k 2 \
    --max_new_tokens_exact 48 \
    --max_new_tokens_summary 120 \
    --dtype float16 \
    --attn_implementation sdpa \
    --runtime_methods full_kv_cache,variable_budget_kv_planner \
    --variable_budget_planner_path "$planner" \
    --variable_budget_policy tail_risk \
    --variable_budget_tail_threshold "$threshold" \
    --variable_budget_temperature 1.0 \
    --seed 2026070701 \
    >>"$log" 2>&1
  local rc=$?
  echo "$(date -Is) done $name rc=$rc" | tee -a "$log"
  return "$rc"
}

run_16k_case2_for_tag() {
  local tag="$1"
  local planner="$2"
  local threshold="$3"
  local jobs=0
  local gpu_index=0
  local gpu_count="${#GPUS[@]}"

  for task in "${TASKS[@]}"; do
    local gpu="${GPUS[$gpu_index]}"
    run_16k_case "$gpu" "$task" 2 "$planner" "$threshold" "$tag" &
    jobs=$((jobs + 1))
    gpu_index=$(((gpu_index + 1) % gpu_count))
    if [ $((jobs % gpu_count)) -eq 0 ]; then
      wait
    fi
  done
  wait
}

run_ruler 0 4096 5 "$BESTCAL" 0.35 bestcal_tail035 &
run_ruler 5 8192 5 "$BESTCAL" 0.35 bestcal_tail035 &
run_ruler 6 4096 5 "$CONFORMAL" -1 conformal_auto &
run_ruler 7 8192 5 "$CONFORMAL" -1 conformal_auto &
wait

run_16k_case2_for_tag bestcal_tail035 "$BESTCAL" 0.35
run_16k_case2_for_tag conformal_auto "$CONFORMAL" -1
