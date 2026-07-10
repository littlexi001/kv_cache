#!/usr/bin/env bash
set -u

ROOT=/home/fdong/ymluo/projects/learned_hierarchical_summary_memory
PY=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
LONGDIR=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
RULERDIR=/home/fdong/ymluo/external/KVCache-Factory/data/RULER
CONFORMAL=$ROOT/outputs/conformal_tailrisk_multiseed_qwen8b_m4_plus_longbench12_alpha005_addone_20260707/conformal_seed_2026070811/conformal_tailrisk_planner.pt
LOGDIR=$ROOT/outputs/variable_budget_ruler_m5_conformal_floor2_20260707_logs

mkdir -p "$LOGDIR"
cd "$ROOT" || exit 1

run_ruler() {
  local gpu="$1"
  local ctx="$2"
  local name="ruler${ctx}_m5_conformal_floor2"
  local out="$ROOT/outputs/variable_budget_runtime_qwen8b_ruler${ctx}_m5_conformal_floor2_20260707"
  local log="$LOGDIR/${name}.log"

  if [ -f "$out/summary.csv" ]; then
    echo "$(date -Is) skip $name: summary.csv exists" | tee -a "$log"
    return 0
  fi
  if [ ! -f "$CONFORMAL" ]; then
    echo "$(date -Is) missing conformal planner: $CONFORMAL" | tee -a "$log"
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
    --max_examples_per_task 5 \
    --max_context_tokens "$ctx" \
    --page_tokens 512 \
    --top_k 2 \
    --max_new_tokens_exact 48 \
    --max_new_tokens_summary 120 \
    --dtype float16 \
    --attn_implementation sdpa \
    --runtime_methods full_kv_cache,variable_budget_kv_planner \
    --variable_budget_planner_path "$CONFORMAL" \
    --variable_budget_policy tail_risk \
    --variable_budget_tail_threshold -1 \
    --variable_budget_min_budget 2 \
    --variable_budget_temperature 1.0 \
    --seed 2026070701 \
    >>"$log" 2>&1
  local rc=$?
  echo "$(date -Is) done $name rc=$rc" | tee -a "$log"
  return "$rc"
}

run_ruler 2 4096 &
run_ruler 6 8192 &
wait
