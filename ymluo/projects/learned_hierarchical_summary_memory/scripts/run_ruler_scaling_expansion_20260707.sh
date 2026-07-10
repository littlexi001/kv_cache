#!/usr/bin/env bash
set -u

ROOT=/home/fdong/ymluo/projects/learned_hierarchical_summary_memory
PY=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
VERIFIER=$ROOT/outputs/output_level_verifier_multiseed_qwen8b_m4_plus_longbench12_20260707/verifier_seed_2026070811/output_level_risk_verifier.pt
LONGDIR=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
RULERDIR=/home/fdong/ymluo/external/KVCache-Factory/data/RULER
LOGDIR=$ROOT/outputs/ruler_scaling_expansion_20260707_logs

mkdir -p "$LOGDIR"
cd "$ROOT" || exit 1

run_ruler() {
  local gpu="$1"
  local ctx="$2"
  local m="$3"
  local min_budget="$4"
  local name="ruler${ctx}_m${m}_floor${min_budget}"
  local out="$ROOT/outputs/output_verifier_runtime_qwen8b_ruler_${ctx}_m${m}_tau07_prefix_floor${min_budget}_20260707"
  local log="$LOGDIR/${name}.log"

  if [ -f "$out/summary.csv" ]; then
    echo "$(date -Is) skip $name: summary.csv exists" | tee -a "$log"
    return 0
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
    --runtime_methods full_kv_cache,output_level_risk_kv_planner \
    --output_verifier_path "$VERIFIER" \
    --output_verifier_threshold 0.7 \
    --output_verifier_budgets 1,2,3,4,6,8 \
    --output_verifier_mode prefix \
    --output_verifier_min_budget "$min_budget" \
    --output_verifier_long_ruler_min_budget "$min_budget" \
    --output_verifier_long_ruler_context_threshold 8192 \
    --seed 2026070701 \
    >>"$log" 2>&1
  local rc=$?
  echo "$(date -Is) done $name rc=$rc" | tee -a "$log"
  return "$rc"
}

run_ruler 4 4096 3 2 &
run_ruler 5 8192 3 2 &
run_ruler 6 16384 2 2 &
wait
