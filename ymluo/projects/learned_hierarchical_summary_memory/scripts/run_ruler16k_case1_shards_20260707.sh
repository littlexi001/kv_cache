#!/usr/bin/env bash
set -u

ROOT=/home/fdong/ymluo/projects/learned_hierarchical_summary_memory
PY=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
VERIFIER=$ROOT/outputs/output_level_verifier_multiseed_qwen8b_m4_plus_longbench12_20260707/verifier_seed_2026070811/output_level_risk_verifier.pt
LONGDIR=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
RULERDIR=/home/fdong/ymluo/external/KVCache-Factory/data/RULER
LOGDIR=$ROOT/outputs/ruler16k_case1_shards_20260707_logs

mkdir -p "$LOGDIR"
cd "$ROOT" || exit 1

run_case1() {
  local gpu="$1"
  local task="$2"
  local out="$ROOT/outputs/output_verifier_runtime_qwen8b_ruler_16k_${task}_case1_tau07_prefix_floor2_20260707"
  local log="$LOGDIR/${task}_case1.log"

  if [ -f "$out/summary.csv" ]; then
    echo "$(date -Is) skip $task case1: summary.csv exists" | tee -a "$log"
    return 0
  fi

  echo "$(date -Is) launch $task case1 on gpu $gpu" | tee -a "$log"
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
    --max_examples_per_task 2 \
    --case_start 1 \
    --case_limit 1 \
    --max_context_tokens 16384 \
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
    --output_verifier_min_budget 2 \
    --output_verifier_long_ruler_min_budget 2 \
    --output_verifier_long_ruler_context_threshold 8192 \
    --seed 2026070701 \
    >>"$log" 2>&1
  local rc=$?
  echo "$(date -Is) done $task case1 rc=$rc" | tee -a "$log"
  return "$rc"
}

run_case1 2 niah_single_1 &
run_case1 3 niah_single_2 &
run_case1 6 niah_multikey_1 &
run_case1 7 niah_multiquery &
wait

run_case1 2 niah_multivalue &
run_case1 3 vt &
run_case1 6 cwe &
run_case1 7 fwe &
wait
