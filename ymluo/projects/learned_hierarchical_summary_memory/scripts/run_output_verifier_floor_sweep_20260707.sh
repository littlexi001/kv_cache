#!/usr/bin/env bash
set -u

ROOT=/home/fdong/ymluo/projects/learned_hierarchical_summary_memory
PY=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
VERIFIER=$ROOT/outputs/output_level_verifier_multiseed_qwen8b_m4_plus_longbench12_20260707/verifier_seed_2026070811/output_level_risk_verifier.pt
LONGDIR=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
RULERDIR=/home/fdong/ymluo/external/KVCache-Factory/data/RULER
LOGDIR=$ROOT/outputs/output_verifier_floor_sweep_20260707_logs

mkdir -p "$LOGDIR"
cd "$ROOT" || exit 1

run_case() {
  local gpu="$1"
  local name="$2"
  local out="$3"
  shift 3
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
    --seed 2026070701 \
    "$@" \
    >>"$log" 2>&1
  local rc=$?
  echo "$(date -Is) done $name rc=$rc" | tee -a "$log"
  return "$rc"
}

run_case 0 longbench_m4_floor2 \
  "$ROOT/outputs/output_verifier_runtime_qwen8b_longbench_m4_tau07_prefix_floor2_20260707" \
  --ruler_tasks "" \
  --max_examples_per_task 4 \
  --max_context_tokens 4096 \
  --output_verifier_min_budget 2 &

run_case 1 longbench_m4_floor3 \
  "$ROOT/outputs/output_verifier_runtime_qwen8b_longbench_m4_tau07_prefix_floor3_20260707" \
  --ruler_tasks "" \
  --max_examples_per_task 4 \
  --max_context_tokens 4096 \
  --output_verifier_min_budget 3 &

run_case 2 mixed13_floor2 \
  "$ROOT/outputs/output_verifier_runtime_qwen8b_13tasks_m1_tau07_prefix_floor2_20260707" \
  --max_examples_per_task 1 \
  --max_context_tokens 4096 \
  --output_verifier_min_budget 2 &

run_case 3 ruler4k_floor2 \
  "$ROOT/outputs/output_verifier_runtime_qwen8b_ruler_4k_m1_tau07_prefix_floor2_20260707" \
  --longbench_tasks "" \
  --ruler_context_lengths 4096 \
  --max_examples_per_task 1 \
  --max_context_tokens 4096 \
  --output_verifier_min_budget 2 &

wait
