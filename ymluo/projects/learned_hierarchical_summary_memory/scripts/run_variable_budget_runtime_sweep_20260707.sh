#!/usr/bin/env bash
set -u

ROOT=/home/fdong/ymluo/projects/learned_hierarchical_summary_memory
PY=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
LONGDIR=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
RULERDIR=/home/fdong/ymluo/external/KVCache-Factory/data/RULER
BESTCAL=$ROOT/outputs/variable_budget_planner_qwen8b_m4_plus_longbench12_k1k2k3k4k6k8_best_calibrated_ls005_cp001_20260707/variable_budget_planner.pt
MINSAFE=$ROOT/outputs/variable_budget_planner_qwen8b_m4_plus_longbench12_k1k2k3k4k6k8_min_safe_20260707/variable_budget_planner.pt
CONFORMAL=$ROOT/outputs/conformal_tailrisk_multiseed_qwen8b_m4_plus_longbench12_alpha005_addone_20260707/conformal_seed_2026070811/conformal_tailrisk_planner.pt
LOGDIR=$ROOT/outputs/variable_budget_runtime_sweep_20260707_logs

mkdir -p "$LOGDIR"
cd "$ROOT" || exit 1

run_case() {
  local gpu="$1"
  local name="$2"
  local planner="$3"
  local threshold="$4"
  local out="$5"
  shift 5
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
    "$@" \
    >>"$log" 2>&1
  local rc=$?
  echo "$(date -Is) done $name rc=$rc" | tee -a "$log"
  return "$rc"
}

run_case 0 longbench_m4_bestcal_tail035 "$BESTCAL" 0.35 \
  "$ROOT/outputs/variable_budget_runtime_qwen8b_longbench_m4_bestcal_tail035_20260707" \
  --ruler_tasks "" \
  --max_examples_per_task 4 \
  --max_context_tokens 4096 &

run_case 1 longbench_m4_minsafe_tail035 "$MINSAFE" 0.35 \
  "$ROOT/outputs/variable_budget_runtime_qwen8b_longbench_m4_minsafe_tail035_20260707" \
  --ruler_tasks "" \
  --max_examples_per_task 4 \
  --max_context_tokens 4096 &

run_case 2 mixed13_bestcal_tail035 "$BESTCAL" 0.35 \
  "$ROOT/outputs/variable_budget_runtime_qwen8b_mixed13_m1_bestcal_tail035_20260707" \
  --max_examples_per_task 1 \
  --max_context_tokens 4096 &

run_case 3 mixed13_minsafe_tail035 "$MINSAFE" 0.35 \
  "$ROOT/outputs/variable_budget_runtime_qwen8b_mixed13_m1_minsafe_tail035_20260707" \
  --max_examples_per_task 1 \
  --max_context_tokens 4096 &

run_case 7 ruler8k_m1_bestcal_tail035 "$BESTCAL" 0.35 \
  "$ROOT/outputs/variable_budget_runtime_qwen8b_ruler8k_m1_bestcal_tail035_20260707" \
  --longbench_tasks "" \
  --ruler_context_lengths 8192 \
  --max_examples_per_task 1 \
  --max_context_tokens 8192 &

wait

run_case 0 longbench_m4_conformal_auto "$CONFORMAL" -1 \
  "$ROOT/outputs/variable_budget_runtime_qwen8b_longbench_m4_conformal_auto_20260707" \
  --ruler_tasks "" \
  --max_examples_per_task 4 \
  --max_context_tokens 4096 &

run_case 1 mixed13_conformal_auto "$CONFORMAL" -1 \
  "$ROOT/outputs/variable_budget_runtime_qwen8b_mixed13_m1_conformal_auto_20260707" \
  --max_examples_per_task 1 \
  --max_context_tokens 4096 &

run_case 2 ruler8k_m1_conformal_auto "$CONFORMAL" -1 \
  "$ROOT/outputs/variable_budget_runtime_qwen8b_ruler8k_m1_conformal_auto_20260707" \
  --longbench_tasks "" \
  --ruler_context_lengths 8192 \
  --max_examples_per_task 1 \
  --max_context_tokens 8192 &

wait
