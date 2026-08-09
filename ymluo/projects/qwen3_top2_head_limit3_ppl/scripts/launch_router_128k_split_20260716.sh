#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
OUT=$ROOT/results/20260716_router_128k_split
LOG_ROOT=$ROOT/outputs/logs/20260716_router_128k_split

export PATH=/home/fdong/miniconda3/envs/moe/bin:$PATH
mkdir -p "$OUT" "$LOG_ROOT"
cd "$ROOT"

collect_case() {
  local topic=$1
  local window=$2
  local devices=$3
  local cpus
  if [[ "$devices" == 0,* ]]; then cpus=0-23,48-71; else cpus=24-47,72-95; fi
  taskset -pc "$cpus" "$BASHPID" >/dev/null
  local output="$OUT/counterfactual_${topic}_w${window}.json"
  if [[ -s "$output" ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES="$devices" "$PYTHON" \
    src/collect_onpolicy_counterfactual_router_20260715.py \
    --model_name_or_path "$MODEL" \
    --output "$output" \
    --full_reference \
      "$ROOT/results/20260716_128k_multitopic_windows_w3/${topic}_w${window}_full.json" \
    --topic "$topic" \
    --history_tokens 128000 \
    --query_tokens 256 \
    --eval_tokens 256 \
    --window_index "$window" \
    --window_stride_tokens 128512 \
    --projection_dim 64 \
    --index_bits 4 \
    --low_fraction 0.01 \
    --mid_fraction 0.015 \
    --high_fraction 0.02 \
    --low_stream_group_size 2 \
    --mid_stream_group_size 2 \
    --high_stream_group_size 1 \
    --exact_cache_fraction 0.032 \
    --safe_nll_tolerance 0.05 \
    --behavior_action high \
    --prefill_chunk_tokens 2048 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    > "$LOG_ROOT/counterfactual_${topic}_w${window}.log" 2>&1
}

collect_case computer 0 0,1,2,3 &
pid0=$!
collect_case sports 0 4,5,6,7 &
pid1=$!
wait "$pid0" "$pid1"
collect_case medicine 0 0,1,2,3 &
pid0=$!
collect_case space 0 4,5,6,7 &
pid1=$!
wait "$pid0" "$pid1"
collect_case politics 0 0,1,2,3 &
pid0=$!
collect_case religion 0 4,5,6,7 &
pid1=$!
wait "$pid0" "$pid1"
collect_case medicine 1 0,1,2,3 &
pid0=$!
collect_case politics 1 4,5,6,7 &
pid1=$!
wait "$pid0" "$pid1"
collect_case computer 1 0,1,2,3

train_fold() {
  local fold=$1
  local calibration_topic=$2
  shift 2
  local args=()
  local topic
  for topic in "$@"; do
    args+=(--train "$OUT/counterfactual_${topic}_w0.json")
  done
  "$PYTHON" src/train_onpolicy_counterfactual_router_20260715.py \
    "${args[@]}" \
    --calibration "$OUT/counterfactual_${calibration_topic}_w1.json" \
    --output "$OUT/router_${fold}.pkl" \
    --report "$OUT/train_${fold}_report.json" \
    --target_kind teacher_action \
    --calibration_objective joint_safety \
    --minimum_required_action_recall 0.95 \
    --target_retention_to_high 0.95 \
    --require_full_reference_labels \
    --model_type extra_trees \
    --trees 600 \
    --min_samples_leaf 4 \
    > "$LOG_ROOT/train_${fold}.log" 2>&1
}

# Each fold keeps both test topics out of model fitting and threshold calibration.
train_fold fold0 medicine space politics religion
train_fold fold1 politics computer sports religion
train_fold fold2 computer sports medicine space

run_test() {
  local fold=$1
  local topic=$2
  local devices=$3
  local refresh=${4:-1}
  local cpus
  if [[ "$devices" == 0,* ]]; then cpus=0-23,48-71; else cpus=24-47,72-95; fi
  taskset -pc "$cpus" "$BASHPID" >/dev/null
  local suffix=""
  if [[ "$refresh" -ne 1 ]]; then suffix="_refresh${refresh}"; fi
  local output="$OUT/test_${fold}_${topic}_w2${suffix}.json"
  if [[ -s "$output" ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES="$devices" "$PYTHON" \
    src/run_shifted_dynamic_physical_cache_ppl_20260715.py \
    --model_name_or_path "$MODEL" \
    --router_path "$OUT/router_${fold}.pkl" \
    --output "$output" \
    --topic "$topic" \
    --history_tokens 128000 \
    --query_tokens 256 \
    --eval_tokens 256 \
    --window_index 2 \
    --window_stride_tokens 128512 \
    --projection_dim 64 \
    --index_bits 4 \
    --exact_cache_fraction 0.032 \
    --candidate_refresh_interval "$refresh" \
    --query_action mid \
    --prefill_chunk_tokens 2048 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    > "$LOG_ROOT/test_${fold}_${topic}_w2${suffix}.log" 2>&1
}

run_test fold0 computer 0,1,2,3 &
pid0=$!
run_test fold0 sports 4,5,6,7 &
pid1=$!
wait "$pid0" "$pid1"
run_test fold1 medicine 0,1,2,3 &
pid0=$!
run_test fold1 space 4,5,6,7 &
pid1=$!
wait "$pid0" "$pid1"
run_test fold2 politics 0,1,2,3 &
pid0=$!
run_test fold2 religion 4,5,6,7 &
pid1=$!
wait "$pid0" "$pid1"

# Matched deployment ablation: the frozen router and target tokens are unchanged.
run_test fold0 computer 0,1,2,3 2 &
pid0=$!
run_test fold0 sports 4,5,6,7 2 &
pid1=$!
wait "$pid0" "$pid1"
run_test fold1 medicine 0,1,2,3 2 &
pid0=$!
run_test fold1 space 4,5,6,7 2 &
pid1=$!
wait "$pid0" "$pid1"
run_test fold2 politics 0,1,2,3 2 &
pid0=$!
run_test fold2 religion 4,5,6,7 2 &
pid1=$!
wait "$pid0" "$pid1"

touch "$OUT/COMPLETE"
