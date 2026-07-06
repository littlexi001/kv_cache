#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate moe
cd /home/fdong

BASE="ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_recent_plus_bench_m4_parallel_20260706"
MODEL="/home/fdong/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
ADAPTER="ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_lora_4k_1ksteps_no_bench_20260705/adapter"
SCRIPT="ymluo/projects/learned_hierarchical_summary_memory/src/run_qwen8b_paper_benchmarks.py"
METHODS="full_raw,recent_plus_summary1_8,recent_plus_summary1_4,recent_plus_summary1_2,recent_plus_static_hier,recent_plus_retrieval_raw_k1,recent_plus_retrieval_raw_k2,recent_plus_retrieval_raw_k3,recent_plus_retrieval_raw_k4,recent_plus_retrieval_raw_k8"
RULER_TASKS="niah_single_1,niah_single_2,niah_multikey_1,niah_multiquery,niah_multivalue,vt,cwe,fwe"

mkdir -p "$BASE/logs" "$BASE/pids"

run_job() {
  local gpu="$1"
  local name="$2"
  local longbench_tasks="$3"
  local ruler_tasks="$4"
  local ruler_lengths="$5"
  local out="$BASE/$name"
  local log="$BASE/logs/$name.log"
  rm -rf "$out"
  (
    CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT" \
      --output_dir "$out" \
      --model_name_or_path "$MODEL" \
      --adapter_path "$ADAPTER" \
      --longbench_data_dir ymluo/external/KVCache-Factory/data/LongBench \
      --ruler_data_dir ymluo/external/KVCache-Factory/data/RULER \
      --longbench_tasks "$longbench_tasks" \
      --ruler_tasks "$ruler_tasks" \
      --ruler_context_lengths "$ruler_lengths" \
      --methods "$METHODS" \
      --max_examples_per_task 4 \
      --max_new_tokens_exact 48 \
      --max_new_tokens_summary 120 \
      --dtype float16 \
      --attn_implementation sdpa \
      --device_map auto \
      --cuda_visible_devices "$gpu"
  ) >"$log" 2>&1 &
  echo $! > "$BASE/pids/$name.pid"
  echo "started $name on gpu $gpu pid $(cat "$BASE/pids/$name.pid")"
}

run_job 4 "longbench_exact" "hotpotqa,2wikimqa,musique,passage_retrieval_en,passage_count,qasper" "" ""
run_job 5 "longbench_summary" "gov_report,multi_news" "" ""
run_job 6 "ruler_4k8k" "" "$RULER_TASKS" "4096,8192"
run_job 7 "ruler_16k" "" "$RULER_TASKS" "16384"

echo "$BASE"
