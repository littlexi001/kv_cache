#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
OUT=$ROOT/results/20260716_128k_numa_ablation
LOG_ROOT=$ROOT/outputs/logs/20260716_128k_numa_ablation

export PATH=/home/fdong/miniconda3/envs/moe/bin:$PATH
mkdir -p "$OUT" "$LOG_ROOT"
cd "$ROOT"
nvidia-smi topo -m > "$OUT/nvidia_topology.txt"
lscpu > "$OUT/lscpu.txt"
grep -E 'Cpus_allowed_list|Mems_allowed_list' /proc/self/status \
  > "$OUT/cgroup_affinity.txt"
for path in /sys/devices/system/node/node*/cpulist; do
  printf '%s %s\n' "$path" "$(cat "$path")"
done > "$OUT/numa_nodes.txt"

run_binding() {
  local name=$1
  local devices=$2
  if [[ -s "$OUT/${name}.json" ]]; then return; fi
  CUDA_VISIBLE_DEVICES="$devices" "$PYTHON" src/numa_exec_20260716.py \
    --node 0 -- "$PYTHON" \
    src/run_hierarchical_physical_cache_ppl_20260715.py \
    --model_name_or_path "$MODEL" \
    --output "$OUT/${name}.json" \
    --topic religion \
    --history_tokens 128000 \
    --query_tokens 256 \
    --eval_tokens 256 \
    --window_index 1 \
    --window_stride_tokens 128512 \
    --projection_dim 64 \
    --index_bits 4 \
    --candidate_fraction 0.015 \
    --attention_fraction 0.015 \
    --candidate_selection_mode per_head_stream \
    --stream_group_size 2 \
    --candidate_refresh_interval 1 \
    --exact_cache_fraction 0.032 \
    --directory_backend fused \
    --prefill_cache_mode dynamic \
    --prefill_chunk_tokens 2048 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    > "$LOG_ROOT/${name}.log" 2>&1
}

run_microbenchmark() {
  local name=$1
  local gpu=$2
  if [[ -s "$OUT/${name}.json" ]]; then return; fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" src/numa_exec_20260716.py \
    --node 0 -- "$PYTHON" \
    src/benchmark_mapped_host_gather_20260715.py \
    --history_count 131072 \
    --selected_fraction 0.015 \
    --attention_fraction 0.015 \
    --cache_fraction 0.032 \
    --cache_hit_rate 0.79 \
    --warmup 10 \
    --repeats 100 \
    --output "$OUT/${name}.json" \
    > "$LOG_ROOT/${name}.log" 2>&1
}

# This server's cgroup permits memory only on NUMA node 0. Keep CPU and memory
# fixed on node 0, then move only the GPU across the NUMA/PCIe boundary.
run_binding local_gpu0_mem0 0,1,2,3
run_binding remote_gpu4_mem0 4,5,6,7
run_microbenchmark micro_local_gpu0_mem0 0
run_microbenchmark micro_remote_gpu4_mem0 4

"$PYTHON" src/summarize_128k_numa_ablation_20260716.py \
  --input_dir "$OUT" \
  > "$LOG_ROOT/summary.log" 2>&1

touch "$OUT/COMPLETE"
