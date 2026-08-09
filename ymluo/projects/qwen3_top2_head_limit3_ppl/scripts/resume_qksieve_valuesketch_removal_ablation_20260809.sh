#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/fdong/qksieve_iclr2027
PY=/home/fdong/miniconda3/envs/nanogpt/bin/python
RUN_ROOT="$ROOT/results/20260809_valuesketch_removal_ablation_32k96k_oas_v1"
RUNNER="$ROOT/src/run_qksieve_coldskip_longcontext_quality_20260730.py"
MODEL="$ROOT/models/Meta-Llama-3.1-8B-Instruct-ms"
DATA="$ROOT/data/ablation"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/home/fdong/miniconda3/envs/nanogpt/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
export QKSIEVE_PRELOAD_EXTENSIONS=1
export QKSIEVE_PRELOAD_QMSE_RATE_TABLES=1
export QKSIEVE_MAX_QUANTILE_SAMPLE_COUNT=512

mkdir -p "$RUN_ROOT/logs"

run_one() {
  local devices=$1 length=$2 eval_tokens=$3 name=$4 text_file=$5 variants=$6
  local output="$RUN_ROOT/${name}"
  CUDA_VISIBLE_DEVICES="$devices" "$PY" -u "$RUNNER" \
    --model_name_or_path "$MODEL" \
    --template "$ROOT/data/unused_requestlocal_template.pt" \
    --output_dir "$output" \
    --history_tokens "$length" \
    --eval_tokens "$eval_tokens" \
    --text_file "$text_file" \
    --repeat_topic_stream_if_short \
    --prefill_chunk_tokens 1024 \
    --protect_recent_tokens 0 \
    --dataset_cache_dir "$ROOT/data/sklearn" \
    --seed 20260809 \
    --dtype float16 \
    --device cuda \
    --device_map balanced \
    --max_memory_per_gpu_gib 22 \
    --variants "$variants" \
    >"$RUN_ROOT/logs/${name}.log" 2>&1
}

vs=qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64_k1280
pair=qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64_novalue_k1280,qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64_k1280

# First validate the fixed ValueSketch path while one two-GPU 96K stream runs.
run_one 0 32768 64 h32768_vsfix_war_and_peace "$DATA/war_and_peace_pg2600.txt" "$vs" & p0=$!
run_one 2 32768 64 h32768_vsfix_monte_cristo "$DATA/count_monte_cristo_pg1184.txt" "$vs" & p1=$!
run_one 4 32768 64 h32768_vsfix_qksieve_code "$ROOT/src/run_head_top2_targeted_ppl_20260714.py" "$vs" & p2=$!
run_one 1,3 98304 32 h98304_2gpu_war_and_peace "$DATA/war_and_peace_pg2600.txt" "$pair" & p3=$!
status=0
for pid in $p0 $p1 $p2 $p3; do wait "$pid" || status=1; done

# The remaining 96K streams use two GPUs each and preserve the first-wave results.
run_one 0,1 98304 32 h98304_2gpu_monte_cristo "$DATA/count_monte_cristo_pg1184.txt" "$pair" & p4=$!
run_one 2,3 98304 32 h98304_2gpu_qksieve_code "$ROOT/src/run_head_top2_targeted_ppl_20260714.py" "$pair" & p5=$!
for pid in $p4 $p5; do wait "$pid" || status=1; done

if [[ $status -eq 0 ]]; then
  touch "$RUN_ROOT/RESUME_ALL_COMPLETE"
else
  touch "$RUN_ROOT/RESUME_FAILED"
fi
exit "$status"
