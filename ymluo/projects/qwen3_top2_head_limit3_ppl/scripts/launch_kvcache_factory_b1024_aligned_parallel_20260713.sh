#!/usr/bin/env bash
set -uo pipefail

PROJECT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
EXTERNAL=/home/fdong/ymluo/external/KVCache-Factory
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
PYDEPS=/home/fdong/ymluo/pydeps/kvcache_factory_tf444
RUNNER="$PROJECT/src/run_kvcache_factory_longbench_aligned.py"

SAMPLES="${SAMPLES:-20}"
STAMP="${STAMP:-20260713_m${SAMPLES}_v1}"
OUT="$PROJECT/outputs/kvcache_factory_aligned_b1024_${STAMP}"
LOG_ROOT="$PROJECT/outputs/logs/kvcache_factory_aligned_b1024_${STAMP}"
mkdir -p "$OUT" "$LOG_ROOT"

QA_GROUP="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique"
OTHER_GROUP="trec,triviaqa,passage_count,passage_retrieval_en,lcc,repobench-p"
OTHER_NO_TREC="triviaqa,passage_count,passage_retrieval_en,lcc,repobench-p"

pids=()
labels=()

launch_job() {
  local gpu="$1"
  local method="$2"
  local tasks="$3"
  local label="$4"
  local delay="$5"
  local log="$LOG_ROOT/${label}.log"

  (
    sleep "$delay"
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="$PYDEPS:$EXTERNAL" \
      "$PYTHON" "$RUNNER" \
        --method "$method" \
        --model_path "$MODEL" \
        --max_capacity_prompts 1024 \
        --attn_implementation sdpa \
        --dtype float16 \
        --kv_cache_granularity kv_head \
        --longbench_data_dir "$EXTERNAL/data/LongBench" \
        --save_dir "$OUT" \
        --datasets "$tasks" \
        --max_num_examples "$SAMPLES" \
        --sample_method topk \
        >"$log" 2>&1
  ) &
  pids+=("$!")
  labels+=("$label")
  echo "launched label=$label gpu=$gpu pid=$! delay=$delay log=$log"
}

# M20 timings balance the seven shards to roughly equal total generation time.
launch_job 0 H2O "gov_report,multi_news,qmsum,samsum,trec" h2o_global 0
launch_job 1 H2O "$QA_GROUP,$OTHER_NO_TREC" h2o_other 10
launch_job 2 SnapKV "gov_report,multi_news,samsum" snapkv_global 20
launch_job 4 SnapKV "qmsum,$QA_GROUP,$OTHER_GROUP" snapkv_other 30
launch_job 5 AdaKV "multi_news,samsum" adakv_multi 40
launch_job 6 AdaKV "gov_report,qmsum" adakv_gov 50
launch_job 7 AdaKV "$QA_GROUP,$OTHER_GROUP" adakv_other 60

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "completed label=${labels[$index]}"
  else
    status=$?
    echo "failed label=${labels[$index]} status=$status" >&2
    failed=1
  fi
done

echo "output=$OUT"
exit "$failed"
