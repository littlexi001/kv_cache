#!/usr/bin/env bash
set -uo pipefail

PROJECT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
EXTERNAL=/home/fdong/ymluo/external/KVCache-Factory
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
PYDEPS=/home/fdong/ymluo/pydeps/kvcache_factory_tf444
RUNNER="$PROJECT/src/run_kvcache_factory_longbench_aligned.py"
SUMMARIZER="$PROJECT/src/summarize_kvcache_factory_aligned_comparison.py"

OUT="$PROJECT/outputs/kvcache_factory_aligned_b1024_20260713_m100_v1"
MODEL_DIR="$OUT/llm-research-meta-llama-3.1-8b-instruct-ms_1024"
LOG_ROOT="$PROJECT/outputs/logs/kvcache_factory_aligned_b1024_20260713_m100_v1"
ANALYSIS="$OUT/analysis"
RAG_CSV="$PROJECT/outputs/longbench_text_rag_hybrid_recent_m100_v1/task_results.csv"
FULL_CSV="$PROJECT/outputs/riskkv_fullkv_m100_same_samples_20260710/task_results.csv"
EXPECTED_ROWS=4800
TASKS=(
  gov_report multi_news qmsum samsum
  narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique
  trec triviaqa passage_count passage_retrieval_en lcc repobench-p
)

mkdir -p "$LOG_ROOT" "$ANALYSIS"

count_rows() {
  find "$OUT" -mindepth 3 -name '*.json' -exec cat {} + 2>/dev/null | wc -l
}

run_missing_for_method() {
  local gpu="$1"
  local method="$2"
  local task file rows log

  for task in "${TASKS[@]}"; do
    file="$MODEL_DIR/$task/$method.json"
    rows=0
    if [[ -f "$file" ]]; then
      rows=$(wc -l < "$file")
    fi
    if [[ "$rows" -eq 100 ]]; then
      continue
    fi

    log="$LOG_ROOT/retry_${method}_${task}.log"
    echo "$(date -Is) retry method=$method task=$task old_rows=$rows gpu=$gpu"
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
        --datasets "$task" \
        --max_num_examples 100 \
        --sample_method topk \
        >"$log" 2>&1
  done
}

echo "$(date -Is) supervisor started rows=$(count_rows)"

# The original workers are tied to the interactive SSH command. Wait for them;
# after a disconnect, repair only incomplete method/task files.
while pgrep -f "$RUNNER.*--save_dir $OUT" >/dev/null; do
  sleep 30
done

rows=$(count_rows)
echo "$(date -Is) original workers stopped rows=$rows"
if [[ "$rows" -lt "$EXPECTED_ROWS" ]]; then
  run_missing_for_method 0 H2O &
  h2o_pid=$!
  run_missing_for_method 2 SnapKV &
  snapkv_pid=$!
  run_missing_for_method 5 AdaKV &
  adakv_pid=$!
  wait "$h2o_pid" "$snapkv_pid" "$adakv_pid"
fi

rows=$(count_rows)
echo "$(date -Is) generation finished rows=$rows"
if [[ "$rows" -ne "$EXPECTED_ROWS" ]]; then
  echo "expected $EXPECTED_ROWS rows, found $rows" >&2
  exit 1
fi

"$PYTHON" "$SUMMARIZER" \
  --input_dir "$OUT" \
  --rag_csv "$RAG_CSV" \
  --full_csv "$FULL_CSV" \
  --output_dir "$ANALYSIS"

echo "$(date -Is) analysis finished output=$ANALYSIS"
