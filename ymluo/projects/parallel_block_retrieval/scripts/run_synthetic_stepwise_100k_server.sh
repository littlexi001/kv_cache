#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

HELDOUT_TEMPLATE_SET="${HELDOUT_TEMPLATE_SET:-v4}"
CORPUS_DIR="${CORPUS_DIR:-${PROJECT_DIR}/data/synthetic_controlled_100k_500_${HELDOUT_TEMPLATE_SET}}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_DIR}/outputs/synthetic_stepwise_100k_${STAMP}}"
BM25_DIR="${BM25_DIR:-${OUT_ROOT}/bm25}"
RULE_DIR="${RULE_DIR:-${OUT_ROOT}/rule_iterative}"
MODEL_DIR="${MODEL_DIR:-${OUT_ROOT}/model_guided}"
EVAL_DIR="${EVAL_DIR:-${OUT_ROOT}/evaluation}"

GPU_IDS="${GPU_IDS:-auto}"
IDLE_MEM_MB="${IDLE_MEM_MB:-1024}"
RUN_NLL="${RUN_NLL:-false}"
FORCE_PREPARE="${FORCE_PREPARE:-false}"
TARGET_BLOCKS="${TARGET_BLOCKS:-3}"

cd "$PROJECT_DIR"
mkdir -p "$OUT_ROOT" "$BM25_DIR" "$RULE_DIR" "$MODEL_DIR" "$EVAL_DIR"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

if [[ "$FORCE_PREPARE" == "true" || ! -f "$CORPUS_DIR/summary.json" ]]; then
  "$PY" src/prepare_synthetic_controlled_corpus.py \
    --model_name_or_path "$MODEL" \
    --output_dir "$CORPUS_DIR" \
    --seq_tokens 100000 \
    --block_tokens 256 \
    --num_queries 500 \
    --num_records 4 \
    --seed 20260714 \
    --split_disjoint_templates \
    --heldout_template_set "$HELDOUT_TEMPLATE_SET" \
    2>&1 | tee "$OUT_ROOT/prepare.log"
fi

"$PY" src/run_lexical_block_retrieval.py \
  --model_name_or_path "$MODEL" \
  --corpus_dir "$CORPUS_DIR" \
  --output_dir "$BM25_DIR" \
  --target_blocks 39 \
  --record_allocations 20,30,39 \
  --min_df 1 \
  --max_df 1.0 \
  2>&1 | tee "$OUT_ROOT/bm25.log"

"$PY" src/run_iterative_condition_retrieval.py \
  --model_name_or_path "$MODEL" \
  --corpus_dir "$CORPUS_DIR" \
  --output_dir "$RULE_DIR" \
  --target_blocks 39 \
  --min_df 1 \
  --max_df 1.0 \
  2>&1 | tee "$OUT_ROOT/rule_iterative.log"

if [[ "$GPU_IDS" == "auto" ]]; then
  mapfile -t FREE_GPU_ARRAY < <(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F, -v lim="$IDLE_MEM_MB" '{gsub(/ /,"",$1); gsub(/ /,"",$2); if (($2 + 0) <= (lim + 0)) print $1}'
  )
else
  IFS=',' read -r -a FREE_GPU_ARRAY <<< "$GPU_IDS"
fi

FREE_COUNT="${#FREE_GPU_ARRAY[@]}"
if [[ "$FREE_COUNT" -lt 1 ]]; then
  echo "[stepwise] no idle GPU found after CPU baselines" >&2
  exit 1
fi
if [[ "$FREE_COUNT" -gt 8 ]]; then
  FREE_GPU_ARRAY=("${FREE_GPU_ARRAY[@]:0:8}")
  FREE_COUNT=8
fi
DEVICES="$(IFS=,; echo "${FREE_GPU_ARRAY[*]}")"
echo "[stepwise] devices=$DEVICES"

CUDA_VISIBLE_DEVICES="$DEVICES" "$PY" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$FREE_COUNT" \
  src/run_model_guided_condition_retrieval.py \
    --device cuda \
    --model_name_or_path "$MODEL" \
    --corpus_dir "$CORPUS_DIR" \
    --output_dir "$MODEL_DIR" \
    --target_blocks "$TARGET_BLOCKS" \
    --candidate_blocks 16 \
    --candidate_sentence_scan 96 \
    --anchor_candidate_blocks 8 \
    --anchor_terms 1 \
    --bm25_weight 0.25 \
    --choice_weight 0 \
    --invalid_status_penalty 3 \
    --completion_thresholds 0,2,3,4 \
  2>&1 | tee "$OUT_ROOT/model_guided.log"

"$PY" src/evaluate_synthetic_benchmark.py \
  --corpus_dir "$CORPUS_DIR" \
  --output_dir "$EVAL_DIR" \
  --retrieval_result "bm25=${BM25_DIR}/query_results.csv" \
  --retrieval_result "rule=${RULE_DIR}/query_results.csv" \
  --retrieval_result "model=${MODEL_DIR}/query_results.csv" \
  --budgets 1,2,3 \
  2>&1 | tee "$OUT_ROOT/evaluation.log"

"$PY" src/analyze_model_guided_routes.py \
  --corpus_dir "$CORPUS_DIR" \
  --diagnostics "$MODEL_DIR/route_diagnostics.jsonl" \
  --output_dir "$MODEL_DIR/route_analysis" \
  --failure_examples 20 \
  2>&1 | tee "$OUT_ROOT/route_analysis.log"

if [[ "$RUN_NLL" == "true" ]]; then
  mkdir -p "$OUT_ROOT/answer_nll"
  CUDA_VISIBLE_DEVICES="$DEVICES" "$PY" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$FREE_COUNT" \
    src/evaluate_retrieved_answer_nll.py \
      --model_name_or_path "$MODEL" \
      --corpus_dir "$CORPUS_DIR" \
      --retrieval_results "$BM25_DIR/query_results.csv" \
      --retrieval_results "$MODEL_DIR/query_results.csv" \
      --output_dir "$OUT_ROOT/answer_nll" \
      --methods bm25_block,model_iterative_dynamic_0.25_threshold_3 \
      --target_blocks "$TARGET_BLOCKS" \
      --splits test \
      --dtype float16 \
      --attn_implementation sdpa \
    2>&1 | tee "$OUT_ROOT/answer_nll.log"
fi

echo "[stepwise] done: $OUT_ROOT"
