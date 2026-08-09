#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/etc/profile.d/conda.sh
conda activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
DATA_JSON="${DATA_JSON:-/home/fdong/ymluo/datasets/LongBench-v2/data.json}"
GPUS="${GPUS:-5,6,7}"
METHOD="${METHOD:-ours_page_gather}"
POLICY="${POLICY-configs/riskkv_operator_contract_v466_retrieve896_code256_20260713.json}"
LABEL="${LABEL:-v466}"
SAMPLES="${SAMPLES:-503}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-32000}"
DOMAINS="${DOMAINS:-}"
CHOICE_DECODE="${CHOICE_DECODE:-0}"
FORCE_DECODE_TOKENS="${FORCE_DECODE_TOKENS:-0}"
SPARSE_QUERY_TOKENWISE="${SPARSE_QUERY_TOKENWISE:-0}"
SPARSE_QUERY_PHYSICAL_MASK="${SPARSE_QUERY_PHYSICAL_MASK:-0}"
SPARSE_POSITION_MODE="${SPARSE_POSITION_MODE:-original}"
STAMP="${STAMP:-20260714_longbench_v2}"
LOG_ROOT="outputs/logs"
LOCK_ROOT="${LOCK_ROOT:-/tmp/riskkv_longbench_v2_gpu_locks_${USER:-user}}"
GPU_MAX_USED_MB="${GPU_MAX_USED_MB:-1000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-15}"
mkdir -p "$LOG_ROOT" "$LOCK_ROOT"

choose_gpu() {
  while true; do
    IFS=',' read -ra ids <<< "$GPUS"
    for gpu in "${ids[@]}"; do
      gpu="${gpu//[[:space:]]/}"
      [[ -z "$gpu" ]] && continue
      read -r used util < <(
        nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$gpu" \
          | tr -d ',' | awk '{print $1, $2}'
      )
      if [[ "${used:-99999}" -lt "$GPU_MAX_USED_MB" && "${util:-100}" -lt "$GPU_MAX_UTIL" ]]; then
        local lock_dir="$LOCK_ROOT/gpu${gpu}.lock"
        if mkdir "$lock_dir" 2>/dev/null; then
          echo "$$" > "$lock_dir/pid"
          echo "$gpu"
          return 0
        fi
        local lock_pid=""
        [[ -f "$lock_dir/pid" ]] && lock_pid="$(cat "$lock_dir/pid" 2>/dev/null || true)"
        if [[ -n "$lock_pid" ]] && ! kill -0 "$lock_pid" 2>/dev/null; then
          rm -rf "$lock_dir"
        fi
      fi
    done
    echo "WAIT_GPU label=$LABEL gpus=$GPUS $(date -Is)" >&2
    sleep 60
  done
}

GPU="$(choose_gpu)"
GPU_LOCK_DIR="$LOCK_ROOT/gpu${GPU}.lock"
trap 'rm -rf "$GPU_LOCK_DIR"' EXIT

OUT="outputs/${STAMP}_${LABEL}_m${SAMPLES}_c${MAX_CONTEXT_TOKENS}"
LOG="$LOG_ROOT/${STAMP}_${LABEL}_m${SAMPLES}_c${MAX_CONTEXT_TOKENS}.log"
mkdir -p "$OUT"

POLICY_ARGS=()
if [[ -n "$POLICY" ]]; then
  POLICY_ARGS+=(--ours_task_policy_json "$POLICY")
fi
DOMAIN_ARGS=()
if [[ -n "$DOMAINS" ]]; then
  DOMAIN_ARGS+=(--longbench_v2_domains "$DOMAINS")
fi
CHOICE_ARGS=()
if [[ "$CHOICE_DECODE" == "1" ]]; then
  CHOICE_ARGS+=(--constrained_choice_decode)
fi
if [[ "$SPARSE_QUERY_TOKENWISE" == "1" ]]; then
  CHOICE_ARGS+=(--sparse_query_tokenwise)
fi
if [[ "$SPARSE_QUERY_PHYSICAL_MASK" == "1" ]]; then
  CHOICE_ARGS+=(--sparse_query_physical_mask)
fi
CHOICE_ARGS+=(--sparse_position_mode "$SPARSE_POSITION_MODE")

echo "START label=$LABEL method=$METHOD gpu=$GPU samples=$SAMPLES context=$MAX_CONTEXT_TOKENS $(date -Is)"
CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python src/run_controlled_public_kv_benchmark_v1.py \
  --model_name_or_path "$MODEL" \
  --output_dir "$OUT" \
  --benchmarks longbench_v2 \
  --longbench_v2_json_path "$DATA_JSON" \
  --max_samples_per_task "$SAMPLES" \
  --max_context_tokens "$MAX_CONTEXT_TOKENS" \
  --force_decode_tokens "$FORCE_DECODE_TOKENS" \
  --prefill_chunk_tokens 2048 \
  --methods "$METHOD" \
  --budget_tokens 1024 \
  --sink_tokens 32 \
  --recent_tokens 64 \
  --page_tokens 16 \
  --ours_scorer hybrid_late_mmr_multiscale_idf_flow \
  --ours_multiscale_group_pages 4 \
  --ours_multiscale_weight 0.22 \
  --ours_flow_neighbor_radius 1 \
  --ours_flow_neighbor_budget_fraction 0.16 \
  --ours_flow_neighbor_min_score 0.12 \
  --ours_flow_score_smooth_weight 0.16 \
  --ours_flow_anchor_boost 0.20 \
  --ours_bridge_budget_fraction 0.16 \
  --ours_bridge_min_score 0.0 \
  --ours_bridge_max_terms 24 \
  --dtype float16 \
  --attn_implementation sdpa \
  --prompt_wrapper llama3 \
  --log_every 10 \
  "${POLICY_ARGS[@]}" \
  "${DOMAIN_ARGS[@]}" \
  "${CHOICE_ARGS[@]}" \
  > "$LOG" 2>&1

echo "DONE label=$LABEL output=$OUT $(date -Is)"
cat "$OUT/summary.csv"
