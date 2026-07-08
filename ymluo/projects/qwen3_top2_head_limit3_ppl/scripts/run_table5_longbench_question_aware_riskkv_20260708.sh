#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/etc/profile.d/conda.sh
conda activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

export TRANSFORMERS_VERBOSITY=error

STAMP="${STAMP:-20260708_table5_question_aware}"
TASKS="${TASKS:-narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p}"
BUDGETS="${BUDGETS:-128 256 512 1024 2048}"
GPU_LIST="${GPU_LIST:-4 5 6 7}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-1000000}"
MAX_CONTEXT="${MAX_CONTEXT:-7500}"
PAGE_TOKENS="${PAGE_TOKENS:-128}"
PREFILL_CHUNK="${PREFILL_CHUNK:-2048}"
SINK_TOKENS="${SINK_TOKENS:-64}"
RECENT_TOKENS="${RECENT_TOKENS:-64}"
OURS_SCORER="${OURS_SCORER:-hybrid_late_mmr}"
ATTN="${ATTN:-sdpa}"
DTYPE="${DTYPE:-float16}"
RUN_FULL_SANITY="${RUN_FULL_SANITY:-1}"
FULL_SANITY_SAMPLES_PER_TASK="${FULL_SANITY_SAMPLES_PER_TASK:-1000000}"

OUT_ROOT="/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/table5_question_aware_riskkv_${STAMP}"
LOG_ROOT="/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/logs"
RUNNER="/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/src/run_controlled_public_kv_benchmark_v1.py"
LB_SRC="/home/fdong/ymluo/external/KVCache-Factory/data/LongBench"
LB_ZIP="$OUT_ROOT/longbench_data.zip"
STATUS="$OUT_ROOT/run_status.csv"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
echo "kind,budget,gpu,status,output_dir,log" > "$STATUS"

find_model() {
  local candidates=(
    "/home/fdong/models/meta-llama/Llama-3.1-8B-Instruct"
    "/home/fdong/qwen/Llama-3.1-8B-Instruct"
    "/home/fdong/qwen/Meta-Llama-3.1-8B-Instruct"
    "/home/fdong/Llama-3.1-8B-Instruct"
  )
  for path in "${candidates[@]}"; do
    if [[ -f "$path/config.json" ]]; then
      echo "$path"
      return 0
    fi
  done
  for root in \
    "/home/fdong/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots" \
    "/home/fdong/ymluo/hf_cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots"; do
    if [[ -d "$root" ]]; then
      for snapshot in "$root"/*; do
        if [[ -f "$snapshot/config.json" ]]; then
          echo "$snapshot"
          return 0
        fi
      done
    fi
  done
  return 1
}

MODEL="${MODEL:-}"
if [[ -z "$MODEL" ]]; then
  if MODEL="$(find_model)"; then
    echo "Using local model: $MODEL"
  else
    MODEL="/home/fdong/models/meta-llama/Llama-3.1-8B-Instruct"
    echo "Local Llama-3.1-8B-Instruct not found; trying gated official repo then public mirror."
    python - <<'PY'
from huggingface_hub import snapshot_download
attempts = [
    ("meta-llama/Llama-3.1-8B-Instruct", "/home/fdong/models/meta-llama/Llama-3.1-8B-Instruct"),
    ("NousResearch/Meta-Llama-3.1-8B-Instruct", "/home/fdong/models/NousResearch-Meta-Llama-3.1-8B-Instruct"),
]
last_error = None
for repo_id, local_dir in attempts:
    try:
        print(f"DOWNLOAD {repo_id} -> {local_dir}", flush=True)
        snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
        print(f"MODEL_READY {local_dir}", flush=True)
        break
    except Exception as exc:
        last_error = exc
        print(f"DOWNLOAD_FAILED {repo_id}: {type(exc).__name__}: {str(exc).splitlines()[0]}", flush=True)
else:
    raise last_error
PY
    if [[ -f "/home/fdong/models/NousResearch-Meta-Llama-3.1-8B-Instruct/config.json" ]]; then
      MODEL="/home/fdong/models/NousResearch-Meta-Llama-3.1-8B-Instruct"
    fi
  fi
fi

python - "$MODEL" <<'PY'
import sys
from transformers import AutoTokenizer
model = sys.argv[1]
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
print("MODEL", model)
print("HAS_CHAT_TEMPLATE", bool(getattr(tok, "chat_template", None)))
if not bool(getattr(tok, "chat_template", None)):
    raise SystemExit("Refusing to run Table 5: tokenizer has no chat_template, likely not an Instruct checkpoint.")
PY

if [[ ! -f "$LB_ZIP" ]]; then
  python - "$LB_SRC" "$LB_ZIP" "$TASKS" <<'PY'
import sys
import zipfile
from pathlib import Path

src = Path(sys.argv[1])
out = Path(sys.argv[2])
tasks = [item.strip() for item in sys.argv[3].split(",") if item.strip()]
out.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for task in tasks:
        path = src / f"{task}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        zf.write(path, arcname=f"data/{task}.jsonl")
print(out)
PY
fi

read -r -a GPUS <<< "$GPU_LIST"

run_job() {
  local kind="$1"
  local gpu="$2"
  local budget="$3"
  local methods="$4"
  local samples="$5"
  local out="$OUT_ROOT/${kind}_b${budget}"
  local log="$LOG_ROOT/table5_${kind}_b${budget}_${STAMP}.log"
  mkdir -p "$out"
  echo "START kind=$kind budget=$budget gpu=$gpu methods=$methods samples=$samples out=$out $(date -Is)"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    set +e
    python "$RUNNER" \
      --model_name_or_path "$MODEL" \
      --output_dir "$out" \
      --benchmarks longbench \
      --longbench_tasks "$TASKS" \
      --max_samples_per_task "$samples" \
      --max_context_tokens "$MAX_CONTEXT" \
      --prefill_chunk_tokens "$PREFILL_CHUNK" \
      --methods "$methods" \
      --budget_tokens "$budget" \
      --sink_tokens "$SINK_TOKENS" \
      --recent_tokens "$RECENT_TOKENS" \
      --page_tokens "$PAGE_TOKENS" \
      --ours_scorer "$OURS_SCORER" \
      --dtype "$DTYPE" \
      --attn_implementation "$ATTN" \
      --prompt_wrapper llama3 \
      --longbench_zip_path "$LB_ZIP" \
      --log_every 20 \
      > "$log" 2>&1
    status=$?
    set -e
    if [[ "$status" -eq 0 ]]; then
      echo "$kind,$budget,$gpu,OK,$out,$log" >> "$STATUS"
    else
      echo "$kind,$budget,$gpu,FAILED,$out,$log" >> "$STATUS"
    fi
    echo "DONE kind=$kind budget=$budget gpu=$gpu status=$status $(date -Is)"
    exit "$status"
  ) &
}

active=0
idx=0
for budget in $BUDGETS; do
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  run_job "riskkv_question_aware" "$gpu" "$budget" "ours_page_gather" "$SAMPLES_PER_TASK"
  active=$((active + 1))
  idx=$((idx + 1))
  if [[ "$active" -ge "${#GPUS[@]}" ]]; then
    wait || true
    active=0
  fi
done
wait || true

if [[ "$RUN_FULL_SANITY" == "1" ]]; then
  gpu="${GPUS[0]}"
  run_job "fullkv_sanity" "$gpu" "2048" "full_kv" "$FULL_SANITY_SAMPLES_PER_TASK"
  wait || true
fi

echo "ALL TABLE5 QUESTION-AWARE RISKKV JOBS DONE $(date -Is)"
