#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
source /home/fdong/miniconda3/bin/activate moe

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

STAMP="${STAMP:-20260703_shortcut_ablation_v9}"
MODEL="${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}"
PY="${PY:-python}"
CONTEXT_TOKENS="${CONTEXT_TOKENS:-10000,20000}"
TASKS_PER_LENGTH="${TASKS_PER_LENGTH:-1}"
EVAL_TOKENS="${EVAL_TOKENS:-32}"
MODES="${MODES:-full,chain_typedhier_role_auto_p1}"
LAYOUTS="${LAYOUTS:-e05_d90,e20_d80,e35_d70}"
ABLATIONS="${ABLATIONS:-answerline_override answerline_no_override natural_no_override raw_sparse_kv}"

PARAGRAPH_MIN_TOKENS="${PARAGRAPH_MIN_TOKENS:-256}"
PARAGRAPH_MAX_TOKENS="${PARAGRAPH_MAX_TOKENS:-1024}"
SECTION_MAX_PARAGRAPHS="${SECTION_MAX_PARAGRAPHS:-8}"
CHAPTER_MAX_PAGES="${CHAPTER_MAX_PAGES:-64}"
GLOBAL_INDEX_UPDATE_PAGES="${GLOBAL_INDEX_UPDATE_PAGES:-128}"

ROOT="outputs/range_sdpa_shortcut_ablation_v9_${STAMP}"
LOG_ROOT="outputs/logs"
mkdir -p "$ROOT" "$LOG_ROOT"
STATUS="$ROOT/run_status.csv"
echo "ablation,status,output_dir,log" > "$STATUS"

for ABLATION in $ABLATIONS; do
  TYPED_RECORD_MODE="extractive"
  TYPED_RECORD_FORMAT="answerline_summary"
  TYPED_SUMMARY_SOURCE_MODE="chain_typedhier_auto_p1"
  TYPED_RECORD_ANSWER_OVERRIDE="false"
  SKIP_LM_ANSWER_WHEN_OVERRIDE="false"
  TYPED_RECORD_INSERT="true"

  case "$ABLATION" in
    answerline_override)
      TYPED_RECORD_FORMAT="answerline_summary"
      TYPED_RECORD_ANSWER_OVERRIDE="true"
      SKIP_LM_ANSWER_WHEN_OVERRIDE="true"
      ;;
    answerline_no_override)
      TYPED_RECORD_FORMAT="answerline_summary"
      ;;
    natural_no_override)
      TYPED_RECORD_FORMAT="natural_summary"
      ;;
    raw_sparse_kv)
      TYPED_RECORD_MODE="none"
      TYPED_RECORD_FORMAT="verbose"
      TYPED_SUMMARY_SOURCE_MODE=""
      TYPED_RECORD_INSERT="false"
      ;;
    *)
      echo "Unknown ablation: $ABLATION" >&2
      exit 2
      ;;
  esac

  OUT="$ROOT/$ABLATION"
  LOG="$LOG_ROOT/range_sdpa_shortcut_ablation_v9_${ABLATION}_${STAMP}.log"
  mkdir -p "$OUT"
  set +e
  "$PY" src/run_longrange_book_index_sparse_eval.py \
    --model_name_or_path "$MODEL" \
    --output_dir "$OUT" \
    --context_tokens "$CONTEXT_TOKENS" \
    --tasks_per_length "$TASKS_PER_LENGTH" \
    --eval_tokens "$EVAL_TOKENS" \
    --task_variant chain_story_conflict \
    --suite_layouts "$LAYOUTS" \
    --modes "$MODES" \
    --score_query_ppl true \
    --score_calibrated true \
    --balanced_labels true \
    --answer_score_format gated_sentence \
    --sparse_attention_impl range_sdpa \
    --typed_record_mode "$TYPED_RECORD_MODE" \
    --typed_record_format "$TYPED_RECORD_FORMAT" \
    --typed_summary_source_mode "$TYPED_SUMMARY_SOURCE_MODE" \
    --typed_record_answer_override "$TYPED_RECORD_ANSWER_OVERRIDE" \
    --typed_record_insert "$TYPED_RECORD_INSERT" \
    --skip_lm_answer_when_override "$SKIP_LM_ANSWER_WHEN_OVERRIDE" \
    --paragraph_min_tokens "$PARAGRAPH_MIN_TOKENS" \
    --paragraph_max_tokens "$PARAGRAPH_MAX_TOKENS" \
    --section_max_paragraphs "$SECTION_MAX_PARAGRAPHS" \
    --chapter_max_pages "$CHAPTER_MAX_PAGES" \
    --global_index_update_pages "$GLOBAL_INDEX_UPDATE_PAGES" \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    --attn_implementation eager \
    2>&1 | tee "$LOG"
  STATUS_CODE=${PIPESTATUS[0]}
  set -e
  if [[ "$STATUS_CODE" -eq 0 ]]; then
    echo "$ABLATION,OK,$OUT,$LOG" >> "$STATUS"
  else
    echo "$ABLATION,FAILED,$OUT,$LOG" >> "$STATUS"
  fi
done

"$PY" - "$ROOT" <<'PY'
from pathlib import Path
import csv
import sys

root = Path(sys.argv[1])
rows = []
for summary in sorted(root.glob("*/sparse_summary.csv")):
    ablation = summary.parent.name
    with summary.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row["ablation"] = ablation
            rows.append(row)

fields = [
    "ablation",
    "context_tokens",
    "mode",
    "tasks",
    "accuracy",
    "calibrated_accuracy",
    "query_ppl",
    "mean_eval_seconds",
    "mean_query_pipeline_seconds",
    "mean_end_to_end_seconds",
    "mean_kept_fraction",
    "mean_kept_tokens",
    "answer_override_rate",
    "lm_answer_scored_rate",
    "typed_record_coverage",
    "typed_record_accuracy",
    "decoy_hit_rate",
    "evidence_hit_rate",
]
out = root / "combined_summary.csv"
with out.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
print(out)
PY

echo "$ROOT"
