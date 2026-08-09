#!/usr/bin/env bash
set -u

BASE=/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
OUTPUT="$BASE/outputs/overnight_onehop_mechanisms_20260728"
EXISTING="$BASE/outputs/incremental_nine_newline_136k_144k_20260726/points.csv"
CRITICAL="$BASE/outputs/age_distractor_prerope_ablation_v2_20260726/counterfactual_heads.csv"

mkdir -p "$OUTPUT/logs"
printf '%s\n' "$$" > "$OUTPUT/launcher.pid"
rm -f "$OUTPUT/launcher.done"

export CUDA_VISIBLE_DEVICES=6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$BASE/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$BASE" || exit 1

overall_status=0

printf '%s output_variants started\n' "$(date --iso-8601=seconds)" \
    | tee "$OUTPUT/logs/output_variants.stage"
"$PYTHON" -u src/run_overnight_onehop_output_variants_8b.py \
    --model-name-or-path "$MODEL" \
    --output-dir "$OUTPUT" \
    --existing-points-csv "$EXISTING" \
    --prefill-chunk-size 128 \
    --dtype bfloat16 \
    --device-map balanced \
    --attn-implementation sdpa \
    --original-max-position-embeddings 40960 \
    --fixed-rope-factor 4 \
    --fixed-max-position-embeddings 147456 \
    --checkpoint-every 50 \
    --generation-max-new-tokens 32 \
    > "$OUTPUT/logs/output_variants.log" 2>&1
output_status=$?
printf '%s output_variants exit=%s\n' \
    "$(date --iso-8601=seconds)" "$output_status" \
    | tee -a "$OUTPUT/logs/output_variants.stage"
if [[ "$output_status" -ne 0 ]]; then
    overall_status=1
fi

printf '%s q_rope_probe started\n' "$(date --iso-8601=seconds)" \
    | tee "$OUTPUT/logs/q_rope_probe.stage"
"$PYTHON" -u src/run_overnight_onehop_q_rope_probe_8b.py \
    --model-name-or-path "$MODEL" \
    --critical-heads-csv "$CRITICAL" \
    --output-dir "$OUTPUT/q_rope_probe" \
    --prefill-chunk-size 128 \
    --dtype bfloat16 \
    --device-map balanced \
    --attn-implementation sdpa \
    --original-max-position-embeddings 40960 \
    --fixed-rope-factor 4 \
    --fixed-max-position-embeddings 147456 \
    > "$OUTPUT/logs/q_rope_probe.log" 2>&1
q_rope_status=$?
printf '%s q_rope_probe exit=%s\n' \
    "$(date --iso-8601=seconds)" "$q_rope_status" \
    | tee -a "$OUTPUT/logs/q_rope_probe.stage"
if [[ "$q_rope_status" -ne 0 ]]; then
    overall_status=1
fi

analysis_status=skipped
if [[ "$output_status" -eq 0 && "$q_rope_status" -eq 0 ]]; then
    printf '%s analysis started\n' "$(date --iso-8601=seconds)" \
        | tee "$OUTPUT/logs/analysis.stage"
    "$PYTHON" -u src/analyze_overnight_onehop_experiments.py \
        --output-root "$OUTPUT" \
        --existing-points-csv "$EXISTING" \
        > "$OUTPUT/logs/analysis.log" 2>&1
    analysis_status=$?
    printf '%s analysis exit=%s\n' \
        "$(date --iso-8601=seconds)" "$analysis_status" \
        | tee -a "$OUTPUT/logs/analysis.stage"
    if [[ "$analysis_status" -ne 0 ]]; then
        overall_status=1
    fi
fi

printf 'finished=%s\noutput_variants=%s\nq_rope_probe=%s\nanalysis=%s\noverall=%s\n' \
    "$(date --iso-8601=seconds)" \
    "$output_status" \
    "$q_rope_status" \
    "$analysis_status" \
    "$overall_status" \
    > "$OUTPUT/launcher.done"
exit "$overall_status"
