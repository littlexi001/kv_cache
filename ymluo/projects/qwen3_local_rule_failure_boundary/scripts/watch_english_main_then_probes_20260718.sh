#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary}"
MAIN="$PROJECT/outputs/attention_confidence_qwen3_8b_english_single_token_128k_20260718"
PROBES="$PROJECT/outputs/attention_confidence_qwen3_8b_english_mechanism_probes_20260718"

mkdir -p "$PROBES"
while [[ ! -f "$MAIN/launcher.done" ]]; do
  if [[ -f "$MAIN/launcher.failed" ]]; then
    echo "main experiment failed; probes not started" >&2
    date -Is > "$PROBES/watcher.failed"
    exit 1
  fi
  sleep 60
done

exec bash "$PROJECT/scripts/run_english_mechanism_probes_128k_server.sh"

