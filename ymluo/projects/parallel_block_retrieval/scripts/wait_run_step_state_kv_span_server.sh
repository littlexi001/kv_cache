#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-21600}"
started="$SECONDS"

while ((SECONDS - started < MAX_WAIT_SECONDS)); do
  free_count="$({
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
      awk -F',' '{gsub(/ /, "", $0); if (($2 + 0) <= 1024 && ($3 + 0) < 10) count += 1} END {print count + 0}'
  })"
  if ((free_count >= 1)); then
    echo "$(date -Iseconds) found $free_count idle GPU(s); starting step-state KV span run"
    exec bash "$ROOT/scripts/run_step_state_kv_span_server.sh"
  fi
  echo "$(date -Iseconds) waiting for an idle GPU"
  sleep "$POLL_SECONDS"
done

echo "Timed out after ${MAX_WAIT_SECONDS}s without an idle GPU" >&2
exit 2
