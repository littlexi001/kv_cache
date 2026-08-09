#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HOST_FILE="${1:-${PROJECT_ROOT}/configs/sixteen_hosts.conf}"
: "${REMOTE_PROJECT:=/mnt/workspace/lym_code/scripts/rope_exp/qwen3_06b_aliyun_pe_pretraining_iclr27}"
if [[ ! -f "${HOST_FILE}" ]]; then echo "host file not found: ${HOST_FILE}" >&2; exit 2; fi
mapfile -t ROWS < <(sed 's/\r$//' "${HOST_FILE}" | awk 'NF && $1 !~ /^#/ {print $1 " " $2}')
if [[ "${#ROWS[@]}" -ne 16 ]]; then echo "expected 16 hosts" >&2; exit 2; fi
for task_id in $(seq 0 15); do
  read -r host _ <<< "${ROWS[${task_id}]}"
  strategy="$(python "${PROJECT_ROOT}/src/sixteen_machine_plan.py" --machine-id "${task_id}" --field strategy)"
  printf '\n=== task=%02d host=%s strategy=%s ===\n' "${task_id}" "${host}" "${strategy}"
  ssh -o BatchMode=yes "${host}" \
    "cd '${REMOTE_PROJECT}' && source scripts/common.sh && bash scripts/status.sh | grep -E '^${strategy}[[:space:]]'; tail -n 1 \"\$(run_dir_for '${strategy}')/controller_events.jsonl\" 2>/dev/null || true"
done
