#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HOST_FILE="${1:-${PROJECT_ROOT}/configs/four_hosts.conf}"
: "${REMOTE_PROJECT:=/mnt/workspace/lym_code/scripts/rope_exp/qwen3_06b_aliyun_pe_pretraining_iclr27}"

if [[ ! -f "${HOST_FILE}" ]]; then
  echo "host file not found: ${HOST_FILE}" >&2
  echo "copy configs/four_hosts.example to configs/four_hosts.conf first" >&2
  exit 2
fi

mapfile -t ROWS < <(sed 's/\r$//' "${HOST_FILE}" | awk 'NF && $1 !~ /^#/ {print $1 " " $2}')
if [[ "${#ROWS[@]}" -ne 4 ]]; then
  echo "expected exactly four non-comment host rows, found ${#ROWS[@]}" >&2
  exit 2
fi

pids=()
for machine_id in 0 1 2 3; do
  read -r host gpu_list <<< "${ROWS[${machine_id}]}"
  if [[ -z "${host}" || ! "${gpu_list}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "invalid row for machine ${machine_id}: ${ROWS[${machine_id}]}" >&2
    exit 2
  fi
  echo "dispatch machine=${machine_id} host=${host} gpus=${gpu_list}"
  ssh -o BatchMode=yes "${host}" \
    "cd '${REMOTE_PROJECT}' && GPU_LIST='${gpu_list}' bash scripts/run_four_machine_worker.sh '${machine_id}'" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
if [[ "${failed}" -ne 0 ]]; then
  echo "one or more SSH dispatches failed; successful remote jobs remain running" >&2
  exit 1
fi
echo "all four background controllers were launched"
