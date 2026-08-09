#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HOST_FILE="${1:-${PROJECT_ROOT}/configs/sixteen_hosts.conf}"
: "${REMOTE_PROJECT:=/mnt/workspace/lym_code/scripts/rope_exp/qwen3_06b_aliyun_pe_pretraining_iclr27}"
if [[ ! -f "${HOST_FILE}" ]]; then
  echo "host file not found: ${HOST_FILE}" >&2
  echo "copy configs/sixteen_hosts.example to configs/sixteen_hosts.conf first" >&2
  exit 2
fi
mapfile -t ROWS < <(sed 's/\r$//' "${HOST_FILE}" | awk 'NF && $1 !~ /^#/ {print $1 " " $2}')
if [[ "${#ROWS[@]}" -ne 16 ]]; then
  echo "expected exactly 16 host rows, found ${#ROWS[@]}" >&2
  exit 2
fi
pids=()
for task_id in $(seq 0 15); do
  read -r host gpu_list <<< "${ROWS[${task_id}]}"
  if [[ -z "${host}" || ! "${gpu_list}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "invalid host row ${task_id}: ${ROWS[${task_id}]}" >&2
    exit 2
  fi
  echo "dispatch task=${task_id} host=${host} gpus=${gpu_list}"
  ssh -o BatchMode=yes "${host}" \
    "cd '${REMOTE_PROJECT}' && GPU_LIST='${gpu_list}' bash scripts/run_pretrain_worker.sh '${task_id}'" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
if [[ "${failed}" -ne 0 ]]; then
  echo "one or more SSH dispatches failed; successful background jobs remain running" >&2
  exit 1
fi
echo "all sixteen pretraining controllers and TensorBoard servers were dispatched"
