#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTDIR="${OUTDIR:-fdong_embedding_dim/outputs/toy_gradient_interference}"
STEPS="${STEPS:-2000}"
RECORD_EVERY="${RECORD_EVERY:-20}"
SEED="${SEED:-0}"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp}"

run_one() {
  local name="$1"
  local dim="$2"
  local layout="$3"
  local probs="$4"
  echo "================================================================================"
  echo "Running ${name}"
  "${PYTHON_BIN}" -u fdong_embedding_dim/scripts/analyze_toy_gradient_interference.py \
    --experiment_name "${name}" \
    --dim "${dim}" \
    --init_layout "${layout}" \
    --group_probs "${probs}" \
    --steps "${STEPS}" \
    --record_every "${RECORD_EVERY}" \
    --seed "${SEED}" \
    --outdir "${OUTDIR}"
}

UNIFORM_PROBS="common:0.70,tail1:0.10,tail2:0.10,tail3:0.10"
ZIPF_PROBS="common:0.70,tail1:0.20,tail2:0.07,tail3:0.03"

# Mechanism test A: if tail starts outside common direction, uniform tail should
# keep higher residual effective rank and similar tail quality.
run_one "toy2d_uniform_spread_grad_interference" 2 "spread" "${UNIFORM_PROBS}"
run_one "toy3d_uniform_spread_grad_interference" 3 "spread" "${UNIFORM_PROBS}"

# Mechanism test B: tail-internal Zipf should create a tail-high/tail-low SIR and
# margin ordering, even when tail starts outside common direction.
run_one "toy2d_zipf_spread_grad_interference" 2 "spread" "${ZIPF_PROBS}"
run_one "toy3d_zipf_spread_grad_interference" 3 "spread" "${ZIPF_PROBS}"

# Mechanism test C: packed initialization should keep the tail in a lower
# effective-dimensional residual subspace, despite raw dimension being larger.
run_one "toy2d_uniform_packed_common_grad_interference" 2 "packed_common" "${UNIFORM_PROBS}"
run_one "toy3d_uniform_packed_common_grad_interference" 3 "packed_common" "${UNIFORM_PROBS}"

# Mechanism test D: packed + Zipf combines low residual d_eff with tail-internal
# frequency competition.
run_one "toy2d_zipf_packed_common_grad_interference" 2 "packed_common" "${ZIPF_PROBS}"
run_one "toy3d_zipf_packed_common_grad_interference" 3 "packed_common" "${ZIPF_PROBS}"

"${PYTHON_BIN}" - "${OUTDIR}" <<'PY'
import csv
import json
import sys
from pathlib import Path

outdir = Path(sys.argv[1])
rows = []
for path in sorted(outdir.glob("*/final_compact.json")):
    data = json.loads(path.read_text())
    name = data["experiment_name"]
    parts = name.split("_")
    rows.append({
        "experiment_name": name,
        "dim": parts[0].replace("toy", "").replace("d", ""),
        "tail_distribution": parts[1],
        "init_layout": "packed_common" if "packed_common" in name else "spread",
        "final_loss": data["final_loss"],
        "final_tail_grad_eff_rank": data["final_tail_grad_eff_rank"],
        "final_tail_rep_residual_eff_rank": data["final_tail_rep_residual_eff_rank"],
        "final_tail_grad_cosine_mean": data["final_tail_grad_cosine_mean"],
        "final_common_tail_grad_cosine_mean": data["final_common_tail_grad_cosine_mean"],
        "final_tail_sir_mean": sum(data["final_tail_sir"].values()) / len(data["final_tail_sir"]),
        **{f"sir_{k}": v for k, v in data["final_tail_sir"].items()},
    })

summary_path = outdir / "all_runs_summary.csv"
summary_path.parent.mkdir(parents=True, exist_ok=True)
fields = [
    "experiment_name",
    "dim",
    "tail_distribution",
    "init_layout",
    "final_loss",
    "final_tail_grad_eff_rank",
    "final_tail_rep_residual_eff_rank",
    "final_tail_grad_cosine_mean",
    "final_common_tail_grad_cosine_mean",
    "final_tail_sir_mean",
    "sir_tail1",
    "sir_tail2",
    "sir_tail3",
]
with summary_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {summary_path}")
PY

echo "================================================================================"
if [[ "${OUTDIR}" = /* ]]; then
  echo "Toy gradient-interference experiments finished. Results: ${OUTDIR}"
else
  echo "Toy gradient-interference experiments finished. Results: ${REPO_ROOT}/${OUTDIR}"
fi
