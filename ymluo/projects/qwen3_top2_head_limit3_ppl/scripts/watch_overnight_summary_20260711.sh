#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
DOC_ROOT="${DOC_ROOT:-/home/fdong/ymluo/doc}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
cd "$ROOT"

B16_TABLE="outputs/riskkv_v19_v301_v302_b16_group_sweep_20260711/summary_table.csv"
BOUNDED_TABLE="outputs/riskkv_v19_v304_v305_bounded_fallback_20260711/summary_table.csv"
REPORT_DIR="outputs/riskkv_v19_overnight_summary_20260711"
REPORT_MD="$REPORT_DIR/report.md"
DOC_MD="$DOC_ROOT/section149_overnight_exploration_results_20260711.md"
mkdir -p "$REPORT_DIR" "$DOC_ROOT"

while [[ ! -f "$B16_TABLE" || ! -f "$BOUNDED_TABLE" ]]; do
  echo "WAIT overnight summary b16=$([[ -f "$B16_TABLE" ]] && echo yes || echo no) bounded=$([[ -f "$BOUNDED_TABLE" ]] && echo yes || echo no) $(date -Is)"
  sleep 300
done

"$PYTHON" - <<'PY' > "$REPORT_MD"
import csv
from pathlib import Path


def read_csv(path: str) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fnum(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def fmt_pct(value: str) -> str:
    return f"{100.0 * fnum(value):.2f}%"


def fmt_float(value: str) -> str:
    return f"{fnum(value):.6f}"


def task_rows(directory: str) -> dict[str, dict[str, str]]:
    path = Path(directory) / "summary.csv"
    if not path.exists():
        return {}
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("benchmark") == "longbench" and row.get("task") != "ALL":
                rows[row["task"]] = row
    return rows


b16 = read_csv("outputs/riskkv_v19_v301_v302_b16_group_sweep_20260711/summary_table.csv")
bounded = read_csv("outputs/riskkv_v19_v304_v305_bounded_fallback_20260711/summary_table.csv")

all_methods = []
seen = set()
for row in b16 + bounded:
    key = row["method"]
    if key not in seen:
        seen.add(key)
        all_methods.append(row)

v300 = next(row for row in all_methods if row["method"] == "v300_main")
best_score = max(all_methods, key=lambda row: fnum(row["score"]))
best_kv_under_score = max(
    all_methods,
    key=lambda row: (fnum(row["score"]) >= 0.995 * fnum(v300["score"]), -fnum(row["kv_keep"])),
)
lowest_kv = min(all_methods, key=lambda row: fnum(row["kv_keep"]))

lines = []
lines.append("# Section 149: overnight exploration results")
lines.append("")
lines.append("日期：2026-07-11")
lines.append("")
lines.append("## Overall")
lines.append("")
lines.append("| Method | Samples | Score | KV keep | Online seconds |")
lines.append("|---|---:|---:|---:|---:|")
for row in all_methods:
    lines.append(
        f"| {row['method']} | {row['samples']} | {fmt_float(row['score'])} | {fmt_pct(row['kv_keep'])} | {float(row['online_seconds']):.4f} |"
    )
lines.append("")
lines.append("## Automatic read")
lines.append("")
lines.append(
    f"- v300 baseline for this comparison: score {fmt_float(v300['score'])}, KV {fmt_pct(v300['kv_keep'])}, online {float(v300['online_seconds']):.4f}s."
)
lines.append(
    f"- Best score point: `{best_score['method']}` with score {fmt_float(best_score['score'])}, KV {fmt_pct(best_score['kv_keep'])}."
)
lines.append(
    f"- Lowest KV point: `{lowest_kv['method']}` with score {fmt_float(lowest_kv['score'])}, KV {fmt_pct(lowest_kv['kv_keep'])}."
)
lines.append(
    f"- Best KV among points within 99.5% of v300 score by this simple gate: `{best_kv_under_score['method']}`."
)
lines.append("")

b16_rows = {row["method"]: row for row in b16}
if "v301_b16_group4" in b16_rows and "v302_b16_group2" in b16_rows:
    lines.append("## b16 group-size conclusion")
    lines.append("")
    for name in ["v301_b16_group4", "v302_b16_group2", "v303_v289_b16_group8"]:
        if name in b16_rows:
            row = b16_rows[name]
            delta = fnum(row["score"]) - fnum(v300["score"])
            lines.append(f"- `{name}`: score delta vs v300 = {delta:+.6f}, KV = {fmt_pct(row['kv_keep'])}.")
    lines.append("")
    if max(fnum(b16_rows[name]["score"]) for name in b16_rows if name != "v300_main") > fnum(v300["score"]):
        lines.append("b16 group-size sweep has at least one positive point; inspect task-level deltas before adopting.")
    else:
        lines.append("b16 group-size sweep does not beat v300 overall; keep b16 as ablation / task-specific locator rather than main method.")
    lines.append("")

bounded_rows = {row["method"]: row for row in bounded}
if "v304_bounded4k" in bounded_rows and "v305_bounded3k" in bounded_rows:
    lines.append("## bounded fallback conclusion")
    lines.append("")
    for name in ["v304_bounded4k", "v305_bounded3k"]:
        row = bounded_rows[name]
        score_delta = fnum(row["score"]) - fnum(v300["score"])
        kv_delta = fnum(row["kv_keep"]) - fnum(v300["kv_keep"])
        online_delta = fnum(row["online_seconds"]) - fnum(v300["online_seconds"])
        lines.append(
            f"- `{name}`: score delta {score_delta:+.6f}, KV delta {100 * kv_delta:+.2f} pp, online delta {online_delta:+.4f}s."
        )
    lines.append("")
    if any(
        fnum(bounded_rows[name]["score"]) >= 0.995 * fnum(v300["score"])
        and fnum(bounded_rows[name]["kv_keep"]) < fnum(v300["kv_keep"])
        for name in ["v304_bounded4k", "v305_bounded3k"]
    ):
        lines.append("bounded fallback produced a likely useful Pareto point; next step is M150 / extra-50 validation.")
    else:
        lines.append("bounded fallback did not produce a clean overall Pareto point; use task-level deltas to train a selective router.")
    lines.append("")

lines.append("## bounded fallback task deltas")
lines.append("")
base_tasks = task_rows(v300["output_dir"])
for method_name in ["v304_bounded4k", "v305_bounded3k"]:
    if method_name not in bounded_rows:
        continue
    candidate_tasks = task_rows(bounded_rows[method_name]["output_dir"])
    lines.append(f"### {method_name}")
    lines.append("")
    lines.append("| Task | Score delta | KV delta pp | Online delta s |")
    lines.append("|---|---:|---:|---:|")
    for task in ["narrativeqa", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "qmsum", "repobench-p"]:
        if task not in base_tasks or task not in candidate_tasks:
            continue
        b = base_tasks[task]
        c = candidate_tasks[task]
        lines.append(
            f"| {task} | {fnum(c['score']) - fnum(b['score']):+.6f} | {100 * (fnum(c['mean_keep_fraction']) - fnum(b['mean_keep_fraction'])):+.2f} | {fnum(c['mean_online_seconds']) - fnum(b['mean_online_seconds']):+.4f} |"
        )
    lines.append("")

lines.append("## Next action")
lines.append("")
lines.append("1. If bounded fallback has a task-level win, train/encode a selective action router instead of applying it globally.")
lines.append("2. If b16 wins only on hotpot-like tasks, keep it as a task-local locator action.")
lines.append("3. Validate any selected point on M150 and extra-50 before promoting it to the paper mainline.")
lines.append("")

print("\n".join(lines))
PY

cp "$REPORT_MD" "$DOC_MD"
cat "$REPORT_MD"
echo "DONE overnight summary $(date -Is)"
