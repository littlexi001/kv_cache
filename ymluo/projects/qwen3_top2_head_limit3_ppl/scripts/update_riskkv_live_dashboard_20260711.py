#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


ROOT = Path("/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl")
DOC_PATH = Path("/home/fdong/ymluo/doc/section154_live_dashboard_20260711.md")
OUT_DIR = ROOT / "outputs/riskkv_v19_live_dashboard_20260711"
V300_DIR = ROOT / "outputs/riskkv_v19_v300_v294_no_2wiki_20260711_m100_bDyn_pDyn"
FULL_DIR = ROOT / "outputs/riskkv_fullkv_m100_same_samples_20260710"

CURRENT_PREFIXES = (
    "riskkv_v19_v289_",
    "riskkv_v19_v301_",
    "riskkv_v19_v302_",
    "riskkv_v19_v304_",
    "riskkv_v19_v305_",
    "riskkv_v19_v306_",
    "riskkv_v19_v307_",
    "riskkv_v19_v308_",
    "riskkv_v19_v309_",
    "riskkv_v19_v310_",
    "riskkv_v19_v311_",
    "riskkv_v19_v312_",
    "riskkv_v19_v313_",
    "riskkv_v19_v314_",
    "riskkv_v19_v315_",
    "riskkv_v19_v316_",
    "riskkv_v19_v317_",
    "riskkv_v19_v318_",
    "riskkv_v19_v319_",
)

FINAL_REPORT_DIRS = [
    "outputs/riskkv_v19_v301_v302_b16_group_sweep_20260711",
    "outputs/riskkv_v19_v304_v305_bounded_fallback_20260711",
    "outputs/riskkv_v19_v307_v308_b16_purefine_sweep_20260711",
    "outputs/riskkv_v19_v309_v310_b16_microspan_sweep_20260711",
    "outputs/riskkv_v19_v306_repobench_retry_20260711",
    "outputs/riskkv_v19_v311_safe_speedpatch_20260711",
    "outputs/riskkv_v19_v312_v313_b16_windowvote_sweep_20260711",
    "outputs/riskkv_v19_v314_v315_bm25_bridge_smoke_20260711",
    "outputs/riskkv_v19_v316_v317_qa_shortdecode_smoke_20260711",
    "outputs/riskkv_v19_v318_v319_qasper_bm25_budget_smoke_20260711",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fnum(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def read_summary_by_task(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in read_csv(path):
        if row.get("benchmark") not in {"ALL", "longbench"}:
            continue
        task = row.get("task", "")
        if not task:
            continue
        out[task] = {
            "score": fnum(row, "score"),
            "kv": fnum(row, "mean_keep_fraction"),
            "online": fnum(row, "mean_online_seconds"),
            "samples": fnum(row, "samples"),
        }
    return out


def summarize_task_results(path: Path) -> dict[str, object] | None:
    rows = read_csv(path)
    if not rows:
        return None
    label = path.parent.name.replace("riskkv_v19_", "")
    task = rows[0].get("task", "")
    score = mean([fnum(row, "score") for row in rows])
    kv = mean([fnum(row, "keep_fraction") for row in rows])
    online = mean([fnum(row, "online_seconds") for row in rows])
    total = mean([fnum(row, "total_seconds") for row in rows])
    return {
        "label": label,
        "task": task,
        "samples": len(rows),
        "score": score,
        "kv": kv,
        "online": online,
        "total": total,
        "path": str(path.parent.relative_to(ROOT)),
    }


def final_summary_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for report_dir in FINAL_REPORT_DIRS:
        path = ROOT / report_dir / "summary_table.csv"
        if not path.exists():
            rows.append(
                {
                    "report": report_dir,
                    "method": "<pending>",
                    "samples": "",
                    "score": "",
                    "kv_keep": "",
                    "online_seconds": "",
                    "output_dir": "",
                }
            )
            continue
        for row in read_csv(path):
            item = {"report": report_dir}
            item.update(row)
            rows.append(item)
    return rows


def status_note(row: dict[str, object], full: dict[str, float], v300: dict[str, float]) -> str:
    score = float(row["score"])
    kv = float(row["kv"])
    online = float(row["online"])
    full_score = full.get("score", 0.0)
    full_online = full.get("online", 0.0)
    v300_score = v300.get("score", 0.0)
    full_ratio = score / full_score if full_score > 0 else 0.0
    v300_ratio = score / v300_score if v300_score > 0 else 0.0
    speed = full_online / online if online > 0 else 0.0
    if full_ratio >= 0.95 and 0.10 <= kv <= 0.30 and speed >= 2.5:
        return "meets target"
    if v300_ratio >= 0.98 and kv < v300.get("kv", 1.0):
        return "promising vs v300"
    if full_ratio < 0.90:
        return "quality drop"
    if kv > 0.30:
        return "KV high"
    if speed < 2.5:
        return "speed weak"
    return "watch"


def write_dashboard() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v300 = read_summary_by_task(V300_DIR / "summary.csv")
    full = read_summary_by_task(FULL_DIR / "summary.csv")

    task_rows: list[dict[str, object]] = []
    for path in sorted((ROOT / "outputs").glob("*/task_results.csv")):
        name = path.parent.name
        if not name.startswith(CURRENT_PREFIXES):
            continue
        row = summarize_task_results(path)
        if row is not None:
            task_rows.append(row)

    task_rows.sort(key=lambda row: (str(row["task"]), str(row["label"])))
    payload = {
        "updated": datetime.now().isoformat(timespec="seconds"),
        "task_rows": task_rows,
        "final_rows": final_summary_rows(),
    }
    (OUT_DIR / "live_dashboard.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Section 154: RiskKV live dashboard")
    lines.append("")
    lines.append(f"更新时间：{payload['updated']}")
    lines.append("")
    lines.append("## Baselines")
    lines.append("")
    lines.append("| Method | Samples | Score | KV | Online | Total |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, directory in [("full_kv", FULL_DIR), ("v300_main", V300_DIR)]:
        summary = read_csv(directory / "summary.csv")
        overall = next((row for row in summary if row.get("benchmark") == "ALL" and row.get("task") == "ALL"), None)
        if overall:
            lines.append(
                "| {} | {} | {:.6f} | {:.2f}% | {:.4f}s | {:.4f}s |".format(
                    name,
                    overall.get("samples", ""),
                    fnum(overall, "score"),
                    100 * fnum(overall, "mean_keep_fraction"),
                    fnum(overall, "mean_online_seconds"),
                    fnum(overall, "mean_total_seconds"),
                )
            )
    lines.append("")
    lines.append("## Final Summary Tables")
    lines.append("")
    lines.append("| Report | Method | Samples | Score | KV | Online |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in payload["final_rows"]:
        if row["method"] == "<pending>":
            lines.append(f"| `{row['report']}` | pending |  |  |  |  |")
        else:
            lines.append(
                "| `{}` | {} | {} | {:.6f} | {:.2f}% | {:.4f}s |".format(
                    row["report"],
                    row.get("method", ""),
                    row.get("samples", ""),
                    float(row.get("score", 0.0) or 0.0),
                    100 * float(row.get("kv_keep", 0.0) or 0.0),
                    float(row.get("online_seconds", 0.0) or 0.0),
                )
            )
    lines.append("")
    lines.append("## Partial Task Results")
    lines.append("")
    lines.append("| Task | Run | N | Score | vs full | vs v300 | KV | Online | full online speed | Note |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in task_rows:
        task = str(row["task"])
        full_task = full.get(task, {})
        v300_task = v300.get(task, {})
        full_score = full_task.get("score", 0.0)
        v300_score = v300_task.get("score", 0.0)
        full_online = full_task.get("online", 0.0)
        score = float(row["score"])
        online = float(row["online"])
        note = status_note(row, full_task, v300_task)
        lines.append(
            "| {} | `{}` | {} | {:.6f} | {:.1f}% | {:.1f}% | {:.2f}% | {:.4f}s | {:.2f}x | {} |".format(
                task,
                row["label"],
                row["samples"],
                score,
                100 * score / full_score if full_score > 0 else 0.0,
                100 * score / v300_score if v300_score > 0 else 0.0,
                100 * float(row["kv"]),
                online,
                full_online / online if online > 0 else 0.0,
                note,
            )
        )
    lines.append("")
    lines.append("## Current Interpretation")
    lines.append("")
    lines.append("- `meets target` 表示该单任务点同时满足：score >= full 的 95%，KV 在 10%-30%，online speed >= full 的 2.5x。")
    lines.append("- `promising vs v300` 表示相对当前主线分数基本持平且 KV 更低，适合进入组合/验证。")
    lines.append("- b16 / bounded fallback 若显示 `quality drop`，不要直接纳入主线；优先看 v309/v310 的 span repack 是否改善。")
    lines.append("- v311 是否成为当前 practical best，取决于 v306 repobench bounded retry 的最终分数和 KV。")
    lines.append("")
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_DIR / "live_dashboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"doc": str(DOC_PATH), "tasks": len(task_rows), "updated": payload["updated"]}, ensure_ascii=False))


if __name__ == "__main__":
    write_dashboard()
