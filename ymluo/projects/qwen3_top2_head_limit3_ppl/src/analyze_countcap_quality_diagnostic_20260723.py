from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


METHODS = (
    "full_kv",
    "exact_top2_fullprompt",
    "exact_massadaptive_fullprompt",
    "countcap_fullprompt",
    "countcap_massadaptive_fullprompt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose fixed-budget, retrieval, and adaptive-budget quality gaps."
    )
    parser.add_argument("--run_root", required=True, type=Path)
    return parser.parse_args()


def read_rows(run_root: Path) -> list[dict[str, str]]:
    merged = run_root / "merged" / "sample_results.csv"
    paths = [merged] if merged.exists() else sorted(
        run_root.glob("shard[0-9]*/sample_results.csv")
    )
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no diagnostic rows under {run_root}")
    return rows


def macro_score(rows: list[dict[str, str]], method: str) -> float:
    by_task: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] == method:
            by_task[row["task"]].append(float(row["score"]))
    if not by_task:
        raise ValueError(f"missing method: {method}")
    return sum(sum(values) / len(values) for values in by_task.values()) / len(by_task)


def mean(rows: list[dict[str, str]], method: str, key: str) -> float:
    values = [float(row[key]) for row in rows if row["method"] == method]
    return sum(values) / len(values) if values else 0.0


def analyze(rows: list[dict[str, str]]) -> dict[str, Any]:
    expected = set(METHODS)
    counts = Counter(row["method"] for row in rows)
    pairs: dict[tuple[str, str], set[str]] = defaultdict(set)
    keyed: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["task"], row["sample_id"])
        pairs[key].add(row["method"])
        keyed[(*key, row["method"])] = row
    if set(counts) != expected:
        raise ValueError(f"method mismatch: {counts}")
    if not all(methods == expected for methods in pairs.values()):
        raise ValueError("diagnostic samples are not strictly paired")

    full_macro = macro_score(rows, "full_kv")
    summaries: dict[str, dict[str, float]] = {}
    for method in METHODS:
        score = macro_score(rows, method)
        full_seconds = 0.0
        method_seconds = 0.0
        for task, sample_id in pairs:
            full_seconds += float(keyed[(task, sample_id, "full_kv")]["online_seconds"])
            method_seconds += float(keyed[(task, sample_id, method)]["online_seconds"])
        summaries[method] = {
            "macro_score": score,
            "quality_retention": score / full_macro if full_macro else 0.0,
            "paired_online_speedup": (
                full_seconds / method_seconds if method_seconds else 0.0
            ),
            "mean_configured_attention_fraction": mean(
                rows, method, "configured_attention_fraction"
            ),
            "mean_measured_attention_link_ratio": mean(
                rows, method, "attention_link_ratio"
            ),
        }

    per_task: list[dict[str, Any]] = []
    for task in sorted({row["task"] for row in rows}):
        task_rows = [row for row in rows if row["task"] == task]
        task_full = sum(
            float(row["score"]) for row in task_rows if row["method"] == "full_kv"
        ) / sum(row["method"] == "full_kv" for row in task_rows)
        entry: dict[str, Any] = {"task": task, "full_kv": task_full}
        for method in METHODS[1:]:
            values = [
                float(row["score"]) for row in task_rows if row["method"] == method
            ]
            score = sum(values) / len(values)
            entry[method] = score
            entry[f"{method}_retention"] = score / task_full if task_full else 0.0
        per_task.append(entry)

    exact_top2 = summaries["exact_top2_fullprompt"]["macro_score"]
    approximate_top2 = summaries["countcap_fullprompt"]["macro_score"]
    exact_adaptive = summaries["exact_massadaptive_fullprompt"]["macro_score"]
    approximate_adaptive = summaries["countcap_massadaptive_fullprompt"][
        "macro_score"
    ]
    decomposition = {
        "fixed_top2_budget_gap": full_macro - exact_top2,
        "top2_retrieval_gap": exact_top2 - approximate_top2,
        "exact_adaptive_recovery": exact_adaptive - exact_top2,
        "adaptive_retrieval_gap": exact_adaptive - approximate_adaptive,
    }
    diagnosis = {
        "fixed_top2_budget_is_safe": exact_top2 >= 0.95 * full_macro,
        "approximate_top2_matches_exact": approximate_top2 >= 0.98 * exact_top2,
        "exact_adaptive_meets_quality_floor": exact_adaptive >= 0.95 * full_macro,
        "approximate_adaptive_matches_exact": (
            approximate_adaptive >= 0.98 * exact_adaptive
        ),
    }
    return {
        "samples": len(pairs),
        "tasks": len({task for task, _ in pairs}),
        "method_counts": dict(counts),
        "methods": summaries,
        "gap_decomposition": decomposition,
        "diagnosis": diagnosis,
        "per_task": per_task,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# CountCap 质量损失诊断",
        "",
        f"严格配对样本：{result['samples']}；任务：{result['tasks']}。",
        "",
        "| 方法 | Macro | 质量保持率 | Online speed | 实测 attention links |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = result["methods"][method]
        lines.append(
            f"| {method} | {row['macro_score']:.5f} | "
            f"{100.0 * row['quality_retention']:.2f}% | "
            f"{row['paired_online_speedup']:.3f}x | "
            f"{100.0 * row['mean_measured_attention_link_ratio']:.2f}% |"
        )
    gap = result["gap_decomposition"]
    diagnosis = result["diagnosis"]
    lines.extend(
        [
            "",
            "## 损失分解",
            "",
            f"- 固定 top-2% 预算缺口：{gap['fixed_top2_budget_gap']:+.5f}",
            f"- top-2% 近似检索缺口：{gap['top2_retrieval_gap']:+.5f}",
            f"- 精确动态预算恢复量：{gap['exact_adaptive_recovery']:+.5f}",
            f"- 动态预算近似检索缺口：{gap['adaptive_retrieval_gap']:+.5f}",
            "",
            "## 自动判断",
            "",
            f"- 精确固定 2% 是否达到 95% 质量线：{diagnosis['fixed_top2_budget_is_safe']}",
            f"- PCA-INT4 固定 2% 是否达到精确方法的 98%：{diagnosis['approximate_top2_matches_exact']}",
            f"- 精确动态预算是否达到 95% 质量线：{diagnosis['exact_adaptive_meets_quality_floor']}",
            f"- 近似动态预算是否达到精确动态方法的 98%：{diagnosis['approximate_adaptive_matches_exact']}",
            "",
            "## 分任务",
            "",
            "| Task | Full | Exact 2% | Approx 2% | Exact adaptive | Approx adaptive |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["per_task"]:
        lines.append(
            f"| {row['task']} | {row['full_kv']:.4f} | "
            f"{row['exact_top2_fullprompt']:.4f} | "
            f"{row['countcap_fullprompt']:.4f} | "
            f"{row['exact_massadaptive_fullprompt']:.4f} | "
            f"{row['countcap_massadaptive_fullprompt']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    result = analyze(read_rows(args.run_root))
    (args.run_root / "quality_diagnostic.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = render_markdown(result)
    (args.run_root / "quality_diagnostic_zh.md").write_text(
        report, encoding="utf-8"
    )
    print(report)


if __name__ == "__main__":
    main()
