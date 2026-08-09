from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_root", required=True, type=Path)
    parser.add_argument("--m20_root", required=True, type=Path)
    parser.add_argument("--crossover_root", required=True, type=Path)
    parser.add_argument("--auto_root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate(
    rows: list[dict[str, str]],
    expected_counts: dict[str, int],
    expected_tasks: int,
) -> None:
    counts = Counter(row["method"] for row in rows)
    if counts != Counter(expected_counts):
        raise AssertionError((counts, expected_counts))
    if len({row["task"] for row in rows}) != expected_tasks:
        raise AssertionError("unexpected task count")
    expected_methods = set(expected_counts)
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        grouped[(row["task"], row["sample_id"])].add(row["method"])
    if not grouped or any(methods != expected_methods for methods in grouped.values()):
        raise AssertionError("method/sample pairing is incomplete")


def macro_score(rows: list[dict[str, str]]) -> float:
    by_task: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(float(row["score"]))
    return statistics.mean(statistics.mean(values) for values in by_task.values())


def paired_speed(
    rows: list[dict[str, str]], method: str, key: str
) -> float:
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["task"], row["sample_id"])][row["method"]] = row
    pairs = [
        methods
        for methods in grouped.values()
        if "full_kv" in methods and method in methods
    ]
    denominator = sum(float(pair[method][key]) for pair in pairs)
    return (
        sum(float(pair["full_kv"][key]) for pair in pairs) / denominator
        if denominator > 0.0
        else 0.0
    )


def summarize_methods(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    full_macro = macro_score([row for row in rows if row["method"] == "full_kv"])
    output = []
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        score = macro_score(subset)
        output.append(
            {
                "method": method,
                "samples": len(subset),
                "macro_score": score,
                "quality_retention": score / full_macro if full_macro else None,
                "query_speedup": 1.0 if method == "full_kv" else paired_speed(rows, method, "query_seconds"),
                "decode_speedup": 1.0 if method == "full_kv" else paired_speed(rows, method, "decode_seconds"),
                "online_speedup": 1.0 if method == "full_kv" else paired_speed(rows, method, "online_seconds"),
                "total_speedup": 1.0 if method == "full_kv" else paired_speed(rows, method, "total_seconds"),
            }
        )
    return output


def per_task(rows: list[dict[str, str]], method: str) -> list[dict[str, Any]]:
    output = []
    for task in sorted({row["task"] for row in rows}):
        full = [row for row in rows if row["task"] == task and row["method"] == "full_kv"]
        candidate = [row for row in rows if row["task"] == task and row["method"] == method]
        full_score = statistics.mean(float(row["score"]) for row in full)
        score = statistics.mean(float(row["score"]) for row in candidate)
        output.append(
            {
                "task": task,
                "samples": len(candidate),
                "full_score": full_score,
                "score": score,
                "quality_retention": score / full_score if full_score else None,
                "online_speedup": (
                    sum(float(row["online_seconds"]) for row in full)
                    / sum(float(row["online_seconds"]) for row in candidate)
                ),
            }
        )
    return output


def fmt(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def method_table(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| 方法 | 样本 | Macro | 质量保持率 | Query | Decode | Online | Total |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {samples} | {score:.4f} | {ret} | {query}x | {decode}x | {online}x | {total}x |".format(
                method=row["method"],
                samples=row["samples"],
                score=row["macro_score"],
                ret=fmt(row["quality_retention"]),
                query=fmt(row["query_speedup"]),
                decode=fmt(row["decode_speedup"]),
                online=fmt(row["online_speedup"]),
                total=fmt(row["total_speedup"]),
            )
        )
    return lines


def main() -> None:
    args = parse_args()
    main_rows = read_rows(args.main_root / "merged" / "sample_results.csv")
    m20_rows = read_rows(args.m20_root / "merged" / "sample_results.csv")
    auto_rows = read_rows(args.auto_root / "merged" / "sample_results.csv")
    validate(main_rows, {"full_kv": 3750, "countcap": 3750}, 16)
    validate(
        m20_rows,
        {
            "full_kv": 320,
            "countcap_fullprompt": 320,
            "countcap_fullprompt_keypca": 320,
        },
        16,
    )
    validate(auto_rows, {"full_kv": 320, "countcap_auto": 320}, 16)
    crossover = json.loads(
        (args.crossover_root / "crossover_analysis.json").read_text(encoding="utf-8")
    )
    auto_paths = Counter(
        row.get("executed_path", "")
        for row in auto_rows
        if row["method"] == "countcap_auto"
    )

    payload = {
        "main": summarize_methods(main_rows),
        "main_countcap_per_task": per_task(main_rows, "countcap"),
        "m20": summarize_methods(m20_rows),
        "crossover": crossover,
        "auto": summarize_methods(auto_rows),
        "auto_executed_paths": dict(auto_paths),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output.with_suffix(".json")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = ["# CountCap 短序列优化最终结果", ""]
    lines.extend(method_table("完整 LongBench：旧 CountCap", payload["main"]))
    lines.extend(["", "### 旧 CountCap 分任务", "", "| 任务 | 样本 | Full | CountCap | 保持率 | Online |", "|---|---:|---:|---:|---:|---:|"])
    for row in payload["main_countcap_per_task"]:
        lines.append(
            f"| {row['task']} | {row['samples']} | {row['full_score']:.4f} | {row['score']:.4f} | {fmt(row['quality_retention'])} | {fmt(row['online_speedup'])}x |"
        )
    lines.extend([""] + method_table("Dense suffix 与 Key-PCA 消融", payload["m20"]))
    lines.extend([""] + method_table("解析式门控独立验证", payload["auto"]))
    lines.extend(
        [
            "",
            f"门控实际路径：`{json.dumps(dict(auto_paths), ensure_ascii=False)}`。",
            "",
            "## 长度交叉点",
            "",
            f"实测插值得到的交叉点：`{crossover.get('estimated_speed_crossover_tokens')}` tokens。",
            "",
            "完整的各长度成本模型、质量下界和 break-even 生成步数见同目录 JSON。",
        ]
    )
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
