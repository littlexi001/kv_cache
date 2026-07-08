from __future__ import annotations

import argparse
import collections
import csv
import math
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def mean(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else math.nan


def filter_complete_cases(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    methods = sorted({row["method"] for row in rows})
    by_case: dict[tuple[str, str, str], list[dict[str, str]]] = collections.defaultdict(list)
    for row in rows:
        by_case[(row["benchmark"], row["task"], row["case_id"])].append(row)

    out: list[dict[str, str]] = []
    expected = set(methods)
    for items in by_case.values():
        got = {row["method"] for row in items}
        if expected.issubset(got):
            out.extend(items)
    return out


def summarize_trials(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = collections.defaultdict(list)
    for row in rows:
        grouped[(row["benchmark"], row["task"], row["method"])].append(row)
        grouped[("__overall__", "__overall__", row["method"])].append(row)

    full_tokens: dict[tuple[str, str], float] = {}
    full_seconds: dict[tuple[str, str], float] = {}
    for (bench, task, method), items in grouped.items():
        if method == "full_raw":
            full_tokens[(bench, task)] = mean([fnum(row, "prompt_tokens") for row in items])
            full_seconds[(bench, task)] = mean([fnum(row, "seconds") for row in items])

    out: list[dict[str, str]] = []
    for (bench, task, method), items in sorted(grouped.items()):
        key = (bench, task)
        score = mean([fnum(row, "score") for row in items])
        tokens = mean([fnum(row, "prompt_tokens") for row in items])
        seconds = mean([fnum(row, "seconds") for row in items])
        ft = full_tokens.get(key, tokens)
        fs = full_seconds.get(key, seconds)
        out.append(
            {
                "benchmark": bench,
                "task": task,
                "method": method,
                "samples": str(len(items)),
                "avg_score": f"{score:.12g}",
                "token_ratio_vs_full_raw": f"{(tokens / ft if ft else math.nan):.12g}",
                "speedup_vs_full_raw": f"{(fs / seconds if seconds else math.nan):.12g}",
            }
        )
    return out


def select_floor(rows: list[dict[str, str]], tol: float) -> tuple[list[dict[str, float | str]], dict[str, float | str] | None]:
    per_task = [
        row
        for row in rows
        if row.get("benchmark") != "__overall__" and row.get("task") != "__overall__"
    ]
    overall = {
        row["method"]: row
        for row in rows
        if row.get("benchmark") == "__overall__" and row.get("task") == "__overall__"
    }
    full_by_task = {
        (row["benchmark"], row["task"]): fnum(row, "avg_score")
        for row in per_task
        if row["method"] == "full_raw"
    }
    methods = sorted({row["method"] for row in rows if row["method"] != "full_raw"})
    reports: list[dict[str, float | str]] = []
    for method in methods:
        gaps: list[float] = []
        min_score = math.inf
        worst_task = ""
        for row in per_task:
            if row["method"] != method:
                continue
            key = (row["benchmark"], row["task"])
            full_score = full_by_task.get(key, math.nan)
            score = fnum(row, "avg_score")
            gap = score - full_score
            gaps.append(gap)
            if score < min_score:
                min_score = score
                worst_task = f"{row['benchmark']}/{row['task']}"
        safe = bool(gaps) and min(gaps) >= -tol
        o = overall.get(method, {})
        reports.append(
            {
                "method": method,
                "samples": int(float(o.get("samples", "0") or 0)),
                "overall_score": fnum(o, "avg_score"),
                "token": fnum(o, "token_ratio_vs_full_raw"),
                "speed": fnum(o, "speedup_vs_full_raw"),
                "min_gap_vs_full": min(gaps) if gaps else math.nan,
                "worst_task": worst_task,
                "safe": "yes" if safe else "no",
            }
        )
    safe_reports = [row for row in reports if row["safe"] == "yes"]
    selected = min(safe_reports, key=lambda row: float(row["token"])) if safe_reports else None
    return reports, selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs"))
    parser.add_argument("--names", required=True, help="Comma-separated output directory names.")
    parser.add_argument("--tol", type=float, default=0.0)
    parser.add_argument("--allow_incomplete_partial", action="store_true")
    args = parser.parse_args()

    for name in [item.strip() for item in args.names.split(",") if item.strip()]:
        directory = args.base / name
        final_rows = read_rows(directory / "summary.csv")
        source = "summary.csv"
        rows = final_rows
        if not rows:
            trial_rows = read_rows(directory / "trials.partial.csv")
            if not args.allow_incomplete_partial:
                trial_rows = filter_complete_cases(trial_rows)
            rows = summarize_trials(trial_rows)
            source = "trials.partial.csv complete-cases"

        reports, selected = select_floor(rows, args.tol)
        print(f"\n{name} source={source} tol={args.tol}")
        for row in sorted(reports, key=lambda r: (str(r["safe"]) != "yes", float(r["token"]) if not math.isnan(float(r["token"])) else 999.0)):
            print(
                f"{row['method']:34s} safe={row['safe']:3s} "
                f"score={float(row['overall_score']):.4f} token={float(row['token']):.4f} "
                f"speed={float(row['speed']):.3f} min_gap={float(row['min_gap_vs_full']):+.4f} "
                f"worst={row['worst_task']} n={row['samples']}"
            )
        if selected:
            print(
                f"SELECT {selected['method']} token={float(selected['token']):.4f} "
                f"score={float(selected['overall_score']):.4f}"
            )
        else:
            print("SELECT none")


if __name__ == "__main__":
    main()
