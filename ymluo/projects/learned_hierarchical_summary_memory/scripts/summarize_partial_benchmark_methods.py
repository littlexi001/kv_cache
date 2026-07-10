from __future__ import annotations

import argparse
import collections
import csv
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
        return 0.0


def summarize_trials(rows: list[dict[str, str]]) -> list[tuple[str, int, float, float, float]]:
    by_method: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    out = []
    full_sec = None
    for method, items in sorted(by_method.items()):
        score = sum(fnum(row, "score") for row in items) / len(items)
        token = sum(fnum(row, "token_ratio_vs_full_raw") for row in items) / len(items)
        sec = sum(fnum(row, "seconds") for row in items) / len(items)
        if method == "full_raw":
            full_sec = sec
        out.append((method, len(items), score, token, sec))
    return [
        (method, n, score, token, (full_sec / sec if full_sec and sec > 0 else 1.0))
        for method, n, score, token, sec in out
    ]


def summarize_final(rows: list[dict[str, str]]) -> list[tuple[str, int, float, float, float]]:
    out = []
    for row in rows:
        if row.get("benchmark") == "__overall__" and row.get("task") == "__overall__":
            out.append(
                (
                    row["method"],
                    int(float(row["samples"])),
                    fnum(row, "avg_score"),
                    fnum(row, "token_ratio_vs_full_raw"),
                    fnum(row, "speedup_vs_full_raw"),
                )
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs"))
    parser.add_argument("--names", required=True, help="Comma-separated output directory names.")
    args = parser.parse_args()

    for name in [item.strip() for item in args.names.split(",") if item.strip()]:
        directory = args.base / name
        final_rows = read_rows(directory / "summary.csv")
        if final_rows:
            rows = summarize_final(final_rows)
            source = "summary.csv"
        else:
            rows = summarize_trials(read_rows(directory / "trials.partial.csv"))
            source = "trials.partial.csv"
        print(f"\n{name} source={source}")
        for method, n, score, token, speed in rows:
            print(f"{method:34s} n={n:3d} score={score:.4f} token={token:.4f} speed={speed:.3f}")


if __name__ == "__main__":
    main()
