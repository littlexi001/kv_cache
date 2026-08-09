from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_length(row: dict[str, str]) -> str:
    task = row.get("task", "")
    for part in reversed(task.split("_")):
        if part.isdigit():
            return part
    sample_id = row.get("sample_id", "")
    for part in sample_id.replace("/", "_").split("_"):
        if part.isdigit() and int(part) >= 1024:
            return part
    return "unknown"


def aggregate(rows: list[dict[str, str]]) -> dict[str, float]:
    return {
        "n": float(len(rows)),
        "score": sum(float(row.get("score") or 0.0) for row in rows) / len(rows),
        "kv": sum(float(row.get("keep_fraction") or 0.0) for row in rows) / len(rows),
        "online": sum(float(row.get("online_seconds") or 0.0) for row in rows) / len(rows),
    }


def print_grouped(name: str, rows: list[dict[str, str]], full_by_len: dict[str, dict[str, float]]) -> None:
    if not rows:
        print(f"{name} RUN/MISS")
        return
    overall = aggregate(rows)
    print(
        f"{name:18s} overall n={int(overall['n']):4d} "
        f"score={overall['score']:.4f} kv={overall['kv']:.2%} online={overall['online']:.4f}s"
    )
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[parse_length(row)].append(row)
    for length in sorted(groups, key=lambda x: int(x) if x.isdigit() else 10**9):
        agg = aggregate(groups[length])
        full = full_by_len.get(length)
        suffix = ""
        if full is not None:
            suffix = f" vs_full={agg['score']/max(full['score'], 1e-9):.2%} speed={full['online']/max(agg['online'], 1e-9):.2f}x"
        print(
            f"  len={length:7s} n={int(agg['n']):4d} "
            f"score={agg['score']:.4f} kv={agg['kv']:.2%} online={agg['online']:.4f}s{suffix}"
        )


def main() -> None:
    full_rows = read_rows(Path("outputs/riskkv_full_kv_ruler_m50_20260712/task_results.csv"))
    ours_rows = read_rows(Path("outputs/riskkv_v427_ruler_m50_b384_20260712/task_results.csv"))
    lowkv_rows = read_rows(Path("outputs/riskkv_v436_ruler_lowkv_b224_m50_20260712/task_results.csv"))
    full_by_len: dict[str, dict[str, float]] = {}
    if full_rows:
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in full_rows:
            groups[parse_length(row)].append(row)
        full_by_len = {length: aggregate(rows) for length, rows in groups.items()}
    print_grouped("full_kv_ruler", full_rows, {})
    print_grouped("v427_ruler_b384", ours_rows, full_by_len)
    print_grouped("v436_ruler_b224", lowkv_rows, full_by_len)


if __name__ == "__main__":
    main()
