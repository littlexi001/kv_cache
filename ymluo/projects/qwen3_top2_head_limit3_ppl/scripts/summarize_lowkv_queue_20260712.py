from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path


FULL_M100_SCORE = 0.36581658127460975
FULL_M100_ONLINE = 3.0988


RUNS = {
    "v427_m100": "outputs/riskkv_v19_v427_v417_source_v421_winners_20260712_v427_m100_bDyn_pDyn/task_results.csv",
    "v428_m100": "outputs/riskkv_v19_v428_v427_plus_repobench_20260712_v428_m100_bDyn_pDyn/task_results.csv",
    "v427_m200": "outputs/riskkv_v19_v427_v417_source_v421_winners_20260712_v427_m200_validate_m200_bDyn_pDyn/task_results.csv",
    "v428_m200": "outputs/riskkv_v19_v428_v427_plus_repobench_20260712_v428_m200_validate_m200_bDyn_pDyn/task_results.csv",
    "full_m200": "outputs/riskkv_full_kv_longbench_m200_20260712/task_results.csv",
    "v429_m100": "outputs/riskkv_v19_v429_source_best_frontiers_20260712_v429_m100_m100_bDyn_pDyn/task_results.csv",
    "v430_m100": "outputs/riskkv_v19_v430_composer_kv06_speed6_task20_20260712_v430_m100_m100_bDyn_pDyn/task_results.csv",
    "v431_m100": "outputs/riskkv_v19_v431_composer_kv08_speed5_task25_20260712_v431_m100_m100_bDyn_pDyn/task_results.csv",
    "v433_m100": "outputs/riskkv_v19_v433_dpcomposer_kv06_speed6_task20_20260712_v433_m100_m100_bDyn_pDyn/task_results.csv",
    "v434_m100": "outputs/riskkv_v19_v434_dpcomposer_kv08_speed5_task25_20260712_v434_m100_m100_bDyn_pDyn/task_results.csv",
    "v435_m100": "outputs/riskkv_v19_v435_dpcomposer_kv10_speed35_task35_20260712_v435_m100_m100_bDyn_pDyn/task_results.csv",
    "full_ruler_m50": "outputs/riskkv_full_kv_ruler_m50_20260712/task_results.csv",
    "v427_ruler_m50": "outputs/riskkv_v427_ruler_m50_b384_20260712/task_results.csv",
    "v436_ruler_m50": "outputs/riskkv_v436_ruler_lowkv_b224_m50_20260712/task_results.csv",
}

LOGS = {
    "v427_m200": "outputs/logs/riskkv_v19_v427_v417_source_v421_winners_20260712_v427_m200_validate_m200_bDyn_pDyn.log",
    "v428_m200": "outputs/logs/riskkv_v19_v428_v427_plus_repobench_20260712_v428_m200_validate_m200_bDyn_pDyn.log",
    "full_m200": "outputs/logs/riskkv_full_kv_longbench_m200_20260712.log",
    "v429_m100": "outputs/logs/riskkv_v19_v429_source_best_frontiers_20260712_v429_m100_m100_bDyn_pDyn.log",
    "v430_m100": "outputs/logs/riskkv_v19_v430_composer_kv06_speed6_task20_20260712_v430_m100_m100_bDyn_pDyn.log",
    "v431_m100": "outputs/logs/riskkv_v19_v431_composer_kv08_speed5_task25_20260712_v431_m100_m100_bDyn_pDyn.log",
    "v433_m100": "outputs/logs/riskkv_v19_v433_dpcomposer_kv06_speed6_task20_20260712_v433_m100_m100_bDyn_pDyn.log",
    "v434_m100": "outputs/logs/riskkv_v19_v434_dpcomposer_kv08_speed5_task25_20260712_v434_m100_m100_bDyn_pDyn.log",
    "v435_m100": "outputs/logs/riskkv_v19_v435_dpcomposer_kv10_speed35_task35_20260712_v435_m100_m100_bDyn_pDyn.log",
    "full_ruler_m50": "outputs/logs/riskkv_full_kv_ruler_m50_20260712.log",
    "v427_ruler_m50": "outputs/logs/riskkv_v427_ruler_m50_b384_20260712.log",
    "v436_ruler_m50": "outputs/logs/riskkv_v436_ruler_lowkv_b224_m50_20260712.log",
}

EXPECTED = {
    "v427_m200": 3150,
    "v428_m200": 3150,
    "full_m200": 3150,
    "v429_m100": 1600,
    "v430_m100": 1600,
    "v431_m100": 1600,
    "v433_m100": 1600,
    "v434_m100": 1600,
    "v435_m100": 1600,
    "full_ruler_m50": 1350,
    "v427_ruler_m50": 1350,
    "v436_ruler_m50": 1350,
}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate(rows: list[dict[str, str]]) -> dict[str, float]:
    return {
        "n": float(len(rows)),
        "score": sum(float(row.get("score") or 0.0) for row in rows) / len(rows),
        "kv": sum(float(row.get("keep_fraction") or 0.0) for row in rows) / len(rows),
        "online": sum(float(row.get("online_seconds") or 0.0) for row in rows) / len(rows),
    }


def progress_from_log(path: Path) -> int:
    if not path.exists():
        return 0
    text = path.read_text(errors="ignore")
    return text.count("ours_page_gather: score=") + text.count("full_kv: score=")


def error_from_log(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="ignore")
    tail = text[-8000:]
    for pattern in ["Traceback", "ValueError", "RuntimeError", "CUDA out of memory", "Killed"]:
        if pattern in tail:
            return pattern
    return ""


def parse_length(task: str, sample_id: str) -> str:
    for part in reversed(task.split("_")):
        if part.isdigit():
            return part
    match = re.search(r"(4096|8192|16384|32768)", sample_id)
    return match.group(1) if match else "unknown"


def print_run(name: str, full_score: float, full_online: float) -> None:
    rows = read_rows(Path(RUNS[name]))
    if rows:
        agg = aggregate(rows)
        print(
            f"{name:18s} DONE n={int(agg['n']):4d} "
            f"score={agg['score']:.4f} vs_full={agg['score']/max(full_score, 1e-9):.2%} "
            f"kv={agg['kv']:.2%} online={agg['online']:.4f}s "
            f"speed={full_online/max(agg['online'], 1e-9):.2f}x"
        )
        if "ruler" in name:
            by_len: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in rows:
                by_len[parse_length(row.get("task", ""), row.get("sample_id", ""))].append(row)
            for length in sorted(by_len, key=lambda value: int(value) if value.isdigit() else 10**9):
                sub = aggregate(by_len[length])
                print(
                    f"  len={length:7s} n={int(sub['n']):4d} score={sub['score']:.4f} "
                    f"kv={sub['kv']:.2%} online={sub['online']:.4f}s"
                )
        return
    log = Path(LOGS.get(name, ""))
    progress = progress_from_log(log)
    expected = EXPECTED.get(name, 0)
    err = error_from_log(log)
    status = f"RUN {progress}/{expected}" if expected else f"RUN {progress}"
    if err:
        status += f" ERR={err}"
    print(f"{name:18s} {status}")


def main() -> None:
    full_m200 = read_rows(Path(RUNS["full_m200"]))
    if full_m200:
        full_agg = aggregate(full_m200)
        full_score = full_agg["score"]
        full_online = full_agg["online"]
        print(f"baseline=M200 full_kv score={full_score:.4f} online={full_online:.4f}s")
    else:
        full_score = FULL_M100_SCORE
        full_online = FULL_M100_ONLINE
        print(f"baseline=M100 fallback score={full_score:.4f} online={full_online:.4f}s")
    for name in RUNS:
        if name == "full_m200":
            print_run(name, full_score, full_online)
            continue
        print_run(name, full_score, full_online)


if __name__ == "__main__":
    main()
