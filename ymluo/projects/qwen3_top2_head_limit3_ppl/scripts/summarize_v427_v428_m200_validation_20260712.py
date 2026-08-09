from __future__ import annotations

import csv
from pathlib import Path


M100_FULL_SCORE = 0.36581658127460975
M100_FULL_ONLINE = 3.0988


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate(path: Path) -> dict[str, float] | None:
    rows = read_rows(path)
    if not rows:
        return None
    return {
        "n": float(len(rows)),
        "score": sum(float(row.get("score") or 0.0) for row in rows) / len(rows),
        "kv": sum(float(row.get("keep_fraction") or 0.0) for row in rows) / len(rows),
        "online": sum(float(row.get("online_seconds") or 0.0) for row in rows) / len(rows),
    }


def main() -> None:
    full_path = Path("outputs/riskkv_full_kv_longbench_m200_20260712/task_results.csv")
    full = aggregate(full_path)
    if full is None:
        full_score = M100_FULL_SCORE
        full_online = M100_FULL_ONLINE
        full_note = "M100 full baseline fallback"
    else:
        full_score = full["score"]
        full_online = full["online"]
        full_note = "M200 full_kv"
        print(
            f"full_kv M200 DONE n={int(full['n'])} "
            f"score={full_score:.4f} online={full_online:.4f}s"
        )

    items = {
        "v427_source_frontier": Path(
            "outputs/riskkv_v19_v427_v417_source_v421_winners_20260712_v427_m100_bDyn_pDyn/task_results.csv"
        ),
        "v428_plus_repobench": Path(
            "outputs/riskkv_v19_v428_v427_plus_repobench_20260712_v428_m100_bDyn_pDyn/task_results.csv"
        ),
        "v427_source_frontier_m200": Path(
            "outputs/riskkv_v19_v427_v417_source_v421_winners_20260712_v427_m200_validate_m200_bDyn_pDyn/task_results.csv"
        ),
        "v428_plus_repobench_m200": Path(
            "outputs/riskkv_v19_v428_v427_plus_repobench_20260712_v428_m200_validate_m200_bDyn_pDyn/task_results.csv"
        ),
        "v429_source_best_m100": Path(
            "outputs/riskkv_v19_v429_source_best_frontiers_20260712_v429_m100_m100_bDyn_pDyn/task_results.csv"
        ),
        "v430_composer_kv06_speed6_m100": Path(
            "outputs/riskkv_v19_v430_composer_kv06_speed6_task20_20260712_v430_m100_m100_bDyn_pDyn/task_results.csv"
        ),
        "v431_composer_kv08_speed5_m100": Path(
            "outputs/riskkv_v19_v431_composer_kv08_speed5_task25_20260712_v431_m100_m100_bDyn_pDyn/task_results.csv"
        ),
    }
    print(f"baseline_for_vs_full={full_note} score={full_score:.4f} online={full_online:.4f}s")
    for name, path in items.items():
        agg = aggregate(path)
        if agg is None:
            print(f"{name:28s} RUN/MISS path={path}")
            continue
        speed = full_online / max(agg["online"], 1e-9)
        print(
            f"{name:28s} DONE n={int(agg['n']):4d} "
            f"score={agg['score']:.4f} vs_full={agg['score']/full_score:.2%} "
            f"kv={agg['kv']:.2%} online={agg['online']:.4f}s speed={speed:.2f}x"
        )


if __name__ == "__main__":
    main()
