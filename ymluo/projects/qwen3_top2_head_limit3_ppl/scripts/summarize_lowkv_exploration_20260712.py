#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl")
OUT = ROOT / "outputs"
DOC = ROOT / "doc"

FULL_BASELINE_SCORE = 0.3658
FULL_BASELINE_ONLINE = 3.0988
PRACTICAL_BASELINE_SCORE = 0.43923514197399705
PRACTICAL_BASELINE_KV = 0.2741166578171013
PRACTICAL_BASELINE_ONLINE = 0.5632476073285215


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fnum(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def summarize_file(path: Path) -> dict[str, object] | None:
    try:
        rows = read_csv(path)
    except Exception:
        return None
    if not rows:
        return None
    score = sum(fnum(row, "score") for row in rows) / len(rows)
    kv = sum(fnum(row, "keep_fraction") for row in rows) / len(rows)
    online = sum(fnum(row, "online_seconds") for row in rows) / len(rows)
    tasks = sorted({row.get("task", "") for row in rows if row.get("task", "")})
    per_task = []
    for task in tasks:
        subset = [row for row in rows if row.get("task") == task]
        per_task.append(
            {
                "task": task,
                "n": len(subset),
                "score": sum(fnum(row, "score") for row in subset) / len(subset),
                "kv": sum(fnum(row, "keep_fraction") for row in subset) / len(subset),
                "online": sum(fnum(row, "online_seconds") for row in subset) / len(subset),
            }
        )
    return {
        "dir": str(path.parent.relative_to(ROOT)),
        "samples": len(rows),
        "tasks": len(tasks),
        "score": score,
        "kv": kv,
        "online": online,
        "score_vs_full": score / FULL_BASELINE_SCORE,
        "speed_vs_full": FULL_BASELINE_ONLINE / max(1e-9, online),
        "score_vs_practical": score / PRACTICAL_BASELINE_SCORE,
        "kv_vs_practical": kv / PRACTICAL_BASELINE_KV,
        "speed_vs_practical": PRACTICAL_BASELINE_ONLINE / max(1e-9, online),
        "hit_full_target": (
            0.01 <= kv <= 0.10
            and score / FULL_BASELINE_SCORE >= 0.95
            and FULL_BASELINE_ONLINE / max(1e-9, online) >= 2.50
        ),
        "hit_practical_target": (
            0.01 <= kv <= 0.10
            and score / PRACTICAL_BASELINE_SCORE >= 0.95
            and PRACTICAL_BASELINE_ONLINE / max(1e-9, online) >= 2.50
        ),
        "per_task": per_task,
    }


def collect_rows() -> list[dict[str, object]]:
    rows = []
    for pattern in [
        "riskkv_v19_v360_*20260712*",
        "riskkv_v19_v361_*20260712*",
        "riskkv_v19_v362_*20260712*",
        "riskkv_v19_v363_*20260712*",
        "riskkv_v19_v364_*20260712*",
        "riskkv_v19_v365_*20260712*",
        "riskkv_v19_v366_*20260712*",
        "riskkv_v19_v367_*20260712*",
        "riskkv_v19_v368_*20260712*",
        "riskkv_v19_v369_*20260712*",
        "riskkv_v19_v370_*20260712*",
        "riskkv_v19_v371_*20260712*",
        "riskkv_v19_v372_*20260712*",
        "riskkv_v19_v373_*20260712*",
        "riskkv_v19_v374_*20260712*",
        "riskkv_v19_v375_*20260712*",
        "riskkv_v19_v376_*20260712*",
        "riskkv_v19_v377_*20260712*",
        "riskkv_v19_v378_*20260712*",
        "riskkv_v19_v379_*20260712*",
        "riskkv_v19_v380_*20260712*",
        "riskkv_v19_v381_*20260712*",
        "riskkv_v19_v382_*20260712*",
        "riskkv_v19_v383_*20260712*",
        "riskkv_v19_v384_*20260712*",
        "riskkv_v19_v385_*20260712*",
        "riskkv_v19_v386_*20260712*",
        "riskkv_v19_v387_*20260712*",
        "riskkv_v19_v388_*20260712*",
        "riskkv_v19_v389_*20260712*",
        "riskkv_v19_v390_*20260712*",
        "riskkv_v19_v391_*20260712*",
        "riskkv_v19_v392_*20260712*",
    ]:
        for directory in OUT.glob(pattern):
            path = directory / "task_results.csv"
            if path.exists():
                summary = summarize_file(path)
                if summary:
                    rows.append(summary)
    rows.sort(
        key=lambda item: (
            not bool(item["hit_full_target"]),
            -float(item["score_vs_full"]),
            float(item["kv"]),
        )
    )
    return rows


def write_outputs(rows: list[dict[str, object]]) -> tuple[Path, Path]:
    DOC.mkdir(parents=True, exist_ok=True)
    json_path = OUT / "riskkv_lowkv_exploration_summary_20260712.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# section170: Low-KV overnight exploration 自动汇总",
        "",
        "日期：2026-07-12",
        "",
        "目标：寻找 1%-10% KV keep、速度 2.5x+、分数达到 baseline 95%+ 的可用现象。",
        "",
        f"近似 full KV baseline：score={FULL_BASELINE_SCORE:.4f}, online={FULL_BASELINE_ONLINE:.4f}s。",
        f"当前 practical baseline(v300)：score={PRACTICAL_BASELINE_SCORE:.4f}, KV keep={PRACTICAL_BASELINE_KV:.2%}, online={PRACTICAL_BASELINE_ONLINE:.4f}s。",
        "",
        "## 全局结果",
        "",
        "| run | samples | tasks | score | vs full | vs v300 | KV keep | speed/full | speed/v300 | hit full target |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {dir} | {samples} | {tasks} | {score:.4f} | {score_vs_full:.2%} | {score_vs_practical:.2%} | {kv:.2%} | {speed_vs_full:.2f}x | {speed_vs_practical:.2f}x | {hit_full_target} |".format(
                **row
            )
        )

    low_kv_rows = [row for row in rows if 0.01 <= float(row["kv"]) <= 0.10]
    lines += [
        "",
        "## 低 KV 候选",
        "",
        "| run | score | vs full | vs v300 | KV keep | speed/full | speed/v300 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(low_kv_rows, key=lambda item: -float(item["score_vs_full"]))[:12]:
        lines.append(
            "| {dir} | {score:.4f} | {score_vs_full:.2%} | {score_vs_practical:.2%} | {kv:.2%} | {speed_vs_full:.2f}x | {speed_vs_practical:.2f}x |".format(
                **row
            )
        )
    if not low_kv_rows:
        lines.append("| 暂无完成的 1%-10% KV 候选 |  |  |  |  |  |  |")

    lines += ["", "## 任务级结果", ""]
    for row in rows:
        lines += [
            f"### {row['dir']}",
            "",
            "| task | n | score | KV keep | online |",
            "|---|---:|---:|---:|---:|",
        ]
        for task_row in row["per_task"]:
            lines.append("| {task} | {n} | {score:.4f} | {kv:.2%} | {online:.4f} |".format(**task_row))
        lines.append("")

    md_path = DOC / "section170_lowkv_overnight_exploration_20260712.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> None:
    rows = collect_rows()
    json_path, md_path = write_outputs(rows)
    print(json_path)
    print(md_path)
    for row in rows[:20]:
        print(
            json.dumps(
                {
                    key: row[key]
                    for key in [
                        "dir",
                        "samples",
                        "tasks",
                        "score",
                        "score_vs_full",
                        "score_vs_practical",
                        "kv",
                        "speed_vs_full",
                        "speed_vs_practical",
                        "hit_full_target",
                    ]
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
