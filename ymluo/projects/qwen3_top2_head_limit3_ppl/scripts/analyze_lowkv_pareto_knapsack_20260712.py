#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_RUNS = {
    "v360": "outputs/riskkv_v19_v360_lowkv_certificate_all_20260712_lowkv_extreme_1to10_m20_bDyn_pDyn/task_results.csv",
    "v363": "outputs/riskkv_v19_v363_taskwise_lowkv_mix_all_20260712_lowkv_extreme_1to10_m20_bDyn_pDyn/task_results.csv",
    "v365": "outputs/riskkv_v19_v365_ultra_skeleton_all_20260712_lowkv_extreme_1to10_m20_bDyn_pDyn/task_results.csv",
    "v368": "outputs/riskkv_v19_v368_direct_operator_extreme_mix_all_20260712_lowkv_extreme_1to10_m20_bDyn_pDyn/task_results.csv",
    "v372": "outputs/riskkv_v19_v372_extractive_qa_direct_all_20260712_lowkv_extreme_1to10_m20_bDyn_pDyn/task_results.csv",
    "v373": "outputs/riskkv_v19_v373_selective_direct_ladder_all_20260712_lowkv_extreme_1to10_m20_bDyn_pDyn/task_results.csv",
}


def fnum(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["task"], []).append(row)
    result = {}
    for task, task_rows in grouped.items():
        result[task] = {
            "n": float(len(task_rows)),
            "score": sum(fnum(row, "score") for row in task_rows) / len(task_rows),
            "kv": sum(fnum(row, "keep_fraction") for row in task_rows) / len(task_rows),
            "online": sum(fnum(row, "online_seconds") for row in task_rows) / len(task_rows),
        }
    return result


def non_dominated(candidates: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    kept = []
    for cand in candidates:
        dominated = False
        for other in candidates:
            if other is cand:
                continue
            better_score = float(other["score"]) >= float(cand["score"]) - 1e-12
            lower_kv = float(other["kv"]) <= float(cand["kv"]) + 1e-12
            strictly = float(other["score"]) > float(cand["score"]) + 1e-12 or float(other["kv"]) < float(cand["kv"]) - 1e-12
            if better_score and lower_kv and strictly:
                dominated = True
                break
        if not dominated:
            kept.append(cand)
    return sorted(kept, key=lambda item: (float(item["kv"]), -float(item["score"]), str(item["run"])))


def solve_knapsack(
    per_task: dict[str, list[dict[str, float | str]]],
    kv_limit: float,
    kv_scale: int,
) -> dict[str, object]:
    tasks = sorted(per_task)
    total_limit = int(kv_limit * len(tasks) * kv_scale + 1e-9)
    dp: dict[int, tuple[float, float, list[dict[str, float | str]]]] = {0: (0.0, 0.0, [])}
    for task in tasks:
        next_dp: dict[int, tuple[float, float, list[dict[str, float | str]]]] = {}
        for used, (score_sum, online_sum, choices) in dp.items():
            for cand in per_task[task]:
                kv_units = int(round(float(cand["kv"]) * kv_scale))
                new_used = used + kv_units
                if new_used > total_limit:
                    continue
                new_score = score_sum + float(cand["score"])
                new_online = online_sum + float(cand["online"])
                choice = {**cand, "task": task}
                old = next_dp.get(new_used)
                if old is None or new_score > old[0] or (abs(new_score - old[0]) < 1e-12 and new_online < old[1]):
                    next_dp[new_used] = (new_score, new_online, choices + [choice])
        best_seen = -1.0
        pruned = {}
        for used in sorted(next_dp):
            if next_dp[used][0] > best_seen + 1e-9:
                pruned[used] = next_dp[used]
                best_seen = next_dp[used][0]
        dp = pruned
    if not dp:
        return {"tasks": len(tasks), "score": 0.0, "kv": 0.0, "online": 0.0, "choices": []}
    used, best = max(dp.items(), key=lambda item: item[1][0])
    return {
        "tasks": len(tasks),
        "score": best[0] / len(tasks),
        "kv": used / (kv_scale * len(tasks)),
        "online": best[1] / len(tasks),
        "choices": best[2],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl")
    parser.add_argument("--kv-limit", type=float, default=0.10)
    parser.add_argument("--kv-scale", type=int, default=10000)
    parser.add_argument("--out-json", default="outputs/riskkv_lowkv_pareto_knapsack_20260712.json")
    parser.add_argument("--out-md", default="doc/section174_lowkv_pareto_knapsack_20260712.md")
    args = parser.parse_args()

    root = Path(args.root)
    runs = {}
    for name, relpath in DEFAULT_RUNS.items():
        path = root / relpath
        if path.exists():
            runs[name] = aggregate(read_csv(path))

    common_tasks = sorted(set.intersection(*[set(per_task) for per_task in runs.values()])) if runs else []
    per_task: dict[str, list[dict[str, float | str]]] = {}
    for task in common_tasks:
        candidates = []
        for run, task_rows in runs.items():
            row = task_rows[task]
            candidates.append(
                {
                    "run": run,
                    "score": row["score"],
                    "kv": row["kv"],
                    "online": row["online"],
                }
            )
        per_task[task] = non_dominated(candidates)

    frontier = []
    for limit in [0.03, 0.05, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15]:
        frontier.append({"kv_limit": limit, **solve_knapsack(per_task, limit, args.kv_scale)})
    selected = solve_knapsack(per_task, args.kv_limit, args.kv_scale)
    payload = {
        "runs": sorted(runs),
        "kv_limit": args.kv_limit,
        "selected": selected,
        "frontier": frontier,
        "per_task_candidates": per_task,
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# section174：Low-KV Pareto/knapsack 自动分析",
        "",
        "这个脚本把已完成的任务级 policy 结果转成非支配候选，再在全局平均 KV 约束下做 knapsack 选择。",
        "",
        "## 全局 frontier",
        "",
        "| avg KV limit | score | KV keep | online |",
        "|---:|---:|---:|---:|",
    ]
    for item in frontier:
        lines.append("| {kv_limit:.0%} | {score:.4f} | {kv:.2%} | {online:.4f}s |".format(**item))
    lines += [
        "",
        f"## 选中组合：avg KV <= {args.kv_limit:.0%}",
        "",
        "| task | selected run | score | KV keep | online |",
        "|---|---|---:|---:|---:|",
    ]
    for choice in selected["choices"]:
        lines.append(
            "| {task} | {run} | {score:.4f} | {kv:.2%} | {online:.4f}s |".format(**choice)
        )
    lines += ["", "## 非支配候选", ""]
    for task, candidates in per_task.items():
        lines += [f"### {task}", "", "| run | score | KV keep | online |", "|---|---:|---:|---:|"]
        for cand in candidates:
            lines.append("| {run} | {score:.4f} | {kv:.2%} | {online:.4f}s |".format(**cand))
        lines.append("")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(out_json)
    print(out_md)
    print(json.dumps({"selected": selected, "frontier": frontier}, ensure_ascii=False))


if __name__ == "__main__":
    main()
