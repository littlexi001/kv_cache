from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Candidate:
    name: str
    policy: str
    results: Path


@dataclass(frozen=True)
class TaskStats:
    score: float
    kv: float
    online: float


DEFAULT_CANDIDATES = [
    Candidate(
        "v417",
        "riskkv_task_policy_v417_expanded_knapsack030_20260712.json",
        Path("outputs/riskkv_v19_v417_expanded_knapsack030_20260712_expanded030_v417_m100_bDyn_pDyn/task_results.csv"),
    ),
    Candidate(
        "v413",
        "riskkv_task_policy_v413_expanded_knapsack035_20260712.json",
        Path("outputs/riskkv_v19_v413_expanded_knapsack035_20260712_expanded035_v413_m100_bDyn_pDyn/task_results.csv"),
    ),
    Candidate(
        "v415",
        "riskkv_task_policy_v415_expanded_knapsack045_20260712.json",
        Path("outputs/riskkv_v19_v415_expanded_knapsack045_20260712_expanded045_v415_m100_bDyn_pDyn/task_results.csv"),
    ),
    Candidate(
        "v421",
        "riskkv_task_policy_v421_frontier_router035_20260712.json",
        Path("outputs/riskkv_v19_v421_frontier_router035_20260712_frontier_v421_m100_bDyn_pDyn/task_results.csv"),
    ),
    Candidate(
        "v424",
        "riskkv_task_policy_v424_latency_frontier_router060_20260712.json",
        Path("outputs/riskkv_v19_v424_latency_frontier_router060_20260712_latency_v424_m100_bDyn_pDyn/task_results.csv"),
    ),
    Candidate(
        "v396",
        "riskkv_task_policy_v396_m100_task_knapsack05_exact_20260712.json",
        Path("outputs/riskkv_v19_v396_m100_task_knapsack05_exact_20260712_m100_task_knapsack05_exact_v396_m100_bDyn_pDyn/task_results.csv"),
    ),
    Candidate(
        "v397",
        "riskkv_task_policy_v397_cost_aware_router_after_pareto_20260712.json",
        Path("outputs/riskkv_v19_v397_cost_aware_router_after_pareto_20260712_after_pareto_v397_m100_bDyn_pDyn/task_results.csv"),
    ),
    Candidate(
        "v428",
        "riskkv_task_policy_v428_v427_plus_repobench_20260712.json",
        Path("outputs/riskkv_v19_v428_v427_plus_repobench_20260712_v428_m100_bDyn_pDyn/task_results.csv"),
    ),
]


def read_stats(path: Path) -> dict[str, TaskStats]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["task"], []).append(row)
    out: dict[str, TaskStats] = {}
    for task, subset in grouped.items():
        out[task] = TaskStats(
            score=sum(float(row.get("score") or 0.0) for row in subset) / len(subset),
            kv=sum(float(row.get("keep_fraction") or 0.0) for row in subset) / len(subset),
            online=sum(float(row.get("online_seconds") or 0.0) for row in subset) / len(subset),
        )
    return out


def select_with_lagrange(
    stats: dict[str, dict[str, TaskStats]],
    tasks: list[str],
    base: str,
    max_kv: float,
    max_online: float,
    max_task_kv: float,
) -> tuple[dict[str, str], dict[str, float]]:
    best_selection: dict[str, str] | None = None
    best_metrics: dict[str, float] | None = None
    best_feasible_score = -1e9
    best_violation = 1e9

    # Search a penalty grid. The task count is small and candidate count is modest;
    # the grid gives reproducible constrained choices without requiring scipy/pulp.
    lambda_kv_values = [0.0, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    lambda_online_values = [0.0, 0.01, 0.03, 0.06, 0.10, 0.18, 0.30, 0.50, 0.80, 1.20]
    task_penalty_values = [0.0, 0.5, 1.0, 2.0, 4.0]

    for lambda_kv in lambda_kv_values:
        for lambda_online in lambda_online_values:
            for task_penalty in task_penalty_values:
                selection: dict[str, str] = {}
                for task in tasks:
                    best_name = base
                    best_value = -1e18
                    for name, by_task in stats.items():
                        if task not in by_task:
                            continue
                        item = by_task[task]
                        over_task = max(0.0, item.kv - max_task_kv)
                        value = item.score - lambda_kv * item.kv - lambda_online * item.online - task_penalty * over_task
                        if value > best_value:
                            best_value = value
                            best_name = name
                    selection[task] = best_name
                metrics = aggregate(selection, stats, tasks)
                violation = max(0.0, metrics["kv"] - max_kv) + max(0.0, metrics["online"] - max_online)
                if metrics["max_task_kv"] > max_task_kv:
                    violation += metrics["max_task_kv"] - max_task_kv
                feasible = violation <= 1e-9
                if feasible and metrics["score"] > best_feasible_score:
                    best_feasible_score = metrics["score"]
                    best_selection = selection
                    best_metrics = metrics
                elif best_selection is None and violation < best_violation:
                    best_violation = violation
                    best_selection = selection
                    best_metrics = metrics

    if best_selection is None or best_metrics is None:
        raise RuntimeError("no selection produced")
    return best_selection, best_metrics


def select_with_dp(
    stats: dict[str, dict[str, TaskStats]],
    tasks: list[str],
    base: str,
    max_kv: float,
    max_online: float,
    max_task_kv: float,
    kv_bin: float,
    online_bin: float,
) -> tuple[dict[str, str], dict[str, float]]:
    max_kv_bins = int(math.floor(max_kv * len(tasks) / kv_bin + 1e-9))
    max_online_bins = int(math.floor(max_online * len(tasks) / online_bin + 1e-9))
    if max_kv_bins <= 0 or max_online_bins <= 0:
        raise ValueError("DP constraints are too small for the chosen bins")

    states: dict[tuple[int, int], tuple[float, tuple[str, ...]]] = {(0, 0): (0.0, tuple())}
    for task in tasks:
        next_states: dict[tuple[int, int], tuple[float, tuple[str, ...]]] = {}
        choices: list[tuple[str, TaskStats, int, int]] = []
        for name, by_task in stats.items():
            if task not in by_task:
                continue
            item = by_task[task]
            if item.kv > max_task_kv + 1e-12:
                continue
            choices.append(
                (
                    name,
                    item,
                    int(math.ceil(item.kv / kv_bin - 1e-12)),
                    int(math.ceil(item.online / online_bin - 1e-12)),
                )
            )
        if not choices:
            raise RuntimeError(f"no candidate remains for task {task!r}; relax --max-task-kv")
        for (kv_state, online_state), (score_state, picked) in states.items():
            for name, item, kv_cost, online_cost in choices:
                new_kv = kv_state + kv_cost
                new_online = online_state + online_cost
                if new_kv > max_kv_bins or new_online > max_online_bins:
                    continue
                key = (new_kv, new_online)
                new_score = score_state + item.score
                current = next_states.get(key)
                if current is None or new_score > current[0]:
                    next_states[key] = (new_score, picked + (name,))
        if not next_states:
            raise RuntimeError(
                f"DP found no feasible partial selection after task {task!r}; "
                "relax --max-kv, --min-speed, --max-task-kv, or bins."
            )
        states = prune_states(next_states)

    best_key, (best_score, picked) = max(states.items(), key=lambda item: item[1][0])
    selection = {task: picked[index] for index, task in enumerate(tasks)}
    metrics = aggregate(selection, stats, tasks)
    metrics["dp_kv_bins"] = float(best_key[0])
    metrics["dp_online_bins"] = float(best_key[1])
    metrics["dp_score_sum"] = best_score
    return selection, metrics


def prune_states(
    states: dict[tuple[int, int], tuple[float, tuple[str, ...]]]
) -> dict[tuple[int, int], tuple[float, tuple[str, ...]]]:
    # Remove dominated states. A state is dominated if another state uses no more
    # KV and no more online bins while achieving at least the same score.
    items = sorted(states.items(), key=lambda item: (item[0][0], item[0][1], -item[1][0]))
    pruned: dict[tuple[int, int], tuple[float, tuple[str, ...]]] = {}
    frontier: list[tuple[int, float]] = []
    for (kv_bin, online_bin), value in items:
        score = value[0]
        dominated = False
        for prev_online, prev_score in frontier:
            if prev_online <= online_bin and prev_score >= score:
                dominated = True
                break
        if dominated:
            continue
        pruned[(kv_bin, online_bin)] = value
        frontier.append((online_bin, score))
        frontier = [
            (candidate_online, candidate_score)
            for candidate_online, candidate_score in frontier
            if not (candidate_online >= online_bin and candidate_score <= score and candidate_online != online_bin)
        ]
    return pruned


def aggregate(selection: dict[str, str], stats: dict[str, dict[str, TaskStats]], tasks: list[str]) -> dict[str, float]:
    score = 0.0
    kv = 0.0
    online = 0.0
    max_task_kv = 0.0
    for task in tasks:
        item = stats[selection[task]][task]
        score += item.score
        kv += item.kv
        online += item.online
        max_task_kv = max(max_task_kv, item.kv)
    n = float(len(tasks))
    return {
        "score": score / n,
        "kv": kv / n,
        "online": online / n,
        "max_task_kv": max_task_kv,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="v417")
    parser.add_argument("--max-kv", type=float, default=0.06)
    parser.add_argument("--min-speed", type=float, default=6.0)
    parser.add_argument("--full-online", type=float, default=3.0988)
    parser.add_argument("--max-task-kv", type=float, default=0.20)
    parser.add_argument("--solver", choices=["dp", "lagrange"], default="dp")
    parser.add_argument("--kv-bin", type=float, default=0.002)
    parser.add_argument("--online-bin", type=float, default=0.02)
    parser.add_argument("--config-out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--comment", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = DEFAULT_CANDIDATES
    candidate_by_name = {candidate.name: candidate for candidate in candidates}
    if args.base not in candidate_by_name:
        raise ValueError(f"unknown base {args.base!r}")
    stats = {candidate.name: read_stats(candidate.results) for candidate in candidates}
    tasks = sorted(set.intersection(*(set(by_task) for by_task in stats.values())))
    max_online = args.full_online / args.min_speed
    if args.solver == "dp":
        selection, metrics = select_with_dp(
            stats=stats,
            tasks=tasks,
            base=args.base,
            max_kv=args.max_kv,
            max_online=max_online,
            max_task_kv=args.max_task_kv,
            kv_bin=args.kv_bin,
            online_bin=args.online_bin,
        )
    else:
        selection, metrics = select_with_lagrange(
            stats=stats,
            tasks=tasks,
            base=args.base,
            max_kv=args.max_kv,
            max_online=max_online,
            max_task_kv=args.max_task_kv,
        )

    config: dict[str, object] = {
        "__extends": candidate_by_name[args.base].policy,
        "__comment": args.comment
        or (
            f"Constrained source composer generated from completed M100 frontiers. "
            f"Target avg KV<={args.max_kv:.3f}, speed>={args.min_speed:.2f}x, "
            f"per-task KV<={args.max_task_kv:.3f}."
        ),
        "__task_sources": {},
    }
    task_sources: dict[str, dict[str, str]] = {}
    rows: list[dict[str, object]] = []
    for task in tasks:
        name = selection[task]
        item = stats[name][task]
        if name != args.base:
            task_sources[task] = {"policy": candidate_by_name[name].policy}
        rows.append(
            {
                "task": task,
                "source": name,
                "score": item.score,
                "kv": item.kv,
                "online": item.online,
            }
        )
    config["__task_sources"] = task_sources
    Path(args.config_out).write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "base": args.base,
        "constraints": {
            "max_kv": args.max_kv,
            "min_speed": args.min_speed,
            "max_online": max_online,
            "max_task_kv": args.max_task_kv,
            "solver": args.solver,
            "kv_bin": args.kv_bin,
            "online_bin": args.online_bin,
        },
        "metrics": {
            **metrics,
            "speed": args.full_online / max(metrics["online"], 1e-9),
        },
        "selection": rows,
    }
    Path(args.summary_out).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"WROTE {args.config_out} score={metrics['score']:.4f} "
        f"kv={metrics['kv']:.2%} max_task_kv={metrics['max_task_kv']:.2%} "
        f"online={metrics['online']:.4f}s speed={args.full_online / max(metrics['online'], 1e-9):.2f}x"
    )
    for row in rows:
        print(
            f"  {row['task']:22s} {row['source']:5s} "
            f"score={row['score']:.4f} kv={row['kv']:.2%} online={row['online']:.4f}s"
        )


if __name__ == "__main__":
    main()
