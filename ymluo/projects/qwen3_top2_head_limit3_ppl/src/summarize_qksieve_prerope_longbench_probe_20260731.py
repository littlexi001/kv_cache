from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


FULL = "full_kv"
BASE = "qksieve_fullprompt_fixed410_fulltopk"
PREROPE = (
    "qksieve_fullprompt_fixed410_post2xprererank_l00to08_fulltopk"
)
METHODS = (FULL, BASE, PREROPE)
SCORE_MODES = {
    BASE: "pca_hierarchical_fixed410_qkmetric_packed_fulltopk",
    PREROPE: (
        "pca_hierarchical_fixed410_qkmetric_"
        "post2xprererank_l00to08_packed_fulltopk"
    ),
}


def history_bin(history_tokens: int) -> str:
    if history_tokens <= 8_000:
        return "le8k"
    if history_tokens <= 16_000:
        return "8k_to_16k"
    if history_tokens <= 24_000:
        return "16k_to_24k"
    return "gt24k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a strictly paired QKSieve pre-RoPE LongBench probe."
    )
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--samples_per_task", type=int, required=True)
    parser.add_argument("--bootstrap_repetitions", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    return sum(values) / len(values)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def load_rows(run_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    paths = sorted(run_root.glob("shard[0-9]*/sample_results.csv"))
    if not paths:
        raise FileNotFoundError(f"no shard CSVs below {run_root}")
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def bootstrap_task_ratios(
    per_task: dict[str, dict[str, float]],
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    tasks = sorted(per_task)
    ratios: list[float] = []
    deltas: list[float] = []
    for _ in range(repetitions):
        sampled = [rng.choice(tasks) for _ in tasks]
        base = mean([per_task[task][BASE] for task in sampled])
        prerope = mean([per_task[task][PREROPE] for task in sampled])
        ratios.append(100.0 * prerope / base if base else 100.0)
        deltas.append(prerope - base)
    return {
        "prerope_vs_base_ratio_ci95_low": percentile(ratios, 0.025),
        "prerope_vs_base_ratio_ci95_high": percentile(ratios, 0.975),
        "prerope_minus_base_macro_ci95_low": percentile(deltas, 0.025),
        "prerope_minus_base_macro_ci95_high": percentile(deltas, 0.975),
    }


def timing_summary(rows: list[dict[str, str]], method: str) -> dict[str, float]:
    selected = [row for row in rows if row["method"] == method]
    generated = sum(max(1, int(row["generated_tokens"])) for row in selected)
    decode_seconds = sum(float(row["decode_seconds"]) for row in selected)
    online_seconds = sum(float(row["online_seconds"]) for row in selected)
    return {
        "mean_query_seconds": mean(
            [float(row["query_seconds"]) for row in selected]
        ),
        "mean_decode_seconds": mean(
            [float(row["decode_seconds"]) for row in selected]
        ),
        "mean_online_seconds": mean(
            [float(row["online_seconds"]) for row in selected]
        ),
        "decode_ms_per_generated_token": 1000.0 * decode_seconds / generated,
        "online_ms_per_generated_token": 1000.0 * online_seconds / generated,
        "generated_tokens": generated,
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    macro = summary["macro"]
    bootstrap = summary["bootstrap"]
    lines = [
        "# QKSieve pre-RoPE LongBench 小规模严格配对测试",
        "",
        "| 方法 | Macro | 相对 Full | Decode ms/token | Online ms/token |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = macro[method]
        timing = summary["timing"][method]
        lines.append(
            f"| `{method}` | {item['score']:.6f} | "
            f"{item['relative_full_pct']:.3f}% | "
            f"{timing['decode_ms_per_generated_token']:.3f} | "
            f"{timing['online_ms_per_generated_token']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"pre-RoPE / base：**{summary['prerope_vs_base_pct']:.3f}%**，"
            f"task-bootstrap 95% CI "
            f"{bootstrap['prerope_vs_base_ratio_ci95_low']:.3f}%--"
            f"{bootstrap['prerope_vs_base_ratio_ci95_high']:.3f}%。",
            "",
            "| 任务 | Full | Base | pre-RoPE | pre/base |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for task, item in summary["per_task"].items():
        lines.append(
            f"| {task} | {item[FULL]:.6f} | {item[BASE]:.6f} | "
            f"{item[PREROPE]:.6f} | {item['prerope_vs_base_pct']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "| Prefix 长度 | 样本 | Full | Base | pre-RoPE | pre/base |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, item in summary["per_history_bin"].items():
        lines.append(
            f"| {label} | {item['samples']} | {item[FULL]:.6f} | "
            f"{item[BASE]:.6f} | {item[PREROPE]:.6f} | "
            f"{item['prerope_vs_base_pct']:.3f}% |"
        )
    lines.extend(
        [
            "",
            f"严格配对样本：{summary['paired_samples']}；"
            f"pre-RoPE 相对 base 的 win/tie/loss："
            f"{summary['sample_comparison']['wins']}/"
            f"{summary['sample_comparison']['ties']}/"
            f"{summary['sample_comparison']['losses']}。",
            "",
            "这是方向性 probe，不替代完整 LongBench。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    expected_tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    if not expected_tasks or args.samples_per_task <= 0:
        raise ValueError("tasks and samples_per_task must be non-empty")

    rows = load_rows(args.run_root)
    by_key: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        method = row["method"]
        if method not in METHODS:
            raise ValueError(f"unexpected method: {method}")
        key = (row["task"], row["sample_id"])
        if method in by_key[key]:
            raise ValueError(f"duplicate row: {key}/{method}")
        by_key[key][method] = row

    expected_pairs = len(expected_tasks) * args.samples_per_task
    if len(by_key) != expected_pairs:
        raise ValueError(f"expected {expected_pairs} pairs, found {len(by_key)}")
    if len(rows) != expected_pairs * len(METHODS):
        raise ValueError(f"expected {expected_pairs * len(METHODS)} rows, found {len(rows)}")
    if sorted({task for task, _ in by_key}) != sorted(expected_tasks):
        raise ValueError("task set does not match the requested probe")
    for key, methods in by_key.items():
        if set(methods) != set(METHODS):
            raise ValueError(f"incomplete strict pair: {key}/{sorted(methods)}")
        base = methods[BASE]
        prerope = methods[PREROPE]
        for method, row in ((BASE, base), (PREROPE, prerope)):
            if row["executed_path"] != method:
                raise ValueError(f"unexpected executed path: {key}/{method}")
            if row["configured_score_mode"] != SCORE_MODES[method]:
                raise ValueError(f"unexpected score mode: {key}/{method}")
            if float(row["configured_index_bits_per_token"]) != 112.0:
                raise ValueError(f"unexpected index rate: {key}/{method}")
        for field in (
            "prompt_tokens",
            "prefix_tokens",
            "suffix_tokens",
            "configured_attention_tokens",
            "configured_attention_fraction",
            "configured_candidate_fraction",
        ):
            if base[field] != prerope[field]:
                raise ValueError(f"budget mismatch: {key}/{field}")

    per_task_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    per_history_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    wins = ties = losses = 0
    for (task, _), methods in by_key.items():
        length_bin = history_bin(int(methods[FULL]["prefix_tokens"]))
        for method, row in methods.items():
            per_task_values[task][method].append(float(row["score"]))
            per_history_values[length_bin][method].append(float(row["score"]))
        delta = float(methods[PREROPE]["score"]) - float(methods[BASE]["score"])
        if delta > 1.0e-12:
            wins += 1
        elif delta < -1.0e-12:
            losses += 1
        else:
            ties += 1

    per_task: dict[str, dict[str, float]] = {}
    for task in sorted(expected_tasks):
        values = {
            method: mean(per_task_values[task][method]) for method in METHODS
        }
        values["prerope_vs_base_pct"] = (
            100.0 * values[PREROPE] / values[BASE]
            if values[BASE]
            else 100.0
        )
        per_task[task] = values

    per_history_bin: dict[str, dict[str, float | int]] = {}
    for label in ("le8k", "8k_to_16k", "16k_to_24k", "gt24k"):
        if label not in per_history_values:
            continue
        values: dict[str, float | int] = {
            method: mean(per_history_values[label][method])
            for method in METHODS
        }
        values["samples"] = len(per_history_values[label][FULL])
        values["prerope_vs_base_pct"] = (
            100.0 * float(values[PREROPE]) / float(values[BASE])
            if values[BASE]
            else 100.0
        )
        per_history_bin[label] = values

    macro_scores = {
        method: mean([per_task[task][method] for task in per_task])
        for method in METHODS
    }
    full_macro = macro_scores[FULL]
    macro = {
        method: {
            "score": score,
            "relative_full_pct": 100.0 * score / full_macro if full_macro else 100.0,
        }
        for method, score in macro_scores.items()
    }
    timing = {method: timing_summary(rows, method) for method in METHODS}
    timing[BASE]["speedup_vs_full_decode"] = (
        timing[FULL]["decode_ms_per_generated_token"]
        / timing[BASE]["decode_ms_per_generated_token"]
    )
    timing[PREROPE]["speedup_vs_full_decode"] = (
        timing[FULL]["decode_ms_per_generated_token"]
        / timing[PREROPE]["decode_ms_per_generated_token"]
    )
    timing[PREROPE]["speedup_vs_base_decode"] = (
        timing[BASE]["decode_ms_per_generated_token"]
        / timing[PREROPE]["decode_ms_per_generated_token"]
    )

    sparse_rows = [row for row in rows if row["method"] != FULL]
    summary: dict[str, Any] = {
        "schema": "qksieve_prerope_longbench_probe_v1",
        "run_root": str(args.run_root),
        "tasks": sorted(expected_tasks),
        "samples_per_task": args.samples_per_task,
        "paired_samples": len(by_key),
        "rows": len(rows),
        "methods": list(METHODS),
        "mean_prompt_tokens": mean(
            [float(methods[FULL]["prompt_tokens"]) for methods in by_key.values()]
        ),
        "min_prompt_tokens": min(
            int(methods[FULL]["prompt_tokens"]) for methods in by_key.values()
        ),
        "max_prompt_tokens": max(
            int(methods[FULL]["prompt_tokens"]) for methods in by_key.values()
        ),
        "mean_sparse_attention_tokens": mean(
            [float(row["configured_attention_tokens"]) for row in sparse_rows]
        ),
        "mean_sparse_attention_fraction_pct": 100.0
        * mean([float(row["configured_attention_fraction"]) for row in sparse_rows]),
        "configured_index_bits_per_token": sorted(
            {float(row["configured_index_bits_per_token"]) for row in sparse_rows}
        ),
        "strict_budget_and_index_match": True,
        "macro": macro,
        "prerope_vs_base_pct": (
            100.0 * macro_scores[PREROPE] / macro_scores[BASE]
            if macro_scores[BASE]
            else 100.0
        ),
        "per_task": per_task,
        "per_history_bin": per_history_bin,
        "sample_comparison": {"wins": wins, "ties": ties, "losses": losses},
        "bootstrap": bootstrap_task_ratios(
            per_task,
            args.bootstrap_repetitions,
            args.seed,
        ),
        "timing": timing,
    }

    output_json = args.run_root / "probe_summary.json"
    output_md = args.run_root / "probe_summary_zh.md"
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(summary, output_md)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
