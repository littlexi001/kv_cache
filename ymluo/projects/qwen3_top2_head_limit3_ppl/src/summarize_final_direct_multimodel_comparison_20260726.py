from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


RA_BITQ_13_TASKS = {
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "qmsum",
    "triviaqa",
    "passage_retrieval_en",
    "gov_report",
    "multi_news",
    "lcc",
    "repobench-p",
}

SELF_INDEXING_11_TASKS = {
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "qmsum",
    "trec",
    "triviaqa",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama_csv", type=Path, required=True)
    parser.add_argument("--qwen_csv", type=Path, required=True)
    parser.add_argument("--baseline_method_summary", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(len(ordered) - 1, low + 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def paired_macro_ci(
    pairs_by_task: dict[str, list[tuple[float, float]]],
    samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = random.Random(seed)
    draws = []
    tasks = sorted(pairs_by_task)
    for _ in range(samples):
        task_deltas = []
        for task in tasks:
            pairs = pairs_by_task[task]
            sampled = [pairs[rng.randrange(len(pairs))] for _ in pairs]
            task_deltas.append(
                mean(sparse - full for full, sparse in sampled)
            )
        draws.append(mean(task_deltas))
    return quantile(draws, 0.025), quantile(draws, 0.975)


def summarize_model(
    rows: list[dict[str, str]],
    model: str,
    bootstrap_samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    methods = sorted({row["method"] for row in rows})
    if len(methods) != 2 or "full_kv" not in methods:
        raise ValueError(f"expected Full plus one sparse method, got {methods}")
    sparse_method = next(method for method in methods if method != "full_kv")

    keyed: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        keyed[(row["task"], row["sample_id"])][row["method"]] = row
    if not keyed or any(set(pair) != set(methods) for pair in keyed.values()):
        raise ValueError("rows are not strict Full/sparse pairs")

    subsets = {
        "longbench16": set(row["task"] for row in rows),
        "rabitq13": RA_BITQ_13_TASKS,
        "selfindex11": SELF_INDEXING_11_TASKS,
    }
    overall = []
    task_rows = []
    for subset_name, subset_tasks in subsets.items():
        present = set(row["task"] for row in rows)
        if not subset_tasks <= present:
            raise ValueError(
                f"{subset_name} is missing {sorted(subset_tasks - present)}"
            )
        pairs_by_task: dict[str, list[tuple[float, float]]] = defaultdict(list)
        selected_pairs = []
        for (task, _), pair in keyed.items():
            if task not in subset_tasks:
                continue
            full = pair["full_kv"]
            sparse = pair[sparse_method]
            pairs_by_task[task].append(
                (float(full["score"]), float(sparse["score"]))
            )
            selected_pairs.append((full, sparse))

        full_task_scores = {
            task: mean(full for full, _ in pairs)
            for task, pairs in pairs_by_task.items()
        }
        sparse_task_scores = {
            task: mean(sparse for _, sparse in pairs)
            for task, pairs in pairs_by_task.items()
        }
        full_macro = mean(full_task_scores.values())
        sparse_macro = mean(sparse_task_scores.values())
        ci_low, ci_high = paired_macro_ci(
            pairs_by_task,
            bootstrap_samples,
            seed,
        )
        sparse_rows = [sparse for _, sparse in selected_pairs]
        full_rows = [full for full, _ in selected_pairs]
        token_normalized_pairs = [
            (full, sparse)
            for full, sparse in selected_pairs
            if float(full.get("generated_tokens", 0.0)) > 0.0
            and float(sparse.get("generated_tokens", 0.0)) > 0.0
        ]
        decode_normalized_pairs = [
            (full, sparse)
            for full, sparse in token_normalized_pairs
            if float(full.get("decode_seconds", 0.0)) > 0.0
            and float(sparse.get("decode_seconds", 0.0)) > 0.0
        ]
        online_normalized_pairs = [
            (full, sparse)
            for full, sparse in token_normalized_pairs
            if float(full["online_seconds"]) > 0.0
            and float(sparse["online_seconds"]) > 0.0
        ]
        overall.append(
            {
                "model": model,
                "subset": subset_name,
                "tasks": len(subset_tasks),
                "paired_samples": len(selected_pairs),
                "full_macro": full_macro,
                "countcap_macro": sparse_macro,
                "quality_retention": (
                    sparse_macro / full_macro if full_macro else 0.0
                ),
                "macro_delta": sparse_macro - full_macro,
                "macro_delta_ci95_low": ci_low,
                "macro_delta_ci95_high": ci_high,
                "mean_prompt_tokens": mean(
                    float(row["prompt_tokens"]) for row in sparse_rows
                ),
                "mean_attention_tokens": mean(
                    float(row["configured_attention_tokens"])
                    for row in sparse_rows
                ),
                "mean_attention_fraction": mean(
                    float(row["configured_attention_fraction"])
                    for row in sparse_rows
                ),
                "paired_online_speedup": mean(
                    float(full["online_seconds"])
                    / float(sparse["online_seconds"])
                    for full, sparse in selected_pairs
                    if float(sparse["online_seconds"]) > 0.0
                ),
                "paired_total_speedup": mean(
                    float(full["total_seconds"])
                    / float(sparse["total_seconds"])
                    for full, sparse in selected_pairs
                    if float(sparse["total_seconds"]) > 0.0
                ),
                "full_mean_online_seconds": mean(
                    float(row["online_seconds"]) for row in full_rows
                ),
                "countcap_mean_online_seconds": mean(
                    float(row["online_seconds"]) for row in sparse_rows
                ),
                "full_mean_generated_tokens": mean(
                    float(row.get("generated_tokens", 0.0))
                    for row in full_rows
                ),
                "countcap_mean_generated_tokens": mean(
                    float(row.get("generated_tokens", 0.0))
                    for row in sparse_rows
                ),
                "paired_decode_per_token_speedup": (
                    mean(
                        (
                            float(full["decode_seconds"])
                            / float(full["generated_tokens"])
                        )
                        / (
                            float(sparse["decode_seconds"])
                            / float(sparse["generated_tokens"])
                        )
                        for full, sparse in decode_normalized_pairs
                    )
                    if decode_normalized_pairs
                    else None
                ),
                "paired_online_per_token_speedup": (
                    mean(
                        (
                            float(full["online_seconds"])
                            / float(full["generated_tokens"])
                        )
                        / (
                            float(sparse["online_seconds"])
                            / float(sparse["generated_tokens"])
                        )
                        for full, sparse in online_normalized_pairs
                    )
                    if online_normalized_pairs
                    else None
                ),
                "aggregate_decode_per_token_speedup": (
                    (
                        sum(
                            float(full["decode_seconds"])
                            for full, _ in decode_normalized_pairs
                        )
                        / sum(
                            float(full["generated_tokens"])
                            for full, _ in decode_normalized_pairs
                        )
                    )
                    / (
                        sum(
                            float(sparse["decode_seconds"])
                            for _, sparse in decode_normalized_pairs
                        )
                        / sum(
                            float(sparse["generated_tokens"])
                            for _, sparse in decode_normalized_pairs
                        )
                    )
                    if decode_normalized_pairs
                    else None
                ),
                "aggregate_online_per_token_speedup": (
                    (
                        sum(
                            float(full["online_seconds"])
                            for full, _ in online_normalized_pairs
                        )
                        / sum(
                            float(full["generated_tokens"])
                            for full, _ in online_normalized_pairs
                        )
                    )
                    / (
                        sum(
                            float(sparse["online_seconds"])
                            for _, sparse in online_normalized_pairs
                        )
                        / sum(
                            float(sparse["generated_tokens"])
                            for _, sparse in online_normalized_pairs
                        )
                    )
                    if online_normalized_pairs
                    else None
                ),
            }
        )

        if subset_name == "longbench16":
            for task in sorted(subset_tasks):
                full_score = full_task_scores[task]
                sparse_score = sparse_task_scores[task]
                task_rows.append(
                    {
                        "model": model,
                        "task": task,
                        "samples": len(pairs_by_task[task]),
                        "full_score": full_score,
                        "countcap_score": sparse_score,
                        "quality_retention": (
                            sparse_score / full_score if full_score else 0.0
                        ),
                        "delta": sparse_score - full_score,
                    }
                )
    return overall, task_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    overall = []
    tasks = []
    for model, path in (
        ("Llama-3.1-8B-Instruct", args.llama_csv),
        ("Qwen3-4B-Instruct", args.qwen_csv),
    ):
        model_overall, model_tasks = summarize_model(
            read_csv(path),
            model,
            args.bootstrap_samples,
            args.seed,
        )
        overall.extend(model_overall)
        tasks.extend(model_tasks)

    baselines = read_csv(args.baseline_method_summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "countcap_overall.csv", overall)
    write_csv(args.output_dir / "countcap_by_task.csv", tasks)
    write_csv(args.output_dir / "same_environment_baselines.csv", baselines)
    payload = {
        "countcap": overall,
        "countcap_by_task": tasks,
        "same_environment_baselines": baselines,
        "published_reference": {
            "RaBitQCache_Llama31_8B_LongBench13": {
                "full_score": 50.58,
                "rabitq_score": 50.63,
                "mean_budget_ratio": 0.1733,
                "protocol_note": (
                    "Published ICML 2026 result; first two layers use Full "
                    "attention; not a same-run reproduction."
                ),
                "source": "https://arxiv.org/abs/2606.31519",
            },
            "RaBitQCache_paper_LongBench13_table": [
                {
                    "method": "Full",
                    "setting": "-",
                    "score": 50.58,
                    "budget_ratio": 1.0,
                },
                {
                    "method": "RaBitQCache",
                    "setting": "top-p=0.95",
                    "score": 50.63,
                    "budget_ratio": 0.1733,
                },
                {
                    "method": "Quest",
                    "setting": "1024",
                    "score": 46.52,
                    "budget_ratio": 0.1138,
                },
                {
                    "method": "Double Sparsity",
                    "setting": "1024",
                    "score": 50.28,
                    "budget_ratio": 0.1142,
                },
                {
                    "method": "SparQ",
                    "setting": "ratio=0.25",
                    "score": 50.15,
                    "budget_ratio": 0.25,
                },
                {
                    "method": "MagicPIG",
                    "setting": "official default",
                    "score": 49.95,
                    "budget_ratio": None,
                },
                {
                    "method": "PyramidKV",
                    "setting": "official default",
                    "score": 45.09,
                    "budget_ratio": None,
                },
                {
                    "method": "SnapKV",
                    "setting": "official default",
                    "score": 44.91,
                    "budget_ratio": None,
                },
                {
                    "method": "PQCache",
                    "setting": "official default",
                    "score": 50.34,
                    "budget_ratio": None,
                },
                {
                    "method": "KIVI",
                    "setting": "official default",
                    "score": 50.13,
                    "budget_ratio": None,
                },
            ],
            "SelfIndexingKVCache_Llama31_8B_LongBench11": {
                "full_score": 58.7,
                "self_indexing_16bit_score": 58.4,
                "self_indexing_2bit_score": 58.2,
                "qwen25_14b_full_score": 56.9,
                "qwen25_14b_self_indexing_16bit_score": 55.9,
                "qwen25_14b_self_indexing_2bit_score": 55.7,
                "budget_tokens": 160,
                "sink_tokens": 64,
                "dynamic_tokens": 96,
                "protocol_note": (
                    "Published AAAI 2026 result on an 11-task LongBench "
                    "subset; compressed K/V and 1-bit retrieval index; "
                    "not a same-run reproduction."
                ),
                "source": (
                    "https://ojs.aaai.org/index.php/AAAI/"
                    "article/download/39988/43949"
                ),
            },
        },
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
