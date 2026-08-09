from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from run_controlled_public_kv_benchmark_v1 import LONG_BENCH_PROMPTS, score_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize aligned KVCache-Factory LongBench results."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--rag_csv", required=True)
    parser.add_argument("--full_csv", required=True)
    parser.add_argument("--rag_method", default="hybrid_recent_1024")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: Sequence[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def load_factory_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*/*/*.json")):
        task = path.parent.name
        method = path.stem
        if task not in LONG_BENCH_PROMPTS:
            continue
        metric = str(LONG_BENCH_PROMPTS[task]["metric"])
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            prediction = str(record.get("pred", ""))
            answers = [str(item) for item in record.get("answers", [])]
            all_classes = [str(item) for item in (record.get("all_classes") or [])]
            rows.append(
                {
                    "task": task,
                    "sample_id": str(record["_id"]),
                    "method": method,
                    "metric": metric,
                    "score": score_prediction(
                        metric,
                        prediction,
                        answers,
                        all_classes,
                        task=task,
                    ),
                    "prediction": prediction,
                    "answers": json.dumps(answers, ensure_ascii=False),
                    "prompt_tokens": int(record["prompt_tokens"]),
                    "context_tokens": int(record["context_tokens"]),
                    "generated_tokens": int(record["generated_tokens"]),
                    "generation_seconds": float(record["generation_seconds"]),
                    "peak_allocated_mb": float(record["peak_allocated_mb"]),
                    "peak_reserved_mb": float(record["peak_reserved_mb"]),
                    "max_capacity_prompts": int(record["max_capacity_prompts"]),
                    "kv_cache_granularity": str(record["kv_cache_granularity"]),
                }
            )
    return rows


def load_baseline_rows(
    path: Path,
    *,
    label: str,
    method_filter: str | None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in read_csv(path):
        if method_filter is not None and row.get("method") != method_filter:
            continue
        seconds_key = "total_seconds"
        output.append(
            {
                "task": str(row["task"]),
                "sample_id": str(row["sample_id"]),
                "method": label,
                "score": float(row["score"]),
                "generation_seconds": float(row[seconds_key]),
            }
        )
    return output


def key_set(rows: Sequence[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(str(row["task"]), str(row["sample_id"])) for row in rows}


def validate_coverage(
    groups: dict[str, list[dict[str, Any]]], expected: set[tuple[str, str]]
) -> None:
    for method, rows in groups.items():
        actual = key_set(rows)
        if actual != expected:
            missing = sorted(expected - actual)[:5]
            extra = sorted(actual - expected)[:5]
            raise ValueError(
                f"{method} sample coverage mismatch: rows={len(rows)} "
                f"expected={len(expected)} missing={missing} extra={extra}"
            )


def summarize_method(method: str, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    times = [float(row["generation_seconds"]) for row in rows]
    output = {
        "method": method,
        "samples": len(rows),
        "score": sum(float(row["score"]) for row in rows) / len(rows),
        "mean_seconds": sum(times) / len(times),
        "median_seconds": percentile(times, 0.5),
        "p95_seconds": percentile(times, 0.95),
    }
    optional = (
        "prompt_tokens",
        "context_tokens",
        "generated_tokens",
        "peak_allocated_mb",
        "peak_reserved_mb",
    )
    for key in optional:
        values = [float(row[key]) for row in rows if key in row]
        output[f"mean_{key}"] = sum(values) / len(values) if values else ""
    return output


def task_summaries(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method, rows in sorted(groups.items()):
        tasks = sorted({str(row["task"]) for row in rows})
        for task in tasks:
            subset = [row for row in rows if row["task"] == task]
            output.append(
                {
                    "method": method,
                    "task": task,
                    "samples": len(subset),
                    "score": sum(float(row["score"]) for row in subset) / len(subset),
                    "mean_seconds": sum(
                        float(row["generation_seconds"]) for row in subset
                    )
                    / len(subset),
                }
            )
    return output


def paired_comparison(
    left_name: str,
    right_name: str,
    groups: dict[str, list[dict[str, Any]]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    left = {
        (str(row["task"]), str(row["sample_id"])): float(row["score"])
        for row in groups[left_name]
    }
    right = {
        (str(row["task"]), str(row["sample_id"])): float(row["score"])
        for row in groups[right_name]
    }
    keys = sorted(left)
    if set(keys) != set(right):
        raise ValueError(f"Cannot pair {left_name} and {right_name}")
    deltas = np.asarray([left[key] - right[key] for key in keys], dtype=np.float64)
    tasks = sorted({key[0] for key in keys})
    indices = [
        np.asarray([index for index, key in enumerate(keys) if key[0] == task])
        for task in tasks
    ]
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
    for iteration in range(bootstrap_samples):
        sampled = np.concatenate(
            [rng.choice(task_indices, len(task_indices), replace=True) for task_indices in indices]
        )
        bootstrap[iteration] = deltas[sampled].mean()
    permutations = np.empty(bootstrap_samples, dtype=np.float64)
    for start in range(0, bootstrap_samples, 1000):
        count = min(1000, bootstrap_samples - start)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, len(deltas)))
        permutations[start : start + count] = (signs * deltas).mean(axis=1)
    mean_delta = float(deltas.mean())
    epsilon = 1.0e-12
    return {
        "left": left_name,
        "right": right_name,
        "samples": len(keys),
        "left_score": sum(left.values()) / len(left),
        "right_score": sum(right.values()) / len(right),
        "delta": mean_delta,
        "ci95_low": percentile(bootstrap, 0.025),
        "ci95_high": percentile(bootstrap, 0.975),
        "sign_flip_p": float(
            (np.abs(permutations) >= abs(mean_delta)).mean()
        ),
        "wins": int((deltas > epsilon).sum()),
        "ties": int((np.abs(deltas) <= epsilon).sum()),
        "losses": int((deltas < -epsilon).sum()),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    factory_rows = load_factory_rows(Path(args.input_dir))
    rag_rows = load_baseline_rows(
        Path(args.rag_csv), label="HybridRecentRAG", method_filter=args.rag_method
    )
    full_rows = load_baseline_rows(
        Path(args.full_csv), label="FullAttention", method_filter=None
    )
    expected = key_set(factory_rows)
    rag_rows = [row for row in rag_rows if (row["task"], row["sample_id"]) in expected]
    full_rows = [row for row in full_rows if (row["task"], row["sample_id"]) in expected]

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in factory_rows:
        groups.setdefault(str(row["method"]), []).append(row)
    groups["HybridRecentRAG"] = rag_rows
    groups["FullAttention"] = full_rows
    validate_coverage(groups, expected)

    summaries = [summarize_method(method, rows) for method, rows in groups.items()]
    summaries.sort(key=lambda row: float(row["score"]), reverse=True)
    tasks = task_summaries(groups)
    pairs = [
        ("SnapKV", "HybridRecentRAG"),
        ("AdaKV", "HybridRecentRAG"),
        ("H2O", "HybridRecentRAG"),
        ("AdaKV", "SnapKV"),
        ("SnapKV", "FullAttention"),
        ("AdaKV", "FullAttention"),
        ("H2O", "FullAttention"),
    ]
    comparisons = [
        paired_comparison(
            left,
            right,
            groups,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + index,
        )
        for index, (left, right) in enumerate(pairs)
    ]

    write_csv(output_dir / "sample_results.csv", factory_rows)
    write_csv(output_dir / "method_summary.csv", summaries)
    write_csv(output_dir / "task_summary.csv", tasks)
    write_csv(output_dir / "paired_comparisons.csv", comparisons)
    payload = {
        "input_dir": str(Path(args.input_dir)),
        "sample_coverage": len(expected),
        "selection_uses_answers": False,
        "method_summary": summaries,
        "paired_comparisons": comparisons,
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
