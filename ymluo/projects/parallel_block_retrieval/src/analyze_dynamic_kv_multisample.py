from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired analysis for dynamic KV generation.")
    parser.add_argument("--rows", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=50000)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def bootstrap_ci(values: np.ndarray, samples: int) -> list[float]:
    rng = np.random.default_rng(20260711)
    estimates = np.asarray(
        [rng.choice(values, len(values), replace=True).mean() for _ in range(samples)]
    )
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def mean(rows: Sequence[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.rows))
    by = {(int(row["query_id"]), str(row["method"])): row for row in rows}
    query_ids = sorted({int(row["query_id"]) for row in rows})
    methods = [
        "question_only",
        "full_source",
        "static_k3",
        "dynamic_c1k3",
        "dynamic_c3k3",
    ]
    datasets = sorted({str(row["dataset"]) for row in rows})
    dataset_summary = []
    for dataset in datasets:
        for method in methods:
            group = [
                row for row in rows if row["dataset"] == dataset and row["method"] == method
            ]
            dataset_summary.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "queries": len(group),
                    "answer_hit_128": mean(group, "answer_hit_128"),
                    "structured_final_f1": mean(group, "structured_final_f1"),
                }
            )

    pairwise = []
    for method in ["dynamic_c1k3", "dynamic_c3k3"]:
        f1_delta = np.asarray(
            [
                float(by[(query_id, method)]["structured_final_f1"])
                - float(by[(query_id, "static_k3")]["structured_final_f1"])
                for query_id in query_ids
            ]
        )
        hit_delta = np.asarray(
            [
                float(by[(query_id, method)]["answer_hit_128"])
                - float(by[(query_id, "static_k3")]["answer_hit_128"])
                for query_id in query_ids
            ]
        )
        pairwise.append(
            {
                "method": method,
                "baseline": "static_k3",
                "mean_f1_delta": float(f1_delta.mean()),
                "f1_delta_bootstrap_ci95": bootstrap_ci(
                    f1_delta, args.bootstrap_samples
                ),
                "f1_wins": int((f1_delta > 0).sum()),
                "f1_losses": int((f1_delta < 0).sum()),
                "mean_hit_delta": float(hit_delta.mean()),
                "hit_delta_bootstrap_ci95": bootstrap_ci(
                    hit_delta, args.bootstrap_samples
                ),
                "hit_gained_query_ids": [
                    query_id
                    for query_id, value in zip(query_ids, hit_delta)
                    if value > 0
                ],
                "hit_lost_query_ids": [
                    query_id
                    for query_id, value in zip(query_ids, hit_delta)
                    if value < 0
                ],
            }
        )
    output = {
        "queries": len(query_ids),
        "dataset_summary": dataset_summary,
        "pairwise": pairwise,
        "notable_dynamic_c3_exact_query_ids": [
            query_id
            for query_id in query_ids
            if float(by[(query_id, "dynamic_c3k3")]["structured_final_f1"]) >= 0.999
        ],
    }
    Path(args.output).write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
