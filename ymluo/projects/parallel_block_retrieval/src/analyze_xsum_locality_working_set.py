from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the coverage/completeness tradeoff of locality windows."
    )
    parser.add_argument("--rows", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--source_name", default="real XSum 10M")
    parser.add_argument("--bootstrap_samples", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def paired_bootstrap(
    values: list[float], *, samples: int, seed: int
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "bootstrap95": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "wins": int((array < 0).sum()),
        "losses": int((array > 0).sum()),
        "ties": int((array == 0).sum()),
    }


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.rows)
    lookup = {
        (int(row["query_id"]), str(row["method"])): row for row in rows
    }
    query_ids = sorted({int(row["query_id"]) for row in rows})
    query_nll = {
        query_id: float(lookup[(query_id, "query_only")]["mean_nll"])
        for query_id in query_ids
    }
    families = ["bm25", "e5", "bm25_e5_rrf"]
    layouts = ["local_1x8", "local_2x4", "local_4x2"]
    comparisons = []
    conditional_utility = []
    for family in families:
        base = [lookup[(query_id, family)] for query_id in query_ids]
        for layout in layouts:
            method = f"{family}_{layout}"
            selected = [lookup[(query_id, method)] for query_id in query_ids]
            comparisons.append(
                {
                    "base_method": family,
                    "locality_method": method,
                    "delta_nll": paired_bootstrap(
                        [
                            float(right["mean_nll"]) - float(left["mean_nll"])
                            for left, right in zip(base, selected)
                        ],
                        samples=args.bootstrap_samples,
                        seed=args.seed + len(comparisons),
                    ),
                    "delta_source_block_recall": paired_bootstrap(
                        [
                            float(right["source_block_recall"])
                            - float(left["source_block_recall"])
                            for left, right in zip(base, selected)
                        ],
                        samples=args.bootstrap_samples,
                        seed=args.seed + 100 + len(comparisons),
                    ),
                    "delta_source_any_hit": paired_bootstrap(
                        [
                            float(right["source_any_hit"])
                            - float(left["source_any_hit"])
                            for left, right in zip(base, selected)
                        ],
                        samples=args.bootstrap_samples,
                        seed=args.seed + 200 + len(comparisons),
                    ),
                }
            )

    for method in [
        "bm25_e5_rrf",
        "bm25_e5_rrf_local_1x8",
        "bm25_e5_rrf_local_2x4",
        "bm25_e5_rrf_local_4x2",
    ]:
        group = [lookup[(query_id, method)] for query_id in query_ids]
        buckets = []
        for lower, upper in ((0, 0), (1, 2), (3, 4), (5, 8)):
            members = [
                row
                for row in group
                if lower <= round(float(row["source_block_recall"]) * 8) <= upper
            ]
            if not members:
                continue
            buckets.append(
                {
                    "source_blocks": f"{lower}-{upper}",
                    "queries": len(members),
                    "mean_delta_nll_vs_query_only": mean(
                        float(row["mean_nll"]) - query_nll[int(row["query_id"])]
                        for row in members
                    ),
                }
            )
        conditional_utility.append({"method": method, "buckets": buckets})

    summary = {
        "source": f"{args.source_name} locality-shaped 512-token working sets",
        "queries": len(query_ids),
        "contains_synthetic_text": False,
        "selection_uses_target": False,
        "comparisons": comparisons,
        "conditional_utility": conditional_utility,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
