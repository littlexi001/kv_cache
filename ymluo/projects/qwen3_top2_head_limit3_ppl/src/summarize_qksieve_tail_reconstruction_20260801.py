"""Aggregate completed QKSieve tail-reconstruction cases across run roots."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def bootstrap_interval(
    deltas: list[float],
    *,
    draws: int = 20_000,
    seed: int = 20260801,
) -> list[float]:
    rng = random.Random(seed)
    count = len(deltas)
    samples = sorted(
        math.exp(
            -sum(deltas[rng.randrange(count)] for _ in range(count))
            / count
        )
        for _ in range(draws)
    )
    return [samples[int(0.025 * draws)], samples[int(0.975 * draws)]]


def load_cases(run_roots: list[Path]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for root in run_roots:
        for path in sorted(root.glob("*/summary.json")):
            if not (path.parent / "ALL_COMPLETE").exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            identity = (
                str(payload["topic"]),
                int(payload["seed"]),
                int(payload["history_tokens"]),
            )
            if identity in seen:
                raise ValueError(f"duplicate case across roots: {identity}")
            seen.add(identity)
            rows = {row["variant"]: row for row in payload["rows"]}
            if "full_attention" not in rows:
                raise ValueError(f"missing Full row: {path}")
            cases.append(
                {
                    "case": path.parent.name,
                    "root": str(root),
                    "topic": payload["topic"],
                    "seed": int(payload["seed"]),
                    "history_tokens": int(payload["history_tokens"]),
                    "eval_tokens": int(payload["eval_tokens"]),
                    "rows": rows,
                }
            )
    if not cases:
        raise ValueError("no completed cases")
    return cases


def aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for case in cases:
        full = case["rows"]["full_attention"]
        for variant, row in case["rows"].items():
            grouped[variant].append((full, row, case))

    rows: list[dict[str, Any]] = []
    for variant, pairs in sorted(grouped.items()):
        deltas = [
            float(row["nll"]) - float(full["nll"])
            for full, row, _ in pairs
        ]
        full_nll = mean([float(full["nll"]) for full, _, _ in pairs])
        method_nll = mean([float(row["nll"]) for _, row, _ in pairs])
        rows.append(
            {
                "variant": variant,
                "cases": len(pairs),
                "eval_tokens": sum(case["eval_tokens"] for _, _, case in pairs),
                "geometric_full_ppl": math.exp(full_nll),
                "geometric_method_ppl": math.exp(method_nll),
                "quality_retention": math.exp(full_nll - method_nll),
                "topic_bootstrap_quality_ci95": bootstrap_interval(deltas),
                "worst_case_quality_retention": min(math.exp(-x) for x in deltas),
                "top1_agreement": mean(
                    [float(row.get("top1_agreement", 1.0)) for _, row, _ in pairs]
                ),
                "kl_full_to_sparse": mean(
                    [float(row.get("kl_full_to_sparse_mean", 0.0)) for _, row, _ in pairs]
                ),
                "index_ratio_of_full_kv": mean(
                    [float(row.get("packed_index_ratio_of_full_kv", 0.0)) for _, row, _ in pairs]
                ),
                "steady_seconds_per_token_python_prototype": mean(
                    [float(row.get("steady_sparse_seconds_per_step", 0.0)) for _, row, _ in pairs]
                ),
                "per_case": [
                    {
                        "case": case["case"],
                        "topic": case["topic"],
                        "seed": case["seed"],
                        "quality_retention": math.exp(
                            float(full["nll"]) - float(row["nll"])
                        ),
                        "full_ppl": float(full["ppl"]),
                        "method_ppl": float(row["ppl"]),
                        "top1_agreement": float(row.get("top1_agreement", 1.0)),
                        "kl_full_to_sparse": float(
                            row.get("kl_full_to_sparse_mean", 0.0)
                        ),
                    }
                    for full, row, case in pairs
                ],
            }
        )
    return {
        "schema": "qksieve_tail_reconstruction_robustness_v1",
        "case_count": len(cases),
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    report = aggregate(load_cases(args.run_roots))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
