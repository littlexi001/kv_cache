"""Aggregate strict-pair QKSieve bit-profile quality experiments."""

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
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def bootstrap_quality_interval(
    delta_nll: list[float],
    *,
    samples: int = 20_000,
    seed: int = 20260801,
) -> tuple[float, float]:
    if not delta_nll:
        raise ValueError("delta_nll must not be empty")
    rng = random.Random(seed)
    count = len(delta_nll)
    values = sorted(
        math.exp(
            -sum(delta_nll[rng.randrange(count)] for _ in range(count))
            / count
        )
        for _ in range(samples)
    )
    return values[int(0.025 * samples)], values[int(0.975 * samples)]


def aggregate(run_root: Path) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for path in sorted(run_root.glob("*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not (path.parent / "ALL_COMPLETE").exists():
            continue
        rows = {row["variant"]: row for row in payload["rows"]}
        if "full_attention" not in rows:
            raise ValueError(f"missing Full row: {path}")
        cases.append(
            {
                "case": path.parent.name,
                "history_tokens": int(payload["history_tokens"]),
                "topic": payload["topic"],
                "synthetic": payload.get("synthetic_rope") is not None,
                "rows": rows,
            }
        )
    if not cases:
        raise ValueError("no complete cases found")

    grouped: dict[
        tuple[int, str],
        list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
    ] = defaultdict(list)
    for case in cases:
        full = case["rows"]["full_attention"]
        for variant, row in case["rows"].items():
            grouped[(case["history_tokens"], variant)].append(
                (full, row, case)
            )

    rows: list[dict[str, Any]] = []
    for (history_tokens, variant), pairs in sorted(grouped.items()):
        full_nll = mean([float(full["nll"]) for full, _, _ in pairs])
        method_nll = mean([float(row["nll"]) for _, row, _ in pairs])
        delta_nll = [
            float(row["nll"]) - float(full["nll"])
            for full, row, _ in pairs
        ]
        ci_low, ci_high = bootstrap_quality_interval(delta_nll)
        result: dict[str, Any] = {
            "history_tokens": history_tokens,
            "variant": variant,
            "cases": len(pairs),
            "tokens": sum(int(row["tokens"]) for _, row, _ in pairs),
            "geometric_ppl": math.exp(method_nll),
            "quality_retention": math.exp(full_nll - method_nll),
            "topic_bootstrap_quality_ci95": [ci_low, ci_high],
            "worst_case_quality_retention": min(
                math.exp(-value) for value in delta_nll
            ),
            "better_than_full_case_fraction": mean(
                [float(value < 0.0) for value in delta_nll]
            ),
            "top1_agreement": mean(
                [
                    float(row.get("top1_agreement", 1.0))
                    for _, row, _ in pairs
                ]
            ),
            "kl_full_to_sparse": mean(
                [
                    float(row.get("kl_full_to_sparse_mean", 0.0))
                    for _, row, _ in pairs
                ]
            ),
            "exact_topk_recall": mean(
                [
                    float(row.get("exact_topk_recall_mean", 0.0))
                    for _, row, _ in pairs
                ]
            ),
            "selected_attention_mass": mean(
                [
                    float(row.get("selected_attention_mass_mean", 0.0))
                    for _, row, _ in pairs
                ]
            ),
            "oracle_topk_attention_mass": mean(
                [
                    float(row.get("oracle_topk_attention_mass_mean", 0.0))
                    for _, row, _ in pairs
                ]
            ),
            "index_ratio_of_full_kv": mean(
                [
                    float(row.get("packed_index_ratio_of_full_kv", 0.0))
                    for _, row, _ in pairs
                ]
            ),
            "steady_seconds_per_token": mean(
                [
                    float(row.get("steady_sparse_seconds_per_step", 0.0))
                    for _, row, _ in pairs
                ]
            ),
            "per_case": [
                {
                    "case": case["case"],
                    "topic": case["topic"],
                    "quality_retention": math.exp(
                        float(full["nll"]) - float(row["nll"])
                    ),
                    "full_ppl": float(full["ppl"]),
                    "method_ppl": float(row["ppl"]),
                }
                for full, row, case in pairs
            ],
        }
        if all("synthetic_gold_nll" in row for _, row, _ in pairs):
            full_gold_nll = mean(
                [
                    float(full["synthetic_gold_nll"])
                    for full, _, _ in pairs
                ]
            )
            method_gold_nll = mean(
                [
                    float(row["synthetic_gold_nll"])
                    for _, row, _ in pairs
                ]
            )
            result["synthetic_gold_quality_retention"] = math.exp(
                full_gold_nll - method_gold_nll
            )
            result["synthetic_gold_accuracy"] = mean(
                [
                    float(row["synthetic_gold_correct"])
                    for _, row, _ in pairs
                ]
            )
        rows.append(result)
    return {
        "schema": "qksieve_bit_profile_robustness_v1",
        "run_root": str(run_root),
        "complete_cases": len(cases),
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    report = aggregate(args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
