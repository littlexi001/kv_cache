from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


METHODS = (
    "full_kv",
    "countcap_fullprompt_keypca_direct",
    "countcap_fullprompt_keypca_direct_fused",
)


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def summarize_case(case_dir: Path) -> dict[str, Any]:
    repetitions: list[dict[str, dict[str, str]]] = []
    for repeat_dir in sorted(
        case_dir.glob("repeat*"),
        key=lambda path: int(path.name.removeprefix("repeat")),
    ):
        with (repeat_dir / "sample_results.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        by_method = {row["method"]: row for row in rows}
        if set(by_method) != set(METHODS):
            raise RuntimeError(f"unexpected methods in {repeat_dir}: {set(by_method)}")
        repetitions.append(by_method)
    full_online = [
        float(repetition["full_kv"]["online_seconds"])
        for repetition in repetitions
    ]
    result: dict[str, Any] = {
        "configured_generation_tokens": int(case_dir.name.removeprefix("g")),
        "prompt_tokens": int(repetitions[0]["full_kv"]["prompt_tokens"]),
        "repetitions": len(repetitions),
        "methods": {},
    }
    for method in METHODS:
        rows = [repetition[method] for repetition in repetitions]
        online = [float(row["online_seconds"]) for row in rows]
        result["methods"][method] = {
            "score": median([float(row["score"]) for row in rows]),
            "generated_tokens": median(
                [float(row["generated_tokens"]) for row in rows]
            ),
            "attention_fraction": median(
                [float(row["configured_attention_fraction"]) for row in rows]
            ),
            "online_seconds": median(online),
            "query_seconds": median(
                [float(row["query_seconds"]) for row in rows]
            ),
            "decode_seconds": median(
                [float(row["decode_seconds"]) for row in rows]
            ),
            "total_seconds": median(
                [float(row["total_seconds"]) for row in rows]
            ),
            "paired_online_speedup": median(
                [
                    full_value / method_value
                    for full_value, method_value in zip(full_online, online)
                ]
            ),
        }
    return result


def fit_cost(cases: list[dict[str, Any]], method: str) -> dict[str, float | None]:
    ordered = sorted(cases, key=lambda case: case["methods"][method]["generated_tokens"])
    low, high = ordered[0], ordered[-1]
    low_tokens = low["methods"][method]["generated_tokens"]
    high_tokens = high["methods"][method]["generated_tokens"]
    step_seconds = (
        high["methods"][method]["online_seconds"]
        - low["methods"][method]["online_seconds"]
    ) / (high_tokens - low_tokens)
    fixed_seconds = (
        low["methods"][method]["online_seconds"]
        - max(0.0, low_tokens - 1.0) * step_seconds
    )
    return {
        "fixed_seconds": fixed_seconds,
        "step_ms": 1000.0 * step_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    for length_dir in sorted(
        args.run_root.glob("length*"),
        key=lambda path: int(path.name.removeprefix("length")),
    ):
        for case_dir in sorted(
            length_dir.glob("g*"),
            key=lambda path: int(path.name.removeprefix("g")),
        ):
            case = summarize_case(case_dir)
            case["configured_prompt_tokens"] = int(
                length_dir.name.removeprefix("length")
            )
            cases.append(case)

    fits: dict[str, dict[str, Any]] = {}
    for length in sorted({case["configured_prompt_tokens"] for case in cases}):
        length_cases = [
            case for case in cases if case["configured_prompt_tokens"] == length
        ]
        fits[str(length)] = {
            method: fit_cost(length_cases, method) for method in METHODS
        }
        full = fits[str(length)]["full_kv"]
        fused = fits[str(length)]["countcap_fullprompt_keypca_direct_fused"]
        denominator = float(full["step_ms"]) - float(fused["step_ms"])
        crossing = None
        if denominator > 0.0:
            crossing = 1.0 + 1000.0 * (
                float(fused["fixed_seconds"]) - float(full["fixed_seconds"])
            ) / denominator
        fits[str(length)]["fused_break_even_tokens"] = crossing

    result = {
        "protocol": "same GovReport sample, three paired repetitions",
        "methods": list(METHODS),
        "cases": cases,
        "cost_fits": fits,
    }
    (args.run_root / "summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
