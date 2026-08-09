from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Iterable[float]) -> float:
    return statistics.fmean(values)


def bootstrap_interval(
    values: list[float],
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    samples = sorted(
        mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(iterations)
    )
    return (
        samples[round(0.025 * (len(samples) - 1))],
        samples[round(0.975 * (len(samples) - 1))],
    )


def load_rows(paths: list[str]) -> list[dict[str, Any]]:
    keyed: dict[tuple[int, int, str], dict[str, Any]] = {}
    for text in paths:
        path = Path(text)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                int(row["target_context_tokens"]),
                int(row["seed"]),
                str(row["variant"]),
            )
            if key in keyed:
                raise ValueError(f"duplicate row: {key}")
            keyed[key] = row
    return [keyed[key] for key in sorted(keyed)]


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (int(row["target_context_tokens"]), str(row["variant"]))
        ].append(row)
    output = []
    metric_fields = (
        "gold_evidence_token_recall",
        "gold_chain_complete_rate",
        "gold_evidence_attention_mass",
        "next_token_correct",
        "query_seconds",
        "semantic_positive_gap_fraction",
        "semantic_positive_gap_mean",
        "semantic_positive_gap_max_mean",
        "semantic_rescue_head_fraction",
    )
    for (length, variant), items in sorted(grouped.items()):
        mean_nll = mean(float(item["gold_nll"]) for item in items)
        result: dict[str, Any] = {
            "target_context_tokens": length,
            "variant": variant,
            "sample_count": len(items),
            "mean_gold_nll": mean_nll,
            "gold_ppl": math.exp(mean_nll),
        }
        for field in metric_fields:
            if all(field in item for item in items):
                result[field] = mean(float(item[field]) for item in items)
        output.append(result)
    return output


def paired_deltas(
    rows: list[dict[str, Any]],
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_case = {
        (
            int(row["target_context_tokens"]),
            int(row["seed"]),
            str(row["variant"]),
        ): row
        for row in rows
    }
    lengths = sorted({key[0] for key in by_case})
    variants = sorted({key[2] for key in by_case})
    seeds = sorted({key[1] for key in by_case})
    output = []
    for baseline in ("full_rope", "rope_top2"):
        if baseline not in variants:
            continue
        for length in lengths:
            for variant in variants:
                if variant == baseline:
                    continue
                deltas = []
                for case_seed in seeds:
                    left = by_case.get((length, case_seed, variant))
                    right = by_case.get((length, case_seed, baseline))
                    if left is not None and right is not None:
                        deltas.append(
                            float(left["gold_nll"]) - float(right["gold_nll"])
                        )
                if not deltas:
                    continue
                low, high = bootstrap_interval(
                    deltas,
                    iterations,
                    seed
                    + length
                    + 997 * variants.index(variant)
                    + 7919 * variants.index(baseline),
                )
                output.append(
                    {
                        "baseline": baseline,
                        "target_context_tokens": length,
                        "variant": variant,
                        "sample_count": len(deltas),
                        "mean_delta_nll": mean(deltas),
                        "quality_ratio": math.exp(-mean(deltas)),
                        "delta_nll_ci_low": low,
                        "delta_nll_ci_high": high,
                        "improved_sample_fraction": mean(
                            float(value < 0.0) for value in deltas
                        ),
                    }
                )
    return output


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_jsonl)
    summary = aggregate(rows)
    deltas = paired_deltas(
        rows,
        args.bootstrap_iterations,
        args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "rows.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(args.output_dir / "rows.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "paired_deltas.csv", deltas)
    payload = {
        "source_files": args.input_jsonl,
        "row_count": len(rows),
        "summary": summary,
        "paired_deltas": deltas,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
