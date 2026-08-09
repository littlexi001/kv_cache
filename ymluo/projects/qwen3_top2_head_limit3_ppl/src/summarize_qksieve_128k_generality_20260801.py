#!/usr/bin/env python
"""Aggregate matched native-128K QKSieve PPL experiments."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--bootstrap_replicates", type=int, default=20000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260801)
    return parser.parse_args()


def experiment_label(path: Path, variant: str, row: dict[str, Any]) -> str:
    alpha = row.get("packed_value_sketch_tail_alpha")
    if alpha is None:
        match = re.search(r"_alpha(\d+p\d+)_", path.as_posix())
        alpha = float(match.group(1).replace("p", ".")) if match else None
    rank_match = re.search(r"valuesketch(8|16|32)i4", variant)
    if rank_match:
        return f"value-r{rank_match.group(1)}-alpha{float(alpha):g}"
    if "sampled" in variant:
        return "ordinary-qksieve"
    return variant


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for root in args.roots:
        for name in sorted(glob.glob(str(root / "*" / "summary.json"))):
            path = Path(name)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload_rows = payload.get("rows")
            if not isinstance(payload_rows, list) or not any(
                isinstance(row, dict) and row.get("method") == "full_attention"
                for row in payload_rows
            ):
                continue
            full = next(
                row for row in payload_rows if row.get("method") == "full_attention"
            )
            for sparse in payload_rows:
                if not isinstance(sparse, dict) or "method" not in sparse:
                    continue
                if sparse["method"] == "full_attention":
                    continue
                variant = str(sparse.get("variant", sparse["method"]))
                label = experiment_label(path, variant, sparse)
                key = (label, str(payload["topic"]), int(payload["seed"]))
                if key in seen:
                    raise ValueError(f"duplicate matched case: {key}")
                seen.add(key)
                rows.append(
                    {
                        "experiment": label,
                        "topic": payload["topic"],
                        "seed": payload["seed"],
                        "variant": variant,
                        "tokens": sparse["tokens"],
                        "full_nll": full["nll"],
                        "sparse_nll": sparse["nll"],
                        "full_ppl": full["ppl"],
                        "sparse_ppl": sparse["ppl"],
                        "ppl_retention_pct": 100.0 * full["ppl"] / sparse["ppl"],
                        "steady_speedup": (
                            full["steady_sparse_seconds_per_step"]
                            / sparse["steady_sparse_seconds_per_step"]
                        ),
                        "amortized_64_token_speedup": (
                            full["sparse_seconds_per_step"]
                            / sparse["sparse_seconds_per_step"]
                        ),
                        "active_tokens_mean": sparse["actual_attention_tokens_mean"],
                        "auxiliary_ratio_pct": 100.0
                        * sparse["packed_total_auxiliary_ratio_of_full_kv"],
                        "source": str(path),
                    }
                )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["experiment"]].append(row)
    summaries: list[dict[str, Any]] = []
    for label, items in sorted(grouped.items()):
        full_nll = sum(float(item["full_nll"]) for item in items) / len(items)
        sparse_nll = sum(float(item["sparse_nll"]) for item in items) / len(items)
        retentions = [float(item["ppl_retention_pct"]) for item in items]
        rng = random.Random(args.bootstrap_seed)
        pooled_bootstrap: list[float] = []
        macro_bootstrap: list[float] = []
        for _ in range(args.bootstrap_replicates):
            sampled = [items[rng.randrange(len(items))] for _ in items]
            sampled_full_nll = sum(float(item["full_nll"]) for item in sampled) / len(sampled)
            sampled_sparse_nll = sum(float(item["sparse_nll"]) for item in sampled) / len(sampled)
            pooled_bootstrap.append(
                100.0 * math.exp(sampled_full_nll - sampled_sparse_nll)
            )
            macro_bootstrap.append(
                sum(float(item["ppl_retention_pct"]) for item in sampled)
                / len(sampled)
            )
        summaries.append(
            {
                "experiment": label,
                "topic_count": len(items),
                "pooled_full_ppl": math.exp(full_nll),
                "pooled_sparse_ppl": math.exp(sparse_nll),
                "pooled_ppl_retention_pct": 100.0 * math.exp(full_nll - sparse_nll),
                "pooled_ppl_retention_ci95_low": percentile(pooled_bootstrap, 0.025),
                "pooled_ppl_retention_ci95_high": percentile(pooled_bootstrap, 0.975),
                "macro_ppl_retention_pct": sum(retentions) / len(retentions),
                "macro_ppl_retention_ci95_low": percentile(macro_bootstrap, 0.025),
                "macro_ppl_retention_ci95_high": percentile(macro_bootstrap, 0.975),
                "worst_ppl_retention_pct": min(retentions),
                "steady_speedup_geomean": geometric_mean(
                    [float(item["steady_speedup"]) for item in items]
                ),
                "amortized_64_token_speedup_geomean": geometric_mean(
                    [float(item["amortized_64_token_speedup"]) for item in items]
                ),
                "auxiliary_ratio_pct_mean": sum(
                    float(item["auxiliary_ratio_pct"]) for item in items
                )
                / len(items),
            }
        )

    def write_csv(path: Path, payload: list[dict[str, Any]]) -> None:
        if not payload:
            return
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(payload[0]))
            writer.writeheader()
            writer.writerows(payload)

    write_csv(args.output_dir / "per_topic.csv", rows)
    write_csv(args.output_dir / "aggregate.csv", summaries)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {"schema": "qksieve-128k-generality-v1", "rows": rows, "summary": summaries},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
