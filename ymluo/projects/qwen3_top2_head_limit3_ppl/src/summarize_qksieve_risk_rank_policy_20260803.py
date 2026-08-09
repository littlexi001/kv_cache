#!/usr/bin/env python
"""Evaluate a numerical Value-rank policy on matched real-QKV rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch


IDENTITY = ("trace", "history_tokens", "layer", "kv_head", "query_head")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rank_csvs",
        required=True,
        help="Comma-separated rank:path pairs.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--risk_thresholds",
        default="0.00025,0.0005,0.00075,0.001,0.0015,0.002",
    )
    parser.add_argument(
        "--relative_risk_thresholds",
        default="0.0025,0.005,0.0075,0.01,0.02,0.03",
    )
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--key_index_bits", type=float, default=240.0)
    parser.add_argument("--full_kv_bits", type=float, default=4096.0)
    parser.add_argument("--value_bits", type=float, default=4.0)
    parser.add_argument("--value_scale_bits", type=float, default=16.0)
    parser.add_argument("--value_scale_block", type=int, default=256)
    return parser.parse_args()


def parse_rank_csvs(specification: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for item in specification.split(","):
        rank_text, path_text = item.split(":", 1)
        result[int(rank_text)] = Path(path_text)
    if not result:
        raise ValueError("at least one rank CSV is required")
    return dict(sorted(result.items()))


def summarize(values: Iterable[float]) -> dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "p99": float(torch.quantile(tensor, 0.99)),
        "maximum": float(tensor.max()),
    }


def pearson(rows: list[dict[str, Any]], left: str, right: str) -> float:
    values = torch.tensor(
        [(float(row[left]), float(row[right])) for row in rows],
        dtype=torch.float64,
    )
    values -= values.mean(dim=0, keepdim=True)
    denominator = torch.sqrt(values.square().sum(dim=0).prod())
    return float((values[:, 0] * values[:, 1]).sum() / denominator)


def load_rows(
    rank_csvs: dict[int, Path], block_size: int
) -> dict[tuple[str, ...], dict[int, dict[str, Any]]]:
    matched: dict[tuple[str, ...], dict[int, dict[str, Any]]] = defaultdict(dict)
    for rank, path in rank_csvs.items():
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["candidate_mode"] != "proxy":
                    continue
                if row["method"] != "block_residual_mean_proxy":
                    continue
                if int(row["block_size"]) != block_size:
                    continue
                identity = tuple(row[field] for field in IDENTITY)
                matched[identity][rank] = row
    required = set(rank_csvs)
    incomplete = [identity for identity, rows in matched.items() if set(rows) != required]
    if incomplete:
        raise RuntimeError(f"{len(incomplete)} identities lack matched ranks")
    return matched


def metrics(
    selected: list[tuple[tuple[str, ...], int, dict[str, Any]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    query_ranks = [rank for _, rank, _ in selected]
    per_kv_head: dict[tuple[str, ...], int] = {}
    for identity, rank, _ in selected:
        kv_identity = identity[:3] + (identity[3],)
        per_kv_head[kv_identity] = max(per_kv_head.get(kv_identity, 0), rank)
    kv_ranks = list(per_kv_head.values())
    value_bits_per_dimension = args.value_bits + (
        args.value_scale_bits / args.value_scale_block
    )
    block_residual_bits_per_token = (
        128.0 * 16.0 / float(args.block_size)
    )
    index_bits = (
        args.key_index_bits
        + value_bits_per_dimension * (sum(kv_ranks) / len(kv_ranks))
        + block_residual_bits_per_token
    )
    result: dict[str, Any] = {
        "query_count": len(selected),
        "kv_head_count": len(kv_ranks),
        "query_rank_mean": sum(query_ranks) / len(query_ranks),
        "kv_head_stored_rank_mean": sum(kv_ranks) / len(kv_ranks),
        "query_rank_histogram": dict(sorted(Counter(query_ranks).items())),
        "kv_head_rank_histogram": dict(sorted(Counter(kv_ranks).items())),
        "auxiliary_index_ratio": index_bits / args.full_kv_bits,
    }
    for field in (
        "absolute_l2",
        "relative_l2",
        "residual_risk_absolute",
        "residual_risk_relative",
    ):
        for statistic, value in summarize(
            float(row[field]) for _, _, row in selected
        ).items():
            result[f"{field}_{statistic}"] = value
    return result


def main() -> None:
    args = parse_args()
    rank_csvs = parse_rank_csvs(args.rank_csvs)
    thresholds = tuple(
        sorted({float(value) for value in args.risk_thresholds.split(",")})
    )
    relative_thresholds = tuple(
        sorted(
            {
                float(value)
                for value in args.relative_risk_thresholds.split(",")
            }
        )
    )
    matched = load_rows(rank_csvs, args.block_size)
    ranks = tuple(rank_csvs)

    fixed = []
    for rank in ranks:
        selected = [
            (identity, rank, rows[rank]) for identity, rows in matched.items()
        ]
        fixed.append({"rank": rank, **metrics(selected, args)})

    policies = []
    for threshold in thresholds:
        selected = []
        for identity, rows in matched.items():
            rank = next(
                (
                    candidate
                    for candidate in ranks
                    if float(rows[candidate]["residual_risk_absolute"])
                    <= threshold
                ),
                ranks[-1],
            )
            selected.append((identity, rank, rows[rank]))
        policies.append(
            {
                "risk_threshold_absolute": threshold,
                **metrics(selected, args),
            }
        )

    relative_policies = []
    for threshold in relative_thresholds:
        selected = []
        for identity, rows in matched.items():
            rank = next(
                (
                    candidate
                    for candidate in ranks
                    if float(rows[candidate]["residual_risk_relative"])
                    <= threshold
                ),
                ranks[-1],
            )
            selected.append((identity, rank, rows[rank]))
        relative_policies.append(
            {
                "risk_threshold_relative": threshold,
                **metrics(selected, args),
            }
        )

    all_rows = [
        row for rows in matched.values() for row in rows.values()
    ]
    report = {
        "schema": "qksieve_risk_rank_policy_v1",
        "rank_csvs": {str(rank): str(path) for rank, path in rank_csvs.items()},
        "block_size": args.block_size,
        "matched_queries": len(matched),
        "risk_correlations": {
            "absolute": pearson(
                all_rows, "residual_risk_absolute", "absolute_l2"
            ),
            "relative": pearson(
                all_rows, "residual_risk_relative", "relative_l2"
            ),
        },
        "fixed_rank": fixed,
        "risk_policy": policies,
        "relative_risk_policy": relative_policies,
        "claim_boundary": (
            "Matched real-QKV local-output audit. Threshold policies are "
            "diagnostic until evaluated on independent traces and model-level PPL."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
