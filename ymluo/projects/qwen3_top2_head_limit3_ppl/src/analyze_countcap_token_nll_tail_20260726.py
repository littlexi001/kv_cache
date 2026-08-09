#!/usr/bin/env python3
"""Decompose paired Full/CountCap PPL differences into token-level tails."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def paired_deltas(paths: list[Path]) -> list[dict[str, Any]]:
    paired: dict[
        tuple[int, str, str, int, int],
        dict[str, dict[str, str]],
    ] = (
        defaultdict(dict)
    )
    for path in paths:
        match = re.search(r"length(\d+)", path.parent.name)
        if match is None:
            raise ValueError(f"Cannot infer history length from {path}")
        history_tokens = int(match.group(1))
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (
                    history_tokens,
                    row["topic"],
                    row["window"],
                    int(row["target_index"]),
                    int(row["token_id"]),
                )
                paired[key][row["method"]] = row

    output = []
    for (
        history_tokens,
        topic,
        window,
        target_index,
        token_id,
    ), methods in sorted(
        paired.items()
    ):
        if set(methods) != {"full_attention", "direct_countcap"}:
            raise ValueError(
                f"unpaired token {topic}/{window}/{target_index}: "
                f"{sorted(methods)}"
            )
        full_nll = float(methods["full_attention"]["nll"])
        direct_nll = float(methods["direct_countcap"]["nll"])
        output.append(
            {
                "history_tokens": history_tokens,
                "topic": topic,
                "window": int(window),
                "target_index": target_index,
                "token_id": token_id,
                "full_nll": full_nll,
                "direct_nll": direct_nll,
                "delta_nll": direct_nll - full_nll,
            }
        )
    return output


def top_fraction_share(positive: np.ndarray, fraction: float) -> float:
    if positive.size == 0 or float(positive.sum()) == 0.0:
        return 0.0
    count = max(1, math.ceil(fraction * positive.size))
    return float(np.sort(positive)[-count:].sum() / positive.sum())


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    delta = np.asarray([row["delta_nll"] for row in rows], dtype=np.float64)
    positive = delta[delta > 0.0]
    worst = sorted(rows, key=lambda row: row["delta_nll"], reverse=True)[:20]
    return {
        "tokens": len(rows),
        "mean_delta_nll": float(delta.mean()),
        "ppl_ratio": float(math.exp(delta.mean())),
        "median_delta_nll": float(np.median(delta)),
        "p90_delta_nll": float(np.quantile(delta, 0.90)),
        "p95_delta_nll": float(np.quantile(delta, 0.95)),
        "p99_delta_nll": float(np.quantile(delta, 0.99)),
        "positive_delta_fraction": float(np.mean(delta > 0.0)),
        "delta_gt_1_fraction": float(np.mean(delta > 1.0)),
        "delta_gt_2_fraction": float(np.mean(delta > 2.0)),
        "positive_loss_top1pct_share": top_fraction_share(positive, 0.01),
        "positive_loss_top5pct_share": top_fraction_share(positive, 0.05),
        "positive_loss_top10pct_share": top_fraction_share(positive, 0.10),
        "worst_tokens": worst,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.run_root.glob("length*/token_results.csv"))
    if not paths:
        raise FileNotFoundError(f"No token_results.csv below {args.run_root}")
    rows = paired_deltas(paths)
    grouped: dict[
        tuple[int, str, int],
        list[dict[str, Any]],
    ] = defaultdict(list)
    by_history: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        history_tokens = int(row["history_tokens"])
        grouped[
            (history_tokens, row["topic"], row["window"])
        ].append(row)
        by_history[history_tokens].append(row)

    output = {
        "overall": summarize(rows),
        "by_history": [
            {
                "history_tokens": history_tokens,
                **summarize(history_rows),
            }
            for history_tokens, history_rows in sorted(by_history.items())
        ],
        "by_case": [
            {
                "history_tokens": history_tokens,
                "topic": topic,
                "window": window,
                **summarize(case_rows),
            }
            for (
                history_tokens,
                topic,
                window,
            ), case_rows in sorted(grouped.items())
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output_json)


if __name__ == "__main__":
    main()
