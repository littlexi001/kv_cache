from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze fixed-page XSum order ablation.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def paired(
    baseline: dict[int, dict[str, Any]],
    treatment: dict[int, dict[str, Any]],
    ids: list[int],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    base = np.asarray([baseline[query_id]["mean_nll"] for query_id in ids])
    new = np.asarray([treatment[query_id]["mean_nll"] for query_id in ids])
    difference = new - base
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(samples, len(ids)))
    sampled = difference[draws].mean(axis=1)
    return {
        "queries": len(ids),
        "baseline_mean_nll": float(base.mean()),
        "treatment_mean_nll": float(new.mean()),
        "treatment_minus_baseline_nll": float(difference.mean()),
        "bootstrap_95_ci": [
            float(np.quantile(sampled, 0.025)),
            float(np.quantile(sampled, 0.975)),
        ],
        "wins": int((difference < 0).sum()),
        "losses": int((difference > 0).sum()),
        "ties": int((difference == 0).sum()),
        "ppl_ratio": math.exp(float(difference.mean())),
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    rows = [
        row
        for rank in range(args.world_size)
        for row in read_jsonl(input_dir / f"rows_rank{rank:03d}.jsonl")
    ]
    by_order: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_order[str(row["order"])][int(row["query_id"])] = row
    ids = sorted(by_order["original_old_to_new"])
    if len(ids) != 100 or any(set(group) != set(ids) for group in by_order.values()):
        raise RuntimeError("expected three orders over the same 100 queries")
    quality = {}
    for order, group in sorted(by_order.items()):
        total_nll = sum(float(group[query_id]["total_nll"]) for query_id in ids)
        total_tokens = sum(int(group[query_id]["target_tokens"]) for query_id in ids)
        micro_nll = total_nll / total_tokens
        quality[order] = {
            "queries": len(ids),
            "micro_nll": micro_nll,
            "ppl": math.exp(micro_nll),
            "mean_forward_seconds": float(
                np.mean([group[query_id]["forward_seconds"] for query_id in ids])
            ),
        }
    output = {
        "protocol": {
            "memory_tokens": 10_000_000,
            "fixed_retrieval_method": rows[0]["method"],
            "fixed_selected_pages": True,
            "selected_tokens": 512,
            "selection_uses_target": False,
            "queries": len(ids),
        },
        "quality": quality,
        "versus_original_old_to_new": {
            order: paired(
                by_order["original_old_to_new"],
                by_order[order],
                ids,
                samples=args.bootstrap_samples,
                seed=args.seed + index,
            )
            for index, order in enumerate(
                ("reverse_new_to_old", "retrieval_score_order")
            )
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
