from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from analyze_residual_certified_pca_20260717 import (
    exact_rerank,
    selection_metrics,
    summarize,
)
from analyze_verify_then_expand_pca_20260717 import (
    load_records,
    parse_float_list,
    record_head_terms,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training-free progressive PCA retrieval using attention stability."
    )
    parser.add_argument("--trace_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument(
        "--candidate_fractions", default="0.02,0.0225,0.025,0.03,0.04,0.06,0.08"
    )
    parser.add_argument("--tv_thresholds", default="0.001,0.0025,0.005,0.01,0.02")
    parser.add_argument("--stable_steps", default="1,2")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sparse_probability(
    exact_scores: torch.Tensor, selected: torch.Tensor
) -> torch.Tensor:
    probability = torch.zeros_like(exact_scores, dtype=torch.float32)
    probability[selected] = torch.softmax(exact_scores[selected].float(), dim=0)
    return probability


def total_variation(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(0.5 * (left - right).abs().sum().item())


def progressive_stability_select(
    exact: torch.Tensor,
    approximate: torch.Tensor,
    top_count: int,
    fractions: tuple[float, ...],
    tv_threshold: float,
    required_stable_steps: int,
) -> tuple[torch.Tensor, float, list[float]]:
    if required_stable_steps <= 0:
        raise ValueError("required_stable_steps must be positive")
    history_count = int(exact.numel())
    approximate_order = torch.argsort(approximate, descending=True)
    previous_probability = None
    stable_steps = 0
    transition_tv: list[float] = []
    selected = None
    chosen_ratio = fractions[-1]
    for fraction in fractions:
        candidate_count = min(
            history_count, max(top_count, math.ceil(fraction * history_count))
        )
        selected = exact_rerank(
            exact, approximate_order[:candidate_count], top_count
        )
        probability = sparse_probability(exact, selected)
        if previous_probability is not None:
            change = total_variation(previous_probability, probability)
            transition_tv.append(change)
            stable_steps = stable_steps + 1 if change <= tv_threshold else 0
            if stable_steps >= required_stable_steps:
                chosen_ratio = candidate_count / history_count
                break
        previous_probability = probability
        chosen_ratio = candidate_count / history_count
    assert selected is not None
    return selected, chosen_ratio, transition_tv


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    fractions = parse_float_list(args.candidate_fractions)
    thresholds = parse_float_list(args.tv_thresholds)
    stable_steps_values = tuple(
        sorted({int(value) for value in args.stable_steps.split(",") if value.strip()})
    )
    if fractions[0] < args.top_fraction or fractions[-1] > 1.0:
        raise ValueError("candidate fractions must lie in [top_fraction, 1]")
    if thresholds[0] <= 0.0 or thresholds[-1] >= 1.0:
        raise ValueError("TV thresholds must lie in (0, 1)")
    if not stable_steps_values or stable_steps_values[0] <= 0:
        raise ValueError("stable_steps must contain positive integers")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    records = load_records(args.trace_path)
    rows: list[dict[str, Any]] = []

    for record_index, record in enumerate(records):
        for terms in record_head_terms(
            record, args.rank, args.sample_stride, device
        ):
            history_count = int(terms["history_count"])
            top_count = max(1, math.ceil(args.top_fraction * history_count))
            true_indices = torch.topk(terms["exact"], k=top_count).indices
            oracle_mass = float(terms["attention"][true_indices].sum().item())
            oracle_probability = sparse_probability(terms["exact"], true_indices)
            for threshold in thresholds:
                for required_steps in stable_steps_values:
                    selected, candidate_ratio, transition_tv = (
                        progressive_stability_select(
                            terms["exact"],
                            terms["approximate"],
                            top_count,
                            fractions,
                            threshold,
                            required_steps,
                        )
                    )
                    selected_probability = sparse_probability(
                        terms["exact"], selected
                    )
                    rows.append(
                        {
                            "topic": args.topic,
                            "record_index": record_index,
                            "layer": int(terms["layer"]),
                            "query_head": int(terms["query_head"]),
                            "tv_threshold": threshold,
                            "required_stable_steps": required_steps,
                            "candidate_ratio": candidate_ratio,
                            "tv_to_exact_top2": total_variation(
                                selected_probability, oracle_probability
                            ),
                            "last_transition_tv": (
                                transition_tv[-1] if transition_tv else 0.0
                            ),
                            "transitions": len(transition_tv),
                            **selection_metrics(
                                selected,
                                true_indices,
                                terms["attention"],
                                oracle_mass,
                            ),
                        }
                    )
        print(
            f"topic={args.topic} record={record_index + 1}/{len(records)} "
            f"layer={int(record['layer'])}",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summaries = []
    for threshold in thresholds:
        for required_steps in stable_steps_values:
            items = [
                row
                for row in rows
                if row["tv_threshold"] == threshold
                and row["required_stable_steps"] == required_steps
            ]
            candidate_values = [float(row["candidate_ratio"]) for row in items]
            recall_values = [float(row["top2_recall"]) for row in items]
            mass_values = [
                float(row["top2_attention_mass_recall"]) for row in items
            ]
            oracle_tv_values = [float(row["tv_to_exact_top2"]) for row in items]
            histogram = Counter(round(value, 6) for value in candidate_values)
            summaries.append(
                {
                    "tv_threshold": threshold,
                    "required_stable_steps": required_steps,
                    "cases": len(items),
                    "candidate_ratio": summarize(candidate_values),
                    "candidate_histogram": {
                        str(key): value for key, value in sorted(histogram.items())
                    },
                    "top2_recall": summarize(recall_values),
                    "top2_attention_mass_recall": summarize(mass_values),
                    "tv_to_exact_top2": summarize(oracle_tv_values),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_head_query.csv", rows)
    report = {
        "method": "progressive_attention_stability",
        "topic": args.topic,
        "rank": args.rank,
        "top_fraction": args.top_fraction,
        "candidate_fractions": fractions,
        "summaries": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
