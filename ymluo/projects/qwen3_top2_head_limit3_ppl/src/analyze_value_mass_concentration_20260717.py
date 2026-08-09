from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure how many top-attention values carry retained mass."
    )
    parser.add_argument("--trace_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--attention_fraction", type=float, default=0.02)
    parser.add_argument("--mass_thresholds", default="0.95,0.99,0.995,0.999")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": float(array.max()),
    }


@torch.inference_mode()
def analyze_trace(
    trace_path: Path,
    attention_fraction: float,
    thresholds: tuple[float, ...],
    device: torch.device,
) -> dict[str, object]:
    payload = torch.load(trace_path, map_location="cpu", weights_only=False)
    fractions_by_threshold = {threshold: [] for threshold in thresholds}
    counts_by_threshold = {threshold: [] for threshold in thresholds}
    bounded_fractions_by_threshold = {threshold: [] for threshold in thresholds}
    bounded_mass_by_threshold = {threshold: [] for threshold in thresholds}
    retained_counts = []
    for record in payload["records"]:
        query = record["query"].to(device).float()[0, :, 0, :]
        key = record["key"].to(device).float()[0]
        history_key = key[:, :-1, :]
        current_key = key[:, -1, :]
        query_heads = int(query.shape[0])
        kv_heads = int(history_key.shape[0])
        groups = query_heads // kv_heads
        expanded_history = history_key.repeat_interleave(groups, dim=0)
        expanded_current = current_key.repeat_interleave(groups, dim=0)
        history_count = int(history_key.shape[1])
        retained_count = max(1, math.ceil(attention_fraction * history_count))
        retained_counts.extend([retained_count] * query_heads)
        scaling = float(record["scaling"])
        exact_scores = torch.einsum(
            "hkd,hd->hk", expanded_history, query
        ) * scaling
        retained_scores = torch.topk(
            exact_scores,
            k=retained_count,
            dim=-1,
            largest=True,
            sorted=True,
        ).values
        self_scores = (query * expanded_current).sum(dim=-1) * scaling
        max_scores = torch.maximum(retained_scores[:, 0], self_scores)
        retained_weights = torch.exp(retained_scores - max_scores.unsqueeze(-1))
        self_weights = torch.exp(self_scores - max_scores)
        total_weights = retained_weights.sum(dim=-1) + self_weights
        cumulative = retained_weights.cumsum(dim=-1) + self_weights.unsqueeze(-1)
        for threshold in thresholds:
            reached = cumulative >= threshold * total_weights.unsqueeze(-1)
            required = reached.float().argmax(dim=-1) + 1
            missing = ~reached.any(dim=-1)
            required = torch.where(
                missing,
                torch.full_like(required, retained_count),
                required,
            )
            counts = required.cpu().tolist()
            counts_by_threshold[threshold].extend(float(value) for value in counts)
            fractions_by_threshold[threshold].extend(
                float(value) / retained_count for value in counts
            )
            safe_cutoff = (
                (1.0 - threshold)
                * total_weights
                / float(retained_count + 1)
            )
            bounded_keep = retained_weights >= safe_cutoff.unsqueeze(-1)
            bounded_count = bounded_keep.sum(dim=-1)
            bounded_mass = (
                (retained_weights * bounded_keep).sum(dim=-1) + self_weights
            ) / total_weights
            bounded_fractions_by_threshold[threshold].extend(
                float(value) / retained_count
                for value in bounded_count.cpu().tolist()
            )
            bounded_mass_by_threshold[threshold].extend(
                float(value) for value in bounded_mass.cpu().tolist()
            )
        del expanded_history, exact_scores, retained_scores, retained_weights
        torch.cuda.empty_cache()
    return {
        "trace": trace_path.stem,
        "head_examples": len(retained_counts),
        "retained_count": summarize([float(value) for value in retained_counts]),
        "thresholds": {
            str(threshold): {
                "fraction_of_top2_values_read": summarize(
                    fractions_by_threshold[threshold]
                ),
                "value_count": summarize(counts_by_threshold[threshold]),
                "bounded_cutoff_fraction_of_top2_values_read": summarize(
                    bounded_fractions_by_threshold[threshold]
                ),
                "bounded_cutoff_retained_mass": summarize(
                    bounded_mass_by_threshold[threshold]
                ),
            }
            for threshold in thresholds
        },
    }


def main() -> None:
    args = parse_args()
    thresholds = tuple(
        float(value.strip()) for value in args.mass_thresholds.split(",")
    )
    if not thresholds or any(not 0.0 < value <= 1.0 for value in thresholds):
        raise ValueError("mass thresholds must be in (0, 1]")
    device = torch.device(args.device)
    traces = [
        analyze_trace(
            path,
            args.attention_fraction,
            thresholds,
            device,
        )
        for path in args.trace_paths
    ]
    payload = {
        "attention_fraction": args.attention_fraction,
        "traces": traces,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
