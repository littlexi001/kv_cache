#!/usr/bin/env python
"""Audit temporal access-frequency sparsity in QKSieve retrieval.

The experiment is deliberately trace based.  It measures whether the tokens
selected by one Query head form a persistent hot set and evaluates a causal
two-speed selector:

* refresh steps scan the complete QKSieve index;
* reuse steps score only the previous selection plus a historical hot set;
* exact K/V attention is still evaluated on the final sparse selection.

No future selection, task label, learned router, or oracle frequency is used by
the proposed selector.  Exact scores are used only to report held-out quality.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from run_head_top2_targeted_ppl_20260714 import (
    _hierarchical_qmse_rate_allocation,
    _hierarchical_quantize_band,
    _qk_metric_projection_factors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibration_steps", type=int, default=8)
    parser.add_argument("--query_shrinkage", type=float, default=0.75)
    parser.add_argument("--key_stride", type=int, default=32)
    parser.add_argument("--base_fraction", type=float, default=0.02)
    parser.add_argument(
        "--refresh_intervals",
        default="2,4,8",
        help="Comma-separated full-index refresh intervals.",
    )
    parser.add_argument(
        "--hot_fractions",
        default="0.0025,0.005,0.01",
        help="Causal frequency-hot-set sizes as fractions of history.",
    )
    parser.add_argument(
        "--output_fractions",
        default="0.01,0.015,0.02",
        help="Final exact-attention budgets.",
    )
    parser.add_argument(
        "--frequency_decay",
        type=float,
        default=0.95,
        help="EWMA decay applied before every observed QKSieve selection.",
    )
    return parser.parse_args()


def parse_numbers(text: str, cast: Any) -> tuple[Any, ...]:
    return tuple(cast(item) for item in text.split(",") if item.strip())


def second_moment(values: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhnd,bhne->bhde", values, values) / max(
        1, values.shape[-2]
    )


def mapped_queries(records: list[dict[str, Any]]) -> torch.Tensor:
    queries = torch.cat(
        [record["query"].float() for record in records], dim=2
    )
    batch, query_heads, steps, head_dim = queries.shape
    kv_heads = int(records[0]["key"].shape[1])
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    groups = query_heads // kv_heads
    return (
        queries.reshape(batch, kv_heads, groups, steps, head_dim)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch, kv_heads, steps * groups, head_dim)
        .contiguous()
    )


def qksieve_reconstruct(
    projected_key: torch.Tensor,
    allocation: torch.Tensor,
) -> torch.Tensor:
    reconstructed = torch.empty_like(projected_key)
    for head in range(projected_key.shape[1]):
        for band in range(8):
            start = 16 * band
            bits = int(allocation[0, head, band].item())
            reconstructed[:, head, :, start : start + 16] = (
                _hierarchical_quantize_band(
                    projected_key[:, head, :, start : start + 16],
                    bits,
                )
            )
    return reconstructed


def flatten_query_heads(
    query: torch.Tensor,
    kv_heads: int,
) -> torch.Tensor:
    batch, query_heads, _, head_dim = query.shape
    if batch != 1 or query_heads % kv_heads:
        raise ValueError("unsupported Query trace shape")
    groups = query_heads // kv_heads
    return query[:, :, 0, :].reshape(kv_heads, groups, head_dim)


def selected_metrics(
    probabilities: torch.Tensor,
    selected: torch.Tensor,
    exact_mask: torch.Tensor,
    exact_top_count: int,
) -> dict[str, float]:
    return {
        "attention_mass": float(
            probabilities.gather(1, selected).sum(dim=1).mean().item()
        ),
        "oracle_top2_recall": float(
            exact_mask.gather(1, selected).float().sum(dim=1).mean().item()
            / max(1, exact_top_count)
        ),
    }


def hot_indices(
    frequency: torch.Tensor,
    count: int,
    observations: int,
) -> torch.Tensor:
    if observations <= 0 or count <= 0:
        return torch.empty(
            frequency.shape[0],
            0,
            dtype=torch.long,
            device=frequency.device,
        )
    return torch.topk(frequency, k=count, dim=1, sorted=False).indices


def shortlist_select(
    proxy_scores: torch.Tensor,
    previous: torch.Tensor,
    hot: torch.Tensor,
    output_count: int,
) -> tuple[torch.Tensor, float]:
    candidate_mask = torch.zeros_like(proxy_scores, dtype=torch.bool)
    candidate_mask.scatter_(1, previous, True)
    if hot.numel():
        candidate_mask.scatter_(1, hot, True)
    pool_count = float(candidate_mask.sum(dim=1).float().mean().item())
    masked_scores = proxy_scores.masked_fill(~candidate_mask, -torch.inf)
    available = int(candidate_mask.sum(dim=1).min().item())
    selected = torch.topk(
        masked_scores,
        k=min(output_count, available),
        dim=1,
        sorted=False,
    ).indices
    return selected, pool_count


def concentration_metrics(
    frequency: torch.Tensor,
    selections_per_head: int,
    fractions: tuple[float, ...],
) -> dict[str, float]:
    sorted_frequency = torch.sort(frequency, dim=1, descending=True).values
    active = frequency > 0
    result = {
        "union_fraction": float(active.float().mean().item()),
        "never_selected_fraction": float((~active).float().mean().item()),
        "mean_frequency_if_selected": float(
            frequency.sum().item() / max(1, int(active.sum().item()))
        ),
    }
    denominator = max(1.0, float(selections_per_head))
    for fraction in fractions:
        count = min(
            frequency.shape[1],
            max(1, math.ceil(frequency.shape[1] * fraction)),
        )
        result[f"top_{fraction:g}_token_selection_share"] = float(
            (sorted_frequency[:, :count].sum(dim=1) / denominator)
            .mean()
            .item()
        )
    return result


def aggregate_rows(
    rows: list[dict[str, Any]],
    group_fields: tuple[str, ...],
    metric_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        item = dict(zip(group_fields, key))
        item["conditions"] = len(group)
        for field in metric_fields:
            item[field] = float(np.mean([float(row[field]) for row in group]))
        output.append(item)
    return output


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    refresh_intervals = parse_numbers(args.refresh_intervals, int)
    hot_fractions = parse_numbers(args.hot_fractions, float)
    output_fractions = parse_numbers(args.output_fractions, float)
    if not 0.0 < args.base_fraction <= 1.0:
        raise ValueError("base_fraction must be in (0, 1]")
    if not 0.0 <= args.frequency_decay <= 1.0:
        raise ValueError("frequency_decay must be in [0, 1]")

    step_rows: list[dict[str, Any]] = []
    concentration_rows: list[dict[str, Any]] = []

    for trace_spec in args.trace:
        topic, path_text = trace_spec.split("=", 1)
        payload = torch.load(path_text, map_location="cpu", weights_only=False)
        by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in payload["records"]:
            by_layer[int(record["layer"])].append(record)

        for layer, records in sorted(by_layer.items()):
            records.sort(key=lambda item: int(item["step"]))
            calibration = records[: args.calibration_steps]
            evaluation = records[args.calibration_steps :]
            if not calibration or not evaluation:
                raise ValueError(f"layer {layer} lacks calibration/evaluation")

            key = calibration[0]["key"].to(device=device).float()
            _, kv_heads, token_count, head_dim = key.shape
            sampled_key = key[..., :: args.key_stride, :].contiguous()
            calibration_query = mapped_queries(calibration).to(device)
            query_basis, key_basis = _qk_metric_projection_factors(
                second_moment(sampled_key),
                second_moment(calibration_query),
                projection_dim=head_dim,
                query_shrinkage=args.query_shrinkage,
            )
            projected_sample = torch.einsum(
                "bhnd,bhdm->bhnm", sampled_key, key_basis
            )
            projected_calibration_query = torch.einsum(
                "bhnd,bhdm->bhnm", calibration_query, query_basis
            )
            allocation = _hierarchical_qmse_rate_allocation(
                projected_sample,
                projected_calibration_query,
                bit_budget_per_coordinate=15,
                allow_zero_bits=True,
                include_scale_metadata=True,
            )
            projected_key = torch.einsum(
                "bhnd,bhdm->bhnm", key, key_basis
            )
            reconstructed_key = qksieve_reconstruct(
                projected_key, allocation
            )

            query_heads = int(calibration[0]["query"].shape[1])
            groups = query_heads // kv_heads
            base_count = max(1, math.ceil(token_count * args.base_fraction))
            exact_top_count = base_count
            frequency = torch.zeros(
                query_heads,
                token_count,
                dtype=torch.float32,
                device=device,
            )
            raw_frequency = torch.zeros_like(frequency)
            previous_by_interval: dict[int, torch.Tensor | None] = {
                interval: None for interval in refresh_intervals
            }

            for evaluation_index, record in enumerate(evaluation):
                query = record["query"].to(device=device).float()
                grouped_query = flatten_query_heads(query, kv_heads)
                projected_query = torch.einsum(
                    "hgd,hdm->hgm", grouped_query, query_basis[0]
                )
                exact_scaled = (
                    torch.einsum(
                        "hgd,hnd->hgn", grouped_query, key[0]
                    ).reshape(query_heads, token_count)
                    * float(record["scaling"])
                )
                proxy_scaled = (
                    torch.einsum(
                        "hgd,hnd->hgn",
                        projected_query,
                        reconstructed_key[0],
                    ).reshape(query_heads, token_count)
                    * float(record["scaling"])
                )
                exact_top = torch.topk(
                    exact_scaled,
                    k=exact_top_count,
                    dim=1,
                    sorted=False,
                ).indices
                probabilities = torch.softmax(exact_scaled, dim=-1)
                exact_mask = torch.zeros_like(exact_scaled, dtype=torch.bool)
                exact_mask.scatter_(1, exact_top, True)
                base_selected = torch.topk(
                    proxy_scaled,
                    k=base_count,
                    dim=1,
                    sorted=False,
                ).indices
                base_metrics = selected_metrics(
                    probabilities,
                    base_selected,
                    exact_mask,
                    exact_top_count,
                )

                for output_fraction in output_fractions:
                    output_count = max(
                        1, math.ceil(token_count * output_fraction)
                    )
                    full_selected = torch.topk(
                        proxy_scaled,
                        k=output_count,
                        dim=1,
                        sorted=False,
                    ).indices
                    metrics = selected_metrics(
                        probabilities,
                        full_selected,
                        exact_mask,
                        exact_top_count,
                    )
                    step_rows.append(
                        {
                            "topic": topic,
                            "layer": layer,
                            "step": int(record["step"]),
                            "method": "qksieve_full_scan",
                            "refresh_interval": 1,
                            "hot_fraction": 0.0,
                            "output_fraction": output_fraction,
                            "output_tokens": output_count,
                            "scan_fraction": 1.0,
                            "attention_mass": metrics["attention_mass"],
                            "oracle_top2_recall": metrics[
                                "oracle_top2_recall"
                            ],
                            "base_qksieve_mass": base_metrics[
                                "attention_mass"
                            ],
                        }
                    )

                    for interval in refresh_intervals:
                        refresh = (
                            previous_by_interval[interval] is None
                            or evaluation_index % interval == 0
                        )
                        for hot_fraction in hot_fractions:
                            if refresh:
                                selected = full_selected
                                pool_count = float(token_count)
                            else:
                                count = max(
                                    1,
                                    math.ceil(token_count * hot_fraction),
                                )
                                hot = hot_indices(
                                    frequency,
                                    count,
                                    evaluation_index,
                                )
                                selected, pool_count = shortlist_select(
                                    proxy_scaled,
                                    previous_by_interval[interval],
                                    hot,
                                    output_count,
                                )
                            metrics = selected_metrics(
                                probabilities,
                                selected,
                                exact_mask,
                                exact_top_count,
                            )
                            step_rows.append(
                                {
                                    "topic": topic,
                                    "layer": layer,
                                    "step": int(record["step"]),
                                    "method": "frequency_hotset_reuse",
                                    "refresh_interval": interval,
                                    "hot_fraction": hot_fraction,
                                    "output_fraction": output_fraction,
                                    "output_tokens": output_count,
                                    "scan_fraction": pool_count
                                    / token_count,
                                    "attention_mass": metrics[
                                        "attention_mass"
                                    ],
                                    "oracle_top2_recall": metrics[
                                        "oracle_top2_recall"
                                    ],
                                    "base_qksieve_mass": base_metrics[
                                        "attention_mass"
                                    ],
                                }
                            )

                for interval in refresh_intervals:
                    if (
                        previous_by_interval[interval] is None
                        or evaluation_index % interval == 0
                    ):
                        previous_by_interval[interval] = base_selected
                frequency.mul_(args.frequency_decay)
                frequency.scatter_add_(
                    1,
                    base_selected,
                    torch.ones_like(base_selected, dtype=frequency.dtype),
                )
                raw_frequency.scatter_add_(
                    1,
                    base_selected,
                    torch.ones_like(base_selected, dtype=raw_frequency.dtype),
                )

            concentration_rows.append(
                {
                    "topic": topic,
                    "layer": layer,
                    "query_heads": query_heads,
                    "history_tokens": token_count,
                    "evaluation_steps": len(evaluation),
                    "base_fraction": args.base_fraction,
                    **concentration_metrics(
                        raw_frequency,
                        len(evaluation) * base_count,
                        (0.001, 0.0025, 0.005, 0.01, 0.02, 0.04),
                    ),
                }
            )
            del (
                key,
                sampled_key,
                calibration_query,
                projected_sample,
                projected_calibration_query,
                projected_key,
                reconstructed_key,
                frequency,
                raw_frequency,
            )
            torch.cuda.empty_cache()

    with (args.output_dir / "step_detail.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(step_rows[0]))
        writer.writeheader()
        writer.writerows(step_rows)
    with (args.output_dir / "frequency_concentration.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(concentration_rows[0])
        )
        writer.writeheader()
        writer.writerows(concentration_rows)

    aggregate = aggregate_rows(
        step_rows,
        (
            "method",
            "refresh_interval",
            "hot_fraction",
            "output_fraction",
        ),
        (
            "scan_fraction",
            "attention_mass",
            "oracle_top2_recall",
            "base_qksieve_mass",
        ),
    )
    for row in aggregate:
        row["mass_retention_vs_qksieve_top2"] = (
            row["attention_mass"] / row["base_qksieve_mass"]
        )
        row["scan_speedup_upper_bound"] = 1.0 / row["scan_fraction"]

    concentration_summary = {
        field: float(np.mean([float(row[field]) for row in concentration_rows]))
        for field in concentration_rows[0]
        if field.startswith("top_")
        or field
        in {
            "union_fraction",
            "never_selected_fraction",
            "mean_frequency_if_selected",
        }
    }
    pareto = [
        row
        for row in aggregate
        if row["method"] == "frequency_hotset_reuse"
        and row["mass_retention_vs_qksieve_top2"] >= 0.995
    ]
    pareto.sort(
        key=lambda row: (
            row["scan_fraction"],
            row["output_fraction"],
        )
    )
    summary = {
        "schema": "qksieve_frequency_hotset_audit_v1",
        "traces": args.trace,
        "calibration_steps": args.calibration_steps,
        "frequency_source": (
            "causal previous QKSieve top-2% selections; exact future "
            "attention is evaluation only"
        ),
        "frequency_decay": args.frequency_decay,
        "frequency_concentration": concentration_summary,
        "aggregate": aggregate,
        "best_mass_retention_ge_99_5pct": pareto[0] if pareto else None,
        "caveat": (
            "Attention-mass retention is a screening metric, not PPL or "
            "downstream quality. Passing configurations require a held-out "
            "PPL test before entering the method."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "ALL_COMPLETE").touch()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
