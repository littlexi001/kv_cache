#!/usr/bin/env python
"""Stress-test Gaussian and empirical crossing rescue on synthetic errors."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

import torch


SCENARIOS = (
    "gaussian",
    "student_t3",
    "block_correlated",
    "high_norm_needle",
    "stealth_aligned_needle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--lengths", default="4096,65536,262144")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=1280)
    parser.add_argument("--sample_count", type=int, default=256)
    parser.add_argument("--failure_probability", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def systematic_indices(tokens: int, samples: int, device: torch.device) -> torch.Tensor:
    active = min(tokens, samples)
    return torch.div(
        torch.arange(active, device=device) * tokens,
        active,
        rounding_mode="floor",
    ).long()


def affine_calibrate(
    exact: torch.Tensor,
    proxy: torch.Tensor,
    sample_ids: torch.Tensor,
) -> torch.Tensor:
    sampled_exact = exact.index_select(1, sample_ids)
    sampled_proxy = proxy.index_select(1, sample_ids)
    proxy_centered = sampled_proxy - sampled_proxy.mean(dim=-1, keepdim=True)
    exact_centered = sampled_exact - sampled_exact.mean(dim=-1, keepdim=True)
    slope = (
        (proxy_centered * exact_centered).mean(dim=-1)
        / proxy_centered.square().mean(dim=-1).clamp_min(1.0e-12)
    ).clamp(0.25, 4.0)
    intercept = sampled_exact.mean(dim=-1) - slope * sampled_proxy.mean(dim=-1)
    return slope[:, None] * proxy + intercept[:, None]


def make_case(
    scenario: str,
    trials: int,
    tokens: int,
    sample_ids: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    exact = torch.randn(trials, tokens, generator=generator, device=device)
    raw_scale = torch.exp(
        -1.8
        + 0.35 * torch.randn(trials, tokens, generator=generator, device=device)
    )
    if scenario == "gaussian":
        normalized = torch.randn(
            trials, tokens, generator=generator, device=device
        )
    elif scenario == "student_t3":
        numerator = torch.randn(
            trials, tokens, generator=generator, device=device
        )
        denominator = torch.stack(
            [
                torch.randn(
                    trials, tokens, generator=generator, device=device
                ).square()
                for _ in range(3)
            ],
            dim=0,
        ).sum(dim=0)
        normalized = numerator / torch.sqrt(denominator / 3.0).clamp_min(1.0e-4)
        normalized /= math.sqrt(3.0)
    elif scenario == "block_correlated":
        block_size = 256
        blocks = math.ceil(tokens / block_size)
        block_noise = torch.randn(
            trials, blocks, generator=generator, device=device
        ).repeat_interleave(block_size, dim=-1)[:, :tokens]
        local_noise = torch.randn(
            trials, tokens, generator=generator, device=device
        )
        normalized = 0.9 * block_noise + math.sqrt(0.19) * local_noise
    elif scenario in {"high_norm_needle", "stealth_aligned_needle"}:
        normalized = torch.randn(
            trials, tokens, generator=generator, device=device
        )
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    proxy = exact - raw_scale * normalized
    needle = torch.full((trials,), -1, dtype=torch.long, device=device)
    if scenario in {"high_norm_needle", "stealth_aligned_needle"}:
        sampled = torch.zeros(tokens, dtype=torch.bool, device=device)
        sampled[sample_ids] = True
        unsampled = torch.nonzero(~sampled, as_tuple=False).flatten()
        needle.fill_(int(unsampled[-1].item()))
        rows = torch.arange(trials, device=device)
        exact[rows, needle] = exact.amax(dim=-1) + 5.0
        proxy[rows, needle] = -2.0
        required_error = exact[rows, needle] - proxy[rows, needle]
        if scenario == "high_norm_needle":
            raw_scale[rows, needle] = required_error / 1.5
        else:
            raw_scale[rows, needle] = raw_scale.median(dim=-1).values
    return exact, proxy, raw_scale, needle


def gaussian_probability(
    exact: torch.Tensor,
    proxy: torch.Tensor,
    raw_scale: torch.Tensor,
    boundary: torch.Tensor,
    sample_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = (
        exact.index_select(1, sample_ids)
        - proxy.index_select(1, sample_ids)
    ) / raw_scale.index_select(1, sample_ids).clamp_min(1.0e-12)
    multiplier = normalized.square().mean(dim=-1).sqrt().clamp_min(1.0e-6)
    sigma = raw_scale * multiplier[:, None]
    probability = 0.5 * torch.erfc(
        (boundary - proxy) / (math.sqrt(2.0) * sigma.clamp_min(1.0e-12))
    )
    return probability, normalized


def empirical_probability(
    exact: torch.Tensor,
    proxy: torch.Tensor,
    raw_scale: torch.Tensor,
    boundary: torch.Tensor,
    sample_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = (
        exact.index_select(1, sample_ids)
        - proxy.index_select(1, sample_ids)
    ) / raw_scale.index_select(1, sample_ids).clamp_min(1.0e-12)
    sorted_values = torch.sort(normalized, dim=-1).values.contiguous()
    threshold = ((boundary - proxy) / raw_scale.clamp_min(1.0e-12)).contiguous()
    insertion = torch.searchsorted(sorted_values, threshold, right=False)
    probability = (
        normalized.shape[-1] - insertion + 1
    ).float() / float(normalized.shape[-1] + 1)
    return probability, normalized


def bernstein_counts(
    probability: torch.Tensor,
    delta: float,
    maximum: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected = probability.sum(dim=-1)
    log_inverse = math.log(1.0 / delta)
    counts = torch.ceil(
        expected
        + torch.sqrt(2.0 * expected * log_inverse)
        + 2.0 * log_inverse / 3.0
    ).long().clamp(0, maximum)
    return counts, expected


def evaluate_method(
    exact: torch.Tensor,
    base_indices: torch.Tensor,
    base_mask: torch.Tensor,
    priority: torch.Tensor,
    counts: torch.Tensor,
    needle: torch.Tensor,
) -> dict[str, float]:
    top_k = base_indices.shape[-1]
    exact_top = torch.topk(exact, top_k, dim=-1, sorted=False).indices
    exact_mask = torch.zeros_like(base_mask)
    exact_mask.scatter_(1, exact_top, True)
    missed = exact_mask & ~base_mask
    recalls = []
    final_recalls = []
    needle_recalls = []
    failures = []
    for row in range(exact.shape[0]):
        count = int(counts[row].item())
        rescue = (
            torch.topk(priority[row], count, sorted=False).indices
            if count
            else torch.empty(0, dtype=torch.long, device=exact.device)
        )
        missed_count = int(missed[row].sum().item())
        recovered = int(missed[row].gather(0, rescue).sum().item()) if count else 0
        recalls.append(recovered / max(1, missed_count))
        failures.append(float(recovered < missed_count))
        union = torch.cat((base_indices[row], rescue))
        union_score = exact[row].gather(0, union)
        final = union.gather(0, torch.topk(union_score, top_k, sorted=False).indices)
        final_recalls.append(
            float(exact_mask[row].gather(0, final).float().mean().item())
        )
        if int(needle[row].item()) >= 0:
            needle_recalls.append(
                float(torch.any(final == needle[row]).item())
            )
    return {
        "missed_topk_rescue_recall": fmean(recalls),
        "any_miss_failure_rate": fmean(failures),
        "final_topk_recall": fmean(final_recalls),
        "needle_recall": fmean(needle_recalls) if needle_recalls else 1.0,
    }


def main() -> None:
    args = parse_args()
    lengths = tuple(int(item) for item in args.lengths.split(",") if item)
    if args.trials <= 0 or args.sample_count < 16:
        raise ValueError("trials must be positive and sample_count at least 16")
    if not 0.0 < args.failure_probability < 1.0:
        raise ValueError("failure_probability must lie in (0, 1)")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    rows: list[dict[str, Any]] = []
    for tokens in lengths:
        sample_ids = systematic_indices(tokens, args.sample_count, device)
        active_k = min(args.top_k, tokens)
        for scenario in SCENARIOS:
            exact, raw_proxy, raw_scale, needle = make_case(
                scenario,
                args.trials,
                tokens,
                sample_ids,
                generator,
                device,
            )
            proxy = affine_calibrate(exact, raw_proxy, sample_ids)
            base_indices = torch.topk(
                proxy, active_k, dim=-1, sorted=False
            ).indices
            base_mask = torch.zeros_like(proxy, dtype=torch.bool)
            base_mask.scatter_(1, base_indices, True)
            boundary = proxy.gather(1, base_indices).amin(dim=-1, keepdim=True)
            for method, estimator in (
                ("gaussian", gaussian_probability),
                ("empirical_add_one", empirical_probability),
            ):
                probability, normalized = estimator(
                    exact, proxy, raw_scale, boundary, sample_ids
                )
                probability.masked_fill_(base_mask, 0.0)
                priority = probability.masked_fill(base_mask, -torch.inf)
                counts, expected = bernstein_counts(
                    probability,
                    args.failure_probability,
                    tokens - active_k,
                )
                centered = normalized - normalized.mean(dim=-1, keepdim=True)
                variance = centered.square().mean(dim=-1).clamp_min(1.0e-12)
                kurtosis = centered.pow(4).mean(dim=-1) / variance.square()
                rows.append(
                    {
                        "tokens": tokens,
                        "scenario": scenario,
                        "method": method,
                        "rescue_tokens_mean": float(counts.float().mean().item()),
                        "rescue_ratio_mean": float(
                            counts.float().mean().item() / tokens
                        ),
                        "expected_crossings_mean": float(expected.mean().item()),
                        "normalized_error_kurtosis": float(kurtosis.mean().item()),
                        "normalized_error_q99": float(
                            torch.quantile(normalized, 0.99).item()
                        ),
                        **evaluate_method(
                            exact,
                            base_indices,
                            base_mask,
                            priority,
                            counts,
                            needle,
                        ),
                    }
                )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["scenario"]), str(row["method"]))].append(row)
    summary = []
    for (scenario, method), items in sorted(grouped.items()):
        summary.append(
            {
                "scenario": scenario,
                "method": method,
                "lengths": len(items),
                "final_topk_recall_worst": min(
                    float(item["final_topk_recall"]) for item in items
                ),
                "needle_recall_worst": min(
                    float(item["needle_recall"]) for item in items
                ),
                "failure_rate_worst": max(
                    float(item["any_miss_failure_rate"]) for item in items
                ),
                "rescue_ratio_mean": fmean(
                    float(item["rescue_ratio_mean"]) for item in items
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_condition.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema": "qksieve_crossing_adversarial_v1",
        "parameters": {**vars(args), "output_dir": str(args.output_dir)},
        "summary": summary,
        "rows": rows,
        "claim_boundary": (
            "Synthetic falsification. Bernstein control assumes independent or "
            "weakly dependent crossing events. A normal-residual, query-aligned "
            "needle is intentionally information-theoretically indistinguishable "
            "from ordinary tokens without reading more Key information."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
