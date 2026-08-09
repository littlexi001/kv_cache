from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from analyze_qksieve_output_risk_budget_20260803 import (
    crossfit_affine_softmax_kl,
)


RATES = (15, 19, 23, 27)
DIFFUSE_SIGMA = {15: 0.35, 19: 0.25, 23: 0.16, 27: 0.10}
NEEDLE_ERROR = {15: 8.0, 19: 6.0, 23: 4.0, 27: 2.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--lengths", default="4096,65536,262144")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--sample_count", type=int, default=256)
    parser.add_argument("--kl_tolerance", type=float, default=0.20)
    parser.add_argument("--tail_mass_tolerance", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def make_scores(
    scenario: str,
    rows: int,
    tokens: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    exact = torch.randn(rows, tokens, generator=generator)
    base_error = torch.randn(rows, tokens, generator=generator)
    special = torch.full((rows,), -1, dtype=torch.long)
    if scenario == "near_tied_plateau":
        width = min(256, tokens)
        plateau_indices = torch.topk(exact, width, dim=-1).indices
        plateau_center = exact.gather(1, plateau_indices).mean(dim=-1, keepdim=True)
        plateau = plateau_center + 0.02 * torch.randn(
            rows, width, generator=generator
        )
        exact.scatter_(1, plateau_indices, plateau)
    elif scenario == "block_correlated":
        block_size = 256
        block_count = math.ceil(tokens / block_size)
        block_error = torch.randn(rows, block_count, generator=generator)
        base_error = block_error.repeat_interleave(block_size, dim=-1)[
            :, :tokens
        ]
    elif scenario == "hidden_needle":
        special = torch.randint(tokens, (rows,), generator=generator)
        row_ids = torch.arange(rows)
        exact[row_ids, special] = exact.amax(dim=-1) + 4.0
        base_error[row_ids, special] = 0.0
    elif scenario != "diffuse":
        raise ValueError(f"unknown scenario: {scenario}")
    return exact, base_error, special


def proxy_for_rate(
    exact: torch.Tensor,
    base_error: torch.Tensor,
    special: torch.Tensor,
    rate: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    signed_error = DIFFUSE_SIGMA[rate] * base_error
    if torch.any(special >= 0):
        rows = torch.arange(exact.shape[0])
        signed_error = signed_error.clone()
        signed_error[rows, special] = NEEDLE_ERROR[rate]
    proxy = exact - signed_error
    return proxy, signed_error.abs()


def minimum_interval_mass_prefix(
    proxy: torch.Tensor,
    score_radius: torch.Tensor,
    target_tail_mass: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the smallest UCB prefix with a valid interval tail-mass bound."""

    upper = proxy.float() + score_radius.float()
    lower = proxy.float() - score_radius.float()
    order = torch.argsort(upper, dim=-1, descending=True)
    sorted_upper = upper.gather(1, order)
    sorted_lower = lower.gather(1, order)
    selected_lower_partition = torch.logcumsumexp(sorted_lower, dim=-1)
    reverse_tail = torch.logcumsumexp(sorted_upper.flip(-1), dim=-1).flip(-1)
    tail_upper_partition = torch.full_like(reverse_tail, -torch.inf)
    tail_upper_partition[:, :-1] = reverse_tail[:, 1:]
    tail_mass_upper = torch.sigmoid(
        tail_upper_partition - selected_lower_partition
    )
    passes = tail_mass_upper <= target_tail_mass
    counts = torch.argmax(passes.to(torch.int64), dim=-1) + 1
    no_pass = ~passes.any(dim=-1)
    counts = torch.where(
        no_pass,
        torch.full_like(counts, proxy.shape[-1]),
        counts,
    )
    chosen_bound = tail_mass_upper.gather(1, (counts - 1).unsqueeze(-1)).squeeze(
        -1
    )
    return counts, order, chosen_bound


def selected_true_tail_mass(
    exact: torch.Tensor,
    order: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    probability = torch.softmax(exact.float(), dim=-1)
    ranks = torch.arange(exact.shape[-1]).unsqueeze(0)
    selected = ranks < counts.unsqueeze(-1)
    selected_probability = probability.gather(1, order)
    return 1.0 - (selected_probability * selected).sum(dim=-1)


def quantile(values: torch.Tensor, fraction: float) -> float:
    return float(torch.quantile(values.float(), fraction).item())


def main() -> None:
    args = parse_args()
    lengths = tuple(int(item) for item in args.lengths.split(",") if item)
    if args.trials <= 0 or args.sample_count < 4:
        raise ValueError("trials must be positive and sample_count at least four")
    if not 0.0 < args.tail_mass_tolerance < 1.0:
        raise ValueError("tail mass tolerance must lie in (0, 1)")
    generator = torch.Generator().manual_seed(args.seed)
    rows: list[dict[str, float | int | str]] = []
    scenarios = (
        "diffuse",
        "near_tied_plateau",
        "block_correlated",
        "hidden_needle",
    )
    for tokens in lengths:
        for scenario in scenarios:
            exact, base_error, special = make_scores(
                scenario, args.trials, tokens, generator
            )
            per_rate: dict[int, dict[str, torch.Tensor]] = {}
            for rate in RATES:
                proxy, radius = proxy_for_rate(
                    exact, base_error, special, rate
                )
                sampled_kl = crossfit_affine_softmax_kl(
                    exact, proxy, args.sample_count
                )
                exact_log_probability = torch.log_softmax(exact, dim=-1)
                proxy_log_probability = torch.log_softmax(proxy, dim=-1)
                full_kl = (
                    exact_log_probability.exp()
                    * (exact_log_probability - proxy_log_probability)
                ).sum(dim=-1)
                counts, order, bound = minimum_interval_mass_prefix(
                    proxy, radius, args.tail_mass_tolerance
                )
                true_tail = selected_true_tail_mass(exact, order, counts)
                per_rate[rate] = {
                    "proxy": proxy,
                    "sampled_kl": sampled_kl,
                    "full_kl": full_kl,
                    "counts": counts,
                    "order": order,
                    "bound": bound,
                    "true_tail": true_tail,
                }

            sampled_rates = torch.full((args.trials,), RATES[-1], dtype=torch.long)
            unresolved = torch.ones(args.trials, dtype=torch.bool)
            for rate in RATES:
                accepted = unresolved & (
                    per_rate[rate]["sampled_kl"] <= args.kl_tolerance
                )
                sampled_rates[accepted] = rate
                unresolved &= ~accepted

            for trial in range(args.trials):
                chosen_rate = int(sampled_rates[trial])
                chosen = per_rate[chosen_rate]
                top_k = min(1280, tokens)
                sampled_order = torch.argsort(
                    chosen["proxy"][trial], descending=True
                )
                fixed_selected = sampled_order[:top_k]
                fixed_tail = 1.0 - torch.softmax(
                    exact[trial], dim=-1
                ).gather(0, fixed_selected).sum()
                needle = int(special[trial])
                fixed_has_needle = (
                    needle < 0 or bool(torch.any(fixed_selected == needle))
                )
                count = int(chosen["counts"][trial])
                certified_selected = chosen["order"][trial, :count]
                certified_has_needle = (
                    needle < 0
                    or bool(torch.any(certified_selected == needle))
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "tokens": tokens,
                        "trial": trial,
                        "chosen_rate": chosen_rate,
                        "sampled_crossfit_kl": float(
                            chosen["sampled_kl"][trial]
                        ),
                        "full_softmax_kl": float(chosen["full_kl"][trial]),
                        "fixed_top1280_true_tail_mass": float(fixed_tail),
                        "fixed_top1280_has_needle": int(fixed_has_needle),
                        "certificate_tokens": count,
                        "certificate_ratio": count / tokens,
                        "certificate_tail_mass_upper": float(
                            chosen["bound"][trial]
                        ),
                        "certificate_true_tail_mass": float(
                            chosen["true_tail"][trial]
                        ),
                        "certificate_covers": int(
                            chosen["bound"][trial]
                            + 1.0e-6
                            >= chosen["true_tail"][trial]
                        ),
                        "certificate_has_needle": int(certified_has_needle),
                    }
                )

    summaries = []
    for tokens in lengths:
        for scenario in scenarios:
            group = [
                row
                for row in rows
                if row["tokens"] == tokens and row["scenario"] == scenario
            ]
            summaries.append(
                {
                    "scenario": scenario,
                    "tokens": tokens,
                    "chosen_rate_mean": sum(
                        float(row["chosen_rate"]) for row in group
                    )
                    / len(group),
                    "sampled_kl_mean": sum(
                        float(row["sampled_crossfit_kl"]) for row in group
                    )
                    / len(group),
                    "full_kl_mean": sum(
                        float(row["full_softmax_kl"]) for row in group
                    )
                    / len(group),
                    "fixed_needle_recall": sum(
                        int(row["fixed_top1280_has_needle"]) for row in group
                    )
                    / len(group),
                    "certificate_needle_recall": sum(
                        int(row["certificate_has_needle"]) for row in group
                    )
                    / len(group),
                    "certificate_coverage": sum(
                        int(row["certificate_covers"]) for row in group
                    )
                    / len(group),
                    "certificate_ratio_mean": sum(
                        float(row["certificate_ratio"]) for row in group
                    )
                    / len(group),
                    "certificate_ratio_p90": quantile(
                        torch.tensor(
                            [float(row["certificate_ratio"]) for row in group]
                        ),
                        0.90,
                    ),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "trials.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema": "qksieve_dual_certificate_adversarial_v1",
        "parameters": {
            **vars(args),
            "output_dir": str(args.output_dir),
            "rates": RATES,
            "diffuse_sigma": DIFFUSE_SIGMA,
            "needle_error": NEEDLE_ERROR,
        },
        "summary": summaries,
        "claim_boundary": (
            "Synthetic falsification only. The interval certificate uses the "
            "true absolute proxy error as an optimistic valid radius; a "
            "deployable quantized radius can only be looser."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
