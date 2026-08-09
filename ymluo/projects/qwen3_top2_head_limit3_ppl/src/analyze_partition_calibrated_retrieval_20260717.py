from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from run_head_top2_targeted_ppl_20260714 import _pca_int4_partial_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate sample-calibrated omitted softmax partition estimates."
    )
    parser.add_argument("--trace_paths", nargs="+", required=True, type=Path)
    parser.add_argument("--output_path", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--sample_fraction", type=float, default=0.0025)
    parser.add_argument("--sample_repeats", type=int, default=8)
    parser.add_argument(
        "--candidate_fractions", default="0.005,0.01,0.02,0.03,0.04,0.06,0.08"
    )
    parser.add_argument("--target_masses", default="0.80,0.90,0.95,0.98")
    parser.add_argument("--ucb_z_values", default="1.0,1.64,2.33")
    return parser.parse_args()


def parse_floats(spec: str) -> list[float]:
    values = sorted({float(item) for item in spec.split(",") if item.strip()})
    if not values or values[0] <= 0.0 or values[-1] > 1.0:
        raise ValueError("fractions must be in (0, 1]")
    return values


def parse_nonnegative_floats(spec: str) -> list[float]:
    values = sorted({float(item) for item in spec.split(",") if item.strip()})
    if not values or values[0] < 0.0:
        raise ValueError("values must be nonnegative")
    return values


def summarize(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "p95": float(torch.quantile(tensor, 0.95)),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


def ucb_name(z_value: float) -> str:
    return f"control_variate_ucb_z{z_value:.2f}".replace(".", "p")


def estimate_candidate_masses(
    exact_scores: torch.Tensor,
    proxy_scores: torch.Tensor,
    candidate_fractions: list[float],
    sample_fraction: float,
    sample_offset: int,
    ucb_z_values: list[float],
) -> tuple[dict[str, list[torch.Tensor]], dict[str, torch.Tensor]]:
    head_count, history_count = exact_scores.shape
    max_candidate_count = min(
        history_count, math.ceil(candidate_fractions[-1] * history_count)
    )
    proxy_top_scores, proxy_top_indices = torch.topk(
        proxy_scores, max_candidate_count, dim=-1, sorted=True
    )
    exact_top_scores = torch.gather(exact_scores, -1, proxy_top_indices)

    sample_count = min(
        history_count, max(16, math.ceil(sample_fraction * history_count))
    )
    stride = max(1, history_count // sample_count)
    offset = int(sample_offset) % stride
    sample_indices = offset + torch.arange(sample_count, device=exact_scores.device) * stride
    sample_indices = sample_indices.clamp_max(history_count - 1)
    exact_sample_scores = exact_scores.index_select(-1, sample_indices)
    proxy_sample_scores = proxy_scores.index_select(-1, sample_indices)

    candidate_rank = torch.full(
        (head_count, history_count),
        max_candidate_count,
        dtype=torch.int32,
        device=exact_scores.device,
    )
    rank_values = torch.arange(
        max_candidate_count, dtype=torch.int32, device=exact_scores.device
    ).view(1, -1).expand(head_count, -1)
    candidate_rank.scatter_(1, proxy_top_indices, rank_values)
    sample_rank = candidate_rank.index_select(-1, sample_indices)

    center = torch.maximum(
        exact_scores.amax(dim=-1), proxy_scores.amax(dim=-1)
    ).unsqueeze(-1)
    exact_weights = torch.exp(exact_scores - center)
    proxy_weights = torch.exp(proxy_scores - center)
    exact_total = exact_weights.sum(dim=-1)
    proxy_total = proxy_weights.sum(dim=-1)
    exact_top_weights = torch.exp(exact_top_scores - center)
    proxy_top_weights = torch.exp(proxy_top_scores - center)
    exact_prefix = exact_top_weights.cumsum(dim=-1)
    proxy_prefix = proxy_top_weights.cumsum(dim=-1)
    exact_sample_weights = torch.exp(exact_sample_scores - center)
    proxy_sample_weights = torch.exp(proxy_sample_scores - center)

    estimates: dict[str, list[torch.Tensor]] = {
        "proxy": [],
        "ratio": [],
        "control_variate": [],
        "actual": [],
    }
    for z_value in ucb_z_values:
        estimates[ucb_name(z_value)] = []
    candidate_counts: list[int] = []
    for fraction in candidate_fractions:
        candidate_count = min(history_count, max(1, math.ceil(fraction * history_count)))
        candidate_counts.append(candidate_count)
        exact_kept = exact_prefix[:, candidate_count - 1]
        proxy_kept = proxy_prefix[:, candidate_count - 1]
        actual_mass = exact_kept / exact_total.clamp_min(1.0e-30)
        proxy_omitted = (proxy_total - proxy_kept).clamp_min(0.0)

        outside_sample = sample_rank >= candidate_count
        outside_count = outside_sample.sum(dim=-1).clamp_min(1)
        outside_size = history_count - candidate_count
        exact_sample_omitted = (exact_sample_weights * outside_sample).sum(dim=-1)
        proxy_sample_omitted = (proxy_sample_weights * outside_sample).sum(dim=-1)

        ratio = exact_sample_omitted / proxy_sample_omitted.clamp_min(1.0e-30)
        ratio_omitted = proxy_omitted * ratio
        sample_difference = exact_sample_weights - proxy_sample_weights
        difference_mean = (sample_difference * outside_sample).sum(dim=-1) / outside_count
        difference_variance = (
            (sample_difference - difference_mean.unsqueeze(-1)).square()
            * outside_sample
        ).sum(dim=-1) / (outside_count - 1).clamp_min(1)
        difference_standard_error = (
            difference_variance / outside_count
        ).clamp_min(0.0).sqrt()
        correction = outside_size * difference_mean
        control_variate_omitted = (proxy_omitted + correction).clamp_min(0.0)

        estimates["actual"].append(actual_mass)
        estimates["proxy"].append(
            exact_kept / (exact_kept + proxy_omitted).clamp_min(1.0e-30)
        )
        estimates["ratio"].append(
            exact_kept / (exact_kept + ratio_omitted).clamp_min(1.0e-30)
        )
        estimates["control_variate"].append(
            exact_kept
            / (exact_kept + control_variate_omitted).clamp_min(1.0e-30)
        )
        for z_value in ucb_z_values:
            upper_omitted = (
                proxy_omitted
                + outside_size
                * (difference_mean + z_value * difference_standard_error)
            ).clamp_min(0.0)
            estimates[ucb_name(z_value)].append(
                exact_kept / (exact_kept + upper_omitted).clamp_min(1.0e-30)
            )

    exact_keep_count = min(history_count, max(1, math.ceil(0.02 * history_count)))
    exact_best_mass = torch.topk(
        exact_scores, exact_keep_count, dim=-1, sorted=False
    ).values
    exact_best_mass = torch.exp(exact_best_mass - center).sum(dim=-1) / exact_total
    diagnostics = {
        "candidate_counts": torch.tensor(candidate_counts),
        "exact_top2_mass": exact_best_mass,
        "proxy_top2_mass": estimates["actual"][
            min(range(len(candidate_fractions)), key=lambda i: abs(candidate_fractions[i] - 0.02))
        ],
    }
    return estimates, diagnostics


def main() -> None:
    args = parse_args()
    candidate_fractions = parse_floats(args.candidate_fractions)
    target_masses = parse_floats(args.target_masses)
    ucb_z_values = parse_nonnegative_floats(args.ucb_z_values)
    if args.sample_repeats <= 0:
        raise ValueError("sample_repeats must be positive")
    if candidate_fractions[-1] > 0.25:
        raise ValueError("this diagnostic is intended for sparse candidate prefixes")

    estimator_names = [
        "proxy",
        "ratio",
        "control_variate",
        *(ucb_name(z_value) for z_value in ucb_z_values),
    ]
    all_estimates: dict[str, list[torch.Tensor]] = {
        "proxy": [],
        "ratio": [],
        "control_variate": [],
        "actual": [],
    }
    for name in estimator_names[3:]:
        all_estimates[name] = []
    top2_exact: list[torch.Tensor] = []
    top2_proxy: list[torch.Tensor] = []
    case_rows: list[dict[str, Any]] = []

    device = torch.device(args.device)
    for trace_path in args.trace_paths:
        trace = torch.load(trace_path, map_location="cpu", weights_only=False)
        topic = str(trace.get("config", {}).get("topic", trace_path.stem))
        for record_index, record in enumerate(trace["records"]):
            query = record["query"].to(device)
            key = record["key"].to(device)
            scaling = float(record["scaling"])
            batch_count, query_head_count, _, head_dim = query.shape
            kv_head_count = int(key.shape[1])
            group_count = query_head_count // kv_head_count
            grouped_query = query.squeeze(2).reshape(
                batch_count, kv_head_count, group_count, head_dim
            )
            exact_scores = torch.einsum(
                "bhgd,bhnd->bhgn", grouped_query.float(), key.float()
            ).reshape(query_head_count, key.shape[2]) * scaling

            pca_state: dict[str, Any] = {}
            proxy_scores = _pca_int4_partial_scores(
                query.squeeze(2),
                key,
                pca_state,
                args.projection_dim,
                use_chunked_layout=True,
            ).squeeze(0).float() * scaling
            diagnostics = None
            for sample_repeat in range(args.sample_repeats):
                estimates, diagnostics = estimate_candidate_masses(
                    exact_scores,
                    proxy_scores,
                    candidate_fractions,
                    args.sample_fraction,
                    sample_offset=37 * int(record["layer"]) + 53 * sample_repeat,
                    ucb_z_values=ucb_z_values,
                )
                for name, values in estimates.items():
                    all_estimates[name].append(torch.stack(values, dim=-1).cpu())
            assert diagnostics is not None
            top2_exact.append(diagnostics["exact_top2_mass"].cpu())
            top2_proxy.append(diagnostics["proxy_top2_mass"].cpu())
            case_rows.append(
                {
                    "topic": topic,
                    "record_index": record_index,
                    "layer": int(record["layer"]),
                    "exact_top2_mass_mean": float(diagnostics["exact_top2_mass"].mean()),
                    "proxy_top2_mass_mean": float(diagnostics["proxy_top2_mass"].mean()),
                }
            )
            print(
                f"[{topic}] layer={record['layer']} "
                f"exact_top2_mass={case_rows[-1]['exact_top2_mass_mean']:.4f} "
                f"proxy_top2_mass={case_rows[-1]['proxy_top2_mass_mean']:.4f}",
                flush=True,
            )
            del query, key, exact_scores, proxy_scores, pca_state
            torch.cuda.empty_cache()

    stacked = {
        name: torch.cat(values, dim=0) for name, values in all_estimates.items()
    }
    actual = stacked["actual"]
    estimator_quality: dict[str, Any] = {}
    for name in estimator_names:
        error = stacked[name] - actual
        estimator_quality[name] = {
            "mean_error": float(error.mean()),
            "mean_absolute_error": float(error.abs().mean()),
            "p95_absolute_error": float(torch.quantile(error.abs(), 0.95)),
            "max_absolute_error": float(error.abs().max()),
            "underestimate_omitted_mass_rate": float((error > 0.0).float().mean()),
        }

    policy_quality: dict[str, Any] = {}
    fraction_tensor = torch.tensor(candidate_fractions).view(1, -1)
    for target_mass in target_masses:
        target_key = f"mass_{target_mass:g}".replace(".", "p")
        policy_quality[target_key] = {}
        oracle_valid = actual >= target_mass
        oracle_attainable = oracle_valid.any(dim=-1)
        oracle_choice = torch.where(
            oracle_valid.any(dim=-1),
            oracle_valid.float().argmax(dim=-1),
            torch.full((actual.shape[0],), actual.shape[1] - 1, dtype=torch.long),
        )
        oracle_fraction = fraction_tensor.expand(actual.shape[0], -1).gather(
            1, oracle_choice.unsqueeze(-1)
        ).squeeze(-1)
        for name in estimator_names:
            predicted = stacked[name]
            valid = predicted >= target_mass
            choice = torch.where(
                valid.any(dim=-1),
                valid.float().argmax(dim=-1),
                torch.full((predicted.shape[0],), predicted.shape[1] - 1, dtype=torch.long),
            )
            selected_fraction = fraction_tensor.expand(predicted.shape[0], -1).gather(
                1, choice.unsqueeze(-1)
            ).squeeze(-1)
            selected_actual = actual.gather(1, choice.unsqueeze(-1)).squeeze(-1)
            policy_quality[target_key][name] = {
                "selected_fraction": summarize(selected_fraction.tolist()),
                "actual_mass": summarize(selected_actual.tolist()),
                "violation_rate": float((selected_actual < target_mass).float().mean()),
                "attainable_violation_rate": float(
                    (selected_actual[oracle_attainable] < target_mass).float().mean()
                )
                if bool(oracle_attainable.any())
                else 0.0,
                "mean_shortfall": float((target_mass - selected_actual).clamp_min(0.0).mean()),
                "mean_fraction_minus_grid_oracle": float(
                    (selected_fraction - oracle_fraction).mean()
                ),
            }
        policy_quality[target_key]["grid_oracle_fraction"] = summarize(
            oracle_fraction.tolist()
        )
        policy_quality[target_key]["grid_unattainable_rate"] = float(
            (~oracle_attainable).float().mean()
        )

    output = {
        "config": {
            "trace_paths": [str(path) for path in args.trace_paths],
            "projection_dim": args.projection_dim,
            "sample_fraction": args.sample_fraction,
            "sample_repeats": args.sample_repeats,
            "candidate_fractions": candidate_fractions,
            "target_masses": target_masses,
            "ucb_z_values": ucb_z_values,
            "head_case_count": int(actual.shape[0]),
        },
        "top2_mass": {
            "exact_top2": summarize(torch.cat(top2_exact).tolist()),
            "proxy_top2": summarize(torch.cat(top2_proxy).tolist()),
        },
        "estimator_quality": estimator_quality,
        "policy_quality": policy_quality,
        "cases": case_rows,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
