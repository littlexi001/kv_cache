from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from run_head_top2_targeted_ppl_20260714 import (
    _partition_global_sample_budget_ladder,
    _partition_proxy_ucb_budget_ladder,
    _pca_int4_partial_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze one-shot sample-calibrated partition budgets on QKV traces."
    )
    parser.add_argument("--trace_paths", nargs="+", required=True, type=Path)
    parser.add_argument("--output_path", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--sample_fraction", type=float, default=0.0025)
    parser.add_argument("--sample_repeats", type=int, default=8)
    parser.add_argument("--budget_fractions", default="0.005,0.01,0.02,0.03,0.04,0.06,0.08")
    parser.add_argument("--target_mass", type=float, default=0.70)
    parser.add_argument("--ucb_z", type=float, default=0.0)
    parser.add_argument("--overfetch_factor", type=int, default=2)
    parser.add_argument("--budget_estimator", choices=["proxy", "global"], default="proxy")
    parser.add_argument("--verify_target", type=float, default=0.0)
    parser.add_argument("--verify_ucb_z", type=float, default=0.0)
    return parser.parse_args()


def parse_fractions(spec: str) -> tuple[float, ...]:
    fractions = tuple(sorted({float(item) for item in spec.split(",") if item.strip()}))
    if not fractions or fractions[0] <= 0.0 or fractions[-1] > 1.0:
        raise ValueError("budget fractions must be in (0, 1]")
    return fractions


def summarize(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "p10": float(torch.quantile(tensor, 0.10)),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "p95": float(torch.quantile(tensor, 0.95)),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


def append_tensor(rows: dict[str, list[float]], name: str, value: torch.Tensor) -> None:
    rows[name].extend(value.detach().float().cpu().flatten().tolist())


def analyze_record(
    record: dict[str, Any],
    device: torch.device,
    projection_dim: int,
    sample_fraction: float,
    sample_repeats: int,
    fractions: tuple[float, ...],
    target_mass: float,
    ucb_z: float,
    overfetch_factor: int,
    verify_target: float,
    verify_ucb_z: float,
    budget_estimator: str,
) -> dict[str, list[float]]:
    query = record["query"].to(device)
    key = record["key"].to(device)
    value = record["value"].to(device)
    scaling = float(record["scaling"])
    batch_count, query_head_count, _, head_dim = query.shape
    if batch_count != 1:
        raise ValueError("trace analyzer currently expects batch size one")
    kv_head_count = int(key.shape[1])
    group_count = query_head_count // kv_head_count
    history_count = int(key.shape[2]) - 1
    query_raw = query.squeeze(2)
    grouped_query = query_raw.reshape(1, kv_head_count, group_count, head_dim)
    exact_all = torch.einsum(
        "bhgd,bhnd->bhgn", grouped_query.float(), key.float()
    ).reshape(query_head_count, key.shape[2]) * scaling
    exact_history = exact_all[:, :history_count]
    self_scores = exact_all[:, history_count].unsqueeze(0)

    pca_state: dict[str, Any] = {}
    proxy_scores = _pca_int4_partial_scores(
        query_raw,
        key[..., :history_count, :],
        pca_state,
        projection_dim,
        use_chunked_layout=True,
    ).float() * scaling
    max_candidate_count = min(
        history_count, math.ceil(fractions[-1] * history_count)
    )
    _, candidate_indices = torch.topk(
        proxy_scores, max_candidate_count, dim=-1, sorted=True
    )
    exact_candidate_scores = torch.gather(
        exact_history.unsqueeze(0), -1, candidate_indices
    )
    keep_counts = tuple(
        min(history_count, max(1, math.ceil(fraction * history_count)))
        for fraction in fractions
    )
    count_options = torch.tensor(keep_counts, dtype=torch.long, device=device)
    fraction_options = torch.tensor(fractions, dtype=torch.float32, device=device)

    expanded_value = value.repeat_interleave(group_count, dim=1).squeeze(0).float()
    full_weights = torch.softmax(exact_all, dim=-1)
    full_output = torch.einsum("hn,hnd->hd", full_weights, expanded_value)
    exact_center = exact_all.amax(dim=-1, keepdim=True)
    exact_total = torch.exp(exact_all - exact_center).sum(dim=-1)

    exact_top2_count = min(history_count, max(1, math.ceil(0.02 * history_count)))
    exact_top2_scores, exact_top2_indices = torch.topk(
        exact_history, exact_top2_count, dim=-1, sorted=True
    )
    exact_top2_partition = (
        torch.exp(exact_top2_scores - exact_center).sum(dim=-1)
        + torch.exp(exact_all[:, -1] - exact_center.squeeze(-1))
    )
    exact_top2_mass = exact_top2_partition / exact_total
    exact_top2_value = torch.gather(
        expanded_value[:, :history_count],
        1,
        exact_top2_indices.unsqueeze(-1).expand(-1, -1, head_dim),
    )
    exact_top2_value = torch.cat((exact_top2_value, expanded_value[:, -1:, :]), dim=1)
    exact_top2_weight = torch.softmax(
        torch.cat((exact_top2_scores, exact_all[:, -1:]), dim=-1), dim=-1
    )
    exact_top2_output = torch.einsum("hk,hkd->hd", exact_top2_weight, exact_top2_value)
    exact_top2_error = (
        (exact_top2_output - full_output).norm(dim=-1)
        / full_output.norm(dim=-1).clamp_min(1.0e-8)
    )

    positions = torch.arange(max_candidate_count, device=device).view(1, 1, -1)
    selected_rank = torch.arange(keep_counts[-1], device=device).view(1, 1, -1)

    def select_for_rung(
        rung: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        chosen_keep = count_options[rung]
        pool_counts = (chosen_keep * overfetch_factor).clamp_max(max_candidate_count)
        scored_candidates = exact_candidate_scores.masked_fill(
            positions >= pool_counts.unsqueeze(-1), -torch.inf
        )
        selected_scores, selected_positions = torch.topk(
            scored_candidates, keep_counts[-1], dim=-1, sorted=True
        )
        selected_indices = torch.gather(candidate_indices, -1, selected_positions)
        selected_valid = selected_rank < chosen_keep.unsqueeze(-1)
        selected_scores = selected_scores.masked_fill(~selected_valid, -torch.inf)
        selected_partition = (
            torch.exp(selected_scores.squeeze(0) - exact_center).sum(dim=-1)
            + torch.exp(exact_all[:, -1] - exact_center.squeeze(-1))
        )
        actual_mass = selected_partition / exact_total
        selected_value = torch.gather(
            expanded_value[:, :history_count],
            1,
            selected_indices.squeeze(0).unsqueeze(-1).expand(-1, -1, head_dim),
        )
        selected_value = torch.cat((selected_value, expanded_value[:, -1:, :]), dim=1)
        sparse_scores = torch.cat(
            (selected_scores.squeeze(0), exact_all[:, -1:]), dim=-1
        )
        sparse_weights = torch.softmax(sparse_scores, dim=-1)
        sparse_output = torch.einsum("hk,hkd->hd", sparse_weights, selected_value)
        relative_error = (
            (sparse_output - full_output).norm(dim=-1)
            / full_output.norm(dim=-1).clamp_min(1.0e-8)
        )
        return (
            chosen_keep,
            pool_counts,
            selected_scores,
            selected_indices,
            actual_mass,
            relative_error,
        )

    def verify_selected_mass(
        selected_scores: torch.Tensor,
        selected_indices: torch.Tensor,
        selected_keep: torch.Tensor,
        sample_scores: torch.Tensor,
        sample_indices: torch.Tensor,
    ) -> torch.Tensor:
        center = torch.maximum(proxy_scores.amax(dim=-1), sample_scores.amax(dim=-1))
        center = torch.maximum(center, selected_scores.amax(dim=-1))
        center = torch.maximum(center, self_scores).unsqueeze(-1)
        proxy_weights = torch.exp(proxy_scores.float() - center)
        proxy_total = proxy_weights.sum(dim=-1)
        selected_valid = selected_rank < selected_keep.unsqueeze(-1)
        proxy_selected = (
            torch.gather(proxy_weights, -1, selected_indices) * selected_valid
        ).sum(dim=-1)
        selected_partition = (
            torch.exp(selected_scores.float() - center).sum(dim=-1)
            + torch.exp(self_scores.float().unsqueeze(-1) - center).squeeze(-1)
        )
        sample_proxy = proxy_weights.index_select(-1, sample_indices)
        sample_exact = torch.exp(sample_scores.float() - center)
        sample_positions = sample_indices.view(1, 1, 1, -1)
        sample_membership = (
            (selected_indices.unsqueeze(-1) == sample_positions)
            & selected_valid.unsqueeze(-1)
        ).any(dim=-2)
        outside_sample = ~sample_membership
        outside_count = outside_sample.sum(dim=-1).clamp_min(1)
        difference = sample_exact - sample_proxy
        difference_mean = (difference * outside_sample).sum(dim=-1) / outside_count
        difference_variance = (
            (difference - difference_mean.unsqueeze(-1)).square() * outside_sample
        ).sum(dim=-1) / (outside_count - 1).clamp_min(1)
        standard_error = (difference_variance / outside_count).clamp_min(0.0).sqrt()
        proxy_tail = (proxy_total - proxy_selected).clamp_min(0.0)
        upper_tail = (
            proxy_tail
            + (history_count - selected_keep)
            * (difference_mean + verify_ucb_z * standard_error)
        ).clamp_min(0.0)
        return selected_partition / (selected_partition + upper_tail).clamp_min(1.0e-30)

    metrics: dict[str, list[float]] = defaultdict(list)
    for repeat in range(sample_repeats):
        sample_count = min(
            history_count, max(1, math.ceil(sample_fraction * history_count))
        )
        stride = max(1, history_count // sample_count)
        offset = (37 * int(record["layer"]) + 53 * repeat) % stride
        sample_indices = offset + torch.arange(sample_count, device=device) * stride
        sample_indices = sample_indices.clamp_max(history_count - 1)
        sample_scores = exact_history.index_select(-1, sample_indices).unsqueeze(0)
        budget_ladder = (
            _partition_global_sample_budget_ladder
            if budget_estimator == "global"
            else _partition_proxy_ucb_budget_ladder
        )
        chosen_rung, estimated_mass, _ = budget_ladder(
            proxy_scores,
            candidate_indices,
            sample_scores,
            sample_indices,
            self_scores,
            keep_counts,
            target_mass,
            ucb_z,
        )
        (
            initial_keep,
            _,
            initial_scores,
            initial_indices,
            initial_actual_mass,
            initial_relative_error,
        ) = select_for_rung(chosen_rung)
        initial_verified_mass = verify_selected_mass(
            initial_scores,
            initial_indices,
            initial_keep,
            sample_scores,
            sample_indices,
        )
        expandable = chosen_rung < len(keep_counts) - 1
        expanded = (
            (initial_verified_mass < verify_target) & expandable
            if verify_target > 0.0
            else torch.zeros_like(expandable)
        )
        final_rung = torch.where(expanded, chosen_rung + 1, chosen_rung)
        (
            final_keep,
            pool_counts,
            final_scores,
            final_indices,
            actual_mass,
            relative_error,
        ) = select_for_rung(final_rung)
        verified_mass = verify_selected_mass(
            final_scores,
            final_indices,
            final_keep,
            sample_scores,
            sample_indices,
        )

        append_tensor(metrics, "estimated_mass", estimated_mass)
        append_tensor(metrics, "verified_mass", verified_mass)
        append_tensor(metrics, "actual_mass", actual_mass)
        append_tensor(metrics, "mass_estimation_error", verified_mass.squeeze(0) - actual_mass)
        append_tensor(
            metrics,
            "predicted_mass_estimation_error",
            estimated_mass.squeeze(0) - initial_actual_mass,
        )
        append_tensor(metrics, "initial_actual_mass", initial_actual_mass)
        append_tensor(metrics, "initial_relative_output_error", initial_relative_error)
        append_tensor(metrics, "expanded", expanded.float())
        append_tensor(metrics, "final_fraction", fraction_options[final_rung])
        append_tensor(metrics, "exact_qk_fraction", pool_counts.float() / history_count)
        append_tensor(metrics, "relative_output_error", relative_error)
        append_tensor(metrics, "exact_top2_mass", exact_top2_mass)
        append_tensor(metrics, "exact_top2_relative_output_error", exact_top2_error)
    return metrics


def main() -> None:
    args = parse_args()
    fractions = parse_fractions(args.budget_fractions)
    if args.sample_repeats < 1 or args.overfetch_factor < 1:
        raise ValueError("sample repeats and overfetch factor must be positive")
    device = torch.device(args.device)
    aggregate: dict[str, list[float]] = defaultdict(list)
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for trace_path in args.trace_paths:
        trace = torch.load(trace_path, map_location="cpu", weights_only=False)
        topic = str(trace.get("config", {}).get("topic", trace_path.stem))
        for record in trace["records"]:
            metrics = analyze_record(
                record,
                device,
                args.projection_dim,
                args.sample_fraction,
                args.sample_repeats,
                fractions,
                args.target_mass,
                args.ucb_z,
                args.overfetch_factor,
                args.verify_target,
                args.verify_ucb_z,
                args.budget_estimator,
            )
            group_name = f"{topic}/layer{int(record['layer'])}"
            for name, values in metrics.items():
                aggregate[name].extend(values)
                grouped[group_name][name].extend(values)
            print(
                f"[{group_name}] mass={sum(metrics['actual_mass']) / len(metrics['actual_mass']):.4f} "
                f"final={sum(metrics['final_fraction']) / len(metrics['final_fraction']):.4f} "
                f"exact_qk={sum(metrics['exact_qk_fraction']) / len(metrics['exact_qk_fraction']):.4f}",
                flush=True,
            )
            torch.cuda.empty_cache()

    def build_summary(rows: dict[str, list[float]]) -> dict[str, Any]:
        actual_mass = torch.tensor(rows["actual_mass"])
        estimation_error = torch.tensor(rows["mass_estimation_error"])
        predicted_estimation_error = torch.tensor(
            rows["predicted_mass_estimation_error"]
        )
        return {
            "head_sample_count": len(rows["actual_mass"]),
            "estimated_mass": summarize(rows["estimated_mass"]),
            "verified_mass": summarize(rows["verified_mass"]),
            "actual_mass": summarize(rows["actual_mass"]),
            "initial_actual_mass": summarize(rows["initial_actual_mass"]),
            "final_fraction": summarize(rows["final_fraction"]),
            "exact_qk_fraction": summarize(rows["exact_qk_fraction"]),
            "relative_output_error": summarize(rows["relative_output_error"]),
            "initial_relative_output_error": summarize(
                rows["initial_relative_output_error"]
            ),
            "expansion_rate": float(torch.tensor(rows["expanded"]).mean()),
            "exact_top2_mass": summarize(rows["exact_top2_mass"]),
            "exact_top2_relative_output_error": summarize(
                rows["exact_top2_relative_output_error"]
            ),
            "actual_mass_below_0p70_rate": float((actual_mass < 0.70).float().mean()),
            "actual_mass_below_0p80_rate": float((actual_mass < 0.80).float().mean()),
            "actual_mass_below_0p90_rate": float((actual_mass < 0.90).float().mean()),
            "mass_overestimate_rate": float((estimation_error > 0.0).float().mean()),
            "mass_overestimate_gt_0p05_rate": float((estimation_error > 0.05).float().mean()),
            "predicted_mass_overestimate_rate": float(
                (predicted_estimation_error > 0.0).float().mean()
            ),
            "predicted_mass_overestimate_gt_0p05_rate": float(
                (predicted_estimation_error > 0.05).float().mean()
            ),
        }

    output = {
        "config": {
            "trace_paths": [str(path) for path in args.trace_paths],
            "projection_dim": args.projection_dim,
            "sample_fraction": args.sample_fraction,
            "sample_repeats": args.sample_repeats,
            "budget_fractions": fractions,
            "target_mass": args.target_mass,
            "ucb_z": args.ucb_z,
            "overfetch_factor": args.overfetch_factor,
            "verify_target": args.verify_target,
            "verify_ucb_z": args.verify_ucb_z,
            "budget_estimator": args.budget_estimator,
        },
        "overall": build_summary(aggregate),
        "groups": {name: build_summary(rows) for name, rows in grouped.items()},
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["overall"], indent=2), flush=True)


if __name__ == "__main__":
    main()
