from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from run_head_top2_targeted_ppl_20260714 import _pca_int4_partial_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure exact rerank overfetch needed at each final attention budget."
    )
    parser.add_argument("--trace_paths", nargs="+", required=True, type=Path)
    parser.add_argument("--output_path", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--final_fractions", default="0.005,0.01,0.02,0.03,0.04")
    parser.add_argument("--overfetch_factors", default="1,2,4")
    parser.add_argument("--fixed_candidate_fraction", type=float, default=0.08)
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "p10": float(torch.quantile(tensor, 0.10)),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "min": float(tensor.min()),
    }


def main() -> None:
    args = parse_args()
    final_fractions = sorted(
        {float(item) for item in args.final_fractions.split(",") if item.strip()}
    )
    overfetch_factors = sorted(
        {int(item) for item in args.overfetch_factors.split(",") if item.strip()}
    )
    if (
        not final_fractions
        or final_fractions[0] <= 0.0
        or final_fractions[-1] > args.fixed_candidate_fraction
    ):
        raise ValueError("final fractions must fit within the fixed candidate fraction")
    if not overfetch_factors or overfetch_factors[0] < 1:
        raise ValueError("overfetch factors must be positive integers")

    metrics: dict[tuple[float, str, str], list[float]] = defaultdict(list)
    device = torch.device(args.device)
    head_cases = 0
    for trace_path in args.trace_paths:
        trace = torch.load(trace_path, map_location="cpu", weights_only=False)
        for record in trace["records"]:
            query = record["query"].to(device)
            key = record["key"].to(device)
            scaling = float(record["scaling"])
            batch_count, query_head_count, _, head_dim = query.shape
            kv_head_count = int(key.shape[1])
            group_count = query_head_count // kv_head_count
            history_count = int(key.shape[2])
            grouped_query = query.squeeze(2).reshape(
                batch_count, kv_head_count, group_count, head_dim
            )
            exact_scores = torch.einsum(
                "bhgd,bhnd->bhgn", grouped_query.float(), key.float()
            ).reshape(query_head_count, history_count) * scaling
            pca_state: dict[str, Any] = {}
            proxy_scores = _pca_int4_partial_scores(
                query.squeeze(2),
                key,
                pca_state,
                args.projection_dim,
                use_chunked_layout=True,
            ).reshape(query_head_count, history_count).float() * scaling

            center = exact_scores.amax(dim=-1, keepdim=True)
            exact_weights = torch.exp(exact_scores - center)
            exact_total = exact_weights.sum(dim=-1)
            fixed_count = min(
                history_count,
                max(1, math.ceil(args.fixed_candidate_fraction * history_count)),
            )
            fixed_proxy_indices = torch.topk(
                proxy_scores, fixed_count, dim=-1, sorted=False
            ).indices
            for fraction in final_fractions:
                final_count = min(history_count, max(1, math.ceil(fraction * history_count)))
                oracle_indices = torch.topk(
                    exact_scores, final_count, dim=-1, sorted=False
                ).indices
                oracle_weight = torch.gather(exact_weights, -1, oracle_indices).sum(dim=-1)
                oracle_mass = oracle_weight / exact_total

                policies: list[tuple[str, torch.Tensor, int]] = []
                for factor in overfetch_factors:
                    candidate_count = min(
                        fixed_count,
                        max(final_count, factor * final_count),
                    )
                    candidate_indices = torch.topk(
                        proxy_scores, candidate_count, dim=-1, sorted=False
                    ).indices
                    candidate_exact = torch.gather(exact_scores, -1, candidate_indices)
                    local_top = torch.topk(
                        candidate_exact, final_count, dim=-1, sorted=False
                    ).indices
                    policies.append(
                        (
                            f"progressive_{factor}x",
                            torch.gather(candidate_indices, -1, local_top),
                            candidate_count,
                        )
                    )
                fixed_exact = torch.gather(exact_scores, -1, fixed_proxy_indices)
                fixed_local_top = torch.topk(
                    fixed_exact, final_count, dim=-1, sorted=False
                ).indices
                policies.append(
                    (
                        "fixed_8pct",
                        torch.gather(fixed_proxy_indices, -1, fixed_local_top),
                        fixed_count,
                    )
                )

                oracle_set = torch.zeros_like(exact_scores, dtype=torch.bool)
                oracle_set.scatter_(1, oracle_indices, True)
                for policy_name, selected_indices, scored_count in policies:
                    selected_weight = torch.gather(
                        exact_weights, -1, selected_indices
                    ).sum(dim=-1)
                    selected_mass = selected_weight / exact_total
                    token_recall = torch.gather(
                        oracle_set, -1, selected_indices
                    ).float().mean(dim=-1)
                    metrics[(fraction, policy_name, "oracle_mass_recall")].extend(
                        (selected_weight / oracle_weight.clamp_min(1.0e-30)).cpu().tolist()
                    )
                    metrics[(fraction, policy_name, "full_attention_mass")].extend(
                        selected_mass.cpu().tolist()
                    )
                    metrics[(fraction, policy_name, "exact_top_token_recall")].extend(
                        token_recall.cpu().tolist()
                    )
                    metrics[(fraction, policy_name, "exact_qk_fraction")].extend(
                        [scored_count / history_count] * query_head_count
                    )
                metrics[(fraction, "oracle", "full_attention_mass")].extend(
                    oracle_mass.cpu().tolist()
                )
            head_cases += query_head_count
            print(f"[{trace_path.stem}] layer={record['layer']} complete", flush=True)
            del query, key, exact_scores, proxy_scores, exact_weights, pca_state
            torch.cuda.empty_cache()

    rows = [
        {
            "final_fraction": fraction,
            "policy": policy,
            "metric": metric,
            **summarize(values),
        }
        for (fraction, policy, metric), values in sorted(metrics.items())
    ]
    output = {
        "config": {
            "trace_paths": [str(path) for path in args.trace_paths],
            "projection_dim": args.projection_dim,
            "final_fractions": final_fractions,
            "overfetch_factors": overfetch_factors,
            "fixed_candidate_fraction": args.fixed_candidate_fraction,
            "head_cases": head_cases,
        },
        "rows": rows,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
