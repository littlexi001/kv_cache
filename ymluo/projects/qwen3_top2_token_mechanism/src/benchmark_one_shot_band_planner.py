from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

import qabs_cuda_kernels
from run_head_top2_targeted_ppl_20260714 import (
    _gaussian_tail_density_outside_crossings,
    _plan_one_shot_density_band_counts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", default="8192,32768,65536,131072")
    parser.add_argument("--kv_heads", type=int, default=8)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--band_size", type=int, default=16)
    parser.add_argument("--keep_fraction", type=float, default=0.02)
    parser.add_argument("--candidate_fraction", type=float, default=0.08)
    parser.add_argument("--target_recall", type=float, default=0.95)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser.parse_args()


def elapsed_ms(callable_, repeats: int) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        callable_()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / repeats


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    if args.projection_dim % args.band_size != 0:
        raise ValueError("band size must divide projection dimension")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    query_heads = args.kv_heads * args.groups
    rows: list[dict[str, float | int]] = []

    for token_count in [int(value) for value in args.lengths.split(",")]:
        keep_count = max(1, math.ceil(args.keep_fraction * token_count))
        candidate_count = max(
            keep_count + 1, math.ceil(args.candidate_fraction * token_count)
        )
        scores = torch.randn(
            1, query_heads, token_count, device=device, dtype=torch.float32
        )
        projected_query = torch.randn(
            1,
            args.kv_heads,
            args.groups,
            args.projection_dim,
            device=device,
            dtype=torch.float32,
        )
        anchor = projected_query.clone()
        anchor[..., args.band_size :] = 0.0
        spectral_weights = torch.rand(
            1, args.kv_heads, args.projection_dim, device=device
        ).add_(0.05)
        keep_counts = torch.full(
            (1, query_heads), keep_count, device=device, dtype=torch.long
        )
        residual_sigma = torch.rand(1, query_heads, device=device).add_(0.05)
        planned_bands = torch.empty(
            1, args.kv_heads, device=device, dtype=torch.int32
        )
        crossing_risk = torch.empty(
            1, query_heads, device=device, dtype=torch.float32
        )

        def iterative_three_checks() -> None:
            for _ in range(3):
                _gaussian_tail_density_outside_crossings(
                    scores,
                    residual_sigma,
                    keep_counts,
                    candidate_count,
                )

        def fused_one_shot() -> None:
            top_values = torch.topk(scores, k=candidate_count, dim=-1).values
            qabs_cuda_kernels.one_shot_band_plan(
                projected_query,
                spectral_weights,
                anchor,
                top_values,
                keep_counts,
                planned_bands,
                crossing_risk,
                token_count,
                args.target_recall,
            )

        top_values = torch.topk(scores, k=candidate_count, dim=-1).values

        def candidate_topk_only() -> None:
            torch.topk(scores, k=candidate_count, dim=-1)

        def planner_given_top_values() -> None:
            qabs_cuda_kernels.one_shot_band_plan(
                projected_query,
                spectral_weights,
                anchor,
                top_values,
                keep_counts,
                planned_bands,
                crossing_risk,
                token_count,
                args.target_recall,
            )

        reference_bands, reference_risk = _plan_one_shot_density_band_counts(
            projected_query,
            anchor,
            spectral_weights,
            top_values,
            keep_counts,
            token_count,
            args.target_recall,
            args.band_size,
        )
        qabs_cuda_kernels.one_shot_band_plan(
            projected_query,
            spectral_weights,
            anchor,
            top_values,
            keep_counts,
            planned_bands,
            crossing_risk,
            token_count,
            args.target_recall,
        )
        torch.cuda.synchronize()
        band_match = float((planned_bands == reference_bands).float().mean().item())
        risk_max_error = float((crossing_risk - reference_risk).abs().max().item())

        for _ in range(args.warmup):
            iterative_three_checks()
            fused_one_shot()
            candidate_topk_only()
            planner_given_top_values()
        iterative_ms = elapsed_ms(iterative_three_checks, args.repeats)
        fused_ms = elapsed_ms(fused_one_shot, args.repeats)
        topk_ms = elapsed_ms(candidate_topk_only, args.repeats)
        planner_only_ms = elapsed_ms(planner_given_top_values, args.repeats)
        rows.append(
            {
                "tokens": token_count,
                "keep_count": keep_count,
                "candidate_count": candidate_count,
                "iterative_three_checks_ms": iterative_ms,
                "fused_one_shot_ms": fused_ms,
                "candidate_topk_ms": topk_ms,
                "planner_given_top_values_ms": planner_only_ms,
                "planner_speedup": iterative_ms / fused_ms,
                "planned_band_match": band_match,
                "risk_max_abs_error": risk_max_error,
            }
        )

    payload = {
        "configuration": vars(args) | {"output": str(args.output)},
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
