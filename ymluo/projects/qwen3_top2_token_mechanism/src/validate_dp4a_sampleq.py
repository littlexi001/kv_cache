from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import torch

import qabs_cuda_kernels
from run_head_top2_targeted_ppl_20260714 import (
    qabs_sampled_head_adaptive_attention,
)


def extend_history(tensor: torch.Tensor, target_count: int) -> torch.Tensor:
    repeats = math.ceil(target_count / tensor.shape[-2])
    return tensor.repeat(1, 1, repeats, 1)[..., :target_count, :].contiguous()


def measure_ms(callback: Callable[[], Any], warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        callback()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        callback()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / repeats)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history_tokens", type=int, default=131072)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    payload = torch.load(args.trace, map_location="cpu", weights_only=False)
    records = [
        row for row in payload["records"] if int(row["layer"]) == args.layer
    ]
    first = next(row for row in records if row.get("key") is not None)
    query = records[0]["query"].cuda().half().contiguous()
    key = extend_history(first["key"].cuda().half(), args.history_tokens + 1)
    value = extend_history(first["value"].cuda().half(), args.history_tokens + 1)
    state: dict[str, Any] = {"rope_theta": 5_000_000.0}
    qabs_sampled_head_adaptive_attention(
        query,
        key,
        value,
        attention_mask=None,
        scaling=float(first["scaling"]),
        mass_threshold=0.75,
        budget_fractions=(0.02,),
        sample_fraction=0.0025,
        qabs_dim_count=8,
        candidate_fraction=0.06,
        use_cuda_kernels=True,
        skip_candidate_rerank=False,
        score_mode="pca_int4_chunked_logscale16_qkmetric_sampleq_autosplit",
        projection_dim=48,
        pca_state=state,
    )

    history_count = args.history_tokens
    candidate_capacity = math.ceil(0.12 * history_count)

    def run(use_dp4a: bool) -> tuple[torch.Tensor, ...]:
        return qabs_cuda_kernels.pca_int4_logscale16_sampled_quantile_candidates(
            state["last_projected_query_codes"].contiguous(),
            state["packed_chunked"],
            state["scales"],
            state["logscale_exponents"],
            history_count,
            256,
            0.06,
            candidate_capacity,
            use_dp4a=use_dp4a,
        )

    scalar = run(False)
    dp4a = run(True)
    torch.cuda.synchronize()
    (
        scalar_indices,
        scalar_scores,
        scalar_counts,
        scalar_boundaries,
        scalar_overflow,
    ) = scalar
    (
        dp4a_indices,
        dp4a_scores,
        dp4a_counts,
        dp4a_boundaries,
        dp4a_overflow,
    ) = dp4a
    count_equal = bool(torch.equal(scalar_counts, dp4a_counts))
    boundary_max_abs = float(
        (scalar_boundaries - dp4a_boundaries).abs().max().item()
    )
    set_equal = count_equal
    score_max_abs = 0.0
    if count_equal:
        for row in range(scalar_indices.shape[1]):
            count = int(scalar_counts[0, row].item())
            scalar_order = torch.argsort(scalar_indices[0, row, :count])
            dp4a_order = torch.argsort(dp4a_indices[0, row, :count])
            scalar_sorted = scalar_indices[0, row, :count][scalar_order]
            dp4a_sorted = dp4a_indices[0, row, :count][dp4a_order]
            if not torch.equal(scalar_sorted, dp4a_sorted):
                set_equal = False
                continue
            error = (
                scalar_scores[0, row, :count][scalar_order]
                - dp4a_scores[0, row, :count][dp4a_order]
            ).abs().max()
            score_max_abs = max(score_max_abs, float(error.item()))

    scalar_ms = measure_ms(lambda: run(False), args.warmup, args.repeats)
    dp4a_ms = measure_ms(lambda: run(True), args.warmup, args.repeats)

    report = {
        "hardware": torch.cuda.get_device_name(),
        "trace": str(args.trace),
        "history_tokens": history_count,
        "layer": args.layer,
        "candidate_fraction_mean": float(scalar_counts.float().mean().item())
        / history_count,
        "count_equal": count_equal,
        "overflow_equal": bool(torch.equal(scalar_overflow, dp4a_overflow)),
        "candidate_set_equal": set_equal,
        "boundary_max_abs": boundary_max_abs,
        "candidate_score_max_abs": score_max_abs,
        "scalar_ms": scalar_ms,
        "dp4a_ms": dp4a_ms,
        "dp4a_speedup": scalar_ms / dp4a_ms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
