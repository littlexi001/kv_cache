from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

from run_head_top2_targeted_ppl_20260714 import (
    qabs_sampled_head_adaptive_attention,
)


def extend_history(tensor: torch.Tensor, target_count: int) -> torch.Tensor:
    repeats = math.ceil(target_count / tensor.shape[-2])
    return tensor.repeat(1, 1, repeats, 1)[..., :target_count, :].contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history_tokens", type=int, default=131072)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--rope_theta", type=float, default=5_000_000.0)
    parser.add_argument(
        "--score_mode",
        default="pca_int4_chunked_logscale16_qkmetric_sampleq_autosplit",
    )
    args = parser.parse_args()

    payload = torch.load(args.trace, map_location="cpu", weights_only=False)
    records = [row for row in payload["records"] if int(row["layer"]) == args.layer]
    first = next(row for row in records if row.get("key") is not None)
    queries = [row["query"].cuda().half().contiguous() for row in records]
    key = extend_history(first["key"].cuda().half(), args.history_tokens + 1)
    value = extend_history(first["value"].cuda().half(), args.history_tokens + 1)
    scaling = float(first["scaling"])
    state: dict[str, Any] = {"rope_theta": args.rope_theta}

    def sparse(query: torch.Tensor) -> torch.Tensor:
        return qabs_sampled_head_adaptive_attention(
            query,
            key,
            value,
            attention_mask=None,
            scaling=scaling,
            mass_threshold=0.75,
            budget_fractions=(0.02,),
            sample_fraction=0.0025,
            qabs_dim_count=8,
            candidate_fraction=0.06,
            use_cuda_kernels=True,
            skip_candidate_rerank=False,
            score_mode=args.score_mode,
            projection_dim=48,
            pca_state=state,
        )[0]

    with torch.inference_mode():
        for index in range(args.warmup):
            sparse(queries[index % len(queries)])
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for index in range(args.steps):
                sparse(queries[(args.warmup + index) % len(queries)])
        torch.cuda.synchronize()

    events = []
    for event in prof.key_averages():
        cuda_us = float(getattr(event, "self_device_time_total", 0.0))
        if cuda_us <= 0.0:
            continue
        events.append(
            {
                "name": event.key,
                "self_cuda_us_total": cuda_us,
                "self_cuda_us_per_step": cuda_us / args.steps,
                "calls": int(event.count),
            }
        )
    events.sort(key=lambda row: row["self_cuda_us_total"], reverse=True)
    total_cuda_us = sum(row["self_cuda_us_total"] for row in events)
    report = {
        "hardware": torch.cuda.get_device_name(),
        "history_tokens": args.history_tokens,
        "layer": args.layer,
        "score_mode": args.score_mode,
        "profiled_steps": args.steps,
        "total_self_cuda_us_per_step": total_cuda_us / args.steps,
        "events": events,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
