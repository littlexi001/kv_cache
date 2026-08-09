#!/usr/bin/env python
"""Check split sparse-attention determinism and accuracy for GQA decode."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

import qabs_cuda_kernels as sparse_cuda


def measure_ms(
    function: Callable[[], object], warmup: int, iterations: int
) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / iterations


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="32768,65536,131072")
    parser.add_argument("--candidate_count", type=int, default=1280)
    parser.add_argument("--capacity_fraction", type=float, default=0.06)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20260729)
    rows = []
    for history_count in sorted(
        {int(item) for item in args.lengths.split(",") if item}
    ):
        candidate_count = min(args.candidate_count, history_count)
        capacity = max(
            candidate_count,
            math.ceil(args.capacity_fraction * history_count),
        )
        query = torch.randn(
            1, 32, 128, dtype=torch.float16, device="cuda"
        )
        key = torch.randn(
            1,
            8,
            history_count + 1,
            128,
            dtype=torch.float16,
            device="cuda",
        )
        value = torch.randn_like(key)
        indices = torch.empty(
            1, 32, capacity, dtype=torch.long, device="cuda"
        )
        for head in range(32):
            indices[0, head, :candidate_count] = torch.randperm(
                history_count, device="cuda"
            )[:candidate_count]
        counts = torch.full(
            (1, 32),
            candidate_count,
            dtype=torch.long,
            device="cuda",
        )
        scaling = 128.0**-0.5
        expanded_key = key.repeat_interleave(4, dim=1)
        expanded_value = value.repeat_interleave(4, dim=1)
        gather_indices = indices[..., :candidate_count].unsqueeze(-1).expand(
            1, 32, candidate_count, 128
        )
        gathered_key = expanded_key.gather(2, gather_indices)
        gathered_value = expanded_value.gather(2, gather_indices)
        self_key = expanded_key[..., -1:, :]
        self_value = expanded_value[..., -1:, :]
        reference = F.scaled_dot_product_attention(
            query.unsqueeze(2),
            torch.cat((gathered_key, self_key), dim=2),
            torch.cat((gathered_value, self_value), dim=2),
            scale=scaling,
        )

        def unsplit() -> torch.Tensor:
            return sparse_cuda.final_attention_ragged_self(
                query, key, value, indices, counts, scaling
            )

        methods: dict[str, Callable[[], torch.Tensor]] = {
            "unsplit": unsplit
        }
        for split in (2, 4, 8, 16):
            methods[f"split{split}"] = (
                lambda split=split: sparse_cuda.final_attention_ragged_self_split(
                    query,
                    key,
                    value,
                    indices,
                    counts,
                    scaling,
                    split,
                )
            )
        outputs = {}
        for name, function in methods.items():
            first = function()
            torch.cuda.synchronize()
            repeats = []
            for _ in range(4):
                repeats.append(function())
                torch.cuda.synchronize()
            repeat_error = max(
                float((first - output).abs().max().item())
                for output in repeats
            )
            latency = measure_ms(
                function, args.warmup, args.iterations
            )
            outputs[name] = first
            rows.append(
                {
                    "history_count": history_count,
                    "candidate_count": candidate_count,
                    "candidate_capacity": capacity,
                    "method": name,
                    "latency_ms": latency,
                    "repeat_max_abs_error": repeat_error,
                    "reference_max_abs_error": float(
                        (
                            first.transpose(1, 2) - reference
                        ).abs().max().item()
                    ),
                    "reference_mean_abs_error": float(
                        (
                            first.transpose(1, 2) - reference
                        ).abs().mean().item()
                    ),
                }
            )
        for row in rows[-len(methods) :]:
            row["speedup_vs_unsplit"] = (
                next(
                    item["latency_ms"]
                    for item in rows[-len(methods) :]
                    if item["method"] == "unsplit"
                )
                / row["latency_ms"]
            )
            print(json.dumps(row, sort_keys=True), flush=True)
        del (
            query,
            key,
            value,
            indices,
            counts,
            expanded_key,
            expanded_value,
            gathered_key,
            gathered_value,
            reference,
            outputs,
        )
        torch.cuda.empty_cache()
    output = {
        "scope": (
            "Qwen-style GQA, exact K/V, identical fixed candidate order, "
            "implicit self token, and production qabs sparse-attention kernels."
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
