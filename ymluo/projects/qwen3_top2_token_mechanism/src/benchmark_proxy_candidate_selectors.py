from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import torch

import qabs_cuda_kernels as kernels


def measure_ms(callback: Callable[[], object], warmup: int, repeats: int) -> float:
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
def benchmark_length(tokens: int, warmup: int, repeats: int) -> dict[str, object]:
    torch.manual_seed(20260720 + tokens)
    scores = torch.randn((1, 32, tokens), device="cuda", dtype=torch.float32)
    final_count = max(1, math.ceil(0.08 * tokens))
    capacity = math.ceil(1.02 * final_count)
    sampled_capacity = math.ceil(0.20 * tokens)
    sigma = torch.zeros((1, 32), device="cuda", dtype=torch.float32)

    def torch_topk() -> torch.Tensor:
        return torch.topk(scores, final_count, dim=-1, sorted=False).indices

    def radix_select() -> tuple[torch.Tensor, ...]:
        return kernels.direct_uncertainty_candidates(
            scores,
            sigma,
            final_count,
            capacity,
            0.0,
        )

    def sampled_quantile_select() -> tuple[torch.Tensor, ...]:
        return kernels.sampled_quantile_candidates(
            scores,
            sample_count=256,
            selected_fraction=0.12,
            candidate_capacity=sampled_capacity,
        )

    exact = torch_topk()
    selected, counts, _, overflow = radix_select()
    sampled, sampled_counts, _, sampled_overflow = sampled_quantile_select()
    recalls = []
    count_values = []
    for head in range(scores.shape[1]):
        count = min(int(counts[0, head]), capacity)
        count_values.append(count)
        selected_set = set(selected[0, head, :count].cpu().tolist())
        exact_set = set(exact[0, head].cpu().tolist())
        recalls.append(len(selected_set & exact_set) / final_count)
    sampled_recalls = []
    sampled_count_values = []
    for head in range(scores.shape[1]):
        count = min(int(sampled_counts[0, head]), sampled_capacity)
        sampled_count_values.append(count)
        sampled_set = set(sampled[0, head, :count].cpu().tolist())
        exact_set = set(exact[0, head].cpu().tolist())
        sampled_recalls.append(len(sampled_set & exact_set) / final_count)
    return {
        "tokens": tokens,
        "candidate_count": final_count,
        "capacity": capacity,
        "torch_topk_ms": measure_ms(torch_topk, warmup, repeats),
        "radix_select_ms": measure_ms(radix_select, warmup, repeats),
        "sampled_quantile_ms": measure_ms(
            sampled_quantile_select, warmup, repeats
        ),
        "radix_recall_mean": sum(recalls) / len(recalls),
        "radix_recall_min": min(recalls),
        "radix_selected_count_mean": sum(count_values) / len(count_values),
        "radix_selected_count_max": max(count_values),
        "overflow_rate": float(overflow.float().mean()),
        "sampled_recall_mean": sum(sampled_recalls) / len(sampled_recalls),
        "sampled_recall_min": min(sampled_recalls),
        "sampled_selected_count_mean": sum(sampled_count_values)
        / len(sampled_count_values),
        "sampled_selected_count_max": max(sampled_count_values),
        "sampled_overflow_rate": float(sampled_overflow.float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="8192,32768,65536,131072")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "hardware": torch.cuda.get_device_name(),
        "rows": [
            benchmark_length(int(length), args.warmup, args.repeats)
            for length in args.lengths.split(",")
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
