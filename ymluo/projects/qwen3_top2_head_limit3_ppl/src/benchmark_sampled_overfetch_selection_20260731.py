from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import qabs_cuda_kernels as qabs_cuda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="32768,65536,131072")
    parser.add_argument("--multipliers", default="2,3,4")
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def measure_ms(
    callback,
    warmup: int,
    repetitions: int,
) -> float:
    for _ in range(warmup):
        callback()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        callback()
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop)) / repetitions


def final_budget(history_count: int) -> int:
    return 1280 if history_count <= 65536 else 2560


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    lengths = [int(value) for value in args.lengths.split(",")]
    multipliers = [
        float(value) for value in args.multipliers.split(",")
    ]
    rows: list[dict[str, float | int | bool]] = []
    for history_count in lengths:
        budget = final_budget(history_count)
        remote_count = budget - 16 - 128
        remote_available = history_count - 16 - 128
        generator = torch.Generator(device="cuda")
        generator.manual_seed(args.seed + history_count)
        scores = torch.randn(
            1,
            32,
            remote_available,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        exact_target = torch.topk(
            scores,
            k=remote_count,
            dim=-1,
            sorted=False,
        ).indices

        for multiplier in multipliers:
            requested = min(
                remote_available,
                max(
                    remote_count,
                    int(math.ceil(multiplier * remote_count)),
                ),
            )
            selected_fraction = requested / remote_available
            sample_keep = max(
                1,
                math.ceil(selected_fraction * args.sample_count),
            )
            effective_fraction = sample_keep / args.sample_count
            capacity = min(
                remote_available,
                max(
                    requested,
                    math.ceil(
                        remote_available
                        * min(1.0, effective_fraction * 2.0)
                    ),
                ),
            )

            recall_values: list[torch.Tensor] = []
            count_values: list[torch.Tensor] = []
            overflow_count = 0
            underfilled_count = 0
            for trial in range(args.trials):
                if trial:
                    scores.normal_(generator=generator)
                    exact_target = torch.topk(
                        scores,
                        k=remote_count,
                        dim=-1,
                        sorted=False,
                    ).indices
                indices, counts, _, overflow = (
                    qabs_cuda.sampled_quantile_candidates(
                        scores,
                        args.sample_count,
                        selected_fraction,
                        capacity,
                    )
                )
                valid = (
                    torch.arange(capacity, device="cuda")
                    .view(1, 1, capacity)
                    < counts.unsqueeze(-1)
                )
                sentinel = torch.full_like(indices, remote_available)
                safe_indices = torch.where(valid, indices, sentinel)
                selected_mask = torch.zeros(
                    1,
                    scores.shape[1],
                    remote_available + 1,
                    dtype=torch.bool,
                    device="cuda",
                )
                selected_mask.scatter_(2, safe_indices, True)
                recall = (
                    selected_mask.gather(2, exact_target)
                    .float()
                    .mean(dim=-1)
                )
                recall_values.append(recall.cpu())
                count_values.append(counts.cpu())
                underfilled_count += int(
                    (counts < remote_count).sum().item()
                )
                overflow_count += int(overflow.sum().item())

            def torch_topk() -> torch.Tensor:
                return torch.topk(
                    scores,
                    k=requested,
                    dim=-1,
                    sorted=False,
                ).indices

            def sampled_select() -> tuple[torch.Tensor, ...]:
                return qabs_cuda.sampled_quantile_candidates(
                    scores,
                    args.sample_count,
                    selected_fraction,
                    capacity,
                )

            topk_ms = measure_ms(
                torch_topk,
                args.warmup,
                args.repetitions,
            )
            sampled_ms = measure_ms(
                sampled_select,
                args.warmup,
                args.repetitions,
            )
            recall_tensor = torch.cat(recall_values).reshape(-1)
            count_tensor = (
                torch.cat(count_values).reshape(-1).float()
            )
            row_count = int(recall_tensor.numel())
            rows.append(
                {
                    "history_count": history_count,
                    "final_budget": budget,
                    "remote_target_count": remote_count,
                    "requested_multiplier": multiplier,
                    "requested_candidate_count": requested,
                    "sample_count": args.sample_count,
                    "sample_keep": sample_keep,
                    "effective_sample_fraction": effective_fraction,
                    "candidate_capacity": capacity,
                    "rows": row_count,
                    "mean_candidate_count": float(
                        count_tensor.mean().item()
                    ),
                    "p01_candidate_count": float(
                        torch.quantile(count_tensor, 0.01).item()
                    ),
                    "minimum_candidate_count": int(
                        count_tensor.min().item()
                    ),
                    "underfilled_row_rate": (
                        underfilled_count / row_count
                    ),
                    "overflow_row_rate": (
                        overflow_count / row_count
                    ),
                    "mean_exact_target_recall": float(
                        recall_tensor.mean().item()
                    ),
                    "p01_exact_target_recall": float(
                        torch.quantile(recall_tensor, 0.01).item()
                    ),
                    "minimum_exact_target_recall": float(
                        recall_tensor.min().item()
                    ),
                    "torch_topk_ms": topk_ms,
                    "sampled_compaction_ms": sampled_ms,
                    "selection_speedup": topk_ms / sampled_ms,
                }
            )

    payload = {
        "schema": "sampled_overfetch_selection_benchmark_v1",
        "seed": args.seed,
        "trials": args.trials,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
