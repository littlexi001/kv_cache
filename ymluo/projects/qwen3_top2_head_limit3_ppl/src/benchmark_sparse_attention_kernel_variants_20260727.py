from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

import qabs_cuda_kernels as kernels


def parse_ints(value: str) -> list[int]:
    values = sorted({int(item) for item in value.split(",") if item.strip()})
    if not values:
        raise ValueError("expected at least one integer")
    return values


def parse_floats(value: str) -> list[float]:
    values = sorted({float(item) for item in value.split(",") if item.strip()})
    if not values:
        raise ValueError("expected at least one floating-point value")
    return values


def measure_ms(
    function: Callable[[], object],
    warmup: int,
    iterations: int,
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


def make_indices(
    history_tokens: int,
    selected_count: int,
    sorted_by_position: bool,
) -> torch.Tensor:
    rank = torch.arange(selected_count, device="cuda", dtype=torch.long)
    head = torch.arange(32, device="cuda", dtype=torch.long).view(-1, 1)
    indices = (rank.view(1, -1) * 8191 + head * 131071) % history_tokens
    if sorted_by_position:
        indices = torch.sort(indices, dim=-1).values
    return indices.unsqueeze(0).contiguous()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare exact ragged sparse-attention CUDA kernels under GQA."
        )
    )
    parser.add_argument(
        "--lengths", default="8192,16384,32768,65536,131072"
    )
    parser.add_argument("--selected_fractions", default="0.02,0.04,0.06")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--full_iterations", type=int, default=30)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(20260727)
    rows: list[dict[str, float | int | bool | str]] = []

    for history_tokens in parse_ints(args.lengths):
        query = torch.randn(
            1, 32, 128, dtype=torch.float16, device="cuda"
        )
        key = torch.randn(
            1,
            8,
            history_tokens + 1,
            128,
            dtype=torch.float16,
            device="cuda",
        )
        value = torch.randn_like(key)
        scaling = 128.0**-0.5

        def full_attention() -> torch.Tensor:
            return F.scaled_dot_product_attention(
                query.unsqueeze(2),
                key,
                value,
                enable_gqa=True,
            )

        measured_full_iterations = (
            min(args.full_iterations, 20)
            if history_tokens >= 65536
            else args.full_iterations
        )
        full_ms = measure_ms(
            full_attention,
            min(args.warmup, 10),
            measured_full_iterations,
        )

        for selected_fraction in parse_floats(args.selected_fractions):
            selected_count = min(
                history_tokens,
                max(1, math.ceil(selected_fraction * history_tokens)),
            )
            counts = torch.full(
                (1, 32),
                selected_count,
                device="cuda",
                dtype=torch.long,
            )
            measured_iterations = (
                min(args.iterations, 20)
                if history_tokens >= 65536
                else args.iterations
            )

            for order in ("unsorted", "position_sorted"):
                indices = make_indices(
                    history_tokens,
                    selected_count,
                    sorted_by_position=order == "position_sorted",
                )

                functions: dict[str, Callable[[], torch.Tensor]] = {
                    "ragged_self": lambda: kernels.final_attention_ragged_self(
                        query, key, value, indices, counts, scaling
                    ),
                    "ragged_self_warp": (
                        lambda: kernels.final_attention_ragged_self_warp(
                            query, key, value, indices, counts, scaling
                        )
                    ),
                }
                for split_count in (2, 4, 8, 16):
                    functions[f"ragged_self_split{split_count}"] = (
                        lambda split_count=split_count: (
                            kernels.final_attention_ragged_self_split(
                                query,
                                key,
                                value,
                                indices,
                                counts,
                                scaling,
                                split_count,
                            )
                        )
                    )

                reference = functions["ragged_self_warp"]()
                for method, function in functions.items():
                    output = function()
                    max_abs_error = float(
                        (output - reference).abs().max().item()
                    )
                    elapsed_ms = measure_ms(
                        function, args.warmup, measured_iterations
                    )
                    rows.append(
                        {
                            "history_tokens": history_tokens,
                            "selected_fraction": (
                                selected_count / history_tokens
                            ),
                            "selected_count": selected_count,
                            "order": order,
                            "method": method,
                            "elapsed_ms": elapsed_ms,
                            "full_sdpa_ms": full_ms,
                            "full_sdpa_over_sparse": full_ms / elapsed_ms,
                            "max_abs_error_vs_warp": max_abs_error,
                        }
                    )

                self_indices = torch.full(
                    (1, 32, 1),
                    history_tokens,
                    device="cuda",
                    dtype=torch.long,
                )
                indices_with_self = torch.cat((indices, self_indices), dim=-1)
                counts_with_self = counts + 1

                def exact_scores() -> torch.Tensor:
                    return kernels.candidate_compact_scores_ragged(
                        query,
                        key,
                        indices_with_self,
                        counts_with_self,
                        scaling,
                    )

                scores_with_self = exact_scores()
                for split_count in (2, 4, 8, 16):

                    def value_from_scores(
                        split_count: int = split_count,
                    ) -> torch.Tensor:
                        return kernels.final_attention_from_scores_split(
                            value,
                            indices_with_self,
                            scores_with_self,
                            counts_with_self,
                            split_count,
                        )

                    value_output = value_from_scores()
                    error = float(
                        (value_output - reference).abs().max().item()
                    )
                    score_ms = measure_ms(
                        exact_scores, args.warmup, measured_iterations
                    )
                    value_ms = measure_ms(
                        value_from_scores,
                        args.warmup,
                        measured_iterations,
                    )

                    def two_kernel() -> torch.Tensor:
                        current_scores = exact_scores()
                        return kernels.final_attention_from_scores_split(
                            value,
                            indices_with_self,
                            current_scores,
                            counts_with_self,
                            split_count,
                        )

                    total_ms = measure_ms(
                        two_kernel, args.warmup, measured_iterations
                    )
                    rows.append(
                        {
                            "history_tokens": history_tokens,
                            "selected_fraction": (
                                selected_count / history_tokens
                            ),
                            "selected_count": selected_count,
                            "order": order,
                            "method": f"score_then_value_split{split_count}",
                            "elapsed_ms": total_ms,
                            "exact_score_ms": score_ms,
                            "value_from_score_ms": value_ms,
                            "full_sdpa_ms": full_ms,
                            "full_sdpa_over_sparse": full_ms / total_ms,
                            "max_abs_error_vs_warp": error,
                        }
                    )
                del (
                    indices,
                    reference,
                    self_indices,
                    indices_with_self,
                    counts_with_self,
                    scores_with_self,
                )

            del counts
        del query, key, value
        torch.cuda.empty_cache()

    output = {
        "config": {
            **vars(args),
            "output": str(args.output) if args.output else None,
        },
        "scope": (
            "One decode attention layer, 32 query heads, 8 KV heads, "
            "head dimension 128. The implicit current token is included. "
            "Index selection and all non-attention model work are excluded."
        ),
        "rows": rows,
    }
    text = json.dumps(output, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
