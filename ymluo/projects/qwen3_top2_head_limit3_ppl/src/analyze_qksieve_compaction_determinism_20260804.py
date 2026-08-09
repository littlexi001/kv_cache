#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import mixedblock_spectral_cuda_20260729 as mixed_cuda
import qksieve_query_cuda_20260728 as query_cuda
import variablebit_spectral_cuda_20260727 as varbit_cuda
from benchmark_variablebit_spectral_attention_20260727 import (
    ALLOCATION_PROFILES,
)
from validate_qksieve_valuesketch_oneexp_ab_20260804 import (
    allocate_outputs,
    launch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_tokens", type=int, default=32_768)
    parser.add_argument("--sample_count", type=int, default=1024)
    parser.add_argument("--rank", type=int, choices=(8, 12, 16, 32), default=16)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    history = args.history_tokens
    rank = args.rank
    block_size = 256
    scaling = 128.0**-0.5
    selected = min(history, 1280, max(256, math.ceil(0.06 * history)))
    selected_fraction = selected / history

    query = torch.randn(1, 32, 128, dtype=dtype, device=device)
    grouped_query = query.reshape(1, 8, 4, 128)
    key_basis = torch.randn(1, 8, 128, 128, dtype=dtype, device=device)
    query_codes, query_scales = query_cuda.project_quantize(
        grouped_query, key_basis
    )
    allocation = ALLOCATION_PROFILES["qmse_total_b15"].unsqueeze(0).cuda()
    index = varbit_cuda.allocate_packed_index(allocation, history, dtype)
    index["packed_codes"].random_(0, 256)
    index["key_scales"].uniform_(0.01, 0.1)
    block_count = math.ceil(history / block_size)
    value_codes = torch.randint(
        0,
        256,
        (1, 8, history, rank // 2),
        dtype=torch.uint8,
        device=device,
    )
    value_minimum = torch.randn(
        1, 8, block_count, rank, dtype=dtype, device=device
    )
    value_scale = torch.rand_like(value_minimum).mul_(0.1).add_(0.01)
    extension = mixed_cuda.load_extension()

    def run_once() -> dict[str, torch.Tensor]:
        outputs = allocate_outputs(history, rank, device)
        launch(
            extension,
            query_codes,
            query_scales,
            index,
            value_codes,
            value_minimum,
            value_scale,
            outputs,
            history,
            args.sample_count,
            selected_fraction,
            rank,
            block_size,
            scaling,
        )
        torch.cuda.synchronize()
        return {name: tensor.clone() for name, tensor in outputs.items()}

    reference = run_once()
    count_mismatch_runs = 0
    set_mismatch_runs = 0
    order_mismatch_runs = 0
    maximum_tail_denominator_error = 0.0
    maximum_tail_coefficient_error = 0.0
    maximum_order_mismatch_rows = 0
    for _ in range(args.repeats - 1):
        current = run_once()
        if not torch.equal(reference["counts"], current["counts"]):
            count_mismatch_runs += 1
        set_mismatch = False
        order_mismatch_rows = 0
        for row in range(32):
            count = int(reference["counts"].reshape(-1)[row].item())
            current_count = int(current["counts"].reshape(-1)[row].item())
            if count != current_count:
                set_mismatch = True
                order_mismatch_rows += 1
                continue
            reference_indices = reference["indices"].reshape(32, history)[
                row, :count
            ]
            current_indices = current["indices"].reshape(32, history)[
                row, :count
            ]
            if not torch.equal(reference_indices, current_indices):
                order_mismatch_rows += 1
            if not torch.equal(
                reference_indices.sort().values,
                current_indices.sort().values,
            ):
                set_mismatch = True
        if set_mismatch:
            set_mismatch_runs += 1
        if order_mismatch_rows:
            order_mismatch_runs += 1
        maximum_order_mismatch_rows = max(
            maximum_order_mismatch_rows, order_mismatch_rows
        )
        maximum_tail_denominator_error = max(
            maximum_tail_denominator_error,
            float(
                (
                    reference["tail_denominator"]
                    - current["tail_denominator"]
                )
                .abs()
                .max()
                .item()
            ),
        )
        maximum_tail_coefficient_error = max(
            maximum_tail_coefficient_error,
            float(
                (
                    reference["tail_coefficients"]
                    - current["tail_coefficients"]
                )
                .abs()
                .max()
                .item()
            ),
        )

    result = {
        "history_tokens": history,
        "sample_count": args.sample_count,
        "rank": rank,
        "repeats": args.repeats,
        "count_mismatch_runs": count_mismatch_runs,
        "candidate_set_mismatch_runs": set_mismatch_runs,
        "candidate_order_mismatch_runs": order_mismatch_runs,
        "maximum_order_mismatch_rows": maximum_order_mismatch_rows,
        "maximum_tail_denominator_abs_error": maximum_tail_denominator_error,
        "maximum_tail_coefficient_abs_error": maximum_tail_coefficient_error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
