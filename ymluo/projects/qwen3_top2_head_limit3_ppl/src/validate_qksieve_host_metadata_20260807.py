#!/usr/bin/env python
"""Validate one-shot host metadata against per-layer device synchronization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import variablebit_spectral_cuda_20260727 as variablebit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20260807)
    device = torch.device("cuda")
    allocations = torch.tensor(
        [
            [8, 4, 2, 1, 0, 0, 0, 0],
            [4, 4, 4, 2, 1, 0, 0, 0],
            [8, 4, 4, 1, 1, 0, 0, 0],
            [4, 4, 2, 2, 2, 1, 0, 0],
            [8, 2, 2, 2, 1, 1, 0, 0],
            [4, 4, 4, 1, 1, 1, 0, 0],
            [8, 4, 2, 2, 1, 0, 0, 0],
            [4, 4, 2, 1, 1, 1, 0, 0],
        ],
        dtype=torch.int8,
        device=device,
    ).unsqueeze(0)
    allocations_host = allocations.cpu()
    capacity = 320
    token_count = 257
    baseline = variablebit.allocate_packed_index(
        allocations, capacity, torch.float16
    )
    optimized = variablebit.allocate_packed_index(
        allocations,
        capacity,
        torch.float16,
        bit_allocations_host=allocations_host,
    )
    projected = torch.randn(
        1, 8, token_count, 128, dtype=torch.float16, device=device
    )
    variablebit.encode_projected_keys_into(projected, baseline, 0)
    variablebit.encode_projected_keys_into(projected, optimized, 0)
    torch.cuda.synchronize()
    tensor_fields = [
        "bit_allocations",
        "code_offsets",
        "scale_offsets",
        "code_bases",
        "scale_bases",
        "code_strides",
        "scale_strides",
    ]
    metadata_exact = {
        field: bool(torch.equal(baseline[field], optimized[field]))
        for field in tensor_fields
    }
    result = {
        "schema": "qksieve_host_metadata_validation_v1",
        "metadata_exact": metadata_exact,
        "packed_codes_exact": bool(
            torch.equal(baseline["packed_codes"], optimized["packed_codes"])
        ),
        "key_scales_exact": bool(
            torch.equal(baseline["key_scales"], optimized["key_scales"])
        ),
        "total_code_bytes_equal": baseline["total_code_bytes"]
        == optimized["total_code_bytes"],
        "total_scale_values_equal": baseline["total_scale_values"]
        == optimized["total_scale_values"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not (
        all(metadata_exact.values())
        and result["packed_codes_exact"]
        and result["key_scales_exact"]
        and result["total_code_bytes_equal"]
        and result["total_scale_values_equal"]
    ):
        raise SystemExit("host metadata changed the packed index")


if __name__ == "__main__":
    main()
