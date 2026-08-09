from __future__ import annotations

import json

import torch

import qabs_cuda_kernels as kernels


def main() -> None:
    candidates = torch.tensor(
        [[[1, 2, 3], [3, 4, 5], [1, 5, 6], [6, 7, 2]]],
        device="cuda",
        dtype=torch.long,
    )
    union, counts = kernels.candidate_union_compact(
        candidates,
        history_count=16,
        group_count=4,
        output_capacity=10,
    )
    count = int(counts[0, 0].item())
    values = set(union[0, 0, :count].cpu().tolist())
    expected = set(range(1, 8))
    if count != len(expected) or values != expected:
        raise AssertionError((count, values, expected))
    print(json.dumps({"count": count, "union": sorted(values)}))


if __name__ == "__main__":
    main()
