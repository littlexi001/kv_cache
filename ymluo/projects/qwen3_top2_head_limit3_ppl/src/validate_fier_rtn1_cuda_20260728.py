from __future__ import annotations

import argparse
import json

import torch

from fier_rtn1_cuda_20260728 import (
    allocate_packed_index,
    reconstruct_keys,
    scores,
    update_packed_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate packed FIER RTN1 CUDA encode and scan kernels."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this validation requires a CUDA device")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    keys = torch.randn(
        2,
        2,
        69,
        128,
        generator=generator,
        dtype=torch.float16,
        device=device,
    )
    query = torch.randn(
        2,
        8,
        128,
        generator=generator,
        dtype=torch.float16,
        device=device,
    )

    one_shot = allocate_packed_index(2, 2, 96, device)
    update_packed_index(keys, one_shot, 69)
    incremental = allocate_packed_index(2, 2, 96, device)
    for history_count in (1, 17, 32, 33, 64, 69):
        update_packed_index(keys, incremental, history_count)

    for name in ("packed_codes", "lower", "upper"):
        if not torch.equal(one_shot[name], incremental[name]):
            raise AssertionError(
                f"incremental and one-shot {name} differ"
            )

    reconstructed = reconstruct_keys(one_shot, 69)
    expected_scores = torch.einsum(
        "bhgd,bhkd->bhgk",
        query.float().reshape(2, 2, 4, 128),
        reconstructed,
    ).reshape(2, 8, 69)
    actual_scores = scores(query, one_shot, 69)
    score_error = (actual_scores - expected_scores).abs()
    max_error = float(score_error.max().item())
    mean_error = float(score_error.mean().item())
    if max_error > 5.0e-3:
        raise AssertionError(
            f"packed FIER CUDA score error is too large: {max_error}"
        )
    expected_topk = torch.topk(
        expected_scores, k=8, dim=-1, sorted=False
    ).indices
    actual_topk = torch.topk(
        actual_scores, k=8, dim=-1, sorted=False
    ).indices
    topk_overlap = float(
        (
            actual_topk.unsqueeze(-1)
            == expected_topk.unsqueeze(-2)
        )
        .any(dim=-1)
        .float()
        .mean()
        .item()
    )
    if topk_overlap < 1.0:
        raise AssertionError(
            f"packed FIER CUDA top-k mismatch: {topk_overlap}"
        )
    print(
        json.dumps(
            {
                "status": "passed",
                "max_score_abs_error": max_error,
                "mean_score_abs_error": mean_error,
                "top8_overlap": topk_overlap,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
