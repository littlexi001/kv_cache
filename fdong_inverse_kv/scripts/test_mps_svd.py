#!/usr/bin/env python3
"""Check whether the exact SVD path used by QK analysis works on MPS."""

from __future__ import annotations

import argparse
import json
import time
from typing import List, Tuple

import torch


def parse_shapes(spec: str) -> List[Tuple[int, int]]:
    shapes = []
    for item in spec.split(","):
        rows, cols = item.lower().split("x")
        shapes.append((int(rows), int(cols)))
    return shapes


def synchronize() -> None:
    if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def randomized_svd(
    matrix: torch.Tensor,
    rank: int,
    oversample: int,
    power_iters: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_rank = min(rank, matrix.shape[0], matrix.shape[1])
    sketch_rank = min(target_rank + oversample, matrix.shape[0], matrix.shape[1])
    omega = torch.randn(matrix.shape[1], sketch_rank, dtype=matrix.dtype, device=matrix.device)
    y = matrix @ omega
    for _ in range(power_iters):
        # MPS does not implement linalg_qr. QR is only applied to the skinny sketch.
        q, _ = torch.linalg.qr(y.cpu(), mode="reduced")
        q = q.to(matrix.device)
        y = matrix @ (matrix.transpose(0, 1) @ q)
    q, _ = torch.linalg.qr(y.cpu(), mode="reduced")
    q = q.to(matrix.device)

    # Only this compressed matrix uses CPU LAPACK.
    compressed = (q.transpose(0, 1) @ matrix).cpu()
    u_small, s, vh = torch.linalg.svd(compressed, full_matrices=False)
    u = q @ u_small.to(matrix.device)
    return u[:, :target_rank], s[:target_rank].to(matrix.device), vh[:target_rank].to(matrix.device)


def run_case(
    rows: int,
    cols: int,
    seed: int,
    method: str,
    rank: int,
    oversample: int,
    power_iters: int,
) -> dict:
    torch.manual_seed(seed)
    matrix = torch.randn(rows, cols, dtype=torch.float32, device="mps")
    synchronize()

    started = time.perf_counter()
    if method == "exact":
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    else:
        u, s, vh = randomized_svd(matrix, rank, oversample, power_iters)
    synchronize()
    elapsed = time.perf_counter() - started

    reconstructed = (u * s.unsqueeze(0)) @ vh
    relative_error = (
        torch.linalg.vector_norm(reconstructed - matrix)
        / torch.linalg.vector_norm(matrix).clamp_min(1e-12)
    )

    output_rank = s.numel()
    identity = torch.eye(output_rank, dtype=torch.float32, device="mps")
    u_error = torch.linalg.vector_norm(u.transpose(0, 1) @ u - identity) / output_rank
    vh_error = torch.linalg.vector_norm(vh @ vh.transpose(0, 1) - identity) / output_rank
    synchronize()

    return {
        "shape": [rows, cols],
        "method": method,
        "requested_rank": rank,
        "input_dtype": str(matrix.dtype),
        "u_shape": list(u.shape),
        "s_shape": list(s.shape),
        "vh_shape": list(vh.shape),
        "output_device": str(s.device),
        "elapsed_seconds": elapsed,
        "relative_reconstruction_error": float(relative_error.item()),
        "u_orthogonality_error": float(u_error.item()),
        "vh_orthogonality_error": float(vh_error.item()),
        "finite": bool(torch.isfinite(s).all().item()),
        "passed": bool(
            torch.isfinite(s).all().item()
            and (relative_error.item() < 1e-4 if output_rank == min(rows, cols) else relative_error.item() < 1.0)
            and u_error.item() < 1e-3
            and vh_error.item() < 1e-3
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default="128x64")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", choices=["exact", "randomized"], default="exact")
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--oversample", type=int, default=32)
    parser.add_argument("--power_iters", type=int, default=2)
    args = parser.parse_args()

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available in this Python environment.")

    results = []
    for rows, cols in parse_shapes(args.shapes):
        print(f"[mps-svd] testing shape={rows}x{cols} dtype=float32", flush=True)
        result = run_case(
            rows,
            cols,
            args.seed,
            args.method,
            args.rank,
            args.oversample,
            args.power_iters,
        )
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    if not all(item["passed"] for item in results):
        raise RuntimeError("At least one MPS SVD case failed numerical validation.")


if __name__ == "__main__":
    main()
