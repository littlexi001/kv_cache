from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from profile_real_qk import read_jsonl
from profile_sparse_candidate_k import candidate_block_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a train-only low-rank K subspace and project a sparse raw-K index."
    )
    parser.add_argument("--profile_dir", required=True)
    parser.add_argument("--candidate_rows_path", required=True)
    parser.add_argument("--candidate_field", default="lexical_candidates")
    parser.add_argument("--candidate_limit", type=int, default=16)
    parser.add_argument("--basis_splits", default="train")
    parser.add_argument(
        "--reuse_basis_path",
        default="",
        help="Reuse a previously fitted train-only basis instead of fitting a new one.",
    )
    parser.add_argument("--calibration_blocks", type=int, default=512)
    parser.add_argument("--svd_rank", type=int, default=32)
    parser.add_argument("--batch_blocks", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def fit_second_moment_basis(
    raw: np.ndarray,
    calibration_offsets: np.ndarray,
    *,
    rank: int,
    batch_blocks: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[float]]:
    profile_count = int(raw.shape[2])
    head_dim = int(raw.shape[3])
    bases = []
    retained = []
    for profile in range(profile_count):
        second_moment = torch.zeros(
            (head_dim, head_dim), dtype=torch.float32, device=device
        )
        vector_count = 0
        for start in range(0, len(calibration_offsets), batch_blocks):
            offsets = calibration_offsets[start : start + batch_blocks]
            values = torch.from_numpy(
                np.asarray(raw[offsets, :, profile, :], dtype=np.float32)
            ).to(device)
            matrix = values.reshape(-1, head_dim)
            second_moment.addmm_(matrix.transpose(0, 1), matrix)
            vector_count += int(matrix.shape[0])
        eigenvalues, eigenvectors = torch.linalg.eigh(second_moment)
        order = torch.argsort(eigenvalues, descending=True)
        selected = order[:rank]
        basis = eigenvectors[:, selected].to(torch.float32).contiguous()
        positive = eigenvalues.clamp_min(0)
        retained.append(
            float(
                (
                    positive.index_select(0, selected).sum()
                    / positive.sum().clamp_min(1.0e-30)
                ).item()
            )
        )
        bases.append(basis)
        del second_moment, eigenvalues, eigenvectors
    return torch.stack(bases), retained


def main() -> None:
    args = parse_args()
    if min(args.candidate_limit, args.calibration_blocks, args.svd_rank, args.batch_blocks) <= 0:
        raise ValueError("candidate/calibration/rank/batch values must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    profile_dir = Path(args.profile_dir)
    profile_summary = json.loads(
        (profile_dir / "summary.json").read_text(encoding="utf-8")
    )
    raw = np.load(profile_dir / "raw_k.npy", mmap_mode="r")
    block_ids = np.load(profile_dir / "block_ids.npy", mmap_mode="r")
    if int(raw.shape[0]) != len(block_ids):
        raise ValueError("raw K rows and block IDs do not align")

    rank = min(args.svd_rank, int(raw.shape[-1]))
    allowed_splits = {item.strip() for item in args.basis_splits.split(",") if item.strip()}
    if args.reuse_basis_path:
        source_basis = torch.load(
            args.reuse_basis_path, map_location="cpu", weights_only=False
        )
        basis = source_basis["basis"][..., :rank].to(torch.float32).contiguous()
        retained_energy = [float(item) for item in source_basis["retained_energy"]]
        calibration_offsets = np.asarray([], dtype=np.int64)
        fit_seconds = 0.0
    else:
        rows = [
            row
            for row in read_jsonl(Path(args.candidate_rows_path))
            if str(row["split"]) in allowed_splits
        ]
        calibration_ids = candidate_block_ids(
            rows, args.candidate_field, args.candidate_limit
        )
        available_offsets = {
            int(block_id): offset for offset, block_id in enumerate(block_ids)
        }
        calibration_offsets = np.asarray(
            [
                available_offsets[item]
                for item in calibration_ids
                if item in available_offsets
            ],
            dtype=np.int64,
        )
        if not len(calibration_offsets):
            raise ValueError("no calibration blocks are available in the sparse profile")
        rng = np.random.default_rng(args.seed)
        rng.shuffle(calibration_offsets)
        calibration_offsets = np.sort(
            calibration_offsets[: min(args.calibration_blocks, len(calibration_offsets))]
        )
        fit_started = time.perf_counter()
        basis, retained_energy = fit_second_moment_basis(
            raw,
            calibration_offsets,
            rank=rank,
            batch_blocks=args.batch_blocks,
            device=device,
        )
        fit_seconds = time.perf_counter() - fit_started
    torch.save(
        {
            "basis": basis.cpu(),
            "retained_energy": retained_energy,
            "basis_splits": sorted(allowed_splits),
            "calibration_offsets": torch.from_numpy(calibration_offsets),
            "reused_from": args.reuse_basis_path or None,
        },
        profile_dir / "basis.pt",
    )

    projected_path = profile_dir / f"svd{rank}_k.npy"
    projected = np.lib.format.open_memmap(
        projected_path,
        mode="w+",
        dtype=np.float16,
        shape=(*raw.shape[:-1], rank),
    )
    basis_device = basis.to(device=device, dtype=torch.float32)
    project_started = time.perf_counter()
    for start in range(0, int(raw.shape[0]), args.batch_blocks):
        end = min(int(raw.shape[0]), start + args.batch_blocks)
        values = torch.from_numpy(
            np.asarray(raw[start:end], dtype=np.float32)
        ).to(device)
        reduced = torch.einsum("btpd,pdr->btpr", values, basis_device)
        projected[start:end] = reduced.to(torch.float16).cpu().numpy()
    projected.flush()
    project_seconds = time.perf_counter() - project_started
    payload: dict[str, Any] = {
        "source": "train-candidate second-moment K subspace for sparse reranking",
        "contains_synthetic_vectors": False,
        "selection_uses_gold": False,
        "profile_dir": str(profile_dir),
        "candidate_rows_path": args.candidate_rows_path,
        "candidate_field": args.candidate_field,
        "candidate_limit": args.candidate_limit,
        "basis_splits": sorted(allowed_splits),
        "basis_source": args.reuse_basis_path or "fitted_from_candidate_rows",
        "available_blocks": int(raw.shape[0]),
        "calibration_blocks": int(len(calibration_offsets)),
        "calibration_tokens": int(len(calibration_offsets) * raw.shape[1]),
        "profiles": int(raw.shape[2]),
        "head_dim": int(raw.shape[3]),
        "svd_rank": rank,
        "retained_energy": retained_energy,
        "mean_retained_energy": float(np.mean(retained_energy)),
        "basis_fit_seconds": fit_seconds,
        "projection_seconds": project_seconds,
        "projected_k_path": str(projected_path),
        "raw_profile_summary": profile_summary,
    }
    (profile_dir / f"svd{rank}_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
