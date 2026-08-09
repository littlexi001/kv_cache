from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build nested farthest-point K coresets with certified cover radii."
    )
    parser.add_argument("--profile_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_prototypes", type=int, default=16)
    parser.add_argument("--radius_budgets", default="1,2,4,8,16")
    parser.add_argument("--batch_blocks", type=int, default=16)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def gather_tokens(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.gather(
        values,
        dim=2,
        index=indices[:, :, None, None].expand(-1, -1, 1, values.shape[-1]),
    ).squeeze(2)


def main() -> None:
    args = parse_args()
    if args.max_prototypes <= 0 or args.batch_blocks <= 0:
        raise ValueError("max_prototypes and batch_blocks must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    profile_dir = Path(args.profile_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_summary = json.loads(
        (profile_dir / "summary.json").read_text(encoding="utf-8")
    )
    raw = np.load(profile_dir / "raw_k.npy", mmap_mode="r")
    block_ids = np.load(profile_dir / "block_ids.npy", mmap_mode="r")
    block_count, token_count, profile_count, head_dim = map(int, raw.shape)
    if args.max_prototypes > token_count:
        raise ValueError("max_prototypes exceeds token count")
    budgets = sorted(
        {
            int(item.strip())
            for item in args.radius_budgets.split(",")
            if item.strip()
        }
    )
    if not budgets or min(budgets) <= 0 or max(budgets) > args.max_prototypes:
        raise ValueError("radius budgets must lie within the prototype budget")

    prototypes = np.lib.format.open_memmap(
        output_dir / "prototypes.npy",
        mode="w+",
        dtype=np.float16,
        shape=(block_count, args.max_prototypes, profile_count, head_dim),
    )
    prototype_indices = np.lib.format.open_memmap(
        output_dir / "prototype_indices.npy",
        mode="w+",
        dtype=np.uint16,
        shape=(block_count, profile_count, args.max_prototypes),
    )
    cover_radii = np.lib.format.open_memmap(
        output_dir / "cover_radii.npy",
        mode="w+",
        dtype=np.float16,
        shape=(block_count, profile_count, len(budgets)),
    )

    started = time.perf_counter()
    for start in range(0, block_count, args.batch_blocks):
        end = min(block_count, start + args.batch_blocks)
        values = torch.from_numpy(
            np.asarray(raw[start:end], dtype=np.float32)
        ).to(device).permute(0, 2, 1, 3).contiguous()
        mean = values.mean(dim=2, keepdim=True)
        distance_to_mean = (values - mean).square().sum(dim=-1)
        selected = distance_to_mean.argmax(dim=2)
        selected_indices = []
        selected_values = []
        batch_radii = []
        minimum_distance = None
        for prototype_index in range(1, args.max_prototypes + 1):
            prototype = gather_tokens(values, selected)
            selected_indices.append(selected)
            selected_values.append(prototype)
            distance = (values - prototype[:, :, None, :]).square().sum(dim=-1)
            minimum_distance = (
                distance
                if minimum_distance is None
                else torch.minimum(minimum_distance, distance)
            )
            if prototype_index in budgets:
                batch_radii.append(minimum_distance.amax(dim=2).sqrt())
            selected = minimum_distance.argmax(dim=2)

        stacked_values = torch.stack(selected_values, dim=2).permute(0, 2, 1, 3)
        stacked_indices = torch.stack(selected_indices, dim=2)
        stacked_radii = torch.stack(batch_radii, dim=2)
        prototypes[start:end] = stacked_values.to(torch.float16).cpu().numpy()
        prototype_indices[start:end] = stacked_indices.to(torch.uint16).cpu().numpy()
        cover_radii[start:end] = stacked_radii.to(torch.float16).cpu().numpy()
        if (start // args.batch_blocks + 1) % 50 == 0 or end == block_count:
            print(json.dumps({"blocks": end, "total": block_count}), flush=True)

    for array in (prototypes, prototype_indices, cover_radii):
        array.flush()
    elapsed = time.perf_counter() - started
    payload = {
        "source": "nested farthest-point coreset of real pre-RoPE block K",
        "contains_synthetic_vectors": False,
        "selection_uses_gold": False,
        "profile_dir": str(profile_dir),
        "blocks": block_count,
        "tokens_per_block": token_count,
        "profiles": profile_count,
        "head_dim": head_dim,
        "max_prototypes": args.max_prototypes,
        "radius_budgets": budgets,
        "compression_vs_raw_k": token_count / args.max_prototypes,
        "prototype_bytes": int(prototypes.nbytes),
        "raw_k_bytes": int(raw.nbytes),
        "elapsed_seconds": elapsed,
        "pair_specs": profile_summary["pair_specs"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
