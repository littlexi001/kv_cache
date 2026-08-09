from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from profile_sampled_block_k_subspaces import RANKS, rank_for, summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure block K singular spectra from full-record raw profile shards."
    )
    parser.add_argument("--profile_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--sample_blocks", type=int, default=512)
    parser.add_argument("--batch_blocks", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    profile_dir = Path(args.profile_dir)
    source = json.loads((profile_dir / "summary.json").read_text(encoding="utf-8"))
    block_count = int(source["num_blocks"])
    token_count = int(source["block_tokens"])
    profile_count = len(source["pair_specs"])
    head_dim = int(source["head_dim"])
    rng = np.random.default_rng(args.seed)
    sample_count = min(args.sample_blocks, block_count)
    block_ids = np.sort(rng.choice(block_count, size=sample_count, replace=False))
    sampled = np.empty(
        (sample_count, token_count, profile_count, head_dim), dtype=np.float16
    )
    for shard in source["shards"]:
        shard_start = int(shard["block_start"])
        shard_end = int(shard["block_end"])
        mask = (block_ids >= shard_start) & (block_ids < shard_end)
        if not mask.any():
            continue
        path = Path(str(shard["raw_k_path"]))
        if not path.exists():
            path = profile_dir / path.name
        raw = np.load(path, mmap_mode="r")
        sampled[mask] = raw[block_ids[mask] - shard_start]

    metrics: dict[str, list[list[float]]] = {}
    for prefix in ("uncentered", "residual"):
        for suffix in ("rank90", "rank95", "effective_rank"):
            metrics[f"{prefix}_{suffix}"] = [[] for _ in range(profile_count)]
        for rank in RANKS:
            metrics[f"{prefix}_energy_rank{rank}"] = [
                [] for _ in range(profile_count)
            ]

    for start in range(0, sample_count, args.batch_blocks):
        values = torch.from_numpy(
            sampled[start : start + args.batch_blocks].astype(np.float32)
        ).to(device)
        for prefix, matrix in (
            ("uncentered", values),
            ("residual", values - values.mean(dim=1, keepdim=True)),
        ):
            batch_size = len(matrix)
            arranged = matrix.permute(0, 2, 1, 3).reshape(
                -1, token_count, head_dim
            )
            singular = torch.linalg.svdvals(arranged)
            energy = singular.square()
            probability = energy / energy.sum(dim=-1, keepdim=True).clamp_min(1.0e-30)
            cumulative = probability.cumsum(dim=-1).reshape(
                batch_size, profile_count, -1
            )
            probability = probability.reshape(batch_size, profile_count, -1)
            effective = torch.exp(
                -(probability * torch.log(probability.clamp_min(1.0e-30))).sum(dim=-1)
            )
            for profile in range(profile_count):
                metrics[f"{prefix}_rank90"][profile].extend(
                    rank_for(cumulative[:, profile], 0.90).cpu().tolist()
                )
                metrics[f"{prefix}_rank95"][profile].extend(
                    rank_for(cumulative[:, profile], 0.95).cpu().tolist()
                )
                metrics[f"{prefix}_effective_rank"][profile].extend(
                    effective[:, profile].cpu().tolist()
                )
                for rank in RANKS:
                    metrics[f"{prefix}_energy_rank{rank}"][profile].extend(
                        cumulative[:, profile, min(rank, cumulative.shape[-1]) - 1]
                        .cpu()
                        .tolist()
                    )

    profiles = []
    for profile, spec in enumerate(source["pair_specs"]):
        item: dict[str, Any] = {"profile": profile, **spec}
        for name, per_profile in metrics.items():
            item[name] = summarize(per_profile[profile])
        profiles.append(item)
    payload = {
        "source": "sampled full-record causal prefill K singular spectra",
        "contains_synthetic_vectors": False,
        "profile_dir": str(profile_dir),
        "context_mode": "full_record_causal_prefill",
        "corpus_blocks": block_count,
        "block_tokens": token_count,
        "sample_blocks": sample_count,
        "seed": args.seed,
        "profiles": profiles,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
