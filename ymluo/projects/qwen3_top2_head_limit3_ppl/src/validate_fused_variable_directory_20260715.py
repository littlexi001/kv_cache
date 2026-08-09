from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from hierarchical_cache_cuda_20260715 import load_hierarchical_cache_extension


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_count", type=int, default=63744)
    parser.add_argument("--candidate_fraction", type=float, default=0.02)
    parser.add_argument("--cache_fraction", type=float, default=0.032)
    parser.add_argument("--kv_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=16)
    parser.add_argument("--target_hit_rate", type=float, default=0.8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def expected_cache_values(
    host_kv: torch.Tensor, candidates: torch.Tensor, device: torch.device
) -> torch.Tensor:
    rows = []
    candidates_cpu = candidates.cpu().to(torch.long)
    for head in range(candidates.shape[0]):
        rows.append(host_kv[:, head, candidates_cpu[head], :])
    return torch.stack(rows, dim=1).to(device)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    module = load_hierarchical_cache_extension()
    candidate_count = math.ceil(args.history_count * args.candidate_fraction)
    cache_count = math.floor(args.history_count * args.cache_fraction)
    table_size = 1 << math.ceil(math.log2(cache_count * 1.6))
    if candidate_count >= cache_count:
        raise ValueError("candidate count must be smaller than cache count")

    generator = torch.Generator().manual_seed(20260715)
    permutations = [
        torch.randperm(args.history_count, generator=generator)
        for _ in range(args.kv_heads)
    ]
    first = torch.stack(
        [permutation[:candidate_count] for permutation in permutations]
    ).to(device=device, dtype=torch.int32)
    hit_count = round(candidate_count * args.target_hit_rate)
    second_rows = []
    for permutation in permutations:
        hits = permutation[:hit_count]
        misses = permutation[candidate_count : 2 * candidate_count - hit_count]
        row = torch.cat((hits, misses))
        second_rows.append(row[torch.randperm(candidate_count, generator=generator)])
    second = torch.stack(second_rows).to(device=device, dtype=torch.int32)

    resident_keys = torch.full(
        (args.kv_heads, cache_count), -1, dtype=torch.int32, device=device
    )
    resident_ages = torch.zeros_like(resident_keys, dtype=torch.uint8)
    table_keys = torch.full(
        (args.kv_heads, table_size), -1, dtype=torch.int32, device=device
    )
    table_slots = torch.empty_like(table_keys)
    host_kv = torch.randn(
        2,
        args.kv_heads,
        args.history_count,
        args.head_dim,
        dtype=torch.float16,
        pin_memory=True,
        generator=generator,
    )
    device_cache = torch.empty(
        2,
        args.kv_heads,
        cache_count + 1,
        args.head_dim,
        dtype=torch.float16,
        device=device,
    )

    first_lookup = module.hash_lookup_forward(first, table_keys, table_slots)
    first_slots = module.variable_lru_update_forward(
        resident_keys, resident_ages, first, first_lookup
    )
    module.mapped_host_fill_variable_forward(
        host_kv, device_cache, first, first_lookup, first_slots
    )
    table_keys.fill_(-1)
    module.hash_rebuild_forward(resident_keys, table_keys, table_slots)
    torch.cuda.synchronize(device)

    first_resident = torch.gather(resident_keys, 1, first_slots.to(torch.long))
    first_cache = torch.gather(
        device_cache,
        2,
        first_slots.reshape(1, args.kv_heads, candidate_count, 1)
        .expand(2, -1, -1, args.head_dim)
        .to(torch.long),
    )
    first_expected = expected_cache_values(host_kv, first, device)

    second_lookup = module.hash_lookup_forward(second, table_keys, table_slots)
    measured_hit_rate = float(second_lookup.ge(0).float().mean().item())
    second_slots = module.variable_lru_update_forward(
        resident_keys, resident_ages, second, second_lookup
    )
    module.mapped_host_fill_variable_forward(
        host_kv, device_cache, second, second_lookup, second_slots
    )
    table_keys.fill_(-1)
    module.hash_rebuild_forward(resident_keys, table_keys, table_slots)
    torch.cuda.synchronize(device)

    second_resident = torch.gather(resident_keys, 1, second_slots.to(torch.long))
    second_cache = torch.gather(
        device_cache,
        2,
        second_slots.reshape(1, args.kv_heads, candidate_count, 1)
        .expand(2, -1, -1, args.head_dim)
        .to(torch.long),
    )
    second_expected = expected_cache_values(host_kv, second, device)
    result = {
        "device": str(device),
        "history_count": args.history_count,
        "candidate_count": candidate_count,
        "cache_count": cache_count,
        "table_size": table_size,
        "target_hit_rate": args.target_hit_rate,
        "first_lookup_miss_fraction": float(first_lookup.lt(0).float().mean().item()),
        "first_slot_min": int(first_slots.min().item()),
        "first_slot_max": int(first_slots.max().item()),
        "first_resident_key_match": float(first_resident.eq(first).float().mean().item()),
        "first_cache_max_abs_error": float(
            (first_cache.float() - first_expected.float()).abs().max().item()
        ),
        "second_measured_hit_rate": measured_hit_rate,
        "second_slot_min": int(second_slots.min().item()),
        "second_slot_max": int(second_slots.max().item()),
        "second_resident_key_match": float(
            second_resident.eq(second).float().mean().item()
        ),
        "second_cache_max_abs_error": float(
            (second_cache.float() - second_expected.float()).abs().max().item()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
