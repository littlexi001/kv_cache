from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def timed_ms(function, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        function()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_count", type=int, default=131_072)
    parser.add_argument("--cache_fraction", type=float, default=0.033)
    parser.add_argument("--selected_fraction", type=float, default=0.02)
    parser.add_argument("--hit_rate", type=float, default=0.793276366422321)
    parser.add_argument("--kv_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--projection_dim", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not 0.0 < args.selected_fraction < args.cache_fraction < 1.0:
        raise ValueError("expected 0 < selected_fraction < cache_fraction < 1")
    cache_count = max(1, math.floor(args.history_count * args.cache_fraction))
    selected_count = max(1, math.ceil(args.history_count * args.selected_fraction))
    hit_count = min(selected_count, round(selected_count * args.hit_rate))
    miss_count = selected_count - hit_count

    resident_id_rows = []
    resident_slot_rows = []
    candidate_rows = []
    generator = torch.Generator().manual_seed(20260715)
    for _ in range(args.kv_heads):
        permutation = torch.randperm(args.history_count, generator=generator)
        resident_unsorted = permutation[:cache_count]
        resident_ids, order = torch.sort(resident_unsorted.to(torch.int32))
        resident_slots = torch.arange(cache_count, dtype=torch.int32)[order]
        hit_choice = torch.randperm(cache_count, generator=generator)[:hit_count]
        hits = resident_unsorted[hit_choice].to(torch.int32)
        misses = permutation[cache_count : cache_count + miss_count].to(torch.int32)
        candidates = torch.cat((hits, misses))
        candidates = candidates[
            torch.randperm(selected_count, generator=generator)
        ]
        resident_id_rows.append(resident_ids)
        resident_slot_rows.append(resident_slots)
        candidate_rows.append(candidates)

    resident_ids = torch.stack(resident_id_rows).cuda()
    resident_slots = torch.stack(resident_slot_rows).cuda()
    candidates = torch.stack(candidate_rows).cuda()
    resident_ages = torch.randint(
        0,
        200,
        (args.kv_heads, cache_count),
        dtype=torch.int32,
        device="cuda",
    )

    def lookup() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = torch.searchsorted(resident_ids, candidates)
        safe_positions = positions.clamp_max(cache_count - 1)
        hit = torch.gather(resident_ids, 1, safe_positions).eq(candidates)
        slots = torch.where(
            hit,
            torch.gather(resident_slots, 1, safe_positions),
            torch.full_like(safe_positions, -1),
        )
        return safe_positions, hit, slots

    positions, hit, slots = lookup()
    measured_hit_rate = float(hit.float().mean().item())
    hit_positions = positions[hit].reshape(args.kv_heads, hit_count)
    miss_ids = candidates[~hit].reshape(args.kv_heads, miss_count)
    resident_hit_mask = torch.zeros_like(resident_ids, dtype=torch.bool)
    resident_hit_mask.scatter_(1, hit_positions, True)

    def choose_evictions() -> torch.Tensor:
        eligible_ages = resident_ages.masked_fill(resident_hit_mask, 256)
        return torch.topk(
            eligible_ages,
            k=miss_count,
            dim=1,
            largest=False,
            sorted=False,
        ).indices

    eviction_positions = choose_evictions()

    def update_directory() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current_ids = resident_ids.clone()
        current_ages = resident_ages.clone()
        current_ids.scatter_(1, eviction_positions, miss_ids)
        current_ages.scatter_(1, eviction_positions, 255)
        current_ages.scatter_(1, hit_positions, 255)
        sorted_ids, order = torch.sort(current_ids, dim=1)
        sorted_slots = torch.gather(resident_slots, 1, order)
        sorted_ages = torch.gather(current_ages, 1, order)
        return sorted_ids, sorted_slots, sorted_ages

    lookup_ms = timed_ms(lookup, args.warmup, args.repeats)
    eviction_ms = timed_ms(choose_evictions, args.warmup, args.repeats)
    update_ms = timed_ms(update_directory, args.warmup, args.repeats)

    full_kv_bytes = (
        2 * args.kv_heads * args.history_count * args.head_dim * 2
    )
    pca_code_bytes = args.kv_heads * args.history_count * args.projection_dim
    pca_scale_bytes = args.kv_heads * args.history_count * 2
    pca_basis_bytes = args.kv_heads * args.head_dim * args.projection_dim * 2
    exact_cache_bytes = 2 * args.kv_heads * cache_count * args.head_dim * 2
    # Sorted token id, physical slot, and an 8-bit recency epoch per entry.
    directory_bytes = args.kv_heads * cache_count * (4 + 2 + 1)
    resident_bytes = (
        pca_code_bytes
        + pca_scale_bytes
        + pca_basis_bytes
        + exact_cache_bytes
        + directory_bytes
    )
    result = {
        "history_count": args.history_count,
        "cache_fraction_requested": args.cache_fraction,
        "cache_count_per_kv_head": cache_count,
        "cache_fraction_actual": cache_count / args.history_count,
        "selected_count_per_kv_head": selected_count,
        "target_hit_rate": args.hit_rate,
        "measured_hit_rate": measured_hit_rate,
        "miss_count_per_kv_head": miss_count,
        "sorted_lookup_ms_per_layer": lookup_ms,
        "eviction_selection_ms_per_layer": eviction_ms,
        "directory_update_ms_per_layer": update_ms,
        "pca_code_scale_basis_fraction": (
            pca_code_bytes + pca_scale_bytes + pca_basis_bytes
        )
        / full_kv_bytes,
        "exact_cache_fraction": exact_cache_bytes / full_kv_bytes,
        "directory_fraction": directory_bytes / full_kv_bytes,
        "total_resident_fraction": resident_bytes / full_kv_bytes,
        "directory_bytes_per_layer": directory_bytes,
        "slot_checksum": int(slots.clamp_min(0).sum().item()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
