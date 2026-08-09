from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline


CPP_SOURCE = r"""
#include <torch/extension.h>

void hash_rebuild_forward(
    torch::Tensor resident_keys,
    torch::Tensor table_keys,
    torch::Tensor table_slots);

torch::Tensor hash_lookup_forward(
    torch::Tensor candidate_keys,
    torch::Tensor table_keys,
    torch::Tensor table_slots);

torch::Tensor lru_histogram_update_forward(
    torch::Tensor resident_keys,
    torch::Tensor resident_ages,
    torch::Tensor hit_slots,
    torch::Tensor miss_keys);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hash_rebuild_forward", &hash_rebuild_forward, "Rebuild resident KV hash directory");
  m.def("hash_lookup_forward", &hash_lookup_forward, "Lookup KV cache slots");
  m.def("lru_histogram_update_forward", &lru_histogram_update_forward, "Fused recency update and LRU eviction");
}
"""


CUDA_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ uint32_t token_hash(uint32_t key) {
  key ^= key >> 16;
  key *= 0x7feb352dU;
  key ^= key >> 15;
  key *= 0x846ca68bU;
  key ^= key >> 16;
  return key;
}

__global__ void hash_rebuild_kernel(
    const int32_t* __restrict__ resident_keys,
    int32_t* __restrict__ table_keys,
    int32_t* __restrict__ table_slots,
    int cache_count,
    int table_size,
    int total_entries) {
  int mask = table_size - 1;
  for (int flat = blockIdx.x * blockDim.x + threadIdx.x;
       flat < total_entries;
       flat += blockDim.x * gridDim.x) {
    int head = flat / cache_count;
    int slot = flat - head * cache_count;
    int32_t key = resident_keys[flat];
    int table_base = head * table_size;
    int position = static_cast<int>(token_hash(static_cast<uint32_t>(key))) & mask;
    for (int probe = 0; probe < table_size; ++probe) {
      int offset = table_base + position;
      int32_t previous = atomicCAS(table_keys + offset, -1, key);
      if (previous == -1 || previous == key) {
        table_slots[offset] = slot;
        break;
      }
      position = (position + 1) & mask;
    }
  }
}

__global__ void hash_lookup_kernel(
    const int32_t* __restrict__ candidate_keys,
    const int32_t* __restrict__ table_keys,
    const int32_t* __restrict__ table_slots,
    int32_t* __restrict__ output_slots,
    int selected_count,
    int table_size,
    int total_candidates) {
  int mask = table_size - 1;
  for (int flat = blockIdx.x * blockDim.x + threadIdx.x;
       flat < total_candidates;
       flat += blockDim.x * gridDim.x) {
    int head = flat / selected_count;
    int32_t key = candidate_keys[flat];
    int table_base = head * table_size;
    int position = static_cast<int>(token_hash(static_cast<uint32_t>(key))) & mask;
    int32_t slot = -1;
    for (int probe = 0; probe < table_size; ++probe) {
      int offset = table_base + position;
      int32_t table_key = table_keys[offset];
      if (table_key == key) {
        slot = table_slots[offset];
        break;
      }
      if (table_key == -1) {
        break;
      }
      position = (position + 1) & mask;
    }
    output_slots[flat] = slot;
  }
}

__global__ void lru_histogram_update_kernel(
    int32_t* __restrict__ resident_keys,
    uint8_t* __restrict__ resident_ages,
    const int32_t* __restrict__ hit_slots,
    const int32_t* __restrict__ miss_keys,
    int32_t* __restrict__ eviction_slots,
    int cache_count,
    int hit_count,
    int miss_count) {
  __shared__ int histogram[256];
  __shared__ int threshold;
  __shared__ int threshold_quota;
  __shared__ int threshold_counter;
  __shared__ int eviction_counter;
  int head = blockIdx.x;
  int tid = threadIdx.x;
  int cache_base = head * cache_count;
  int hit_base = head * hit_count;
  int miss_base = head * miss_count;

  for (int slot = tid; slot < cache_count; slot += blockDim.x) {
    uint8_t age = resident_ages[cache_base + slot];
    resident_ages[cache_base + slot] = age > 0 ? age - 1 : 0;
  }
  __syncthreads();
  for (int hit = tid; hit < hit_count; hit += blockDim.x) {
    int32_t slot = hit_slots[hit_base + hit];
    if (slot >= 0 && slot < cache_count) {
      resident_ages[cache_base + slot] = 255;
    }
  }
  for (int age = tid; age < 256; age += blockDim.x) {
    histogram[age] = 0;
  }
  __syncthreads();

  for (int slot = tid; slot < cache_count; slot += blockDim.x) {
    atomicAdd(histogram + resident_ages[cache_base + slot], 1);
  }
  __syncthreads();
  if (tid == 0) {
    int cumulative = 0;
    threshold = 255;
    threshold_quota = miss_count;
    for (int age = 0; age < 256; ++age) {
      if (cumulative + histogram[age] >= miss_count) {
        threshold = age;
        threshold_quota = miss_count - cumulative;
        break;
      }
      cumulative += histogram[age];
    }
    threshold_counter = 0;
    eviction_counter = 0;
  }
  __syncthreads();

  for (int slot = tid; slot < cache_count; slot += blockDim.x) {
    uint8_t age = resident_ages[cache_base + slot];
    bool evict = age < threshold;
    if (age == threshold) {
      int rank = atomicAdd(&threshold_counter, 1);
      evict = rank < threshold_quota;
    }
    if (evict) {
      int rank = atomicAdd(&eviction_counter, 1);
      if (rank < miss_count) {
        resident_keys[cache_base + slot] = miss_keys[miss_base + rank];
        resident_ages[cache_base + slot] = 255;
        eviction_slots[miss_base + rank] = slot;
      }
    }
  }
}

void hash_rebuild_forward(
    torch::Tensor resident_keys,
    torch::Tensor table_keys,
    torch::Tensor table_slots) {
  TORCH_CHECK(resident_keys.is_cuda() && table_keys.is_cuda() && table_slots.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(resident_keys.scalar_type() == at::kInt, "resident keys must be int32");
  TORCH_CHECK(table_keys.scalar_type() == at::kInt && table_slots.scalar_type() == at::kInt, "table tensors must be int32");
  TORCH_CHECK(resident_keys.dim() == 2 && table_keys.dim() == 2 && table_slots.sizes() == table_keys.sizes(), "invalid shapes");
  TORCH_CHECK(resident_keys.size(0) == table_keys.size(0), "head count mismatch");
  int table_size = static_cast<int>(table_keys.size(1));
  TORCH_CHECK(table_size > 0 && (table_size & (table_size - 1)) == 0, "table size must be a power of two");
  int total_entries = static_cast<int>(resident_keys.numel());
  int threads = 256;
  int blocks = std::min(1024, (total_entries + threads - 1) / threads);
  hash_rebuild_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      resident_keys.data_ptr<int32_t>(),
      table_keys.data_ptr<int32_t>(),
      table_slots.data_ptr<int32_t>(),
      static_cast<int>(resident_keys.size(1)),
      table_size,
      total_entries);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor hash_lookup_forward(
    torch::Tensor candidate_keys,
    torch::Tensor table_keys,
    torch::Tensor table_slots) {
  TORCH_CHECK(candidate_keys.is_cuda() && table_keys.is_cuda() && table_slots.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(candidate_keys.scalar_type() == at::kInt, "candidate keys must be int32");
  TORCH_CHECK(table_keys.scalar_type() == at::kInt && table_slots.scalar_type() == at::kInt, "table tensors must be int32");
  TORCH_CHECK(candidate_keys.dim() == 2 && table_keys.dim() == 2 && table_slots.sizes() == table_keys.sizes(), "invalid shapes");
  TORCH_CHECK(candidate_keys.size(0) == table_keys.size(0), "head count mismatch");
  auto output = torch::empty_like(candidate_keys);
  int total_candidates = static_cast<int>(candidate_keys.numel());
  int threads = 256;
  int blocks = std::min(1024, (total_candidates + threads - 1) / threads);
  hash_lookup_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      candidate_keys.data_ptr<int32_t>(),
      table_keys.data_ptr<int32_t>(),
      table_slots.data_ptr<int32_t>(),
      output.data_ptr<int32_t>(),
      static_cast<int>(candidate_keys.size(1)),
      static_cast<int>(table_keys.size(1)),
      total_candidates);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor lru_histogram_update_forward(
    torch::Tensor resident_keys,
    torch::Tensor resident_ages,
    torch::Tensor hit_slots,
    torch::Tensor miss_keys) {
  TORCH_CHECK(resident_keys.is_cuda() && resident_ages.is_cuda(), "resident tensors must be CUDA");
  TORCH_CHECK(hit_slots.is_cuda() && miss_keys.is_cuda(), "update tensors must be CUDA");
  TORCH_CHECK(resident_keys.scalar_type() == at::kInt && hit_slots.scalar_type() == at::kInt && miss_keys.scalar_type() == at::kInt, "keys and slots must be int32");
  TORCH_CHECK(resident_ages.scalar_type() == at::kByte, "resident ages must be uint8");
  TORCH_CHECK(resident_keys.dim() == 2 && resident_ages.sizes() == resident_keys.sizes(), "invalid resident shapes");
  TORCH_CHECK(hit_slots.dim() == 2 && miss_keys.dim() == 2 && hit_slots.size(0) == resident_keys.size(0) && miss_keys.size(0) == resident_keys.size(0), "invalid update shapes");
  auto output = torch::empty_like(miss_keys);
  lru_histogram_update_kernel<<<
      resident_keys.size(0),
      256,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
          resident_keys.data_ptr<int32_t>(),
          resident_ages.data_ptr<uint8_t>(),
          hit_slots.data_ptr<int32_t>(),
          miss_keys.data_ptr<int32_t>(),
          output.data_ptr<int32_t>(),
          static_cast<int>(resident_keys.size(1)),
          static_cast<int>(hit_slots.size(1)),
          static_cast<int>(miss_keys.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
"""


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
    parser.add_argument("--cache_fraction", type=float, default=0.032)
    parser.add_argument("--selected_fraction", type=float, default=0.02)
    parser.add_argument("--hit_rate", type=float, default=0.79)
    parser.add_argument("--kv_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--projection_dim", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cache_count = max(1, math.floor(args.history_count * args.cache_fraction))
    selected_count = max(1, math.ceil(args.history_count * args.selected_fraction))
    hit_count = min(selected_count, round(selected_count * args.hit_rate))
    miss_count = selected_count - hit_count
    table_size = 1 << math.ceil(math.log2(cache_count * 1.6))

    resident_rows = []
    candidate_rows = []
    generator = torch.Generator().manual_seed(20260715)
    for _ in range(args.kv_heads):
        permutation = torch.randperm(args.history_count, generator=generator)
        resident = permutation[:cache_count].to(torch.int32)
        hit_choice = torch.randperm(cache_count, generator=generator)[:hit_count]
        hits = resident[hit_choice]
        misses = permutation[cache_count : cache_count + miss_count].to(torch.int32)
        candidates = torch.cat((hits, misses))
        candidates = candidates[torch.randperm(selected_count, generator=generator)]
        resident_rows.append(resident)
        candidate_rows.append(candidates)

    resident_keys = torch.stack(resident_rows).cuda()
    candidates = torch.stack(candidate_rows).cuda()
    resident_ages = torch.randint(
        0, 500, (args.kv_heads, cache_count), dtype=torch.int32, device="cuda"
    )
    resident_ages_u8 = torch.randint(
        0, 256, (args.kv_heads, cache_count), dtype=torch.uint8, device="cuda"
    )
    table_keys = torch.full(
        (args.kv_heads, table_size), -1, dtype=torch.int32, device="cuda"
    )
    table_slots = torch.empty_like(table_keys)
    module = load_inline(
        name="hash_lru_directory_ext_v3",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=None,
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )

    def rebuild() -> None:
        table_keys.fill_(-1)
        module.hash_rebuild_forward(resident_keys, table_keys, table_slots)

    rebuild()

    def lookup() -> torch.Tensor:
        return module.hash_lookup_forward(candidates, table_keys, table_slots)

    slots = lookup()
    hit = slots.ge(0)
    measured_hit_rate = float(hit.float().mean().item())
    hit_slots = slots[hit].reshape(args.kv_heads, hit_count).to(torch.long)
    miss_ids = candidates[~hit].reshape(args.kv_heads, miss_count)
    hit_slots_i32 = slots[hit].reshape(args.kv_heads, hit_count)

    def select_evictions() -> torch.Tensor:
        eligible_ages = resident_ages.clone()
        eligible_ages.scatter_(1, hit_slots, 65536)
        return torch.topk(
            eligible_ages,
            k=miss_count,
            dim=1,
            largest=False,
            sorted=False,
        ).indices

    eviction_slots = select_evictions()

    def apply_updates() -> None:
        resident_ages.scatter_(1, hit_slots, 65535)
        resident_keys.scatter_(1, eviction_slots, miss_ids)
        resident_ages.scatter_(1, eviction_slots, 65535)

    def fused_histogram_update() -> torch.Tensor:
        return module.lru_histogram_update_forward(
            resident_keys,
            resident_ages_u8,
            hit_slots_i32,
            miss_ids,
        )

    lookup_ms = timed_ms(lookup, args.warmup, args.repeats)
    eviction_ms = timed_ms(select_evictions, args.warmup, args.repeats)
    update_ms = timed_ms(apply_updates, args.warmup, args.repeats)
    rebuild_ms = timed_ms(rebuild, args.warmup, args.repeats)
    fused_update_ms = timed_ms(
        fused_histogram_update, args.warmup, args.repeats
    )

    full_kv_bytes = 2 * args.kv_heads * args.history_count * args.head_dim * 2
    pca_bytes = (
        args.kv_heads * args.history_count * (args.projection_dim + 2)
        + args.kv_heads * args.head_dim * args.projection_dim * 2
    )
    exact_cache_bytes = 2 * args.kv_heads * cache_count * args.head_dim * 2
    # Persistent hash keys + uint16 slots, inverse token ids + uint8 ages.
    directory_bytes = (
        args.kv_heads * table_size * (4 + 2)
        + args.kv_heads * cache_count * (4 + 1)
    )
    result = {
        "history_count": args.history_count,
        "cache_count_per_kv_head": cache_count,
        "cache_fraction_actual": cache_count / args.history_count,
        "selected_count_per_kv_head": selected_count,
        "table_size_per_kv_head": table_size,
        "hash_load_factor": cache_count / table_size,
        "target_hit_rate": args.hit_rate,
        "measured_hit_rate": measured_hit_rate,
        "miss_count_per_kv_head": miss_count,
        "hash_lookup_ms_per_layer": lookup_ms,
        "lru_eviction_selection_ms_per_layer": eviction_ms,
        "slot_update_ms_per_layer": update_ms,
        "hash_rebuild_ms_per_layer": rebuild_ms,
        "directory_total_ms_per_layer": lookup_ms + eviction_ms + update_ms + rebuild_ms,
        "fused_histogram_update_ms_per_layer": fused_update_ms,
        "fused_directory_total_ms_per_layer": lookup_ms + fused_update_ms + rebuild_ms,
        "pca_index_fraction": pca_bytes / full_kv_bytes,
        "exact_cache_fraction": exact_cache_bytes / full_kv_bytes,
        "directory_fraction": directory_bytes / full_kv_bytes,
        "total_resident_fraction": (pca_bytes + exact_cache_bytes + directory_bytes)
        / full_kv_bytes,
        "directory_bytes_per_layer": directory_bytes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
