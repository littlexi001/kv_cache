from __future__ import annotations

from functools import lru_cache

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

torch::Tensor hash_lookup_ragged_forward(
    torch::Tensor candidate_keys,
    torch::Tensor candidate_counts,
    torch::Tensor table_keys,
    torch::Tensor table_slots);

torch::Tensor variable_lru_update_forward(
    torch::Tensor resident_keys,
    torch::Tensor resident_ages,
    torch::Tensor candidate_keys,
    torch::Tensor lookup_slots);

torch::Tensor variable_lru_update_ragged_forward(
    torch::Tensor resident_keys,
    torch::Tensor resident_ages,
    torch::Tensor candidate_keys,
    torch::Tensor candidate_counts,
    torch::Tensor lookup_slots);

void mapped_host_fill_variable_forward(
    torch::Tensor host_kv,
    torch::Tensor device_cache,
    torch::Tensor candidate_keys,
    torch::Tensor lookup_slots,
    torch::Tensor final_slots);

void mapped_host_fill_variable_ragged_forward(
    torch::Tensor host_kv,
    torch::Tensor device_cache,
    torch::Tensor candidate_keys,
    torch::Tensor candidate_counts,
    torch::Tensor lookup_slots,
    torch::Tensor final_slots);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hash_rebuild_forward", &hash_rebuild_forward);
  m.def("hash_lookup_forward", &hash_lookup_forward);
  m.def("hash_lookup_ragged_forward", &hash_lookup_ragged_forward);
  m.def("variable_lru_update_forward", &variable_lru_update_forward);
  m.def(
      "variable_lru_update_ragged_forward",
      &variable_lru_update_ragged_forward);
  m.def("mapped_host_fill_variable_forward", &mapped_host_fill_variable_forward);
  m.def(
      "mapped_host_fill_variable_ragged_forward",
      &mapped_host_fill_variable_ragged_forward);
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
    if (key < 0) {
      continue;
    }
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
    const int64_t* __restrict__ candidate_counts,
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
    int candidate = flat - head * selected_count;
    if (candidate_counts != nullptr
        && candidate >= static_cast<int>(candidate_counts[head])) {
      output_slots[flat] = -1;
      continue;
    }
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

__global__ void variable_lru_update_kernel(
    int32_t* __restrict__ resident_keys,
    uint8_t* __restrict__ resident_ages,
    const int32_t* __restrict__ candidate_keys,
    const int64_t* __restrict__ candidate_counts,
    const int32_t* __restrict__ lookup_slots,
    int32_t* __restrict__ final_slots,
    int cache_count,
    int selected_count) {
  extern __shared__ int shared[];
  int* histogram = shared;
  int* miss_positions = shared + 256;
  __shared__ int miss_count;
  __shared__ int threshold;
  __shared__ int threshold_quota;
  __shared__ int threshold_counter;
  __shared__ int eviction_counter;

  int head = blockIdx.x;
  int tid = threadIdx.x;
  int cache_base = head * cache_count;
  int candidate_base = head * selected_count;
  int valid_count = candidate_counts == nullptr
      ? selected_count
      : min(selected_count, static_cast<int>(candidate_counts[head]));

  if (tid == 0) {
    miss_count = 0;
  }
  for (int age = tid; age < 256; age += blockDim.x) {
    histogram[age] = 0;
  }
  for (int slot = tid; slot < cache_count; slot += blockDim.x) {
    uint8_t age = resident_ages[cache_base + slot];
    resident_ages[cache_base + slot] = age > 0 ? age - 1 : 0;
  }
  __syncthreads();

  for (int candidate = tid; candidate < valid_count; candidate += blockDim.x) {
    int32_t slot = lookup_slots[candidate_base + candidate];
    final_slots[candidate_base + candidate] = slot;
    if (slot >= 0) {
      resident_ages[cache_base + slot] = 255;
    } else {
      int rank = atomicAdd(&miss_count, 1);
      miss_positions[rank] = candidate;
    }
  }
  for (int candidate = valid_count + tid;
       candidate < selected_count;
       candidate += blockDim.x) {
    final_slots[candidate_base + candidate] = -1;
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
        int candidate = miss_positions[rank];
        resident_keys[cache_base + slot] =
            candidate_keys[candidate_base + candidate];
        resident_ages[cache_base + slot] = 255;
        final_slots[candidate_base + candidate] = slot;
      }
    }
  }
}

template <typename scalar_t>
__global__ void mapped_host_fill_variable_kernel(
    const scalar_t* __restrict__ host_kv,
    scalar_t* __restrict__ device_cache,
    const int32_t* __restrict__ candidate_keys,
    const int64_t* __restrict__ candidate_counts,
    const int32_t* __restrict__ lookup_slots,
    const int32_t* __restrict__ final_slots,
    int kv_head_count,
    int history_capacity,
    int cache_count,
    int selected_count,
    int head_dim,
    int total_elements) {
  for (int flat = blockIdx.x * blockDim.x + threadIdx.x;
       flat < total_elements;
       flat += blockDim.x * gridDim.x) {
    int dim = flat % head_dim;
    int candidate = (flat / head_dim) % selected_count;
    int kv_head = (flat / (head_dim * selected_count)) % kv_head_count;
    int kv_kind = flat / (head_dim * selected_count * kv_head_count);
    if (candidate_counts != nullptr
        && candidate >= static_cast<int>(candidate_counts[kv_head])) {
      continue;
    }
    int directory_offset = kv_head * selected_count + candidate;
    if (lookup_slots[directory_offset] >= 0) {
      continue;
    }
    int32_t token = candidate_keys[directory_offset];
    int32_t destination = final_slots[directory_offset];
    int64_t source = (
        (static_cast<int64_t>(kv_kind) * kv_head_count + kv_head)
        * history_capacity + token) * head_dim + dim;
    int64_t target = (
        (static_cast<int64_t>(kv_kind) * kv_head_count + kv_head)
        * cache_count + destination) * head_dim + dim;
    device_cache[target] = host_kv[source];
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
  c10::cuda::CUDAGuard device_guard(resident_keys.device());
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
  TORCH_CHECK(candidate_keys.dim() == 2 && table_keys.dim() == 2 && table_slots.sizes() == table_keys.sizes(), "invalid shapes");
  TORCH_CHECK(candidate_keys.size(0) == table_keys.size(0), "head count mismatch");
  c10::cuda::CUDAGuard device_guard(candidate_keys.device());
  auto output = torch::empty_like(candidate_keys);
  int total_candidates = static_cast<int>(candidate_keys.numel());
  int threads = 256;
  int blocks = std::min(1024, (total_candidates + threads - 1) / threads);
  hash_lookup_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      candidate_keys.data_ptr<int32_t>(),
      nullptr,
      table_keys.data_ptr<int32_t>(),
      table_slots.data_ptr<int32_t>(),
      output.data_ptr<int32_t>(),
      static_cast<int>(candidate_keys.size(1)),
      static_cast<int>(table_keys.size(1)),
      total_candidates);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor hash_lookup_ragged_forward(
    torch::Tensor candidate_keys,
    torch::Tensor candidate_counts,
    torch::Tensor table_keys,
    torch::Tensor table_slots) {
  TORCH_CHECK(
      candidate_keys.is_cuda() && candidate_counts.is_cuda()
          && table_keys.is_cuda() && table_slots.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(
      candidate_keys.scalar_type() == at::kInt,
      "candidate keys must be int32");
  TORCH_CHECK(
      candidate_counts.scalar_type() == at::kLong,
      "candidate counts must be int64");
  TORCH_CHECK(
      candidate_keys.dim() == 2 && candidate_counts.dim() == 1
          && table_keys.dim() == 2
          && table_slots.sizes() == table_keys.sizes(),
      "invalid shapes");
  TORCH_CHECK(
      candidate_keys.size(0) == table_keys.size(0)
          && candidate_counts.size(0) == candidate_keys.size(0),
      "head count mismatch");
  c10::cuda::CUDAGuard device_guard(candidate_keys.device());
  auto output = torch::empty_like(candidate_keys);
  int total_candidates = static_cast<int>(candidate_keys.numel());
  int threads = 256;
  int blocks = std::min(1024, (total_candidates + threads - 1) / threads);
  hash_lookup_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      candidate_keys.data_ptr<int32_t>(),
      candidate_counts.data_ptr<int64_t>(),
      table_keys.data_ptr<int32_t>(),
      table_slots.data_ptr<int32_t>(),
      output.data_ptr<int32_t>(),
      static_cast<int>(candidate_keys.size(1)),
      static_cast<int>(table_keys.size(1)),
      total_candidates);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor variable_lru_update_forward(
    torch::Tensor resident_keys,
    torch::Tensor resident_ages,
    torch::Tensor candidate_keys,
    torch::Tensor lookup_slots) {
  TORCH_CHECK(resident_keys.is_cuda() && resident_ages.is_cuda(), "resident tensors must be CUDA");
  TORCH_CHECK(candidate_keys.is_cuda() && lookup_slots.is_cuda(), "candidate tensors must be CUDA");
  TORCH_CHECK(resident_keys.scalar_type() == at::kInt && candidate_keys.scalar_type() == at::kInt && lookup_slots.scalar_type() == at::kInt, "keys and slots must be int32");
  TORCH_CHECK(resident_ages.scalar_type() == at::kByte, "resident ages must be uint8");
  TORCH_CHECK(resident_keys.dim() == 2 && resident_ages.sizes() == resident_keys.sizes(), "invalid resident shapes");
  TORCH_CHECK(candidate_keys.dim() == 2 && lookup_slots.sizes() == candidate_keys.sizes(), "invalid candidate shapes");
  TORCH_CHECK(candidate_keys.size(0) == resident_keys.size(0), "head count mismatch");
  TORCH_CHECK(candidate_keys.size(1) <= resident_keys.size(1), "candidate count exceeds cache count");
  c10::cuda::CUDAGuard device_guard(resident_keys.device());
  auto output = torch::empty_like(candidate_keys);
  int selected_count = static_cast<int>(candidate_keys.size(1));
  size_t shared_bytes = static_cast<size_t>(256 + selected_count) * sizeof(int);
  variable_lru_update_kernel<<<
      resident_keys.size(0), 256, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
          resident_keys.data_ptr<int32_t>(),
          resident_ages.data_ptr<uint8_t>(),
          candidate_keys.data_ptr<int32_t>(),
          nullptr,
          lookup_slots.data_ptr<int32_t>(),
          output.data_ptr<int32_t>(),
          static_cast<int>(resident_keys.size(1)),
          selected_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor variable_lru_update_ragged_forward(
    torch::Tensor resident_keys,
    torch::Tensor resident_ages,
    torch::Tensor candidate_keys,
    torch::Tensor candidate_counts,
    torch::Tensor lookup_slots) {
  TORCH_CHECK(
      resident_keys.is_cuda() && resident_ages.is_cuda(),
      "resident tensors must be CUDA");
  TORCH_CHECK(
      candidate_keys.is_cuda() && candidate_counts.is_cuda()
          && lookup_slots.is_cuda(),
      "candidate tensors must be CUDA");
  TORCH_CHECK(
      resident_keys.scalar_type() == at::kInt
          && candidate_keys.scalar_type() == at::kInt
          && lookup_slots.scalar_type() == at::kInt,
      "keys and slots must be int32");
  TORCH_CHECK(
      candidate_counts.scalar_type() == at::kLong,
      "candidate counts must be int64");
  TORCH_CHECK(
      resident_ages.scalar_type() == at::kByte,
      "resident ages must be uint8");
  TORCH_CHECK(
      resident_keys.dim() == 2
          && resident_ages.sizes() == resident_keys.sizes(),
      "invalid resident shapes");
  TORCH_CHECK(
      candidate_keys.dim() == 2 && candidate_counts.dim() == 1
          && lookup_slots.sizes() == candidate_keys.sizes(),
      "invalid candidate shapes");
  TORCH_CHECK(
      candidate_keys.size(0) == resident_keys.size(0)
          && candidate_counts.size(0) == candidate_keys.size(0),
      "head count mismatch");
  TORCH_CHECK(
      candidate_keys.size(1) <= resident_keys.size(1),
      "candidate count exceeds cache count");
  c10::cuda::CUDAGuard device_guard(resident_keys.device());
  auto output = torch::empty_like(candidate_keys);
  int selected_count = static_cast<int>(candidate_keys.size(1));
  size_t shared_bytes =
      static_cast<size_t>(256 + selected_count) * sizeof(int);
  variable_lru_update_kernel<<<
      resident_keys.size(0),
      256,
      shared_bytes,
      at::cuda::getCurrentCUDAStream()>>>(
          resident_keys.data_ptr<int32_t>(),
          resident_ages.data_ptr<uint8_t>(),
          candidate_keys.data_ptr<int32_t>(),
          candidate_counts.data_ptr<int64_t>(),
          lookup_slots.data_ptr<int32_t>(),
          output.data_ptr<int32_t>(),
          static_cast<int>(resident_keys.size(1)),
          selected_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

void mapped_host_fill_variable_forward(
    torch::Tensor host_kv,
    torch::Tensor device_cache,
    torch::Tensor candidate_keys,
    torch::Tensor lookup_slots,
    torch::Tensor final_slots) {
  TORCH_CHECK(!host_kv.is_cuda() && host_kv.is_pinned() && host_kv.is_contiguous(), "host_kv must be contiguous pinned CPU memory");
  TORCH_CHECK(device_cache.is_cuda() && device_cache.is_contiguous(), "device cache must be contiguous CUDA memory");
  TORCH_CHECK(candidate_keys.is_cuda() && candidate_keys.scalar_type() == at::kInt, "candidate keys must be CUDA int32");
  TORCH_CHECK(lookup_slots.is_cuda() && lookup_slots.scalar_type() == at::kInt, "lookup slots must be CUDA int32");
  TORCH_CHECK(final_slots.is_cuda() && final_slots.scalar_type() == at::kInt, "final slots must be CUDA int32");
  TORCH_CHECK(host_kv.dim() == 4 && device_cache.dim() == 4 && host_kv.size(0) == 2 && device_cache.size(0) == 2, "invalid KV shapes");
  TORCH_CHECK(candidate_keys.dim() == 2 && lookup_slots.sizes() == candidate_keys.sizes() && final_slots.sizes() == candidate_keys.sizes(), "invalid directory shapes");
  TORCH_CHECK(host_kv.scalar_type() == device_cache.scalar_type(), "dtype mismatch");
  c10::cuda::CUDAGuard device_guard(device_cache.device());
  void* mapped_pointer = nullptr;
  cudaError_t status = cudaHostGetDevicePointer(&mapped_pointer, host_kv.data_ptr(), 0);
  TORCH_CHECK(status == cudaSuccess, "cudaHostGetDevicePointer failed: ", cudaGetErrorString(status));
  int kv_head_count = static_cast<int>(host_kv.size(1));
  int history_capacity = static_cast<int>(host_kv.size(2));
  int cache_count = static_cast<int>(device_cache.size(2));
  int selected_count = static_cast<int>(candidate_keys.size(1));
  int head_dim = static_cast<int>(host_kv.size(3));
  TORCH_CHECK(candidate_keys.size(0) == kv_head_count, "head count mismatch");
  int total_elements = 2 * kv_head_count * selected_count * head_dim;
  int threads = 256;
  int blocks = std::min(4096, (total_elements + threads - 1) / threads);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      device_cache.scalar_type(),
      "mapped_host_fill_variable_forward",
      [&] {
        mapped_host_fill_variable_kernel<scalar_t><<<
            blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                static_cast<const scalar_t*>(mapped_pointer),
                device_cache.data_ptr<scalar_t>(),
                candidate_keys.data_ptr<int32_t>(),
                nullptr,
                lookup_slots.data_ptr<int32_t>(),
                final_slots.data_ptr<int32_t>(),
                kv_head_count,
                history_capacity,
                cache_count,
                selected_count,
                head_dim,
                total_elements);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void mapped_host_fill_variable_ragged_forward(
    torch::Tensor host_kv,
    torch::Tensor device_cache,
    torch::Tensor candidate_keys,
    torch::Tensor candidate_counts,
    torch::Tensor lookup_slots,
    torch::Tensor final_slots) {
  TORCH_CHECK(
      !host_kv.is_cuda() && host_kv.is_pinned() && host_kv.is_contiguous(),
      "host_kv must be contiguous pinned CPU memory");
  TORCH_CHECK(
      device_cache.is_cuda() && device_cache.is_contiguous(),
      "device cache must be contiguous CUDA memory");
  TORCH_CHECK(
      candidate_keys.is_cuda()
          && candidate_keys.scalar_type() == at::kInt,
      "candidate keys must be CUDA int32");
  TORCH_CHECK(
      candidate_counts.is_cuda()
          && candidate_counts.scalar_type() == at::kLong,
      "candidate counts must be CUDA int64");
  TORCH_CHECK(
      lookup_slots.is_cuda() && lookup_slots.scalar_type() == at::kInt,
      "lookup slots must be CUDA int32");
  TORCH_CHECK(
      final_slots.is_cuda() && final_slots.scalar_type() == at::kInt,
      "final slots must be CUDA int32");
  TORCH_CHECK(
      host_kv.dim() == 4 && device_cache.dim() == 4
          && host_kv.size(0) == 2 && device_cache.size(0) == 2,
      "invalid KV shapes");
  TORCH_CHECK(
      candidate_keys.dim() == 2 && candidate_counts.dim() == 1
          && lookup_slots.sizes() == candidate_keys.sizes()
          && final_slots.sizes() == candidate_keys.sizes(),
      "invalid directory shapes");
  TORCH_CHECK(
      host_kv.scalar_type() == device_cache.scalar_type(),
      "dtype mismatch");
  c10::cuda::CUDAGuard device_guard(device_cache.device());
  void* mapped_pointer = nullptr;
  cudaError_t status =
      cudaHostGetDevicePointer(&mapped_pointer, host_kv.data_ptr(), 0);
  TORCH_CHECK(
      status == cudaSuccess,
      "cudaHostGetDevicePointer failed: ",
      cudaGetErrorString(status));
  int kv_head_count = static_cast<int>(host_kv.size(1));
  int history_capacity = static_cast<int>(host_kv.size(2));
  int cache_count = static_cast<int>(device_cache.size(2));
  int selected_count = static_cast<int>(candidate_keys.size(1));
  int head_dim = static_cast<int>(host_kv.size(3));
  TORCH_CHECK(
      candidate_keys.size(0) == kv_head_count
          && candidate_counts.size(0) == kv_head_count,
      "head count mismatch");
  int total_elements = 2 * kv_head_count * selected_count * head_dim;
  int threads = 256;
  int blocks = std::min(4096, (total_elements + threads - 1) / threads);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      device_cache.scalar_type(),
      "mapped_host_fill_variable_ragged_forward",
      [&] {
        mapped_host_fill_variable_kernel<scalar_t><<<
            blocks,
            threads,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
                static_cast<const scalar_t*>(mapped_pointer),
                device_cache.data_ptr<scalar_t>(),
                candidate_keys.data_ptr<int32_t>(),
                candidate_counts.data_ptr<int64_t>(),
                lookup_slots.data_ptr<int32_t>(),
                final_slots.data_ptr<int32_t>(),
                kv_head_count,
                history_capacity,
                cache_count,
                selected_count,
                head_dim,
                total_elements);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


@lru_cache(maxsize=1)
def load_hierarchical_cache_extension():
    return load_inline(
        name="hierarchical_cache_ext_v3",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=None,
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )
