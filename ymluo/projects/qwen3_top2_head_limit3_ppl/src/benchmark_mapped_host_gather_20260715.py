from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

import qabs_cuda_kernels as qabs_kernels


CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor mapped_host_gather_forward(
    torch::Tensor host_kv,
    torch::Tensor indices);

torch::Tensor hybrid_mapped_attention_forward(
    torch::Tensor host_kv,
    torch::Tensor device_cache,
    torch::Tensor query,
    torch::Tensor token_indices,
    torch::Tensor cache_slots,
    torch::Tensor destination_slots,
    double scaling,
    bool update_cache);

torch::Tensor hybrid_mapped_pack_forward(
    torch::Tensor host_kv,
    torch::Tensor device_cache,
    torch::Tensor token_indices,
    torch::Tensor cache_slots,
    torch::Tensor destination_slots,
    bool update_cache);

torch::Tensor gqa_hybrid_mapped_attention_forward(
    torch::Tensor host_kv,
    torch::Tensor device_cache,
    torch::Tensor query,
    torch::Tensor token_indices,
    torch::Tensor cache_slots,
    torch::Tensor destination_slots,
    double scaling,
    bool update_cache);

void mapped_host_fill_cache_forward(
    torch::Tensor host_kv,
    torch::Tensor device_cache,
    torch::Tensor miss_tokens,
    torch::Tensor destination_slots);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mapped_host_gather_forward", &mapped_host_gather_forward, "Mapped pinned-host KV gather");
  m.def("hybrid_mapped_attention_forward", &hybrid_mapped_attention_forward, "Hybrid GPU-cache and mapped-host attention");
  m.def("hybrid_mapped_pack_forward", &hybrid_mapped_pack_forward, "Pack hybrid GPU-cache and mapped-host KV");
  m.def("gqa_hybrid_mapped_attention_forward", &gqa_hybrid_mapped_attention_forward, "GQA-fused GPU-cache and mapped-host attention");
  m.def("mapped_host_fill_cache_forward", &mapped_host_fill_cache_forward, "Fill GPU cache misses from mapped host KV");
}
"""


CUDA_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void mapped_host_gather_kernel(
    const scalar_t* __restrict__ host_kv,
    const int64_t* __restrict__ indices,
    scalar_t* __restrict__ output,
    int kv_head_count,
    int history_count,
    int selected_count,
    int head_dim,
    int total_elements) {
  for (int flat = blockIdx.x * blockDim.x + threadIdx.x;
       flat < total_elements;
       flat += blockDim.x * gridDim.x) {
    int head_dim_index = flat % head_dim;
    int selected_index = (flat / head_dim) % selected_count;
    int kv_head = (flat / (head_dim * selected_count)) % kv_head_count;
    int kv_kind = flat / (head_dim * selected_count * kv_head_count);
    int64_t token = indices[kv_head * selected_count + selected_index];
    int64_t source = (
        ((static_cast<int64_t>(kv_kind) * kv_head_count + kv_head) * history_count + token)
        * head_dim + head_dim_index);
    output[flat] = host_kv[source];
  }
}

torch::Tensor mapped_host_gather_forward(
    torch::Tensor host_kv,
    torch::Tensor indices) {
  TORCH_CHECK(!host_kv.is_cuda(), "host_kv must be a CPU tensor");
  TORCH_CHECK(host_kv.is_pinned(), "host_kv must use pinned memory");
  TORCH_CHECK(host_kv.is_contiguous(), "host_kv must be contiguous");
  TORCH_CHECK(indices.is_cuda() && indices.scalar_type() == at::kLong, "indices must be CUDA int64");
  TORCH_CHECK(host_kv.dim() == 4 && host_kv.size(0) == 2, "host_kv must have shape [2, heads, history, dim]");
  TORCH_CHECK(indices.dim() == 2 && indices.size(0) == host_kv.size(1), "indices must have shape [heads, selected]");
  c10::cuda::CUDAGuard device_guard(indices.device());
  void* mapped_pointer = nullptr;
  cudaError_t status = cudaHostGetDevicePointer(&mapped_pointer, host_kv.data_ptr(), 0);
  TORCH_CHECK(status == cudaSuccess, "cudaHostGetDevicePointer failed: ", cudaGetErrorString(status));
  int kv_head_count = static_cast<int>(host_kv.size(1));
  int history_count = static_cast<int>(host_kv.size(2));
  int head_dim = static_cast<int>(host_kv.size(3));
  int selected_count = static_cast<int>(indices.size(1));
  auto output = torch::empty(
      {2, kv_head_count, selected_count, head_dim},
      indices.options().dtype(host_kv.scalar_type()));
  int total_elements = static_cast<int>(output.numel());
  int threads = 256;
  int blocks = std::min(4096, (total_elements + threads - 1) / threads);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      host_kv.scalar_type(),
      "mapped_host_gather_forward",
      [&] {
        mapped_host_gather_kernel<scalar_t><<<
            blocks,
            threads,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
                static_cast<const scalar_t*>(mapped_pointer),
                indices.data_ptr<int64_t>(),
                output.data_ptr<scalar_t>(),
                kv_head_count,
                history_count,
                selected_count,
                head_dim,
                total_elements);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

template <typename scalar_t>
__global__ void mapped_host_fill_cache_kernel(
    const scalar_t* __restrict__ host_kv,
    scalar_t* __restrict__ device_cache,
    const int64_t* __restrict__ miss_tokens,
    const int32_t* __restrict__ destination_slots,
    int kv_head_count,
    int history_count,
    int cache_count,
    int miss_count,
    int head_dim,
    int total_elements) {
  for (int flat = blockIdx.x * blockDim.x + threadIdx.x;
       flat < total_elements;
       flat += blockDim.x * gridDim.x) {
    int dim = flat % head_dim;
    int miss = (flat / head_dim) % miss_count;
    int kv_head = (flat / (head_dim * miss_count)) % kv_head_count;
    int kv_kind = flat / (head_dim * miss_count * kv_head_count);
    int directory_offset = kv_head * miss_count + miss;
    int64_t token = miss_tokens[directory_offset];
    int32_t destination = destination_slots[directory_offset];
    int64_t source = (
        (static_cast<int64_t>(kv_kind) * kv_head_count + kv_head) * history_count
        + token) * head_dim + dim;
    int64_t target = (
        (static_cast<int64_t>(kv_kind) * kv_head_count + kv_head) * cache_count
        + destination) * head_dim + dim;
    device_cache[target] = host_kv[source];
  }
}

void mapped_host_fill_cache_forward(
    torch::Tensor host_kv,
    torch::Tensor device_cache,
    torch::Tensor miss_tokens,
    torch::Tensor destination_slots) {
  TORCH_CHECK(!host_kv.is_cuda() && host_kv.is_pinned() && host_kv.is_contiguous(), "host_kv must be contiguous pinned CPU memory");
  TORCH_CHECK(device_cache.is_cuda(), "device cache must be CUDA");
  TORCH_CHECK(miss_tokens.is_cuda() && miss_tokens.scalar_type() == at::kLong, "miss tokens must be CUDA int64");
  TORCH_CHECK(destination_slots.is_cuda() && destination_slots.scalar_type() == at::kInt, "destination slots must be CUDA int32");
  TORCH_CHECK(host_kv.dim() == 4 && device_cache.dim() == 4 && host_kv.size(0) == 2 && device_cache.size(0) == 2, "invalid KV shapes");
  TORCH_CHECK(miss_tokens.dim() == 2 && destination_slots.sizes() == miss_tokens.sizes(), "invalid miss shapes");
  TORCH_CHECK(host_kv.scalar_type() == device_cache.scalar_type(), "dtype mismatch");
  c10::cuda::CUDAGuard device_guard(device_cache.device());
  void* mapped_pointer = nullptr;
  cudaError_t status = cudaHostGetDevicePointer(&mapped_pointer, host_kv.data_ptr(), 0);
  TORCH_CHECK(status == cudaSuccess, "cudaHostGetDevicePointer failed: ", cudaGetErrorString(status));
  int kv_head_count = static_cast<int>(host_kv.size(1));
  int history_count = static_cast<int>(host_kv.size(2));
  int cache_count = static_cast<int>(device_cache.size(2));
  int miss_count = static_cast<int>(miss_tokens.size(1));
  int head_dim = static_cast<int>(host_kv.size(3));
  int total_elements = 2 * kv_head_count * miss_count * head_dim;
  int threads = 256;
  int blocks = std::min(4096, (total_elements + threads - 1) / threads);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      device_cache.scalar_type(),
      "mapped_host_fill_cache_forward",
      [&] {
        mapped_host_fill_cache_kernel<scalar_t><<<
            blocks,
            threads,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
                static_cast<const scalar_t*>(mapped_pointer),
                device_cache.data_ptr<scalar_t>(),
                miss_tokens.data_ptr<int64_t>(),
                destination_slots.data_ptr<int32_t>(),
                kv_head_count,
                history_count,
                cache_count,
                miss_count,
                head_dim,
                total_elements);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
__global__ void hybrid_mapped_attention_kernel(
    const scalar_t* __restrict__ host_kv,
    scalar_t* __restrict__ device_cache,
    const scalar_t* __restrict__ query,
    const int64_t* __restrict__ token_indices,
    const int32_t* __restrict__ cache_slots,
    const int32_t* __restrict__ destination_slots,
    scalar_t* __restrict__ output,
    int query_head_count,
    int kv_head_count,
    int history_count,
    int cache_count,
    int selected_count,
    int head_dim,
    float scaling,
    bool update_cache) {
  extern __shared__ float shared[];
  float* reduction = shared;
  float* weights = shared + blockDim.x;
  int query_head = blockIdx.x;
  int tid = threadIdx.x;
  int groups = query_head_count / kv_head_count;
  int kv_head = query_head / groups;
  int group = query_head - kv_head * groups;
  const scalar_t* q_row = query + query_head * head_dim;
  const int64_t* token_row = token_indices + kv_head * selected_count;
  const int32_t* slot_row = cache_slots + kv_head * selected_count;
  const int32_t* destination_row = destination_slots + kv_head * selected_count;
  const scalar_t* host_k = host_kv + kv_head * history_count * head_dim;
  const scalar_t* host_v = host_kv +
      (static_cast<int64_t>(kv_head_count) * history_count +
       static_cast<int64_t>(kv_head) * history_count) * head_dim;
  scalar_t* cache_k = device_cache + kv_head * cache_count * head_dim;
  scalar_t* cache_v = device_cache +
      (static_cast<int64_t>(kv_head_count) * cache_count +
       static_cast<int64_t>(kv_head) * cache_count) * head_dim;

  float local_max = -1.0e30f;
  for (int selected = tid; selected < selected_count; selected += blockDim.x) {
    int32_t slot = slot_row[selected];
    int64_t token = token_row[selected];
    const scalar_t* key_vector = slot >= 0
        ? cache_k + static_cast<int64_t>(slot) * head_dim
        : host_k + token * head_dim;
    float score = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
      scalar_t key_value = key_vector[dim];
      score += static_cast<float>(q_row[dim]) * static_cast<float>(key_value);
      if (update_cache && group == 0 && slot < 0) {
        int32_t destination = destination_row[selected];
        if (destination >= 0 && destination < cache_count) {
          cache_k[static_cast<int64_t>(destination) * head_dim + dim] = key_value;
        }
      }
    }
    score *= scaling;
    weights[selected] = score;
    local_max = fmaxf(local_max, score);
  }
  reduction[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
    }
    __syncthreads();
  }
  float max_score = reduction[0];

  float local_denom = 0.0f;
  for (int selected = tid; selected < selected_count; selected += blockDim.x) {
    float weight = expf(weights[selected] - max_score);
    weights[selected] = weight;
    local_denom += weight;
  }
  reduction[tid] = local_denom;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }
  float denominator = fmaxf(reduction[0], 1.0e-20f);

  for (int dim = tid; dim < head_dim; dim += blockDim.x) {
    float accumulated = 0.0f;
    for (int selected = 0; selected < selected_count; ++selected) {
      int32_t slot = slot_row[selected];
      int64_t token = token_row[selected];
      const scalar_t* value_vector = slot >= 0
          ? cache_v + static_cast<int64_t>(slot) * head_dim
          : host_v + token * head_dim;
      scalar_t value = value_vector[dim];
      accumulated += (weights[selected] / denominator) * static_cast<float>(value);
      if (update_cache && group == 0 && slot < 0) {
        int32_t destination = destination_row[selected];
        if (destination >= 0 && destination < cache_count) {
          cache_v[static_cast<int64_t>(destination) * head_dim + dim] = value;
        }
      }
    }
    output[query_head * head_dim + dim] = static_cast<scalar_t>(accumulated);
  }
}

template <typename scalar_t>
__global__ void gqa_hybrid_mapped_attention_kernel(
    const scalar_t* __restrict__ host_kv,
    scalar_t* __restrict__ device_cache,
    const scalar_t* __restrict__ query,
    const int64_t* __restrict__ token_indices,
    const int32_t* __restrict__ cache_slots,
    const int32_t* __restrict__ destination_slots,
    scalar_t* __restrict__ output,
    int kv_head_count,
    int history_count,
    int cache_count,
    int selected_count,
    int head_dim,
    float scaling,
    bool update_cache) {
  constexpr int groups = 4;
  extern __shared__ float shared[];
  float* reduction = shared;
  float* weights = shared + blockDim.x;
  float* denominators = weights + groups * selected_count;
  int kv_head = blockIdx.x;
  int tid = threadIdx.x;
  const scalar_t* query_base = query + kv_head * groups * head_dim;
  const int64_t* token_row = token_indices + kv_head * selected_count;
  const int32_t* slot_row = cache_slots + kv_head * selected_count;
  const int32_t* destination_row = destination_slots + kv_head * selected_count;
  const scalar_t* host_k = host_kv + kv_head * history_count * head_dim;
  const scalar_t* host_v = host_kv +
      (static_cast<int64_t>(kv_head_count) * history_count +
       static_cast<int64_t>(kv_head) * history_count) * head_dim;
  scalar_t* cache_k = device_cache + kv_head * cache_count * head_dim;
  scalar_t* cache_v = device_cache +
      (static_cast<int64_t>(kv_head_count) * cache_count +
       static_cast<int64_t>(kv_head) * cache_count) * head_dim;

  for (int selected = tid; selected < selected_count; selected += blockDim.x) {
    int32_t slot = slot_row[selected];
    int64_t token = token_row[selected];
    const scalar_t* key_vector = slot >= 0
        ? cache_k + static_cast<int64_t>(slot) * head_dim
        : host_k + token * head_dim;
    float scores[groups] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int dim = 0; dim < head_dim; ++dim) {
      scalar_t key_value = key_vector[dim];
#pragma unroll
      for (int group = 0; group < groups; ++group) {
        scores[group] +=
            static_cast<float>(query_base[group * head_dim + dim]) *
            static_cast<float>(key_value);
      }
      if (update_cache && slot < 0) {
        int32_t destination = destination_row[selected];
        if (destination >= 0 && destination < cache_count) {
          cache_k[static_cast<int64_t>(destination) * head_dim + dim] = key_value;
        }
      }
    }
#pragma unroll
    for (int group = 0; group < groups; ++group) {
      weights[group * selected_count + selected] = scores[group] * scaling;
    }
  }
  __syncthreads();

#pragma unroll
  for (int group = 0; group < groups; ++group) {
    float local_max = -1.0e30f;
    for (int selected = tid; selected < selected_count; selected += blockDim.x) {
      local_max = fmaxf(local_max, weights[group * selected_count + selected]);
    }
    reduction[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
      }
      __syncthreads();
    }
    float max_score = reduction[0];
    float local_denom = 0.0f;
    for (int selected = tid; selected < selected_count; selected += blockDim.x) {
      int offset = group * selected_count + selected;
      float weight = expf(weights[offset] - max_score);
      weights[offset] = weight;
      local_denom += weight;
    }
    reduction[tid] = local_denom;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduction[tid] += reduction[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      denominators[group] = fmaxf(reduction[0], 1.0e-20f);
    }
    __syncthreads();
  }

  for (int dim = tid; dim < head_dim; dim += blockDim.x) {
    float accumulated[groups] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int selected = 0; selected < selected_count; ++selected) {
      int32_t slot = slot_row[selected];
      int64_t token = token_row[selected];
      const scalar_t* value_vector = slot >= 0
          ? cache_v + static_cast<int64_t>(slot) * head_dim
          : host_v + token * head_dim;
      scalar_t value_scalar = value_vector[dim];
      float value = static_cast<float>(value_scalar);
#pragma unroll
      for (int group = 0; group < groups; ++group) {
        accumulated[group] +=
            (weights[group * selected_count + selected] / denominators[group]) *
            value;
      }
      if (update_cache && slot < 0) {
        int32_t destination = destination_row[selected];
        if (destination >= 0 && destination < cache_count) {
          cache_v[static_cast<int64_t>(destination) * head_dim + dim] = value_scalar;
        }
      }
    }
#pragma unroll
    for (int group = 0; group < groups; ++group) {
      output[(kv_head * groups + group) * head_dim + dim] =
          static_cast<scalar_t>(accumulated[group]);
    }
  }
}

torch::Tensor hybrid_mapped_attention_forward(
    torch::Tensor host_kv,
    torch::Tensor device_cache,
    torch::Tensor query,
    torch::Tensor token_indices,
    torch::Tensor cache_slots,
    torch::Tensor destination_slots,
    double scaling,
    bool update_cache) {
  TORCH_CHECK(!host_kv.is_cuda() && host_kv.is_pinned() && host_kv.is_contiguous(), "host_kv must be contiguous pinned CPU memory");
  TORCH_CHECK(device_cache.is_cuda() && query.is_cuda(), "cache and query must be CUDA");
  TORCH_CHECK(token_indices.is_cuda() && token_indices.scalar_type() == at::kLong, "token indices must be CUDA int64");
  TORCH_CHECK(cache_slots.is_cuda() && cache_slots.scalar_type() == at::kInt, "cache slots must be CUDA int32");
  TORCH_CHECK(destination_slots.is_cuda() && destination_slots.scalar_type() == at::kInt, "destination slots must be CUDA int32");
  TORCH_CHECK(host_kv.dim() == 4 && device_cache.dim() == 4 && host_kv.size(0) == 2 && device_cache.size(0) == 2, "invalid KV shapes");
  TORCH_CHECK(query.dim() == 3 && query.size(0) == 1, "query must have shape [1, heads, dim]");
  TORCH_CHECK(token_indices.dim() == 2 && cache_slots.sizes() == token_indices.sizes() && destination_slots.sizes() == token_indices.sizes(), "invalid directory shapes");
  TORCH_CHECK(host_kv.scalar_type() == device_cache.scalar_type() && host_kv.scalar_type() == query.scalar_type(), "dtype mismatch");
  c10::cuda::CUDAGuard device_guard(query.device());
  void* mapped_pointer = nullptr;
  cudaError_t status = cudaHostGetDevicePointer(&mapped_pointer, host_kv.data_ptr(), 0);
  TORCH_CHECK(status == cudaSuccess, "cudaHostGetDevicePointer failed: ", cudaGetErrorString(status));
  int query_head_count = static_cast<int>(query.size(1));
  int kv_head_count = static_cast<int>(host_kv.size(1));
  int history_count = static_cast<int>(host_kv.size(2));
  int cache_count = static_cast<int>(device_cache.size(2));
  int selected_count = static_cast<int>(token_indices.size(1));
  int head_dim = static_cast<int>(query.size(2));
  TORCH_CHECK(query_head_count % kv_head_count == 0, "invalid GQA grouping");
  auto output = torch::empty({1, query_head_count, head_dim}, query.options());
  int threads = 128;
  size_t shared_bytes = static_cast<size_t>(threads + selected_count) * sizeof(float);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "hybrid_mapped_attention_forward",
      [&] {
        hybrid_mapped_attention_kernel<scalar_t><<<
            query_head_count,
            threads,
            shared_bytes,
            at::cuda::getCurrentCUDAStream()>>>(
                static_cast<const scalar_t*>(mapped_pointer),
                device_cache.data_ptr<scalar_t>(),
                query.data_ptr<scalar_t>(),
                token_indices.data_ptr<int64_t>(),
                cache_slots.data_ptr<int32_t>(),
                destination_slots.data_ptr<int32_t>(),
                output.data_ptr<scalar_t>(),
                query_head_count,
                kv_head_count,
                history_count,
                cache_count,
                selected_count,
                head_dim,
                static_cast<float>(scaling),
                update_cache);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor gqa_hybrid_mapped_attention_forward(
    torch::Tensor host_kv,
    torch::Tensor device_cache,
    torch::Tensor query,
    torch::Tensor token_indices,
    torch::Tensor cache_slots,
    torch::Tensor destination_slots,
    double scaling,
    bool update_cache) {
  TORCH_CHECK(!host_kv.is_cuda() && host_kv.is_pinned() && host_kv.is_contiguous(), "host_kv must be contiguous pinned CPU memory");
  TORCH_CHECK(device_cache.is_cuda() && query.is_cuda(), "cache and query must be CUDA");
  TORCH_CHECK(token_indices.is_cuda() && token_indices.scalar_type() == at::kLong, "token indices must be CUDA int64");
  TORCH_CHECK(cache_slots.is_cuda() && cache_slots.scalar_type() == at::kInt, "cache slots must be CUDA int32");
  TORCH_CHECK(destination_slots.is_cuda() && destination_slots.scalar_type() == at::kInt, "destination slots must be CUDA int32");
  TORCH_CHECK(host_kv.dim() == 4 && device_cache.dim() == 4 && host_kv.size(0) == 2 && device_cache.size(0) == 2, "invalid KV shapes");
  TORCH_CHECK(query.dim() == 3 && query.size(0) == 1, "query must have shape [1, heads, dim]");
  TORCH_CHECK(token_indices.dim() == 2 && cache_slots.sizes() == token_indices.sizes() && destination_slots.sizes() == token_indices.sizes(), "invalid directory shapes");
  TORCH_CHECK(host_kv.scalar_type() == device_cache.scalar_type() && host_kv.scalar_type() == query.scalar_type(), "dtype mismatch");
  c10::cuda::CUDAGuard device_guard(query.device());
  void* mapped_pointer = nullptr;
  cudaError_t status = cudaHostGetDevicePointer(&mapped_pointer, host_kv.data_ptr(), 0);
  TORCH_CHECK(status == cudaSuccess, "cudaHostGetDevicePointer failed: ", cudaGetErrorString(status));
  int query_head_count = static_cast<int>(query.size(1));
  int kv_head_count = static_cast<int>(host_kv.size(1));
  int history_count = static_cast<int>(host_kv.size(2));
  int cache_count = static_cast<int>(device_cache.size(2));
  int selected_count = static_cast<int>(token_indices.size(1));
  int head_dim = static_cast<int>(query.size(2));
  TORCH_CHECK(query_head_count == kv_head_count * 4, "kernel currently requires four query heads per KV head");
  auto output = torch::empty({1, query_head_count, head_dim}, query.options());
  int threads = 128;
  size_t shared_bytes = static_cast<size_t>(threads + 4 * selected_count + 4) * sizeof(float);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "gqa_hybrid_mapped_attention_forward",
      [&] {
        gqa_hybrid_mapped_attention_kernel<scalar_t><<<
            kv_head_count,
            threads,
            shared_bytes,
            at::cuda::getCurrentCUDAStream()>>>(
                static_cast<const scalar_t*>(mapped_pointer),
                device_cache.data_ptr<scalar_t>(),
                query.data_ptr<scalar_t>(),
                token_indices.data_ptr<int64_t>(),
                cache_slots.data_ptr<int32_t>(),
                destination_slots.data_ptr<int32_t>(),
                output.data_ptr<scalar_t>(),
                kv_head_count,
                history_count,
                cache_count,
                selected_count,
                head_dim,
                static_cast<float>(scaling),
                update_cache);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

template <typename scalar_t>
__global__ void hybrid_mapped_pack_kernel(
    const scalar_t* __restrict__ host_kv,
    scalar_t* __restrict__ device_cache,
    const int64_t* __restrict__ token_indices,
    const int32_t* __restrict__ cache_slots,
    const int32_t* __restrict__ destination_slots,
    scalar_t* __restrict__ output,
    int kv_head_count,
    int history_count,
    int cache_count,
    int selected_count,
    int head_dim,
    int total_elements,
    bool update_cache) {
  for (int flat = blockIdx.x * blockDim.x + threadIdx.x;
       flat < total_elements;
       flat += blockDim.x * gridDim.x) {
    int dim = flat % head_dim;
    int selected = (flat / head_dim) % selected_count;
    int kv_head = (flat / (head_dim * selected_count)) % kv_head_count;
    int kv_kind = flat / (head_dim * selected_count * kv_head_count);
    int directory_offset = kv_head * selected_count + selected;
    int32_t slot = cache_slots[directory_offset];
    int64_t token = token_indices[directory_offset];
    int64_t source;
    if (slot >= 0) {
      source = (
          (static_cast<int64_t>(kv_kind) * kv_head_count + kv_head) * cache_count
          + slot) * head_dim + dim;
      output[flat] = device_cache[source];
    } else {
      source = (
          (static_cast<int64_t>(kv_kind) * kv_head_count + kv_head) * history_count
          + token) * head_dim + dim;
      scalar_t value = host_kv[source];
      output[flat] = value;
      if (update_cache) {
        int32_t destination = destination_slots[directory_offset];
        if (destination >= 0 && destination < cache_count) {
          int64_t cache_destination = (
              (static_cast<int64_t>(kv_kind) * kv_head_count + kv_head) * cache_count
              + destination) * head_dim + dim;
          device_cache[cache_destination] = value;
        }
      }
    }
  }
}

torch::Tensor hybrid_mapped_pack_forward(
    torch::Tensor host_kv,
    torch::Tensor device_cache,
    torch::Tensor token_indices,
    torch::Tensor cache_slots,
    torch::Tensor destination_slots,
    bool update_cache) {
  TORCH_CHECK(!host_kv.is_cuda() && host_kv.is_pinned() && host_kv.is_contiguous(), "host_kv must be contiguous pinned CPU memory");
  TORCH_CHECK(device_cache.is_cuda(), "device cache must be CUDA");
  TORCH_CHECK(token_indices.is_cuda() && token_indices.scalar_type() == at::kLong, "token indices must be CUDA int64");
  TORCH_CHECK(cache_slots.is_cuda() && cache_slots.scalar_type() == at::kInt, "cache slots must be CUDA int32");
  TORCH_CHECK(destination_slots.is_cuda() && destination_slots.scalar_type() == at::kInt, "destination slots must be CUDA int32");
  TORCH_CHECK(host_kv.dim() == 4 && device_cache.dim() == 4 && host_kv.size(0) == 2 && device_cache.size(0) == 2, "invalid KV shapes");
  TORCH_CHECK(token_indices.dim() == 2 && cache_slots.sizes() == token_indices.sizes() && destination_slots.sizes() == token_indices.sizes(), "invalid directory shapes");
  TORCH_CHECK(host_kv.scalar_type() == device_cache.scalar_type(), "dtype mismatch");
  c10::cuda::CUDAGuard device_guard(device_cache.device());
  void* mapped_pointer = nullptr;
  cudaError_t status = cudaHostGetDevicePointer(&mapped_pointer, host_kv.data_ptr(), 0);
  TORCH_CHECK(status == cudaSuccess, "cudaHostGetDevicePointer failed: ", cudaGetErrorString(status));
  int kv_head_count = static_cast<int>(host_kv.size(1));
  int history_count = static_cast<int>(host_kv.size(2));
  int cache_count = static_cast<int>(device_cache.size(2));
  int selected_count = static_cast<int>(token_indices.size(1));
  int head_dim = static_cast<int>(host_kv.size(3));
  auto output = torch::empty(
      {2, kv_head_count, selected_count, head_dim},
      device_cache.options());
  int total_elements = static_cast<int>(output.numel());
  int threads = 256;
  int blocks = std::min(4096, (total_elements + threads - 1) / threads);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      device_cache.scalar_type(),
      "hybrid_mapped_pack_forward",
      [&] {
        hybrid_mapped_pack_kernel<scalar_t><<<
            blocks,
            threads,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
                static_cast<const scalar_t*>(mapped_pointer),
                device_cache.data_ptr<scalar_t>(),
                token_indices.data_ptr<int64_t>(),
                cache_slots.data_ptr<int32_t>(),
                destination_slots.data_ptr<int32_t>(),
                output.data_ptr<scalar_t>(),
                kv_head_count,
                history_count,
                cache_count,
                selected_count,
                head_dim,
                total_elements,
                update_cache);
      });
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
    parser.add_argument("--selected_fraction", type=float, default=0.0040613)
    parser.add_argument("--attention_fraction", type=float, default=0.02)
    parser.add_argument("--cache_fraction", type=float, default=0.032)
    parser.add_argument("--cache_hit_rate", type=float, default=0.79)
    parser.add_argument("--index_order", choices=("random", "token_sorted"), default="random")
    parser.add_argument("--kv_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    module = load_inline(
        name="mapped_host_gather_ext_v6",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=None,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    selected_count = max(1, math.ceil(args.history_count * args.selected_fraction))
    host_kv = torch.empty(
        (2, args.kv_heads, args.history_count, args.head_dim),
        dtype=torch.float16,
        pin_memory=True,
    )
    host_kv.random_(-64, 64)
    indices = torch.randint(
        0,
        args.history_count,
        (args.kv_heads, selected_count),
        dtype=torch.long,
        device="cuda",
    )
    expected = torch.gather(
        host_kv,
        2,
        indices.cpu().unsqueeze(0).unsqueeze(-1).expand(2, -1, -1, args.head_dim),
    )
    actual = module.mapped_host_gather_forward(host_kv, indices)
    torch.cuda.synchronize()
    max_error = float((actual.cpu() - expected).abs().max().item())

    def mapped_gather() -> torch.Tensor:
        return module.mapped_host_gather_forward(host_kv, indices)

    mapped_ms = timed_ms(mapped_gather, args.warmup, args.repeats)
    attention_count = max(1, math.ceil(args.history_count * args.attention_fraction))
    device_selected = torch.randn(
        (2, args.kv_heads, attention_count, args.head_dim),
        dtype=torch.float16,
        device="cuda",
    )
    destination_positions = torch.arange(
        selected_count, dtype=torch.long, device="cuda"
    ).unsqueeze(0).expand(args.kv_heads, -1)
    destination_gather = (
        destination_positions.unsqueeze(0)
        .unsqueeze(-1)
        .expand(2, -1, -1, args.head_dim)
    )
    query_heads = args.kv_heads * 4
    query = torch.randn((1, query_heads, args.head_dim), dtype=torch.float16, device="cuda")
    attention_indices = (
        torch.arange(attention_count, dtype=torch.long, device="cuda")
        .reshape(1, 1, -1)
        .expand(1, query_heads, -1)
        .contiguous()
    )
    attention_counts = torch.full(
        (1, query_heads), attention_count, dtype=torch.long, device="cuda"
    )

    def final_attention() -> torch.Tensor:
        return qabs_kernels.final_attention_ragged(
            query,
            device_selected[0].unsqueeze(0),
            device_selected[1].unsqueeze(0),
            attention_indices,
            attention_counts,
            args.head_dim**-0.5,
        )

    def mapped_fetch_and_attention() -> torch.Tensor:
        miss_data = module.mapped_host_gather_forward(host_kv, indices)
        device_selected.scatter_(2, destination_gather, miss_data)
        return final_attention()

    final_attention_ms = timed_ms(final_attention, args.warmup, args.repeats)
    fetch_and_attention_ms = timed_ms(
        mapped_fetch_and_attention, args.warmup, args.repeats
    )

    cache_count = max(1, math.floor(args.history_count * args.cache_fraction))
    hit_count = min(attention_count, round(attention_count * args.cache_hit_rate))
    miss_count = attention_count - hit_count
    device_cache = torch.randn(
        (2, args.kv_heads, cache_count, args.head_dim),
        dtype=torch.float16,
        device="cuda",
    )
    token_rows = []
    cache_slot_rows = []
    destination_rows = []
    unsorted_token_rows = []
    unsorted_cache_slot_rows = []
    unsorted_destination_rows = []
    generator = torch.Generator().manual_seed(20260715)
    for _ in range(args.kv_heads):
        tokens = torch.randperm(args.history_count, generator=generator)[:attention_count]
        slots = torch.randperm(cache_count, generator=generator)[:attention_count]
        hit_mask = torch.zeros(attention_count, dtype=torch.bool)
        hit_mask[:hit_count] = True
        permutation = torch.randperm(attention_count, generator=generator)
        hit_mask = hit_mask[permutation]
        tokens = tokens[permutation]
        source_slots = torch.where(hit_mask, slots, torch.full_like(slots, -1))
        unsorted_token_rows.append(tokens.clone())
        unsorted_cache_slot_rows.append(source_slots.to(torch.int32).clone())
        unsorted_destination_rows.append(slots.to(torch.int32).clone())
        if args.index_order == "token_sorted":
            order = torch.argsort(tokens)
            tokens = tokens[order]
            source_slots = source_slots[order]
            slots = slots[order]
        token_rows.append(tokens)
        cache_slot_rows.append(source_slots.to(torch.int32))
        destination_rows.append(slots.to(torch.int32))
    token_indices = torch.stack(token_rows).cuda()
    cache_slots = torch.stack(cache_slot_rows).cuda()
    destination_slots = torch.stack(destination_rows).cuda()
    unsorted_token_indices = torch.stack(unsorted_token_rows).cuda()
    unsorted_cache_slots = torch.stack(unsorted_cache_slot_rows).cuda()
    unsorted_destination_slots = torch.stack(unsorted_destination_rows).cuda()

    def materialize_selected_kv(
        selected_tokens: torch.Tensor,
        selected_slots: torch.Tensor,
    ) -> torch.Tensor:
        materialized = torch.empty(
            (2, args.kv_heads, attention_count, args.head_dim),
            dtype=torch.float16,
            device="cuda",
        )
        for head in range(args.kv_heads):
            head_slots = selected_slots[head]
            head_tokens = selected_tokens[head]
            hit_mask = head_slots.ge(0)
            materialized[:, head, hit_mask] = device_cache[
                :, head, head_slots[hit_mask].to(torch.long)
            ]
            materialized[:, head, ~hit_mask] = host_kv[
                :, head, head_tokens[~hit_mask].cpu()
            ].cuda()
        return materialized

    selected_kv = materialize_selected_kv(token_indices, cache_slots)
    unsorted_selected_kv = materialize_selected_kv(
        unsorted_token_indices,
        unsorted_cache_slots,
    )
    online_sorted_order = torch.argsort(unsorted_token_indices, dim=-1)
    online_sorted_selected_kv = materialize_selected_kv(
        torch.gather(unsorted_token_indices, 1, online_sorted_order),
        torch.gather(unsorted_cache_slots, 1, online_sorted_order),
    )

    def hybrid_attention(update_cache: bool) -> torch.Tensor:
        return module.hybrid_mapped_attention_forward(
            host_kv,
            device_cache,
            query,
            token_indices,
            cache_slots,
            destination_slots,
            args.head_dim**-0.5,
            update_cache,
        )

    hybrid_output = hybrid_attention(False)
    reference_output = qabs_kernels.final_attention_ragged(
        query,
        selected_kv[0].unsqueeze(0),
        selected_kv[1].unsqueeze(0),
        attention_indices,
        attention_counts,
        args.head_dim**-0.5,
    ).squeeze(1)
    online_sorted_reference_output = qabs_kernels.final_attention_ragged(
        query,
        online_sorted_selected_kv[0].unsqueeze(0),
        online_sorted_selected_kv[1].unsqueeze(0),
        attention_indices,
        attention_counts,
        args.head_dim**-0.5,
    ).squeeze(1)
    online_unsorted_reference_output = qabs_kernels.final_attention_ragged(
        query,
        unsorted_selected_kv[0].unsqueeze(0),
        unsorted_selected_kv[1].unsqueeze(0),
        attention_indices,
        attention_counts,
        args.head_dim**-0.5,
    ).squeeze(1)
    torch.cuda.synchronize()
    hybrid_max_error = float(
        (hybrid_output.float() - reference_output.float()).abs().max().item()
    )
    hybrid_attention_ms = timed_ms(
        lambda: hybrid_attention(False), args.warmup, args.repeats
    )
    hybrid_attention_update_ms = timed_ms(
        lambda: hybrid_attention(True), args.warmup, args.repeats
    )

    def gqa_hybrid_attention(update_cache: bool) -> torch.Tensor:
        return module.gqa_hybrid_mapped_attention_forward(
            host_kv,
            device_cache,
            query,
            token_indices,
            cache_slots,
            destination_slots,
            args.head_dim**-0.5,
            update_cache,
        )

    gqa_hybrid_output = gqa_hybrid_attention(False)
    torch.cuda.synchronize()
    gqa_hybrid_max_error = float(
        (gqa_hybrid_output.float() - reference_output.float()).abs().max().item()
    )
    gqa_hybrid_attention_ms = timed_ms(
        lambda: gqa_hybrid_attention(False), args.warmup, args.repeats
    )
    gqa_hybrid_attention_update_ms = timed_ms(
        lambda: gqa_hybrid_attention(True), args.warmup, args.repeats
    )

    miss_mask = cache_slots.lt(0)
    compact_miss_tokens = token_indices[miss_mask].reshape(
        args.kv_heads, miss_count
    )
    compact_miss_destinations = destination_slots[miss_mask].reshape(
        args.kv_heads, miss_count
    )
    final_cache_slots = torch.where(
        miss_mask, destination_slots, cache_slots
    )
    final_cache_indices = (
        final_cache_slots.unsqueeze(1)
        .expand(-1, 4, -1)
        .reshape(1, query_heads, attention_count)
        .to(torch.long)
        .contiguous()
    )

    def fill_cache_misses() -> None:
        module.mapped_host_fill_cache_forward(
            host_kv,
            device_cache,
            compact_miss_tokens,
            compact_miss_destinations,
        )

    def resident_cache_attention() -> torch.Tensor:
        return qabs_kernels.final_attention_ragged(
            query,
            device_cache[0].unsqueeze(0),
            device_cache[1].unsqueeze(0),
            final_cache_indices,
            attention_counts,
            args.head_dim**-0.5,
        )

    expanded_key_cache = (
        device_cache[0]
        .unsqueeze(1)
        .expand(args.kv_heads, 4, cache_count, args.head_dim)
        .reshape(1, query_heads, cache_count, args.head_dim)
    )
    expanded_value_cache = (
        device_cache[1]
        .unsqueeze(1)
        .expand(args.kv_heads, 4, cache_count, args.head_dim)
        .reshape(1, query_heads, cache_count, args.head_dim)
    )
    gather_index = final_cache_indices.unsqueeze(-1).expand(
        -1, -1, -1, args.head_dim
    )

    def gather_selected_cache() -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.gather(expanded_key_cache, 2, gather_index),
            torch.gather(expanded_value_cache, 2, gather_index),
        )

    selected_cache_key, selected_cache_value = gather_selected_cache()

    def selected_sdpa_attention() -> torch.Tensor:
        return F.scaled_dot_product_attention(
            query.unsqueeze(2),
            selected_cache_key,
            selected_cache_value,
            dropout_p=0.0,
            is_causal=False,
        )

    def gather_cache_sdpa_attention() -> torch.Tensor:
        selected_key, selected_value = gather_selected_cache()
        return F.scaled_dot_product_attention(
            query.unsqueeze(2),
            selected_key,
            selected_value,
            dropout_p=0.0,
            is_causal=False,
        )

    fill_cache_misses()
    cache_attention_output = resident_cache_attention().squeeze(1)
    sdpa_attention_output = selected_sdpa_attention().squeeze(2)
    torch.cuda.synchronize()
    resident_cache_max_error = float(
        (cache_attention_output.float() - reference_output.float()).abs().max().item()
    )
    sdpa_attention_max_error = float(
        (sdpa_attention_output.float() - reference_output.float()).abs().max().item()
    )

    def fill_cache_and_attention() -> torch.Tensor:
        fill_cache_misses()
        return resident_cache_attention()

    def online_address_sort_metadata() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        order = torch.argsort(unsorted_token_indices, dim=-1)
        return (
            torch.gather(unsorted_token_indices, 1, order),
            torch.gather(unsorted_cache_slots, 1, order),
            torch.gather(unsorted_destination_slots, 1, order),
        )

    def online_address_sort_fill_cache_attention() -> torch.Tensor:
        sorted_tokens, sorted_cache_slots, sorted_destinations = (
            online_address_sort_metadata()
        )
        sorted_miss_mask = sorted_cache_slots.lt(0)
        sorted_miss_tokens = sorted_tokens[sorted_miss_mask].reshape(
            args.kv_heads, miss_count
        )
        sorted_miss_destinations = sorted_destinations[
            sorted_miss_mask
        ].reshape(args.kv_heads, miss_count)
        module.mapped_host_fill_cache_forward(
            host_kv,
            device_cache,
            sorted_miss_tokens,
            sorted_miss_destinations,
        )
        sorted_final_slots = torch.where(
            sorted_miss_mask, sorted_destinations, sorted_cache_slots
        )
        sorted_final_indices = (
            sorted_final_slots.unsqueeze(1)
            .expand(-1, 4, -1)
            .reshape(1, query_heads, attention_count)
            .to(torch.long)
            .contiguous()
        )
        return qabs_kernels.final_attention_ragged(
            query,
            device_cache[0].unsqueeze(0),
            device_cache[1].unsqueeze(0),
            sorted_final_indices,
            attention_counts,
            args.head_dim**-0.5,
        )

    def online_miss_compact_sort_fill_cache_attention() -> torch.Tensor:
        online_miss_mask = unsorted_cache_slots.lt(0)
        online_miss_tokens = unsorted_token_indices[online_miss_mask].reshape(
            args.kv_heads, miss_count
        )
        online_miss_destinations = unsorted_destination_slots[
            online_miss_mask
        ].reshape(args.kv_heads, miss_count)
        miss_order = torch.argsort(online_miss_tokens, dim=-1)
        sorted_miss_tokens = torch.gather(online_miss_tokens, 1, miss_order)
        sorted_miss_destinations = torch.gather(
            online_miss_destinations, 1, miss_order
        )
        module.mapped_host_fill_cache_forward(
            host_kv,
            device_cache,
            sorted_miss_tokens,
            sorted_miss_destinations,
        )
        online_final_slots = torch.where(
            online_miss_mask,
            unsorted_destination_slots,
            unsorted_cache_slots,
        )
        online_final_indices = (
            online_final_slots.unsqueeze(1)
            .expand(-1, 4, -1)
            .reshape(1, query_heads, attention_count)
            .to(torch.long)
            .contiguous()
        )
        return qabs_kernels.final_attention_ragged(
            query,
            device_cache[0].unsqueeze(0),
            device_cache[1].unsqueeze(0),
            online_final_indices,
            attention_counts,
            args.head_dim**-0.5,
        )

    online_address_sort_output = online_address_sort_fill_cache_attention().squeeze(1)
    online_miss_sort_output = (
        online_miss_compact_sort_fill_cache_attention().squeeze(1)
    )
    torch.cuda.synchronize()
    online_address_sort_max_error = float(
        (
            online_address_sort_output.float()
            - online_sorted_reference_output.float()
        )
        .abs()
        .max()
        .item()
    )
    online_miss_sort_max_error = float(
        (
            online_miss_sort_output.float()
            - online_unsorted_reference_output.float()
        )
        .abs()
        .max()
        .item()
    )

    fill_cache_ms = timed_ms(
        fill_cache_misses, args.warmup, args.repeats
    )
    fill_cache_attention_ms = timed_ms(
        fill_cache_and_attention, args.warmup, args.repeats
    )
    online_address_sort_metadata_ms = timed_ms(
        online_address_sort_metadata,
        args.warmup,
        args.repeats,
    )
    online_address_sort_fill_cache_attention_ms = timed_ms(
        online_address_sort_fill_cache_attention,
        args.warmup,
        args.repeats,
    )
    online_miss_compact_sort_fill_cache_attention_ms = timed_ms(
        online_miss_compact_sort_fill_cache_attention,
        args.warmup,
        args.repeats,
    )
    gather_cache_ms = timed_ms(
        gather_selected_cache, args.warmup, args.repeats
    )
    selected_sdpa_attention_ms = timed_ms(
        selected_sdpa_attention, args.warmup, args.repeats
    )
    gather_cache_sdpa_attention_ms = timed_ms(
        gather_cache_sdpa_attention, args.warmup, args.repeats
    )

    def hybrid_pack(update_cache: bool) -> torch.Tensor:
        return module.hybrid_mapped_pack_forward(
            host_kv,
            device_cache,
            token_indices,
            cache_slots,
            destination_slots,
            update_cache,
        )

    hybrid_packed = hybrid_pack(False)
    torch.cuda.synchronize()
    hybrid_pack_max_error = float(
        (hybrid_packed.float() - selected_kv.float()).abs().max().item()
    )

    def hybrid_pack_attention(update_cache: bool) -> torch.Tensor:
        packed = hybrid_pack(update_cache)
        return qabs_kernels.final_attention_ragged(
            query,
            packed[0].unsqueeze(0),
            packed[1].unsqueeze(0),
            attention_indices,
            attention_counts,
            args.head_dim**-0.5,
        )

    hybrid_pack_ms = timed_ms(
        lambda: hybrid_pack(False), args.warmup, args.repeats
    )
    hybrid_pack_update_ms = timed_ms(
        lambda: hybrid_pack(True), args.warmup, args.repeats
    )
    hybrid_pack_attention_ms = timed_ms(
        lambda: hybrid_pack_attention(False), args.warmup, args.repeats
    )
    hybrid_pack_update_attention_ms = timed_ms(
        lambda: hybrid_pack_attention(True), args.warmup, args.repeats
    )
    transferred_bytes = actual.numel() * actual.element_size()
    result = {
        "history_count": args.history_count,
        "selected_fraction": args.selected_fraction,
        "selected_count_per_kv_head": selected_count,
        "transferred_mib_per_layer": transferred_bytes / 2**20,
        "mapped_gather_ms_per_layer": mapped_ms,
        "attention_fraction": args.attention_fraction,
        "attention_count_per_kv_head": attention_count,
        "packed_final_attention_ms_per_layer": final_attention_ms,
        "mapped_fetch_scatter_attention_ms_per_layer": fetch_and_attention_ms,
        "cache_fraction": args.cache_fraction,
        "cache_hit_rate": args.cache_hit_rate,
        "index_order": args.index_order,
        "hybrid_mapped_attention_ms_per_layer": hybrid_attention_ms,
        "hybrid_mapped_attention_cache_update_ms_per_layer": hybrid_attention_update_ms,
        "hybrid_max_abs_error": hybrid_max_error,
        "gqa_hybrid_mapped_attention_ms_per_layer": gqa_hybrid_attention_ms,
        "gqa_hybrid_mapped_attention_cache_update_ms_per_layer": gqa_hybrid_attention_update_ms,
        "gqa_hybrid_max_abs_error": gqa_hybrid_max_error,
        "mapped_host_fill_cache_ms_per_layer": fill_cache_ms,
        "mapped_host_fill_cache_attention_ms_per_layer": fill_cache_attention_ms,
        "online_address_sort_metadata_ms_per_layer": online_address_sort_metadata_ms,
        "online_address_sort_fill_cache_attention_ms_per_layer": (
            online_address_sort_fill_cache_attention_ms
        ),
        "online_miss_compact_sort_fill_cache_attention_ms_per_layer": (
            online_miss_compact_sort_fill_cache_attention_ms
        ),
        "online_address_sort_max_abs_error": online_address_sort_max_error,
        "online_miss_sort_max_abs_error": online_miss_sort_max_error,
        "resident_cache_attention_max_abs_error": resident_cache_max_error,
        "gather_resident_cache_ms_per_layer": gather_cache_ms,
        "selected_sdpa_attention_ms_per_layer": selected_sdpa_attention_ms,
        "gather_resident_cache_sdpa_ms_per_layer": gather_cache_sdpa_attention_ms,
        "selected_sdpa_attention_max_abs_error": sdpa_attention_max_error,
        "hybrid_pack_ms_per_layer": hybrid_pack_ms,
        "hybrid_pack_cache_update_ms_per_layer": hybrid_pack_update_ms,
        "hybrid_pack_attention_ms_per_layer": hybrid_pack_attention_ms,
        "hybrid_pack_cache_update_attention_ms_per_layer": hybrid_pack_update_attention_ms,
        "hybrid_pack_max_abs_error": hybrid_pack_max_error,
        "effective_gib_per_second": transferred_bytes / 2**30 / (mapped_ms / 1000.0),
        "max_abs_error": max_error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
