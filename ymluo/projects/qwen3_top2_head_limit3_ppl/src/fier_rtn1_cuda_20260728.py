from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import torch
from torch.utils.cpp_extension import load_inline


CPP_SOURCE = r"""
#include <torch/extension.h>

void fier_rtn1_encode_groups_out_cuda(
    torch::Tensor keys,
    torch::Tensor packed_codes,
    torch::Tensor lower,
    torch::Tensor upper,
    int64_t history_count,
    int64_t first_group);

torch::Tensor fier_rtn1_scores_cuda(
    torch::Tensor query,
    torch::Tensor packed_codes,
    torch::Tensor lower,
    torch::Tensor upper,
    int64_t history_count);

void fier_rtn1_sampled_compact_out_cuda(
    torch::Tensor query,
    torch::Tensor packed_codes,
    torch::Tensor lower,
    torch::Tensor upper,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor overflow,
    int64_t history_count,
    int64_t sample_count,
    double selected_fraction);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "encode_groups_out",
      &fier_rtn1_encode_groups_out_cuda,
      "Encode sequence-group RTN1 Keys into packed bit planes");
  m.def(
      "scores",
      &fier_rtn1_scores_cuda,
      "Scan a packed sequence-group RTN1 index");
  m.def(
      "sampled_compact_out",
      &fier_rtn1_sampled_compact_out_cuda,
      "Fused sampled threshold and compact RTN1 candidates");
}
"""


CUDA_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr int kHeadDim = 128;
constexpr int kGroupSize = 32;
constexpr int kMaximumSampleCount = 2048;

template <typename key_t>
__global__ void encode_groups_kernel(
    const key_t* __restrict__ keys,
    int32_t* __restrict__ packed_codes,
    at::Half* __restrict__ lower,
    at::Half* __restrict__ upper,
    int batch_count,
    int kv_head_count,
    int source_token_count,
    int capacity_groups,
    int history_count,
    int first_group,
    int encoded_group_count) {
  int row = blockIdx.x;
  int coordinate = threadIdx.x;
  if (coordinate >= kHeadDim) {
    return;
  }
  int local_group = row % encoded_group_count;
  int batch_kv = row / encoded_group_count;
  int group = first_group + local_group;
  int start = group * kGroupSize;
  int valid = min(kGroupSize, history_count - start);

  float low = FLT_MAX;
  float high = -FLT_MAX;
#pragma unroll
  for (int token = 0; token < kGroupSize; ++token) {
    if (token < valid) {
      int64_t source_offset =
          (static_cast<int64_t>(batch_kv) * source_token_count
           + start + token) * kHeadDim + coordinate;
      float value = static_cast<float>(keys[source_offset]);
      low = fminf(low, value);
      high = fmaxf(high, value);
    }
  }
  float threshold = 0.5f * (low + high);
  uint32_t bit_plane = 0u;
#pragma unroll
  for (int token = 0; token < kGroupSize; ++token) {
    if (token < valid) {
      int64_t source_offset =
          (static_cast<int64_t>(batch_kv) * source_token_count
           + start + token) * kHeadDim + coordinate;
      float value = static_cast<float>(keys[source_offset]);
      if (value >= threshold) {
        bit_plane |= (uint32_t{1} << token);
      }
    }
  }
  int64_t destination_offset =
      (static_cast<int64_t>(batch_kv) * capacity_groups + group)
          * kHeadDim
      + coordinate;
  packed_codes[destination_offset] =
      static_cast<int32_t>(bit_plane);
  lower[destination_offset] = static_cast<at::Half>(low);
  upper[destination_offset] = static_cast<at::Half>(high);
}

template <typename query_t>
__global__ void scores_kernel(
    const query_t* __restrict__ query,
    const int32_t* __restrict__ packed_codes,
    const at::Half* __restrict__ lower,
    const at::Half* __restrict__ upper,
    float* __restrict__ scores,
    int batch_count,
    int query_head_count,
    int kv_head_count,
    int capacity_groups,
    int history_count,
    int group_count) {
  int group = blockIdx.x;
  int query_head = blockIdx.y;
  int batch = blockIdx.z;
  int coordinate = threadIdx.x;
  int query_groups = query_head_count / kv_head_count;
  int kv_head = query_head / query_groups;
  int batch_kv = batch * kv_head_count + kv_head;
  int64_t metadata_base =
      (static_cast<int64_t>(batch_kv) * capacity_groups + group)
      * kHeadDim;

  __shared__ float shared_query[kHeadDim];
  __shared__ float shared_lower[kHeadDim];
  __shared__ float shared_upper[kHeadDim];
  if (coordinate < kHeadDim) {
    int64_t query_offset =
        (static_cast<int64_t>(batch) * query_head_count + query_head)
        * kHeadDim + coordinate;
    shared_query[coordinate] =
        static_cast<float>(query[query_offset]);
    shared_lower[coordinate] =
        static_cast<float>(lower[metadata_base + coordinate]);
    shared_upper[coordinate] =
        static_cast<float>(upper[metadata_base + coordinate]);
  }
  __syncthreads();

  if (coordinate >= kGroupSize) {
    return;
  }
  int token = group * kGroupSize + coordinate;
  if (token >= history_count) {
    return;
  }
  float score = 0.0f;
#pragma unroll 4
  for (int dimension = 0; dimension < kHeadDim; ++dimension) {
    uint32_t bit_plane = static_cast<uint32_t>(
        packed_codes[metadata_base + dimension]);
    float reconstructed = ((bit_plane >> coordinate) & 1u)
        ? shared_upper[dimension]
        : shared_lower[dimension];
    score = fmaf(shared_query[dimension], reconstructed, score);
  }
  scores[
      (static_cast<int64_t>(batch) * query_head_count + query_head)
          * history_count
      + token] = score;
}

template <typename query_t>
__device__ __forceinline__ float score_token(
    const query_t* __restrict__ query,
    const int32_t* __restrict__ packed_codes,
    const at::Half* __restrict__ lower,
    const at::Half* __restrict__ upper,
    int row,
    int query_head_count,
    int kv_head_count,
    int capacity_groups,
    int token) {
  int query_groups = query_head_count / kv_head_count;
  int batch = row / query_head_count;
  int query_head = row - batch * query_head_count;
  int kv_head = query_head / query_groups;
  int batch_kv = batch * kv_head_count + kv_head;
  int group = token / kGroupSize;
  int bit = token - group * kGroupSize;
  int64_t query_base =
      static_cast<int64_t>(row) * kHeadDim;
  int64_t metadata_base =
      (static_cast<int64_t>(batch_kv) * capacity_groups + group)
          * kHeadDim;
  float score = 0.0f;
#pragma unroll 4
  for (int dimension = 0; dimension < kHeadDim; ++dimension) {
    uint32_t bit_plane = static_cast<uint32_t>(
        packed_codes[metadata_base + dimension]);
    float reconstructed = ((bit_plane >> bit) & 1u)
        ? static_cast<float>(upper[metadata_base + dimension])
        : static_cast<float>(lower[metadata_base + dimension]);
    score = fmaf(
        static_cast<float>(query[query_base + dimension]),
        reconstructed,
        score);
  }
  return score;
}

template <typename query_t>
__global__ void sampled_threshold_kernel(
    const query_t* __restrict__ query,
    const int32_t* __restrict__ packed_codes,
    const at::Half* __restrict__ lower,
    const at::Half* __restrict__ upper,
    float* __restrict__ thresholds,
    int query_head_count,
    int kv_head_count,
    int capacity_groups,
    int history_count,
    int sample_count,
    int selected_keep) {
  int row = blockIdx.x;
  int thread = threadIdx.x;
  int sort_count = 1;
  while (sort_count < sample_count) {
    sort_count <<= 1;
  }
  extern __shared__ float samples[];
  for (int sample = thread; sample < sort_count; sample += blockDim.x) {
    float score = -INFINITY;
    if (sample < sample_count) {
      int segment = max(1, history_count / sample_count);
      int phase = (row * 131 + 17) % segment;
      int64_t centered =
          (static_cast<int64_t>(2 * sample + 1) * history_count)
          / (2 * sample_count);
      int token = static_cast<int>((centered + phase) % history_count);
      score = score_token(
          query,
          packed_codes,
          lower,
          upper,
          row,
          query_head_count,
          kv_head_count,
          capacity_groups,
          token);
    }
    samples[sample] = score;
  }
  __syncthreads();
  for (int size = 2; size <= sort_count; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int sample = thread; sample < sort_count; sample += blockDim.x) {
        int other = sample ^ stride;
        if (other > sample) {
          bool ascending = (sample & size) == 0;
          float left = samples[sample];
          float right = samples[other];
          if ((left > right) == ascending) {
            samples[sample] = right;
            samples[other] = left;
          }
        }
      }
      __syncthreads();
    }
  }
  if (thread == 0) {
    thresholds[row] = samples[sort_count - selected_keep];
  }
}

template <typename query_t>
__global__ void threshold_compact_kernel(
    const query_t* __restrict__ query,
    const int32_t* __restrict__ packed_codes,
    const at::Half* __restrict__ lower,
    const at::Half* __restrict__ upper,
    const float* __restrict__ thresholds,
    int64_t* __restrict__ candidate_indices,
    int64_t* __restrict__ candidate_counts,
    bool* __restrict__ overflow,
    int query_head_count,
    int kv_head_count,
    int capacity_groups,
    int history_count,
    int candidate_capacity) {
  int group = blockIdx.x;
  int query_head = blockIdx.y;
  int batch = blockIdx.z;
  int coordinate = threadIdx.x;
  int query_groups = query_head_count / kv_head_count;
  int kv_head = query_head / query_groups;
  int batch_kv = batch * kv_head_count + kv_head;
  int row = batch * query_head_count + query_head;
  int64_t metadata_base =
      (static_cast<int64_t>(batch_kv) * capacity_groups + group)
          * kHeadDim;

  __shared__ float shared_query[kHeadDim];
  __shared__ float shared_lower[kHeadDim];
  __shared__ float shared_upper[kHeadDim];
  if (coordinate < kHeadDim) {
    int64_t query_offset =
        static_cast<int64_t>(row) * kHeadDim + coordinate;
    shared_query[coordinate] =
        static_cast<float>(query[query_offset]);
    shared_lower[coordinate] =
        static_cast<float>(lower[metadata_base + coordinate]);
    shared_upper[coordinate] =
        static_cast<float>(upper[metadata_base + coordinate]);
  }
  __syncthreads();
  if (coordinate >= kGroupSize) {
    return;
  }
  int token = group * kGroupSize + coordinate;
  if (token >= history_count) {
    return;
  }
  float score = 0.0f;
#pragma unroll 4
  for (int dimension = 0; dimension < kHeadDim; ++dimension) {
    uint32_t bit_plane = static_cast<uint32_t>(
        packed_codes[metadata_base + dimension]);
    float reconstructed = ((bit_plane >> coordinate) & 1u)
        ? shared_upper[dimension]
        : shared_lower[dimension];
    score = fmaf(shared_query[dimension], reconstructed, score);
  }
  if (score < thresholds[row]) {
    return;
  }
  unsigned long long* count =
      reinterpret_cast<unsigned long long*>(candidate_counts + row);
  unsigned long long slot = atomicAdd(count, 1ULL);
  if (slot < static_cast<unsigned long long>(candidate_capacity)) {
    candidate_indices[
        static_cast<int64_t>(row) * candidate_capacity + slot] = token;
  } else {
    overflow[row] = true;
  }
}

__global__ void finalize_counts_kernel(
    int64_t* __restrict__ candidate_counts,
    bool* __restrict__ overflow,
    int row_count,
    int candidate_capacity) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) {
    return;
  }
  if (candidate_counts[row] > candidate_capacity) {
    candidate_counts[row] = candidate_capacity;
    overflow[row] = true;
  }
}

}  // namespace

void fier_rtn1_encode_groups_out_cuda(
    torch::Tensor keys,
    torch::Tensor packed_codes,
    torch::Tensor lower,
    torch::Tensor upper,
    int64_t history_count,
    int64_t first_group) {
  TORCH_CHECK(keys.is_cuda(), "keys must be CUDA");
  TORCH_CHECK(packed_codes.is_cuda(), "packed codes must be CUDA");
  TORCH_CHECK(lower.is_cuda() && upper.is_cuda(), "metadata must be CUDA");
  TORCH_CHECK(keys.dim() == 4 && keys.size(3) == kHeadDim,
              "keys must have shape [B, Hkv, N, 128]");
  TORCH_CHECK(
      packed_codes.scalar_type() == at::kInt,
      "packed codes must be int32");
  TORCH_CHECK(
      lower.scalar_type() == at::kHalf
          && upper.scalar_type() == at::kHalf,
      "FIER bounds must be float16");
  TORCH_CHECK(
      packed_codes.sizes() == lower.sizes()
          && lower.sizes() == upper.sizes(),
      "FIER packed tensors must have identical shapes");
  TORCH_CHECK(
      packed_codes.dim() == 4
          && packed_codes.size(0) == keys.size(0)
          && packed_codes.size(1) == keys.size(1)
          && packed_codes.size(3) == kHeadDim,
      "packed index shape mismatch");
  TORCH_CHECK(
      history_count > 0 && history_count <= keys.size(2),
      "history_count is outside the source Key tensor");
  int64_t last_group = (history_count + kGroupSize - 1) / kGroupSize;
  TORCH_CHECK(
      first_group >= 0 && first_group < last_group,
      "first_group must identify a non-empty group range");
  TORCH_CHECK(
      last_group <= packed_codes.size(2),
      "packed index capacity is too small");

  int batch_count = static_cast<int>(keys.size(0));
  int kv_head_count = static_cast<int>(keys.size(1));
  int encoded_group_count = static_cast<int>(last_group - first_group);
  int row_count = batch_count * kv_head_count * encoded_group_count;
  c10::cuda::CUDAGuard device_guard(keys.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      keys.scalar_type(),
      "fier_rtn1_encode_groups",
      [&] {
        encode_groups_kernel<scalar_t><<<row_count, kHeadDim, 0, stream>>>(
            keys.data_ptr<scalar_t>(),
            packed_codes.data_ptr<int32_t>(),
            lower.data_ptr<at::Half>(),
            upper.data_ptr<at::Half>(),
            batch_count,
            kv_head_count,
            static_cast<int>(keys.size(2)),
            static_cast<int>(packed_codes.size(2)),
            static_cast<int>(history_count),
            static_cast<int>(first_group),
            encoded_group_count);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor fier_rtn1_scores_cuda(
    torch::Tensor query,
    torch::Tensor packed_codes,
    torch::Tensor lower,
    torch::Tensor upper,
    int64_t history_count) {
  TORCH_CHECK(query.is_cuda(), "query must be CUDA");
  TORCH_CHECK(packed_codes.is_cuda(), "packed codes must be CUDA");
  TORCH_CHECK(lower.is_cuda() && upper.is_cuda(), "metadata must be CUDA");
  TORCH_CHECK(
      query.dim() == 3 && query.size(2) == kHeadDim,
      "query must have shape [B, Hq, 128]");
  TORCH_CHECK(
      packed_codes.scalar_type() == at::kInt,
      "packed codes must be int32");
  TORCH_CHECK(
      lower.scalar_type() == at::kHalf
          && upper.scalar_type() == at::kHalf,
      "FIER bounds must be float16");
  TORCH_CHECK(
      packed_codes.sizes() == lower.sizes()
          && lower.sizes() == upper.sizes(),
      "FIER packed tensors must have identical shapes");
  TORCH_CHECK(
      packed_codes.dim() == 4
          && packed_codes.size(0) == query.size(0)
          && packed_codes.size(3) == kHeadDim,
      "packed index shape mismatch");
  TORCH_CHECK(
      query.size(1) % packed_codes.size(1) == 0,
      "query heads must be divisible by KV heads");
  TORCH_CHECK(history_count > 0, "history_count must be positive");
  int64_t group_count = (history_count + kGroupSize - 1) / kGroupSize;
  TORCH_CHECK(
      group_count <= packed_codes.size(2),
      "packed index does not cover history_count");

  auto scores = torch::empty(
      {query.size(0), query.size(1), history_count},
      query.options().dtype(torch::kFloat));
  c10::cuda::CUDAGuard device_guard(query.device());
  dim3 grid(
      static_cast<unsigned int>(group_count),
      static_cast<unsigned int>(query.size(1)),
      static_cast<unsigned int>(query.size(0)));
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "fier_rtn1_scores",
      [&] {
        scores_kernel<scalar_t><<<grid, kHeadDim, 0, stream>>>(
            query.data_ptr<scalar_t>(),
            packed_codes.data_ptr<int32_t>(),
            lower.data_ptr<at::Half>(),
            upper.data_ptr<at::Half>(),
            scores.data_ptr<float>(),
            static_cast<int>(query.size(0)),
            static_cast<int>(query.size(1)),
            static_cast<int>(packed_codes.size(1)),
            static_cast<int>(packed_codes.size(2)),
            static_cast<int>(history_count),
            static_cast<int>(group_count));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return scores;
}

void fier_rtn1_sampled_compact_out_cuda(
    torch::Tensor query,
    torch::Tensor packed_codes,
    torch::Tensor lower,
    torch::Tensor upper,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor overflow,
    int64_t history_count,
    int64_t sample_count,
    double selected_fraction) {
  TORCH_CHECK(query.is_cuda(), "query must be CUDA");
  TORCH_CHECK(packed_codes.is_cuda(), "packed codes must be CUDA");
  TORCH_CHECK(lower.is_cuda() && upper.is_cuda(), "metadata must be CUDA");
  TORCH_CHECK(
      query.dim() == 3 && query.size(2) == kHeadDim,
      "query must have shape [B, Hq, 128]");
  TORCH_CHECK(
      packed_codes.scalar_type() == at::kInt,
      "packed codes must be int32");
  TORCH_CHECK(
      lower.scalar_type() == at::kHalf
          && upper.scalar_type() == at::kHalf,
      "FIER bounds must be float16");
  TORCH_CHECK(
      candidate_indices.scalar_type() == at::kLong
          && candidate_counts.scalar_type() == at::kLong,
      "candidate tensors must be int64");
  TORCH_CHECK(
      thresholds.scalar_type() == at::kFloat,
      "thresholds must be float32");
  TORCH_CHECK(
      overflow.scalar_type() == at::kBool,
      "overflow must be bool");
  TORCH_CHECK(
      sample_count > 0 && sample_count <= kMaximumSampleCount,
      "invalid sample count");
  TORCH_CHECK(
      selected_fraction > 0.0 && selected_fraction <= 1.0,
      "selected fraction must be in (0, 1]");
  int batch_count = static_cast<int>(query.size(0));
  int query_head_count = static_cast<int>(query.size(1));
  int kv_head_count = static_cast<int>(packed_codes.size(1));
  TORCH_CHECK(
      query_head_count % kv_head_count == 0,
      "query heads must be divisible by KV heads");
  int64_t group_count = (history_count + kGroupSize - 1) / kGroupSize;
  TORCH_CHECK(
      group_count <= packed_codes.size(2),
      "packed index does not cover history_count");
  int row_count = batch_count * query_head_count;
  int candidate_capacity = static_cast<int>(candidate_indices.size(2));
  int selected_keep = std::max(
      1, static_cast<int>(ceil(selected_fraction * sample_count)));

  c10::cuda::CUDAGuard device_guard(query.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(
      candidate_counts.data_ptr<int64_t>(),
      0,
      candidate_counts.numel() * sizeof(int64_t),
      stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      overflow.data_ptr<bool>(),
      0,
      overflow.numel() * sizeof(bool),
      stream));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "fier_rtn1_sampled_compact",
      [&] {
        sampled_threshold_kernel<scalar_t><<<
            row_count,
            256,
            kMaximumSampleCount * sizeof(float),
            stream>>>(
                query.data_ptr<scalar_t>(),
                packed_codes.data_ptr<int32_t>(),
                lower.data_ptr<at::Half>(),
                upper.data_ptr<at::Half>(),
                thresholds.data_ptr<float>(),
                query_head_count,
                kv_head_count,
                static_cast<int>(packed_codes.size(2)),
                static_cast<int>(history_count),
                static_cast<int>(sample_count),
                selected_keep);
        dim3 grid(
            static_cast<unsigned int>(group_count),
            static_cast<unsigned int>(query_head_count),
            static_cast<unsigned int>(batch_count));
        threshold_compact_kernel<scalar_t><<<
            grid, kHeadDim, 0, stream>>>(
                query.data_ptr<scalar_t>(),
                packed_codes.data_ptr<int32_t>(),
                lower.data_ptr<at::Half>(),
                upper.data_ptr<at::Half>(),
                thresholds.data_ptr<float>(),
                candidate_indices.data_ptr<int64_t>(),
                candidate_counts.data_ptr<int64_t>(),
                overflow.data_ptr<bool>(),
                query_head_count,
                kv_head_count,
                static_cast<int>(packed_codes.size(2)),
                static_cast<int>(history_count),
                candidate_capacity);
      });
  finalize_counts_kernel<<<
      (row_count + 255) / 256, 256, 0, stream>>>(
          candidate_counts.data_ptr<int64_t>(),
          overflow.data_ptr<bool>(),
          row_count,
          candidate_capacity);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


@lru_cache(maxsize=1)
def load_extension():
    return load_inline(
        name="fier_rtn1_ext_20260729_v2",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=None,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )


def allocate_packed_index(
    batch_count: int,
    kv_head_count: int,
    capacity_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    if batch_count <= 0 or kv_head_count <= 0 or capacity_tokens <= 0:
        raise ValueError("FIER index dimensions must be positive")
    capacity_groups = math.ceil(capacity_tokens / 32)
    shape = (batch_count, kv_head_count, capacity_groups, 128)
    return {
        "packed_codes": torch.empty(
            shape, dtype=torch.int32, device=device
        ),
        "lower": torch.empty(shape, dtype=torch.float16, device=device),
        "upper": torch.empty(shape, dtype=torch.float16, device=device),
        "capacity": capacity_groups * 32,
        "capacity_groups": capacity_groups,
        "indexed_count": 0,
        "group_size": 32,
        "logical_bits_per_head_token": 256,
    }


def _encode_groups_reference(
    keys: torch.Tensor,
    packed_index: dict[str, Any],
    history_count: int,
    first_group: int,
) -> None:
    group_size = int(packed_index["group_size"])
    last_group = math.ceil(history_count / group_size)
    for group in range(first_group, last_group):
        start = group * group_size
        stop = min(history_count, start + group_size)
        values = keys[..., start:stop, :].float()
        lower = values.amin(dim=-2)
        upper = values.amax(dim=-2)
        threshold = (lower + upper) * 0.5
        high = values >= threshold.unsqueeze(-2)
        shifts = torch.arange(
            stop - start,
            dtype=torch.int64,
            device=keys.device,
        )
        codes = (
            high.to(torch.int64)
            * torch.bitwise_left_shift(
                torch.ones_like(shifts), shifts
            ).view(1, 1, -1, 1)
        ).sum(dim=-2)
        packed_index["packed_codes"][..., group, :].copy_(
            codes.to(torch.int32)
        )
        packed_index["lower"][..., group, :].copy_(
            lower.to(torch.float16)
        )
        packed_index["upper"][..., group, :].copy_(
            upper.to(torch.float16)
        )


def encode_groups_into(
    keys: torch.Tensor,
    packed_index: dict[str, Any],
    history_count: int,
    first_group: int,
) -> None:
    if keys.ndim != 4 or keys.shape[-1] != 128:
        raise ValueError("keys must have shape [B, Hkv, N, 128]")
    if history_count <= 0 or history_count > int(keys.shape[-2]):
        raise ValueError("history_count is outside the source Key tensor")
    if history_count > int(packed_index["capacity"]):
        raise ValueError("FIER packed index capacity is too small")
    last_group = math.ceil(history_count / 32)
    if first_group < 0 or first_group >= last_group:
        raise ValueError("first_group must identify a non-empty range")
    if keys.is_cuda:
        load_extension().encode_groups_out(
            keys.contiguous(),
            packed_index["packed_codes"],
            packed_index["lower"],
            packed_index["upper"],
            history_count,
            first_group,
        )
    else:
        _encode_groups_reference(
            keys, packed_index, history_count, first_group
        )
    packed_index["indexed_count"] = history_count


def update_packed_index(
    keys: torch.Tensor,
    packed_index: dict[str, Any],
    history_count: int,
) -> None:
    indexed_count = int(packed_index["indexed_count"])
    if history_count < indexed_count:
        raise ValueError("FIER index cannot move backwards")
    if history_count == indexed_count:
        return
    first_group = indexed_count // 32
    encode_groups_into(
        keys, packed_index, history_count, first_group
    )


def reconstruct_keys(
    packed_index: dict[str, Any],
    history_count: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if history_count <= 0 or history_count > int(
        packed_index["indexed_count"]
    ):
        raise ValueError("requested history is not indexed")
    token_indices = torch.arange(
        history_count,
        device=packed_index["packed_codes"].device,
        dtype=torch.int64,
    )
    groups = torch.div(token_indices, 32, rounding_mode="floor")
    bits = torch.remainder(token_indices, 32)
    codes = packed_index["packed_codes"][
        ..., groups, :
    ].to(torch.int64)
    code_values = torch.bitwise_and(
        torch.bitwise_right_shift(
            codes,
            bits.view(1, 1, -1, 1),
        ),
        1,
    )
    lower = packed_index["lower"][..., groups, :].float()
    upper = packed_index["upper"][..., groups, :].float()
    return torch.where(code_values.bool(), upper, lower).to(dtype)


def scores(
    query: torch.Tensor,
    packed_index: dict[str, Any],
    history_count: int,
) -> torch.Tensor:
    if query.ndim != 3 or query.shape[-1] != 128:
        raise ValueError("query must have shape [B, Hq, 128]")
    if history_count <= 0 or history_count > int(
        packed_index["indexed_count"]
    ):
        raise ValueError("requested history is not indexed")
    kv_head_count = int(packed_index["packed_codes"].shape[1])
    if query.shape[1] % kv_head_count:
        raise ValueError("query heads must be divisible by KV heads")
    if query.is_cuda:
        return load_extension().scores(
            query.contiguous(),
            packed_index["packed_codes"],
            packed_index["lower"],
            packed_index["upper"],
            history_count,
        )
    reconstructed = reconstruct_keys(
        packed_index, history_count, dtype=torch.float32
    )
    query_groups = query.shape[1] // kv_head_count
    grouped_query = query.float().reshape(
        query.shape[0], kv_head_count, query_groups, 128
    )
    return torch.einsum(
        "bhgd,bhkd->bhgk", grouped_query, reconstructed
    ).reshape(query.shape[0], query.shape[1], history_count)


def sampled_threshold_compact_out(
    query: torch.Tensor,
    packed_index: dict[str, Any],
    candidate_indices: torch.Tensor,
    candidate_counts: torch.Tensor,
    thresholds: torch.Tensor,
    overflow: torch.Tensor,
    history_count: int,
    sample_count: int,
    selected_fraction: float,
) -> tuple[torch.Tensor, ...]:
    if not query.is_cuda:
        raise ValueError("sampled FIER compaction requires CUDA")
    load_extension().sampled_compact_out(
        query.contiguous(),
        packed_index["packed_codes"],
        packed_index["lower"],
        packed_index["upper"],
        candidate_indices,
        candidate_counts,
        thresholds,
        overflow,
        history_count,
        sample_count,
        selected_fraction,
    )
    return candidate_indices, candidate_counts, thresholds, overflow


def allocated_bytes(packed_index: dict[str, Any]) -> int:
    return sum(
        int(packed_index[name].numel())
        * int(packed_index[name].element_size())
        for name in ("packed_codes", "lower", "upper")
    )
