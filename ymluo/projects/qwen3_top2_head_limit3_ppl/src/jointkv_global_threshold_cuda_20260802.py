"""Per-query sampled-threshold selection for the packed JointKV index.

The older JointKV CUDA path keeps a fixed quota from every 32-token warp and
shares one candidate set across GQA query heads.  Real text violates both
assumptions: useful tokens cluster in a few spans and different query heads can
need different tokens.  This extension scans the same compact index but emits
one global, variable-length candidate list per query head.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from torch.utils.cpp_extension import load_inline


CPP_SOURCE = r"""
#include <torch/extension.h>
#include <vector>

void jointkv_sampled_threshold_compact_cuda(
    torch::Tensor packed_query,
    torch::Tensor query_lut,
    torch::Tensor base_codes,
    torch::Tensor residual_codes,
    torch::Tensor joint_ids,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor overflow,
    int64_t sample_count,
    double selected_fraction,
    double overfetch,
    int64_t joint_offset);

void jointkv_append_suffix_cuda(
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    int64_t suffix_start,
    int64_t suffix_stop);

void jointkv_project_query_lut_out_cuda(
    torch::Tensor query,
    torch::Tensor query_matrix,
    torch::Tensor packed_query,
    torch::Tensor query_lut,
    int64_t base_offset,
    int64_t residual_offset,
    int64_t base_chunks,
    int64_t residual_chunks);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "sampled_threshold_compact",
      &jointkv_sampled_threshold_compact_cuda,
      "JointKV per-query sampled-threshold compact (CUDA)");
  m.def(
      "append_suffix",
      &jointkv_append_suffix_cuda,
      "Append a contiguous exact suffix to every query row (CUDA)");
  m.def(
      "project_query_lut_out",
      &jointkv_project_query_lut_out_cuda,
      "Fused head-specific query projection and byte LUT build (CUDA)");
}
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

namespace {

constexpr int kMaxSamples = 512;

template <typename scalar_t>
__global__ void project_query_lut_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ query_matrix,
    scalar_t* __restrict__ packed_query,
    float* __restrict__ query_lut,
    int query_head_count,
    int kv_head_count,
    int query_groups,
    int head_dim,
    int query_width,
    int base_offset,
    int residual_offset,
    int base_chunks,
    int residual_chunks) {
  int row = blockIdx.x;
  int query_head = row % query_head_count;
  int kv_head = query_head / query_groups;
  const scalar_t* query_row = query + static_cast<int64_t>(row) * head_dim;
  const scalar_t* matrix = query_matrix
      + static_cast<int64_t>(kv_head) * head_dim * query_width;
  scalar_t* packed_row = packed_query
      + static_cast<int64_t>(row) * query_width;
  for (int output = threadIdx.x; output < query_width; output += blockDim.x) {
    float accumulator = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
      accumulator += static_cast<float>(query_row[dim])
          * static_cast<float>(matrix[dim * query_width + output]);
    }
    packed_row[output] = static_cast<scalar_t>(accumulator);
  }
  __syncthreads();
  int total_chunks = base_chunks + residual_chunks;
  int lut_elements = total_chunks * 256;
  float* lut_row = query_lut
      + static_cast<int64_t>(row) * lut_elements;
  for (int index = threadIdx.x; index < lut_elements; index += blockDim.x) {
    int pattern = index & 255;
    int chunk = index >> 8;
    int source = chunk < base_chunks
        ? base_offset + chunk * 8
        : residual_offset + (chunk - base_chunks) * 8;
    float score = 0.0f;
#pragma unroll
    for (int bit = 0; bit < 8; ++bit) {
      float probe = static_cast<float>(packed_row[source + bit]);
      score += ((pattern >> bit) & 1) ? probe : -probe;
    }
    lut_row[index] = score;
  }
}

template <typename scalar_t>
__device__ __forceinline__ float score_token(
    const scalar_t* __restrict__ packed_query,
    const float* __restrict__ query_lut,
    const int64_t* __restrict__ base_codes,
    const int64_t* __restrict__ residual_codes,
    const uint8_t* __restrict__ joint_ids,
    int row,
    int batch_kv,
    int token,
    int token_count,
    int query_width,
    int total_chunks,
    int base_chunks,
    int joint_offset) {
  int64_t token_offset = static_cast<int64_t>(batch_kv) * token_count + token;
  unsigned long long base = static_cast<unsigned long long>(base_codes[token_offset]);
  unsigned long long residual =
      static_cast<unsigned long long>(residual_codes[token_offset]);
  int lut_base = row * total_chunks * 256;
  float score = 0.0f;
#pragma unroll
  for (int chunk = 0; chunk < 8; ++chunk) {
    if (chunk < base_chunks) {
      int pattern = static_cast<int>((base >> (8 * chunk)) & 255ull);
      score += query_lut[lut_base + chunk * 256 + pattern];
    }
  }
  int residual_chunks = total_chunks - base_chunks;
#pragma unroll
  for (int chunk = 0; chunk < 8; ++chunk) {
    if (chunk < residual_chunks) {
      int pattern = static_cast<int>((residual >> (8 * chunk)) & 255ull);
      score += query_lut[
          lut_base + (base_chunks + chunk) * 256 + pattern];
    }
  }
  int joint = static_cast<int>(joint_ids[token_offset] & 63u);
  score += static_cast<float>(
      packed_query[static_cast<int64_t>(row) * query_width + joint_offset + joint]);
  return score;
}

template <typename scalar_t>
__global__ void sample_threshold_kernel(
    const scalar_t* __restrict__ packed_query,
    const float* __restrict__ query_lut,
    const int64_t* __restrict__ base_codes,
    const int64_t* __restrict__ residual_codes,
    const uint8_t* __restrict__ joint_ids,
    float* __restrict__ thresholds,
    int kv_head_count,
    int query_groups,
    int token_count,
    int query_width,
    int total_chunks,
    int base_chunks,
    int sample_count,
    int selected_samples,
    int joint_offset) {
  int row = blockIdx.x;
  int thread = threadIdx.x;
  int query_head_count = kv_head_count * query_groups;
  int batch = row / query_head_count;
  int query_head = row - batch * query_head_count;
  int kv_head = query_head / query_groups;
  int batch_kv = batch * kv_head_count + kv_head;
  int sort_count = 1;
  while (sort_count < sample_count) {
    sort_count <<= 1;
  }
  extern __shared__ float samples[];
  for (int sample = thread; sample < sort_count; sample += blockDim.x) {
    float score = -INFINITY;
    if (sample < sample_count) {
      int segment = max(1, token_count / sample_count);
      int phase = (row * 131 + 17) % segment;
      int64_t centered =
          (static_cast<int64_t>(2 * sample + 1) * token_count)
          / (2 * sample_count);
      int token = static_cast<int>((centered + phase) % token_count);
      score = score_token(
          packed_query, query_lut, base_codes, residual_codes, joint_ids,
          row, batch_kv, token, token_count, query_width, total_chunks,
          base_chunks, joint_offset);
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
    thresholds[row] = samples[sort_count - selected_samples];
  }
}

template <typename scalar_t>
__global__ void compact_kernel(
    const scalar_t* __restrict__ packed_query,
    const float* __restrict__ query_lut,
    const int64_t* __restrict__ base_codes,
    const int64_t* __restrict__ residual_codes,
    const uint8_t* __restrict__ joint_ids,
    const float* __restrict__ thresholds,
    int64_t* __restrict__ candidate_indices,
    int64_t* __restrict__ candidate_counts,
    bool* __restrict__ overflow,
    int kv_head_count,
    int query_groups,
    int token_count,
    int query_width,
    int total_chunks,
    int base_chunks,
    int capacity,
    int joint_offset) {
  int token = blockIdx.y * blockDim.x + threadIdx.x;
  if (token >= token_count) {
    return;
  }
  int batch_kv = blockIdx.x;
  int batch = batch_kv / kv_head_count;
  int kv_head = batch_kv - batch * kv_head_count;
  int query_head_count = kv_head_count * query_groups;
  for (int group = 0; group < query_groups; ++group) {
    int row = batch * query_head_count + kv_head * query_groups + group;
    float score = score_token(
        packed_query, query_lut, base_codes, residual_codes, joint_ids,
        row, batch_kv, token, token_count, query_width, total_chunks,
        base_chunks, joint_offset);
    if (score < thresholds[row]) {
      continue;
    }
    unsigned long long* count = reinterpret_cast<unsigned long long*>(
        candidate_counts + row);
    unsigned long long slot = atomicAdd(count, 1ULL);
    if (slot < static_cast<unsigned long long>(capacity)) {
      candidate_indices[static_cast<int64_t>(row) * capacity + slot] = token;
    } else {
      overflow[row] = true;
    }
  }
}

__global__ void finalize_counts_kernel(
    int64_t* __restrict__ counts,
    const bool* __restrict__ overflow,
    int rows,
    int capacity) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows && (overflow[row] || counts[row] > capacity)) {
    counts[row] = capacity;
  }
}

__global__ void append_suffix_kernel(
    int64_t* __restrict__ indices,
    int64_t* __restrict__ counts,
    int rows,
    int capacity,
    int64_t suffix_start,
    int suffix_count) {
  int row = blockIdx.x;
  int64_t base = min(max(counts[row], static_cast<int64_t>(0)),
                     static_cast<int64_t>(capacity));
  int available = max(0, capacity - static_cast<int>(base));
  int append_count = min(suffix_count, available);
  for (int offset = threadIdx.x; offset < append_count; offset += blockDim.x) {
    indices[static_cast<int64_t>(row) * capacity + base + offset] =
        suffix_start + offset;
  }
  if (threadIdx.x == 0) {
    counts[row] = base + append_count;
  }
}

}  // namespace

void jointkv_project_query_lut_out_cuda(
    torch::Tensor query,
    torch::Tensor query_matrix,
    torch::Tensor packed_query,
    torch::Tensor query_lut,
    int64_t base_offset,
    int64_t residual_offset,
    int64_t base_chunks,
    int64_t residual_chunks) {
  TORCH_CHECK(query.is_cuda() && query_matrix.is_cuda(),
              "query projection inputs must be CUDA tensors");
  TORCH_CHECK(query.dim() == 3, "query must be [B,QH,D]");
  TORCH_CHECK(query_matrix.dim() == 3, "query_matrix must be [KVH,D,W]");
  TORCH_CHECK(packed_query.dim() == 4,
              "packed_query output must be [B,KVH,G,W]");
  TORCH_CHECK(query_lut.dim() == 5 && query_lut.scalar_type() == at::kFloat,
              "query_lut output must be float32 [B,KVH,G,C,256]");
  TORCH_CHECK(query.scalar_type() == query_matrix.scalar_type()
                  && query.scalar_type() == packed_query.scalar_type(),
              "query, matrix, and packed output dtypes must match");
  int batches = static_cast<int>(query.size(0));
  int query_heads = static_cast<int>(query.size(1));
  int head_dim = static_cast<int>(query.size(2));
  int kv_heads = static_cast<int>(query_matrix.size(0));
  int query_width = static_cast<int>(query_matrix.size(2));
  TORCH_CHECK(query_matrix.size(1) == head_dim,
              "query matrix head dimension mismatch");
  TORCH_CHECK(query_heads % kv_heads == 0,
              "query heads must be divisible by KV heads");
  int groups = query_heads / kv_heads;
  int total_chunks = static_cast<int>(base_chunks + residual_chunks);
  TORCH_CHECK(packed_query.size(0) == batches
                  && packed_query.size(1) == kv_heads
                  && packed_query.size(2) == groups
                  && packed_query.size(3) == query_width,
              "packed query output shape mismatch");
  TORCH_CHECK(query_lut.size(0) == batches
                  && query_lut.size(1) == kv_heads
                  && query_lut.size(2) == groups
                  && query_lut.size(3) == total_chunks
                  && query_lut.size(4) == 256,
              "query LUT output shape mismatch");
  TORCH_CHECK(base_offset >= 0 && residual_offset >= 0
                  && base_offset + base_chunks * 8 <= query_width
                  && residual_offset + residual_chunks * 8 <= query_width,
              "query code ranges leave query width");
  c10::cuda::CUDAGuard guard(query.device());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "jointkv_project_query_lut_out",
      [&] {
        project_query_lut_kernel<scalar_t><<<
            batches * query_heads,
            256,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
                query.data_ptr<scalar_t>(),
                query_matrix.data_ptr<scalar_t>(),
                packed_query.data_ptr<scalar_t>(),
                query_lut.data_ptr<float>(),
                query_heads,
                kv_heads,
                groups,
                head_dim,
                query_width,
                static_cast<int>(base_offset),
                static_cast<int>(residual_offset),
                static_cast<int>(base_chunks),
                static_cast<int>(residual_chunks));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void jointkv_sampled_threshold_compact_cuda(
    torch::Tensor packed_query,
    torch::Tensor query_lut,
    torch::Tensor base_codes,
    torch::Tensor residual_codes,
    torch::Tensor joint_ids,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor overflow,
    int64_t sample_count,
    double selected_fraction,
    double overfetch,
    int64_t joint_offset) {
  TORCH_CHECK(packed_query.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(packed_query.dim() == 4, "packed_query must be [B,KVH,G,W]");
  TORCH_CHECK(query_lut.dim() == 5 && query_lut.scalar_type() == at::kFloat,
              "query_lut must be float32 [B,KVH,G,C,256]");
  TORCH_CHECK(base_codes.dim() == 3 && base_codes.scalar_type() == at::kLong,
              "base_codes must be int64 [B,KVH,N]");
  TORCH_CHECK(residual_codes.sizes() == base_codes.sizes()
                  && residual_codes.scalar_type() == at::kLong,
              "residual_codes must match base_codes");
  TORCH_CHECK(joint_ids.sizes() == base_codes.sizes()
                  && joint_ids.scalar_type() == at::kByte,
              "joint_ids must be uint8 and match base_codes");
  TORCH_CHECK(candidate_indices.dim() == 3
                  && candidate_indices.scalar_type() == at::kLong,
              "candidate_indices must be int64 [B,QH,C]");
  TORCH_CHECK(candidate_counts.dim() == 2
                  && candidate_counts.scalar_type() == at::kLong,
              "candidate_counts must be int64 [B,QH]");
  TORCH_CHECK(thresholds.sizes() == candidate_counts.sizes()
                  && thresholds.scalar_type() == at::kFloat,
              "thresholds must be float32 [B,QH]");
  TORCH_CHECK(overflow.sizes() == candidate_counts.sizes()
                  && overflow.scalar_type() == at::kBool,
              "overflow must be bool [B,QH]");
  TORCH_CHECK(sample_count > 0 && sample_count <= kMaxSamples,
              "sample_count must lie in [1,512]");
  TORCH_CHECK(selected_fraction > 0.0 && selected_fraction <= 1.0,
              "selected_fraction must lie in (0,1]");
  TORCH_CHECK(overfetch >= 1.0, "overfetch must be at least one");

  auto query_c = packed_query.contiguous();
  auto lut_c = query_lut.contiguous();
  auto base_c = base_codes.contiguous();
  auto residual_c = residual_codes.contiguous();
  auto ids_c = joint_ids.contiguous();
  int batches = static_cast<int>(query_c.size(0));
  int kv_heads = static_cast<int>(query_c.size(1));
  int groups = static_cast<int>(query_c.size(2));
  int query_width = static_cast<int>(query_c.size(3));
  int tokens = static_cast<int>(base_c.size(2));
  int total_chunks = static_cast<int>(lut_c.size(3));
  int base_chunks = 8;
  int query_heads = kv_heads * groups;
  int rows = batches * query_heads;
  int capacity = static_cast<int>(candidate_indices.size(2));
  TORCH_CHECK(total_chunks >= base_chunks,
              "query LUT must contain the 64-bit base code");
  TORCH_CHECK(candidate_indices.size(0) == batches
                  && candidate_indices.size(1) == query_heads,
              "candidate workspace shape mismatch");
  TORCH_CHECK(candidate_counts.size(0) == batches
                  && candidate_counts.size(1) == query_heads,
              "count workspace shape mismatch");
  TORCH_CHECK(joint_offset >= 0 && joint_offset + 64 <= query_width,
              "joint lookup leaves packed query");

  int selected_samples = std::max(
      1, std::min(static_cast<int>(sample_count),
                  static_cast<int>(std::ceil(
                      selected_fraction * overfetch * sample_count))));
  c10::cuda::CUDAGuard guard(query_c.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(
      candidate_counts.data_ptr<int64_t>(), 0,
      candidate_counts.numel() * sizeof(int64_t), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      overflow.data_ptr<bool>(), 0,
      overflow.numel() * sizeof(bool), stream));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "jointkv_sampled_threshold_compact",
      [&] {
        sample_threshold_kernel<scalar_t><<<
            rows, 256, kMaxSamples * sizeof(float), stream>>>(
                query_c.data_ptr<scalar_t>(),
                lut_c.data_ptr<float>(),
                base_c.data_ptr<int64_t>(),
                residual_c.data_ptr<int64_t>(),
                ids_c.data_ptr<uint8_t>(),
                thresholds.data_ptr<float>(),
                kv_heads,
                groups,
                tokens,
                query_width,
                total_chunks,
                base_chunks,
                static_cast<int>(sample_count),
                selected_samples,
                static_cast<int>(joint_offset));
        dim3 blocks(batches * kv_heads, (tokens + 255) / 256);
        compact_kernel<scalar_t><<<blocks, 256, 0, stream>>>(
            query_c.data_ptr<scalar_t>(),
            lut_c.data_ptr<float>(),
            base_c.data_ptr<int64_t>(),
            residual_c.data_ptr<int64_t>(),
            ids_c.data_ptr<uint8_t>(),
            thresholds.data_ptr<float>(),
            candidate_indices.data_ptr<int64_t>(),
            candidate_counts.data_ptr<int64_t>(),
            overflow.data_ptr<bool>(),
            kv_heads,
            groups,
            tokens,
            query_width,
            total_chunks,
            base_chunks,
            capacity,
            static_cast<int>(joint_offset));
      });
  finalize_counts_kernel<<<(rows + 255) / 256, 256, 0, stream>>>(
      candidate_counts.data_ptr<int64_t>(),
      overflow.data_ptr<bool>(),
      rows,
      capacity);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void jointkv_append_suffix_cuda(
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    int64_t suffix_start,
    int64_t suffix_stop) {
  TORCH_CHECK(candidate_indices.is_cuda() && candidate_counts.is_cuda(),
              "suffix append inputs must be CUDA tensors");
  TORCH_CHECK(candidate_indices.dim() == 3
                  && candidate_indices.scalar_type() == at::kLong,
              "candidate_indices must be int64 [B,QH,C]");
  TORCH_CHECK(candidate_counts.dim() == 2
                  && candidate_counts.scalar_type() == at::kLong,
              "candidate_counts must be int64 [B,QH]");
  TORCH_CHECK(candidate_indices.size(0) == candidate_counts.size(0)
                  && candidate_indices.size(1) == candidate_counts.size(1),
              "suffix append workspace shapes do not align");
  TORCH_CHECK(suffix_stop >= suffix_start,
              "suffix stop must not precede suffix start");
  int rows = static_cast<int>(candidate_counts.numel());
  int capacity = static_cast<int>(candidate_indices.size(2));
  int suffix_count = static_cast<int>(suffix_stop - suffix_start);
  if (suffix_count == 0) {
    return;
  }
  c10::cuda::CUDAGuard guard(candidate_indices.device());
  append_suffix_kernel<<<
      rows, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
          candidate_indices.data_ptr<int64_t>(),
          candidate_counts.data_ptr<int64_t>(),
          rows,
          capacity,
          suffix_start,
          suffix_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


@lru_cache(maxsize=1)
def load_extension() -> Any:
    return load_inline(
        name="jointkv_global_threshold_cuda_20260802_v4",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=None,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )


def allocate_workspace(
    packed_query: torch.Tensor,
    capacity: int,
) -> dict[str, torch.Tensor]:
    if capacity <= 0:
        raise ValueError("candidate capacity must be positive")
    batches, kv_heads, groups, _ = packed_query.shape
    query_heads = kv_heads * groups
    return {
        "indices": torch.empty(
            batches,
            query_heads,
            capacity,
            dtype=torch.int64,
            device=packed_query.device,
        ),
        "counts": torch.empty(
            batches,
            query_heads,
            dtype=torch.int64,
            device=packed_query.device,
        ),
        "thresholds": torch.empty(
            batches,
            query_heads,
            dtype=torch.float32,
            device=packed_query.device,
        ),
        "overflow": torch.empty(
            batches,
            query_heads,
            dtype=torch.bool,
            device=packed_query.device,
        ),
    }


def sampled_threshold_compact_out(
    packed_query: torch.Tensor,
    query_lut: torch.Tensor,
    base_codes: torch.Tensor,
    residual_codes: torch.Tensor,
    joint_ids: torch.Tensor,
    workspace: dict[str, torch.Tensor],
    *,
    sample_count: int,
    selected_fraction: float,
    overfetch: float = 1.0,
    joint_offset: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    load_extension().sampled_threshold_compact(
        packed_query,
        query_lut,
        base_codes,
        residual_codes,
        joint_ids,
        workspace["indices"],
        workspace["counts"],
        workspace["thresholds"],
        workspace["overflow"],
        sample_count,
        selected_fraction,
        overfetch,
        joint_offset,
    )
    return (
        workspace["indices"],
        workspace["counts"],
        workspace["thresholds"],
        workspace["overflow"],
    )


def append_contiguous_suffix(
    candidate_indices: torch.Tensor,
    candidate_counts: torch.Tensor,
    suffix_start: int,
    suffix_stop: int,
) -> None:
    load_extension().append_suffix(
        candidate_indices,
        candidate_counts,
        suffix_start,
        suffix_stop,
    )


def allocate_query_workspace(
    query: torch.Tensor,
    query_matrix: torch.Tensor,
    *,
    base_chunks: int = 8,
    residual_chunks: int = 6,
) -> dict[str, torch.Tensor]:
    batches, query_heads, _ = query.shape
    kv_heads, _, query_width = query_matrix.shape
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    groups = query_heads // kv_heads
    return {
        "packed_query": torch.empty(
            batches,
            kv_heads,
            groups,
            query_width,
            dtype=query.dtype,
            device=query.device,
        ),
        "query_lut": torch.empty(
            batches,
            kv_heads,
            groups,
            base_chunks + residual_chunks,
            256,
            dtype=torch.float32,
            device=query.device,
        ),
    }


def project_query_lut_out(
    query: torch.Tensor,
    query_matrix: torch.Tensor,
    workspace: dict[str, torch.Tensor],
    *,
    base_offset: int = 0,
    residual_offset: int = 64,
    base_chunks: int = 8,
    residual_chunks: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    load_extension().project_query_lut_out(
        query,
        query_matrix,
        workspace["packed_query"],
        workspace["query_lut"],
        base_offset,
        residual_offset,
        base_chunks,
        residual_chunks,
    )
    return workspace["packed_query"], workspace["query_lut"]
