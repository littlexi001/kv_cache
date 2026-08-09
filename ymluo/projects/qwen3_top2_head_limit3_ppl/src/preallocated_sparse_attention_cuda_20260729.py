from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from torch.utils.cpp_extension import load_inline


CPP_SOURCE = r"""
#include <torch/extension.h>

void preallocated_sparse_attention_split_out_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor indices,
    torch::Tensor counts,
    torch::Tensor partial_output,
    torch::Tensor partial_max,
    torch::Tensor partial_sum,
    torch::Tensor output,
    double scaling,
    int64_t split_count);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "forward_out",
      &preallocated_sparse_attention_split_out_cuda,
      "Preallocated split sparse attention");
}
"""


CUDA_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

template <typename scalar_t>
__global__ void sparse_attention_split_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const int64_t* __restrict__ indices,
    const int64_t* __restrict__ counts,
    float* __restrict__ partial_output,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    int query_head_count,
    int kv_head_count,
    int key_count,
    int max_select_count,
    int head_dim,
    int split_count,
    int64_t key_stride_b,
    int64_t key_stride_h,
    int64_t key_stride_k,
    int64_t key_stride_d,
    int64_t value_stride_b,
    int64_t value_stride_h,
    int64_t value_stride_k,
    int64_t value_stride_d,
    float scaling) {
  extern __shared__ float shared[];
  float* reduction = shared;
  float* weights = shared + blockDim.x;
  int block = blockIdx.x;
  int row = block / split_count;
  int split = block - row * split_count;
  int tid = threadIdx.x;
  int batch = row / query_head_count;
  int head = row - batch * query_head_count;
  int groups = query_head_count / kv_head_count;
  int kv_head = head / groups;
  int candidate_count = min(
      max(static_cast<int>(counts[row]), 0), max_select_count);
  int row_select_count = candidate_count + 1;
  int chunk = (row_select_count + split_count - 1) / split_count;
  int start = min(split * chunk, row_select_count);
  int end = min(start + chunk, row_select_count);
  int local_count = end - start;

  const scalar_t* q_row = query + row * head_dim;
  const scalar_t* k_base =
      key + batch * key_stride_b + kv_head * key_stride_h;
  const scalar_t* v_base =
      value + batch * value_stride_b + kv_head * value_stride_h;
  const int64_t* idx_row = indices + row * max_select_count;
  float* partial_row =
      partial_output + (row * split_count + split) * head_dim;

  float local_max = -INFINITY;
  for (int local = tid; local < local_count; local += blockDim.x) {
    int selected = start + local;
    int64_t token = selected == candidate_count
        ? static_cast<int64_t>(key_count - 1)
        : idx_row[selected];
    float score = -INFINITY;
    if (token >= 0 && token < key_count) {
      const scalar_t* key_row = k_base + token * key_stride_k;
      float accumulator = 0.0f;
      for (int dim = 0; dim < head_dim; ++dim) {
        accumulator += static_cast<float>(q_row[dim])
            * static_cast<float>(key_row[dim * key_stride_d]);
      }
      score = accumulator * scaling;
      local_max = fmaxf(local_max, score);
    }
    weights[local] = score;
  }
  reduction[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
    }
    __syncthreads();
  }
  float block_max = isfinite(reduction[0]) ? reduction[0] : 0.0f;

  float local_sum = 0.0f;
  for (int local = tid; local < local_count; local += blockDim.x) {
    float weight = isfinite(weights[local])
        ? expf(weights[local] - block_max)
        : 0.0f;
    weights[local] = weight;
    local_sum += weight;
  }
  reduction[tid] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    partial_max[row * split_count + split] = block_max;
    partial_sum[row * split_count + split] = reduction[0];
  }

  for (int dim = tid; dim < head_dim; dim += blockDim.x) {
    float accumulator = 0.0f;
    for (int local = 0; local < local_count; ++local) {
      int selected = start + local;
      int64_t token = selected == candidate_count
          ? static_cast<int64_t>(key_count - 1)
          : idx_row[selected];
      if (token >= 0 && token < key_count) {
        accumulator += weights[local]
            * static_cast<float>(
                v_base[
                    token * value_stride_k + dim * value_stride_d]);
      }
    }
    partial_row[dim] = accumulator;
  }
}

template <typename scalar_t>
__global__ void reduce_attention_splits_kernel(
    const float* __restrict__ partial_output,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    scalar_t* __restrict__ output,
    int head_dim,
    int split_count) {
  __shared__ float global_max;
  __shared__ float inverse_denom;
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (tid == 0) {
    float maximum = -INFINITY;
    for (int split = 0; split < split_count; ++split) {
      if (partial_sum[row * split_count + split] > 0.0f) {
        maximum = fmaxf(
            maximum, partial_max[row * split_count + split]);
      }
    }
    maximum = isfinite(maximum) ? maximum : 0.0f;
    float denominator = 0.0f;
    for (int split = 0; split < split_count; ++split) {
      denominator += partial_sum[row * split_count + split]
          * expf(partial_max[row * split_count + split] - maximum);
    }
    global_max = maximum;
    inverse_denom = 1.0f / fmaxf(denominator, 1.0e-20f);
  }
  __syncthreads();

  const float* partial_row =
      partial_output + row * split_count * head_dim;
  scalar_t* output_row = output + row * head_dim;
  for (int dim = tid; dim < head_dim; dim += blockDim.x) {
    float accumulator = 0.0f;
    for (int split = 0; split < split_count; ++split) {
      float scale = expf(
          partial_max[row * split_count + split] - global_max);
      accumulator += scale * partial_row[split * head_dim + dim];
    }
    output_row[dim] = static_cast<scalar_t>(
        accumulator * inverse_denom);
  }
}

void preallocated_sparse_attention_split_out_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor indices,
    torch::Tensor counts,
    torch::Tensor partial_output,
    torch::Tensor partial_max,
    torch::Tensor partial_sum,
    torch::Tensor output,
    double scaling,
    int64_t split_count) {
  TORCH_CHECK(
      query.is_cuda() && key.is_cuda() && value.is_cuda()
          && indices.is_cuda() && counts.is_cuda()
          && partial_output.is_cuda() && partial_max.is_cuda()
          && partial_sum.is_cuda() && output.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(
      query.device() == key.device() && query.device() == value.device()
          && query.device() == indices.device()
          && query.device() == counts.device()
          && query.device() == partial_output.device()
          && query.device() == output.device(),
      "input device mismatch");
  TORCH_CHECK(query.dim() == 3, "query must be [batch, heads, dim]");
  TORCH_CHECK(key.dim() == 4, "key must be [batch, heads, key, dim]");
  TORCH_CHECK(value.sizes() == key.sizes(), "value must match key");
  TORCH_CHECK(indices.dim() == 3, "indices must be rank three");
  TORCH_CHECK(counts.dim() == 2, "counts must be rank two");
  TORCH_CHECK(indices.scalar_type() == at::kLong, "indices must be int64");
  TORCH_CHECK(counts.scalar_type() == at::kLong, "counts must be int64");
  TORCH_CHECK(partial_output.scalar_type() == at::kFloat,
              "partial output must be float32");
  TORCH_CHECK(partial_max.scalar_type() == at::kFloat
              && partial_sum.scalar_type() == at::kFloat,
              "partial statistics must be float32");
  TORCH_CHECK(
      split_count >= 2 && split_count <= 16,
      "split count must be in [2, 16]");

  auto query_c = query.contiguous();
  auto indices_c = indices.contiguous();
  auto counts_c = counts.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int query_head_count = static_cast<int>(query_c.size(1));
  int kv_head_count = static_cast<int>(key.size(1));
  int head_dim = static_cast<int>(query_c.size(2));
  int key_count = static_cast<int>(key.size(2));
  int max_select_count = static_cast<int>(indices_c.size(2));
  int splits = static_cast<int>(split_count);
  int rows = batch_count * query_head_count;
  TORCH_CHECK(
      query_head_count % kv_head_count == 0,
      "query heads must be divisible by KV heads");
  TORCH_CHECK(
      partial_output.dim() == 3
          && partial_output.size(0) == rows
          && partial_output.size(1) == splits
          && partial_output.size(2) == head_dim,
      "partial output shape mismatch");
  TORCH_CHECK(
      partial_max.dim() == 2
          && partial_max.size(0) == rows
          && partial_max.size(1) == splits
          && partial_sum.sizes() == partial_max.sizes(),
      "partial statistic shape mismatch");
  TORCH_CHECK(
      output.dim() == 3
          && output.size(0) == batch_count
          && output.size(1) == query_head_count
          && output.size(2) == head_dim,
      "output shape mismatch");
  int threads = 128;
  int max_local_count = (
      max_select_count + 1 + splits - 1) / splits;
  size_t shared_bytes = static_cast<size_t>(
      threads + max_local_count) * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "preallocated_sparse_attention_split_out",
      [&] {
        sparse_attention_split_kernel<scalar_t><<<
            rows * splits, threads, shared_bytes,
            at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<scalar_t>(),
                key.data_ptr<scalar_t>(),
                value.data_ptr<scalar_t>(),
                indices_c.data_ptr<int64_t>(),
                counts_c.data_ptr<int64_t>(),
                partial_output.data_ptr<float>(),
                partial_max.data_ptr<float>(),
                partial_sum.data_ptr<float>(),
                query_head_count,
                kv_head_count,
                key_count,
                max_select_count,
                head_dim,
                splits,
                key.stride(0),
                key.stride(1),
                key.stride(2),
                key.stride(3),
                value.stride(0),
                value.stride(1),
                value.stride(2),
                value.stride(3),
                static_cast<float>(scaling));
        reduce_attention_splits_kernel<scalar_t><<<
            rows, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                partial_output.data_ptr<float>(),
                partial_max.data_ptr<float>(),
                partial_sum.data_ptr<float>(),
                output.data_ptr<scalar_t>(),
                head_dim,
                splits);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


@lru_cache(maxsize=1)
def load_extension() -> Any:
    return load_inline(
        name="qksieve_preallocated_sparse_attention_20260729_v1",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=None,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )


def allocate_workspace(
    query: torch.Tensor,
    split_count: int,
) -> dict[str, torch.Tensor]:
    batch_count, query_head_count, head_dim = query.shape
    rows = batch_count * query_head_count
    return {
        "partial_output": torch.empty(
            rows,
            split_count,
            head_dim,
            dtype=torch.float32,
            device=query.device,
        ),
        "partial_max": torch.empty(
            rows,
            split_count,
            dtype=torch.float32,
            device=query.device,
        ),
        "partial_sum": torch.empty(
            rows,
            split_count,
            dtype=torch.float32,
            device=query.device,
        ),
        "output": torch.empty_like(query),
    }


def forward_out(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    workspace: dict[str, torch.Tensor],
    scaling: float,
    split_count: int,
) -> torch.Tensor:
    load_extension().forward_out(
        query,
        key,
        value,
        indices,
        counts,
        workspace["partial_output"],
        workspace["partial_max"],
        workspace["partial_sum"],
        workspace["output"],
        scaling,
        split_count,
    )
    return workspace["output"]
