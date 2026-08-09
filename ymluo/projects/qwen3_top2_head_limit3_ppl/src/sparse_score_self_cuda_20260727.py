from __future__ import annotations

from functools import lru_cache

import torch
from torch.utils.cpp_extension import load_inline


CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor sparse_score_self_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor indices,
    torch::Tensor scores,
    torch::Tensor counts,
    double scaling,
    int64_t split_count);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "sparse_score_self_attention_forward",
      &sparse_score_self_attention_forward,
      "Split sparse attention from exact scores with an implicit self token");
}
"""


CUDA_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>

template <typename scalar_t>
__global__ void sparse_softmax_with_self_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const float* __restrict__ scores,
    const int64_t* __restrict__ counts,
    float* __restrict__ weights,
    float* __restrict__ self_weights,
    int query_head_count,
    int kv_head_count,
    int key_count,
    int max_select_count,
    int head_dim,
    int64_t key_stride_b,
    int64_t key_stride_h,
    int64_t key_stride_k,
    int64_t key_stride_d,
    float scaling) {
  extern __shared__ float reduction[];
  __shared__ float self_score;
  __shared__ float reference_max;
  __shared__ float inverse_denominator;
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int batch = row / query_head_count;
  int query_head = row - batch * query_head_count;
  int groups = query_head_count / kv_head_count;
  int kv_head = query_head / groups;
  int row_count = min(
      max(static_cast<int>(counts[row]), 0), max_select_count);
  const scalar_t* query_row = query + row * head_dim;
  const scalar_t* self_key = key
      + batch * key_stride_b
      + kv_head * key_stride_h
      + (key_count - 1) * key_stride_k;
  const float* score_row = scores + row * max_select_count;
  float* weight_row = weights + row * max_select_count;

  float dot = 0.0f;
  for (int dim = tid; dim < head_dim; dim += blockDim.x) {
    dot += static_cast<float>(query_row[dim])
        * static_cast<float>(self_key[dim * key_stride_d]);
  }
  reduction[tid] = dot;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    self_score = reduction[0] * scaling;
  }
  __syncthreads();

  float local_max = tid == 0 ? self_score : -INFINITY;
  for (int selected = tid; selected < row_count; selected += blockDim.x) {
    local_max = fmaxf(local_max, score_row[selected]);
  }
  reduction[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    reference_max = isfinite(reduction[0]) ? reduction[0] : 0.0f;
  }
  __syncthreads();

  float local_sum = tid == 0
      ? expf(self_score - reference_max)
      : 0.0f;
  for (int selected = tid; selected < row_count; selected += blockDim.x) {
    float weight = expf(score_row[selected] - reference_max);
    weight_row[selected] = weight;
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
    inverse_denominator = 1.0f / fmaxf(reduction[0], 1.0e-20f);
    self_weights[row] = expf(self_score - reference_max)
        * inverse_denominator;
  }
  __syncthreads();
  for (int selected = tid; selected < row_count; selected += blockDim.x) {
    weight_row[selected] *= inverse_denominator;
  }
}

template <typename scalar_t>
__global__ void sparse_split_value_kernel(
    const scalar_t* __restrict__ value,
    const int64_t* __restrict__ indices,
    const float* __restrict__ weights,
    const int64_t* __restrict__ counts,
    float* __restrict__ partial_output,
    int query_head_count,
    int kv_head_count,
    int key_count,
    int max_select_count,
    int head_dim,
    int split_count,
    int64_t value_stride_b,
    int64_t value_stride_h,
    int64_t value_stride_k,
    int64_t value_stride_d) {
  int block = blockIdx.x;
  int row = block / split_count;
  int split = block - row * split_count;
  int tid = threadIdx.x;
  int batch = row / query_head_count;
  int query_head = row - batch * query_head_count;
  int groups = query_head_count / kv_head_count;
  int kv_head = query_head / groups;
  int row_count = min(
      max(static_cast<int>(counts[row]), 0), max_select_count);
  int chunk = (row_count + split_count - 1) / split_count;
  int start = min(split * chunk, row_count);
  int end = min(start + chunk, row_count);
  const scalar_t* value_base = value
      + batch * value_stride_b + kv_head * value_stride_h;
  const int64_t* index_row = indices + row * max_select_count;
  const float* weight_row = weights + row * max_select_count;
  float* partial_row = partial_output
      + (row * split_count + split) * head_dim;

  for (int dim = tid; dim < head_dim; dim += blockDim.x) {
    float accumulator = 0.0f;
    for (int selected = start; selected < end; ++selected) {
      int64_t token = index_row[selected];
      if (token >= 0 && token < key_count) {
        accumulator += weight_row[selected]
            * static_cast<float>(
                value_base[
                    token * value_stride_k + dim * value_stride_d]);
      }
    }
    partial_row[dim] = accumulator;
  }
}

template <typename scalar_t>
__global__ void sparse_reduce_with_self_kernel(
    const scalar_t* __restrict__ value,
    const float* __restrict__ partial_output,
    const float* __restrict__ self_weights,
    scalar_t* __restrict__ output,
    int query_head_count,
    int kv_head_count,
    int key_count,
    int head_dim,
    int split_count,
    int64_t value_stride_b,
    int64_t value_stride_h,
    int64_t value_stride_k,
    int64_t value_stride_d) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int batch = row / query_head_count;
  int query_head = row - batch * query_head_count;
  int groups = query_head_count / kv_head_count;
  int kv_head = query_head / groups;
  const float* partial_row =
      partial_output + row * split_count * head_dim;
  const scalar_t* self_value = value
      + batch * value_stride_b
      + kv_head * value_stride_h
      + (key_count - 1) * value_stride_k;
  scalar_t* output_row = output + row * head_dim;
  float self_weight = self_weights[row];

  for (int dim = tid; dim < head_dim; dim += blockDim.x) {
    float accumulator = self_weight
        * static_cast<float>(self_value[dim * value_stride_d]);
    for (int split = 0; split < split_count; ++split) {
      accumulator += partial_row[split * head_dim + dim];
    }
    output_row[dim] = static_cast<scalar_t>(accumulator);
  }
}

torch::Tensor sparse_score_self_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor indices,
    torch::Tensor scores,
    torch::Tensor counts,
    double scaling,
    int64_t split_count) {
  TORCH_CHECK(
      query.is_cuda() && key.is_cuda() && value.is_cuda()
          && indices.is_cuda() && scores.is_cuda() && counts.is_cuda(),
      "all inputs must be CUDA tensors");
  TORCH_CHECK(query.dim() == 3, "query must be [batch, query_heads, dim]");
  TORCH_CHECK(key.dim() == 4, "key must be [batch, kv_heads, tokens, dim]");
  TORCH_CHECK(value.sizes() == key.sizes(), "value must match key");
  TORCH_CHECK(indices.dim() == 3, "indices must be [batch, query_heads, max]");
  TORCH_CHECK(scores.sizes() == indices.sizes(), "scores must match indices");
  TORCH_CHECK(counts.dim() == 2, "counts must be [batch, query_heads]");
  TORCH_CHECK(
      query.scalar_type() == key.scalar_type()
          && query.scalar_type() == value.scalar_type(),
      "query, key, and value dtypes must match");
  TORCH_CHECK(indices.scalar_type() == at::kLong, "indices must be int64");
  TORCH_CHECK(scores.scalar_type() == at::kFloat, "scores must be float32");
  TORCH_CHECK(counts.scalar_type() == at::kLong, "counts must be int64");
  TORCH_CHECK(
      split_count >= 1 && split_count <= 64,
      "split_count must be in [1, 64]");

  auto query_c = query.contiguous();
  auto indices_c = indices.contiguous();
  auto scores_c = scores.contiguous();
  auto counts_c = counts.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int query_head_count = static_cast<int>(query_c.size(1));
  int head_dim = static_cast<int>(query_c.size(2));
  int kv_head_count = static_cast<int>(key.size(1));
  int key_count = static_cast<int>(key.size(2));
  int max_select_count = static_cast<int>(indices_c.size(2));
  int rows = batch_count * query_head_count;
  int splits = static_cast<int>(split_count);
  TORCH_CHECK(
      key.size(0) == batch_count && key.size(3) == head_dim,
      "key shape mismatch");
  TORCH_CHECK(
      indices_c.size(0) == batch_count
          && indices_c.size(1) == query_head_count,
      "indices shape mismatch");
  TORCH_CHECK(
      counts_c.size(0) == batch_count
          && counts_c.size(1) == query_head_count,
      "counts shape mismatch");
  TORCH_CHECK(
      query_head_count % kv_head_count == 0,
      "query heads must be divisible by KV heads");
  TORCH_CHECK(key_count > 0, "key must include the self token");
  TORCH_CHECK(max_select_count > 0, "candidate capacity must be positive");

  auto weights = torch::empty_like(scores_c);
  auto self_weights = torch::empty(
      {rows}, scores_c.options());
  auto partial_output = torch::empty(
      {rows, splits, head_dim}, scores_c.options());
  auto output = torch::empty(
      {batch_count, query_head_count, head_dim}, query_c.options());
  int threads = 128;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "sparse_score_self_attention_forward",
      [&] {
        sparse_softmax_with_self_kernel<scalar_t><<<
            rows,
            threads,
            threads * sizeof(float),
            at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<scalar_t>(),
                key.data_ptr<scalar_t>(),
                scores_c.data_ptr<float>(),
                counts_c.data_ptr<int64_t>(),
                weights.data_ptr<float>(),
                self_weights.data_ptr<float>(),
                query_head_count,
                kv_head_count,
                key_count,
                max_select_count,
                head_dim,
                key.stride(0),
                key.stride(1),
                key.stride(2),
                key.stride(3),
                static_cast<float>(scaling));
        sparse_split_value_kernel<scalar_t><<<
            rows * splits,
            threads,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
                value.data_ptr<scalar_t>(),
                indices_c.data_ptr<int64_t>(),
                weights.data_ptr<float>(),
                counts_c.data_ptr<int64_t>(),
                partial_output.data_ptr<float>(),
                query_head_count,
                kv_head_count,
                key_count,
                max_select_count,
                head_dim,
                splits,
                value.stride(0),
                value.stride(1),
                value.stride(2),
                value.stride(3));
        sparse_reduce_with_self_kernel<scalar_t><<<
            rows,
            threads,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
                value.data_ptr<scalar_t>(),
                partial_output.data_ptr<float>(),
                self_weights.data_ptr<float>(),
                output.data_ptr<scalar_t>(),
                query_head_count,
                kv_head_count,
                key_count,
                head_dim,
                splits,
                value.stride(0),
                value.stride(1),
                value.stride(2),
                value.stride(3));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
"""


@lru_cache(maxsize=1)
def load_extension():
    return load_inline(
        name="sparse_score_self_ext_20260727_v1",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=None,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: torch.Tensor,
    scores: torch.Tensor,
    counts: torch.Tensor,
    scaling: float,
    split_count: int = 16,
) -> torch.Tensor:
    return load_extension().sparse_score_self_attention_forward(
        query,
        key,
        value,
        indices,
        scores.float().contiguous(),
        counts,
        float(scaling),
        int(split_count),
    )
