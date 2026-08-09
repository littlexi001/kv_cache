from __future__ import annotations

import os
from functools import lru_cache

import torch
from torch.utils.cpp_extension import load_inline


CPP_SOURCE = r"""
#include <torch/extension.h>

void qksieve_valuesketch_append_int4_out(
    torch::Tensor value_history,
    torch::Tensor value_mean,
    torch::Tensor value_basis,
    torch::Tensor value_minimum,
    torch::Tensor value_scale,
    torch::Tensor packed_codes,
    int64_t input_start,
    int64_t input_stop,
    int64_t value_block_size);

torch::Tensor qksieve_valuesketch_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor tail_denominator,
    torch::Tensor tail_coefficients,
    torch::Tensor value_mean,
    torch::Tensor value_basis,
    double scaling,
    double tail_alpha);

void qksieve_valuesketch_attention_out(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor tail_denominator,
    torch::Tensor tail_coefficients,
    torch::Tensor value_mean,
    torch::Tensor value_basis,
    torch::Tensor output,
    torch::Tensor partial_output,
    torch::Tensor partial_maximum,
    torch::Tensor partial_sum,
    double scaling,
    double tail_alpha);

#if 0  // CUDA implementation is emitted at the end of CUDA_SOURCE.
void qksieve_valuesketch_attention_active_out(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor tail_denominator,
    torch::Tensor tail_coefficients,
    torch::Tensor value_mean,
    torch::Tensor value_basis,
    torch::Tensor active_key_count,
    torch::Tensor output,
    torch::Tensor partial_output,
    torch::Tensor partial_maximum,
    torch::Tensor partial_sum,
    double scaling,
    double tail_alpha);

void qksieve_append_suffix_candidates_out(
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor active_key_count,
    int64_t prefix_count,
    int64_t physical_key_count);

void qksieve_valuesketch_attention_active_out(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor tail_denominator,
    torch::Tensor tail_coefficients,
    torch::Tensor value_mean,
    torch::Tensor value_basis,
    torch::Tensor active_key_count,
    torch::Tensor output,
    torch::Tensor partial_output,
    torch::Tensor partial_maximum,
    torch::Tensor partial_sum,
    double scaling,
    double tail_alpha) {
  TORCH_CHECK(
      query.is_cuda() && key.is_cuda() && value.is_cuda()
          && candidate_indices.is_cuda() && candidate_counts.is_cuda()
          && thresholds.is_cuda() && tail_denominator.is_cuda()
          && tail_coefficients.is_cuda() && value_mean.is_cuda()
          && value_basis.is_cuda() && active_key_count.is_cuda()
          && output.is_cuda() && partial_output.is_cuda()
          && partial_maximum.is_cuda() && partial_sum.is_cuda(),
      "active ValueSketch inputs and workspaces must be CUDA tensors");
  TORCH_CHECK(query.dim() == 3 && key.dim() == 4 && value.dim() == 4,
              "active ValueSketch query/key/value ranks are invalid");
  TORCH_CHECK(key.sizes() == value.sizes(),
              "active Key and Value tensors must have identical shapes");
  TORCH_CHECK(query.scalar_type() == key.scalar_type()
                  && query.scalar_type() == value.scalar_type()
                  && query.scalar_type() == value_mean.scalar_type()
                  && query.scalar_type() == value_basis.scalar_type()
                  && query.scalar_type() == output.scalar_type(),
              "active floating-point tensors and output must share a dtype");
  TORCH_CHECK(candidate_indices.scalar_type() == at::kLong
                  && candidate_counts.scalar_type() == at::kLong
                  && active_key_count.scalar_type() == at::kInt,
              "active candidate tensors must be int64 and length must be int32");
  TORCH_CHECK(thresholds.scalar_type() == at::kFloat
                  && tail_denominator.scalar_type() == at::kFloat
                  && tail_coefficients.scalar_type() == at::kFloat
                  && partial_output.scalar_type() == at::kFloat
                  && partial_maximum.scalar_type() == at::kFloat
                  && partial_sum.scalar_type() == at::kFloat,
              "active tail statistics and partial workspaces must be float32");
  TORCH_CHECK(query.is_contiguous() && candidate_indices.is_contiguous()
                  && candidate_counts.is_contiguous() && thresholds.is_contiguous()
                  && tail_denominator.is_contiguous()
                  && tail_coefficients.is_contiguous() && value_mean.is_contiguous()
                  && value_basis.is_contiguous() && active_key_count.is_contiguous()
                  && output.is_contiguous() && partial_output.is_contiguous()
                  && partial_maximum.is_contiguous() && partial_sum.is_contiguous(),
              "active ValueSketch tensors must be contiguous");
  TORCH_CHECK(active_key_count.numel() == 1,
              "active_key_count must contain one int32 value");
  int batch_count = static_cast<int>(query.size(0));
  int query_head_count = static_cast<int>(query.size(1));
  int kv_head_count = static_cast<int>(key.size(1));
  int key_count = static_cast<int>(key.size(2));
  int head_dim = static_cast<int>(query.size(2));
  int candidate_capacity = static_cast<int>(candidate_indices.size(2));
  int value_rank = static_cast<int>(tail_coefficients.size(2));
  int row_count = batch_count * query_head_count;
  int split_count = static_cast<int>(partial_output.size(1));
  TORCH_CHECK(key.stride(3) == 1 && key.stride(2) == head_dim
                  && value.stride(3) == 1 && value.stride(2) == head_dim,
              "active Key/Value token rows must be contiguous");
  TORCH_CHECK(split_count == 1 || split_count == 2 || split_count == 4
                  || split_count == 8 || split_count == 16,
              "active ValueSketch split count is invalid");
  TORCH_CHECK(query_head_count % kv_head_count == 0,
              "active Query heads must be divisible by KV heads");
  TORCH_CHECK(head_dim == key.size(3) && head_dim == value_mean.size(2),
              "active head dimensions do not match");
  TORCH_CHECK(value_rank > 0 && value_rank <= head_dim
                  && value_basis.size(3) == value_rank,
              "active Value basis rank is invalid");
  TORCH_CHECK(tail_alpha >= 0.0 && tail_alpha <= 1.0,
              "active Value-tail alpha must lie in [0, 1]");
  TORCH_CHECK(candidate_indices.numel()
                  == static_cast<int64_t>(row_count) * candidate_capacity
                  && candidate_counts.numel() == row_count,
              "active candidate shapes are invalid");
  TORCH_CHECK(thresholds.numel() >= row_count
                  && tail_denominator.numel() == row_count
                  && tail_coefficients.numel()
                      == static_cast<int64_t>(row_count) * value_rank,
              "active tail-statistic shapes are invalid");
  TORCH_CHECK(output.sizes() == query.sizes(),
              "active output workspace must match query shape");
  TORCH_CHECK(partial_output.dim() == 3
                  && partial_output.size(0) == row_count
                  && partial_output.size(1) == split_count
                  && partial_output.size(2) == head_dim
                  && partial_maximum.dim() == 2
                  && partial_maximum.size(0) == row_count
                  && partial_maximum.size(1) == split_count
                  && partial_sum.sizes() == partial_maximum.sizes(),
              "active partial workspace shapes are invalid");
  int threads = 128;
  int maximum_local_count =
      (candidate_capacity + 1 + split_count - 1) / split_count;
  size_t shared_bytes = static_cast<size_t>(
      maximum_local_count + threads) * sizeof(float);
  TORCH_CHECK(shared_bytes <= 48 * 1024,
              "active candidate capacity exceeds the shared-memory limit");
  c10::cuda::CUDAGuard device_guard(query.device());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "qksieve_valuesketch_attention_active_out",
      [&] {
        qksieve_valuesketch_exact_split_kernel<scalar_t><<<
            row_count * split_count, threads, shared_bytes,
            at::cuda::getCurrentCUDAStream()>>>(
                query.data_ptr<scalar_t>(),
                key.data_ptr<scalar_t>(),
                value.data_ptr<scalar_t>(),
                candidate_indices.data_ptr<int64_t>(),
                candidate_counts.data_ptr<int64_t>(),
                partial_output.data_ptr<float>(),
                partial_maximum.data_ptr<float>(),
                partial_sum.data_ptr<float>(),
                query_head_count,
                kv_head_count,
                key_count,
                active_key_count.data_ptr<int32_t>(),
                candidate_capacity,
                head_dim,
                split_count,
                key.stride(0),
                key.stride(1),
                value.stride(0),
                value.stride(1),
                static_cast<float>(scaling));
        qksieve_valuesketch_reduce_tail_kernel<scalar_t><<<
            row_count, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                partial_output.data_ptr<float>(),
                partial_maximum.data_ptr<float>(),
                partial_sum.data_ptr<float>(),
                thresholds.data_ptr<float>(),
                tail_denominator.data_ptr<float>(),
                tail_coefficients.data_ptr<float>(),
                value_mean.data_ptr<scalar_t>(),
                value_basis.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                query_head_count,
                kv_head_count,
                head_dim,
                value_rank,
                split_count,
                static_cast<float>(scaling),
                static_cast<float>(tail_alpha));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#endif

void qksieve_valuesketch_attention_active_out(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor tail_denominator,
    torch::Tensor tail_coefficients,
    torch::Tensor value_mean,
    torch::Tensor value_basis,
    torch::Tensor active_key_count,
    torch::Tensor output,
    torch::Tensor partial_output,
    torch::Tensor partial_maximum,
    torch::Tensor partial_sum,
    double scaling,
    double tail_alpha);

void qksieve_append_suffix_candidates_out(
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor active_key_count,
    int64_t prefix_count,
    int64_t physical_key_count);

void qksieve_valuesketch_attention_tiled_out(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor tail_denominator,
    torch::Tensor tail_coefficients,
    torch::Tensor value_mean,
    torch::Tensor value_basis,
    torch::Tensor output,
    torch::Tensor partial_output,
    torch::Tensor partial_maximum,
    torch::Tensor partial_sum,
    double scaling,
    double tail_alpha);

torch::Tensor qksieve_condtail_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor tail_denominator,
    torch::Tensor tail_numerator,
    double scaling);

torch::Tensor qksieve_condtail_attention_shared_gqa_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor tail_denominator,
    torch::Tensor tail_numerator,
    double scaling);

torch::Tensor qksieve_condtail_reduce_moments(
    torch::Tensor tail_block_denominator,
    torch::Tensor tail_weighted_x,
    torch::Tensor mean_x,
    torch::Tensor mean_v,
    torch::Tensor linear_map);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "qksieve_valuesketch_attention_forward",
      &qksieve_valuesketch_attention_forward,
      "Exact selected attention plus an INT4 low-rank Value tail");
  m.def(
      "qksieve_valuesketch_attention_out",
      &qksieve_valuesketch_attention_out,
      "Exact selected attention plus an INT4 low-rank Value tail, using caller-owned workspaces");
  m.def(
      "qksieve_valuesketch_attention_active_out",
      &qksieve_valuesketch_attention_active_out,
      "Graph-safe selected attention with a device-side active KV length");
  m.def(
      "qksieve_append_suffix_candidates_out",
      &qksieve_append_suffix_candidates_out,
      "Append every active suffix token to each prefix candidate row");
  m.def(
      "qksieve_valuesketch_attention_tiled_out",
      &qksieve_valuesketch_attention_tiled_out,
      "Warp-tiled exact selected attention plus an INT4 low-rank Value tail");
  m.def(
      "qksieve_condtail_attention_forward",
      &qksieve_condtail_attention_forward,
      "Exact selected attention plus a direct conditional Value tail");
  m.def(
      "qksieve_condtail_attention_shared_gqa_forward",
      &qksieve_condtail_attention_shared_gqa_forward,
      "GQA-shared exact selected attention plus a conditional Value tail");
  m.def(
      "qksieve_condtail_reduce_moments",
      &qksieve_condtail_reduce_moments,
      "Reduce block moments through a shared K-to-V map");
  m.def(
      "qksieve_valuesketch_append_int4_out",
      &qksieve_valuesketch_append_int4_out,
      "Append projected INT4 Value-sketch coefficients");
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
__global__ void qksieve_valuesketch_append_int4_kernel(
    const scalar_t* __restrict__ value_history,
    const scalar_t* __restrict__ value_mean,
    const scalar_t* __restrict__ value_basis,
    const scalar_t* __restrict__ value_minimum,
    const scalar_t* __restrict__ value_scale,
    uint8_t* __restrict__ packed_codes,
    int batch_count,
    int kv_head_count,
    int token_count,
    int head_dim,
    int value_rank,
    int value_block_count,
    int packed_capacity,
    int input_start,
    int value_block_size,
    int64_t value_batch_stride,
    int64_t value_head_stride,
    int64_t value_token_stride) {
  int row = blockIdx.x;
  int token_offset = row % token_count;
  int batch_kv = row / token_count;
  int batch = batch_kv / kv_head_count;
  int kv_head = batch_kv - batch * kv_head_count;
  int token = input_start + token_offset;
  int block = token / value_block_size;
  const scalar_t* value_row = value_history
      + static_cast<int64_t>(batch) * value_batch_stride
      + static_cast<int64_t>(kv_head) * value_head_stride
      + static_cast<int64_t>(token) * value_token_stride;
  const scalar_t* mean_row = value_mean
      + static_cast<int64_t>(batch_kv) * head_dim;
  const scalar_t* basis_row = value_basis
      + static_cast<int64_t>(batch_kv) * head_dim * value_rank;
  const scalar_t* minimum_row = value_minimum
      + (static_cast<int64_t>(batch_kv) * value_block_count + block)
          * value_rank;
  const scalar_t* scale_row = value_scale
      + (static_cast<int64_t>(batch_kv) * value_block_count + block)
          * value_rank;
  uint8_t* output_row = packed_codes
      + (static_cast<int64_t>(batch_kv) * packed_capacity + token)
          * (value_rank / 2);
  for (int pair = threadIdx.x; pair < value_rank / 2;
       pair += blockDim.x) {
    uint8_t packed = 0;
    for (int lane = 0; lane < 2; ++lane) {
      int rank = 2 * pair + lane;
      float coefficient = 0.0f;
      for (int dimension = 0; dimension < head_dim; ++dimension) {
        coefficient += (
            static_cast<float>(value_row[dimension])
            - static_cast<float>(mean_row[dimension]))
            * static_cast<float>(
                basis_row[dimension * value_rank + rank]);
      }
      float minimum = static_cast<float>(minimum_row[rank]);
      float scale = fmaxf(static_cast<float>(scale_row[rank]), 1.0e-12f);
      int code = static_cast<int>(nearbyintf(
          (coefficient - minimum) / scale));
      code = max(0, min(15, code));
      packed |= static_cast<uint8_t>(code << (4 * lane));
    }
    output_row[pair] = packed;
  }
}

template <typename scalar_t>
__global__ void qksieve_valuesketch_exact_split_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const int64_t* __restrict__ candidate_indices,
    const int64_t* __restrict__ candidate_counts,
    float* __restrict__ partial_output,
    float* __restrict__ partial_maximum,
    float* __restrict__ partial_sum,
    int query_head_count,
    int kv_head_count,
    int key_count,
    const int32_t* __restrict__ active_key_count,
    int candidate_capacity,
    int head_dim,
    int split_count,
    int64_t key_batch_stride,
    int64_t key_head_stride,
    int64_t value_batch_stride,
    int64_t value_head_stride,
    float scaling) {
  extern __shared__ float shared[];
  float* reduction = shared;
  float* weights = shared + blockDim.x;
  int block = blockIdx.x;
  int row = block / split_count;
  int split = block - row * split_count;
  int tid = threadIdx.x;
  int batch = row / query_head_count;
  int query_head = row - batch * query_head_count;
  int query_groups = query_head_count / kv_head_count;
  int kv_head = query_head / query_groups;
  int batch_kv = batch * kv_head_count + kv_head;
  int logical_key_count = active_key_count == nullptr
      ? key_count
      : min(max(static_cast<int>(active_key_count[0]), 1), key_count);
  int selected_history = min(
      max(static_cast<int>(candidate_counts[row]), 0),
      candidate_capacity);
  int selected_count = selected_history + 1;
  int chunk = (selected_count + split_count - 1) / split_count;
  int start = min(split * chunk, selected_count);
  int end = min(start + chunk, selected_count);
  int local_count = end - start;
  int self_token = logical_key_count - 1;

  const scalar_t* query_row = query + row * head_dim;
  const scalar_t* key_base = key
      + static_cast<int64_t>(batch) * key_batch_stride
      + static_cast<int64_t>(kv_head) * key_head_stride;
  const scalar_t* value_base = value
      + static_cast<int64_t>(batch) * value_batch_stride
      + static_cast<int64_t>(kv_head) * value_head_stride;
  const int64_t* index_row =
      candidate_indices + static_cast<int64_t>(row) * candidate_capacity;
  float* output_row = partial_output
      + static_cast<int64_t>(row * split_count + split) * head_dim;

  float local_maximum = -INFINITY;
  for (int local = tid; local < local_count; local += blockDim.x) {
    int selected = start + local;
    int token = selected == selected_history
        ? self_token
        : static_cast<int>(index_row[selected]);
    float score = -INFINITY;
    if (token >= 0 && token < logical_key_count) {
      const scalar_t* key_row = key_base + token * head_dim;
      float accumulator = 0.0f;
      for (int dimension = 0; dimension < head_dim; ++dimension) {
        accumulator += static_cast<float>(query_row[dimension])
            * static_cast<float>(key_row[dimension]);
      }
      score = accumulator * scaling;
      local_maximum = fmaxf(local_maximum, score);
    }
    weights[local] = score;
  }
  reduction[tid] = local_maximum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
    }
    __syncthreads();
  }
  float maximum = isfinite(reduction[0]) ? reduction[0] : 0.0f;
  float local_sum = 0.0f;
  for (int local = tid; local < local_count; local += blockDim.x) {
    float weight = isfinite(weights[local])
        ? expf(weights[local] - maximum)
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
    partial_maximum[row * split_count + split] = maximum;
    partial_sum[row * split_count + split] = reduction[0];
  }
  for (int dimension = tid; dimension < head_dim;
       dimension += blockDim.x) {
    float accumulator = 0.0f;
    for (int local = 0; local < local_count; ++local) {
      int selected = start + local;
      int token = selected == selected_history
          ? self_token
          : static_cast<int>(index_row[selected]);
      if (token >= 0 && token < logical_key_count) {
        accumulator += weights[local]
            * static_cast<float>(
                value_base[token * head_dim + dimension]);
      }
    }
    output_row[dimension] = accumulator;
  }
}

__global__ void qksieve_append_suffix_candidates_kernel(
    int64_t* __restrict__ candidate_indices,
    int64_t* __restrict__ candidate_counts,
    const int32_t* __restrict__ active_key_count,
    int row_count,
    int candidate_capacity,
    int prefix_count,
    int physical_key_count) {
  int row = blockIdx.x;
  if (row >= row_count) {
    return;
  }
  int active_count = min(
      max(static_cast<int>(active_key_count[0]), prefix_count + 1),
      physical_key_count);
  int suffix_history_count = max(0, active_count - prefix_count - 1);
  int base_count = min(
      max(static_cast<int>(candidate_counts[row]), 0),
      candidate_capacity);
  int append_count = min(
      suffix_history_count,
      candidate_capacity - base_count);
  int64_t* row_indices = candidate_indices
      + static_cast<int64_t>(row) * candidate_capacity;
  for (int offset = threadIdx.x; offset < append_count;
       offset += blockDim.x) {
    row_indices[base_count + offset] = prefix_count + offset;
  }
  if (threadIdx.x == 0) {
    candidate_counts[row] = base_count + append_count;
  }
}

void qksieve_append_suffix_candidates_out(
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor active_key_count,
    int64_t prefix_count,
    int64_t physical_key_count) {
  TORCH_CHECK(
      candidate_indices.is_cuda() && candidate_counts.is_cuda()
          && active_key_count.is_cuda(),
      "suffix candidate tensors must be CUDA tensors");
  TORCH_CHECK(
      candidate_indices.scalar_type() == at::kLong
          && candidate_counts.scalar_type() == at::kLong
          && active_key_count.scalar_type() == at::kInt,
      "suffix candidates require int64 indices/counts and int32 active length");
  TORCH_CHECK(
      candidate_indices.dim() == 3 && candidate_counts.dim() == 2
          && active_key_count.numel() == 1,
      "suffix candidate shapes are invalid");
  TORCH_CHECK(
      candidate_indices.is_contiguous() && candidate_counts.is_contiguous()
          && active_key_count.is_contiguous(),
      "suffix candidate tensors must be contiguous");
  TORCH_CHECK(
      candidate_indices.size(0) == candidate_counts.size(0)
          && candidate_indices.size(1) == candidate_counts.size(1),
      "suffix candidate row shapes do not match");
  TORCH_CHECK(prefix_count >= 0, "prefix_count must be non-negative");
  TORCH_CHECK(
      physical_key_count > prefix_count,
      "physical_key_count must exceed prefix_count");
  int row_count = static_cast<int>(candidate_counts.numel());
  int candidate_capacity = static_cast<int>(candidate_indices.size(2));
  c10::cuda::CUDAGuard device_guard(candidate_indices.device());
  qksieve_append_suffix_candidates_kernel<<<
      row_count, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
          candidate_indices.data_ptr<int64_t>(),
          candidate_counts.data_ptr<int64_t>(),
          active_key_count.data_ptr<int32_t>(),
          row_count,
          candidate_capacity,
          static_cast<int>(prefix_count),
          static_cast<int>(physical_key_count));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__device__ __forceinline__ float qksieve_warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

// The reference kernel assigns one complete QK dot product to one thread,
// which makes adjacent lanes read Key rows 256 bytes apart.  This variant
// assigns one candidate to a warp: every lane reads four dimensions and all
// four 32-value transactions are contiguous.  Candidate order, softmax, AV,
// and ValueSketch reduction remain unchanged.
template <typename scalar_t>
__global__ void qksieve_valuesketch_exact_split_tiled_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const int64_t* __restrict__ candidate_indices,
    const int64_t* __restrict__ candidate_counts,
    float* __restrict__ partial_output,
    float* __restrict__ partial_maximum,
    float* __restrict__ partial_sum,
    int query_head_count,
    int kv_head_count,
    int key_count,
    int candidate_capacity,
    int head_dim,
    int split_count,
    int64_t key_batch_stride,
    int64_t key_head_stride,
    int64_t value_batch_stride,
    int64_t value_head_stride,
    float scaling) {
  extern __shared__ float shared[];
  float* reduction = shared;
  float* weights = shared + blockDim.x;
  int block = blockIdx.x;
  int row = block / split_count;
  int split = block - row * split_count;
  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp = tid >> 5;
  int warp_count = blockDim.x >> 5;
  int batch = row / query_head_count;
  int query_head = row - batch * query_head_count;
  int query_groups = query_head_count / kv_head_count;
  int kv_head = query_head / query_groups;
  int batch_kv = batch * kv_head_count + kv_head;
  int selected_history = min(
      max(static_cast<int>(candidate_counts[row]), 0),
      candidate_capacity);
  int selected_count = selected_history + 1;
  int chunk = (selected_count + split_count - 1) / split_count;
  int start = min(split * chunk, selected_count);
  int end = min(start + chunk, selected_count);
  int local_count = end - start;
  int self_token = key_count - 1;

  const scalar_t* query_row = query + row * head_dim;
  const scalar_t* key_base = key
      + static_cast<int64_t>(batch) * key_batch_stride
      + static_cast<int64_t>(kv_head) * key_head_stride;
  const scalar_t* value_base = value
      + static_cast<int64_t>(batch) * value_batch_stride
      + static_cast<int64_t>(kv_head) * value_head_stride;
  const int64_t* index_row =
      candidate_indices + static_cast<int64_t>(row) * candidate_capacity;
  float* output_row = partial_output
      + static_cast<int64_t>(row * split_count + split) * head_dim;

  float query_values[4];
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    query_values[item] = static_cast<float>(query_row[lane + item * 32]);
  }

  for (int local = warp; local < local_count; local += warp_count) {
    int token = -1;
    if (lane == 0) {
      int selected = start + local;
      token = selected == selected_history
          ? self_token
          : static_cast<int>(index_row[selected]);
    }
    token = __shfl_sync(0xffffffffu, token, 0);
    bool valid = token >= 0 && token < key_count;
    float accumulator = 0.0f;
    if (valid) {
      const scalar_t* key_row = key_base
          + static_cast<int64_t>(token) * head_dim;
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        int dimension = lane + item * 32;
        accumulator += query_values[item]
            * static_cast<float>(key_row[dimension]);
      }
    }
    accumulator = qksieve_warp_sum(accumulator);
    if (lane == 0) {
      weights[local] = valid ? accumulator * scaling : -INFINITY;
    }
  }
  __syncthreads();

  float local_maximum = -INFINITY;
  for (int local = tid; local < local_count; local += blockDim.x) {
    local_maximum = fmaxf(local_maximum, weights[local]);
  }
  reduction[tid] = local_maximum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
    }
    __syncthreads();
  }
  float maximum = isfinite(reduction[0]) ? reduction[0] : 0.0f;
  float local_sum = 0.0f;
  for (int local = tid; local < local_count; local += blockDim.x) {
    float weight = isfinite(weights[local])
        ? expf(weights[local] - maximum)
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
    partial_maximum[row * split_count + split] = maximum;
    partial_sum[row * split_count + split] = reduction[0];
  }
  for (int dimension = tid; dimension < head_dim;
       dimension += blockDim.x) {
    float accumulator = 0.0f;
    for (int local = 0; local < local_count; ++local) {
      int selected = start + local;
      int token = selected == selected_history
          ? self_token
          : static_cast<int>(index_row[selected]);
      if (token >= 0 && token < key_count) {
        accumulator += weights[local]
            * static_cast<float>(value_base[
                static_cast<int64_t>(token) * head_dim + dimension]);
      }
    }
    output_row[dimension] = accumulator;
  }
}

template <typename scalar_t>
__global__ void qksieve_condtail_exact_shared_gqa4_split_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const int64_t* __restrict__ candidate_indices,
    const int64_t* __restrict__ candidate_counts,
    float* __restrict__ partial_output,
    float* __restrict__ partial_maximum,
    float* __restrict__ partial_sum,
    int query_head_count,
    int kv_head_count,
    int key_count,
    int candidate_capacity,
    int head_dim,
    int split_count,
    int maximum_local_count,
    float scaling) {
  constexpr int query_groups = 4;
  extern __shared__ float shared[];
  float* weights = shared;
  float* reduction = weights + query_groups * maximum_local_count;
  int block = blockIdx.x;
  int batch_kv = block / split_count;
  int split = block - batch_kv * split_count;
  int tid = threadIdx.x;
  int batch = batch_kv / kv_head_count;
  int kv_head = batch_kv - batch * kv_head_count;
  int query_head_base = batch * query_head_count + kv_head * query_groups;
  int selected_history = min(
      max(static_cast<int>(candidate_counts[batch_kv]), 0),
      candidate_capacity);
  int selected_count = selected_history + 1;
  int chunk = (selected_count + split_count - 1) / split_count;
  int start = min(split * chunk, selected_count);
  int end = min(start + chunk, selected_count);
  int local_count = end - start;
  int self_token = key_count - 1;

  const scalar_t* query_base =
      query + static_cast<int64_t>(query_head_base) * head_dim;
  const scalar_t* key_base =
      key + static_cast<int64_t>(batch_kv) * key_count * head_dim;
  const scalar_t* value_base =
      value + static_cast<int64_t>(batch_kv) * key_count * head_dim;
  const int64_t* index_row = candidate_indices
      + static_cast<int64_t>(batch_kv) * candidate_capacity;

  float local_maximum[query_groups] = {
      -INFINITY, -INFINITY, -INFINITY, -INFINITY};
  for (int local = tid; local < local_count; local += blockDim.x) {
    int selected = start + local;
    int token = selected == selected_history
        ? self_token
        : static_cast<int>(index_row[selected]);
    float scores[query_groups] = {
        -INFINITY, -INFINITY, -INFINITY, -INFINITY};
    if (token >= 0 && token < key_count) {
      const scalar_t* key_row = key_base + token * head_dim;
      float accumulators[query_groups] = {0.0f, 0.0f, 0.0f, 0.0f};
      for (int dimension = 0; dimension < head_dim; ++dimension) {
        float key_value = static_cast<float>(key_row[dimension]);
#pragma unroll
        for (int group = 0; group < query_groups; ++group) {
          accumulators[group] += static_cast<float>(
              query_base[group * head_dim + dimension]) * key_value;
        }
      }
#pragma unroll
      for (int group = 0; group < query_groups; ++group) {
        scores[group] = accumulators[group] * scaling;
        local_maximum[group] = fmaxf(local_maximum[group], scores[group]);
      }
    }
#pragma unroll
    for (int group = 0; group < query_groups; ++group) {
      weights[group * maximum_local_count + local] = scores[group];
    }
  }

#pragma unroll
  for (int group = 0; group < query_groups; ++group) {
    reduction[tid] = local_maximum[group];
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
      }
      __syncthreads();
    }
    float maximum = isfinite(reduction[0]) ? reduction[0] : 0.0f;
    float local_sum = 0.0f;
    for (int local = tid; local < local_count; local += blockDim.x) {
      float score = weights[group * maximum_local_count + local];
      float weight = isfinite(score) ? expf(score - maximum) : 0.0f;
      weights[group * maximum_local_count + local] = weight;
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
      int row = query_head_base + group;
      partial_maximum[row * split_count + split] = maximum;
      partial_sum[row * split_count + split] = reduction[0];
    }
    __syncthreads();
  }

  for (int dimension = tid; dimension < head_dim;
       dimension += blockDim.x) {
    float accumulators[query_groups] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int local = 0; local < local_count; ++local) {
      int selected = start + local;
      int token = selected == selected_history
          ? self_token
          : static_cast<int>(index_row[selected]);
      if (token >= 0 && token < key_count) {
        float value_element = static_cast<float>(
            value_base[token * head_dim + dimension]);
#pragma unroll
        for (int group = 0; group < query_groups; ++group) {
          accumulators[group] +=
              weights[group * maximum_local_count + local] * value_element;
        }
      }
    }
#pragma unroll
    for (int group = 0; group < query_groups; ++group) {
      int row = query_head_base + group;
      partial_output[
          (static_cast<int64_t>(row) * split_count + split) * head_dim
          + dimension] = accumulators[group];
    }
  }
}

template <typename scalar_t>
__global__ void qksieve_valuesketch_reduce_tail_kernel(
    const float* __restrict__ partial_output,
    const float* __restrict__ partial_maximum,
    const float* __restrict__ partial_sum,
    const float* __restrict__ thresholds,
    const float* __restrict__ tail_denominator,
    const float* __restrict__ tail_coefficients,
    const scalar_t* __restrict__ value_mean,
    const scalar_t* __restrict__ value_basis,
    scalar_t* __restrict__ output,
    int query_head_count,
    int kv_head_count,
    int head_dim,
    int value_rank,
    int split_count,
    float scaling,
    float tail_alpha) {
  __shared__ float global_maximum;
  __shared__ float inverse_denominator;
  __shared__ float tail_factor;
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int batch = row / query_head_count;
  int query_head = row - batch * query_head_count;
  int query_groups = query_head_count / kv_head_count;
  int kv_head = query_head / query_groups;
  int batch_kv = batch * kv_head_count + kv_head;
  if (tid == 0) {
    float maximum = tail_denominator[row] > 0.0f
        ? thresholds[row] * scaling
        : -INFINITY;
    for (int split = 0; split < split_count; ++split) {
      int partial = row * split_count + split;
      if (partial_sum[partial] > 0.0f) {
        maximum = fmaxf(maximum, partial_maximum[partial]);
      }
    }
    maximum = isfinite(maximum) ? maximum : 0.0f;
    float denominator = 0.0f;
    for (int split = 0; split < split_count; ++split) {
      int partial = row * split_count + split;
      if (partial_sum[partial] > 0.0f) {
        denominator += partial_sum[partial]
            * expf(partial_maximum[partial] - maximum);
      }
    }
    float tail_scale = expf(fminf(
        80.0f,
        fmaxf(-80.0f, thresholds[row] * scaling - maximum)));
    denominator += tail_alpha * tail_denominator[row] * tail_scale;
    global_maximum = maximum;
    tail_factor = tail_scale;
    inverse_denominator = 1.0f / fmaxf(denominator, 1.0e-20f);
  }
  __syncthreads();

  const float* partial_row = partial_output
      + static_cast<int64_t>(row) * split_count * head_dim;
  const scalar_t* mean_row =
      value_mean + static_cast<int64_t>(batch_kv) * head_dim;
  const scalar_t* basis_row = value_basis
      + static_cast<int64_t>(batch_kv) * head_dim * value_rank;
  const float* coefficient_row = tail_coefficients
      + static_cast<int64_t>(row) * value_rank;
  scalar_t* output_row = output + row * head_dim;
  for (int dimension = tid; dimension < head_dim;
       dimension += blockDim.x) {
    float accumulator = 0.0f;
    for (int split = 0; split < split_count; ++split) {
      int partial = row * split_count + split;
      if (partial_sum[partial] > 0.0f) {
        accumulator += expf(
            partial_maximum[partial] - global_maximum)
            * partial_row[split * head_dim + dimension];
      }
    }
    float tail_numerator = tail_denominator[row]
        * static_cast<float>(mean_row[dimension]);
    for (int rank = 0; rank < value_rank; ++rank) {
      tail_numerator += coefficient_row[rank]
          * static_cast<float>(basis_row[dimension * value_rank + rank]);
    }
    output_row[dimension] = static_cast<scalar_t>(
        (accumulator + tail_alpha * tail_factor * tail_numerator)
        * inverse_denominator);
  }
}

template <typename scalar_t>
__global__ void qksieve_condtail_reduce_moments_kernel(
    const float* __restrict__ tail_block_denominator,
    const float* __restrict__ tail_weighted_x,
    const scalar_t* __restrict__ mean_x,
    const scalar_t* __restrict__ mean_v,
    const scalar_t* __restrict__ linear_map,
    float* __restrict__ tail_numerator,
    int query_head_count,
    int kv_head_count,
    int block_count,
    int head_dim,
    int rank) {
  extern __shared__ float shared[];
  float* block_mass = shared;
  float* centered_x = shared + block_count;
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int batch = row / query_head_count;
  int query_head = row - batch * query_head_count;
  int query_groups = query_head_count / kv_head_count;
  int kv_head = query_head / query_groups;
  int batch_kv = batch * kv_head_count + kv_head;
  const float* mass_row = tail_block_denominator
      + static_cast<int64_t>(row) * block_count;
  for (int block = tid; block < block_count; block += blockDim.x) {
    block_mass[block] = mass_row[block];
  }
  __syncthreads();
  for (int component = tid; component < rank; component += blockDim.x) {
    float centered = tail_weighted_x[
        static_cast<int64_t>(row) * rank + component];
    for (int block = 0; block < block_count; ++block) {
      centered -= block_mass[block] * static_cast<float>(
          mean_x[
              (static_cast<int64_t>(batch_kv) * block_count + block)
                  * rank
              + component]);
    }
    centered_x[component] = centered;
  }
  __syncthreads();
  for (int dimension = tid; dimension < head_dim;
       dimension += blockDim.x) {
    float accumulator = 0.0f;
    for (int block = 0; block < block_count; ++block) {
      accumulator += block_mass[block] * static_cast<float>(
          mean_v[
              (static_cast<int64_t>(batch_kv) * block_count + block)
                  * head_dim
              + dimension]);
    }
    const scalar_t* map_row = linear_map
        + (static_cast<int64_t>(batch_kv) * head_dim + dimension) * rank;
    for (int component = 0; component < rank; ++component) {
      accumulator += centered_x[component]
          * static_cast<float>(map_row[component]);
    }
    tail_numerator[
        static_cast<int64_t>(row) * head_dim + dimension] = accumulator;
  }
}

template <typename scalar_t>
__global__ void qksieve_condtail_reduce_kernel(
    const float* __restrict__ partial_output,
    const float* __restrict__ partial_maximum,
    const float* __restrict__ partial_sum,
    const float* __restrict__ thresholds,
    const float* __restrict__ tail_denominator,
    const float* __restrict__ tail_numerator,
    scalar_t* __restrict__ output,
    int head_dim,
    int split_count,
    float scaling) {
  __shared__ float global_maximum;
  __shared__ float inverse_denominator;
  __shared__ float tail_factor;
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (tid == 0) {
    float maximum = tail_denominator[row] > 0.0f
        ? thresholds[row] * scaling
        : -INFINITY;
    for (int split = 0; split < split_count; ++split) {
      int partial = row * split_count + split;
      if (partial_sum[partial] > 0.0f) {
        maximum = fmaxf(maximum, partial_maximum[partial]);
      }
    }
    maximum = isfinite(maximum) ? maximum : 0.0f;
    float denominator = 0.0f;
    for (int split = 0; split < split_count; ++split) {
      int partial = row * split_count + split;
      if (partial_sum[partial] > 0.0f) {
        denominator += partial_sum[partial]
            * expf(partial_maximum[partial] - maximum);
      }
    }
    float scale = expf(fminf(
        80.0f,
        fmaxf(-80.0f, thresholds[row] * scaling - maximum)));
    denominator += tail_denominator[row] * scale;
    global_maximum = maximum;
    tail_factor = scale;
    inverse_denominator = 1.0f / fmaxf(denominator, 1.0e-20f);
  }
  __syncthreads();

  const float* partial_row = partial_output
      + static_cast<int64_t>(row) * split_count * head_dim;
  const float* tail_row = tail_numerator
      + static_cast<int64_t>(row) * head_dim;
  scalar_t* output_row = output + static_cast<int64_t>(row) * head_dim;
  for (int dimension = tid; dimension < head_dim;
       dimension += blockDim.x) {
    float accumulator = 0.0f;
    for (int split = 0; split < split_count; ++split) {
      int partial = row * split_count + split;
      if (partial_sum[partial] > 0.0f) {
        accumulator += expf(
            partial_maximum[partial] - global_maximum)
            * partial_row[split * head_dim + dimension];
      }
    }
    accumulator += tail_factor * tail_row[dimension];
    output_row[dimension] = static_cast<scalar_t>(
        accumulator * inverse_denominator);
  }
}

template <typename scalar_t>
__global__ void qksieve_valuesketch_attention_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const int64_t* __restrict__ candidate_indices,
    const int64_t* __restrict__ candidate_counts,
    const float* __restrict__ thresholds,
    const float* __restrict__ tail_denominator,
    const float* __restrict__ tail_coefficients,
    const scalar_t* __restrict__ value_mean,
    const scalar_t* __restrict__ value_basis,
    scalar_t* __restrict__ output,
    int query_head_count,
    int kv_head_count,
    int key_count,
    int candidate_capacity,
    int head_dim,
    int value_rank,
    float scaling) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int batch = row / query_head_count;
  int query_head = row - batch * query_head_count;
  int query_groups = query_head_count / kv_head_count;
  int kv_head = query_head / query_groups;
  int batch_kv = batch * kv_head_count + kv_head;
  int selected_history = min(
      max(static_cast<int>(candidate_counts[row]), 0),
      candidate_capacity);
  int selected_count = selected_history + 1;
  int self_token = key_count - 1;

  extern __shared__ float shared[];
  float* weights = shared;
  float* reduction = shared + candidate_capacity + 1;
  const scalar_t* query_row = query + row * head_dim;
  const scalar_t* key_base =
      key + static_cast<int64_t>(batch_kv) * key_count * head_dim;
  const scalar_t* value_base =
      value + static_cast<int64_t>(batch_kv) * key_count * head_dim;

  float local_maximum = -INFINITY;
  for (int selected = tid; selected < selected_count;
       selected += blockDim.x) {
    int token = selected == selected_history
        ? self_token
        : static_cast<int>(
            candidate_indices[
                static_cast<int64_t>(row) * candidate_capacity + selected]);
    const scalar_t* key_row = key_base + token * head_dim;
    float score = 0.0f;
    for (int dimension = 0; dimension < head_dim; ++dimension) {
      score += static_cast<float>(query_row[dimension])
          * static_cast<float>(key_row[dimension]);
    }
    score *= scaling;
    weights[selected] = score;
    local_maximum = fmaxf(local_maximum, score);
  }
  reduction[tid] = local_maximum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
    }
    __syncthreads();
  }
  float exact_maximum = reduction[0];

  float local_denominator = 0.0f;
  for (int selected = tid; selected < selected_count;
       selected += blockDim.x) {
    float weight = expf(weights[selected] - exact_maximum);
    weights[selected] = weight;
    local_denominator += weight;
  }
  reduction[tid] = local_denominator;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }
  float exact_denominator = reduction[0];
  float tail_anchor = thresholds[row] * scaling;
  float tail_factor = expf(fminf(
      80.0f, fmaxf(-80.0f, tail_anchor - exact_maximum)));
  float scaled_tail_denominator = tail_denominator[row] * tail_factor;
  float inverse_denominator = 1.0f / fmaxf(
      exact_denominator + scaled_tail_denominator, 1.0e-20f);

  const scalar_t* mean_row =
      value_mean + static_cast<int64_t>(batch_kv) * head_dim;
  const scalar_t* basis_row = value_basis
      + static_cast<int64_t>(batch_kv) * head_dim * value_rank;
  const float* coefficient_row = tail_coefficients
      + static_cast<int64_t>(row) * value_rank;
  scalar_t* output_row = output + row * head_dim;
  for (int dimension = tid; dimension < head_dim;
       dimension += blockDim.x) {
    float exact_numerator = 0.0f;
    for (int selected = 0; selected < selected_count; ++selected) {
      int token = selected == selected_history
          ? self_token
          : static_cast<int>(
              candidate_indices[
                  static_cast<int64_t>(row) * candidate_capacity + selected]);
      exact_numerator += weights[selected]
          * static_cast<float>(value_base[token * head_dim + dimension]);
    }
    float tail_numerator = tail_denominator[row]
        * static_cast<float>(mean_row[dimension]);
    for (int rank = 0; rank < value_rank; ++rank) {
      tail_numerator += coefficient_row[rank]
          * static_cast<float>(basis_row[dimension * value_rank + rank]);
    }
    output_row[dimension] = static_cast<scalar_t>(
        (exact_numerator + tail_factor * tail_numerator)
        * inverse_denominator);
  }
}

void qksieve_valuesketch_append_int4_out(
    torch::Tensor value_history,
    torch::Tensor value_mean,
    torch::Tensor value_basis,
    torch::Tensor value_minimum,
    torch::Tensor value_scale,
    torch::Tensor packed_codes,
    int64_t input_start,
    int64_t input_stop,
    int64_t value_block_size) {
  TORCH_CHECK(value_history.is_cuda() && packed_codes.is_cuda(),
              "Value-sketch tensors must be CUDA tensors");
  TORCH_CHECK(value_history.dim() == 4 && value_mean.dim() == 3
                  && value_basis.dim() == 4,
              "Value-sketch projection shapes are invalid");
  TORCH_CHECK(value_minimum.sizes() == value_scale.sizes()
                  && value_minimum.dim() == 4,
              "Value-sketch metadata shapes are invalid");
  TORCH_CHECK(packed_codes.scalar_type() == at::kByte
                  && packed_codes.dim() == 4,
              "packed Value codes must be uint8");
  TORCH_CHECK(value_history.scalar_type() == value_mean.scalar_type()
                  && value_history.scalar_type() == value_basis.scalar_type()
                  && value_history.scalar_type()
                      == value_minimum.scalar_type()
                  && value_history.scalar_type() == value_scale.scalar_type(),
              "Value-sketch floating tensors must share a dtype");
  int batch_count = static_cast<int>(value_history.size(0));
  int kv_head_count = static_cast<int>(value_history.size(1));
  int history_count = static_cast<int>(value_history.size(2));
  int head_dim = static_cast<int>(value_history.size(3));
  int value_rank = static_cast<int>(value_basis.size(3));
  int value_block_count = static_cast<int>(value_minimum.size(2));
  int packed_capacity = static_cast<int>(packed_codes.size(2));
  TORCH_CHECK(value_rank > 0 && value_rank <= head_dim
                  && value_rank % 2 == 0,
              "Value-sketch append rank must be a positive even value "
              "not exceeding head_dim");
  TORCH_CHECK(value_basis.size(2) == head_dim
                  && value_mean.size(2) == head_dim,
              "Value-sketch head dimensions do not match");
  TORCH_CHECK(input_start >= 0 && input_start <= input_stop
                  && input_stop <= history_count
                  && input_stop <= packed_capacity,
              "Value-sketch append range is invalid");
  TORCH_CHECK(value_block_size > 0
                  && value_block_count
                      >= (input_stop + value_block_size - 1)
                          / value_block_size,
              "Value-sketch metadata does not cover the append range");
  TORCH_CHECK(value_history.stride(3) == 1,
              "Value-sketch head dimension must be contiguous");
  int token_count = static_cast<int>(input_stop - input_start);
  if (token_count == 0) {
    return;
  }
  c10::cuda::CUDAGuard device_guard(value_history.device());
  int threads = 32;
  int blocks = batch_count * kv_head_count * token_count;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      value_history.scalar_type(),
      "qksieve_valuesketch_append_int4_out",
      [&] {
        qksieve_valuesketch_append_int4_kernel<scalar_t><<<
            blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                value_history.data_ptr<scalar_t>(),
                value_mean.data_ptr<scalar_t>(),
                value_basis.data_ptr<scalar_t>(),
                value_minimum.data_ptr<scalar_t>(),
                value_scale.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                batch_count,
                kv_head_count,
                token_count,
                head_dim,
                value_rank,
                value_block_count,
                packed_capacity,
                static_cast<int>(input_start),
                static_cast<int>(value_block_size),
                value_history.stride(0),
                value_history.stride(1),
                value_history.stride(2));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor qksieve_valuesketch_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor tail_denominator,
    torch::Tensor tail_coefficients,
    torch::Tensor value_mean,
    torch::Tensor value_basis,
    double scaling,
    double tail_alpha) {
  TORCH_CHECK(
      query.is_cuda() && key.is_cuda() && value.is_cuda()
          && candidate_indices.is_cuda() && candidate_counts.is_cuda(),
      "inputs must be CUDA tensors");
  TORCH_CHECK(query.dim() == 3 && key.dim() == 4 && value.dim() == 4,
              "query/key/value ranks are invalid");
  TORCH_CHECK(key.sizes() == value.sizes(),
              "Key and Value tensors must have identical shapes");
  TORCH_CHECK(query.scalar_type() == key.scalar_type()
                  && query.scalar_type() == value.scalar_type()
                  && query.scalar_type() == value_mean.scalar_type()
                  && query.scalar_type() == value_basis.scalar_type(),
              "floating-point tensors must share a dtype");
  TORCH_CHECK(candidate_indices.scalar_type() == at::kLong
                  && candidate_counts.scalar_type() == at::kLong,
              "candidate tensors must be int64");
  TORCH_CHECK(thresholds.scalar_type() == at::kFloat
                  && tail_denominator.scalar_type() == at::kFloat
                  && tail_coefficients.scalar_type() == at::kFloat,
              "tail statistics must be float32");
  int batch_count = static_cast<int>(query.size(0));
  int query_head_count = static_cast<int>(query.size(1));
  int kv_head_count = static_cast<int>(key.size(1));
  int key_count = static_cast<int>(key.size(2));
  int head_dim = static_cast<int>(query.size(2));
  int candidate_capacity = static_cast<int>(candidate_indices.size(2));
  int value_rank = static_cast<int>(tail_coefficients.size(2));
  int row_count = batch_count * query_head_count;
  TORCH_CHECK(query_head_count % kv_head_count == 0,
              "Query heads must be divisible by KV heads");
  TORCH_CHECK(head_dim == key.size(3) && head_dim == value_mean.size(2),
              "head dimensions do not match");
  TORCH_CHECK(value_rank > 0 && value_rank <= head_dim,
              "Value-tail rank must be positive and not exceed head_dim");
  TORCH_CHECK(tail_alpha >= 0.0 && tail_alpha <= 1.0,
              "Value-tail alpha must lie in [0, 1]");
  TORCH_CHECK(value_basis.size(3) == value_rank,
              "Value basis rank is invalid");
  TORCH_CHECK(candidate_indices.numel()
                  == static_cast<int64_t>(row_count) * candidate_capacity
                  && candidate_counts.numel() == row_count,
              "candidate shapes are invalid");
  TORCH_CHECK(thresholds.numel() >= row_count
                  && tail_denominator.numel() == row_count
                  && tail_coefficients.numel()
                      == static_cast<int64_t>(row_count) * value_rank,
              "tail-statistic shapes are invalid");
  auto query_c = query.contiguous();
  auto key_c = key.contiguous();
  auto value_c = value.contiguous();
  auto indices_c = candidate_indices.contiguous();
  auto counts_c = candidate_counts.contiguous();
  auto thresholds_c = thresholds.contiguous();
  auto denominator_c = tail_denominator.contiguous();
  auto coefficients_c = tail_coefficients.contiguous();
  auto mean_c = value_mean.contiguous();
  auto basis_c = value_basis.contiguous();
  auto output = torch::empty_like(query_c);
  int split_count = candidate_capacity <= 4096 ? 8 : 4;
  int threads = 128;
  while ((candidate_capacity + split_count - 1) / split_count + threads
         > 12 * 1024) {
    split_count *= 2;
  }
  auto partial_output = torch::empty(
      {row_count, split_count, head_dim},
      query_c.options().dtype(at::kFloat));
  auto partial_maximum = torch::empty(
      {row_count, split_count}, query_c.options().dtype(at::kFloat));
  auto partial_sum = torch::empty_like(partial_maximum);
  int maximum_local_count =
      (candidate_capacity + 1 + split_count - 1) / split_count;
  size_t shared_bytes = static_cast<size_t>(
      maximum_local_count + threads) * sizeof(float);
  TORCH_CHECK(shared_bytes <= 48 * 1024,
              "candidate capacity exceeds the kernel shared-memory limit");
  c10::cuda::CUDAGuard device_guard(query.device());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "qksieve_valuesketch_attention_forward",
      [&] {
        qksieve_valuesketch_exact_split_kernel<scalar_t><<<
            row_count * split_count, threads, shared_bytes,
            at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<scalar_t>(),
                key_c.data_ptr<scalar_t>(),
                value_c.data_ptr<scalar_t>(),
                indices_c.data_ptr<int64_t>(),
                counts_c.data_ptr<int64_t>(),
                partial_output.data_ptr<float>(),
                partial_maximum.data_ptr<float>(),
                partial_sum.data_ptr<float>(),
                query_head_count,
                kv_head_count,
                key_count,
                nullptr,
                candidate_capacity,
                head_dim,
                split_count,
                key_c.stride(0),
                key_c.stride(1),
                value_c.stride(0),
                value_c.stride(1),
                static_cast<float>(scaling));
        qksieve_valuesketch_reduce_tail_kernel<scalar_t><<<
            row_count, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                partial_output.data_ptr<float>(),
                partial_maximum.data_ptr<float>(),
                partial_sum.data_ptr<float>(),
                thresholds_c.data_ptr<float>(),
                denominator_c.data_ptr<float>(),
                coefficients_c.data_ptr<float>(),
                mean_c.data_ptr<scalar_t>(),
                basis_c.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                query_head_count,
                kv_head_count,
                head_dim,
                value_rank,
                split_count,
                static_cast<float>(scaling),
                static_cast<float>(tail_alpha));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

void qksieve_valuesketch_attention_out(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor tail_denominator,
    torch::Tensor tail_coefficients,
    torch::Tensor value_mean,
    torch::Tensor value_basis,
    torch::Tensor output,
    torch::Tensor partial_output,
    torch::Tensor partial_maximum,
    torch::Tensor partial_sum,
    double scaling,
    double tail_alpha) {
  TORCH_CHECK(
      query.is_cuda() && key.is_cuda() && value.is_cuda()
          && candidate_indices.is_cuda() && candidate_counts.is_cuda()
          && thresholds.is_cuda() && tail_denominator.is_cuda()
          && tail_coefficients.is_cuda() && value_mean.is_cuda()
          && value_basis.is_cuda() && output.is_cuda()
          && partial_output.is_cuda() && partial_maximum.is_cuda()
          && partial_sum.is_cuda(),
      "inputs and workspaces must be CUDA tensors");
  TORCH_CHECK(query.dim() == 3 && key.dim() == 4 && value.dim() == 4,
              "query/key/value ranks are invalid");
  TORCH_CHECK(key.sizes() == value.sizes(),
              "Key and Value tensors must have identical shapes");
  TORCH_CHECK(query.scalar_type() == key.scalar_type()
                  && query.scalar_type() == value.scalar_type()
                  && query.scalar_type() == value_mean.scalar_type()
                  && query.scalar_type() == value_basis.scalar_type()
                  && query.scalar_type() == output.scalar_type(),
              "floating-point tensors and output must share a dtype");
  TORCH_CHECK(candidate_indices.scalar_type() == at::kLong
                  && candidate_counts.scalar_type() == at::kLong,
              "candidate tensors must be int64");
  TORCH_CHECK(thresholds.scalar_type() == at::kFloat
                  && tail_denominator.scalar_type() == at::kFloat
                  && tail_coefficients.scalar_type() == at::kFloat
                  && partial_output.scalar_type() == at::kFloat
                  && partial_maximum.scalar_type() == at::kFloat
                  && partial_sum.scalar_type() == at::kFloat,
              "tail statistics and partial workspaces must be float32");
  TORCH_CHECK(query.is_contiguous() && candidate_indices.is_contiguous()
                  && candidate_counts.is_contiguous() && thresholds.is_contiguous()
                  && tail_denominator.is_contiguous()
                  && tail_coefficients.is_contiguous() && value_mean.is_contiguous()
                  && value_basis.is_contiguous() && output.is_contiguous()
                  && partial_output.is_contiguous()
                  && partial_maximum.is_contiguous() && partial_sum.is_contiguous(),
              "persistent ValueSketch inputs and workspaces must be contiguous");
  int batch_count = static_cast<int>(query.size(0));
  int query_head_count = static_cast<int>(query.size(1));
  int kv_head_count = static_cast<int>(key.size(1));
  int key_count = static_cast<int>(key.size(2));
  int head_dim = static_cast<int>(query.size(2));
  TORCH_CHECK(key.stride(3) == 1 && key.stride(2) == head_dim
                  && value.stride(3) == 1 && value.stride(2) == head_dim,
              "Key/Value token rows must be contiguous");
  int candidate_capacity = static_cast<int>(candidate_indices.size(2));
  int value_rank = static_cast<int>(tail_coefficients.size(2));
  int row_count = batch_count * query_head_count;
  int split_count = static_cast<int>(partial_output.size(1));
  TORCH_CHECK(
      split_count == 1 || split_count == 2 || split_count == 4
          || split_count == 8 || split_count == 16,
      "ValueSketch split count must be one of 1,2,4,8,16");
  int threads = 128;
  TORCH_CHECK(query_head_count % kv_head_count == 0,
              "Query heads must be divisible by KV heads");
  TORCH_CHECK(head_dim == key.size(3) && head_dim == value_mean.size(2),
              "head dimensions do not match");
  TORCH_CHECK(value_rank > 0 && value_rank <= head_dim,
              "Value-tail rank must be positive and not exceed head_dim");
  TORCH_CHECK(tail_alpha >= 0.0 && tail_alpha <= 1.0,
              "Value-tail alpha must lie in [0, 1]");
  TORCH_CHECK(value_basis.size(3) == value_rank,
              "Value basis rank is invalid");
  TORCH_CHECK(candidate_indices.numel()
                  == static_cast<int64_t>(row_count) * candidate_capacity
                  && candidate_counts.numel() == row_count,
              "candidate shapes are invalid");
  TORCH_CHECK(thresholds.numel() >= row_count
                  && tail_denominator.numel() == row_count
                  && tail_coefficients.numel()
                      == static_cast<int64_t>(row_count) * value_rank,
              "tail-statistic shapes are invalid");
  TORCH_CHECK(output.sizes() == query.sizes(),
              "output workspace must match query shape");
  TORCH_CHECK(partial_output.dim() == 3
                  && partial_output.size(0) == row_count
                  && partial_output.size(1) == split_count
                  && partial_output.size(2) == head_dim,
              "partial-output workspace shape is invalid");
  TORCH_CHECK(partial_maximum.dim() == 2
                  && partial_maximum.size(0) == row_count
                  && partial_maximum.size(1) == split_count
                  && partial_sum.sizes() == partial_maximum.sizes(),
              "partial scalar workspace shapes are invalid");
  int maximum_local_count =
      (candidate_capacity + 1 + split_count - 1) / split_count;
  size_t shared_bytes = static_cast<size_t>(
      maximum_local_count + threads) * sizeof(float);
  TORCH_CHECK(shared_bytes <= 48 * 1024,
              "candidate capacity exceeds the kernel shared-memory limit");
  c10::cuda::CUDAGuard device_guard(query.device());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "qksieve_valuesketch_attention_out",
      [&] {
        qksieve_valuesketch_exact_split_kernel<scalar_t><<<
            row_count * split_count, threads, shared_bytes,
            at::cuda::getCurrentCUDAStream()>>>(
                query.data_ptr<scalar_t>(),
                key.data_ptr<scalar_t>(),
                value.data_ptr<scalar_t>(),
                candidate_indices.data_ptr<int64_t>(),
                candidate_counts.data_ptr<int64_t>(),
                partial_output.data_ptr<float>(),
                partial_maximum.data_ptr<float>(),
                partial_sum.data_ptr<float>(),
                query_head_count,
                kv_head_count,
                key_count,
                nullptr,
                candidate_capacity,
                head_dim,
                split_count,
                key.stride(0),
                key.stride(1),
                value.stride(0),
                value.stride(1),
                static_cast<float>(scaling));
        qksieve_valuesketch_reduce_tail_kernel<scalar_t><<<
            row_count, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                partial_output.data_ptr<float>(),
                partial_maximum.data_ptr<float>(),
                partial_sum.data_ptr<float>(),
                thresholds.data_ptr<float>(),
                tail_denominator.data_ptr<float>(),
                tail_coefficients.data_ptr<float>(),
                value_mean.data_ptr<scalar_t>(),
                value_basis.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                query_head_count,
                kv_head_count,
                head_dim,
                value_rank,
                split_count,
                static_cast<float>(scaling),
                static_cast<float>(tail_alpha));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void qksieve_valuesketch_attention_tiled_out(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor tail_denominator,
    torch::Tensor tail_coefficients,
    torch::Tensor value_mean,
    torch::Tensor value_basis,
    torch::Tensor output,
    torch::Tensor partial_output,
    torch::Tensor partial_maximum,
    torch::Tensor partial_sum,
    double scaling,
    double tail_alpha) {
  TORCH_CHECK(
      query.is_cuda() && key.is_cuda() && value.is_cuda()
          && candidate_indices.is_cuda() && candidate_counts.is_cuda()
          && thresholds.is_cuda() && tail_denominator.is_cuda()
          && tail_coefficients.is_cuda() && value_mean.is_cuda()
          && value_basis.is_cuda() && output.is_cuda()
          && partial_output.is_cuda() && partial_maximum.is_cuda()
          && partial_sum.is_cuda(),
      "inputs and workspaces must be CUDA tensors");
  TORCH_CHECK(query.dim() == 3 && key.dim() == 4 && value.dim() == 4,
              "query/key/value ranks are invalid");
  TORCH_CHECK(key.sizes() == value.sizes(),
              "Key and Value tensors must have identical shapes");
  TORCH_CHECK(query.scalar_type() == key.scalar_type()
                  && query.scalar_type() == value.scalar_type()
                  && query.scalar_type() == value_mean.scalar_type()
                  && query.scalar_type() == value_basis.scalar_type()
                  && query.scalar_type() == output.scalar_type(),
              "floating-point tensors and output must share a dtype");
  TORCH_CHECK(candidate_indices.scalar_type() == at::kLong
                  && candidate_counts.scalar_type() == at::kLong,
              "candidate tensors must be int64");
  TORCH_CHECK(thresholds.scalar_type() == at::kFloat
                  && tail_denominator.scalar_type() == at::kFloat
                  && tail_coefficients.scalar_type() == at::kFloat
                  && partial_output.scalar_type() == at::kFloat
                  && partial_maximum.scalar_type() == at::kFloat
                  && partial_sum.scalar_type() == at::kFloat,
              "tail statistics and partial workspaces must be float32");
  TORCH_CHECK(query.is_contiguous() && candidate_indices.is_contiguous()
                  && candidate_counts.is_contiguous() && thresholds.is_contiguous()
                  && tail_denominator.is_contiguous()
                  && tail_coefficients.is_contiguous() && value_mean.is_contiguous()
                  && value_basis.is_contiguous() && output.is_contiguous()
                  && partial_output.is_contiguous()
                  && partial_maximum.is_contiguous() && partial_sum.is_contiguous(),
              "persistent ValueSketch inputs and workspaces must be contiguous");
  int batch_count = static_cast<int>(query.size(0));
  int query_head_count = static_cast<int>(query.size(1));
  int kv_head_count = static_cast<int>(key.size(1));
  int key_count = static_cast<int>(key.size(2));
  int head_dim = static_cast<int>(query.size(2));
  int candidate_capacity = static_cast<int>(candidate_indices.size(2));
  int value_rank = static_cast<int>(tail_coefficients.size(2));
  int row_count = batch_count * query_head_count;
  int split_count = candidate_capacity <= 4096 ? 8 : 4;
  int threads = 128;
  while ((candidate_capacity + split_count - 1) / split_count + threads
         > 12 * 1024) {
    split_count *= 2;
  }
  TORCH_CHECK(query_head_count % kv_head_count == 0,
              "Query heads must be divisible by KV heads");
  TORCH_CHECK(head_dim == 128 && head_dim == key.size(3)
                  && head_dim == value_mean.size(2),
              "tiled ValueSketch currently requires head_dim 128");
  TORCH_CHECK(key.stride(3) == 1 && key.stride(2) == head_dim
                  && value.stride(3) == 1 && value.stride(2) == head_dim,
              "Key/Value token rows must be contiguous");
  TORCH_CHECK(value_rank > 0 && value_rank <= head_dim,
              "Value-tail rank must be positive and not exceed head_dim");
  TORCH_CHECK(tail_alpha >= 0.0 && tail_alpha <= 1.0,
              "Value-tail alpha must lie in [0, 1]");
  TORCH_CHECK(value_basis.size(3) == value_rank,
              "Value basis rank is invalid");
  TORCH_CHECK(candidate_indices.numel()
                  == static_cast<int64_t>(row_count) * candidate_capacity
                  && candidate_counts.numel() == row_count,
              "candidate shapes are invalid");
  TORCH_CHECK(thresholds.numel() >= row_count
                  && tail_denominator.numel() == row_count
                  && tail_coefficients.numel()
                      == static_cast<int64_t>(row_count) * value_rank,
              "tail-statistic shapes are invalid");
  TORCH_CHECK(output.sizes() == query.sizes(),
              "output workspace must match query shape");
  TORCH_CHECK(partial_output.dim() == 3
                  && partial_output.size(0) == row_count
                  && partial_output.size(1) == split_count
                  && partial_output.size(2) == head_dim,
              "partial-output workspace shape is invalid");
  TORCH_CHECK(partial_maximum.dim() == 2
                  && partial_maximum.size(0) == row_count
                  && partial_maximum.size(1) == split_count
                  && partial_sum.sizes() == partial_maximum.sizes(),
              "partial scalar workspace shapes are invalid");
  int maximum_local_count =
      (candidate_capacity + 1 + split_count - 1) / split_count;
  size_t shared_bytes = static_cast<size_t>(
      maximum_local_count + threads) * sizeof(float);
  TORCH_CHECK(shared_bytes <= 48 * 1024,
              "candidate capacity exceeds the kernel shared-memory limit");
  c10::cuda::CUDAGuard device_guard(query.device());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "qksieve_valuesketch_attention_tiled_out",
      [&] {
        qksieve_valuesketch_exact_split_tiled_kernel<scalar_t><<<
            row_count * split_count, threads, shared_bytes,
            at::cuda::getCurrentCUDAStream()>>>(
                query.data_ptr<scalar_t>(),
                key.data_ptr<scalar_t>(),
                value.data_ptr<scalar_t>(),
                candidate_indices.data_ptr<int64_t>(),
                candidate_counts.data_ptr<int64_t>(),
                partial_output.data_ptr<float>(),
                partial_maximum.data_ptr<float>(),
                partial_sum.data_ptr<float>(),
                query_head_count,
                kv_head_count,
                key_count,
                candidate_capacity,
                head_dim,
                split_count,
                key.stride(0),
                key.stride(1),
                value.stride(0),
                value.stride(1),
                static_cast<float>(scaling));
        qksieve_valuesketch_reduce_tail_kernel<scalar_t><<<
            row_count, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                partial_output.data_ptr<float>(),
                partial_maximum.data_ptr<float>(),
                partial_sum.data_ptr<float>(),
                thresholds.data_ptr<float>(),
                tail_denominator.data_ptr<float>(),
                tail_coefficients.data_ptr<float>(),
                value_mean.data_ptr<scalar_t>(),
                value_basis.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                query_head_count,
                kv_head_count,
                head_dim,
                value_rank,
                split_count,
                static_cast<float>(scaling),
                static_cast<float>(tail_alpha));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor qksieve_condtail_reduce_moments(
    torch::Tensor tail_block_denominator,
    torch::Tensor tail_weighted_x,
    torch::Tensor mean_x,
    torch::Tensor mean_v,
    torch::Tensor linear_map) {
  TORCH_CHECK(
      tail_block_denominator.is_cuda() && tail_weighted_x.is_cuda()
          && mean_x.is_cuda() && mean_v.is_cuda() && linear_map.is_cuda(),
      "conditional-tail moment tensors must be CUDA tensors");
  TORCH_CHECK(tail_block_denominator.scalar_type() == at::kFloat
                  && tail_weighted_x.scalar_type() == at::kFloat,
              "accumulated conditional-tail moments must be float32");
  TORCH_CHECK(mean_x.scalar_type() == mean_v.scalar_type()
                  && mean_x.scalar_type() == linear_map.scalar_type(),
              "conditional-tail model tensors must share a dtype");
  TORCH_CHECK(tail_block_denominator.dim() == 3
                  && tail_weighted_x.dim() == 3
                  && mean_x.dim() == 4 && mean_v.dim() == 4
                  && linear_map.dim() == 4,
              "conditional-tail moment ranks are invalid");
  int batch_count = static_cast<int>(tail_block_denominator.size(0));
  int query_head_count = static_cast<int>(tail_block_denominator.size(1));
  int block_count = static_cast<int>(tail_block_denominator.size(2));
  int kv_head_count = static_cast<int>(mean_x.size(1));
  int rank = static_cast<int>(mean_x.size(3));
  int head_dim = static_cast<int>(mean_v.size(3));
  int row_count = batch_count * query_head_count;
  TORCH_CHECK(query_head_count % kv_head_count == 0,
              "Query heads must be divisible by KV heads");
  TORCH_CHECK(rank == 8 && tail_weighted_x.size(2) == rank,
              "conditional-tail reducer currently requires rank 8");
  TORCH_CHECK(mean_x.size(0) == batch_count
                  && mean_x.size(2) == block_count
                  && mean_v.size(0) == batch_count
                  && mean_v.size(1) == kv_head_count
                  && mean_v.size(2) == block_count,
              "conditional-tail block-model shapes do not match");
  TORCH_CHECK(linear_map.size(0) == batch_count
                  && linear_map.size(1) == kv_head_count
                  && linear_map.size(2) == head_dim
                  && linear_map.size(3) == rank,
              "conditional-tail linear-map shape is invalid");
  auto mass_c = tail_block_denominator.contiguous();
  auto weighted_x_c = tail_weighted_x.contiguous();
  auto mean_x_c = mean_x.contiguous();
  auto mean_v_c = mean_v.contiguous();
  auto map_c = linear_map.contiguous();
  auto output = torch::empty(
      {batch_count, query_head_count, head_dim},
      mass_c.options().dtype(at::kFloat));
  int threads = 128;
  size_t shared_bytes = static_cast<size_t>(block_count + rank)
      * sizeof(float);
  TORCH_CHECK(shared_bytes <= 48 * 1024,
              "conditional-tail block count exceeds shared-memory capacity");
  c10::cuda::CUDAGuard device_guard(tail_block_denominator.device());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      mean_x.scalar_type(),
      "qksieve_condtail_reduce_moments",
      [&] {
        qksieve_condtail_reduce_moments_kernel<scalar_t><<<
            row_count, threads, shared_bytes,
            at::cuda::getCurrentCUDAStream()>>>(
                mass_c.data_ptr<float>(),
                weighted_x_c.data_ptr<float>(),
                mean_x_c.data_ptr<scalar_t>(),
                mean_v_c.data_ptr<scalar_t>(),
                map_c.data_ptr<scalar_t>(),
                output.data_ptr<float>(),
                query_head_count,
                kv_head_count,
                block_count,
                head_dim,
                rank);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qksieve_condtail_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor tail_denominator,
    torch::Tensor tail_numerator,
    double scaling) {
  TORCH_CHECK(
      query.is_cuda() && key.is_cuda() && value.is_cuda()
          && candidate_indices.is_cuda() && candidate_counts.is_cuda(),
      "inputs must be CUDA tensors");
  TORCH_CHECK(query.dim() == 3 && key.dim() == 4 && value.dim() == 4,
              "query/key/value ranks are invalid");
  TORCH_CHECK(key.sizes() == value.sizes(),
              "Key and Value tensors must have identical shapes");
  TORCH_CHECK(query.scalar_type() == key.scalar_type()
                  && query.scalar_type() == value.scalar_type(),
              "Q/K/V tensors must share a dtype");
  TORCH_CHECK(candidate_indices.scalar_type() == at::kLong
                  && candidate_counts.scalar_type() == at::kLong,
              "candidate tensors must be int64");
  TORCH_CHECK(thresholds.scalar_type() == at::kFloat
                  && tail_denominator.scalar_type() == at::kFloat
                  && tail_numerator.scalar_type() == at::kFloat,
              "tail statistics must be float32");
  int batch_count = static_cast<int>(query.size(0));
  int query_head_count = static_cast<int>(query.size(1));
  int kv_head_count = static_cast<int>(key.size(1));
  int key_count = static_cast<int>(key.size(2));
  int head_dim = static_cast<int>(query.size(2));
  int candidate_capacity = static_cast<int>(candidate_indices.size(2));
  int row_count = batch_count * query_head_count;
  TORCH_CHECK(query_head_count % kv_head_count == 0,
              "Query heads must be divisible by KV heads");
  TORCH_CHECK(head_dim == key.size(3), "head dimensions do not match");
  TORCH_CHECK(candidate_indices.numel()
                  == static_cast<int64_t>(row_count) * candidate_capacity
                  && candidate_counts.numel() == row_count,
              "candidate shapes are invalid");
  TORCH_CHECK(thresholds.numel() >= row_count
                  && tail_denominator.numel() == row_count
                  && tail_numerator.numel()
                      == static_cast<int64_t>(row_count) * head_dim,
              "conditional-tail shapes are invalid");
  auto query_c = query.contiguous();
  auto key_c = key.contiguous();
  auto value_c = value.contiguous();
  auto indices_c = candidate_indices.contiguous();
  auto counts_c = candidate_counts.contiguous();
  auto thresholds_c = thresholds.contiguous();
  auto denominator_c = tail_denominator.contiguous();
  auto numerator_c = tail_numerator.contiguous();
  auto output = torch::empty_like(query_c);
  int split_count = candidate_capacity <= 4096 ? 8 : 4;
  int threads = 128;
  while ((candidate_capacity + split_count - 1) / split_count + threads
         > 12 * 1024) {
    split_count *= 2;
  }
  auto partial_output = torch::empty(
      {row_count, split_count, head_dim},
      query_c.options().dtype(at::kFloat));
  auto partial_maximum = torch::empty(
      {row_count, split_count}, query_c.options().dtype(at::kFloat));
  auto partial_sum = torch::empty_like(partial_maximum);
  int maximum_local_count =
      (candidate_capacity + 1 + split_count - 1) / split_count;
  size_t shared_bytes = static_cast<size_t>(
      maximum_local_count + threads) * sizeof(float);
  TORCH_CHECK(shared_bytes <= 48 * 1024,
              "candidate capacity exceeds the kernel shared-memory limit");
  c10::cuda::CUDAGuard device_guard(query.device());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "qksieve_condtail_attention_forward",
      [&] {
        qksieve_valuesketch_exact_split_kernel<scalar_t><<<
            row_count * split_count, threads, shared_bytes,
            at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<scalar_t>(),
                key_c.data_ptr<scalar_t>(),
                value_c.data_ptr<scalar_t>(),
                indices_c.data_ptr<int64_t>(),
                counts_c.data_ptr<int64_t>(),
                partial_output.data_ptr<float>(),
                partial_maximum.data_ptr<float>(),
                partial_sum.data_ptr<float>(),
                query_head_count,
                kv_head_count,
                key_count,
                nullptr,
                candidate_capacity,
                head_dim,
                split_count,
                key_c.stride(0),
                key_c.stride(1),
                value_c.stride(0),
                value_c.stride(1),
                static_cast<float>(scaling));
        qksieve_condtail_reduce_kernel<scalar_t><<<
            row_count, threads, 0,
            at::cuda::getCurrentCUDAStream()>>>(
                partial_output.data_ptr<float>(),
                partial_maximum.data_ptr<float>(),
                partial_sum.data_ptr<float>(),
                thresholds_c.data_ptr<float>(),
                denominator_c.data_ptr<float>(),
                numerator_c.data_ptr<float>(),
                output.data_ptr<scalar_t>(),
                head_dim,
                split_count,
                static_cast<float>(scaling));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qksieve_condtail_attention_shared_gqa_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor tail_denominator,
    torch::Tensor tail_numerator,
    double scaling) {
  TORCH_CHECK(
      query.is_cuda() && key.is_cuda() && value.is_cuda()
          && candidate_indices.is_cuda() && candidate_counts.is_cuda(),
      "inputs must be CUDA tensors");
  TORCH_CHECK(query.dim() == 3 && key.dim() == 4 && value.dim() == 4,
              "query/key/value ranks are invalid");
  TORCH_CHECK(key.sizes() == value.sizes(),
              "Key and Value tensors must have identical shapes");
  TORCH_CHECK(query.scalar_type() == key.scalar_type()
                  && query.scalar_type() == value.scalar_type(),
              "Q/K/V tensors must share a dtype");
  TORCH_CHECK(candidate_indices.scalar_type() == at::kLong
                  && candidate_counts.scalar_type() == at::kLong,
              "candidate tensors must be int64");
  TORCH_CHECK(thresholds.scalar_type() == at::kFloat
                  && tail_denominator.scalar_type() == at::kFloat
                  && tail_numerator.scalar_type() == at::kFloat,
              "tail statistics must be float32");
  int batch_count = static_cast<int>(query.size(0));
  int query_head_count = static_cast<int>(query.size(1));
  int kv_head_count = static_cast<int>(key.size(1));
  int key_count = static_cast<int>(key.size(2));
  int head_dim = static_cast<int>(query.size(2));
  int candidate_capacity = static_cast<int>(candidate_indices.size(2));
  int query_groups = query_head_count / kv_head_count;
  int kv_row_count = batch_count * kv_head_count;
  int row_count = batch_count * query_head_count;
  TORCH_CHECK(query_head_count % kv_head_count == 0 && query_groups == 4,
              "shared consumer currently requires GQA group size four");
  TORCH_CHECK(head_dim == key.size(3), "head dimensions do not match");
  TORCH_CHECK(candidate_indices.dim() == 3
                  && candidate_indices.size(0) == batch_count
                  && candidate_indices.size(1) == kv_head_count
                  && candidate_indices.numel()
                      == static_cast<int64_t>(kv_row_count)
                          * candidate_capacity
                  && candidate_counts.numel() == kv_row_count,
              "shared candidate shapes are invalid");
  TORCH_CHECK(thresholds.numel() >= row_count
                  && tail_denominator.numel() == row_count
                  && tail_numerator.numel()
                      == static_cast<int64_t>(row_count) * head_dim,
              "conditional-tail shapes are invalid");
  auto query_c = query.contiguous();
  auto key_c = key.contiguous();
  auto value_c = value.contiguous();
  auto indices_c = candidate_indices.contiguous();
  auto counts_c = candidate_counts.contiguous();
  auto thresholds_c = thresholds.contiguous();
  auto denominator_c = tail_denominator.contiguous();
  auto numerator_c = tail_numerator.contiguous();
  auto output = torch::empty_like(query_c);
  int split_count = candidate_capacity <= 4096 ? 8 : 4;
  int threads = 128;
  while ((candidate_capacity + split_count - 1) / split_count + threads
         > 12 * 1024) {
    split_count *= 2;
  }
  auto partial_output = torch::empty(
      {row_count, split_count, head_dim},
      query_c.options().dtype(at::kFloat));
  auto partial_maximum = torch::empty(
      {row_count, split_count}, query_c.options().dtype(at::kFloat));
  auto partial_sum = torch::empty_like(partial_maximum);
  int maximum_local_count =
      (candidate_capacity + 1 + split_count - 1) / split_count;
  size_t shared_bytes = static_cast<size_t>(
      query_groups * maximum_local_count + threads) * sizeof(float);
  TORCH_CHECK(shared_bytes <= 48 * 1024,
              "candidate capacity exceeds the shared GQA kernel limit");
  c10::cuda::CUDAGuard device_guard(query.device());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "qksieve_condtail_attention_shared_gqa_forward",
      [&] {
        qksieve_condtail_exact_shared_gqa4_split_kernel<scalar_t><<<
            kv_row_count * split_count, threads, shared_bytes,
            at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<scalar_t>(),
                key_c.data_ptr<scalar_t>(),
                value_c.data_ptr<scalar_t>(),
                indices_c.data_ptr<int64_t>(),
                counts_c.data_ptr<int64_t>(),
                partial_output.data_ptr<float>(),
                partial_maximum.data_ptr<float>(),
                partial_sum.data_ptr<float>(),
                query_head_count,
                kv_head_count,
                key_count,
                candidate_capacity,
                head_dim,
                split_count,
                maximum_local_count,
                static_cast<float>(scaling));
        qksieve_condtail_reduce_kernel<scalar_t><<<
            row_count, threads, 0,
            at::cuda::getCurrentCUDAStream()>>>(
                partial_output.data_ptr<float>(),
                partial_maximum.data_ptr<float>(),
                partial_sum.data_ptr<float>(),
                thresholds_c.data_ptr<float>(),
                denominator_c.data_ptr<float>(),
                numerator_c.data_ptr<float>(),
                output.data_ptr<scalar_t>(),
                head_dim,
                split_count,
                static_cast<float>(scaling));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

void qksieve_valuesketch_attention_active_out(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor tail_denominator,
    torch::Tensor tail_coefficients,
    torch::Tensor value_mean,
    torch::Tensor value_basis,
    torch::Tensor active_key_count,
    torch::Tensor output,
    torch::Tensor partial_output,
    torch::Tensor partial_maximum,
    torch::Tensor partial_sum,
    double scaling,
    double tail_alpha) {
  TORCH_CHECK(
      query.is_cuda() && key.is_cuda() && value.is_cuda()
          && candidate_indices.is_cuda() && candidate_counts.is_cuda()
          && thresholds.is_cuda() && tail_denominator.is_cuda()
          && tail_coefficients.is_cuda() && value_mean.is_cuda()
          && value_basis.is_cuda() && active_key_count.is_cuda()
          && output.is_cuda() && partial_output.is_cuda()
          && partial_maximum.is_cuda() && partial_sum.is_cuda(),
      "active ValueSketch inputs and workspaces must be CUDA tensors");
  TORCH_CHECK(query.dim() == 3 && key.dim() == 4 && value.dim() == 4,
              "active ValueSketch query/key/value ranks are invalid");
  TORCH_CHECK(key.sizes() == value.sizes(),
              "active Key and Value tensors must have identical shapes");
  TORCH_CHECK(query.scalar_type() == key.scalar_type()
                  && query.scalar_type() == value.scalar_type()
                  && query.scalar_type() == value_mean.scalar_type()
                  && query.scalar_type() == value_basis.scalar_type()
                  && query.scalar_type() == output.scalar_type(),
              "active floating-point tensors and output must share a dtype");
  TORCH_CHECK(candidate_indices.scalar_type() == at::kLong
                  && candidate_counts.scalar_type() == at::kLong
                  && active_key_count.scalar_type() == at::kInt,
              "active candidate tensors must be int64 and length must be int32");
  TORCH_CHECK(thresholds.scalar_type() == at::kFloat
                  && tail_denominator.scalar_type() == at::kFloat
                  && tail_coefficients.scalar_type() == at::kFloat
                  && partial_output.scalar_type() == at::kFloat
                  && partial_maximum.scalar_type() == at::kFloat
                  && partial_sum.scalar_type() == at::kFloat,
              "active tail statistics and partial workspaces must be float32");
  TORCH_CHECK(query.is_contiguous() && candidate_indices.is_contiguous()
                  && candidate_counts.is_contiguous() && thresholds.is_contiguous()
                  && tail_denominator.is_contiguous()
                  && tail_coefficients.is_contiguous() && value_mean.is_contiguous()
                  && value_basis.is_contiguous() && active_key_count.is_contiguous()
                  && output.is_contiguous() && partial_output.is_contiguous()
                  && partial_maximum.is_contiguous() && partial_sum.is_contiguous(),
              "active ValueSketch tensors must be contiguous");
  TORCH_CHECK(active_key_count.numel() == 1,
              "active_key_count must contain one int32 value");
  int batch_count = static_cast<int>(query.size(0));
  int query_head_count = static_cast<int>(query.size(1));
  int kv_head_count = static_cast<int>(key.size(1));
  int key_count = static_cast<int>(key.size(2));
  int head_dim = static_cast<int>(query.size(2));
  int candidate_capacity = static_cast<int>(candidate_indices.size(2));
  int value_rank = static_cast<int>(tail_coefficients.size(2));
  int row_count = batch_count * query_head_count;
  int split_count = static_cast<int>(partial_output.size(1));
  TORCH_CHECK(key.stride(3) == 1 && key.stride(2) == head_dim
                  && value.stride(3) == 1 && value.stride(2) == head_dim,
              "active Key/Value token rows must be contiguous");
  TORCH_CHECK(split_count == 1 || split_count == 2 || split_count == 4
                  || split_count == 8 || split_count == 16,
              "active ValueSketch split count is invalid");
  TORCH_CHECK(query_head_count % kv_head_count == 0,
              "active Query heads must be divisible by KV heads");
  TORCH_CHECK(head_dim == key.size(3) && head_dim == value_mean.size(2),
              "active head dimensions do not match");
  TORCH_CHECK(value_rank > 0 && value_rank <= head_dim
                  && value_basis.size(3) == value_rank,
              "active Value basis rank is invalid");
  TORCH_CHECK(tail_alpha >= 0.0 && tail_alpha <= 1.0,
              "active Value-tail alpha must lie in [0, 1]");
  TORCH_CHECK(candidate_indices.numel()
                  == static_cast<int64_t>(row_count) * candidate_capacity
                  && candidate_counts.numel() == row_count,
              "active candidate shapes are invalid");
  TORCH_CHECK(thresholds.numel() >= row_count
                  && tail_denominator.numel() == row_count
                  && tail_coefficients.numel()
                      == static_cast<int64_t>(row_count) * value_rank,
              "active tail-statistic shapes are invalid");
  TORCH_CHECK(output.sizes() == query.sizes(),
              "active output workspace must match query shape");
  TORCH_CHECK(partial_output.dim() == 3
                  && partial_output.size(0) == row_count
                  && partial_output.size(1) == split_count
                  && partial_output.size(2) == head_dim
                  && partial_maximum.dim() == 2
                  && partial_maximum.size(0) == row_count
                  && partial_maximum.size(1) == split_count
                  && partial_sum.sizes() == partial_maximum.sizes(),
              "active partial workspace shapes are invalid");
  int threads = 128;
  int maximum_local_count =
      (candidate_capacity + 1 + split_count - 1) / split_count;
  size_t shared_bytes = static_cast<size_t>(
      maximum_local_count + threads) * sizeof(float);
  TORCH_CHECK(shared_bytes <= 48 * 1024,
              "active candidate capacity exceeds the shared-memory limit");
  c10::cuda::CUDAGuard device_guard(query.device());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "qksieve_valuesketch_attention_active_out",
      [&] {
        qksieve_valuesketch_exact_split_kernel<scalar_t><<<
            row_count * split_count, threads, shared_bytes,
            at::cuda::getCurrentCUDAStream()>>>(
                query.data_ptr<scalar_t>(),
                key.data_ptr<scalar_t>(),
                value.data_ptr<scalar_t>(),
                candidate_indices.data_ptr<int64_t>(),
                candidate_counts.data_ptr<int64_t>(),
                partial_output.data_ptr<float>(),
                partial_maximum.data_ptr<float>(),
                partial_sum.data_ptr<float>(),
                query_head_count,
                kv_head_count,
                key_count,
                active_key_count.data_ptr<int32_t>(),
                candidate_capacity,
                head_dim,
                split_count,
                key.stride(0),
                key.stride(1),
                value.stride(0),
                value.stride(1),
                static_cast<float>(scaling));
        qksieve_valuesketch_reduce_tail_kernel<scalar_t><<<
            row_count, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                partial_output.data_ptr<float>(),
                partial_maximum.data_ptr<float>(),
                partial_sum.data_ptr<float>(),
                thresholds.data_ptr<float>(),
                tail_denominator.data_ptr<float>(),
                tail_coefficients.data_ptr<float>(),
                value_mean.data_ptr<scalar_t>(),
                value_basis.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                query_head_count,
                kv_head_count,
                head_dim,
                value_rank,
                split_count,
                static_cast<float>(scaling),
                static_cast<float>(tail_alpha));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


@lru_cache(maxsize=1)
def load_extension() -> object:
    return load_inline(
        name="qksieve_valuesketch_attention_20260809_v16_contiguous_contract",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )


def attention_split_count(candidate_capacity: int) -> int:
    if candidate_capacity <= 0:
        raise ValueError("candidate_capacity must be positive")
    split_override = int(os.environ.get("QKSIEVE_VALUE_ATTENTION_SPLITS", "0"))
    if split_override not in {0, 1, 2, 4, 8, 16}:
        raise ValueError(
            "QKSIEVE_VALUE_ATTENTION_SPLITS must be one of 0,1,2,4,8,16"
        )
    split_count = (
        split_override
        if split_override > 0
        else 8
        if candidate_capacity <= 4096
        else 4
    )
    while (
        candidate_capacity + split_count - 1
    ) // split_count + 128 > 12 * 1024:
        split_count *= 2
    return split_count


def allocate_attention_workspace(
    query: torch.Tensor,
    candidate_capacity: int,
) -> dict[str, torch.Tensor]:
    if query.ndim != 3:
        raise ValueError("query must have shape [batch, query_heads, head_dim]")
    split_count = attention_split_count(candidate_capacity)
    row_count = int(query.shape[0]) * int(query.shape[1])
    head_dim = int(query.shape[2])
    output = torch.empty_like(query, memory_format=torch.contiguous_format)
    return {
        "output": output,
        "output_view": output.unsqueeze(1),
        "partial_output": torch.empty(
            row_count,
            split_count,
            head_dim,
            dtype=torch.float32,
            device=query.device,
        ),
        "partial_maximum": torch.empty(
            row_count,
            split_count,
            dtype=torch.float32,
            device=query.device,
        ),
        "partial_sum": torch.empty(
            row_count,
            split_count,
            dtype=torch.float32,
            device=query.device,
        ),
    }


def exact_selected_plus_tail(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_counts: torch.Tensor,
    thresholds: torch.Tensor,
    tail_denominator: torch.Tensor,
    tail_coefficients: torch.Tensor,
    value_mean: torch.Tensor,
    value_basis: torch.Tensor,
    scaling: float,
    tail_alpha: float = 1.0,
) -> torch.Tensor:
    output = load_extension().qksieve_valuesketch_attention_forward(
        query,
        key,
        value,
        candidate_indices,
        candidate_counts,
        thresholds,
        tail_denominator,
        tail_coefficients,
        value_mean,
        value_basis,
        float(scaling),
        float(tail_alpha),
    )
    return output[:, None, :, :].contiguous()


def exact_selected_plus_tail_out(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_counts: torch.Tensor,
    thresholds: torch.Tensor,
    tail_denominator: torch.Tensor,
    tail_coefficients: torch.Tensor,
    value_mean: torch.Tensor,
    value_basis: torch.Tensor,
    workspace: dict[str, torch.Tensor],
    scaling: float,
    tail_alpha: float = 1.0,
) -> torch.Tensor:
    load_extension().qksieve_valuesketch_attention_out(
        query,
        key,
        value,
        candidate_indices,
        candidate_counts,
        thresholds,
        tail_denominator,
        tail_coefficients,
        value_mean,
        value_basis,
        workspace["output"],
        workspace["partial_output"],
        workspace["partial_maximum"],
        workspace["partial_sum"],
        float(scaling),
        float(tail_alpha),
    )
    return workspace["output_view"]


def exact_selected_plus_tail_tiled_out(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_counts: torch.Tensor,
    thresholds: torch.Tensor,
    tail_denominator: torch.Tensor,
    tail_coefficients: torch.Tensor,
    value_mean: torch.Tensor,
    value_basis: torch.Tensor,
    workspace: dict[str, torch.Tensor],
    scaling: float,
    tail_alpha: float = 1.0,
) -> torch.Tensor:
    load_extension().qksieve_valuesketch_attention_tiled_out(
        query,
        key,
        value,
        candidate_indices,
        candidate_counts,
        thresholds,
        tail_denominator,
        tail_coefficients,
        value_mean,
        value_basis,
        workspace["output"],
        workspace["partial_output"],
        workspace["partial_maximum"],
        workspace["partial_sum"],
        float(scaling),
        float(tail_alpha),
    )
    return workspace["output_view"]


def exact_selected_plus_conditional_tail(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_counts: torch.Tensor,
    thresholds: torch.Tensor,
    tail_denominator: torch.Tensor,
    tail_numerator: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    output = load_extension().qksieve_condtail_attention_forward(
        query,
        key,
        value,
        candidate_indices,
        candidate_counts,
        thresholds,
        tail_denominator,
        tail_numerator,
        float(scaling),
    )
    return output[:, None, :, :].contiguous()


def exact_shared_gqa_selected_plus_conditional_tail(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_counts: torch.Tensor,
    thresholds: torch.Tensor,
    tail_denominator: torch.Tensor,
    tail_numerator: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    output = load_extension().qksieve_condtail_attention_shared_gqa_forward(
        query,
        key,
        value,
        candidate_indices,
        candidate_counts,
        thresholds,
        tail_denominator,
        tail_numerator,
        float(scaling),
    )
    return output[:, None, :, :].contiguous()


def reduce_conditional_tail_moments(
    tail_block_denominator: torch.Tensor,
    tail_weighted_x: torch.Tensor,
    mean_x: torch.Tensor,
    mean_v: torch.Tensor,
    linear_map: torch.Tensor,
) -> torch.Tensor:
    return load_extension().qksieve_condtail_reduce_moments(
        tail_block_denominator,
        tail_weighted_x,
        mean_x,
        mean_v,
        linear_map,
    )


def append_int4_out(
    value_history: torch.Tensor,
    value_mean: torch.Tensor,
    value_basis: torch.Tensor,
    value_minimum: torch.Tensor,
    value_scale: torch.Tensor,
    packed_codes: torch.Tensor,
    input_start: int,
    input_stop: int,
    value_block_size: int,
) -> None:
    load_extension().qksieve_valuesketch_append_int4_out(
        value_history,
        value_mean,
        value_basis,
        value_minimum,
        value_scale,
        packed_codes,
        int(input_start),
        int(input_stop),
        int(value_block_size),
    )
