from __future__ import annotations

from functools import lru_cache

import torch
from torch.utils.cpp_extension import load_inline


CPP_SOURCE = r"""
#include <torch/extension.h>

void qksieve_masked_valuesketch_attention_out(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor selection_masks,
    torch::Tensor thresholds,
    torch::Tensor tail_denominator,
    torch::Tensor tail_coefficients,
    torch::Tensor value_mean,
    torch::Tensor value_basis,
    torch::Tensor output,
    torch::Tensor partial_output,
    torch::Tensor partial_maximum,
    torch::Tensor partial_sum,
    int64_t split_tokens,
    double scaling,
    double tail_alpha);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "qksieve_masked_valuesketch_attention_out",
      &qksieve_masked_valuesketch_attention_out,
      "Exact GQA attention driven directly by QKSieve selection masks");
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
#include <cstdint>

template <typename scalar_t>
__global__ void qksieve_masked_exact_gqa4_split_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const int32_t* __restrict__ selection_masks,
    float* __restrict__ partial_output,
    float* __restrict__ partial_maximum,
    float* __restrict__ partial_sum,
    int query_head_count,
    int kv_head_count,
    int key_count,
    int history_count,
    int mask_stride_words,
    int head_dim,
    int split_tokens,
    int split_count,
    int maximum_local_count,
    float scaling) {
  constexpr int query_groups = 4;
  extern __shared__ unsigned char shared_raw[];
  int32_t* token_indices = reinterpret_cast<int32_t*>(shared_raw);
  uint8_t* memberships = reinterpret_cast<uint8_t*>(
      token_indices + maximum_local_count);
  uintptr_t score_address = reinterpret_cast<uintptr_t>(
      memberships + maximum_local_count);
  score_address = (score_address + alignof(float) - 1)
      & ~(static_cast<uintptr_t>(alignof(float) - 1));
  float* scores = reinterpret_cast<float*>(score_address);
  float* reduction = scores + query_groups * maximum_local_count;
  __shared__ int local_count_shared;

  int batch_kv = blockIdx.x / split_count;
  int split = blockIdx.x - batch_kv * split_count;
  int batch = batch_kv / kv_head_count;
  int kv_head = batch_kv - batch * kv_head_count;
  int query_head_base = batch * query_head_count + kv_head * query_groups;
  int token_start = split * split_tokens;
  int token_stop = min(token_start + split_tokens, history_count);
  int word_start = token_start >> 5;
  int word_stop = (token_stop + 31) >> 5;
  int tid = threadIdx.x;

  // One thread performs a deterministic compaction inside this CUDA block.
  // The list never leaves shared memory, so no global candidate array is built.
  if (tid == 0) {
    int local_count = 0;
    for (int word = word_start; word < word_stop; ++word) {
      uint32_t group_words[query_groups];
      uint32_t union_word = 0u;
#pragma unroll
      for (int group = 0; group < query_groups; ++group) {
        int row = query_head_base + group;
        group_words[group] = static_cast<uint32_t>(selection_masks[
            static_cast<int64_t>(row) * mask_stride_words + word]);
        union_word |= group_words[group];
      }
      while (union_word != 0u) {
        int bit = __ffs(static_cast<int>(union_word)) - 1;
        int token = word * 32 + bit;
        if (token >= token_start && token < token_stop) {
          uint8_t membership = 0u;
#pragma unroll
          for (int group = 0; group < query_groups; ++group) {
            membership |= static_cast<uint8_t>(
                ((group_words[group] >> bit) & 1u) << group);
          }
          token_indices[local_count] = token;
          memberships[local_count] = membership;
          ++local_count;
        }
        union_word &= union_word - 1u;
      }
    }
    if (split == split_count - 1) {
      token_indices[local_count] = key_count - 1;
      memberships[local_count] = 0x0f;
      ++local_count;
    }
    local_count_shared = local_count;
  }
  __syncthreads();
  int local_count = local_count_shared;

  const scalar_t* query_base = query
      + static_cast<int64_t>(query_head_base) * head_dim;
  const scalar_t* key_base = key
      + static_cast<int64_t>(batch_kv) * key_count * head_dim;
  const scalar_t* value_base = value
      + static_cast<int64_t>(batch_kv) * key_count * head_dim;

  for (int local = tid; local < local_count; local += blockDim.x) {
    int token = token_indices[local];
    uint8_t membership = memberships[local];
    float accumulators[query_groups] = {0.0f, 0.0f, 0.0f, 0.0f};
    const scalar_t* key_row = key_base
        + static_cast<int64_t>(token) * head_dim;
    for (int dimension = 0; dimension < head_dim; ++dimension) {
      float key_element = static_cast<float>(key_row[dimension]);
#pragma unroll
      for (int group = 0; group < query_groups; ++group) {
        if ((membership & (1u << group)) != 0u) {
          accumulators[group] += static_cast<float>(
              query_base[group * head_dim + dimension]) * key_element;
        }
      }
    }
#pragma unroll
    for (int group = 0; group < query_groups; ++group) {
      scores[group * maximum_local_count + local] =
          (membership & (1u << group)) != 0u
          ? accumulators[group] * scaling
          : -INFINITY;
    }
  }
  __syncthreads();

#pragma unroll
  for (int group = 0; group < query_groups; ++group) {
    float local_maximum = -INFINITY;
    for (int local = tid; local < local_count; local += blockDim.x) {
      local_maximum = fmaxf(
          local_maximum,
          scores[group * maximum_local_count + local]);
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
      float score = scores[group * maximum_local_count + local];
      float weight = isfinite(score) ? expf(score - maximum) : 0.0f;
      scores[group * maximum_local_count + local] = weight;
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
      int partial = row * split_count + split;
      partial_maximum[partial] = maximum;
      partial_sum[partial] = reduction[0];
    }
    __syncthreads();
  }

  for (int dimension = tid; dimension < head_dim;
       dimension += blockDim.x) {
    float accumulators[query_groups] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int local = 0; local < local_count; ++local) {
      int token = token_indices[local];
      float value_element = static_cast<float>(value_base[
          static_cast<int64_t>(token) * head_dim + dimension]);
#pragma unroll
      for (int group = 0; group < query_groups; ++group) {
        accumulators[group] +=
            scores[group * maximum_local_count + local] * value_element;
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
  const scalar_t* mean_row = value_mean
      + static_cast<int64_t>(batch_kv) * head_dim;
  const scalar_t* basis_row = value_basis
      + static_cast<int64_t>(batch_kv) * head_dim * value_rank;
  const float* coefficient_row = tail_coefficients
      + static_cast<int64_t>(row) * value_rank;
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

void qksieve_masked_valuesketch_attention_out(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor selection_masks,
    torch::Tensor thresholds,
    torch::Tensor tail_denominator,
    torch::Tensor tail_coefficients,
    torch::Tensor value_mean,
    torch::Tensor value_basis,
    torch::Tensor output,
    torch::Tensor partial_output,
    torch::Tensor partial_maximum,
    torch::Tensor partial_sum,
    int64_t split_tokens,
    double scaling,
    double tail_alpha) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda(),
              "query, key, and value must be CUDA tensors");
  TORCH_CHECK(selection_masks.is_cuda() && thresholds.is_cuda(),
              "selection masks and thresholds must be CUDA tensors");
  TORCH_CHECK(tail_denominator.is_cuda() && tail_coefficients.is_cuda()
                  && value_mean.is_cuda() && value_basis.is_cuda(),
              "ValueSketch tensors must be CUDA tensors");
  TORCH_CHECK(output.is_cuda() && partial_output.is_cuda()
                  && partial_maximum.is_cuda() && partial_sum.is_cuda(),
              "output workspaces must be CUDA tensors");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous()
                  && value.is_contiguous() && selection_masks.is_contiguous()
                  && thresholds.is_contiguous()
                  && tail_denominator.is_contiguous()
                  && tail_coefficients.is_contiguous()
                  && value_mean.is_contiguous() && value_basis.is_contiguous()
                  && output.is_contiguous() && partial_output.is_contiguous()
                  && partial_maximum.is_contiguous()
                  && partial_sum.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(query.dim() == 3 && key.dim() == 4 && value.dim() == 4,
              "query must be [B,QH,D], key/value must be [B,KVH,K,D]");
  TORCH_CHECK(query.scalar_type() == key.scalar_type()
                  && query.scalar_type() == value.scalar_type()
                  && query.scalar_type() == value_mean.scalar_type()
                  && query.scalar_type() == value_basis.scalar_type()
                  && output.scalar_type() == query.scalar_type(),
              "floating-point tensor dtypes must match");
  TORCH_CHECK(selection_masks.scalar_type() == at::kInt,
              "selection_masks must be int32");
  TORCH_CHECK(thresholds.scalar_type() == at::kFloat
                  && tail_denominator.scalar_type() == at::kFloat
                  && tail_coefficients.scalar_type() == at::kFloat
                  && partial_output.scalar_type() == at::kFloat
                  && partial_maximum.scalar_type() == at::kFloat
                  && partial_sum.scalar_type() == at::kFloat,
              "threshold, tail, and partial tensors must be float32");
  int batch_count = static_cast<int>(query.size(0));
  int query_head_count = static_cast<int>(query.size(1));
  int head_dim = static_cast<int>(query.size(2));
  int kv_head_count = static_cast<int>(key.size(1));
  int key_count = static_cast<int>(key.size(2));
  int history_count = key_count - 1;
  int value_rank = static_cast<int>(tail_coefficients.size(2));
  TORCH_CHECK(query_head_count == 4 * kv_head_count,
              "this kernel requires GQA-4");
  TORCH_CHECK(head_dim == 128, "this kernel currently requires head_dim=128");
  TORCH_CHECK(key.sizes() == value.sizes(), "key/value shapes must match");
  TORCH_CHECK(static_cast<int>(key.size(0)) == batch_count
                  && static_cast<int>(key.size(3)) == head_dim,
              "query and key/value shapes are incompatible");
  TORCH_CHECK(split_tokens > 0 && split_tokens <= 2048,
              "split_tokens must be in [1,2048]");
  int split_count = (history_count + split_tokens - 1) / split_tokens;
  int words_per_row = (history_count + 31) / 32;
  TORCH_CHECK(selection_masks.dim() == 3
                  && selection_masks.size(0) == batch_count
                  && selection_masks.size(1) == query_head_count
                  && selection_masks.size(2) >= words_per_row,
              "selection mask shape does not match history length");
  int mask_stride_words = static_cast<int>(selection_masks.size(2));
  int row_count = batch_count * query_head_count;
  int kv_row_count = batch_count * kv_head_count;
  TORCH_CHECK(thresholds.numel() == row_count
                  && tail_denominator.numel() == row_count
                  && tail_coefficients.numel()
                      == static_cast<int64_t>(row_count) * value_rank,
              "tail tensor shapes are invalid");
  TORCH_CHECK(value_mean.numel()
                  == static_cast<int64_t>(batch_count) * kv_head_count
                      * head_dim
                  && value_basis.numel()
                      == static_cast<int64_t>(batch_count) * kv_head_count
                          * head_dim * value_rank,
              "ValueSketch basis shapes are invalid");
  TORCH_CHECK(output.numel()
                  == static_cast<int64_t>(row_count) * head_dim
                  && partial_output.numel()
                      == static_cast<int64_t>(row_count) * split_count
                          * head_dim
                  && partial_maximum.numel()
                      == static_cast<int64_t>(row_count) * split_count
                  && partial_sum.numel()
                      == static_cast<int64_t>(row_count) * split_count,
              "output workspace shapes are invalid");

  int threads = 128;
  int maximum_local_count = static_cast<int>(split_tokens) + 1;
  size_t shared_bytes =
      static_cast<size_t>(maximum_local_count) * sizeof(int32_t)
      + static_cast<size_t>(maximum_local_count) * sizeof(uint8_t)
      + 3
      + static_cast<size_t>(4 * maximum_local_count + threads)
          * sizeof(float);
  TORCH_CHECK(shared_bytes <= 48 * 1024,
              "requested split size exceeds the shared-memory limit");
  c10::cuda::CUDAGuard device_guard(query.device());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "qksieve_masked_valuesketch_attention_out",
      [&] {
        qksieve_masked_exact_gqa4_split_kernel<scalar_t><<<
            kv_row_count * split_count,
            threads,
            shared_bytes,
            at::cuda::getCurrentCUDAStream()>>>(
                query.data_ptr<scalar_t>(),
                key.data_ptr<scalar_t>(),
                value.data_ptr<scalar_t>(),
                selection_masks.data_ptr<int32_t>(),
                partial_output.data_ptr<float>(),
                partial_maximum.data_ptr<float>(),
                partial_sum.data_ptr<float>(),
                query_head_count,
                kv_head_count,
                key_count,
                history_count,
                mask_stride_words,
                head_dim,
                static_cast<int>(split_tokens),
                split_count,
                maximum_local_count,
                static_cast<float>(scaling));
        qksieve_valuesketch_reduce_tail_kernel<scalar_t><<<
            row_count,
            threads,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
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
        name="qksieve_masked_valuesketch_attention_20260807_v1",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )


def split_count(history_tokens: int, split_tokens: int = 2048) -> int:
    if history_tokens <= 0:
        raise ValueError("history_tokens must be positive")
    if not 0 < split_tokens <= 2048:
        raise ValueError("split_tokens must be in [1, 2048]")
    return (history_tokens + split_tokens - 1) // split_tokens


def allocate_workspace(
    query: torch.Tensor,
    history_tokens: int,
    split_tokens: int = 2048,
) -> dict[str, torch.Tensor]:
    if query.ndim != 3:
        raise ValueError("query must have shape [batch, query_heads, head_dim]")
    splits = split_count(history_tokens, split_tokens)
    rows = int(query.shape[0]) * int(query.shape[1])
    head_dim = int(query.shape[2])
    output = torch.empty_like(query, memory_format=torch.contiguous_format)
    return {
        "output": output,
        "output_view": output.unsqueeze(1),
        "partial_output": torch.empty(
            rows,
            splits,
            head_dim,
            dtype=torch.float32,
            device=query.device,
        ),
        "partial_maximum": torch.empty(
            rows, splits, dtype=torch.float32, device=query.device
        ),
        "partial_sum": torch.empty(
            rows, splits, dtype=torch.float32, device=query.device
        ),
    }


def exact_masked_gqa4_plus_tail_out(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selection_masks: torch.Tensor,
    thresholds: torch.Tensor,
    tail_denominator: torch.Tensor,
    tail_coefficients: torch.Tensor,
    value_mean: torch.Tensor,
    value_basis: torch.Tensor,
    workspace: dict[str, torch.Tensor],
    scaling: float,
    tail_alpha: float = 1.0,
    split_tokens: int = 2048,
) -> torch.Tensor:
    load_extension().qksieve_masked_valuesketch_attention_out(
        query,
        key,
        value,
        selection_masks,
        thresholds,
        tail_denominator,
        tail_coefficients,
        value_mean,
        value_basis,
        workspace["output"],
        workspace["partial_output"],
        workspace["partial_maximum"],
        workspace["partial_sum"],
        int(split_tokens),
        float(scaling),
        float(tail_alpha),
    )
    return workspace["output_view"]
