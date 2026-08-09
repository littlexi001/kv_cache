from __future__ import annotations

from functools import lru_cache

import torch
from torch.utils.cpp_extension import load_inline


CPP_SOURCE = r"""
#include <torch/extension.h>
#include <vector>

torch::Tensor hier841_core_scores_forward(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor core_int8,
    torch::Tensor middle_int4,
    torch::Tensor key_scales);

torch::Tensor hier841_tail_scores_forward(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor tail_sign,
    torch::Tensor key_scales,
    torch::Tensor candidate_indices);

std::vector<torch::Tensor> hier841_sampled_threshold_compact_forward(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor core_int8,
    torch::Tensor middle_int4,
    torch::Tensor tail_sign,
    torch::Tensor key_scales,
    int64_t sample_count,
    double candidate_fraction,
    double selected_fraction,
    int64_t candidate_capacity);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "hier841_core_scores_forward",
      &hier841_core_scores_forward,
      "Packed hierarchical 8/4 core scores");
  m.def(
      "hier841_tail_scores_forward",
      &hier841_tail_scores_forward,
      "Packed hierarchical 1-bit candidate-tail scores");
  m.def(
      "hier841_sampled_threshold_compact_forward",
      &hier841_sampled_threshold_compact_forward,
      "Fused sampled-threshold hierarchical scan and compaction");
}
"""


CUDA_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

__device__ __forceinline__ int pack_signed_bytes(
    int a,
    int b,
    int c,
    int d) {
  return (a & 0xff)
      | ((b & 0xff) << 8)
      | ((c & 0xff) << 16)
      | ((d & 0xff) << 24);
}

__device__ __forceinline__ int sign_extend_int4(int value) {
  return value < 8 ? value : value - 16;
}

template <typename scale_t>
__device__ __forceinline__ float hier841_core_score_one(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const int8_t* __restrict__ core_int8,
    const uint8_t* __restrict__ middle_int4,
    const scale_t* __restrict__ key_scales,
    int batch,
    int kv_head,
    int query_group,
    int kv_head_count,
    int query_groups,
    int history_count,
    int token) {
  int64_t query_row = (
      (static_cast<int64_t>(batch) * kv_head_count + kv_head)
      * query_groups + query_group);
  const int8_t* query = query_codes + query_row * 128;
  const scale_t* qscale = query_scales + query_row * 8;
  int64_t key_token = (
      (static_cast<int64_t>(batch) * kv_head_count + kv_head)
      * history_count + token);
  const int8_t* core = core_int8 + key_token * 16;
  const scale_t* token_scales = key_scales + key_token * 4;
  int core_dot = 0;
#pragma unroll
  for (int word = 0; word < 4; ++word) {
    core_dot = __dp4a(
        reinterpret_cast<const int*>(core)[word],
        reinterpret_cast<const int*>(query)[word],
        core_dot);
  }
  float score = static_cast<float>(core_dot)
      * static_cast<float>(qscale[0])
      * static_cast<float>(token_scales[0]);
#pragma unroll
  for (int group = 0; group < 2; ++group) {
    int64_t packed_base = (
        (((static_cast<int64_t>(batch) * kv_head_count + kv_head) * 2
           + group)
          * history_count + token)
        * 8);
    const uint8_t* packed = middle_int4 + packed_base;
    const int8_t* qband = query + 16 * (group + 1);
    int dot = 0;
#pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
      uint8_t byte0 = packed[2 * chunk];
      uint8_t byte1 = packed[2 * chunk + 1];
      int key_word = pack_signed_bytes(
          sign_extend_int4(byte0 & 0xf),
          sign_extend_int4(byte0 >> 4),
          sign_extend_int4(byte1 & 0xf),
          sign_extend_int4(byte1 >> 4));
      dot = __dp4a(
          key_word,
          reinterpret_cast<const int*>(qband)[chunk],
          dot);
    }
    score += static_cast<float>(dot)
        * static_cast<float>(qscale[group + 1])
        * static_cast<float>(token_scales[group + 1]);
  }
  return score;
}

template <typename scale_t>
__device__ __forceinline__ float hier841_tail_score_one(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ tail_sign,
    const scale_t* __restrict__ key_scales,
    int batch,
    int kv_head,
    int query_group,
    int kv_head_count,
    int query_groups,
    int history_count,
    int token) {
  int64_t query_row = (
      (static_cast<int64_t>(batch) * kv_head_count + kv_head)
      * query_groups + query_group);
  const int8_t* query = query_codes + query_row * 128;
  const scale_t* qscale = query_scales + query_row * 8;
  int64_t key_token = (
      (static_cast<int64_t>(batch) * kv_head_count + kv_head)
      * history_count + token);
  const uint8_t* packed = tail_sign + key_token * 10;
  float tail_scale = static_cast<float>(key_scales[key_token * 4 + 3]);
  float score = 0.0f;
#pragma unroll
  for (int group = 0; group < 5; ++group) {
    const int8_t* qband = query + 16 * (group + 3);
    int dot = 0;
#pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
      uint8_t bits = packed[2 * group + chunk / 2];
      bits = chunk % 2 == 0 ? (bits & 0xf) : (bits >> 4);
      int key_word = pack_signed_bytes(
          (bits & 0x1) ? 1 : -1,
          (bits & 0x2) ? 1 : -1,
          (bits & 0x4) ? 1 : -1,
          (bits & 0x8) ? 1 : -1);
      dot = __dp4a(
          key_word,
          reinterpret_cast<const int*>(qband)[chunk],
          dot);
    }
    score += static_cast<float>(dot)
        * static_cast<float>(qscale[group + 3])
        * tail_scale;
  }
  return score;
}

template <typename scale_t>
__global__ void hier841_sample_thresholds_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const int8_t* __restrict__ core_int8,
    const uint8_t* __restrict__ middle_int4,
    const uint8_t* __restrict__ tail_sign,
    const scale_t* __restrict__ key_scales,
    float* __restrict__ thresholds,
    int kv_head_count,
    int query_groups,
    int history_count,
    int sample_count,
    int candidate_keep,
    int selected_keep) {
  int row = blockIdx.x;
  int thread = threadIdx.x;
  int query_head_count = kv_head_count * query_groups;
  int batch = row / query_head_count;
  int query_head = row - batch * query_head_count;
  int kv_head = query_head / query_groups;
  int query_group = query_head - kv_head * query_groups;
  extern __shared__ float shared[];
  float* core_samples = shared;
  float* full_samples = shared + 256;
  float core = -INFINITY;
  float full = -INFINITY;
  if (thread < sample_count) {
    int segment = max(1, history_count / sample_count);
    int phase = (row * 131 + 17) % segment;
    int64_t centered = (
        (static_cast<int64_t>(2 * thread + 1) * history_count)
        / (2 * sample_count));
    int token = static_cast<int>((centered + phase) % history_count);
    core = hier841_core_score_one(
        query_codes,
        query_scales,
        core_int8,
        middle_int4,
        key_scales,
        batch,
        kv_head,
        query_group,
        kv_head_count,
        query_groups,
        history_count,
        token);
    full = core + hier841_tail_score_one(
        query_codes,
        query_scales,
        tail_sign,
        key_scales,
        batch,
        kv_head,
        query_group,
        kv_head_count,
        query_groups,
        history_count,
        token);
  }
  core_samples[thread] = core;
  full_samples[thread] = full;
  __syncthreads();

  for (int size = 2; size <= 256; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      int other = thread ^ stride;
      if (other > thread) {
        bool ascending = (thread & size) == 0;
        float core_left = core_samples[thread];
        float core_right = core_samples[other];
        float full_left = full_samples[thread];
        float full_right = full_samples[other];
        if ((core_left > core_right) == ascending) {
          core_samples[thread] = core_right;
          core_samples[other] = core_left;
        }
        if ((full_left > full_right) == ascending) {
          full_samples[thread] = full_right;
          full_samples[other] = full_left;
        }
      }
      __syncthreads();
    }
  }
  if (thread == 0) {
    thresholds[row * 2] = core_samples[256 - candidate_keep];
    thresholds[row * 2 + 1] = full_samples[256 - selected_keep];
  }
}

template <typename scale_t>
__global__ void hier841_threshold_compact_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const int8_t* __restrict__ core_int8,
    const uint8_t* __restrict__ middle_int4,
    const uint8_t* __restrict__ tail_sign,
    const scale_t* __restrict__ key_scales,
    const float* __restrict__ thresholds,
    int64_t* __restrict__ candidate_indices,
    float* __restrict__ candidate_scores,
    int64_t* __restrict__ candidate_counts,
    bool* __restrict__ overflow,
    int kv_head_count,
    int query_groups,
    int history_count,
    int candidate_capacity) {
  int token = blockIdx.y * blockDim.x + threadIdx.x;
  if (token >= history_count) {
    return;
  }
  int batch_kv = blockIdx.x;
  int batch = batch_kv / kv_head_count;
  int kv_head = batch_kv - batch * kv_head_count;
  int query_head_count = kv_head_count * query_groups;
  for (int query_group = 0; query_group < query_groups; ++query_group) {
    int query_head = kv_head * query_groups + query_group;
    int row = batch * query_head_count + query_head;
    float core = hier841_core_score_one(
        query_codes,
        query_scales,
        core_int8,
        middle_int4,
        key_scales,
        batch,
        kv_head,
        query_group,
        kv_head_count,
        query_groups,
        history_count,
        token);
    if (core < thresholds[row * 2]) {
      continue;
    }
    float full = core + hier841_tail_score_one(
        query_codes,
        query_scales,
        tail_sign,
        key_scales,
        batch,
        kv_head,
        query_group,
        kv_head_count,
        query_groups,
        history_count,
        token);
    if (full < thresholds[row * 2 + 1]) {
      continue;
    }
    unsigned long long* count = reinterpret_cast<unsigned long long*>(
        candidate_counts + row);
    unsigned long long slot = atomicAdd(count, 1ULL);
    if (slot < static_cast<unsigned long long>(candidate_capacity)) {
      int64_t output_offset = (
          static_cast<int64_t>(row) * candidate_capacity + slot);
      candidate_indices[output_offset] = token;
      candidate_scores[output_offset] = full;
    } else {
      overflow[row] = true;
    }
  }
}

__global__ void hier841_finalize_counts_kernel(
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

template <typename scale_t>
__global__ void hier841_core_scores_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const int8_t* __restrict__ core_int8,
    const uint8_t* __restrict__ middle_int4,
    const scale_t* __restrict__ key_scales,
    float* __restrict__ output,
    int kv_head_count,
    int query_groups,
    int history_count) {
  int token = blockIdx.y * blockDim.x + threadIdx.x;
  if (token >= history_count) {
    return;
  }
  int batch_kv = blockIdx.x;
  int batch = batch_kv / kv_head_count;
  int kv_head = batch_kv - batch * kv_head_count;
  int64_t key_token = (
      (static_cast<int64_t>(batch) * kv_head_count + kv_head)
      * history_count + token);
  const int8_t* core = core_int8 + key_token * 16;
  const scale_t* token_scales = key_scales + key_token * 4;
  int core_words[4];
#pragma unroll
  for (int word = 0; word < 4; ++word) {
    core_words[word] =
        reinterpret_cast<const int*>(core)[word];
  }

  for (int query_group = 0; query_group < query_groups; ++query_group) {
    int64_t query_row = (
        (static_cast<int64_t>(batch) * kv_head_count + kv_head)
        * query_groups + query_group);
    const int8_t* query = query_codes + query_row * 128;
    const scale_t* qscale = query_scales + query_row * 8;
    int core_dot = 0;
#pragma unroll
    for (int word = 0; word < 4; ++word) {
      core_dot = __dp4a(
          core_words[word],
          reinterpret_cast<const int*>(query)[word],
          core_dot);
    }
    float score = static_cast<float>(core_dot)
        * static_cast<float>(qscale[0])
        * static_cast<float>(token_scales[0]);

#pragma unroll
    for (int group = 0; group < 2; ++group) {
      int64_t packed_base = (
          (((static_cast<int64_t>(batch) * kv_head_count + kv_head) * 2
             + group)
            * history_count + token)
          * 8);
      const uint8_t* packed = middle_int4 + packed_base;
      const int8_t* qband = query + 16 * (group + 1);
      int dot = 0;
#pragma unroll
      for (int chunk = 0; chunk < 4; ++chunk) {
        uint8_t byte0 = packed[2 * chunk];
        uint8_t byte1 = packed[2 * chunk + 1];
        int key_word = pack_signed_bytes(
            sign_extend_int4(byte0 & 0xf),
            sign_extend_int4(byte0 >> 4),
            sign_extend_int4(byte1 & 0xf),
            sign_extend_int4(byte1 >> 4));
        dot = __dp4a(
            key_word,
            reinterpret_cast<const int*>(qband)[chunk],
            dot);
      }
      score += static_cast<float>(dot)
          * static_cast<float>(qscale[group + 1])
          * static_cast<float>(token_scales[group + 1]);
    }
    int query_head = kv_head * query_groups + query_group;
    int64_t output_offset = (
        (static_cast<int64_t>(batch) * kv_head_count * query_groups
         + query_head)
        * history_count + token);
    output[output_offset] = score;
  }
}

template <typename scale_t>
__global__ void hier841_tail_scores_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ tail_sign,
    const scale_t* __restrict__ key_scales,
    const int64_t* __restrict__ candidate_indices,
    float* __restrict__ output,
    int query_head_count,
    int kv_head_count,
    int query_groups,
    int history_count,
    int candidate_count,
    int total_candidates) {
  for (int flat = blockIdx.x * blockDim.x + threadIdx.x;
       flat < total_candidates;
       flat += blockDim.x * gridDim.x) {
    int candidate = flat % candidate_count;
    int row = flat / candidate_count;
    int batch = row / query_head_count;
    int query_head = row - batch * query_head_count;
    int kv_head = query_head / query_groups;
    int query_group = query_head - kv_head * query_groups;
    int64_t token = candidate_indices[flat];
    if (token < 0 || token >= history_count) {
      output[flat] = -INFINITY;
      continue;
    }
    int64_t query_row = (
        (static_cast<int64_t>(batch) * kv_head_count + kv_head)
        * query_groups + query_group);
    const int8_t* query = query_codes + query_row * 128;
    const scale_t* qscale = query_scales + query_row * 8;
    int64_t key_token = (
        (static_cast<int64_t>(batch) * kv_head_count + kv_head)
        * history_count + token);
    const uint8_t* packed = tail_sign + key_token * 10;
    float tail_scale = static_cast<float>(key_scales[key_token * 4 + 3]);
    float score = 0.0f;
#pragma unroll
    for (int group = 0; group < 5; ++group) {
      const int8_t* qband = query + 16 * (group + 3);
      int dot = 0;
#pragma unroll
      for (int chunk = 0; chunk < 4; ++chunk) {
        uint8_t bits = packed[2 * group + chunk / 2];
        bits = chunk % 2 == 0 ? (bits & 0xf) : (bits >> 4);
        int key_word = pack_signed_bytes(
            (bits & 0x1) ? 1 : -1,
            (bits & 0x2) ? 1 : -1,
            (bits & 0x4) ? 1 : -1,
            (bits & 0x8) ? 1 : -1);
        dot = __dp4a(
            key_word,
            reinterpret_cast<const int*>(qband)[chunk],
            dot);
      }
      score += static_cast<float>(dot)
          * static_cast<float>(qscale[group + 3])
          * tail_scale;
    }
    output[flat] = score;
  }
}

torch::Tensor hier841_core_scores_forward(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor core_int8,
    torch::Tensor middle_int4,
    torch::Tensor key_scales) {
  TORCH_CHECK(
      query_codes.is_cuda() && query_scales.is_cuda()
          && core_int8.is_cuda() && middle_int4.is_cuda()
          && key_scales.is_cuda(),
      "all inputs must be CUDA tensors");
  TORCH_CHECK(
      query_codes.scalar_type() == at::kChar
          && core_int8.scalar_type() == at::kChar,
      "query and 8-bit core codes must be int8");
  TORCH_CHECK(
      middle_int4.scalar_type() == at::kByte,
      "packed 4-bit middle codes must be uint8");
  TORCH_CHECK(
      query_scales.scalar_type() == key_scales.scalar_type(),
      "query and key scale dtypes must match");
  TORCH_CHECK(
      query_codes.dim() == 4 && query_codes.size(3) == 128,
      "query codes must have shape [batch, kv_heads, groups, 128]");
  TORCH_CHECK(
      query_scales.dim() == 4 && query_scales.size(3) == 8,
      "query scales must have shape [batch, kv_heads, groups, 8]");
  TORCH_CHECK(
      core_int8.dim() == 4 && core_int8.size(3) == 16,
      "8-bit core must have shape [batch, kv_heads, history, 16]");
  TORCH_CHECK(
      middle_int4.dim() == 5
          && middle_int4.size(2) == 2
          && middle_int4.size(4) == 8,
      "4-bit middle must have shape [batch, kv_heads, 2, history, 8]");
  TORCH_CHECK(
      key_scales.dim() == 4 && key_scales.size(3) == 4,
      "key scales must have shape [batch, kv_heads, history, 4]");
  TORCH_CHECK(
      query_codes.size(0) == core_int8.size(0)
          && query_codes.size(1) == core_int8.size(1),
      "batch/KV-head dimensions must match");
  int batch_count = static_cast<int>(query_codes.size(0));
  int kv_head_count = static_cast<int>(query_codes.size(1));
  int query_groups = static_cast<int>(query_codes.size(2));
  int history_count = static_cast<int>(core_int8.size(2));
  TORCH_CHECK(
      query_scales.size(0) == batch_count
          && query_scales.size(1) == kv_head_count
          && query_scales.size(2) == query_groups,
      "query-scale shape mismatch");
  TORCH_CHECK(
      middle_int4.size(0) == batch_count
          && middle_int4.size(1) == kv_head_count
          && middle_int4.size(3) == history_count,
      "middle-code shape mismatch");
  TORCH_CHECK(
      key_scales.size(0) == batch_count
          && key_scales.size(1) == kv_head_count
          && key_scales.size(2) == history_count,
      "key-scale shape mismatch");
  c10::cuda::CUDAGuard device_guard(query_codes.device());
  auto output = torch::empty(
      {batch_count, kv_head_count * query_groups, history_count},
      query_codes.options().dtype(at::kFloat));
  dim3 blocks(
      batch_count * kv_head_count,
      (history_count + 255) / 256);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_scales.scalar_type(),
      "hier841_core_scores_forward",
      [&] {
        hier841_core_scores_kernel<scalar_t><<<
            blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                core_int8.data_ptr<int8_t>(),
                middle_int4.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                output.data_ptr<float>(),
                kv_head_count,
                query_groups,
                history_count);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor hier841_tail_scores_forward(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor tail_sign,
    torch::Tensor key_scales,
    torch::Tensor candidate_indices) {
  TORCH_CHECK(
      query_codes.is_cuda() && query_scales.is_cuda()
          && tail_sign.is_cuda() && key_scales.is_cuda()
          && candidate_indices.is_cuda(),
      "all inputs must be CUDA tensors");
  TORCH_CHECK(
      query_codes.scalar_type() == at::kChar,
      "query codes must be int8");
  TORCH_CHECK(
      tail_sign.scalar_type() == at::kByte,
      "packed tail signs must be uint8");
  TORCH_CHECK(
      candidate_indices.scalar_type() == at::kLong,
      "candidate indices must be int64");
  TORCH_CHECK(
      query_codes.dim() == 4 && query_codes.size(3) == 128,
      "query codes must have shape [batch, kv_heads, groups, 128]");
  TORCH_CHECK(
      query_scales.dim() == 4 && query_scales.size(3) == 8,
      "query scales must have shape [batch, kv_heads, groups, 8]");
  TORCH_CHECK(
      tail_sign.dim() == 4 && tail_sign.size(3) == 10,
      "tail signs must have shape [batch, kv_heads, history, 10]");
  TORCH_CHECK(
      key_scales.dim() == 4 && key_scales.size(3) == 4,
      "key scales must have shape [batch, kv_heads, history, 4]");
  TORCH_CHECK(
      candidate_indices.dim() == 3,
      "candidate indices must have shape [batch, query_heads, count]");
  int batch_count = static_cast<int>(query_codes.size(0));
  int kv_head_count = static_cast<int>(query_codes.size(1));
  int query_groups = static_cast<int>(query_codes.size(2));
  int query_head_count = kv_head_count * query_groups;
  int history_count = static_cast<int>(tail_sign.size(2));
  int candidate_count = static_cast<int>(candidate_indices.size(2));
  TORCH_CHECK(
      candidate_indices.size(0) == batch_count
          && candidate_indices.size(1) == query_head_count,
      "candidate shape mismatch");
  TORCH_CHECK(
      query_scales.scalar_type() == key_scales.scalar_type(),
      "query and key scale dtypes must match");
  c10::cuda::CUDAGuard device_guard(query_codes.device());
  auto output = torch::empty(
      candidate_indices.sizes(),
      query_codes.options().dtype(at::kFloat));
  int total_candidates = static_cast<int>(candidate_indices.numel());
  int blocks = std::min(4096, (total_candidates + 255) / 256);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_scales.scalar_type(),
      "hier841_tail_scores_forward",
      [&] {
        hier841_tail_scores_kernel<scalar_t><<<
            blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                tail_sign.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                candidate_indices.data_ptr<int64_t>(),
                output.data_ptr<float>(),
                query_head_count,
                kv_head_count,
                query_groups,
                history_count,
                candidate_count,
                total_candidates);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> hier841_sampled_threshold_compact_forward(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor core_int8,
    torch::Tensor middle_int4,
    torch::Tensor tail_sign,
    torch::Tensor key_scales,
    int64_t sample_count,
    double candidate_fraction,
    double selected_fraction,
    int64_t candidate_capacity) {
  TORCH_CHECK(
      query_codes.is_cuda() && query_scales.is_cuda()
          && core_int8.is_cuda() && middle_int4.is_cuda()
          && tail_sign.is_cuda() && key_scales.is_cuda(),
      "all inputs must be CUDA tensors");
  TORCH_CHECK(
      query_codes.scalar_type() == at::kChar
          && core_int8.scalar_type() == at::kChar,
      "query and 8-bit core codes must be int8");
  TORCH_CHECK(
      middle_int4.scalar_type() == at::kByte
          && tail_sign.scalar_type() == at::kByte,
      "packed middle and tail codes must be uint8");
  TORCH_CHECK(
      query_scales.scalar_type() == key_scales.scalar_type(),
      "query and key scale dtypes must match");
  TORCH_CHECK(
      query_codes.dim() == 4 && query_codes.size(3) == 128,
      "query codes must have shape [batch, kv_heads, groups, 128]");
  TORCH_CHECK(
      query_scales.dim() == 4 && query_scales.size(3) == 8,
      "query scales must have shape [batch, kv_heads, groups, 8]");
  TORCH_CHECK(
      core_int8.dim() == 4 && core_int8.size(3) == 16,
      "8-bit core must have shape [batch, kv_heads, history, 16]");
  TORCH_CHECK(
      middle_int4.dim() == 5 && middle_int4.size(2) == 2
          && middle_int4.size(4) == 8,
      "4-bit middle must have shape [batch, kv_heads, 2, history, 8]");
  TORCH_CHECK(
      tail_sign.dim() == 4 && tail_sign.size(3) == 10,
      "tail signs must have shape [batch, kv_heads, history, 10]");
  TORCH_CHECK(
      key_scales.dim() == 4 && key_scales.size(3) == 4,
      "key scales must have shape [batch, kv_heads, history, 4]");
  TORCH_CHECK(
      sample_count > 0 && sample_count <= 256,
      "sample count must be in [1, 256]");
  TORCH_CHECK(
      candidate_fraction > 0.0 && candidate_fraction < 1.0,
      "candidate fraction must be in (0, 1)");
  TORCH_CHECK(
      selected_fraction > 0.0
          && selected_fraction <= candidate_fraction,
      "selected fraction must be in (0, candidate_fraction]");

  int batch_count = static_cast<int>(query_codes.size(0));
  int kv_head_count = static_cast<int>(query_codes.size(1));
  int query_groups = static_cast<int>(query_codes.size(2));
  int query_head_count = kv_head_count * query_groups;
  int history_count = static_cast<int>(core_int8.size(2));
  int row_count = batch_count * query_head_count;
  TORCH_CHECK(
      sample_count <= history_count,
      "sample count cannot exceed history count");
  TORCH_CHECK(
      candidate_capacity > 0 && candidate_capacity <= history_count,
      "candidate capacity must be in [1, history_count]");
  TORCH_CHECK(
      query_scales.size(0) == batch_count
          && query_scales.size(1) == kv_head_count
          && query_scales.size(2) == query_groups,
      "query-scale shape mismatch");
  TORCH_CHECK(
      middle_int4.size(0) == batch_count
          && middle_int4.size(1) == kv_head_count
          && middle_int4.size(3) == history_count,
      "middle-code shape mismatch");
  TORCH_CHECK(
      tail_sign.size(0) == batch_count
          && tail_sign.size(1) == kv_head_count
          && tail_sign.size(2) == history_count,
      "tail-code shape mismatch");
  TORCH_CHECK(
      key_scales.size(0) == batch_count
          && key_scales.size(1) == kv_head_count
          && key_scales.size(2) == history_count,
      "key-scale shape mismatch");

  c10::cuda::CUDAGuard device_guard(query_codes.device());
  auto thresholds = torch::empty(
      {batch_count, query_head_count, 2},
      query_codes.options().dtype(at::kFloat));
  auto indices = torch::zeros(
      {batch_count, query_head_count, candidate_capacity},
      query_codes.options().dtype(at::kLong));
  auto scores = torch::full(
      {batch_count, query_head_count, candidate_capacity},
      -INFINITY,
      query_codes.options().dtype(at::kFloat));
  auto counts = torch::zeros(
      {batch_count, query_head_count},
      query_codes.options().dtype(at::kLong));
  auto overflow = torch::zeros(
      {batch_count, query_head_count},
      query_codes.options().dtype(at::kBool));
  int candidate_keep = std::max(
      1,
      static_cast<int>(ceil(candidate_fraction * sample_count)));
  int selected_keep = std::max(
      1,
      static_cast<int>(ceil(selected_fraction * sample_count)));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_scales.scalar_type(),
      "hier841_sampled_threshold_compact_forward",
      [&] {
        hier841_sample_thresholds_kernel<scalar_t><<<
            row_count,
            256,
            2 * 256 * sizeof(float),
            at::cuda::getCurrentCUDAStream()>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                core_int8.data_ptr<int8_t>(),
                middle_int4.data_ptr<uint8_t>(),
                tail_sign.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                thresholds.data_ptr<float>(),
                kv_head_count,
                query_groups,
                history_count,
                static_cast<int>(sample_count),
                candidate_keep,
                selected_keep);
        dim3 compact_blocks(
            batch_count * kv_head_count,
            (history_count + 255) / 256);
        hier841_threshold_compact_kernel<scalar_t><<<
            compact_blocks,
            256,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                core_int8.data_ptr<int8_t>(),
                middle_int4.data_ptr<uint8_t>(),
                tail_sign.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                thresholds.data_ptr<float>(),
                indices.data_ptr<int64_t>(),
                scores.data_ptr<float>(),
                counts.data_ptr<int64_t>(),
                overflow.data_ptr<bool>(),
                kv_head_count,
                query_groups,
                history_count,
                static_cast<int>(candidate_capacity));
      });
  hier841_finalize_counts_kernel<<<
      (row_count + 255) / 256,
      256,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
          counts.data_ptr<int64_t>(),
          overflow.data_ptr<bool>(),
          row_count,
          static_cast<int>(candidate_capacity));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {indices, scores, counts, thresholds, overflow};
}
"""


@lru_cache(maxsize=1)
def load_extension():
    return load_inline(
        name="hierarchical_spectral_ext_20260727_v2",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=None,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )


def core_scores(
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    core_int8: torch.Tensor,
    middle_int4: torch.Tensor,
    key_scales: torch.Tensor,
) -> torch.Tensor:
    return load_extension().hier841_core_scores_forward(
        query_codes,
        query_scales,
        core_int8,
        middle_int4,
        key_scales,
    )


def tail_candidate_scores(
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    tail_sign: torch.Tensor,
    key_scales: torch.Tensor,
    candidate_indices: torch.Tensor,
) -> torch.Tensor:
    return load_extension().hier841_tail_scores_forward(
        query_codes,
        query_scales,
        tail_sign,
        key_scales,
        candidate_indices,
    )


def sampled_threshold_compact(
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    core_int8: torch.Tensor,
    middle_int4: torch.Tensor,
    tail_sign: torch.Tensor,
    key_scales: torch.Tensor,
    sample_count: int,
    candidate_fraction: float,
    selected_fraction: float,
    candidate_capacity: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return load_extension().hier841_sampled_threshold_compact_forward(
        query_codes,
        query_scales,
        core_int8,
        middle_int4,
        tail_sign,
        key_scales,
        int(sample_count),
        float(candidate_fraction),
        float(selected_fraction),
        int(candidate_capacity),
    )
