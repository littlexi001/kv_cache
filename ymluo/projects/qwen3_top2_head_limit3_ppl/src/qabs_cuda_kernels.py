from __future__ import annotations

from functools import lru_cache

import torch
from torch.utils.cpp_extension import load_inline


CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor qabs_partial_scores_forward(
    torch::Tensor query,
    torch::Tensor key,
    int64_t dim_count);

torch::Tensor qabs_partial_scores_dim_major_forward(
    torch::Tensor query,
    torch::Tensor key_dim_major,
    torch::Tensor dim_indices);

torch::Tensor qabs_candidate_full_scores_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor current_candidate,
    torch::Tensor previous_candidate,
    bool has_previous_candidate,
    torch::Tensor previous_final,
    bool has_previous_final,
    int64_t protect_sink_tokens,
    int64_t protect_recent_tokens,
    double scaling);

torch::Tensor qabs_final_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor indices,
    torch::Tensor valid,
    double scaling);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qabs_partial_scores_forward", &qabs_partial_scores_forward, "QABS partial candidate scores");
  m.def("qabs_partial_scores_dim_major_forward", &qabs_partial_scores_dim_major_forward, "QABS partial candidate scores on dim-major K");
  m.def("qabs_candidate_full_scores_forward", &qabs_candidate_full_scores_forward, "QABS candidate full-QK scores");
  m.def("qabs_final_attention_forward", &qabs_final_attention_forward, "QABS final sparse attention forward");
}
"""


CUDA_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

constexpr int QABS_MAX_DIMS = 128;
constexpr int QABS_TILE_THREADS = 256;

template <typename scalar_t>
__global__ void qabs_partial_scores_dim_major_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key_dim_major,
    const int64_t* __restrict__ dim_indices,
    float* __restrict__ output,
    int key_count,
    int head_dim,
    int selected_count,
    int64_t key_stride_b,
    int64_t key_stride_h,
    int64_t key_stride_d,
    int64_t key_stride_k,
    int head_count) {
  int row = blockIdx.x;
  int tile = blockIdx.y;
  int tid = threadIdx.x;
  int batch = row / head_count;
  int head = row - batch * head_count;
  int token = tile * blockDim.x + tid;
  if (token >= key_count) {
    return;
  }

  const scalar_t* q_row = query + row * head_dim;
  const int64_t* dim_row = dim_indices + row * selected_count;
  const scalar_t* k_base = key_dim_major + batch * key_stride_b + head * key_stride_h;
  float acc = 0.0f;
  for (int s = 0; s < selected_count; ++s) {
    int64_t dim = dim_row[s];
    acc += static_cast<float>(q_row[dim]) * static_cast<float>(k_base[dim * key_stride_d + token * key_stride_k]);
  }
  output[row * key_count + token] = acc;
}

template <typename scalar_t>
__global__ void qabs_partial_scores_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    float* __restrict__ output,
    int row_count,
    int key_count,
    int head_dim,
    int dim_count,
    int64_t key_stride_b,
    int64_t key_stride_h,
    int64_t key_stride_k,
    int64_t key_stride_d,
    int head_count) {
  __shared__ int selected_idx[QABS_MAX_DIMS];
  __shared__ float selected_q[QABS_MAX_DIMS];

  int row = blockIdx.x;
  int tile = blockIdx.y;
  int tid = threadIdx.x;
  int batch = row / head_count;
  int head = row - batch * head_count;
  int selected_count = min(min(max(dim_count, 1), head_dim), QABS_MAX_DIMS);
  const scalar_t* q_row = query + row * head_dim;

  if (tid == 0) {
    for (int s = 0; s < selected_count; ++s) {
      int best_idx = 0;
      float best_abs = -1.0f;
      for (int d = 0; d < head_dim; ++d) {
        bool used = false;
        for (int u = 0; u < s; ++u) {
          used = used || selected_idx[u] == d;
        }
        if (used) {
          continue;
        }
        float value = static_cast<float>(q_row[d]);
        float magnitude = fabsf(value);
        if (magnitude > best_abs) {
          best_abs = magnitude;
          best_idx = d;
        }
      }
      selected_idx[s] = best_idx;
      selected_q[s] = static_cast<float>(q_row[best_idx]);
    }
  }
  __syncthreads();

  int token = tile * blockDim.x + tid;
  if (token >= key_count) {
    return;
  }
  const scalar_t* k_row = key + batch * key_stride_b + head * key_stride_h + token * key_stride_k;
  float acc = 0.0f;
  for (int s = 0; s < selected_count; ++s) {
    acc += selected_q[s] * static_cast<float>(k_row[selected_idx[s] * key_stride_d]);
  }
  output[row * key_count + token] = acc;
}

template <typename scalar_t>
__global__ void qabs_candidate_full_scores_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const bool* __restrict__ current_candidate,
    const bool* __restrict__ previous_candidate,
    bool has_previous_candidate,
    const bool* __restrict__ previous_final,
    bool has_previous_final,
    float* __restrict__ output,
    int key_count,
    int head_dim,
    int protect_sink_tokens,
    int protect_recent_tokens,
    float scaling,
    int64_t key_stride_b,
    int64_t key_stride_h,
    int64_t key_stride_k,
    int64_t key_stride_d,
    int head_count) {
  int row = blockIdx.x;
  int tile = blockIdx.y;
  int tid = threadIdx.x;
  int batch = row / head_count;
  int head = row - batch * head_count;
  int token = tile * blockDim.x + tid;
  if (token >= key_count) {
    return;
  }

  int offset = row * key_count + token;
  bool selected = current_candidate[offset];
  if (has_previous_candidate) {
    selected = selected || previous_candidate[offset];
  }
  if (has_previous_final) {
    selected = selected || previous_final[offset];
  }
  if (protect_sink_tokens > 0 && token < protect_sink_tokens) {
    selected = true;
  }
  if (protect_recent_tokens > 0 && token >= max(0, key_count - protect_recent_tokens)) {
    selected = true;
  }
  if (!selected) {
    output[offset] = -CUDART_INF_F;
    return;
  }

  const scalar_t* q_row = query + row * head_dim;
  const scalar_t* k_row = key + batch * key_stride_b + head * key_stride_h + token * key_stride_k;
  float acc = 0.0f;
  for (int d = 0; d < head_dim; ++d) {
    acc += static_cast<float>(q_row[d]) * static_cast<float>(k_row[d * key_stride_d]);
  }
  output[offset] = acc * scaling;
}

template <typename scalar_t>
__global__ void qabs_final_attention_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const int64_t* __restrict__ indices,
    const uint8_t* __restrict__ valid,
    scalar_t* __restrict__ output,
    int batch_count,
    int head_count,
    int key_count,
    int select_count,
    int head_dim,
    float scaling) {
  extern __shared__ float shared[];
  float* reduction = shared;
  float* weights = shared + blockDim.x;
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int batch = row / head_count;
  int head = row - batch * head_count;

  const scalar_t* q_row = query + row * head_dim;
  const scalar_t* k_base = key + ((batch * head_count + head) * key_count) * head_dim;
  const scalar_t* v_base = value + ((batch * head_count + head) * key_count) * head_dim;
  const int64_t* idx_row = indices + row * select_count;
  const uint8_t* valid_row = valid + row * select_count;
  scalar_t* out_row = output + row * head_dim;

  for (int s = tid; s < select_count; s += blockDim.x) {
    weights[s] = 0.0f;
  }
  __syncthreads();

  float max_score = -CUDART_INF_F;
  for (int s = 0; s < select_count; ++s) {
    float local = 0.0f;
    int64_t idx = idx_row[s];
    bool is_valid = valid_row[s] != 0 && idx >= 0 && idx < key_count;
    if (is_valid) {
      const scalar_t* k_vec = k_base + idx * head_dim;
      for (int d = tid; d < head_dim; d += blockDim.x) {
        local += static_cast<float>(q_row[d]) * static_cast<float>(k_vec[d]);
      }
    }
    reduction[tid] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduction[tid] += reduction[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0 && is_valid) {
      float score = reduction[0] * scaling;
      max_score = fmaxf(max_score, score);
    }
    __syncthreads();
  }

  __shared__ float shared_max;
  __shared__ float shared_denom;
  if (tid == 0) {
    shared_max = isfinite(max_score) ? max_score : 0.0f;
    shared_denom = 0.0f;
  }
  __syncthreads();

  float denom_local = 0.0f;
  for (int s = 0; s < select_count; ++s) {
    float local = 0.0f;
    int64_t idx = idx_row[s];
    bool is_valid = valid_row[s] != 0 && idx >= 0 && idx < key_count;
    if (is_valid) {
      const scalar_t* k_vec = k_base + idx * head_dim;
      for (int d = tid; d < head_dim; d += blockDim.x) {
        local += static_cast<float>(q_row[d]) * static_cast<float>(k_vec[d]);
      }
    }
    reduction[tid] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduction[tid] += reduction[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0 && is_valid) {
      float weight = expf(reduction[0] * scaling - shared_max);
      weights[s] = weight;
      denom_local += weight;
    }
    __syncthreads();
  }
  if (tid == 0) {
    shared_denom = fmaxf(denom_local, 1.0e-20f);
  }
  __syncthreads();

  for (int s = tid; s < select_count; s += blockDim.x) {
    weights[s] = weights[s] / shared_denom;
  }
  __syncthreads();

  for (int d = tid; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int s = 0; s < select_count; ++s) {
      int64_t idx = idx_row[s];
      bool is_valid = valid_row[s] != 0 && idx >= 0 && idx < key_count;
      if (is_valid) {
        acc += weights[s] * static_cast<float>(v_base[idx * head_dim + d]);
      }
    }
    out_row[d] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t>
__global__ void qabs_final_attention_token_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const int64_t* __restrict__ indices,
    const uint8_t* __restrict__ valid,
    scalar_t* __restrict__ output,
    int batch_count,
    int head_count,
    int key_count,
    int select_count,
    int head_dim,
    float scaling) {
  extern __shared__ float shared[];
  float* reduction = shared;
  float* weights = shared + blockDim.x;
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int batch = row / head_count;
  int head = row - batch * head_count;

  const scalar_t* q_row = query + row * head_dim;
  const scalar_t* k_base = key + ((batch * head_count + head) * key_count) * head_dim;
  const scalar_t* v_base = value + ((batch * head_count + head) * key_count) * head_dim;
  const int64_t* idx_row = indices + row * select_count;
  const uint8_t* valid_row = valid + row * select_count;
  scalar_t* out_row = output + row * head_dim;

  float local_max = -CUDART_INF_F;
  for (int s = tid; s < select_count; s += blockDim.x) {
    int64_t idx = idx_row[s];
    bool is_valid = valid_row[s] != 0 && idx >= 0 && idx < key_count;
    float score = -CUDART_INF_F;
    if (is_valid) {
      const scalar_t* k_vec = k_base + idx * head_dim;
      float acc = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        acc += static_cast<float>(q_row[d]) * static_cast<float>(k_vec[d]);
      }
      score = acc * scaling;
      local_max = fmaxf(local_max, score);
    }
    weights[s] = score;
  }
  reduction[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
    }
    __syncthreads();
  }
  float max_score = isfinite(reduction[0]) ? reduction[0] : 0.0f;

  float local_denom = 0.0f;
  for (int s = tid; s < select_count; s += blockDim.x) {
    float weight = isfinite(weights[s]) ? expf(weights[s] - max_score) : 0.0f;
    weights[s] = weight;
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
  float denom = fmaxf(reduction[0], 1.0e-20f);

  for (int d = tid; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int s = 0; s < select_count; ++s) {
      float weight = weights[s];
      if (weight != 0.0f) {
        int64_t idx = idx_row[s];
        if (idx >= 0 && idx < key_count) {
          acc += (weight / denom) * static_cast<float>(v_base[idx * head_dim + d]);
        }
      }
    }
    out_row[d] = static_cast<scalar_t>(acc);
  }
}

torch::Tensor qabs_partial_scores_forward(
    torch::Tensor query,
    torch::Tensor key,
    int64_t dim_count) {
  TORCH_CHECK(query.is_cuda(), "query must be CUDA");
  TORCH_CHECK(key.is_cuda(), "key must be CUDA");
  TORCH_CHECK(query.dim() == 3, "query must have shape [batch, heads, dim]");
  TORCH_CHECK(key.dim() == 4, "key must have shape [batch, heads, key, dim]");
  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query/key dtype mismatch");
  TORCH_CHECK(query.size(0) == key.size(0) && query.size(1) == key.size(1) && query.size(2) == key.size(3), "query/key shape mismatch");

  auto query_c = query.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int head_count = static_cast<int>(query_c.size(1));
  int head_dim = static_cast<int>(query_c.size(2));
  int key_count = static_cast<int>(key.size(2));
  int selected_dim_count = static_cast<int>(std::min<int64_t>(std::max<int64_t>(dim_count, 1), std::min<int64_t>(head_dim, QABS_MAX_DIMS)));
  auto output = torch::empty({batch_count, head_count, key_count}, query_c.options().dtype(at::kFloat));
  dim3 blocks(batch_count * head_count, (key_count + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, query_c.scalar_type(), "qabs_partial_scores_forward", [&] {
    qabs_partial_scores_kernel<scalar_t><<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        query_c.data_ptr<scalar_t>(),
        key.data_ptr<scalar_t>(),
        output.data_ptr<float>(),
        batch_count * head_count,
        key_count,
        head_dim,
        selected_dim_count,
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        head_count);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_partial_scores_dim_major_forward(
    torch::Tensor query,
    torch::Tensor key_dim_major,
    torch::Tensor dim_indices) {
  TORCH_CHECK(query.is_cuda(), "query must be CUDA");
  TORCH_CHECK(key_dim_major.is_cuda(), "key_dim_major must be CUDA");
  TORCH_CHECK(dim_indices.is_cuda(), "dim_indices must be CUDA");
  TORCH_CHECK(query.dim() == 3, "query must have shape [batch, heads, dim]");
  TORCH_CHECK(key_dim_major.dim() == 4, "key_dim_major must have shape [batch, heads, dim, key]");
  TORCH_CHECK(dim_indices.dim() == 3, "dim_indices must have shape [batch, heads, selected_dim]");
  TORCH_CHECK(query.scalar_type() == key_dim_major.scalar_type(), "query/key dtype mismatch");
  TORCH_CHECK(dim_indices.scalar_type() == at::kLong, "dim_indices must be int64");
  TORCH_CHECK(query.size(0) == key_dim_major.size(0) && query.size(1) == key_dim_major.size(1) && query.size(2) == key_dim_major.size(2), "query/key shape mismatch");
  TORCH_CHECK(dim_indices.size(0) == query.size(0) && dim_indices.size(1) == query.size(1), "dim_indices shape mismatch");

  auto query_c = query.contiguous();
  auto dim_indices_c = dim_indices.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int head_count = static_cast<int>(query_c.size(1));
  int head_dim = static_cast<int>(query_c.size(2));
  int key_count = static_cast<int>(key_dim_major.size(3));
  int selected_count = static_cast<int>(dim_indices_c.size(2));
  auto output = torch::empty({batch_count, head_count, key_count}, query_c.options().dtype(at::kFloat));
  dim3 blocks(batch_count * head_count, (key_count + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, query_c.scalar_type(), "qabs_partial_scores_dim_major_forward", [&] {
    qabs_partial_scores_dim_major_kernel<scalar_t><<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        query_c.data_ptr<scalar_t>(),
        key_dim_major.data_ptr<scalar_t>(),
        dim_indices_c.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        key_count,
        head_dim,
        selected_count,
        key_dim_major.stride(0),
        key_dim_major.stride(1),
        key_dim_major.stride(2),
        key_dim_major.stride(3),
        head_count);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_candidate_full_scores_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor current_candidate,
    torch::Tensor previous_candidate,
    bool has_previous_candidate,
    torch::Tensor previous_final,
    bool has_previous_final,
    int64_t protect_sink_tokens,
    int64_t protect_recent_tokens,
    double scaling) {
  TORCH_CHECK(query.is_cuda(), "query must be CUDA");
  TORCH_CHECK(key.is_cuda(), "key must be CUDA");
  TORCH_CHECK(current_candidate.is_cuda(), "current_candidate must be CUDA");
  TORCH_CHECK(query.dim() == 3, "query must have shape [batch, heads, dim]");
  TORCH_CHECK(key.dim() == 4, "key must have shape [batch, heads, key, dim]");
  TORCH_CHECK(current_candidate.dim() == 3, "current_candidate must have shape [batch, heads, key]");
  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query/key dtype mismatch");
  TORCH_CHECK(current_candidate.scalar_type() == at::kBool, "current_candidate must be bool");
  TORCH_CHECK(!has_previous_candidate || previous_candidate.scalar_type() == at::kBool, "previous_candidate must be bool");
  TORCH_CHECK(!has_previous_final || previous_final.scalar_type() == at::kBool, "previous_final must be bool");
  TORCH_CHECK(query.size(0) == key.size(0) && query.size(1) == key.size(1) && query.size(2) == key.size(3), "query/key shape mismatch");
  TORCH_CHECK(current_candidate.size(0) == key.size(0) && current_candidate.size(1) == key.size(1) && current_candidate.size(2) == key.size(2), "candidate shape mismatch");

  auto query_c = query.contiguous();
  auto current_c = current_candidate.contiguous();
  auto previous_candidate_c = has_previous_candidate ? previous_candidate.contiguous() : previous_candidate;
  auto previous_final_c = has_previous_final ? previous_final.contiguous() : previous_final;
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int head_count = static_cast<int>(query_c.size(1));
  int head_dim = static_cast<int>(query_c.size(2));
  int key_count = static_cast<int>(key.size(2));
  auto output = torch::empty({batch_count, head_count, key_count}, query_c.options().dtype(at::kFloat));
  dim3 blocks(batch_count * head_count, (key_count + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);

  const bool* previous_candidate_ptr = has_previous_candidate ? previous_candidate_c.data_ptr<bool>() : current_c.data_ptr<bool>();
  const bool* previous_final_ptr = has_previous_final ? previous_final_c.data_ptr<bool>() : current_c.data_ptr<bool>();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, query_c.scalar_type(), "qabs_candidate_full_scores_forward", [&] {
    qabs_candidate_full_scores_kernel<scalar_t><<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        query_c.data_ptr<scalar_t>(),
        key.data_ptr<scalar_t>(),
        current_c.data_ptr<bool>(),
        previous_candidate_ptr,
        has_previous_candidate,
        previous_final_ptr,
        has_previous_final,
        output.data_ptr<float>(),
        key_count,
        head_dim,
        static_cast<int>(protect_sink_tokens),
        static_cast<int>(protect_recent_tokens),
        static_cast<float>(scaling),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        head_count);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_final_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor indices,
    torch::Tensor valid,
    double scaling) {
  TORCH_CHECK(query.is_cuda(), "query must be CUDA");
  TORCH_CHECK(key.is_cuda(), "key must be CUDA");
  TORCH_CHECK(value.is_cuda(), "value must be CUDA");
  TORCH_CHECK(indices.is_cuda(), "indices must be CUDA");
  TORCH_CHECK(valid.is_cuda(), "valid must be CUDA");
  TORCH_CHECK(query.device() == key.device(), "query/key device mismatch");
  TORCH_CHECK(query.device() == value.device(), "query/value device mismatch");
  TORCH_CHECK(query.device() == indices.device(), "query/indices device mismatch");
  TORCH_CHECK(query.device() == valid.device(), "query/valid device mismatch");
  TORCH_CHECK(query.dim() == 3, "query must have shape [batch, heads, dim]");
  TORCH_CHECK(key.dim() == 4, "key must have shape [batch, heads, key, dim]");
  TORCH_CHECK(value.sizes() == key.sizes(), "value must match key shape");
  TORCH_CHECK(indices.dim() == 3, "indices must have shape [batch, heads, selected]");
  TORCH_CHECK(valid.sizes() == indices.sizes(), "valid must match indices shape");
  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query/key dtype mismatch");
  TORCH_CHECK(query.scalar_type() == value.scalar_type(), "query/value dtype mismatch");
  TORCH_CHECK(indices.scalar_type() == at::kLong, "indices must be int64");
  TORCH_CHECK(valid.scalar_type() == at::kByte, "valid must be uint8");

  auto query_c = query.contiguous();
  auto key_c = key.contiguous();
  auto value_c = value.contiguous();
  auto indices_c = indices.contiguous();
  auto valid_c = valid.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());

  int batch_count = static_cast<int>(query_c.size(0));
  int head_count = static_cast<int>(query_c.size(1));
  int head_dim = static_cast<int>(query_c.size(2));
  int key_count = static_cast<int>(key_c.size(2));
  int select_count = static_cast<int>(indices_c.size(2));
  TORCH_CHECK(key_c.size(0) == batch_count && key_c.size(1) == head_count && key_c.size(3) == head_dim, "key shape mismatch");
  TORCH_CHECK(indices_c.size(0) == batch_count && indices_c.size(1) == head_count, "indices shape mismatch");
  TORCH_CHECK(select_count > 0, "select_count must be positive");

  auto output = torch::empty({batch_count, head_count, head_dim}, query_c.options());
  int threads = 1;
  while (threads < head_dim) {
    threads <<= 1;
  }
  threads = threads < 32 ? 32 : threads;
  threads = threads > 256 ? 256 : threads;
  int blocks = batch_count * head_count;
  size_t shared_bytes = static_cast<size_t>(threads + select_count) * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, query_c.scalar_type(), "qabs_final_attention_forward", [&] {
    qabs_final_attention_token_kernel<scalar_t><<<blocks, threads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        query_c.data_ptr<scalar_t>(),
        key_c.data_ptr<scalar_t>(),
        value_c.data_ptr<scalar_t>(),
        indices_c.data_ptr<int64_t>(),
        valid_c.data_ptr<uint8_t>(),
        output.data_ptr<scalar_t>(),
        batch_count,
        head_count,
        key_count,
        select_count,
        head_dim,
        static_cast<float>(scaling));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
"""


@lru_cache(maxsize=1)
def _load_extension():
    return load_inline(
        name="qabs_sparse_attention_ext_v5",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )


def partial_scores(query: torch.Tensor, key_history: torch.Tensor, dim_count: int) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_partial_scores_forward(query, key_history, int(dim_count))


def partial_scores_dim_major(
    query: torch.Tensor,
    key_dim_major: torch.Tensor,
    dim_indices: torch.Tensor,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_partial_scores_dim_major_forward(query, key_dim_major, dim_indices)


def candidate_full_scores(
    query: torch.Tensor,
    key_history: torch.Tensor,
    current_candidate: torch.Tensor,
    previous_candidate: torch.Tensor | None,
    previous_final: torch.Tensor | None,
    protect_sink_tokens: int,
    protect_recent_tokens: int,
    scaling: float,
) -> torch.Tensor:
    module = _load_extension()
    empty_candidate = torch.empty(0, dtype=torch.bool, device=query.device)
    return module.qabs_candidate_full_scores_forward(
        query,
        key_history,
        current_candidate,
        previous_candidate if previous_candidate is not None else empty_candidate,
        previous_candidate is not None,
        previous_final if previous_final is not None else empty_candidate,
        previous_final is not None,
        int(protect_sink_tokens),
        int(protect_recent_tokens),
        float(scaling),
    )


def final_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: torch.Tensor,
    valid: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    module = _load_extension()
    valid_u8 = valid.to(dtype=torch.uint8)
    output = module.qabs_final_attention_forward(query, key, value, indices, valid_u8, float(scaling))
    return output[:, None, :, :].contiguous()
