from __future__ import annotations

from functools import lru_cache

import torch
from torch.utils.cpp_extension import load_inline


CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor qabs_final_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor indices,
    torch::Tensor valid,
    double scaling);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qabs_final_attention_forward", &qabs_final_attention_forward, "QABS final sparse attention forward");
}
"""


CUDA_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

template <typename scalar_t>
__global__ void qabs_final_attention_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const int64_t* __restrict__ indices,
    const bool* __restrict__ valid,
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
  const bool* valid_row = valid + row * select_count;
  scalar_t* out_row = output + row * head_dim;

  for (int s = tid; s < select_count; s += blockDim.x) {
    weights[s] = 0.0f;
  }
  __syncthreads();

  float max_score = -CUDART_INF_F;
  for (int s = 0; s < select_count; ++s) {
    float local = 0.0f;
    int64_t idx = idx_row[s];
    bool is_valid = valid_row[s] && idx >= 0 && idx < key_count;
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
    bool is_valid = valid_row[s] && idx >= 0 && idx < key_count;
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
      bool is_valid = valid_row[s] && idx >= 0 && idx < key_count;
      if (is_valid) {
        acc += weights[s] * static_cast<float>(v_base[idx * head_dim + d]);
      }
    }
    out_row[d] = static_cast<scalar_t>(acc);
  }
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
  TORCH_CHECK(query.dim() == 3, "query must have shape [batch, heads, dim]");
  TORCH_CHECK(key.dim() == 4, "key must have shape [batch, heads, key, dim]");
  TORCH_CHECK(value.sizes() == key.sizes(), "value must match key shape");
  TORCH_CHECK(indices.dim() == 3, "indices must have shape [batch, heads, selected]");
  TORCH_CHECK(valid.sizes() == indices.sizes(), "valid must match indices shape");
  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query/key dtype mismatch");
  TORCH_CHECK(query.scalar_type() == value.scalar_type(), "query/value dtype mismatch");
  TORCH_CHECK(indices.scalar_type() == at::kLong, "indices must be int64");
  TORCH_CHECK(valid.scalar_type() == at::kBool, "valid must be bool");

  auto query_c = query.contiguous();
  auto key_c = key.contiguous();
  auto value_c = value.contiguous();
  auto indices_c = indices.contiguous();
  auto valid_c = valid.contiguous();

  int batch_count = static_cast<int>(query_c.size(0));
  int head_count = static_cast<int>(query_c.size(1));
  int head_dim = static_cast<int>(query_c.size(2));
  int key_count = static_cast<int>(key_c.size(2));
  int select_count = static_cast<int>(indices_c.size(2));
  TORCH_CHECK(key_c.size(0) == batch_count && key_c.size(1) == head_count && key_c.size(3) == head_dim, "key shape mismatch");
  TORCH_CHECK(indices_c.size(0) == batch_count && indices_c.size(1) == head_count, "indices shape mismatch");

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
    qabs_final_attention_kernel<scalar_t><<<blocks, threads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        query_c.data_ptr<scalar_t>(),
        key_c.data_ptr<scalar_t>(),
        value_c.data_ptr<scalar_t>(),
        indices_c.data_ptr<int64_t>(),
        valid_c.data_ptr<bool>(),
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
        name="qabs_final_attention_ext",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
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
    output = module.qabs_final_attention_forward(query, key, value, indices, valid, float(scaling))
    return output[:, None, :, :].contiguous()
