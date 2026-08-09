from __future__ import annotations

from functools import lru_cache

import torch
from torch.utils.cpp_extension import load_inline


CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor qksieve_mean_value_blend_forward(
    torch::Tensor sparse_output,
    torch::Tensor value_mean,
    torch::Tensor selected_mass);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "qksieve_mean_value_blend_forward",
      &qksieve_mean_value_blend_forward,
      "QKSieve fused selected-mass and mean-Value blend");
}
"""


CUDA_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void qksieve_mean_value_blend_kernel(
    const scalar_t* __restrict__ sparse_output,
    const float* __restrict__ value_mean,
    const float* __restrict__ selected_mass,
    scalar_t* __restrict__ output,
    int query_head_count,
    int kv_head_count,
    int head_dim,
    int query_groups,
    int element_count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  int dimension = index % head_dim;
  int query_head = (index / head_dim) % query_head_count;
  int batch = index / (query_head_count * head_dim);
  int kv_head = query_head / query_groups;
  float alpha = fminf(
      1.0f,
      fmaxf(0.0f, selected_mass[batch * query_head_count + query_head]));
  float selected = static_cast<float>(sparse_output[index]);
  float mean = value_mean[
      (batch * kv_head_count + kv_head) * head_dim + dimension];
  output[index] = static_cast<scalar_t>(
      alpha * selected + (1.0f - alpha) * mean);
}

torch::Tensor qksieve_mean_value_blend_forward(
    torch::Tensor sparse_output,
    torch::Tensor value_mean,
    torch::Tensor selected_mass) {
  TORCH_CHECK(sparse_output.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(value_mean.is_cuda() && selected_mass.is_cuda(),
              "inputs must share a CUDA device");
  TORCH_CHECK(sparse_output.dim() == 4 && sparse_output.size(1) == 1,
              "sparse output must be [B,1,QH,D]");
  TORCH_CHECK(value_mean.dim() == 3,
              "value mean must be [B,KVH,D]");
  TORCH_CHECK(selected_mass.dim() == 2,
              "selected mass must be [B,QH]");
  TORCH_CHECK(value_mean.scalar_type() == at::kFloat,
              "value mean must be float32");
  TORCH_CHECK(selected_mass.scalar_type() == at::kFloat,
              "selected mass must be float32");
  TORCH_CHECK(sparse_output.size(0) == value_mean.size(0)
                  && sparse_output.size(0) == selected_mass.size(0),
              "batch dimensions must match");
  TORCH_CHECK(sparse_output.size(2) == selected_mass.size(1),
              "Query-head dimensions must match");
  TORCH_CHECK(sparse_output.size(3) == value_mean.size(2),
              "head dimensions must match");
  TORCH_CHECK(sparse_output.size(2) % value_mean.size(1) == 0,
              "Query heads must be divisible by KV heads");

  auto sparse_c = sparse_output.contiguous();
  auto mean_c = value_mean.contiguous();
  auto mass_c = selected_mass.contiguous();
  auto output = torch::empty_like(sparse_c);
  int query_head_count = static_cast<int>(sparse_c.size(2));
  int kv_head_count = static_cast<int>(mean_c.size(1));
  int head_dim = static_cast<int>(sparse_c.size(3));
  int query_groups = query_head_count / kv_head_count;
  int element_count = static_cast<int>(sparse_c.numel());

  c10::cuda::CUDAGuard device_guard(sparse_c.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      sparse_c.scalar_type(),
      "qksieve_mean_value_blend_forward",
      [&] {
        qksieve_mean_value_blend_kernel<scalar_t><<<
            (element_count + 255) / 256, 256, 0, stream>>>(
                sparse_c.data_ptr<scalar_t>(),
                mean_c.data_ptr<float>(),
                mass_c.data_ptr<float>(),
                output.data_ptr<scalar_t>(),
                query_head_count,
                kv_head_count,
                head_dim,
                query_groups,
                element_count);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
"""


@lru_cache(maxsize=1)
def load_extension():
    return load_inline(
        name="qksieve_mean_value_blend_ext_v1",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )


def mean_value_blend(
    sparse_output: torch.Tensor,
    value_mean: torch.Tensor,
    selected_mass: torch.Tensor,
) -> torch.Tensor:
    if not sparse_output.is_cuda:
        raise ValueError("fused mean-Value blend requires CUDA")
    return load_extension().qksieve_mean_value_blend_forward(
        sparse_output,
        value_mean.float(),
        selected_mass.float(),
    )
