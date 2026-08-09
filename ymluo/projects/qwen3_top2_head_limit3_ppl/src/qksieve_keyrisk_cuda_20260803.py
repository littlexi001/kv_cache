from __future__ import annotations

from functools import lru_cache

import torch
from torch.utils.cpp_extension import load_inline


CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor qksieve_key_residual_norm_forward(
    torch::Tensor projected_key,
    torch::Tensor bit_allocations);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "qksieve_key_residual_norm_forward",
      &qksieve_key_residual_norm_forward,
      "Fused QKSieve projected-Key quantization residual norm");
}
"""


CUDA_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void qksieve_key_residual_norm_kernel(
    const scalar_t* __restrict__ projected,
    const int8_t* __restrict__ allocations,
    scalar_t* __restrict__ output,
    int token_count) {
  constexpr int kHeadDim = 128;
  constexpr int kBandDim = 16;
  int row = blockIdx.x;
  int dimension = threadIdx.x;
  int batch_head = row / token_count;
  int band = dimension / kBandDim;
  int lane = dimension - band * kBandDim;
  int bits = static_cast<int>(allocations[batch_head * 8 + band]);

  __shared__ float values[kHeadDim];
  __shared__ float scales[8];
  __shared__ float residual_square[kHeadDim];
  float value = static_cast<float>(
      projected[static_cast<int64_t>(row) * kHeadDim + dimension]);
  values[dimension] = value;
  __syncthreads();

  if (lane == 0) {
    float statistic = 0.0f;
#pragma unroll
    for (int offset = 0; offset < kBandDim; ++offset) {
      float magnitude = fabsf(values[band * kBandDim + offset]);
      statistic = bits == 1
          ? statistic + magnitude
          : fmaxf(statistic, magnitude);
    }
    float scale = bits == 0 || bits == 16
        ? 1.0f
        : bits == 1
        ? statistic / static_cast<float>(kBandDim)
        : statistic / static_cast<float>((1 << (bits - 1)) - 1);
    scales[band] = fmaxf(scale, 1.0e-8f);
  }
  __syncthreads();

  float reconstructed = 0.0f;
  if (bits == 16) {
    reconstructed = value;
  } else if (bits == 1) {
    reconstructed = value >= 0.0f ? scales[band] : -scales[band];
  } else if (bits > 1) {
    int maximum = (1 << (bits - 1)) - 1;
    int code = __float2int_rn(value / scales[band]);
    code = max(-maximum, min(maximum, code));
    reconstructed = static_cast<float>(code) * scales[band];
  }
  float residual = value - reconstructed;
  residual_square[dimension] = residual * residual;
  __syncthreads();

  for (int stride = 64; stride > 0; stride >>= 1) {
    if (dimension < stride) {
      residual_square[dimension] += residual_square[dimension + stride];
    }
    __syncthreads();
  }
  if (dimension == 0) {
    output[row] = static_cast<scalar_t>(sqrtf(residual_square[0]));
  }
}

torch::Tensor qksieve_key_residual_norm_forward(
    torch::Tensor projected_key,
    torch::Tensor bit_allocations) {
  TORCH_CHECK(
      projected_key.is_cuda() && bit_allocations.is_cuda(),
      "projected Keys and allocations must be CUDA tensors");
  TORCH_CHECK(
      projected_key.dim() == 4 && projected_key.size(3) == 128,
      "projected Keys must have shape [batch, kv_heads, tokens, 128]");
  TORCH_CHECK(
      bit_allocations.dim() == 3
          && bit_allocations.size(0) == projected_key.size(0)
          && bit_allocations.size(1) == projected_key.size(1)
          && bit_allocations.size(2) == 8,
      "allocations must have shape [batch, kv_heads, 8]");
  TORCH_CHECK(
      bit_allocations.scalar_type() == at::kChar,
      "allocations must use int8");
  TORCH_CHECK(
      projected_key.scalar_type() == at::kHalf
          || projected_key.scalar_type() == at::kBFloat16,
      "projected Keys must use FP16 or BF16");

  auto projected = projected_key.contiguous();
  auto allocations = bit_allocations.contiguous();
  auto output = torch::empty(
      projected.sizes().slice(0, 3), projected.options());
  int token_count = static_cast<int>(projected.size(2));
  int rows = static_cast<int>(
      projected.size(0) * projected.size(1) * projected.size(2));
  c10::cuda::CUDAGuard device_guard(projected.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      projected.scalar_type(),
      "qksieve_key_residual_norm_forward",
      [&] {
        qksieve_key_residual_norm_kernel<scalar_t>
            <<<rows, 128, 0, stream>>>(
                projected.data_ptr<scalar_t>(),
                allocations.data_ptr<int8_t>(),
                output.data_ptr<scalar_t>(),
                token_count);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
"""


@lru_cache(maxsize=1)
def load_extension():
    return load_inline(
        name="qksieve_keyrisk_ext_v1",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )


def residual_norm(
    projected_key: torch.Tensor,
    bit_allocations: torch.Tensor,
) -> torch.Tensor:
    if projected_key.ndim != 4 or projected_key.shape[-1] != 128:
        raise ValueError("projected Keys must have shape [B, KVH, N, 128]")
    if bit_allocations.shape != projected_key.shape[:2] + (8,):
        raise ValueError("allocations must have shape [B, KVH, 8]")
    return load_extension().qksieve_key_residual_norm_forward(
        projected_key.contiguous(),
        bit_allocations.to(device=projected_key.device, dtype=torch.int8).contiguous(),
    )
