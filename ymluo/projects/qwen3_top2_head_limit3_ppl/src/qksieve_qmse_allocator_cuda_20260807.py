from __future__ import annotations

from functools import lru_cache

import torch
from torch.utils.cpp_extension import load_inline


CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor qksieve_qmse_allocate_cuda(
    torch::Tensor costs,
    torch::Tensor feasible_allocations,
    torch::Tensor feasible_bit_indices,
    torch::Tensor feasible_used_bits);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "allocate",
      &qksieve_qmse_allocate_cuda,
      "Fused exact qMSE rate allocation");
}
"""


CUDA_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <limits>

namespace {

constexpr int kBands = 8;
constexpr int kThreads = 256;

__device__ __forceinline__ bool better_candidate(
    double candidate_cost,
    int candidate_used_bits,
    int candidate_index,
    double incumbent_cost,
    int incumbent_used_bits,
    int incumbent_index) {
  return candidate_cost < incumbent_cost ||
      (candidate_cost == incumbent_cost &&
       (candidate_used_bits > incumbent_used_bits ||
        (candidate_used_bits == incumbent_used_bits &&
         candidate_index < incumbent_index)));
}

__global__ void qmse_allocate_kernel(
    const double* __restrict__ costs,
    const int16_t* __restrict__ feasible_allocations,
    const int64_t* __restrict__ feasible_bit_indices,
    const int16_t* __restrict__ feasible_used_bits,
    int16_t* __restrict__ output,
    int row_count,
    int level_count,
    int allocation_count) {
  const int row = blockIdx.x;
  if (row >= row_count) {
    return;
  }

  double best_cost = std::numeric_limits<double>::infinity();
  int best_used_bits = -1;
  int best_index = allocation_count;
  const double* row_costs = costs + row * kBands * level_count;
  for (int allocation = threadIdx.x; allocation < allocation_count;
       allocation += blockDim.x) {
    double total = 0.0;
#pragma unroll
    for (int band = 0; band < kBands; ++band) {
      const int level = static_cast<int>(
          feasible_bit_indices[allocation * kBands + band]);
      total += row_costs[band * level_count + level];
    }
    const int used_bits = static_cast<int>(
        feasible_used_bits[allocation]);
    if (better_candidate(
            total,
            used_bits,
            allocation,
            best_cost,
            best_used_bits,
            best_index)) {
      best_cost = total;
      best_used_bits = used_bits;
      best_index = allocation;
    }
  }

  __shared__ double shared_cost[kThreads];
  __shared__ int shared_used_bits[kThreads];
  __shared__ int shared_index[kThreads];
  shared_cost[threadIdx.x] = best_cost;
  shared_used_bits[threadIdx.x] = best_used_bits;
  shared_index[threadIdx.x] = best_index;
  __syncthreads();

  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const int other = threadIdx.x + stride;
      if (better_candidate(
              shared_cost[other],
              shared_used_bits[other],
              shared_index[other],
              shared_cost[threadIdx.x],
              shared_used_bits[threadIdx.x],
              shared_index[threadIdx.x])) {
        shared_cost[threadIdx.x] = shared_cost[other];
        shared_used_bits[threadIdx.x] = shared_used_bits[other];
        shared_index[threadIdx.x] = shared_index[other];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x < kBands) {
    output[row * kBands + threadIdx.x] = feasible_allocations[
        shared_index[0] * kBands + threadIdx.x];
  }
}

}  // namespace

torch::Tensor qksieve_qmse_allocate_cuda(
    torch::Tensor costs,
    torch::Tensor feasible_allocations,
    torch::Tensor feasible_bit_indices,
    torch::Tensor feasible_used_bits) {
  TORCH_CHECK(costs.is_cuda(), "costs must be CUDA");
  TORCH_CHECK(costs.scalar_type() == torch::kFloat64, "costs must be float64");
  TORCH_CHECK(costs.dim() == 4 && costs.size(2) == kBands, "invalid costs shape");
  TORCH_CHECK(feasible_allocations.is_cuda(), "allocations must be CUDA");
  TORCH_CHECK(feasible_allocations.scalar_type() == torch::kInt16, "allocations must be int16");
  TORCH_CHECK(feasible_bit_indices.is_cuda(), "bit indices must be CUDA");
  TORCH_CHECK(feasible_bit_indices.scalar_type() == torch::kInt64, "bit indices must be int64");
  TORCH_CHECK(feasible_used_bits.is_cuda(), "used bits must be CUDA");
  TORCH_CHECK(feasible_used_bits.scalar_type() == torch::kInt16, "used bits must be int16");
  TORCH_CHECK(costs.is_contiguous(), "costs must be contiguous");
  TORCH_CHECK(feasible_allocations.is_contiguous(), "allocations must be contiguous");
  TORCH_CHECK(feasible_bit_indices.is_contiguous(), "bit indices must be contiguous");
  TORCH_CHECK(feasible_used_bits.is_contiguous(), "used bits must be contiguous");

  const int64_t batch_count = costs.size(0);
  const int64_t head_count = costs.size(1);
  const int64_t level_count = costs.size(3);
  const int64_t allocation_count = feasible_allocations.size(0);
  TORCH_CHECK(feasible_allocations.sizes() == torch::IntArrayRef({allocation_count, kBands}), "invalid allocations shape");
  TORCH_CHECK(feasible_bit_indices.sizes() == torch::IntArrayRef({allocation_count, kBands}), "invalid bit-index shape");
  TORCH_CHECK(feasible_used_bits.numel() == allocation_count, "invalid used-bit shape");

  c10::cuda::CUDAGuard guard(costs.device());
  auto output = torch::empty(
      {batch_count, head_count, kBands},
      costs.options().dtype(torch::kInt16));
  const int row_count = static_cast<int>(batch_count * head_count);
  qmse_allocate_kernel<<<
      row_count,
      kThreads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
          costs.data_ptr<double>(),
          feasible_allocations.data_ptr<int16_t>(),
          feasible_bit_indices.data_ptr<int64_t>(),
          feasible_used_bits.data_ptr<int16_t>(),
          output.data_ptr<int16_t>(),
          row_count,
          static_cast<int>(level_count),
          static_cast<int>(allocation_count));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
"""


@lru_cache(maxsize=1)
def load_extension():
    return load_inline(
        name="qksieve_qmse_allocator_ext_20260807_v1",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=None,
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )


def allocate(
    costs: torch.Tensor,
    feasible_allocations: torch.Tensor,
    feasible_bit_indices: torch.Tensor,
    feasible_used_bits: torch.Tensor,
) -> torch.Tensor:
    return load_extension().allocate(
        costs.contiguous(),
        feasible_allocations.contiguous(),
        feasible_bit_indices.contiguous(),
        feasible_used_bits.contiguous(),
    )
