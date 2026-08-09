from __future__ import annotations

from functools import lru_cache

import torch
from torch.utils.cpp_extension import load_inline


CPP_SOURCE = r"""
#include <torch/extension.h>

std::vector<torch::Tensor> qksieve_prefix_tail_budget_forward(
    torch::Tensor scores,
    torch::Tensor tail_weights,
    torch::Tensor allowed_tail,
    int64_t floor_k);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "qksieve_prefix_tail_budget_forward",
      &qksieve_prefix_tail_budget_forward,
      "Conservative two-level histogram proxy-prefix selection");
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
#include <cstdint>

constexpr int QKSIEVE_THREADS = 256;

__device__ __forceinline__ unsigned int ordered_float(float value) {
  unsigned int bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

__global__ void prefix_tail_threshold_kernel(
    const float* __restrict__ scores,
    const float* __restrict__ tail_weights,
    const float* __restrict__ allowed_tail,
    int64_t* __restrict__ ordered_thresholds,
    int64_t* __restrict__ candidate_counts,
    int history_count,
    int floor_k) {
  __shared__ int count_histogram[256];
  __shared__ float weight_histogram[256];
  __shared__ int boundary_high;
  __shared__ int count_above_high;
  __shared__ float weight_below_high;
  __shared__ unsigned int ordered_threshold;
  __shared__ int selected_count;

  int row = blockIdx.x;
  int tid = threadIdx.x;
  const float* score_row = scores + static_cast<int64_t>(row) * history_count;
  const float* weight_row =
      tail_weights + static_cast<int64_t>(row) * history_count;

  count_histogram[tid] = 0;
  weight_histogram[tid] = 0.0f;
  __syncthreads();
  for (int token = tid; token < history_count; token += blockDim.x) {
    unsigned int ordered = ordered_float(score_row[token]);
    int bin = static_cast<int>(ordered >> 24);
    atomicAdd(count_histogram + bin, 1);
    atomicAdd(weight_histogram + bin, fmaxf(weight_row[token], 0.0f));
  }
  __syncthreads();

  if (tid == 0) {
    int floor_boundary = 255;
    int selected = 0;
    for (int bin = 255; bin >= 0; --bin) {
      selected += count_histogram[bin];
      if (selected >= floor_k) {
        floor_boundary = bin;
        break;
      }
    }
    int risk_boundary = 256;
    float omitted = 0.0f;
    float allowed = fmaxf(allowed_tail[row], 0.0f);
    for (int bin = 0; bin < 256; ++bin) {
      float next = omitted + weight_histogram[bin];
      if (next > allowed) {
        risk_boundary = bin;
        break;
      }
      omitted = next;
    }
    boundary_high = min(floor_boundary, risk_boundary);
    count_above_high = 0;
    weight_below_high = 0.0f;
    for (int bin = boundary_high + 1; bin < 256; ++bin) {
      count_above_high += count_histogram[bin];
    }
    for (int bin = 0; bin < boundary_high; ++bin) {
      weight_below_high += weight_histogram[bin];
    }
  }
  __syncthreads();

  count_histogram[tid] = 0;
  weight_histogram[tid] = 0.0f;
  __syncthreads();
  for (int token = tid; token < history_count; token += blockDim.x) {
    unsigned int ordered = ordered_float(score_row[token]);
    if (static_cast<int>(ordered >> 24) == boundary_high) {
      int bin = static_cast<int>((ordered >> 16) & 0xffu);
      atomicAdd(count_histogram + bin, 1);
      atomicAdd(weight_histogram + bin, fmaxf(weight_row[token], 0.0f));
    }
  }
  __syncthreads();

  if (tid == 0) {
    int floor_boundary = 256;
    int floor_remaining = max(0, floor_k - count_above_high);
    if (floor_remaining > 0) {
      int selected = 0;
      for (int bin = 255; bin >= 0; --bin) {
        selected += count_histogram[bin];
        if (selected >= floor_remaining) {
          floor_boundary = bin;
          break;
        }
      }
    }
    int risk_boundary = 256;
    float remaining = fmaxf(allowed_tail[row] - weight_below_high, 0.0f);
    float omitted = 0.0f;
    for (int bin = 0; bin < 256; ++bin) {
      float next = omitted + weight_histogram[bin];
      if (next > remaining) {
        risk_boundary = bin;
        break;
      }
      omitted = next;
    }
    int boundary_mid = min(floor_boundary, risk_boundary);
    if (boundary_mid > 255) {
      boundary_mid = 255;
    }
    ordered_threshold =
        (static_cast<unsigned int>(boundary_high) << 24)
        | (static_cast<unsigned int>(boundary_mid) << 16);
    selected_count = 0;
  }
  __syncthreads();

  int local_count = 0;
  for (int token = tid; token < history_count; token += blockDim.x) {
    local_count += static_cast<int>(
        ordered_float(score_row[token]) >= ordered_threshold);
  }
  atomicAdd(&selected_count, local_count);
  __syncthreads();
  if (tid == 0) {
    ordered_thresholds[row] = static_cast<int64_t>(ordered_threshold);
    candidate_counts[row] = static_cast<int64_t>(selected_count);
  }
}

__global__ void prefix_tail_compact_kernel(
    const float* __restrict__ scores,
    const int64_t* __restrict__ ordered_thresholds,
    int64_t* __restrict__ candidate_indices,
    int* __restrict__ cursors,
    int history_count,
    int candidate_capacity) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int lane = tid & 31;
  const float* score_row = scores + static_cast<int64_t>(row) * history_count;
  int64_t* index_row =
      candidate_indices + static_cast<int64_t>(row) * candidate_capacity;
  unsigned int threshold = static_cast<unsigned int>(ordered_thresholds[row]);
  for (int token = tid; token < history_count; token += blockDim.x) {
    bool keep = ordered_float(score_row[token]) >= threshold;
    unsigned int active = __ballot_sync(0xffffffffu, keep);
    int active_count = __popc(active);
    int base = 0;
    if (lane == 0 && active_count > 0) {
      base = atomicAdd(cursors + row, active_count);
    }
    base = __shfl_sync(0xffffffffu, base, 0);
    if (keep) {
      unsigned int lower = lane == 0 ? 0u : ((1u << lane) - 1u);
      int position = base + __popc(active & lower);
      if (position < candidate_capacity) {
        index_row[position] = static_cast<int64_t>(token);
      }
    }
  }
}

std::vector<torch::Tensor> qksieve_prefix_tail_budget_forward(
    torch::Tensor scores,
    torch::Tensor tail_weights,
    torch::Tensor allowed_tail,
    int64_t floor_k) {
  TORCH_CHECK(scores.is_cuda(), "scores must be CUDA");
  TORCH_CHECK(scores.dim() == 3, "scores must be [B,H,N]");
  TORCH_CHECK(scores.scalar_type() == at::kFloat, "scores must be float32");
  TORCH_CHECK(tail_weights.sizes() == scores.sizes(), "weights must match scores");
  TORCH_CHECK(tail_weights.scalar_type() == at::kFloat, "weights must be float32");
  TORCH_CHECK(allowed_tail.sizes() == scores.sizes().slice(0, 2), "allowed tail must be [B,H]");
  TORCH_CHECK(allowed_tail.scalar_type() == at::kFloat, "allowed tail must be float32");
  int batch_count = static_cast<int>(scores.size(0));
  int head_count = static_cast<int>(scores.size(1));
  int history_count = static_cast<int>(scores.size(2));
  TORCH_CHECK(floor_k > 0 && floor_k <= history_count, "invalid floor_k");

  auto scores_c = scores.contiguous();
  auto weights_c = tail_weights.contiguous();
  auto allowed_c = allowed_tail.contiguous();
  c10::cuda::CUDAGuard device_guard(scores_c.device());
  auto thresholds = torch::empty(
      {batch_count, head_count}, scores_c.options().dtype(at::kLong));
  auto counts = torch::empty_like(thresholds);
  prefix_tail_threshold_kernel<<<
      batch_count * head_count,
      QKSIEVE_THREADS,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
          scores_c.data_ptr<float>(),
          weights_c.data_ptr<float>(),
          allowed_c.data_ptr<float>(),
          thresholds.data_ptr<int64_t>(),
          counts.data_ptr<int64_t>(),
          history_count,
          static_cast<int>(floor_k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  int64_t candidate_capacity = counts.max().item<int64_t>();
  auto indices = torch::zeros(
      {batch_count, head_count, candidate_capacity},
      scores_c.options().dtype(at::kLong));
  auto cursors = torch::zeros(
      {batch_count, head_count}, scores_c.options().dtype(at::kInt));
  prefix_tail_compact_kernel<<<
      batch_count * head_count,
      QKSIEVE_THREADS,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
          scores_c.data_ptr<float>(),
          thresholds.data_ptr<int64_t>(),
          indices.data_ptr<int64_t>(),
          cursors.data_ptr<int>(),
          history_count,
          static_cast<int>(candidate_capacity));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {indices, counts, thresholds};
}
"""


@lru_cache(maxsize=1)
def load_extension() -> object:
    return load_inline(
        name="qksieve_prefix_budget_cuda_20260803_v1",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=None,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )


def prefix_tail_budget_candidates(
    scores: torch.Tensor,
    tail_weights: torch.Tensor,
    allowed_tail: torch.Tensor,
    floor_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select a conservative score prefix using an O(N) histogram scan."""

    if scores.ndim != 3 or tail_weights.shape != scores.shape:
        raise ValueError("scores and tail weights must have shape [B,H,N]")
    if allowed_tail.shape != scores.shape[:-1]:
        raise ValueError("allowed tail must have shape [B,H]")
    if not scores.is_cuda:
        raise ValueError("histogram prefix selection requires CUDA")
    return load_extension().qksieve_prefix_tail_budget_forward(
        scores.float().contiguous(),
        tail_weights.float().contiguous(),
        allowed_tail.float().contiguous(),
        int(floor_k),
    )
