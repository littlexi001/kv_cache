from __future__ import annotations

from functools import lru_cache

import torch
from torch.utils.cpp_extension import load_inline


CPP_SOURCE = r"""
#include <torch/extension.h>

std::vector<torch::Tensor> qksieve_dual_mass_forward(
    torch::Tensor proxy_logits,
    torch::Tensor residual_log_risk,
    torch::Tensor calibration_slope,
    torch::Tensor calibration_intercept,
    double target_mass,
    int64_t floor_k,
    int64_t candidate_capacity);

std::vector<torch::Tensor> qksieve_dual_mass_value_tail_forward(
    torch::Tensor proxy_logits,
    torch::Tensor residual_log_risk,
    torch::Tensor calibration_slope,
    torch::Tensor calibration_intercept,
    torch::Tensor packed_value_codes,
    torch::Tensor value_minimum,
    torch::Tensor value_scale,
    double target_mass,
    int64_t floor_k,
    int64_t candidate_capacity,
    int64_t value_block_size);

void qksieve_dual_mass_value_tail_out(
    torch::Tensor proxy_logits,
    torch::Tensor residual_log_risk,
    torch::Tensor calibration_slope,
    torch::Tensor calibration_intercept,
    torch::Tensor packed_value_codes,
    torch::Tensor value_minimum,
    torch::Tensor value_scale,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor attention_thresholds,
    torch::Tensor risk_thresholds,
    torch::Tensor overflow,
    torch::Tensor tail_anchor_logits,
    torch::Tensor tail_denominator,
    torch::Tensor tail_coefficients,
    double target_mass,
    int64_t floor_k,
    int64_t value_block_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "qksieve_dual_mass_forward",
      &qksieve_dual_mass_forward,
      "Calibrated attention/residual-risk dual-mass selection");
  m.def(
      "qksieve_dual_mass_value_tail_forward",
      &qksieve_dual_mass_value_tail_forward,
      "Dual-mass selection with fused INT4 Value-tail moments");
  m.def(
      "qksieve_dual_mass_value_tail_out",
      &qksieve_dual_mass_value_tail_out,
      "Persistent-output dual-mass selection and Value-tail moments");
}
"""


CUDA_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cstdint>

constexpr int QKSIEVE_THREADS = 256;

__device__ __forceinline__ unsigned int ordered_float(float value) {
  unsigned int bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

__device__ __forceinline__ float calibrated_score(
    const float* proxy_row,
    int token,
    float slope,
    float intercept) {
  return fmaf(slope, proxy_row[token], intercept);
}

__device__ __forceinline__ int coarse_mass_boundary(
    const int* count_histogram,
    const float* weight_histogram,
    int floor_k,
    float target_mass) {
  int floor_boundary = 255;
  int selected = 0;
  for (int bin = 255; bin >= 0; --bin) {
    selected += count_histogram[bin];
    if (selected >= floor_k) {
      floor_boundary = bin;
      break;
    }
  }
  float total_weight = 0.0f;
  for (int bin = 0; bin < 256; ++bin) {
    total_weight += weight_histogram[bin];
  }
  float allowed_tail = fmaxf((1.0f - target_mass) * total_weight, 0.0f);
  float omitted = 0.0f;
  int mass_boundary = 255;
  for (int bin = 0; bin < 256; ++bin) {
    float next = omitted + weight_histogram[bin];
    if (next > allowed_tail) {
      mass_boundary = bin;
      break;
    }
    omitted = next;
  }
  return min(floor_boundary, mass_boundary);
}

__device__ __forceinline__ int fine_mass_boundary(
    const int* count_histogram,
    const float* weight_histogram,
    int floor_remaining,
    float allowed_tail_remaining) {
  int floor_boundary = 256;
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
  float omitted = 0.0f;
  int mass_boundary = 256;
  for (int bin = 0; bin < 256; ++bin) {
    float next = omitted + weight_histogram[bin];
    if (next > allowed_tail_remaining) {
      mass_boundary = bin;
      break;
    }
    omitted = next;
  }
  return min(255, min(floor_boundary, mass_boundary));
}

template <bool collect_value_tail>
__global__ void dual_mass_kernel(
    const float* __restrict__ proxy_logits,
    const float* __restrict__ residual_log_risk,
    const float* __restrict__ calibration_slope,
    const float* __restrict__ calibration_intercept,
    int64_t* __restrict__ candidate_indices,
    int64_t* __restrict__ candidate_counts,
    int64_t* __restrict__ attention_thresholds,
    int64_t* __restrict__ risk_thresholds,
    unsigned char* __restrict__ overflow,
    const unsigned char* __restrict__ packed_value_codes,
    const float* __restrict__ value_minimum,
    const float* __restrict__ value_scale,
    float* __restrict__ tail_anchor_logits,
    float* __restrict__ tail_denominator,
    float* __restrict__ tail_coefficients,
    int query_head_count,
    int kv_head_count,
    int history_count,
    int candidate_capacity,
    int floor_k,
    float target_mass,
    int value_rank,
    int value_block_count,
    int packed_value_capacity,
    int value_block_size) {
  __shared__ int attention_counts[256];
  __shared__ float attention_weights[256];
  __shared__ int risk_counts[256];
  __shared__ float risk_weights[256];
  __shared__ unsigned int attention_max_ordered;
  __shared__ unsigned int risk_max_ordered;
  __shared__ unsigned int attention_threshold;
  __shared__ unsigned int risk_threshold;
  __shared__ int attention_boundary_high;
  __shared__ int risk_boundary_high;
  __shared__ int attention_count_above;
  __shared__ int risk_count_above;
  __shared__ float attention_weight_below;
  __shared__ float risk_weight_below;
  __shared__ float attention_allowed_tail;
  __shared__ float risk_allowed_tail;
  __shared__ int selected_count;

  int row = blockIdx.x;
  int tid = threadIdx.x;
  int query_head = row % query_head_count;
  int batch = row / query_head_count;
  int groups = query_head_count / kv_head_count;
  int kv_head = query_head / groups;
  const float* proxy_row =
      proxy_logits + static_cast<int64_t>(row) * history_count;
  const float* risk_row = residual_log_risk +
      (static_cast<int64_t>(batch) * kv_head_count + kv_head) * history_count;
  float slope = calibration_slope[row];
  float intercept = calibration_intercept[row];

  if (tid == 0) {
    attention_max_ordered = ordered_float(-CUDART_INF_F);
    risk_max_ordered = ordered_float(-CUDART_INF_F);
    if constexpr (collect_value_tail) {
      tail_denominator[row] = 0.0f;
    }
  }
  if constexpr (collect_value_tail) {
    if (tid < value_rank) {
      tail_coefficients[
          static_cast<int64_t>(row) * value_rank + tid] = 0.0f;
    }
  }
  __syncthreads();
  float local_attention_max = -CUDART_INF_F;
  float local_risk_max = -CUDART_INF_F;
  for (int token = tid; token < history_count; token += blockDim.x) {
    float attention = calibrated_score(proxy_row, token, slope, intercept);
    float risk = attention + risk_row[token];
    local_attention_max = fmaxf(local_attention_max, attention);
    local_risk_max = fmaxf(local_risk_max, risk);
  }
  atomicMax(&attention_max_ordered, ordered_float(local_attention_max));
  atomicMax(&risk_max_ordered, ordered_float(local_risk_max));
  __syncthreads();
  unsigned int attention_max_bits =
      (attention_max_ordered & 0x80000000u)
          ? (attention_max_ordered ^ 0x80000000u)
          : ~attention_max_ordered;
  unsigned int risk_max_bits =
      (risk_max_ordered & 0x80000000u)
          ? (risk_max_ordered ^ 0x80000000u)
          : ~risk_max_ordered;
  float attention_max = __uint_as_float(attention_max_bits);
  float risk_max = __uint_as_float(risk_max_bits);

  attention_counts[tid] = 0;
  attention_weights[tid] = 0.0f;
  risk_counts[tid] = 0;
  risk_weights[tid] = 0.0f;
  __syncthreads();
  for (int token = tid; token < history_count; token += blockDim.x) {
    float attention = calibrated_score(proxy_row, token, slope, intercept);
    float risk = attention + risk_row[token];
    int attention_bin = static_cast<int>(ordered_float(attention) >> 24);
    int risk_bin = static_cast<int>(ordered_float(risk) >> 24);
    atomicAdd(attention_counts + attention_bin, 1);
    atomicAdd(
        attention_weights + attention_bin,
        __expf(attention - attention_max));
    atomicAdd(risk_counts + risk_bin, 1);
    atomicAdd(risk_weights + risk_bin, __expf(risk - risk_max));
  }
  __syncthreads();

  if (tid == 0) {
    attention_boundary_high = coarse_mass_boundary(
        attention_counts, attention_weights, floor_k, target_mass);
    risk_boundary_high = coarse_mass_boundary(
        risk_counts, risk_weights, floor_k, target_mass);
    float attention_total = 0.0f;
    float risk_total = 0.0f;
    attention_count_above = 0;
    risk_count_above = 0;
    attention_weight_below = 0.0f;
    risk_weight_below = 0.0f;
    for (int bin = 0; bin < 256; ++bin) {
      attention_total += attention_weights[bin];
      risk_total += risk_weights[bin];
      if (bin < attention_boundary_high) {
        attention_weight_below += attention_weights[bin];
      } else if (bin > attention_boundary_high) {
        attention_count_above += attention_counts[bin];
      }
      if (bin < risk_boundary_high) {
        risk_weight_below += risk_weights[bin];
      } else if (bin > risk_boundary_high) {
        risk_count_above += risk_counts[bin];
      }
    }
    attention_allowed_tail =
        fmaxf((1.0f - target_mass) * attention_total, 0.0f);
    risk_allowed_tail =
        fmaxf((1.0f - target_mass) * risk_total, 0.0f);
  }
  __syncthreads();

  attention_counts[tid] = 0;
  attention_weights[tid] = 0.0f;
  risk_counts[tid] = 0;
  risk_weights[tid] = 0.0f;
  __syncthreads();
  for (int token = tid; token < history_count; token += blockDim.x) {
    float attention = calibrated_score(proxy_row, token, slope, intercept);
    float risk = attention + risk_row[token];
    unsigned int ordered_attention = ordered_float(attention);
    unsigned int ordered_risk = ordered_float(risk);
    if (static_cast<int>(ordered_attention >> 24) ==
        attention_boundary_high) {
      int bin = static_cast<int>((ordered_attention >> 16) & 0xffu);
      atomicAdd(attention_counts + bin, 1);
      atomicAdd(
          attention_weights + bin,
          __expf(attention - attention_max));
    }
    if (static_cast<int>(ordered_risk >> 24) == risk_boundary_high) {
      int bin = static_cast<int>((ordered_risk >> 16) & 0xffu);
      atomicAdd(risk_counts + bin, 1);
      atomicAdd(risk_weights + bin, __expf(risk - risk_max));
    }
  }
  __syncthreads();

  if (tid == 0) {
    int attention_mid = fine_mass_boundary(
        attention_counts,
        attention_weights,
        max(0, floor_k - attention_count_above),
        fmaxf(attention_allowed_tail - attention_weight_below, 0.0f));
    int risk_mid = fine_mass_boundary(
        risk_counts,
        risk_weights,
        max(0, floor_k - risk_count_above),
        fmaxf(risk_allowed_tail - risk_weight_below, 0.0f));
    attention_threshold =
        (static_cast<unsigned int>(attention_boundary_high) << 24)
        | (static_cast<unsigned int>(attention_mid) << 16);
    risk_threshold =
        (static_cast<unsigned int>(risk_boundary_high) << 24)
        | (static_cast<unsigned int>(risk_mid) << 16);
    attention_thresholds[row] = static_cast<int64_t>(attention_threshold);
    risk_thresholds[row] = static_cast<int64_t>(risk_threshold);
    if constexpr (collect_value_tail) {
      tail_anchor_logits[row] = attention_max;
    }
    selected_count = 0;
  }
  __syncthreads();

  int lane = tid & 31;
  int64_t* index_row = candidate_indices
      + static_cast<int64_t>(row) * candidate_capacity;
  float local_tail_denominator = 0.0f;
  float local_tail_coefficients[16] = {
      0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f};
  for (int token_base = 0; token_base < history_count;
       token_base += blockDim.x) {
    int token = token_base + tid;
    bool valid = token < history_count;
    float attention = valid
        ? calibrated_score(proxy_row, token, slope, intercept)
        : -CUDART_INF_F;
    float risk = valid ? attention + risk_row[token] : -CUDART_INF_F;
    bool keep = valid
        && (ordered_float(attention) >= attention_threshold
            || ordered_float(risk) >= risk_threshold);
    unsigned int active = __ballot_sync(0xffffffffu, keep);
    int active_count = __popc(active);
    int base = 0;
    if (lane == 0 && active_count > 0) {
      base = atomicAdd(&selected_count, active_count);
    }
    base = __shfl_sync(0xffffffffu, base, 0);
    if (keep) {
      unsigned int lower = lane == 0 ? 0u : ((1u << lane) - 1u);
      int position = base + __popc(active & lower);
      if (position < candidate_capacity) {
        index_row[position] = static_cast<int64_t>(token);
      }
    } else if constexpr (collect_value_tail) {
      if (!valid) {
        continue;
      }
      float weight = __expf(attention - attention_max);
      local_tail_denominator += weight;
      int batch_kv = batch * kv_head_count + kv_head;
      int value_block = token / value_block_size;
      const unsigned char* code_row = packed_value_codes
          + (static_cast<int64_t>(batch_kv) * packed_value_capacity + token)
              * (value_rank / 2);
      const float* minimum_row = value_minimum
          + (static_cast<int64_t>(batch_kv) * value_block_count
              + value_block) * value_rank;
      const float* scale_row = value_scale
          + (static_cast<int64_t>(batch_kv) * value_block_count
              + value_block) * value_rank;
#pragma unroll
      for (int rank = 0; rank < 16; ++rank) {
        unsigned char packed = code_row[rank >> 1];
        int code = (packed >> (4 * (rank & 1))) & 0x0f;
        float coefficient = fmaf(
            static_cast<float>(code), scale_row[rank], minimum_row[rank]);
        local_tail_coefficients[rank] += weight * coefficient;
      }
    }
  }
  if constexpr (collect_value_tail) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      local_tail_denominator += __shfl_down_sync(
          0xffffffffu, local_tail_denominator, offset);
#pragma unroll
      for (int rank = 0; rank < 16; ++rank) {
        local_tail_coefficients[rank] += __shfl_down_sync(
            0xffffffffu, local_tail_coefficients[rank], offset);
      }
    }
    if (lane == 0) {
      atomicAdd(tail_denominator + row, local_tail_denominator);
#pragma unroll
      for (int rank = 0; rank < 16; ++rank) {
        atomicAdd(
            tail_coefficients + static_cast<int64_t>(row) * 16 + rank,
            local_tail_coefficients[rank]);
      }
    }
  }
  __syncthreads();
  if (tid == 0) {
    int retained = min(selected_count, candidate_capacity);
    candidate_counts[row] = static_cast<int64_t>(retained);
    overflow[row] = static_cast<unsigned char>(
        selected_count > candidate_capacity);
  }
}

std::vector<torch::Tensor> qksieve_dual_mass_forward(
    torch::Tensor proxy_logits,
    torch::Tensor residual_log_risk,
    torch::Tensor calibration_slope,
    torch::Tensor calibration_intercept,
    double target_mass,
    int64_t floor_k,
    int64_t candidate_capacity) {
  TORCH_CHECK(proxy_logits.is_cuda(), "proxy logits must be CUDA");
  TORCH_CHECK(proxy_logits.dim() == 3, "proxy logits must be [B,Hq,N]");
  TORCH_CHECK(proxy_logits.scalar_type() == at::kFloat, "proxy logits must be float32");
  TORCH_CHECK(residual_log_risk.dim() == 3, "residual risk must be [B,Hkv,N]");
  TORCH_CHECK(residual_log_risk.scalar_type() == at::kFloat, "residual risk must be float32");
  TORCH_CHECK(residual_log_risk.size(0) == proxy_logits.size(0), "batch mismatch");
  TORCH_CHECK(residual_log_risk.size(2) == proxy_logits.size(2), "history mismatch");
  TORCH_CHECK(calibration_slope.sizes() == proxy_logits.sizes().slice(0, 2), "slope must be [B,Hq]");
  TORCH_CHECK(calibration_intercept.sizes() == calibration_slope.sizes(), "intercept must match slope");
  TORCH_CHECK(calibration_slope.scalar_type() == at::kFloat, "slope must be float32");
  TORCH_CHECK(calibration_intercept.scalar_type() == at::kFloat, "intercept must be float32");
  TORCH_CHECK(target_mass > 0.0 && target_mass < 1.0, "invalid target mass");

  int batch_count = static_cast<int>(proxy_logits.size(0));
  int query_head_count = static_cast<int>(proxy_logits.size(1));
  int kv_head_count = static_cast<int>(residual_log_risk.size(1));
  int history_count = static_cast<int>(proxy_logits.size(2));
  TORCH_CHECK(query_head_count % kv_head_count == 0, "invalid GQA grouping");
  TORCH_CHECK(floor_k > 0 && floor_k <= history_count, "invalid floor_k");
  TORCH_CHECK(candidate_capacity > 0 && candidate_capacity <= history_count, "invalid capacity");

  auto proxy = proxy_logits.contiguous();
  auto risk = residual_log_risk.contiguous();
  auto slope = calibration_slope.contiguous();
  auto intercept = calibration_intercept.contiguous();
  c10::cuda::CUDAGuard device_guard(proxy.device());
  // Reference consumers gather the padded tensor before applying counts, so
  // unused entries must remain valid token ids. The fused attention kernel can
  // later consume a persistent pre-zeroed workspace instead.
  auto indices = torch::zeros(
      {batch_count, query_head_count, candidate_capacity},
      proxy.options().dtype(at::kLong));
  auto counts = torch::empty(
      {batch_count, query_head_count},
      proxy.options().dtype(at::kLong));
  auto attention_thresholds = torch::empty_like(counts);
  auto risk_thresholds = torch::empty_like(counts);
  auto overflow = torch::empty(
      {batch_count, query_head_count},
      proxy.options().dtype(at::kByte));

  dual_mass_kernel<false><<<
      batch_count * query_head_count,
      QKSIEVE_THREADS,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
          proxy.data_ptr<float>(),
          risk.data_ptr<float>(),
          slope.data_ptr<float>(),
          intercept.data_ptr<float>(),
          indices.data_ptr<int64_t>(),
          counts.data_ptr<int64_t>(),
          attention_thresholds.data_ptr<int64_t>(),
          risk_thresholds.data_ptr<int64_t>(),
          overflow.data_ptr<unsigned char>(),
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          query_head_count,
          kv_head_count,
          history_count,
          static_cast<int>(candidate_capacity),
          static_cast<int>(floor_k),
          static_cast<float>(target_mass),
          0,
          0,
          0,
          0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {
      indices,
      counts,
      attention_thresholds,
      risk_thresholds,
      overflow};
}

std::vector<torch::Tensor> qksieve_dual_mass_value_tail_forward(
    torch::Tensor proxy_logits,
    torch::Tensor residual_log_risk,
    torch::Tensor calibration_slope,
    torch::Tensor calibration_intercept,
    torch::Tensor packed_value_codes,
    torch::Tensor value_minimum,
    torch::Tensor value_scale,
    double target_mass,
    int64_t floor_k,
    int64_t candidate_capacity,
    int64_t value_block_size) {
  TORCH_CHECK(proxy_logits.is_cuda(), "proxy logits must be CUDA");
  TORCH_CHECK(proxy_logits.dim() == 3, "proxy logits must be [B,Hq,N]");
  TORCH_CHECK(proxy_logits.scalar_type() == at::kFloat, "proxy logits must be float32");
  TORCH_CHECK(residual_log_risk.dim() == 3, "residual risk must be [B,Hkv,N]");
  TORCH_CHECK(residual_log_risk.scalar_type() == at::kFloat, "residual risk must be float32");
  TORCH_CHECK(calibration_slope.scalar_type() == at::kFloat, "slope must be float32");
  TORCH_CHECK(calibration_intercept.scalar_type() == at::kFloat, "intercept must be float32");
  TORCH_CHECK(packed_value_codes.dim() == 4, "Value codes must be [B,Hkv,C,R/2]");
  TORCH_CHECK(packed_value_codes.scalar_type() == at::kByte, "Value codes must be uint8");
  TORCH_CHECK(value_minimum.dim() == 4 && value_scale.dim() == 4, "Value metadata must be rank four");
  TORCH_CHECK(value_minimum.scalar_type() == at::kFloat, "Value minimum must be float32");
  TORCH_CHECK(value_scale.scalar_type() == at::kFloat, "Value scale must be float32");
  TORCH_CHECK(value_minimum.sizes() == value_scale.sizes(), "Value metadata must align");
  TORCH_CHECK(target_mass > 0.0 && target_mass < 1.0, "invalid target mass");

  int batch_count = static_cast<int>(proxy_logits.size(0));
  int query_head_count = static_cast<int>(proxy_logits.size(1));
  int history_count = static_cast<int>(proxy_logits.size(2));
  int kv_head_count = static_cast<int>(residual_log_risk.size(1));
  int packed_value_capacity = static_cast<int>(packed_value_codes.size(2));
  int value_rank = static_cast<int>(packed_value_codes.size(3)) * 2;
  int value_block_count = static_cast<int>(value_minimum.size(2));
  TORCH_CHECK(value_rank == 16, "fused Value tail currently requires rank 16");
  TORCH_CHECK(query_head_count % kv_head_count == 0, "invalid GQA grouping");
  TORCH_CHECK(residual_log_risk.size(0) == batch_count
                  && residual_log_risk.size(2) == history_count,
              "residual-risk shape mismatch");
  TORCH_CHECK(calibration_slope.sizes() == proxy_logits.sizes().slice(0, 2), "slope must be [B,Hq]");
  TORCH_CHECK(calibration_intercept.sizes() == calibration_slope.sizes(), "intercept must match slope");
  TORCH_CHECK(packed_value_codes.size(0) == batch_count
                  && packed_value_codes.size(1) == kv_head_count
                  && packed_value_capacity >= history_count,
              "Value-code shape mismatch");
  TORCH_CHECK(value_minimum.size(0) == batch_count
                  && value_minimum.size(1) == kv_head_count
                  && value_minimum.size(3) == value_rank,
              "Value-metadata shape mismatch");
  TORCH_CHECK(value_block_size > 0
                  && value_block_count * value_block_size >= history_count,
              "invalid Value block layout");
  TORCH_CHECK(floor_k > 0 && floor_k <= history_count, "invalid floor_k");
  TORCH_CHECK(candidate_capacity > 0 && candidate_capacity <= history_count, "invalid capacity");

  auto proxy = proxy_logits.contiguous();
  auto risk = residual_log_risk.contiguous();
  auto slope = calibration_slope.contiguous();
  auto intercept = calibration_intercept.contiguous();
  auto value_codes = packed_value_codes.contiguous();
  auto value_min = value_minimum.contiguous();
  auto value_step = value_scale.contiguous();
  c10::cuda::CUDAGuard device_guard(proxy.device());
  auto indices = torch::zeros(
      {batch_count, query_head_count, candidate_capacity},
      proxy.options().dtype(at::kLong));
  auto counts = torch::empty(
      {batch_count, query_head_count},
      proxy.options().dtype(at::kLong));
  auto attention_thresholds = torch::empty_like(counts);
  auto risk_thresholds = torch::empty_like(counts);
  auto overflow = torch::empty(
      {batch_count, query_head_count},
      proxy.options().dtype(at::kByte));
  auto tail_anchor_logits = torch::empty(
      {batch_count, query_head_count}, proxy.options());
  auto tail_denominator = torch::empty_like(tail_anchor_logits);
  auto tail_coefficients = torch::empty(
      {batch_count, query_head_count, value_rank}, proxy.options());

  dual_mass_kernel<true><<<
      batch_count * query_head_count,
      QKSIEVE_THREADS,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
          proxy.data_ptr<float>(),
          risk.data_ptr<float>(),
          slope.data_ptr<float>(),
          intercept.data_ptr<float>(),
          indices.data_ptr<int64_t>(),
          counts.data_ptr<int64_t>(),
          attention_thresholds.data_ptr<int64_t>(),
          risk_thresholds.data_ptr<int64_t>(),
          overflow.data_ptr<unsigned char>(),
          value_codes.data_ptr<unsigned char>(),
          value_min.data_ptr<float>(),
          value_step.data_ptr<float>(),
          tail_anchor_logits.data_ptr<float>(),
          tail_denominator.data_ptr<float>(),
          tail_coefficients.data_ptr<float>(),
          query_head_count,
          kv_head_count,
          history_count,
          static_cast<int>(candidate_capacity),
          static_cast<int>(floor_k),
          static_cast<float>(target_mass),
          value_rank,
          value_block_count,
          packed_value_capacity,
          static_cast<int>(value_block_size));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {
      indices,
      counts,
      attention_thresholds,
      risk_thresholds,
      overflow,
      tail_anchor_logits,
      tail_denominator,
      tail_coefficients};
}

void qksieve_dual_mass_value_tail_out(
    torch::Tensor proxy_logits,
    torch::Tensor residual_log_risk,
    torch::Tensor calibration_slope,
    torch::Tensor calibration_intercept,
    torch::Tensor packed_value_codes,
    torch::Tensor value_minimum,
    torch::Tensor value_scale,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor attention_thresholds,
    torch::Tensor risk_thresholds,
    torch::Tensor overflow,
    torch::Tensor tail_anchor_logits,
    torch::Tensor tail_denominator,
    torch::Tensor tail_coefficients,
    double target_mass,
    int64_t floor_k,
    int64_t value_block_size) {
  TORCH_CHECK(proxy_logits.is_cuda(), "proxy logits must be CUDA");
  TORCH_CHECK(proxy_logits.dim() == 3, "proxy logits must be [B,Hq,N]");
  TORCH_CHECK(proxy_logits.scalar_type() == at::kFloat, "proxy logits must be float32");
  TORCH_CHECK(residual_log_risk.scalar_type() == at::kFloat, "residual risk must be float32");
  TORCH_CHECK(calibration_slope.scalar_type() == at::kFloat, "slope must be float32");
  TORCH_CHECK(calibration_intercept.scalar_type() == at::kFloat, "intercept must be float32");
  TORCH_CHECK(packed_value_codes.scalar_type() == at::kByte, "Value codes must be uint8");
  TORCH_CHECK(value_minimum.scalar_type() == at::kFloat
                  && value_scale.scalar_type() == at::kFloat,
              "Value metadata must be float32");
  TORCH_CHECK(candidate_indices.scalar_type() == at::kLong
                  && candidate_counts.scalar_type() == at::kLong
                  && attention_thresholds.scalar_type() == at::kLong
                  && risk_thresholds.scalar_type() == at::kLong,
              "candidate outputs must use int64");
  TORCH_CHECK(overflow.scalar_type() == at::kByte,
              "overflow output must use uint8");
  TORCH_CHECK(tail_anchor_logits.scalar_type() == at::kFloat
                  && tail_denominator.scalar_type() == at::kFloat
                  && tail_coefficients.scalar_type() == at::kFloat,
              "tail outputs must be float32");
  TORCH_CHECK(proxy_logits.is_contiguous()
                  && residual_log_risk.is_contiguous()
                  && calibration_slope.is_contiguous()
                  && calibration_intercept.is_contiguous()
                  && packed_value_codes.is_contiguous()
                  && value_minimum.is_contiguous()
                  && value_scale.is_contiguous()
                  && candidate_indices.is_contiguous()
                  && candidate_counts.is_contiguous()
                  && attention_thresholds.is_contiguous()
                  && risk_thresholds.is_contiguous()
                  && overflow.is_contiguous()
                  && tail_anchor_logits.is_contiguous()
                  && tail_denominator.is_contiguous()
                  && tail_coefficients.is_contiguous(),
              "persistent dual-mass tensors must be contiguous");

  int batch_count = static_cast<int>(proxy_logits.size(0));
  int query_head_count = static_cast<int>(proxy_logits.size(1));
  int history_count = static_cast<int>(proxy_logits.size(2));
  int kv_head_count = static_cast<int>(residual_log_risk.size(1));
  int packed_value_capacity = static_cast<int>(packed_value_codes.size(2));
  int value_rank = static_cast<int>(packed_value_codes.size(3)) * 2;
  int value_block_count = static_cast<int>(value_minimum.size(2));
  int candidate_capacity = static_cast<int>(candidate_indices.size(2));
  TORCH_CHECK(value_rank == 16, "fused Value tail currently requires rank 16");
  TORCH_CHECK(query_head_count % kv_head_count == 0, "invalid GQA grouping");
  TORCH_CHECK(residual_log_risk.dim() == 3
                  && residual_log_risk.size(0) == batch_count
                  && residual_log_risk.size(1) == kv_head_count
                  && residual_log_risk.size(2) == history_count,
              "residual-risk shape mismatch");
  TORCH_CHECK(calibration_slope.dim() == 2
                  && calibration_slope.size(0) == batch_count
                  && calibration_slope.size(1) == query_head_count
                  && calibration_intercept.sizes() == calibration_slope.sizes(),
              "calibration shape mismatch");
  TORCH_CHECK(packed_value_codes.size(0) == batch_count
                  && packed_value_codes.size(1) == kv_head_count
                  && packed_value_capacity >= history_count,
              "Value-code shape mismatch");
  TORCH_CHECK(value_minimum.sizes() == value_scale.sizes()
                  && value_minimum.size(0) == batch_count
                  && value_minimum.size(1) == kv_head_count
                  && value_minimum.size(3) == value_rank,
              "Value-metadata shape mismatch");
  TORCH_CHECK(candidate_indices.size(0) == batch_count
                  && candidate_indices.size(1) == query_head_count
                  && candidate_capacity > 0
                  && candidate_capacity <= history_count,
              "candidate-index output shape mismatch");
  TORCH_CHECK(candidate_counts.dim() == 2
                  && candidate_counts.size(0) == batch_count
                  && candidate_counts.size(1) == query_head_count
                  && attention_thresholds.sizes() == candidate_counts.sizes()
                  && risk_thresholds.sizes() == candidate_counts.sizes()
                  && overflow.sizes() == candidate_counts.sizes()
                  && tail_anchor_logits.sizes() == candidate_counts.sizes()
                  && tail_denominator.sizes() == candidate_counts.sizes(),
              "row output shape mismatch");
  TORCH_CHECK(tail_coefficients.dim() == 3
                  && tail_coefficients.size(0) == batch_count
                  && tail_coefficients.size(1) == query_head_count
                  && tail_coefficients.size(2) == value_rank,
              "tail-coefficient output shape mismatch");
  TORCH_CHECK(value_block_size > 0
                  && value_block_count * value_block_size >= history_count,
              "invalid Value block layout");
  TORCH_CHECK(floor_k > 0 && floor_k <= history_count, "invalid floor_k");
  TORCH_CHECK(target_mass > 0.0 && target_mass < 1.0, "invalid target mass");

  c10::cuda::CUDAGuard device_guard(proxy_logits.device());
  dual_mass_kernel<true><<<
      batch_count * query_head_count,
      QKSIEVE_THREADS,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
          proxy_logits.data_ptr<float>(),
          residual_log_risk.data_ptr<float>(),
          calibration_slope.data_ptr<float>(),
          calibration_intercept.data_ptr<float>(),
          candidate_indices.data_ptr<int64_t>(),
          candidate_counts.data_ptr<int64_t>(),
          attention_thresholds.data_ptr<int64_t>(),
          risk_thresholds.data_ptr<int64_t>(),
          overflow.data_ptr<unsigned char>(),
          packed_value_codes.data_ptr<unsigned char>(),
          value_minimum.data_ptr<float>(),
          value_scale.data_ptr<float>(),
          tail_anchor_logits.data_ptr<float>(),
          tail_denominator.data_ptr<float>(),
          tail_coefficients.data_ptr<float>(),
          query_head_count,
          kv_head_count,
          history_count,
          candidate_capacity,
          static_cast<int>(floor_k),
          static_cast<float>(target_mass),
          value_rank,
          value_block_count,
          packed_value_capacity,
          static_cast<int>(value_block_size));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


@lru_cache(maxsize=1)
def load_extension() -> object:
    return load_inline(
        name="qksieve_dual_mass_cuda_20260804_v4",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=None,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )


def dual_mass_candidates(
    proxy_logits: torch.Tensor,
    residual_log_risk: torch.Tensor,
    calibration_slope: torch.Tensor,
    calibration_intercept: torch.Tensor,
    target_mass: float,
    floor_k: int,
    candidate_capacity: int | None = None,
) -> tuple[torch.Tensor, ...]:
    """Select a calibrated attention/risk union with two-level histograms."""

    if proxy_logits.ndim != 3 or residual_log_risk.ndim != 3:
        raise ValueError("proxy and residual-risk tensors must be rank three")
    if proxy_logits.shape[0] != residual_log_risk.shape[0]:
        raise ValueError("proxy and residual-risk batches must match")
    if proxy_logits.shape[-1] != residual_log_risk.shape[-1]:
        raise ValueError("proxy and residual-risk histories must match")
    if proxy_logits.shape[1] % residual_log_risk.shape[1]:
        raise ValueError("query heads must be divisible by KV heads")
    if not proxy_logits.is_cuda:
        raise ValueError("dual-mass histogram selection requires CUDA")
    history_count = int(proxy_logits.shape[-1])
    capacity = (
        history_count
        if candidate_capacity is None
        else int(candidate_capacity)
    )
    return load_extension().qksieve_dual_mass_forward(
        proxy_logits.float().contiguous(),
        residual_log_risk.float().contiguous(),
        calibration_slope.float().contiguous(),
        calibration_intercept.float().contiguous(),
        float(target_mass),
        int(floor_k),
        capacity,
    )


def dual_mass_candidates_with_value_tail(
    proxy_logits: torch.Tensor,
    residual_log_risk: torch.Tensor,
    calibration_slope: torch.Tensor,
    calibration_intercept: torch.Tensor,
    packed_value_codes: torch.Tensor,
    value_minimum: torch.Tensor,
    value_scale: torch.Tensor,
    target_mass: float,
    floor_k: int,
    candidate_capacity: int | None = None,
    value_block_size: int = 256,
) -> tuple[torch.Tensor, ...]:
    """Fuse dual-mass compaction with rank-16 INT4 Value-tail moments."""

    if proxy_logits.ndim != 3 or residual_log_risk.ndim != 3:
        raise ValueError("proxy and residual-risk tensors must be rank three")
    if packed_value_codes.ndim != 4:
        raise ValueError("packed Value codes must have shape [B,Hkv,C,R/2]")
    if value_minimum.shape != value_scale.shape:
        raise ValueError("Value minimum and scale tensors must align")
    if packed_value_codes.shape[-1] != 8:
        raise ValueError("fused Value-tail moments require rank 16")
    if not proxy_logits.is_cuda:
        raise ValueError("dual-mass histogram selection requires CUDA")
    history_count = int(proxy_logits.shape[-1])
    capacity = (
        history_count
        if candidate_capacity is None
        else int(candidate_capacity)
    )
    return load_extension().qksieve_dual_mass_value_tail_forward(
        proxy_logits.float().contiguous(),
        residual_log_risk.float().contiguous(),
        calibration_slope.float().contiguous(),
        calibration_intercept.float().contiguous(),
        packed_value_codes.contiguous(),
        value_minimum.float().contiguous(),
        value_scale.float().contiguous(),
        float(target_mass),
        int(floor_k),
        capacity,
        int(value_block_size),
    )


def dual_mass_candidates_with_value_tail_out(
    proxy_logits: torch.Tensor,
    residual_log_risk: torch.Tensor,
    calibration_slope: torch.Tensor,
    calibration_intercept: torch.Tensor,
    packed_value_codes: torch.Tensor,
    value_minimum: torch.Tensor,
    value_scale: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_counts: torch.Tensor,
    attention_thresholds: torch.Tensor,
    risk_thresholds: torch.Tensor,
    overflow: torch.Tensor,
    tail_anchor_logits: torch.Tensor,
    tail_denominator: torch.Tensor,
    tail_coefficients: torch.Tensor,
    target_mass: float,
    floor_k: int,
    value_block_size: int = 256,
) -> tuple[torch.Tensor, ...]:
    """Write dual-mass and Value-tail outputs into persistent CUDA buffers."""

    inputs = (
        proxy_logits,
        residual_log_risk,
        calibration_slope,
        calibration_intercept,
        value_minimum,
        value_scale,
    )
    if any(tensor.dtype != torch.float32 for tensor in inputs):
        raise ValueError("persistent dual-mass floating inputs must be float32")
    if any(not tensor.is_contiguous() for tensor in inputs):
        raise ValueError("persistent dual-mass inputs must be contiguous")
    if packed_value_codes.dtype != torch.uint8 or not packed_value_codes.is_contiguous():
        raise ValueError("packed Value codes must be contiguous uint8")
    load_extension().qksieve_dual_mass_value_tail_out(
        proxy_logits,
        residual_log_risk,
        calibration_slope,
        calibration_intercept,
        packed_value_codes,
        value_minimum,
        value_scale,
        candidate_indices,
        candidate_counts,
        attention_thresholds,
        risk_thresholds,
        overflow,
        tail_anchor_logits,
        tail_denominator,
        tail_coefficients,
        float(target_mass),
        int(floor_k),
        int(value_block_size),
    )
    return (
        candidate_indices,
        candidate_counts,
        attention_thresholds,
        risk_thresholds,
        overflow,
        tail_anchor_logits,
        tail_denominator,
        tail_coefficients,
    )
