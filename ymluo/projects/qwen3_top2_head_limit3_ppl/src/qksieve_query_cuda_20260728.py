from __future__ import annotations

from functools import lru_cache

import torch
from torch.utils.cpp_extension import load_inline


HEAD_DIM = 128
BAND_DIM = 16
BAND_COUNT = HEAD_DIM // BAND_DIM


CPP_SOURCE = r"""
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> qksieve_project_quantize_forward(
    torch::Tensor grouped_query,
    torch::Tensor basis);

std::vector<torch::Tensor> qksieve_project_quantize_wmma_forward(
    torch::Tensor grouped_query,
    torch::Tensor basis);

void qksieve_project_quantize_wmma_out(
    torch::Tensor grouped_query,
    torch::Tensor basis,
    torch::Tensor output_codes,
    torch::Tensor output_scales);

std::vector<torch::Tensor> qksieve_project_quantize_active_forward(
    torch::Tensor grouped_query,
    torch::Tensor basis,
    torch::Tensor bit_allocations);

void qksieve_project_encode_append_forward(
    torch::Tensor key,
    torch::Tensor basis,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    int64_t start);

std::vector<torch::Tensor> qksieve_project_append_quantize_forward(
    torch::Tensor key,
    torch::Tensor grouped_query,
    torch::Tensor basis,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    int64_t start);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "qksieve_project_quantize_forward",
      &qksieve_project_quantize_forward,
      "Fused QKSieve query projection and bandwise INT8 quantization");
  m.def(
      "qksieve_project_quantize_wmma_forward",
      &qksieve_project_quantize_wmma_forward,
      "Tensor-Core QKSieve query projection and bandwise INT8 quantization");
  m.def(
      "qksieve_project_quantize_wmma_out",
      &qksieve_project_quantize_wmma_out,
      "Tensor-Core QKSieve query projection into persistent output buffers");
  m.def(
      "qksieve_project_quantize_active_forward",
      &qksieve_project_quantize_active_forward,
      "Active-band QKSieve query projection and INT8 quantization");
  m.def(
      "qksieve_project_encode_append_forward",
      &qksieve_project_encode_append_forward,
      "Fused QKSieve one-token Key projection and variable-bit append");
  m.def(
      "qksieve_project_append_quantize_forward",
      &qksieve_project_append_quantize_forward,
      "Joint QKSieve Key append and GQA Query preparation");
}
"""


CUDA_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

__global__ void qksieve_project_quantize_wmma_half_kernel(
    const half* __restrict__ grouped_query,
    const half* __restrict__ basis,
    int8_t* __restrict__ output_codes,
    half* __restrict__ output_scales,
    int group_count) {
  constexpr int kHeadDim = 128;
  constexpr int kPaddedGroups = 16;
  constexpr int kTile = 16;
  constexpr int kWarpCount = kHeadDim / kTile;
  int kv_row = blockIdx.x;
  int tid = threadIdx.x;
  int warp = tid >> 5;
  __shared__ half input[kPaddedGroups][kHeadDim];
  __shared__ float projected[kPaddedGroups][kHeadDim];
  __shared__ float band_scales[kPaddedGroups][8];

  for (int index = tid; index < kPaddedGroups * kHeadDim;
       index += blockDim.x) {
    int row = index / kHeadDim;
    int column = index - row * kHeadDim;
    input[row][column] = row < group_count
        ? grouped_query[
              (static_cast<int64_t>(kv_row) * group_count + row)
                  * kHeadDim
              + column]
        : __float2half(0.0f);
  }
  __syncthreads();

  if (warp < kWarpCount) {
    int output_start = warp * kTile;
    wmma::fragment<
        wmma::accumulator, kTile, kTile, kTile, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
#pragma unroll
    for (int input_start = 0; input_start < kHeadDim;
         input_start += kTile) {
      wmma::fragment<
          wmma::matrix_a, kTile, kTile, kTile, half,
          wmma::row_major> input_fragment;
      wmma::fragment<
          wmma::matrix_b, kTile, kTile, kTile, half,
          wmma::row_major> basis_fragment;
      wmma::load_matrix_sync(
          input_fragment,
          &input[0][input_start],
          kHeadDim);
      wmma::load_matrix_sync(
          basis_fragment,
          basis
              + static_cast<int64_t>(kv_row) * kHeadDim * kHeadDim
              + input_start * kHeadDim + output_start,
          kHeadDim);
      wmma::mma_sync(
          accumulator,
          input_fragment,
          basis_fragment,
          accumulator);
    }
    wmma::store_matrix_sync(
        &projected[0][output_start],
        accumulator,
        kHeadDim,
        wmma::mem_row_major);
  }
  __syncthreads();

  if (tid < kHeadDim) {
    for (int group = 0; group < group_count; ++group) {
      projected[group][tid] = __half2float(
          __float2half_rn(projected[group][tid]));
    }
  }
  __syncthreads();

  if (tid < kHeadDim) {
    int band = tid >> 4;
    int lane = tid & 15;
    if (lane == 0) {
      for (int group = 0; group < group_count; ++group) {
        float maximum = 0.0f;
#pragma unroll
        for (int index = 0; index < 16; ++index) {
          maximum = fmaxf(
              maximum,
              fabsf(projected[group][band * 16 + index]));
        }
        float scale = fmaxf(maximum / 127.0f, 1.0e-8f);
        band_scales[group][band] = scale;
        int query_row = kv_row * group_count + group;
        output_scales[
            static_cast<int64_t>(query_row) * 8 + band] =
            __float2half_rn(scale);
      }
    }
  }
  __syncthreads();

  if (tid < kHeadDim) {
    int band = tid >> 4;
    for (int group = 0; group < group_count; ++group) {
      int code = __float2int_rn(
          projected[group][tid] / band_scales[group][band]);
      code = max(-127, min(127, code));
      int query_row = kv_row * group_count + group;
      output_codes[
          static_cast<int64_t>(query_row) * kHeadDim + tid] =
          static_cast<int8_t>(code);
    }
  }
}

template <typename scalar_t>
__global__ void qksieve_project_quantize_kernel(
    const scalar_t* __restrict__ grouped_query,
    const scalar_t* __restrict__ basis,
    const int8_t* __restrict__ bit_allocations,
    int8_t* __restrict__ output_codes,
    scalar_t* __restrict__ output_scales,
    int group_count,
    bool active_only) {
  constexpr int kMaxGroups = 16;
  int kv_row = blockIdx.x;
  int output_dimension = threadIdx.x;
  int band = output_dimension >> 4;
  int lane = output_dimension & 15;
  bool active = !active_only
      || static_cast<int>(bit_allocations[kv_row * 8 + band]) > 0;
  const scalar_t* basis_row =
      basis + static_cast<int64_t>(kv_row) * 128 * 128;
  __shared__ float projected[kMaxGroups][128];
  __shared__ float band_scales[kMaxGroups][8];
  float accumulators[kMaxGroups];
#pragma unroll
  for (int group = 0; group < kMaxGroups; ++group) {
    accumulators[group] = 0.0f;
  }

  // Threads span consecutive output dimensions.  For every input dimension
  // the basis load is therefore contiguous across a warp, and all GQA Query
  // groups mapped to this KV head reuse the same basis value.
#pragma unroll 8
  for (int input_dimension = 0; input_dimension < 128; ++input_dimension) {
    if (active) {
      float basis_value = static_cast<float>(
          basis_row[input_dimension * 128 + output_dimension]);
      for (int group = 0; group < group_count; ++group) {
        int query_row = kv_row * group_count + group;
        accumulators[group] +=
            static_cast<float>(
                grouped_query[
                    static_cast<int64_t>(query_row) * 128 + input_dimension])
            * basis_value;
      }
    }
  }
  for (int group = 0; group < group_count; ++group) {
    // Match the frozen path's model-dtype projection materialization before
    // the separate INT8 quantization kernel reads it.
    projected[group][output_dimension] = static_cast<float>(
        static_cast<scalar_t>(accumulators[group]));
  }
  __syncthreads();

  if (lane == 0) {
    for (int group = 0; group < group_count; ++group) {
      float maximum = 0.0f;
#pragma unroll
      for (int index = 0; index < 16; ++index) {
        maximum = fmaxf(
            maximum,
            fabsf(projected[group][band * 16 + index]));
      }
      band_scales[group][band] = active
          ? fmaxf(maximum / 127.0f, 1.0e-8f)
          : 1.0f;
      int query_row = kv_row * group_count + group;
      output_scales[static_cast<int64_t>(query_row) * 8 + band] =
          static_cast<scalar_t>(band_scales[group][band]);
    }
  }
  __syncthreads();

  for (int group = 0; group < group_count; ++group) {
    int code = active
        ? __float2int_rn(
            projected[group][output_dimension]
            / band_scales[group][band])
        : 0;
    code = max(-127, min(127, code));
    int query_row = kv_row * group_count + group;
    output_codes[
        static_cast<int64_t>(query_row) * 128 + output_dimension] =
        static_cast<int8_t>(code);
  }
}

template <typename scalar_t>
__global__ void qksieve_project_encode_append_kernel(
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ basis,
    uint8_t* __restrict__ packed_codes,
    scalar_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int64_t* __restrict__ code_bases,
    const int64_t* __restrict__ scale_bases,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    int64_t key_batch_stride,
    int64_t key_head_stride,
    int kv_head_count,
    int start) {
  int batch_kv = blockIdx.x;
  int batch = batch_kv / kv_head_count;
  int kv_head = batch_kv - batch * kv_head_count;
  int output_dimension = threadIdx.x;
  int band = output_dimension >> 4;
  int lane = output_dimension & 15;
  const scalar_t* key_row =
      key + static_cast<int64_t>(batch) * key_batch_stride
      + static_cast<int64_t>(kv_head) * key_head_stride;
  const scalar_t* basis_row =
      basis + static_cast<int64_t>(batch_kv) * 128 * 128;
  __shared__ float projected[128];
  __shared__ float band_scales[8];
  __shared__ int8_t quantized[128];

  int bits = static_cast<int>(
      bit_allocations[batch_kv * 8 + band]);
  float accumulator = 0.0f;
  if (bits > 0) {
#pragma unroll 8
    for (int input_dimension = 0;
         input_dimension < 128;
         ++input_dimension) {
      accumulator += static_cast<float>(key_row[input_dimension])
          * static_cast<float>(
              basis_row[input_dimension * 128 + output_dimension]);
    }
  }
  // Preserve the model-dtype intermediate used by the existing two-kernel
  // path before calculating scales and codes.
  float value = static_cast<float>(static_cast<scalar_t>(accumulator));
  projected[output_dimension] = value;
  __syncthreads();

  if (lane == 0) {
    float statistic = 0.0f;
#pragma unroll
    for (int index = 0; index < 16; ++index) {
      float magnitude = fabsf(projected[band * 16 + index]);
      statistic = bits == 1
          ? statistic + magnitude
          : fmaxf(statistic, magnitude);
    }
    float scale = bits == 0
        ? 0.0f
        : bits == 1
        ? statistic * (1.0f / 16.0f)
        : statistic / static_cast<float>((1 << (bits - 1)) - 1);
    band_scales[band] = fmaxf(scale, 1.0e-8f);
  }
  __syncthreads();

  float scale = band_scales[band];
  int code = 0;
  if (bits == 1) {
    code = value >= 0.0f ? 1 : -1;
  } else if (bits > 1) {
    int maximum = (1 << (bits - 1)) - 1;
    code = __float2int_rn(__fdiv_rn(value, scale));
    code = max(-maximum, min(maximum, code));
  }
  quantized[output_dimension] = static_cast<int8_t>(code);
  if (bits > 0 && lane == 0) {
    int scale_index = static_cast<int>(
        scale_offsets[batch_kv * 8 + band]);
    key_scales[
        scale_bases[batch_kv]
        + static_cast<int64_t>(start) * scale_strides[batch_kv]
        + scale_index] = static_cast<scalar_t>(scale);
  }
  __syncthreads();

  if (bits == 0) {
    return;
  }
  uint8_t* destination = packed_codes
      + code_bases[batch_kv]
      + static_cast<int64_t>(start) * code_strides[batch_kv]
      + code_offsets[batch_kv * 8 + band];
  const int8_t* band_codes = quantized + band * 16;
  if (bits == 8 && lane < 16) {
    destination[lane] = static_cast<uint8_t>(band_codes[lane]);
  } else if (bits == 4 && lane < 8) {
    destination[lane] =
        (static_cast<uint8_t>(band_codes[2 * lane]) & 0xf)
        | ((static_cast<uint8_t>(band_codes[2 * lane + 1]) & 0xf) << 4);
  } else if (bits == 2 && lane < 4) {
    int base = 4 * lane;
    destination[lane] =
        (static_cast<uint8_t>(band_codes[base]) & 0x3)
        | ((static_cast<uint8_t>(band_codes[base + 1]) & 0x3) << 2)
        | ((static_cast<uint8_t>(band_codes[base + 2]) & 0x3) << 4)
        | ((static_cast<uint8_t>(band_codes[base + 3]) & 0x3) << 6);
  } else if (bits == 1 && lane < 2) {
    int base = 8 * lane;
    uint8_t packed = 0;
#pragma unroll
    for (int index = 0; index < 8; ++index) {
      packed |= static_cast<uint8_t>(
          band_codes[base + index] > 0 ? (1 << index) : 0);
    }
    destination[lane] = packed;
  }
}

template <typename scalar_t>
__global__ void qksieve_project_append_quantize_kernel(
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ grouped_query,
    const scalar_t* __restrict__ basis,
    int8_t* __restrict__ query_codes,
    scalar_t* __restrict__ query_scales,
    uint8_t* __restrict__ packed_codes,
    scalar_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int64_t* __restrict__ code_bases,
    const int64_t* __restrict__ scale_bases,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    int64_t key_batch_stride,
    int64_t key_head_stride,
    int kv_head_count,
    int group_count,
    int start) {
  constexpr int kMaxGroups = 16;
  int batch_kv = blockIdx.x;
  int output_dimension = threadIdx.x;
  int batch = batch_kv / kv_head_count;
  int kv_head = batch_kv - batch * kv_head_count;
  int band = output_dimension >> 4;
  int lane = output_dimension & 15;
  const scalar_t* key_row =
      key + static_cast<int64_t>(batch) * key_batch_stride
      + static_cast<int64_t>(kv_head) * key_head_stride;
  const scalar_t* basis_row =
      basis + static_cast<int64_t>(batch_kv) * 128 * 128;
  __shared__ float projected[kMaxGroups + 1][128];
  __shared__ float band_scales[kMaxGroups + 1][8];
  __shared__ int8_t quantized_key[128];
  float key_accumulator = 0.0f;
  float query_accumulators[kMaxGroups];
#pragma unroll
  for (int group = 0; group < kMaxGroups; ++group) {
    query_accumulators[group] = 0.0f;
  }

#pragma unroll 8
  for (int input_dimension = 0;
       input_dimension < 128;
       ++input_dimension) {
    float basis_value = static_cast<float>(
        basis_row[input_dimension * 128 + output_dimension]);
    key_accumulator += static_cast<float>(key_row[input_dimension])
        * basis_value;
    for (int group = 0; group < group_count; ++group) {
      int query_row = batch_kv * group_count + group;
      query_accumulators[group] +=
          static_cast<float>(
              grouped_query[
                  static_cast<int64_t>(query_row) * 128
                  + input_dimension])
          * basis_value;
    }
  }
  projected[0][output_dimension] = static_cast<float>(
      static_cast<scalar_t>(key_accumulator));
  for (int group = 0; group < group_count; ++group) {
    projected[group + 1][output_dimension] = static_cast<float>(
        static_cast<scalar_t>(query_accumulators[group]));
  }
  __syncthreads();

  int key_bits = static_cast<int>(
      bit_allocations[batch_kv * 8 + band]);
  if (lane == 0) {
    float key_statistic = 0.0f;
#pragma unroll
    for (int index = 0; index < 16; ++index) {
      float magnitude = fabsf(projected[0][band * 16 + index]);
      key_statistic = key_bits == 1
          ? key_statistic + magnitude
          : fmaxf(key_statistic, magnitude);
    }
    float key_scale = key_bits == 0
        ? 0.0f
        : key_bits == 1
        ? key_statistic * (1.0f / 16.0f)
        : key_statistic
            / static_cast<float>((1 << (key_bits - 1)) - 1);
    band_scales[0][band] = fmaxf(key_scale, 1.0e-8f);
    for (int group = 0; group < group_count; ++group) {
      float maximum = 0.0f;
#pragma unroll
      for (int index = 0; index < 16; ++index) {
        maximum = fmaxf(
            maximum,
            fabsf(projected[group + 1][band * 16 + index]));
      }
      float scale = fmaxf(maximum / 127.0f, 1.0e-8f);
      band_scales[group + 1][band] = scale;
      int query_row = batch_kv * group_count + group;
      query_scales[static_cast<int64_t>(query_row) * 8 + band] =
          static_cast<scalar_t>(scale);
    }
  }
  __syncthreads();

  float key_scale = band_scales[0][band];
  int key_code = 0;
  if (key_bits == 1) {
    key_code = projected[0][output_dimension] >= 0.0f ? 1 : -1;
  } else if (key_bits > 1) {
    int maximum = (1 << (key_bits - 1)) - 1;
    key_code = __float2int_rn(
        __fdiv_rn(projected[0][output_dimension], key_scale));
    key_code = max(-maximum, min(maximum, key_code));
  }
  quantized_key[output_dimension] = static_cast<int8_t>(key_code);
  if (key_bits > 0 && lane == 0) {
    int scale_index = static_cast<int>(
        scale_offsets[batch_kv * 8 + band]);
    key_scales[
        scale_bases[batch_kv]
        + static_cast<int64_t>(start) * scale_strides[batch_kv]
        + scale_index] = static_cast<scalar_t>(key_scale);
  }
  for (int group = 0; group < group_count; ++group) {
    int code = __float2int_rn(
        projected[group + 1][output_dimension]
        / band_scales[group + 1][band]);
    code = max(-127, min(127, code));
    int query_row = batch_kv * group_count + group;
    query_codes[
        static_cast<int64_t>(query_row) * 128 + output_dimension] =
        static_cast<int8_t>(code);
  }
  __syncthreads();

  if (key_bits == 0) {
    return;
  }
  uint8_t* destination = packed_codes
      + code_bases[batch_kv]
      + static_cast<int64_t>(start) * code_strides[batch_kv]
      + code_offsets[batch_kv * 8 + band];
  const int8_t* band_codes = quantized_key + band * 16;
  if (key_bits == 8 && lane < 16) {
    destination[lane] = static_cast<uint8_t>(band_codes[lane]);
  } else if (key_bits == 4 && lane < 8) {
    destination[lane] =
        (static_cast<uint8_t>(band_codes[2 * lane]) & 0xf)
        | ((static_cast<uint8_t>(band_codes[2 * lane + 1]) & 0xf) << 4);
  } else if (key_bits == 2 && lane < 4) {
    int base = 4 * lane;
    destination[lane] =
        (static_cast<uint8_t>(band_codes[base]) & 0x3)
        | ((static_cast<uint8_t>(band_codes[base + 1]) & 0x3) << 2)
        | ((static_cast<uint8_t>(band_codes[base + 2]) & 0x3) << 4)
        | ((static_cast<uint8_t>(band_codes[base + 3]) & 0x3) << 6);
  } else if (key_bits == 1 && lane < 2) {
    int base = 8 * lane;
    uint8_t packed = 0;
#pragma unroll
    for (int index = 0; index < 8; ++index) {
      packed |= static_cast<uint8_t>(
          band_codes[base + index] > 0 ? (1 << index) : 0);
    }
    destination[lane] = packed;
  }
}

std::vector<torch::Tensor> qksieve_project_quantize_forward(
    torch::Tensor grouped_query,
    torch::Tensor basis) {
  TORCH_CHECK(
      grouped_query.is_cuda() && basis.is_cuda(),
      "grouped query and basis must be CUDA tensors");
  TORCH_CHECK(
      grouped_query.device() == basis.device(),
      "grouped query and basis must be on the same CUDA device");
  TORCH_CHECK(
      grouped_query.dim() == 4
          && grouped_query.size(3) == 128,
      "grouped query must have shape [batch, kv_heads, groups, 128]");
  TORCH_CHECK(
      basis.dim() == 4
          && basis.size(0) == grouped_query.size(0)
          && basis.size(1) == grouped_query.size(1)
          && basis.size(2) == 128
          && basis.size(3) == 128,
      "basis must have shape [batch, kv_heads, 128, 128]");
  TORCH_CHECK(
      grouped_query.scalar_type() == basis.scalar_type(),
      "grouped query and basis dtypes must match");
  TORCH_CHECK(
      grouped_query.scalar_type() == at::kHalf
          || grouped_query.scalar_type() == at::kBFloat16,
      "fused QKSieve query preparation supports FP16 and BF16");
  TORCH_CHECK(
      grouped_query.size(2) > 0 && grouped_query.size(2) <= 16,
      "QKSieve fused Query preparation supports 1-16 GQA groups");

  auto query_c = grouped_query.contiguous();
  auto basis_c = basis.contiguous();
  auto output_codes = torch::empty(
      query_c.sizes(),
      query_c.options().dtype(at::kChar));
  auto scale_shape = query_c.sizes().vec();
  scale_shape.back() = 8;
  auto output_scales = torch::empty(scale_shape, query_c.options());
  int kv_row_count = static_cast<int>(
      query_c.size(0) * query_c.size(1));
  c10::cuda::CUDAGuard device_guard(query_c.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "qksieve_project_quantize_forward",
      [&] {
        qksieve_project_quantize_kernel<scalar_t>
            <<<kv_row_count, 128, 0, stream>>>(
                query_c.data_ptr<scalar_t>(),
                basis_c.data_ptr<scalar_t>(),
                nullptr,
                output_codes.data_ptr<int8_t>(),
                output_scales.data_ptr<scalar_t>(),
                static_cast<int>(query_c.size(2)),
                false);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output_codes, output_scales};
}

std::vector<torch::Tensor> qksieve_project_quantize_wmma_forward(
    torch::Tensor grouped_query,
    torch::Tensor basis) {
  TORCH_CHECK(
      grouped_query.is_cuda() && basis.is_cuda(),
      "grouped query and basis must be CUDA tensors");
  TORCH_CHECK(
      grouped_query.device() == basis.device(),
      "grouped query and basis must be on the same CUDA device");
  TORCH_CHECK(
      grouped_query.dim() == 4
          && grouped_query.size(3) == 128,
      "grouped query must have shape [batch, kv_heads, groups, 128]");
  TORCH_CHECK(
      basis.dim() == 4
          && basis.size(0) == grouped_query.size(0)
          && basis.size(1) == grouped_query.size(1)
          && basis.size(2) == 128
          && basis.size(3) == 128,
      "basis must have shape [batch, kv_heads, 128, 128]");
  TORCH_CHECK(
      grouped_query.scalar_type() == at::kHalf
          && basis.scalar_type() == at::kHalf,
      "WMMA Query preparation currently requires FP16");
  TORCH_CHECK(
      grouped_query.size(2) > 0 && grouped_query.size(2) <= 16,
      "WMMA Query preparation supports 1-16 GQA groups");

  auto query_c = grouped_query.contiguous();
  auto basis_c = basis.contiguous();
  auto output_codes = torch::empty(
      query_c.sizes(), query_c.options().dtype(at::kChar));
  auto scale_shape = query_c.sizes().vec();
  scale_shape.back() = 8;
  auto output_scales = torch::empty(scale_shape, query_c.options());
  int kv_row_count = static_cast<int>(
      query_c.size(0) * query_c.size(1));
  c10::cuda::CUDAGuard device_guard(query_c.device());
  qksieve_project_quantize_wmma_half_kernel<<<
      kv_row_count, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const half*>(
              query_c.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(
              basis_c.data_ptr<at::Half>()),
          output_codes.data_ptr<int8_t>(),
          reinterpret_cast<half*>(
              output_scales.data_ptr<at::Half>()),
          static_cast<int>(query_c.size(2)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output_codes, output_scales};
}

void qksieve_project_quantize_wmma_out(
    torch::Tensor grouped_query,
    torch::Tensor basis,
    torch::Tensor output_codes,
    torch::Tensor output_scales) {
  TORCH_CHECK(
      grouped_query.is_cuda() && basis.is_cuda()
          && output_codes.is_cuda() && output_scales.is_cuda(),
      "all WMMA Query preparation tensors must be CUDA tensors");
  TORCH_CHECK(
      grouped_query.device() == basis.device()
          && grouped_query.device() == output_codes.device()
          && grouped_query.device() == output_scales.device(),
      "all WMMA Query preparation tensors must be on the same device");
  TORCH_CHECK(
      grouped_query.dim() == 4 && grouped_query.size(3) == 128,
      "grouped query must have shape [batch, kv_heads, groups, 128]");
  TORCH_CHECK(
      basis.dim() == 4
          && basis.size(0) == grouped_query.size(0)
          && basis.size(1) == grouped_query.size(1)
          && basis.size(2) == 128
          && basis.size(3) == 128,
      "basis must have shape [batch, kv_heads, 128, 128]");
  TORCH_CHECK(
      grouped_query.scalar_type() == at::kHalf
          && basis.scalar_type() == at::kHalf,
      "WMMA Query preparation currently requires FP16");
  TORCH_CHECK(
      grouped_query.size(2) > 0 && grouped_query.size(2) <= 16,
      "WMMA Query preparation supports 1-16 GQA groups");
  TORCH_CHECK(
      grouped_query.is_contiguous() && basis.is_contiguous(),
      "persistent WMMA Query inputs must be contiguous");
  TORCH_CHECK(
      output_codes.scalar_type() == at::kChar
          && output_codes.sizes() == grouped_query.sizes()
          && output_codes.is_contiguous(),
      "output_codes must be contiguous INT8 with the grouped-query shape");
  TORCH_CHECK(
      output_scales.scalar_type() == at::kHalf
          && output_scales.dim() == 4
          && output_scales.size(0) == grouped_query.size(0)
          && output_scales.size(1) == grouped_query.size(1)
          && output_scales.size(2) == grouped_query.size(2)
          && output_scales.size(3) == 8
          && output_scales.is_contiguous(),
      "output_scales must be contiguous FP16 [batch, kv_heads, groups, 8]");

  c10::cuda::CUDAGuard device_guard(grouped_query.device());
  int kv_row_count = static_cast<int>(
      grouped_query.size(0) * grouped_query.size(1));
  qksieve_project_quantize_wmma_half_kernel<<<
      kv_row_count, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const half*>(
              grouped_query.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(basis.data_ptr<at::Half>()),
          output_codes.data_ptr<int8_t>(),
          reinterpret_cast<half*>(output_scales.data_ptr<at::Half>()),
          static_cast<int>(grouped_query.size(2)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> qksieve_project_quantize_active_forward(
    torch::Tensor grouped_query,
    torch::Tensor basis,
    torch::Tensor bit_allocations) {
  TORCH_CHECK(
      grouped_query.is_cuda() && basis.is_cuda()
          && bit_allocations.is_cuda(),
      "active-band Query inputs must be CUDA tensors");
  TORCH_CHECK(
      grouped_query.dim() == 4 && grouped_query.size(3) == 128,
      "grouped query must have shape [batch, kv_heads, groups, 128]");
  TORCH_CHECK(
      basis.dim() == 4
          && basis.size(0) == grouped_query.size(0)
          && basis.size(1) == grouped_query.size(1)
          && basis.size(2) == 128 && basis.size(3) == 128,
      "basis must have shape [batch, kv_heads, 128, 128]");
  TORCH_CHECK(
      bit_allocations.dim() == 3
          && bit_allocations.size(0) == grouped_query.size(0)
          && bit_allocations.size(1) == grouped_query.size(1)
          && bit_allocations.size(2) == 8
          && bit_allocations.scalar_type() == at::kChar,
      "bit allocations must have shape [batch, kv_heads, 8]");
  TORCH_CHECK(
      grouped_query.scalar_type() == basis.scalar_type(),
      "grouped query and basis dtypes must match");
  TORCH_CHECK(
      grouped_query.scalar_type() == at::kHalf
          || grouped_query.scalar_type() == at::kBFloat16,
      "active-band Query preparation supports FP16 and BF16");
  TORCH_CHECK(
      grouped_query.size(2) > 0 && grouped_query.size(2) <= 16,
      "active-band Query preparation supports 1-16 GQA groups");

  auto query_c = grouped_query.contiguous();
  auto basis_c = basis.contiguous();
  auto allocations_c = bit_allocations.contiguous();
  auto output_codes = torch::empty(
      query_c.sizes(), query_c.options().dtype(at::kChar));
  auto scale_shape = query_c.sizes().vec();
  scale_shape.back() = 8;
  auto output_scales = torch::empty(scale_shape, query_c.options());
  int kv_row_count = static_cast<int>(
      query_c.size(0) * query_c.size(1));
  c10::cuda::CUDAGuard device_guard(query_c.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "qksieve_project_quantize_active_forward",
      [&] {
        qksieve_project_quantize_kernel<scalar_t>
            <<<kv_row_count, 128, 0, stream>>>(
                query_c.data_ptr<scalar_t>(),
                basis_c.data_ptr<scalar_t>(),
                allocations_c.data_ptr<int8_t>(),
                output_codes.data_ptr<int8_t>(),
                output_scales.data_ptr<scalar_t>(),
                static_cast<int>(query_c.size(2)),
                true);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output_codes, output_scales};
}

void qksieve_project_encode_append_forward(
    torch::Tensor key,
    torch::Tensor basis,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    int64_t start) {
  TORCH_CHECK(
      key.is_cuda() && basis.is_cuda() && packed_codes.is_cuda()
          && key_scales.is_cuda() && bit_allocations.is_cuda()
          && code_offsets.is_cuda() && scale_offsets.is_cuda()
          && code_bases.is_cuda() && scale_bases.is_cuda()
          && code_strides.is_cuda() && scale_strides.is_cuda(),
      "all fused Key append tensors must be CUDA tensors");
  TORCH_CHECK(
      key.dim() == 4 && key.size(2) == 1 && key.size(3) == 128,
      "key must have shape [batch, kv_heads, 1, 128]");
  TORCH_CHECK(
      basis.dim() == 4 && basis.size(0) == key.size(0)
          && basis.size(1) == key.size(1)
          && basis.size(2) == 128 && basis.size(3) == 128,
      "basis must have shape [batch, kv_heads, 128, 128]");
  TORCH_CHECK(
      key.scalar_type() == basis.scalar_type()
          && key.scalar_type() == key_scales.scalar_type(),
      "key, basis, and Key scales must share a dtype");
  TORCH_CHECK(
      key.scalar_type() == at::kHalf
          || key.scalar_type() == at::kBFloat16,
      "fused Key append supports FP16 and BF16");
  TORCH_CHECK(packed_codes.scalar_type() == at::kByte, "codes must be uint8");
  TORCH_CHECK(
      bit_allocations.scalar_type() == at::kChar,
      "bit allocations must be int8");
  TORCH_CHECK(code_offsets.scalar_type() == at::kShort, "offsets must be int16");
  TORCH_CHECK(scale_offsets.scalar_type() == at::kChar, "scale offsets must be int8");
  TORCH_CHECK(code_bases.scalar_type() == at::kLong, "code bases must be int64");
  TORCH_CHECK(scale_bases.scalar_type() == at::kLong, "scale bases must be int64");
  TORCH_CHECK(code_strides.scalar_type() == at::kShort, "code strides must be int16");
  TORCH_CHECK(scale_strides.scalar_type() == at::kChar, "scale strides must be int8");
  TORCH_CHECK(start >= 0, "append offset must be non-negative");

  auto basis_c = basis.contiguous();
  int kv_row_count = static_cast<int>(key.size(0) * key.size(1));
  c10::cuda::CUDAGuard device_guard(key.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      key.scalar_type(),
      "qksieve_project_encode_append_forward",
      [&] {
        qksieve_project_encode_append_kernel<scalar_t>
            <<<kv_row_count, 128, 0, stream>>>(
                key.data_ptr<scalar_t>(),
                basis_c.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                bit_allocations.data_ptr<int8_t>(),
                code_offsets.data_ptr<int16_t>(),
                scale_offsets.data_ptr<int8_t>(),
                code_bases.data_ptr<int64_t>(),
                scale_bases.data_ptr<int64_t>(),
                code_strides.data_ptr<int16_t>(),
                scale_strides.data_ptr<int8_t>(),
                key.stride(0),
                key.stride(1),
                static_cast<int>(key.size(1)),
                static_cast<int>(start));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> qksieve_project_append_quantize_forward(
    torch::Tensor key,
    torch::Tensor grouped_query,
    torch::Tensor basis,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    int64_t start) {
  TORCH_CHECK(
      key.is_cuda() && grouped_query.is_cuda() && basis.is_cuda()
          && packed_codes.is_cuda() && key_scales.is_cuda()
          && bit_allocations.is_cuda() && code_offsets.is_cuda()
          && scale_offsets.is_cuda() && code_bases.is_cuda()
          && scale_bases.is_cuda() && code_strides.is_cuda()
          && scale_strides.is_cuda(),
      "all joint preparation tensors must be CUDA tensors");
  TORCH_CHECK(
      key.dim() == 4 && key.size(2) == 1 && key.size(3) == 128,
      "key must have shape [batch, kv_heads, 1, 128]");
  TORCH_CHECK(
      grouped_query.dim() == 4
          && grouped_query.size(0) == key.size(0)
          && grouped_query.size(1) == key.size(1)
          && grouped_query.size(2) > 0
          && grouped_query.size(2) <= 16
          && grouped_query.size(3) == 128,
      "grouped query must have shape [batch, kv_heads, 1-16, 128]");
  TORCH_CHECK(
      basis.dim() == 4 && basis.size(0) == key.size(0)
          && basis.size(1) == key.size(1)
          && basis.size(2) == 128 && basis.size(3) == 128,
      "basis must have shape [batch, kv_heads, 128, 128]");
  TORCH_CHECK(
      key.scalar_type() == grouped_query.scalar_type()
          && key.scalar_type() == basis.scalar_type()
          && key.scalar_type() == key_scales.scalar_type(),
      "key, query, basis, and Key scales must share a dtype");
  TORCH_CHECK(
      key.scalar_type() == at::kHalf
          || key.scalar_type() == at::kBFloat16,
      "joint preparation supports FP16 and BF16");
  TORCH_CHECK(packed_codes.scalar_type() == at::kByte, "codes must be uint8");
  TORCH_CHECK(bit_allocations.scalar_type() == at::kChar, "allocations must be int8");
  TORCH_CHECK(code_offsets.scalar_type() == at::kShort, "offsets must be int16");
  TORCH_CHECK(scale_offsets.scalar_type() == at::kChar, "scale offsets must be int8");
  TORCH_CHECK(code_bases.scalar_type() == at::kLong, "code bases must be int64");
  TORCH_CHECK(scale_bases.scalar_type() == at::kLong, "scale bases must be int64");
  TORCH_CHECK(code_strides.scalar_type() == at::kShort, "code strides must be int16");
  TORCH_CHECK(scale_strides.scalar_type() == at::kChar, "scale strides must be int8");
  TORCH_CHECK(start >= 0, "append offset must be non-negative");

  auto query_c = grouped_query.contiguous();
  auto basis_c = basis.contiguous();
  auto output_codes = torch::empty(
      query_c.sizes(), query_c.options().dtype(at::kChar));
  auto scale_shape = query_c.sizes().vec();
  scale_shape.back() = 8;
  auto output_scales = torch::empty(scale_shape, query_c.options());
  int kv_row_count = static_cast<int>(key.size(0) * key.size(1));
  c10::cuda::CUDAGuard device_guard(key.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      key.scalar_type(),
      "qksieve_project_append_quantize_forward",
      [&] {
        qksieve_project_append_quantize_kernel<scalar_t>
            <<<kv_row_count, 128, 0, stream>>>(
                key.data_ptr<scalar_t>(),
                query_c.data_ptr<scalar_t>(),
                basis_c.data_ptr<scalar_t>(),
                output_codes.data_ptr<int8_t>(),
                output_scales.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                bit_allocations.data_ptr<int8_t>(),
                code_offsets.data_ptr<int16_t>(),
                scale_offsets.data_ptr<int8_t>(),
                code_bases.data_ptr<int64_t>(),
                scale_bases.data_ptr<int64_t>(),
                code_strides.data_ptr<int16_t>(),
                scale_strides.data_ptr<int8_t>(),
                key.stride(0),
                key.stride(1),
                static_cast<int>(key.size(1)),
                static_cast<int>(query_c.size(2)),
                static_cast<int>(start));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output_codes, output_scales};
}
"""


@lru_cache(maxsize=1)
def load_extension():
    return load_inline(
        name="qksieve_query_project_ext_v8_persistent_wmma",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )


def project_quantize(
    grouped_query: torch.Tensor,
    basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_inputs(grouped_query, basis, require_cuda=True)
    return tuple(
        load_extension().qksieve_project_quantize_forward(
            grouped_query.contiguous(),
            basis.contiguous(),
        )
    )


def project_quantize_wmma(
    grouped_query: torch.Tensor,
    basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_inputs(grouped_query, basis, require_cuda=True)
    if grouped_query.dtype != torch.float16:
        raise ValueError("WMMA query projection currently requires float16")
    return tuple(
        load_extension().qksieve_project_quantize_wmma_forward(
            grouped_query.contiguous(),
            basis.contiguous(),
        )
    )


def project_quantize_wmma_unchecked(
    grouped_query: torch.Tensor,
    basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the frozen decode kernel after one-time caller validation."""
    return tuple(
        load_extension().qksieve_project_quantize_wmma_forward(
            grouped_query,
            basis,
        )
    )


def project_quantize_wmma_out_unchecked(
    grouped_query: torch.Tensor,
    basis: torch.Tensor,
    output_codes: torch.Tensor,
    output_scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch frozen WMMA projection into caller-owned persistent buffers."""
    load_extension().qksieve_project_quantize_wmma_out(
        grouped_query,
        basis,
        output_codes,
        output_scales,
    )
    return output_codes, output_scales


def project_quantize_active(
    grouped_query: torch.Tensor,
    basis: torch.Tensor,
    bit_allocations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_inputs(grouped_query, basis, require_cuda=True)
    expected_shape = grouped_query.shape[:2] + (BAND_COUNT,)
    if tuple(bit_allocations.shape) != expected_shape:
        raise ValueError(
            "bit_allocations must have shape "
            f"{tuple(expected_shape)}"
        )
    return tuple(
        load_extension().qksieve_project_quantize_active_forward(
            grouped_query.contiguous(),
            basis.contiguous(),
            bit_allocations.contiguous(),
        )
    )


def project_encode_append(
    key: torch.Tensor,
    basis: torch.Tensor,
    packed_index: dict,
    start: int,
) -> None:
    if key.ndim != 4 or key.shape[-2:] != (1, HEAD_DIM):
        raise ValueError("key must have shape [batch, kv_heads, 1, 128]")
    if tuple(basis.shape) != key.shape[:2] + (HEAD_DIM, HEAD_DIM):
        raise ValueError("basis must have shape [batch, kv_heads, 128, 128]")
    if start < 0 or start >= int(packed_index["capacity"]):
        raise ValueError("fused Key append offset exceeds index capacity")
    score_bias = packed_index.get("score_bias")
    if isinstance(score_bias, torch.Tensor) and score_bias.numel() > 0:
        raise ValueError("fused Key append does not support score bias")
    load_extension().qksieve_project_encode_append_forward(
        key,
        basis.contiguous(),
        packed_index["packed_codes"],
        packed_index["key_scales"],
        packed_index["bit_allocations"],
        packed_index["code_offsets"],
        packed_index["scale_offsets"],
        packed_index["code_bases"],
        packed_index["scale_bases"],
        packed_index["code_strides"],
        packed_index["scale_strides"],
        int(start),
    )
    packed_index["indexed_count"] = start + 1


def project_encode_append_unchecked(
    key: torch.Tensor,
    basis: torch.Tensor,
    packed_index: dict,
    start: int,
) -> None:
    """Append one Key after the frozen decode contract has been validated."""
    load_extension().qksieve_project_encode_append_forward(
        key,
        basis,
        packed_index["packed_codes"],
        packed_index["key_scales"],
        packed_index["bit_allocations"],
        packed_index["code_offsets"],
        packed_index["scale_offsets"],
        packed_index["code_bases"],
        packed_index["scale_bases"],
        packed_index["code_strides"],
        packed_index["scale_strides"],
        int(start),
    )
    packed_index["indexed_count"] = start + 1


def project_append_quantize(
    key: torch.Tensor,
    grouped_query: torch.Tensor,
    basis: torch.Tensor,
    packed_index: dict,
    start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_inputs(grouped_query, basis, require_cuda=True)
    if key.ndim != 4 or key.shape[-2:] != (1, HEAD_DIM):
        raise ValueError("key must have shape [batch, kv_heads, 1, 128]")
    if key.shape[:2] != grouped_query.shape[:2]:
        raise ValueError("key and grouped query head layouts must match")
    if key.dtype != grouped_query.dtype or key.device != grouped_query.device:
        raise ValueError("key and grouped query must share dtype and device")
    if start < 0 or start >= int(packed_index["capacity"]):
        raise ValueError("joint append offset exceeds index capacity")
    score_bias = packed_index.get("score_bias")
    if isinstance(score_bias, torch.Tensor) and score_bias.numel() > 0:
        raise ValueError("joint preparation does not support score bias")
    outputs = tuple(
        load_extension().qksieve_project_append_quantize_forward(
            key,
            grouped_query.contiguous(),
            basis.contiguous(),
            packed_index["packed_codes"],
            packed_index["key_scales"],
            packed_index["bit_allocations"],
            packed_index["code_offsets"],
            packed_index["scale_offsets"],
            packed_index["code_bases"],
            packed_index["scale_bases"],
            packed_index["code_strides"],
            packed_index["scale_strides"],
            int(start),
        )
    )
    packed_index["indexed_count"] = start + 1
    return outputs


def project_quantize_reference(
    grouped_query: torch.Tensor,
    basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation of the fused projection/quantization contract."""
    _validate_inputs(grouped_query, basis, require_cuda=False)
    projected = torch.einsum(
        "bhgd,bhdm->bhgm",
        grouped_query,
        basis,
    )
    bands = projected.float().reshape(
        *projected.shape[:-1],
        BAND_COUNT,
        BAND_DIM,
    )
    scales = bands.abs().amax(dim=-1).clamp_min(1.0e-8) / 127.0
    codes = torch.round(bands / scales.unsqueeze(-1)).clamp(-127, 127)
    return (
        codes.to(torch.int8).reshape_as(projected).contiguous(),
        scales.to(projected.dtype).contiguous(),
    )


def _validate_inputs(
    grouped_query: torch.Tensor,
    basis: torch.Tensor,
    *,
    require_cuda: bool,
) -> None:
    if grouped_query.ndim != 4 or grouped_query.shape[-1] != HEAD_DIM:
        raise ValueError(
            "grouped_query must have shape [batch, kv_heads, groups, 128]"
        )
    if not 1 <= grouped_query.shape[2] <= 16:
        raise ValueError("grouped_query must contain 1-16 GQA groups")
    expected_basis_shape = (
        grouped_query.shape[0],
        grouped_query.shape[1],
        HEAD_DIM,
        HEAD_DIM,
    )
    if tuple(basis.shape) != expected_basis_shape:
        raise ValueError(
            "basis must have shape [batch, kv_heads, 128, 128]; "
            f"expected {expected_basis_shape}, got {tuple(basis.shape)}"
        )
    if grouped_query.dtype != basis.dtype:
        raise ValueError("grouped_query and basis dtypes must match")
    if grouped_query.device != basis.device:
        raise ValueError("grouped_query and basis must share a device")
    if not grouped_query.is_floating_point():
        raise ValueError("grouped_query and basis must be floating point")
    if require_cuda and not grouped_query.is_cuda:
        raise ValueError("fused QKSieve query preparation requires CUDA")
    if require_cuda and grouped_query.dtype not in {
        torch.float16,
        torch.bfloat16,
    }:
        raise ValueError(
            "fused QKSieve query preparation supports FP16 and BF16"
        )
