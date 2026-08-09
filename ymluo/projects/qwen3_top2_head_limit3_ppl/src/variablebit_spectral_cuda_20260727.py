from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from torch.utils.cpp_extension import load_inline


CPP_SOURCE = r"""
#include <torch/extension.h>
#include <vector>

torch::Tensor varbit_spectral_scores_forward(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor score_bias,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    int64_t history_count);

std::vector<torch::Tensor> varbit_spectral_quantize_query_forward(
    torch::Tensor projected_query);

void varbit_spectral_encode_projected_out_forward(
    torch::Tensor projected_keys,
    torch::Tensor scale_metrics,
    torch::Tensor precomputed_scales,
    torch::Tensor exact_query_mean,
    torch::Tensor proxy_query_mean,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor score_bias,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    int64_t start,
    bool metric_scale,
    bool override_scale,
    double bias_shrinkage);

std::vector<torch::Tensor> varbit_spectral_sampled_compact_forward(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor score_bias,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    int64_t history_count,
    int64_t sample_count,
    double selected_fraction,
    int64_t candidate_capacity);

void varbit_spectral_sampled_compact_out_forward(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor score_bias,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_scores,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor overflow,
    int64_t history_count,
    int64_t sample_count,
    double selected_fraction);

torch::Tensor sharedtail_spectral_scores_forward(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    int64_t history_count);

std::vector<torch::Tensor> sharedtail_spectral_quantize_query_forward(
    torch::Tensor projected_query,
    torch::Tensor coordinate_amplitude);

void sharedtail_spectral_encode_projected_out_forward(
    torch::Tensor projected_keys,
    torch::Tensor coordinate_rms,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    int64_t start);

void sharedtail_spectral_sampled_compact_out_forward(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_scores,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor overflow,
    int64_t history_count,
    int64_t sample_count,
    double selected_fraction);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "varbit_spectral_scores_forward",
      &varbit_spectral_scores_forward,
      "Packed variable-bit spectral scores");
  m.def(
      "varbit_spectral_quantize_query_forward",
      &varbit_spectral_quantize_query_forward,
      "Fused variable-bit INT8 query preparation");
  m.def(
      "varbit_spectral_encode_projected_out_forward",
      &varbit_spectral_encode_projected_out_forward,
      "Encode projected keys into a packed variable-bit index");
  m.def(
      "varbit_spectral_sampled_compact_forward",
      &varbit_spectral_sampled_compact_forward,
      "Fused variable-bit sampled threshold and candidate compaction");
  m.def(
      "varbit_spectral_sampled_compact_out_forward",
      &varbit_spectral_sampled_compact_out_forward,
      "Fused variable-bit compaction into persistent output buffers");
  m.def(
      "sharedtail_spectral_scores_forward",
      &sharedtail_spectral_scores_forward,
      "Fixed 4/4/shared-sign spectral scores");
  m.def(
      "sharedtail_spectral_quantize_query_forward",
      &sharedtail_spectral_quantize_query_forward,
      "Fused fixed shared-tail query preparation");
  m.def(
      "sharedtail_spectral_encode_projected_out_forward",
      &sharedtail_spectral_encode_projected_out_forward,
      "Encode fixed 4/4/shared-sign spectral keys");
  m.def(
      "sharedtail_spectral_sampled_compact_out_forward",
      &sharedtail_spectral_sampled_compact_out_forward,
      "Fused fixed shared-tail threshold and compaction");
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
#include <vector>

#define VARBIT_MAX_SAMPLE_COUNT 2048

template <typename scale_t>
__global__ void varbit_encode_projected_kernel(
    const scale_t* __restrict__ projected_keys,
    const float* __restrict__ scale_metrics,
    const scale_t* __restrict__ precomputed_scales,
    const float* __restrict__ exact_query_mean,
    const float* __restrict__ proxy_query_mean,
    uint8_t* __restrict__ packed_codes,
    scale_t* __restrict__ key_scales,
    scale_t* __restrict__ score_bias,
    const int8_t* __restrict__ bit_allocations,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int64_t* __restrict__ code_bases,
    const int64_t* __restrict__ scale_bases,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    int token_count,
    int start,
    bool metric_scale,
    bool override_scale,
    int bias_capacity,
    float bias_shrinkage) {
  int row = blockIdx.x;
  int batch_kv = row / token_count;
  int local_token = row - batch_kv * token_count;
  int token = start + local_token;
  int coordinate = threadIdx.x;
  int band = coordinate >> 4;
  int lane = coordinate & 15;
  __shared__ float magnitudes[128];
  __shared__ float band_scales[8];
  __shared__ float metric_numerators[128];
  __shared__ float metric_denominators[128];
  __shared__ float bias_terms[128];
  __shared__ int8_t quantized[128];

  const scale_t* source =
      projected_keys + static_cast<int64_t>(row) * 128;
  float value = static_cast<float>(source[coordinate]);
  magnitudes[coordinate] = fabsf(value);
  __syncthreads();

  int bits = static_cast<int>(bit_allocations[batch_kv * 8 + band]);
  if (lane == 0) {
    float statistic = 0.0f;
#pragma unroll
    for (int index = 0; index < 16; ++index) {
      float magnitude = magnitudes[band * 16 + index];
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
  quantized[coordinate] = static_cast<int8_t>(code);

  __syncthreads();
  if (metric_scale && bits > 0) {
    const float* metric = scale_metrics
        + static_cast<int64_t>(batch_kv) * 8 * 16 * 16
        + band * 16 * 16;
    float metric_value = 0.0f;
    float metric_code = 0.0f;
#pragma unroll
    for (int other = 0; other < 16; ++other) {
      float weight = metric[lane * 16 + other];
      metric_value += weight * static_cast<float>(
          source[band * 16 + other]);
      metric_code += weight * static_cast<float>(
          quantized[band * 16 + other]);
    }
    float lane_code = static_cast<float>(code);
    metric_numerators[coordinate] = lane_code * metric_value;
    metric_denominators[coordinate] = lane_code * metric_code;
  } else {
    metric_numerators[coordinate] = 0.0f;
    metric_denominators[coordinate] = 0.0f;
  }
  __syncthreads();
  if (metric_scale && bits > 0 && lane == 0) {
    float numerator = 0.0f;
    float denominator = 0.0f;
#pragma unroll
    for (int index = 0; index < 16; ++index) {
      numerator += metric_numerators[band * 16 + index];
      denominator += metric_denominators[band * 16 + index];
    }
    band_scales[band] = denominator > 1.0e-12f
        ? fmaxf(numerator / denominator, 0.0f)
        : band_scales[band];
  }
  __syncthreads();
  if (override_scale && bits > 0 && lane == 0) {
    band_scales[band] = static_cast<float>(
        precomputed_scales[
            (static_cast<int64_t>(batch_kv) * token_count + local_token)
                * 8
            + band]);
  }
  __syncthreads();

  float final_scale = band_scales[band];
  if (bias_capacity > 0) {
    int64_t mean_offset = static_cast<int64_t>(batch_kv) * 128
        + coordinate;
    bias_terms[coordinate] =
        exact_query_mean[mean_offset] * value
        - proxy_query_mean[mean_offset]
            * static_cast<float>(code) * final_scale;
  }
  __syncthreads();
  if (bias_capacity > 0 && coordinate == 0) {
    float residual_mean = 0.0f;
#pragma unroll
    for (int index = 0; index < 128; ++index) {
      residual_mean += bias_terms[index];
    }
    score_bias[
        static_cast<int64_t>(batch_kv) * bias_capacity + token] =
        static_cast<scale_t>(bias_shrinkage * residual_mean);
  }
  if (bits > 0 && lane == 0) {
    int scale_index = static_cast<int>(
        scale_offsets[batch_kv * 8 + band]);
    key_scales[
        scale_bases[batch_kv]
        + static_cast<int64_t>(token) * scale_strides[batch_kv]
        + scale_index] = static_cast<scale_t>(final_scale);
  }
  __syncthreads();

  if (bits == 0) {
    return;
  }
  uint8_t* destination = packed_codes
      + code_bases[batch_kv]
      + static_cast<int64_t>(token) * code_strides[batch_kv]
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

template <typename scale_t>
__global__ void sharedtail_encode_projected_kernel(
    const scale_t* __restrict__ projected_keys,
    const float* __restrict__ coordinate_rms,
    uint8_t* __restrict__ packed_codes,
    scale_t* __restrict__ key_scales,
    int token_count,
    int capacity,
    int start) {
  int row = blockIdx.x;
  int batch_kv = row / token_count;
  int local_token = row - batch_kv * token_count;
  int token = start + local_token;
  int coordinate = threadIdx.x;
  __shared__ float values[128];
  __shared__ float normalized_square[128];
  __shared__ float scales[3];
  __shared__ int8_t core_codes[32];

  float value = static_cast<float>(
      projected_keys[static_cast<int64_t>(row) * 128 + coordinate]);
  float rms = fmaxf(
      static_cast<float>(
          coordinate_rms[static_cast<int64_t>(batch_kv) * 128 + coordinate]),
      1.0e-8f);
  values[coordinate] = value;
  float normalized = value / rms;
  normalized_square[coordinate] = normalized * normalized;
  __syncthreads();

  if (coordinate == 0 || coordinate == 16) {
    int start_coordinate = coordinate;
    float maximum = 0.0f;
#pragma unroll
    for (int index = 0; index < 16; ++index) {
      maximum = fmaxf(maximum, fabsf(values[start_coordinate + index]));
    }
    scales[start_coordinate >> 4] = fmaxf(maximum / 7.0f, 1.0e-8f);
  } else if (coordinate == 32) {
    float mean_square = 0.0f;
#pragma unroll
    for (int index = 0; index < 128; ++index) {
      mean_square += normalized_square[index];
    }
    scales[2] = fmaxf(sqrtf(mean_square * (1.0f / 128.0f)), 1.0e-8f);
  }
  __syncthreads();

  if (coordinate < 32) {
    float scale = scales[coordinate >> 4];
    int code = __float2int_rn(value / scale);
    core_codes[coordinate] = static_cast<int8_t>(max(-7, min(7, code)));
  }
  __syncthreads();

  uint8_t* destination = packed_codes
      + (static_cast<int64_t>(batch_kv) * capacity + token) * 24;
  scale_t* destination_scales = key_scales
      + (static_cast<int64_t>(batch_kv) * capacity + token) * 3;
  if (coordinate < 16 && (coordinate & 1) == 0) {
    int output = coordinate >> 1;
    destination[output] =
        (static_cast<uint8_t>(core_codes[coordinate]) & 0xf)
        | ((static_cast<uint8_t>(core_codes[coordinate + 1]) & 0xf) << 4);
  } else if (coordinate >= 16 && coordinate < 32
             && (coordinate & 1) == 0) {
    int output = 8 + ((coordinate - 16) >> 1);
    destination[output] =
        (static_cast<uint8_t>(core_codes[coordinate]) & 0xf)
        | ((static_cast<uint8_t>(core_codes[coordinate + 1]) & 0xf) << 4);
  } else if (coordinate >= 32 && coordinate < 40) {
    int tail_group = coordinate - 32;
    uint8_t signs = 0;
#pragma unroll
    for (int index = 0; index < 8; ++index) {
      signs |= values[32 + tail_group * 8 + index] >= 0.0f
          ? static_cast<uint8_t>(1 << index)
          : static_cast<uint8_t>(0);
    }
    destination[16 + tail_group] = signs;
  }
  if (coordinate < 3) {
    destination_scales[coordinate] = static_cast<scale_t>(scales[coordinate]);
  }
}

template <typename scale_t>
__global__ void sharedtail_quantize_query_kernel(
    const scale_t* __restrict__ projected_query,
    const float* __restrict__ coordinate_amplitude,
    int8_t* __restrict__ output_codes,
    scale_t* __restrict__ output_scales,
    int query_groups) {
  int row = blockIdx.x;
  int coordinate = threadIdx.x;
  int batch_kv = row / query_groups;
  int band = coordinate >> 4;
  int lane = coordinate & 15;
  __shared__ float query_values[128];
  __shared__ float original_scales[8];
  __shared__ float weighted_tail[64];
  __shared__ float weighted_scales[4];

  float value = static_cast<float>(
      projected_query[static_cast<int64_t>(row) * 128 + coordinate]);
  query_values[coordinate] = value;
  __syncthreads();
  if (lane == 0) {
    float maximum = 0.0f;
#pragma unroll
    for (int index = 0; index < 16; ++index) {
      maximum = fmaxf(
          maximum,
          fabsf(query_values[band * 16 + index]));
    }
    original_scales[band] = fmaxf(maximum / 127.0f, 1.0e-8f);
  }
  __syncthreads();

  int original_code = __float2int_rn(value / original_scales[band]);
  original_code = max(-127, min(127, original_code));
  if (coordinate < 32) {
    output_codes[static_cast<int64_t>(row) * 96 + coordinate] =
        static_cast<int8_t>(original_code);
  } else if (coordinate < 96) {
    weighted_tail[coordinate - 32] =
        static_cast<float>(original_code)
        * original_scales[band]
        * coordinate_amplitude[
            static_cast<int64_t>(batch_kv) * 128 + coordinate];
  }
  __syncthreads();

  if (coordinate >= 32 && coordinate < 96 && lane == 0) {
    int tail_band = band - 2;
    float maximum = 0.0f;
#pragma unroll
    for (int index = 0; index < 16; ++index) {
      maximum = fmaxf(
          maximum,
          fabsf(weighted_tail[tail_band * 16 + index]));
    }
    weighted_scales[tail_band] = fmaxf(maximum / 127.0f, 1.0e-8f);
  }
  __syncthreads();

  if (coordinate >= 32 && coordinate < 96) {
    int tail_coordinate = coordinate - 32;
    int tail_band = tail_coordinate >> 4;
    int weighted_code = __float2int_rn(
        weighted_tail[tail_coordinate] / weighted_scales[tail_band]);
    weighted_code = max(-127, min(127, weighted_code));
    output_codes[
        static_cast<int64_t>(row) * 96 + 32 + tail_coordinate] =
        static_cast<int8_t>(weighted_code);
  }
  if (coordinate < 2) {
    output_scales[static_cast<int64_t>(row) * 6 + coordinate] =
        static_cast<scale_t>(original_scales[coordinate]);
  } else if (coordinate >= 2 && coordinate < 6) {
    output_scales[static_cast<int64_t>(row) * 6 + coordinate] =
        static_cast<scale_t>(weighted_scales[coordinate - 2]);
  }
}

template <typename scale_t>
__global__ void varbit_quantize_query_kernel(
    const scale_t* __restrict__ projected_query,
    int8_t* __restrict__ output_codes,
    scale_t* __restrict__ output_scales) {
  int row = blockIdx.x;
  int coordinate = threadIdx.x;
  int band = coordinate >> 4;
  int lane = coordinate & 15;
  __shared__ float query_values[128];
  __shared__ float scales[8];

  float value = static_cast<float>(
      projected_query[static_cast<int64_t>(row) * 128 + coordinate]);
  query_values[coordinate] = value;
  __syncthreads();
  if (lane == 0) {
    float maximum = 0.0f;
#pragma unroll
    for (int index = 0; index < 16; ++index) {
      maximum = fmaxf(
          maximum,
          fabsf(query_values[band * 16 + index]));
    }
    scales[band] = fmaxf(maximum / 127.0f, 1.0e-8f);
  }
  __syncthreads();

  int code = __float2int_rn(value / scales[band]);
  code = max(-127, min(127, code));
  output_codes[static_cast<int64_t>(row) * 128 + coordinate] =
      static_cast<int8_t>(code);
  if (lane == 0) {
    output_scales[static_cast<int64_t>(row) * 8 + band] =
        static_cast<scale_t>(scales[band]);
  }
}

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

__device__ __forceinline__ int signed_int4(int value) {
  return value < 8 ? value : value - 16;
}

__device__ __forceinline__ int signed_int2(int value) {
  return value < 2 ? value : value - 4;
}

__device__ __forceinline__ int band_dot(
    const uint8_t* __restrict__ packed,
    const int8_t* __restrict__ query,
    int bits) {
  int dot = 0;
  if (bits == 8) {
#pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
      const uint8_t* code = packed + chunk * 4;
      int word = pack_signed_bytes(
          static_cast<int8_t>(code[0]),
          static_cast<int8_t>(code[1]),
          static_cast<int8_t>(code[2]),
          static_cast<int8_t>(code[3]));
      dot = __dp4a(
          word,
          reinterpret_cast<const int*>(query)[chunk],
          dot);
    }
  } else if (bits == 4) {
#pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
      uint8_t byte0 = packed[2 * chunk];
      uint8_t byte1 = packed[2 * chunk + 1];
      int word = pack_signed_bytes(
          signed_int4(byte0 & 0xf),
          signed_int4(byte0 >> 4),
          signed_int4(byte1 & 0xf),
          signed_int4(byte1 >> 4));
      dot = __dp4a(
          word,
          reinterpret_cast<const int*>(query)[chunk],
          dot);
    }
  } else if (bits == 2) {
#pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
      uint8_t byte = packed[chunk];
      int word = pack_signed_bytes(
          signed_int2(byte & 0x3),
          signed_int2((byte >> 2) & 0x3),
          signed_int2((byte >> 4) & 0x3),
          signed_int2(byte >> 6));
      dot = __dp4a(
          word,
          reinterpret_cast<const int*>(query)[chunk],
          dot);
    }
  } else if (bits == 1) {
#pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
      uint8_t nibble = packed[chunk / 2];
      nibble = chunk % 2 == 0 ? (nibble & 0xf) : (nibble >> 4);
      int word = pack_signed_bytes(
          (nibble & 0x1) ? 1 : -1,
          (nibble & 0x2) ? 1 : -1,
          (nibble & 0x4) ? 1 : -1,
          (nibble & 0x8) ? 1 : -1);
      dot = __dp4a(
          word,
          reinterpret_cast<const int*>(query)[chunk],
          dot);
    }
  }
  return dot;
}

template <typename scale_t>
__device__ __forceinline__ float sharedtail_score_one(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    int batch_kv,
    int query_group,
    int query_groups,
    int capacity,
    int token) {
  int64_t query_row =
      static_cast<int64_t>(batch_kv) * query_groups + query_group;
  const int8_t* query = query_codes + query_row * 96;
  const scale_t* qscale = query_scales + query_row * 6;
  const uint8_t* token_codes = packed_codes
      + (static_cast<int64_t>(batch_kv) * capacity + token) * 24;
  const scale_t* token_scales = key_scales
      + (static_cast<int64_t>(batch_kv) * capacity + token) * 3;
  float score = 0.0f;
  score += static_cast<float>(band_dot(token_codes, query, 4))
      * static_cast<float>(qscale[0])
      * static_cast<float>(token_scales[0]);
  score += static_cast<float>(band_dot(token_codes + 8, query + 16, 4))
      * static_cast<float>(qscale[1])
      * static_cast<float>(token_scales[1]);
  float tail = 0.0f;
#pragma unroll
  for (int band = 0; band < 4; ++band) {
    tail += static_cast<float>(
        band_dot(token_codes + 16 + 2 * band, query + 32 + 16 * band, 1))
        * static_cast<float>(qscale[2 + band]);
  }
  score += tail * static_cast<float>(token_scales[2]);
  return score;
}

template <typename scale_t>
__global__ void sharedtail_scores_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    float* __restrict__ output,
    int kv_head_count,
    int query_groups,
    int capacity,
    int history_count) {
  int token = blockIdx.y * blockDim.x + threadIdx.x;
  if (token >= history_count) {
    return;
  }
  int batch_kv = blockIdx.x;
  int batch = batch_kv / kv_head_count;
  int kv_head = batch_kv - batch * kv_head_count;
  int query_head_count = kv_head_count * query_groups;
  for (int query_group = 0; query_group < query_groups; ++query_group) {
    float score = sharedtail_score_one(
        query_codes,
        query_scales,
        packed_codes,
        key_scales,
        batch_kv,
        query_group,
        query_groups,
        capacity,
        token);
    int query_head = kv_head * query_groups + query_group;
    output[
        (static_cast<int64_t>(batch) * query_head_count + query_head)
            * history_count
        + token] = score;
  }
}

template <typename scale_t>
__global__ void sharedtail_sample_threshold_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    float* __restrict__ thresholds,
    int kv_head_count,
    int query_groups,
    int capacity,
    int history_count,
    int sample_count,
    int selected_keep) {
  int row = blockIdx.x;
  int thread = threadIdx.x;
  int query_head_count = kv_head_count * query_groups;
  int batch = row / query_head_count;
  int query_head = row - batch * query_head_count;
  int kv_head = query_head / query_groups;
  int query_group = query_head - kv_head * query_groups;
  int batch_kv = batch * kv_head_count + kv_head;
  int sort_count = 1;
  while (sort_count < sample_count) {
    sort_count <<= 1;
  }
  extern __shared__ float samples[];
  for (
      int sample = thread;
      sample < sort_count;
      sample += blockDim.x) {
    float score = -INFINITY;
    if (sample < sample_count) {
      int segment = max(1, history_count / sample_count);
      int phase = (row * 131 + 17) % segment;
      int64_t centered = (
          static_cast<int64_t>(2 * sample + 1) * history_count)
          / (2 * sample_count);
      int token = static_cast<int>((centered + phase) % history_count);
      score = sharedtail_score_one(
          query_codes,
          query_scales,
          packed_codes,
          key_scales,
          batch_kv,
          query_group,
          query_groups,
          capacity,
          token);
    }
    samples[sample] = score;
  }
  __syncthreads();
  for (int size = 2; size <= sort_count; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (
          int sample = thread;
          sample < sort_count;
          sample += blockDim.x) {
        int other = sample ^ stride;
        if (other > sample) {
          bool ascending = (sample & size) == 0;
          float left = samples[sample];
          float right = samples[other];
          if ((left > right) == ascending) {
            samples[sample] = right;
            samples[other] = left;
          }
        }
      }
      __syncthreads();
    }
  }
  if (thread == 0) {
    thresholds[row] = samples[sort_count - selected_keep];
  }
}

template <typename scale_t>
__global__ void sharedtail_threshold_compact_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const float* __restrict__ thresholds,
    int64_t* __restrict__ candidate_indices,
    float* __restrict__ candidate_scores,
    int64_t* __restrict__ candidate_counts,
    bool* __restrict__ overflow,
    int kv_head_count,
    int query_groups,
    int capacity,
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
    float score = sharedtail_score_one(
        query_codes,
        query_scales,
        packed_codes,
        key_scales,
        batch_kv,
        query_group,
        query_groups,
        capacity,
        token);
    if (score < thresholds[row]) {
      continue;
    }
    unsigned long long* count = reinterpret_cast<unsigned long long*>(
        candidate_counts + row);
    unsigned long long slot = atomicAdd(count, 1ULL);
    if (slot < static_cast<unsigned long long>(candidate_capacity)) {
      int64_t output_offset =
          static_cast<int64_t>(row) * candidate_capacity + slot;
      candidate_indices[output_offset] = token;
      candidate_scores[output_offset] = score;
    } else {
      overflow[row] = true;
    }
  }
}

template <typename scale_t>
__device__ __forceinline__ float score_one(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const scale_t* __restrict__ score_bias,
    const int8_t* __restrict__ bit_allocations,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int64_t* __restrict__ code_bases,
    const int64_t* __restrict__ scale_bases,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    int batch_kv,
    int query_group,
    int query_groups,
    int token,
    int bias_capacity) {
  int64_t query_row =
      static_cast<int64_t>(batch_kv) * query_groups + query_group;
  const int8_t* query = query_codes + query_row * 128;
  const scale_t* qscale = query_scales + query_row * 8;
  int code_stride = static_cast<int>(code_strides[batch_kv]);
  int scale_stride = static_cast<int>(scale_strides[batch_kv]);
  const uint8_t* token_codes = packed_codes
      + code_bases[batch_kv]
      + static_cast<int64_t>(token) * code_stride;
  const scale_t* token_scales = key_scales
      + scale_bases[batch_kv]
      + static_cast<int64_t>(token) * scale_stride;
  const int8_t* allocations = bit_allocations + batch_kv * 8;
  const int16_t* offsets = code_offsets + batch_kv * 8;
  const int8_t* scale_index = scale_offsets + batch_kv * 8;
  float score = 0.0f;
  bool zero_tail =
      allocations[4] == 0 && allocations[5] == 0
      && allocations[6] == 0 && allocations[7] == 0;
  if (
      zero_tail && allocations[0] == 8 && allocations[1] == 4
      && allocations[2] == 0 && allocations[3] == 0) {
    int dot0 = band_dot(token_codes, query, 8);
    int dot1 = band_dot(token_codes + 16, query + 16, 4);
    score =
        static_cast<float>(dot0)
            * static_cast<float>(qscale[0])
            * static_cast<float>(token_scales[0])
        + static_cast<float>(dot1)
            * static_cast<float>(qscale[1])
            * static_cast<float>(token_scales[1]);
  } else if (
      zero_tail && allocations[0] == 8 && allocations[1] == 1
      && allocations[2] == 1 && allocations[3] == 1) {
    int dot0 = band_dot(token_codes, query, 8);
    int dot1 = band_dot(token_codes + 16, query + 16, 1);
    int dot2 = band_dot(token_codes + 18, query + 32, 1);
    int dot3 = band_dot(token_codes + 20, query + 48, 1);
    score =
        static_cast<float>(dot0)
            * static_cast<float>(qscale[0])
            * static_cast<float>(token_scales[0])
        + static_cast<float>(dot1)
            * static_cast<float>(qscale[1])
            * static_cast<float>(token_scales[1])
        + static_cast<float>(dot2)
            * static_cast<float>(qscale[2])
            * static_cast<float>(token_scales[2])
        + static_cast<float>(dot3)
            * static_cast<float>(qscale[3])
            * static_cast<float>(token_scales[3]);
  } else if (
      zero_tail && allocations[0] == 4 && allocations[1] == 4
      && allocations[2] == 4 && allocations[3] == 0) {
    int dot0 = band_dot(token_codes, query, 4);
    int dot1 = band_dot(token_codes + 8, query + 16, 4);
    int dot2 = band_dot(token_codes + 16, query + 32, 4);
    score =
        static_cast<float>(dot0)
            * static_cast<float>(qscale[0])
            * static_cast<float>(token_scales[0])
        + static_cast<float>(dot1)
            * static_cast<float>(qscale[1])
            * static_cast<float>(token_scales[1])
        + static_cast<float>(dot2)
            * static_cast<float>(qscale[2])
            * static_cast<float>(token_scales[2]);
  } else if (
      zero_tail && allocations[0] == 4 && allocations[1] == 4
      && allocations[2] == 1 && allocations[3] == 0) {
    int dot0 = band_dot(token_codes, query, 4);
    int dot1 = band_dot(token_codes + 8, query + 16, 4);
    int dot2 = band_dot(token_codes + 16, query + 32, 1);
    score =
        static_cast<float>(dot0)
            * static_cast<float>(qscale[0])
            * static_cast<float>(token_scales[0])
        + static_cast<float>(dot1)
            * static_cast<float>(qscale[1])
            * static_cast<float>(token_scales[1])
        + static_cast<float>(dot2)
            * static_cast<float>(qscale[2])
            * static_cast<float>(token_scales[2]);
  } else {
#pragma unroll
    for (int band = 0; band < 8; ++band) {
      int bits = static_cast<int>(allocations[band]);
      if (bits == 0) {
        continue;
      }
      int dot = band_dot(
          token_codes + offsets[band],
          query + 16 * band,
          bits);
      score += static_cast<float>(dot)
          * static_cast<float>(qscale[band])
          * static_cast<float>(token_scales[scale_index[band]]);
    }
  }
  if (bias_capacity > 0) {
    score += static_cast<float>(
        score_bias[
            static_cast<int64_t>(batch_kv) * bias_capacity + token]);
  }
  return score;
}

template <typename scale_t>
__global__ void varbit_scores_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const scale_t* __restrict__ score_bias,
    const int8_t* __restrict__ bit_allocations,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int64_t* __restrict__ code_bases,
    const int64_t* __restrict__ scale_bases,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    float* __restrict__ output,
    int kv_head_count,
    int query_groups,
    int history_count,
    int bias_capacity) {
  int token = blockIdx.y * blockDim.x + threadIdx.x;
  if (token >= history_count) {
    return;
  }
  int batch_kv = blockIdx.x;
  int batch = batch_kv / kv_head_count;
  int kv_head = batch_kv - batch * kv_head_count;
  int query_head_count = kv_head_count * query_groups;
  for (int query_group = 0; query_group < query_groups; ++query_group) {
    float score = score_one(
        query_codes,
        query_scales,
        packed_codes,
        key_scales,
        score_bias,
        bit_allocations,
        code_offsets,
        scale_offsets,
        code_bases,
        scale_bases,
        code_strides,
        scale_strides,
        batch_kv,
        query_group,
        query_groups,
        token,
        bias_capacity);
    int query_head = kv_head * query_groups + query_group;
    output[
        (static_cast<int64_t>(batch) * query_head_count + query_head)
            * history_count
        + token] = score;
  }
}

template <typename scale_t>
__global__ void varbit_sample_threshold_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const scale_t* __restrict__ score_bias,
    const int8_t* __restrict__ bit_allocations,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int64_t* __restrict__ code_bases,
    const int64_t* __restrict__ scale_bases,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    float* __restrict__ thresholds,
    int kv_head_count,
    int query_groups,
    int history_count,
    int sample_count,
    int selected_keep,
    int bias_capacity) {
  int row = blockIdx.x;
  int thread = threadIdx.x;
  int query_head_count = kv_head_count * query_groups;
  int batch = row / query_head_count;
  int query_head = row - batch * query_head_count;
  int kv_head = query_head / query_groups;
  int query_group = query_head - kv_head * query_groups;
  int batch_kv = batch * kv_head_count + kv_head;
  int sort_count = 1;
  while (sort_count < sample_count) {
    sort_count <<= 1;
  }
  extern __shared__ float samples[];
  for (
      int sample = thread;
      sample < sort_count;
      sample += blockDim.x) {
    float score = -INFINITY;
    if (sample < sample_count) {
      int segment = max(1, history_count / sample_count);
      int phase = (row * 131 + 17) % segment;
      int64_t centered = (
          static_cast<int64_t>(2 * sample + 1) * history_count)
          / (2 * sample_count);
      int token = static_cast<int>(
          (centered + phase) % history_count);
      score = score_one(
          query_codes,
          query_scales,
          packed_codes,
          key_scales,
          score_bias,
          bit_allocations,
          code_offsets,
          scale_offsets,
          code_bases,
          scale_bases,
          code_strides,
          scale_strides,
          batch_kv,
          query_group,
          query_groups,
          token,
          bias_capacity);
    }
    samples[sample] = score;
  }
  __syncthreads();
  for (
      int size = 2;
      size <= sort_count;
      size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (
          int sample = thread;
          sample < sort_count;
          sample += blockDim.x) {
        int other = sample ^ stride;
        if (other > sample) {
          bool ascending = (sample & size) == 0;
          float left = samples[sample];
          float right = samples[other];
          if ((left > right) == ascending) {
            samples[sample] = right;
            samples[other] = left;
          }
        }
      }
      __syncthreads();
    }
  }
  if (thread == 0) {
    thresholds[row] = samples[sort_count - selected_keep];
  }
}

template <typename scale_t>
__global__ void varbit_threshold_compact_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const scale_t* __restrict__ score_bias,
    const int8_t* __restrict__ bit_allocations,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int64_t* __restrict__ code_bases,
    const int64_t* __restrict__ scale_bases,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    const float* __restrict__ thresholds,
    int64_t* __restrict__ candidate_indices,
    float* __restrict__ candidate_scores,
    int64_t* __restrict__ candidate_counts,
    bool* __restrict__ overflow,
    int kv_head_count,
    int query_groups,
    int history_count,
    int candidate_capacity,
    int bias_capacity) {
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
    float score = score_one(
        query_codes,
        query_scales,
        packed_codes,
        key_scales,
        score_bias,
        bit_allocations,
        code_offsets,
        scale_offsets,
        code_bases,
        scale_bases,
        code_strides,
        scale_strides,
        batch_kv,
        query_group,
        query_groups,
        token,
        bias_capacity);
    if (score < thresholds[row]) {
      continue;
    }
    unsigned long long* count = reinterpret_cast<unsigned long long*>(
        candidate_counts + row);
    unsigned long long slot = atomicAdd(count, 1ULL);
    if (slot < static_cast<unsigned long long>(candidate_capacity)) {
      int64_t output_offset =
          static_cast<int64_t>(row) * candidate_capacity + slot;
      candidate_indices[output_offset] = token;
      candidate_scores[output_offset] = score;
    } else {
      overflow[row] = true;
    }
  }
}

__global__ void finalize_counts_kernel(
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

void check_inputs(
    const torch::Tensor& query_codes,
    const torch::Tensor& query_scales,
    const torch::Tensor& packed_codes,
    const torch::Tensor& key_scales,
    const torch::Tensor& score_bias,
    const torch::Tensor& bit_allocations,
    const torch::Tensor& code_offsets,
    const torch::Tensor& scale_offsets,
    const torch::Tensor& code_bases,
    const torch::Tensor& scale_bases,
    const torch::Tensor& code_strides,
    const torch::Tensor& scale_strides,
    int64_t history_count) {
  TORCH_CHECK(
      query_codes.is_cuda() && query_scales.is_cuda()
          && packed_codes.is_cuda() && key_scales.is_cuda()
          && score_bias.is_cuda()
          && bit_allocations.is_cuda() && code_offsets.is_cuda()
          && scale_offsets.is_cuda() && code_bases.is_cuda()
          && scale_bases.is_cuda() && code_strides.is_cuda()
          && scale_strides.is_cuda(),
      "all inputs must be CUDA tensors");
  TORCH_CHECK(
      query_codes.scalar_type() == at::kChar
          && bit_allocations.scalar_type() == at::kChar
          && scale_offsets.scalar_type() == at::kChar
          && scale_strides.scalar_type() == at::kChar,
      "query/allocation/scale metadata dtypes are invalid");
  TORCH_CHECK(
      packed_codes.scalar_type() == at::kByte,
      "packed codes must be uint8");
  TORCH_CHECK(
      code_offsets.scalar_type() == at::kShort
          && code_strides.scalar_type() == at::kShort,
      "code offsets and strides must be int16");
  TORCH_CHECK(
      code_bases.scalar_type() == at::kLong
          && scale_bases.scalar_type() == at::kLong,
      "head bases must be int64");
  TORCH_CHECK(
      query_scales.scalar_type() == key_scales.scalar_type(),
      "query/key scale dtypes must match");
  TORCH_CHECK(
      score_bias.numel() == 0
          || score_bias.scalar_type() == key_scales.scalar_type(),
      "score bias dtype must match key scales");
  TORCH_CHECK(
      query_codes.dim() == 4 && query_codes.size(3) == 128,
      "query codes must have shape [batch, kv_heads, groups, 128]");
  TORCH_CHECK(
      query_scales.dim() == 4 && query_scales.size(3) == 8,
      "query scales must have shape [batch, kv_heads, groups, 8]");
  TORCH_CHECK(
      bit_allocations.dim() == 3 && bit_allocations.size(2) == 8,
      "bit allocations must have shape [batch, kv_heads, 8]");
  TORCH_CHECK(
      code_offsets.sizes() == bit_allocations.sizes()
          && scale_offsets.sizes() == bit_allocations.sizes(),
      "band metadata shapes must match allocations");
  TORCH_CHECK(
      code_bases.sizes() == code_strides.sizes()
          && scale_bases.sizes() == scale_strides.sizes()
          && code_bases.dim() == 2,
      "head metadata must have shape [batch, kv_heads]");
  if (score_bias.numel() > 0) {
    TORCH_CHECK(
        score_bias.dim() == 3
            && score_bias.size(0) == query_codes.size(0)
            && score_bias.size(1) == query_codes.size(1)
            && history_count <= score_bias.size(2),
        "score bias must have shape [batch, kv_heads, capacity]");
  }
  TORCH_CHECK(history_count > 0, "history count must be positive");
}

std::vector<torch::Tensor> varbit_spectral_quantize_query_forward(
    torch::Tensor projected_query) {
  TORCH_CHECK(
      projected_query.is_cuda(),
      "variable-bit query preparation requires CUDA");
  TORCH_CHECK(
      projected_query.dim() == 4 && projected_query.size(3) == 128,
      "projected query must have shape [batch, kv_heads, groups, 128]");
  int row_count = static_cast<int>(
      projected_query.size(0)
      * projected_query.size(1)
      * projected_query.size(2));
  auto scale_shape = projected_query.sizes().vec();
  scale_shape.back() = 8;
  auto output_codes = torch::empty(
      projected_query.sizes(),
      projected_query.options().dtype(at::kChar));
  auto output_scales = torch::empty(
      scale_shape,
      projected_query.options());
  c10::cuda::CUDAGuard device_guard(projected_query.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      projected_query.scalar_type(),
      "varbit_spectral_quantize_query_forward",
      [&] {
        varbit_quantize_query_kernel<scalar_t><<<
            row_count, 128, 0, stream>>>(
                projected_query.data_ptr<scalar_t>(),
                output_codes.data_ptr<int8_t>(),
                output_scales.data_ptr<scalar_t>());
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output_codes, output_scales};
}

void varbit_spectral_encode_projected_out_forward(
    torch::Tensor projected_keys,
    torch::Tensor scale_metrics,
    torch::Tensor precomputed_scales,
    torch::Tensor exact_query_mean,
    torch::Tensor proxy_query_mean,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor score_bias,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    int64_t start,
    bool metric_scale,
    bool override_scale,
    double bias_shrinkage) {
  TORCH_CHECK(
      projected_keys.is_cuda() && packed_codes.is_cuda()
          && key_scales.is_cuda() && score_bias.is_cuda()
          && bit_allocations.is_cuda()
          && code_offsets.is_cuda() && scale_offsets.is_cuda()
          && code_bases.is_cuda() && scale_bases.is_cuda()
          && code_strides.is_cuda() && scale_strides.is_cuda(),
      "all packed encoder inputs must be CUDA tensors");
  TORCH_CHECK(
      projected_keys.dim() == 4 && projected_keys.size(3) == 128,
      "projected keys must have shape [batch, kv_heads, tokens, 128]");
  TORCH_CHECK(
      projected_keys.scalar_type() == key_scales.scalar_type(),
      "projected keys and key scales must have the same dtype");
  TORCH_CHECK(
      packed_codes.scalar_type() == at::kByte
          && bit_allocations.scalar_type() == at::kChar
          && scale_offsets.scalar_type() == at::kChar
          && scale_strides.scalar_type() == at::kChar
          && code_offsets.scalar_type() == at::kShort
          && code_strides.scalar_type() == at::kShort
          && code_bases.scalar_type() == at::kLong
          && scale_bases.scalar_type() == at::kLong,
      "packed encoder metadata dtypes are invalid");
  bool use_score_bias = score_bias.numel() > 0;
  if (use_score_bias) {
    TORCH_CHECK(
        exact_query_mean.is_cuda() && proxy_query_mean.is_cuda()
            && exact_query_mean.scalar_type() == at::kFloat
            && proxy_query_mean.scalar_type() == at::kFloat,
        "score bias requires CUDA float32 query means");
    TORCH_CHECK(
        exact_query_mean.dim() == 3
            && exact_query_mean.size(0) == projected_keys.size(0)
            && exact_query_mean.size(1) == projected_keys.size(1)
            && exact_query_mean.size(2) == 128
            && proxy_query_mean.sizes() == exact_query_mean.sizes(),
        "query means must have shape [batch, kv_heads, 128]");
    TORCH_CHECK(
        score_bias.dim() == 3
            && score_bias.size(0) == projected_keys.size(0)
            && score_bias.size(1) == projected_keys.size(1)
            && score_bias.scalar_type() == projected_keys.scalar_type(),
        "score bias must match projected-key batch/head/dtype");
    TORCH_CHECK(
        bias_shrinkage >= 0.0 && bias_shrinkage <= 1.0,
        "score-bias shrinkage must be in [0, 1]");
  }
  TORCH_CHECK(start >= 0, "packed encoder start must be nonnegative");
  if (metric_scale) {
    TORCH_CHECK(
        scale_metrics.is_cuda()
            && scale_metrics.scalar_type() == at::kFloat,
        "metric scales require CUDA float32 metric matrices");
    TORCH_CHECK(
        scale_metrics.dim() == 5
            && scale_metrics.size(0) == projected_keys.size(0)
            && scale_metrics.size(1) == projected_keys.size(1)
            && scale_metrics.size(2) == 8
            && scale_metrics.size(3) == 16
            && scale_metrics.size(4) == 16,
        "scale metrics must have shape [batch, kv_heads, 8, 16, 16]");
  }
  if (override_scale) {
    TORCH_CHECK(
        precomputed_scales.is_cuda()
            && precomputed_scales.scalar_type()
                == projected_keys.scalar_type(),
        "precomputed scales must be CUDA and match projected-key dtype");
    TORCH_CHECK(
        precomputed_scales.dim() == 4
            && precomputed_scales.size(0) == projected_keys.size(0)
            && precomputed_scales.size(1) == projected_keys.size(1)
            && precomputed_scales.size(2) == projected_keys.size(2)
            && precomputed_scales.size(3) == 8,
        "precomputed scales must have shape [batch, kv_heads, tokens, 8]");
  }
  int batch_kv = static_cast<int>(
      projected_keys.size(0) * projected_keys.size(1));
  int token_count = static_cast<int>(projected_keys.size(2));
  int bias_capacity = use_score_bias
      ? static_cast<int>(score_bias.size(2))
      : 0;
  TORCH_CHECK(
      start + token_count <= (use_score_bias ? bias_capacity : start + token_count),
      "encoded rows exceed score-bias capacity");
  if (token_count == 0) {
    return;
  }
  c10::cuda::CUDAGuard device_guard(projected_keys.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      projected_keys.scalar_type(),
      "varbit_spectral_encode_projected_out",
      [&] {
        varbit_encode_projected_kernel<scalar_t>
            <<<
                batch_kv * token_count,
                128,
                0,
                stream>>>(
                projected_keys.data_ptr<scalar_t>(),
                metric_scale
                    ? scale_metrics.data_ptr<float>()
                    : nullptr,
                override_scale
                    ? precomputed_scales.data_ptr<scalar_t>()
                    : nullptr,
                use_score_bias
                    ? exact_query_mean.data_ptr<float>()
                    : nullptr,
                use_score_bias
                    ? proxy_query_mean.data_ptr<float>()
                    : nullptr,
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                use_score_bias
                    ? score_bias.data_ptr<scalar_t>()
                    : nullptr,
                bit_allocations.data_ptr<int8_t>(),
                code_offsets.data_ptr<int16_t>(),
                scale_offsets.data_ptr<int8_t>(),
                code_bases.data_ptr<int64_t>(),
                scale_bases.data_ptr<int64_t>(),
                code_strides.data_ptr<int16_t>(),
                scale_strides.data_ptr<int8_t>(),
                token_count,
                static_cast<int>(start),
                metric_scale,
                override_scale,
                bias_capacity,
                static_cast<float>(bias_shrinkage));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor varbit_spectral_scores_forward(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor score_bias,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    int64_t history_count) {
  check_inputs(
      query_codes, query_scales, packed_codes, key_scales, score_bias,
      bit_allocations, code_offsets, scale_offsets, code_bases,
      scale_bases, code_strides, scale_strides, history_count);
  int batch_count = static_cast<int>(query_codes.size(0));
  int kv_head_count = static_cast<int>(query_codes.size(1));
  int query_groups = static_cast<int>(query_codes.size(2));
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
      "varbit_spectral_scores_forward",
      [&] {
        varbit_scores_kernel<scalar_t><<<
            blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                score_bias.numel() > 0
                    ? score_bias.data_ptr<scalar_t>()
                    : nullptr,
                bit_allocations.data_ptr<int8_t>(),
                code_offsets.data_ptr<int16_t>(),
                scale_offsets.data_ptr<int8_t>(),
                code_bases.data_ptr<int64_t>(),
                scale_bases.data_ptr<int64_t>(),
                code_strides.data_ptr<int16_t>(),
                scale_strides.data_ptr<int8_t>(),
                output.data_ptr<float>(),
                kv_head_count,
                query_groups,
                static_cast<int>(history_count),
                score_bias.numel() > 0
                    ? static_cast<int>(score_bias.size(2))
                    : 0);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> varbit_spectral_sampled_compact_forward(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor score_bias,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    int64_t history_count,
    int64_t sample_count,
    double selected_fraction,
    int64_t candidate_capacity) {
  check_inputs(
      query_codes, query_scales, packed_codes, key_scales, score_bias,
      bit_allocations, code_offsets, scale_offsets, code_bases,
      scale_bases, code_strides, scale_strides, history_count);
  TORCH_CHECK(
      sample_count > 0
          && sample_count <= VARBIT_MAX_SAMPLE_COUNT
          && sample_count <= history_count,
      "sample count must be in [1, min(2048, history)]");
  TORCH_CHECK(
      selected_fraction > 0.0 && selected_fraction < 1.0,
      "selected fraction must be in (0, 1)");
  TORCH_CHECK(
      candidate_capacity > 0 && candidate_capacity <= history_count,
      "candidate capacity must be in [1, history]");
  int batch_count = static_cast<int>(query_codes.size(0));
  int kv_head_count = static_cast<int>(query_codes.size(1));
  int query_groups = static_cast<int>(query_codes.size(2));
  int query_head_count = kv_head_count * query_groups;
  int row_count = batch_count * query_head_count;
  int selected_keep = std::max(
      1, static_cast<int>(ceil(selected_fraction * sample_count)));
  c10::cuda::CUDAGuard device_guard(query_codes.device());
  auto thresholds = torch::empty(
      {batch_count, query_head_count},
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
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_scales.scalar_type(),
      "varbit_spectral_sampled_compact_forward",
      [&] {
        varbit_sample_threshold_kernel<scalar_t><<<
            row_count,
            256,
            VARBIT_MAX_SAMPLE_COUNT * sizeof(float),
            at::cuda::getCurrentCUDAStream()>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                score_bias.numel() > 0
                    ? score_bias.data_ptr<scalar_t>()
                    : nullptr,
                bit_allocations.data_ptr<int8_t>(),
                code_offsets.data_ptr<int16_t>(),
                scale_offsets.data_ptr<int8_t>(),
                code_bases.data_ptr<int64_t>(),
                scale_bases.data_ptr<int64_t>(),
                code_strides.data_ptr<int16_t>(),
                scale_strides.data_ptr<int8_t>(),
                thresholds.data_ptr<float>(),
                kv_head_count,
                query_groups,
                static_cast<int>(history_count),
                static_cast<int>(sample_count),
                selected_keep,
                score_bias.numel() > 0
                    ? static_cast<int>(score_bias.size(2))
                    : 0);
        dim3 compact_blocks(
            batch_count * kv_head_count,
            (history_count + 255) / 256);
        varbit_threshold_compact_kernel<scalar_t><<<
            compact_blocks,
            256,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                score_bias.numel() > 0
                    ? score_bias.data_ptr<scalar_t>()
                    : nullptr,
                bit_allocations.data_ptr<int8_t>(),
                code_offsets.data_ptr<int16_t>(),
                scale_offsets.data_ptr<int8_t>(),
                code_bases.data_ptr<int64_t>(),
                scale_bases.data_ptr<int64_t>(),
                code_strides.data_ptr<int16_t>(),
                scale_strides.data_ptr<int8_t>(),
                thresholds.data_ptr<float>(),
                indices.data_ptr<int64_t>(),
                scores.data_ptr<float>(),
                counts.data_ptr<int64_t>(),
                overflow.data_ptr<bool>(),
                kv_head_count,
                query_groups,
                static_cast<int>(history_count),
                static_cast<int>(candidate_capacity),
                score_bias.numel() > 0
                    ? static_cast<int>(score_bias.size(2))
                    : 0);
      });
  finalize_counts_kernel<<<
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

void varbit_spectral_sampled_compact_out_forward(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor score_bias,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_scores,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor overflow,
    int64_t history_count,
    int64_t sample_count,
    double selected_fraction) {
  check_inputs(
      query_codes, query_scales, packed_codes, key_scales, score_bias,
      bit_allocations, code_offsets, scale_offsets, code_bases,
      scale_bases, code_strides, scale_strides, history_count);
  TORCH_CHECK(
      candidate_indices.is_cuda() && candidate_scores.is_cuda()
          && candidate_counts.is_cuda() && thresholds.is_cuda()
          && overflow.is_cuda(),
      "persistent outputs must be CUDA tensors");
  TORCH_CHECK(
      candidate_indices.scalar_type() == at::kLong
          && candidate_scores.scalar_type() == at::kFloat
          && candidate_counts.scalar_type() == at::kLong
          && thresholds.scalar_type() == at::kFloat
          && overflow.scalar_type() == at::kBool,
      "persistent output dtypes are invalid");
  TORCH_CHECK(
      sample_count > 0
          && sample_count <= VARBIT_MAX_SAMPLE_COUNT
          && sample_count <= history_count,
      "sample count must be in [1, min(2048, history)]");
  TORCH_CHECK(
      selected_fraction > 0.0 && selected_fraction < 1.0,
      "selected fraction must be in (0, 1)");
  int batch_count = static_cast<int>(query_codes.size(0));
  int kv_head_count = static_cast<int>(query_codes.size(1));
  int query_groups = static_cast<int>(query_codes.size(2));
  int query_head_count = kv_head_count * query_groups;
  int row_count = batch_count * query_head_count;
  TORCH_CHECK(
      candidate_indices.dim() == 3
          && candidate_indices.size(0) == batch_count
          && candidate_indices.size(1) == query_head_count,
      "candidate index shape mismatch");
  TORCH_CHECK(
      candidate_scores.sizes() == candidate_indices.sizes(),
      "candidate score shape mismatch");
  TORCH_CHECK(
      candidate_counts.dim() == 2
          && candidate_counts.size(0) == batch_count
          && candidate_counts.size(1) == query_head_count,
      "candidate count shape mismatch");
  TORCH_CHECK(
      thresholds.sizes() == candidate_counts.sizes()
          && overflow.sizes() == candidate_counts.sizes(),
      "threshold/overflow shape mismatch");
  int candidate_capacity = static_cast<int>(candidate_indices.size(2));
  TORCH_CHECK(
      candidate_capacity > 0 && candidate_capacity <= history_count,
      "candidate capacity must be in [1, history]");
  int selected_keep = std::max(
      1, static_cast<int>(ceil(selected_fraction * sample_count)));
  c10::cuda::CUDAGuard device_guard(query_codes.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(
      candidate_counts.data_ptr<int64_t>(),
      0,
      candidate_counts.numel() * sizeof(int64_t),
      stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      overflow.data_ptr<bool>(),
      0,
      overflow.numel() * sizeof(bool),
      stream));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_scales.scalar_type(),
      "varbit_spectral_sampled_compact_out_forward",
      [&] {
        varbit_sample_threshold_kernel<scalar_t><<<
            row_count,
            256,
            VARBIT_MAX_SAMPLE_COUNT * sizeof(float),
            stream>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                score_bias.numel() > 0
                    ? score_bias.data_ptr<scalar_t>()
                    : nullptr,
                bit_allocations.data_ptr<int8_t>(),
                code_offsets.data_ptr<int16_t>(),
                scale_offsets.data_ptr<int8_t>(),
                code_bases.data_ptr<int64_t>(),
                scale_bases.data_ptr<int64_t>(),
                code_strides.data_ptr<int16_t>(),
                scale_strides.data_ptr<int8_t>(),
                thresholds.data_ptr<float>(),
                kv_head_count,
                query_groups,
                static_cast<int>(history_count),
                static_cast<int>(sample_count),
                selected_keep,
                score_bias.numel() > 0
                    ? static_cast<int>(score_bias.size(2))
                    : 0);
        dim3 compact_blocks(
            batch_count * kv_head_count,
            (history_count + 255) / 256);
        varbit_threshold_compact_kernel<scalar_t><<<
            compact_blocks,
            256,
            0,
            stream>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                score_bias.numel() > 0
                    ? score_bias.data_ptr<scalar_t>()
                    : nullptr,
                bit_allocations.data_ptr<int8_t>(),
                code_offsets.data_ptr<int16_t>(),
                scale_offsets.data_ptr<int8_t>(),
                code_bases.data_ptr<int64_t>(),
                scale_bases.data_ptr<int64_t>(),
                code_strides.data_ptr<int16_t>(),
                scale_strides.data_ptr<int8_t>(),
                thresholds.data_ptr<float>(),
                candidate_indices.data_ptr<int64_t>(),
                candidate_scores.data_ptr<float>(),
                candidate_counts.data_ptr<int64_t>(),
                overflow.data_ptr<bool>(),
                kv_head_count,
                query_groups,
                static_cast<int>(history_count),
                candidate_capacity,
                score_bias.numel() > 0
                    ? static_cast<int>(score_bias.size(2))
                    : 0);
      });
  finalize_counts_kernel<<<
      (row_count + 255) / 256,
      256,
      0,
      stream>>>(
          candidate_counts.data_ptr<int64_t>(),
          overflow.data_ptr<bool>(),
          row_count,
          candidate_capacity);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_sharedtail_inputs(
    const torch::Tensor& query_codes,
    const torch::Tensor& query_scales,
    const torch::Tensor& packed_codes,
    const torch::Tensor& key_scales,
    int64_t history_count) {
  TORCH_CHECK(
      query_codes.is_cuda() && query_scales.is_cuda()
          && packed_codes.is_cuda() && key_scales.is_cuda(),
      "all shared-tail inputs must be CUDA tensors");
  TORCH_CHECK(
      query_codes.scalar_type() == at::kChar,
      "shared-tail query codes must be int8");
  TORCH_CHECK(
      packed_codes.scalar_type() == at::kByte,
      "shared-tail packed codes must be uint8");
  TORCH_CHECK(
      query_scales.scalar_type() == key_scales.scalar_type(),
      "shared-tail query/key scale dtypes must match");
  TORCH_CHECK(
      query_codes.dim() == 4 && query_codes.size(3) == 96,
      "shared-tail query codes must have shape [batch, kv_heads, groups, 96]");
  TORCH_CHECK(
      query_scales.dim() == 4 && query_scales.size(3) == 6,
      "shared-tail query scales must have shape [batch, kv_heads, groups, 6]");
  TORCH_CHECK(
      packed_codes.dim() == 4 && packed_codes.size(3) == 24,
      "shared-tail codes must have shape [batch, kv_heads, capacity, 24]");
  TORCH_CHECK(
      key_scales.dim() == 4 && key_scales.size(3) == 3,
      "shared-tail scales must have shape [batch, kv_heads, capacity, 3]");
  TORCH_CHECK(
      query_codes.size(0) == packed_codes.size(0)
          && query_codes.size(1) == packed_codes.size(1)
          && query_scales.size(0) == query_codes.size(0)
          && query_scales.size(1) == query_codes.size(1)
          && query_scales.size(2) == query_codes.size(2)
          && key_scales.size(0) == packed_codes.size(0)
          && key_scales.size(1) == packed_codes.size(1)
          && key_scales.size(2) == packed_codes.size(2),
      "shared-tail batch/head/capacity shapes do not match");
  TORCH_CHECK(
      history_count > 0 && history_count <= packed_codes.size(2),
      "shared-tail history count must be in [1, capacity]");
}

std::vector<torch::Tensor> sharedtail_spectral_quantize_query_forward(
    torch::Tensor projected_query,
    torch::Tensor coordinate_amplitude) {
  TORCH_CHECK(
      projected_query.is_cuda() && coordinate_amplitude.is_cuda(),
      "shared-tail query preparation requires CUDA tensors");
  TORCH_CHECK(
      projected_query.dim() == 4 && projected_query.size(3) == 128,
      "projected query must have shape [batch, kv_heads, groups, 128]");
  TORCH_CHECK(
      coordinate_amplitude.dim() == 3
          && coordinate_amplitude.size(0) == projected_query.size(0)
          && coordinate_amplitude.size(1) == projected_query.size(1)
          && coordinate_amplitude.size(2) == 128,
      "coordinate amplitude must have shape [batch, kv_heads, 128]");
  TORCH_CHECK(
      coordinate_amplitude.scalar_type() == at::kFloat,
      "coordinate amplitude must be float32");
  int batch_count = static_cast<int>(projected_query.size(0));
  int kv_head_count = static_cast<int>(projected_query.size(1));
  int query_groups = static_cast<int>(projected_query.size(2));
  int row_count = batch_count * kv_head_count * query_groups;
  auto code_shape = projected_query.sizes().vec();
  code_shape.back() = 96;
  auto scale_shape = projected_query.sizes().vec();
  scale_shape.back() = 6;
  auto output_codes = torch::empty(
      code_shape, projected_query.options().dtype(at::kChar));
  auto output_scales = torch::empty(
      scale_shape, projected_query.options());
  c10::cuda::CUDAGuard device_guard(projected_query.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      projected_query.scalar_type(),
      "sharedtail_spectral_quantize_query_forward",
      [&] {
        sharedtail_quantize_query_kernel<scalar_t><<<
            row_count, 128, 0, stream>>>(
                projected_query.data_ptr<scalar_t>(),
                coordinate_amplitude.data_ptr<float>(),
                output_codes.data_ptr<int8_t>(),
                output_scales.data_ptr<scalar_t>(),
                query_groups);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output_codes, output_scales};
}

void sharedtail_spectral_encode_projected_out_forward(
    torch::Tensor projected_keys,
    torch::Tensor coordinate_rms,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    int64_t start) {
  TORCH_CHECK(
      projected_keys.is_cuda() && coordinate_rms.is_cuda()
          && packed_codes.is_cuda() && key_scales.is_cuda(),
      "all shared-tail encoder inputs must be CUDA tensors");
  TORCH_CHECK(
      projected_keys.dim() == 4 && projected_keys.size(3) == 128,
      "projected keys must have shape [batch, kv_heads, tokens, 128]");
  TORCH_CHECK(
      coordinate_rms.dim() == 3 && coordinate_rms.size(2) == 128,
      "coordinate RMS must have shape [batch, kv_heads, 128]");
  TORCH_CHECK(
      packed_codes.dim() == 4 && packed_codes.size(3) == 24
          && key_scales.dim() == 4 && key_scales.size(3) == 3,
      "shared-tail packed index shapes are invalid");
  TORCH_CHECK(
      projected_keys.size(0) == coordinate_rms.size(0)
          && projected_keys.size(1) == coordinate_rms.size(1)
          && projected_keys.size(0) == packed_codes.size(0)
          && projected_keys.size(1) == packed_codes.size(1)
          && key_scales.size(0) == packed_codes.size(0)
          && key_scales.size(1) == packed_codes.size(1)
          && key_scales.size(2) == packed_codes.size(2),
      "shared-tail encoder batch/head/capacity shapes do not match");
  TORCH_CHECK(
      coordinate_rms.scalar_type() == at::kFloat
          && projected_keys.scalar_type() == key_scales.scalar_type(),
      "shared-tail RMS must be float32 and key scales must match projected keys");
  TORCH_CHECK(
      packed_codes.scalar_type() == at::kByte,
      "shared-tail packed codes must be uint8");
  TORCH_CHECK(start >= 0, "shared-tail encoder start must be nonnegative");
  int token_count = static_cast<int>(projected_keys.size(2));
  int capacity = static_cast<int>(packed_codes.size(2));
  TORCH_CHECK(
      start + token_count <= capacity,
      "shared-tail encoded rows exceed index capacity");
  if (token_count == 0) {
    return;
  }
  int batch_kv = static_cast<int>(
      projected_keys.size(0) * projected_keys.size(1));
  c10::cuda::CUDAGuard device_guard(projected_keys.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      projected_keys.scalar_type(),
      "sharedtail_spectral_encode_projected_out_forward",
      [&] {
        sharedtail_encode_projected_kernel<scalar_t><<<
            batch_kv * token_count, 128, 0, stream>>>(
                projected_keys.data_ptr<scalar_t>(),
                coordinate_rms.data_ptr<float>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                token_count,
                capacity,
                static_cast<int>(start));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor sharedtail_spectral_scores_forward(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    int64_t history_count) {
  check_sharedtail_inputs(
      query_codes, query_scales, packed_codes, key_scales, history_count);
  int batch_count = static_cast<int>(query_codes.size(0));
  int kv_head_count = static_cast<int>(query_codes.size(1));
  int query_groups = static_cast<int>(query_codes.size(2));
  int capacity = static_cast<int>(packed_codes.size(2));
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
      "sharedtail_spectral_scores_forward",
      [&] {
        sharedtail_scores_kernel<scalar_t><<<
            blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                output.data_ptr<float>(),
                kv_head_count,
                query_groups,
                capacity,
                static_cast<int>(history_count));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

void sharedtail_spectral_sampled_compact_out_forward(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_scores,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor overflow,
    int64_t history_count,
    int64_t sample_count,
    double selected_fraction) {
  check_sharedtail_inputs(
      query_codes, query_scales, packed_codes, key_scales, history_count);
  TORCH_CHECK(
      candidate_indices.is_cuda() && candidate_scores.is_cuda()
          && candidate_counts.is_cuda() && thresholds.is_cuda()
          && overflow.is_cuda(),
      "shared-tail persistent outputs must be CUDA tensors");
  TORCH_CHECK(
      candidate_indices.scalar_type() == at::kLong
          && candidate_scores.scalar_type() == at::kFloat
          && candidate_counts.scalar_type() == at::kLong
          && thresholds.scalar_type() == at::kFloat
          && overflow.scalar_type() == at::kBool,
      "shared-tail persistent output dtypes are invalid");
  TORCH_CHECK(
      sample_count > 0
          && sample_count <= VARBIT_MAX_SAMPLE_COUNT
          && sample_count <= history_count,
      "sample count must be in [1, min(2048, history)]");
  TORCH_CHECK(
      selected_fraction > 0.0 && selected_fraction < 1.0,
      "selected fraction must be in (0, 1)");
  int batch_count = static_cast<int>(query_codes.size(0));
  int kv_head_count = static_cast<int>(query_codes.size(1));
  int query_groups = static_cast<int>(query_codes.size(2));
  int query_head_count = kv_head_count * query_groups;
  int row_count = batch_count * query_head_count;
  int capacity = static_cast<int>(packed_codes.size(2));
  TORCH_CHECK(
      candidate_indices.dim() == 3
          && candidate_indices.size(0) == batch_count
          && candidate_indices.size(1) == query_head_count,
      "shared-tail candidate index shape mismatch");
  TORCH_CHECK(
      candidate_scores.sizes() == candidate_indices.sizes(),
      "shared-tail candidate score shape mismatch");
  TORCH_CHECK(
      candidate_counts.dim() == 2
          && candidate_counts.size(0) == batch_count
          && candidate_counts.size(1) == query_head_count,
      "shared-tail candidate count shape mismatch");
  TORCH_CHECK(
      thresholds.sizes() == candidate_counts.sizes()
          && overflow.sizes() == candidate_counts.sizes(),
      "shared-tail threshold/overflow shape mismatch");
  int candidate_capacity = static_cast<int>(candidate_indices.size(2));
  TORCH_CHECK(
      candidate_capacity > 0 && candidate_capacity <= history_count,
      "shared-tail candidate capacity must be in [1, history]");
  int selected_keep = std::max(
      1, static_cast<int>(ceil(selected_fraction * sample_count)));
  c10::cuda::CUDAGuard device_guard(query_codes.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(
      candidate_counts.data_ptr<int64_t>(),
      0,
      candidate_counts.numel() * sizeof(int64_t),
      stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      overflow.data_ptr<bool>(),
      0,
      overflow.numel() * sizeof(bool),
      stream));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_scales.scalar_type(),
      "sharedtail_spectral_sampled_compact_out_forward",
      [&] {
        sharedtail_sample_threshold_kernel<scalar_t><<<
            row_count,
            256,
            VARBIT_MAX_SAMPLE_COUNT * sizeof(float),
            stream>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                thresholds.data_ptr<float>(),
                kv_head_count,
                query_groups,
                capacity,
                static_cast<int>(history_count),
                static_cast<int>(sample_count),
                selected_keep);
        dim3 compact_blocks(
            batch_count * kv_head_count,
            (history_count + 255) / 256);
        sharedtail_threshold_compact_kernel<scalar_t><<<
            compact_blocks, 256, 0, stream>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                thresholds.data_ptr<float>(),
                candidate_indices.data_ptr<int64_t>(),
                candidate_scores.data_ptr<float>(),
                candidate_counts.data_ptr<int64_t>(),
                overflow.data_ptr<bool>(),
                kv_head_count,
                query_groups,
                capacity,
                static_cast<int>(history_count),
                candidate_capacity);
      });
  finalize_counts_kernel<<<
      (row_count + 255) / 256,
      256,
      0,
      stream>>>(
          candidate_counts.data_ptr<int64_t>(),
          overflow.data_ptr<bool>(),
          row_count,
          candidate_capacity);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


@lru_cache(maxsize=1)
def load_extension():
    return load_inline(
        name="variablebit_spectral_ext_20260727_v14",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=None,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )


def sharedtail_calibration_parameters(
    projected_key_sample: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calibrate the shared envelope and per-coordinate sign amplitudes."""
    if (
        projected_key_sample.ndim != 4
        or projected_key_sample.shape[-1] != 128
    ):
        raise ValueError(
            "projected key sample must have shape "
            "[batch, kv_heads, tokens, 128]"
        )
    working = projected_key_sample.float()
    coordinate_rms = (
        working.square().mean(dim=-2).sqrt().clamp_min(1.0e-8)
    )
    normalized = working / coordinate_rms.unsqueeze(-2)
    envelope = normalized.square().mean(dim=-1).sqrt()
    denominator = envelope.square().sum(dim=-1, keepdim=True).clamp_min(
        1.0e-12
    )
    coordinate_amplitude = (
        envelope.unsqueeze(-1) * working.abs()
    ).sum(dim=-2) / denominator
    return coordinate_rms.contiguous(), coordinate_amplitude.contiguous()


def allocate_sharedtail_index(
    batch_count: int,
    kv_head_count: int,
    capacity: int,
    scale_dtype: torch.dtype,
    device: torch.device,
    coordinate_rms: torch.Tensor,
    coordinate_amplitude: torch.Tensor,
) -> dict[str, Any]:
    """Allocate the fixed 240-bit 4/4/shared-sign index."""
    if capacity <= 0:
        raise ValueError("shared-tail index capacity must be positive")
    expected_shape = (batch_count, kv_head_count, 128)
    if coordinate_rms.shape != expected_shape:
        raise ValueError("shared-tail coordinate RMS shape mismatch")
    if coordinate_amplitude.shape != expected_shape:
        raise ValueError("shared-tail coordinate amplitude shape mismatch")
    return {
        "packed_codes": torch.empty(
            batch_count,
            kv_head_count,
            capacity,
            24,
            dtype=torch.uint8,
            device=device,
        ),
        "key_scales": torch.empty(
            batch_count,
            kv_head_count,
            capacity,
            3,
            dtype=scale_dtype,
            device=device,
        ),
        "coordinate_rms": coordinate_rms.to(
            device=device, dtype=torch.float32
        ).contiguous(),
        "coordinate_amplitude": coordinate_amplitude.to(
            device=device, dtype=torch.float32
        ).contiguous(),
        "capacity": int(capacity),
        "indexed_count": 0,
        "bits_per_head_token": 240,
    }


def encode_sharedtail_projected_keys_into(
    projected_keys: torch.Tensor,
    packed_index: dict[str, Any],
    start: int,
) -> None:
    """Append projected K rows to the fixed shared-tail index."""
    if projected_keys.ndim != 4 or projected_keys.shape[-1] != 128:
        raise ValueError(
            "projected keys must have shape [batch, kv_heads, tokens, 128]"
        )
    token_count = int(projected_keys.shape[-2])
    stop = start + token_count
    if start < 0 or stop > int(packed_index["capacity"]):
        raise ValueError("projected key write exceeds shared-tail capacity")
    if not projected_keys.is_cuda:
        raise ValueError("the fixed shared-tail encoder requires CUDA")
    load_extension().sharedtail_spectral_encode_projected_out_forward(
        projected_keys.contiguous(),
        packed_index["coordinate_rms"],
        packed_index["packed_codes"],
        packed_index["key_scales"],
        start,
    )
    packed_index["indexed_count"] = stop


def quantize_sharedtail_projected_query(
    projected_query: torch.Tensor,
    coordinate_amplitude: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare two INT4-core query bands and four weighted sign-tail bands."""
    if projected_query.ndim != 4 or projected_query.shape[-1] != 128:
        raise ValueError(
            "projected query must have shape [batch, kv_heads, groups, 128]"
        )
    if (
        coordinate_amplitude.shape
        != projected_query.shape[:2] + (128,)
    ):
        raise ValueError("shared-tail coordinate amplitude shape mismatch")
    if projected_query.is_cuda:
        return tuple(
            load_extension().sharedtail_spectral_quantize_query_forward(
                projected_query.contiguous(),
                coordinate_amplitude.to(
                    device=projected_query.device,
                    dtype=torch.float32,
                ).contiguous(),
            )
        )
    grouped = projected_query.float().reshape(
        *projected_query.shape[:-1], 8, 16
    )
    query_scales = grouped.abs().amax(dim=-1).clamp_min(1.0e-8) / 127.0
    query_codes = torch.round(
        grouped / query_scales.unsqueeze(-1)
    ).clamp(-127, 127)

    reconstructed_tail = (
        query_codes[..., 2:6, :]
        * query_scales[..., 2:6].unsqueeze(-1)
    )
    tail_amplitude = coordinate_amplitude[
        ..., 32:96
    ].reshape(*coordinate_amplitude.shape[:-1], 4, 16)
    weighted_tail = reconstructed_tail * tail_amplitude.unsqueeze(-3)
    weighted_scales = (
        weighted_tail.abs().amax(dim=-1).clamp_min(1.0e-8) / 127.0
    )
    weighted_codes = torch.round(
        weighted_tail / weighted_scales.unsqueeze(-1)
    ).clamp(-127, 127)

    packed_query_codes = torch.cat(
        (
            query_codes[..., :2, :].reshape(
                *projected_query.shape[:-1], 32
            ),
            weighted_codes.reshape(*projected_query.shape[:-1], 64),
        ),
        dim=-1,
    )
    packed_query_scales = torch.cat(
        (query_scales[..., :2], weighted_scales),
        dim=-1,
    )
    return (
        packed_query_codes.to(torch.int8).contiguous(),
        packed_query_scales.to(projected_query.dtype).contiguous(),
    )


def sharedtail_scores(
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    packed_index: dict[str, Any],
    history_count: int,
) -> torch.Tensor:
    return load_extension().sharedtail_spectral_scores_forward(
        query_codes,
        query_scales,
        packed_index["packed_codes"],
        packed_index["key_scales"],
        history_count,
    )


def sharedtail_sampled_threshold_compact_out(
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    packed_index: dict[str, Any],
    candidate_indices: torch.Tensor,
    candidate_scores: torch.Tensor,
    candidate_counts: torch.Tensor,
    thresholds: torch.Tensor,
    overflow: torch.Tensor,
    history_count: int,
    sample_count: int,
    selected_fraction: float,
) -> tuple[torch.Tensor, ...]:
    load_extension().sharedtail_spectral_sampled_compact_out_forward(
        query_codes,
        query_scales,
        packed_index["packed_codes"],
        packed_index["key_scales"],
        candidate_indices,
        candidate_scores,
        candidate_counts,
        thresholds,
        overflow,
        history_count,
        sample_count,
        selected_fraction,
    )
    return (
        candidate_indices,
        candidate_scores,
        candidate_counts,
        thresholds,
        overflow,
    )


def make_packed_metadata(
    bit_allocations: torch.Tensor,
    capacity: int,
) -> dict[str, Any]:
    """Build the compact per-head layout consumed by the CUDA scan kernel."""
    if capacity <= 0:
        raise ValueError("packed index capacity must be positive")
    if bit_allocations.ndim != 3 or bit_allocations.shape[-1] != 8:
        raise ValueError("bit allocations must have shape [batch, kv_heads, 8]")
    device = bit_allocations.device
    allocations_cpu = bit_allocations.detach().to("cpu", torch.int8)
    batch_count, kv_head_count, _ = allocations_cpu.shape
    code_offsets = torch.zeros_like(allocations_cpu, dtype=torch.int16)
    scale_offsets = torch.full_like(allocations_cpu, -1, dtype=torch.int8)
    code_strides = torch.zeros(
        batch_count, kv_head_count, dtype=torch.int16
    )
    scale_strides = torch.zeros(
        batch_count, kv_head_count, dtype=torch.int8
    )
    code_bases = torch.zeros(
        batch_count, kv_head_count, dtype=torch.int64
    )
    scale_bases = torch.zeros(
        batch_count, kv_head_count, dtype=torch.int64
    )
    code_cursor = 0
    scale_cursor = 0
    for batch_index in range(batch_count):
        for head_index in range(kv_head_count):
            code_bases[batch_index, head_index] = code_cursor
            scale_bases[batch_index, head_index] = scale_cursor
            code_stride = 0
            scale_stride = 0
            for band_index in range(8):
                bits = int(
                    allocations_cpu[
                        batch_index, head_index, band_index
                    ].item()
                )
                if bits not in {0, 1, 2, 4, 8}:
                    raise ValueError(f"unsupported spectral bit width: {bits}")
                code_offsets[
                    batch_index, head_index, band_index
                ] = code_stride
                if bits:
                    scale_offsets[
                        batch_index, head_index, band_index
                    ] = scale_stride
                    code_stride += 2 * bits
                    scale_stride += 1
            code_strides[batch_index, head_index] = code_stride
            scale_strides[batch_index, head_index] = scale_stride
            code_cursor += capacity * code_stride
            scale_cursor += capacity * scale_stride
    return {
        "bit_allocations": allocations_cpu.to(device=device),
        "bit_allocations_host": allocations_cpu,
        "code_offsets": code_offsets.to(device=device),
        "scale_offsets": scale_offsets.to(device=device),
        "code_bases": code_bases.to(device=device),
        "scale_bases": scale_bases.to(device=device),
        "code_strides": code_strides.to(device=device),
        "scale_strides": scale_strides.to(device=device),
        "capacity": int(capacity),
        "total_code_bytes": int(code_cursor),
        "total_scale_values": int(scale_cursor),
    }


def quantize_projected_query(
    projected_query: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize each 16D query band to INT8 with an independent scale."""
    if projected_query.shape[-1] != 128:
        raise ValueError("projected queries must have head dimension 128")
    if projected_query.is_cuda:
        return tuple(
            load_extension().varbit_spectral_quantize_query_forward(
                projected_query.contiguous()
            )
        )
    grouped = projected_query.float().reshape(
        *projected_query.shape[:-1], 8, 16
    )
    scales = grouped.abs().amax(dim=-1).clamp_min(1.0e-8) / 127.0
    codes = torch.round(grouped / scales.unsqueeze(-1)).clamp(-127, 127)
    return (
        codes.to(torch.int8).reshape_as(projected_query).contiguous(),
        scales.to(projected_query.dtype).contiguous(),
    )


def _pack_band_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    unsigned = (codes.to(torch.int16) & ((1 << bits) - 1)).to(torch.uint8)
    if bits == 8:
        return unsigned
    if bits == 4:
        grouped = unsigned.reshape(*unsigned.shape[:-1], 8, 2)
        return grouped[..., 0] | (grouped[..., 1] << 4)
    if bits == 2:
        grouped = unsigned.reshape(*unsigned.shape[:-1], 4, 4)
        return (
            grouped[..., 0]
            | (grouped[..., 1] << 2)
            | (grouped[..., 2] << 4)
            | (grouped[..., 3] << 6)
        )
    if bits == 1:
        positive = (codes > 0).to(torch.uint8).reshape(
            *codes.shape[:-1], 2, 8
        )
        weights = (
            1 << torch.arange(8, dtype=torch.int64, device=codes.device)
        ).to(torch.uint8)
        return (positive * weights).sum(dim=-1).to(torch.uint8)
    raise ValueError(f"unsupported spectral bit width: {bits}")


def allocate_packed_index(
    bit_allocations: torch.Tensor,
    capacity: int,
    scale_dtype: torch.dtype,
    enable_score_bias: bool = False,
) -> dict[str, Any]:
    metadata = make_packed_metadata(bit_allocations, capacity)
    device = bit_allocations.device
    metadata["packed_codes"] = torch.empty(
        metadata["total_code_bytes"],
        dtype=torch.uint8,
        device=device,
    )
    metadata["key_scales"] = torch.empty(
        metadata["total_scale_values"],
        dtype=scale_dtype,
        device=device,
    )
    metadata["score_bias"] = (
        torch.empty(
            *bit_allocations.shape[:2],
            capacity,
            dtype=scale_dtype,
            device=device,
        )
        if enable_score_bias
        else torch.empty(0, dtype=scale_dtype, device=device)
    )
    metadata["indexed_count"] = 0
    return metadata


def metric_optimal_projected_key_scales(
    projected_keys: torch.Tensor,
    bit_allocations: torch.Tensor,
    scale_metrics: torch.Tensor,
) -> torch.Tensor:
    """Compute closed-form QK-metric scales for fixed uniform codes."""
    if projected_keys.ndim != 4 or projected_keys.shape[-1] != 128:
        raise ValueError(
            "projected keys must have shape [batch, kv_heads, tokens, 128]"
        )
    expected_allocation_shape = projected_keys.shape[:2] + (8,)
    if tuple(bit_allocations.shape) != expected_allocation_shape:
        raise ValueError("bit allocations must have shape [batch, kv_heads, 8]")
    expected_metric_shape = projected_keys.shape[:2] + (8, 16, 16)
    if tuple(scale_metrics.shape) != expected_metric_shape:
        raise ValueError(
            "scale metrics must have shape [batch, kv_heads, 8, 16, 16]"
        )
    working = projected_keys.float().reshape(
        *projected_keys.shape[:-1],
        8,
        16,
    )
    allocations = bit_allocations.to(
        device=projected_keys.device,
        dtype=torch.int8,
    )
    codes = torch.zeros_like(working)
    for bits in (1, 2, 4, 8):
        if bits == 1:
            candidate = torch.where(working >= 0.0, 1.0, -1.0)
        else:
            maximum_code = (1 << (bits - 1)) - 1
            initial_scale = (
                working.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8)
                / float(maximum_code)
            )
            candidate = torch.round(working / initial_scale).clamp(
                -maximum_code,
                maximum_code,
            )
        mask = (allocations == bits).unsqueeze(2).unsqueeze(-1)
        codes = torch.where(mask, candidate, codes)
    weighted_codes = torch.einsum(
        "bhkgd,bhgde->bhkge",
        codes,
        scale_metrics.to(
            device=projected_keys.device,
            dtype=torch.float32,
        ),
    )
    numerator = (weighted_codes * working).sum(dim=-1)
    denominator = (
        (weighted_codes * codes).sum(dim=-1).clamp_min(1.0e-12)
    )
    scales = (numerator / denominator).clamp_min(0.0)
    scales = torch.where(
        allocations.unsqueeze(2) > 0,
        scales,
        torch.zeros_like(scales),
    )
    return scales.to(projected_keys.dtype).contiguous()


def encode_projected_keys_into(
    projected_keys: torch.Tensor,
    packed_index: dict[str, Any],
    start: int,
    scale_metrics: torch.Tensor | None = None,
    precomputed_scales: torch.Tensor | None = None,
    exact_query_mean: torch.Tensor | None = None,
    proxy_query_mean: torch.Tensor | None = None,
    bias_shrinkage: float = 0.5,
) -> None:
    """Quantize and append projected K rows to a preallocated packed index."""
    if projected_keys.ndim != 4 or projected_keys.shape[-1] != 128:
        raise ValueError(
            "projected keys must have shape [batch, kv_heads, tokens, 128]"
        )
    token_count = int(projected_keys.shape[-2])
    stop = start + token_count
    capacity = int(packed_index["capacity"])
    if start < 0 or stop > capacity:
        raise ValueError("projected key write exceeds packed index capacity")
    if scale_metrics is not None:
        expected_metric_shape = (
            projected_keys.shape[0],
            projected_keys.shape[1],
            8,
            16,
            16,
        )
        if tuple(scale_metrics.shape) != expected_metric_shape:
            raise ValueError(
                "scale metrics must have shape "
                "[batch, kv_heads, 8, 16, 16]"
            )
    if precomputed_scales is not None:
        expected_scale_shape = projected_keys.shape[:3] + (8,)
        if tuple(precomputed_scales.shape) != expected_scale_shape:
            raise ValueError(
                "precomputed scales must have shape "
                "[batch, kv_heads, tokens, 8]"
            )
    score_bias = packed_index.get("score_bias")
    use_score_bias = (
        isinstance(score_bias, torch.Tensor) and score_bias.numel() > 0
    )
    if use_score_bias:
        expected_mean_shape = projected_keys.shape[:2] + (128,)
        if (
            exact_query_mean is None
            or proxy_query_mean is None
            or tuple(exact_query_mean.shape) != expected_mean_shape
            or tuple(proxy_query_mean.shape) != expected_mean_shape
        ):
            raise ValueError(
                "score-biased index requires exact/proxy query means with "
                "shape [batch, kv_heads, 128]"
            )
        if not 0.0 <= bias_shrinkage <= 1.0:
            raise ValueError("score-bias shrinkage must be in [0, 1]")
    if projected_keys.is_cuda:
        if scale_metrics is None:
            scale_metrics = torch.empty(
                0,
                dtype=torch.float32,
                device=projected_keys.device,
            )
        if precomputed_scales is None:
            precomputed_scales = torch.empty(
                0,
                dtype=projected_keys.dtype,
                device=projected_keys.device,
            )
        if exact_query_mean is None:
            exact_query_mean = torch.empty(
                0,
                dtype=torch.float32,
                device=projected_keys.device,
            )
        if proxy_query_mean is None:
            proxy_query_mean = torch.empty(
                0,
                dtype=torch.float32,
                device=projected_keys.device,
            )
        load_extension().varbit_spectral_encode_projected_out_forward(
            projected_keys.contiguous(),
            scale_metrics.to(
                device=projected_keys.device,
                dtype=torch.float32,
            ).contiguous(),
            precomputed_scales.to(
                device=projected_keys.device,
                dtype=projected_keys.dtype,
            ).contiguous(),
            exact_query_mean.to(
                device=projected_keys.device,
                dtype=torch.float32,
            ).contiguous(),
            proxy_query_mean.to(
                device=projected_keys.device,
                dtype=torch.float32,
            ).contiguous(),
            packed_index["packed_codes"],
            packed_index["key_scales"],
            packed_index["score_bias"],
            packed_index["bit_allocations"],
            packed_index["code_offsets"],
            packed_index["scale_offsets"],
            packed_index["code_bases"],
            packed_index["scale_bases"],
            packed_index["code_strides"],
            packed_index["scale_strides"],
            start,
            bool(scale_metrics.numel()) and not bool(
                precomputed_scales.numel()
            ),
            bool(precomputed_scales.numel()),
            float(bias_shrinkage),
        )
        packed_index["indexed_count"] = stop
        return
    allocations_cpu = packed_index["bit_allocations_host"]
    batch_count, kv_head_count, _, _ = projected_keys.shape
    if allocations_cpu.shape[:2] != (batch_count, kv_head_count):
        raise ValueError("packed index head layout does not match projected keys")
    for batch_index in range(batch_count):
        for head_index in range(kv_head_count):
            bias_values = (
                torch.zeros(token_count, dtype=torch.float32)
                if use_score_bias
                else None
            )
            code_stride = int(
                packed_index["code_strides"][
                    batch_index, head_index
                ].item()
            )
            scale_stride = int(
                packed_index["scale_strides"][
                    batch_index, head_index
                ].item()
            )
            code_base = int(
                packed_index["code_bases"][
                    batch_index, head_index
                ].item()
            )
            scale_base = int(
                packed_index["scale_bases"][
                    batch_index, head_index
                ].item()
            )
            head_codes = packed_index["packed_codes"][
                code_base : code_base + capacity * code_stride
            ].reshape(capacity, code_stride)
            head_scales = packed_index["key_scales"][
                scale_base : scale_base + capacity * scale_stride
            ].reshape(capacity, scale_stride)
            for band_index in range(8):
                bits = int(
                    allocations_cpu[
                        batch_index, head_index, band_index
                    ].item()
                )
                band = projected_keys[
                    batch_index,
                    head_index,
                    :,
                    16 * band_index : 16 * (band_index + 1),
                ].float()
                if bias_values is not None:
                    exact_mean_band = exact_query_mean[
                        batch_index,
                        head_index,
                        16 * band_index : 16 * (band_index + 1),
                    ].float()
                    bias_values.add_(band @ exact_mean_band)
                if bits == 0:
                    continue
                if bits == 1:
                    scales = band.abs().mean(dim=-1).clamp_min(1.0e-8)
                    codes = torch.where(
                        band >= 0.0,
                        torch.ones_like(band),
                        -torch.ones_like(band),
                    ).to(torch.int8)
                else:
                    maximum_code = (1 << (bits - 1)) - 1
                    scales = (
                        band.abs().amax(dim=-1).clamp_min(1.0e-8)
                        / float(maximum_code)
                    )
                    codes = torch.round(
                        band / scales.unsqueeze(-1)
                    ).clamp(-maximum_code, maximum_code).to(torch.int8)
                if precomputed_scales is not None:
                    scales = precomputed_scales[
                        batch_index,
                        head_index,
                        :,
                        band_index,
                    ].float()
                elif scale_metrics is not None:
                    metric = scale_metrics[
                        batch_index,
                        head_index,
                        band_index,
                    ].float()
                    codes_float = codes.float()
                    weighted_codes = codes_float @ metric
                    numerator = (weighted_codes * band).sum(
                        dim=-1
                    )
                    denominator = (
                        (weighted_codes * codes_float)
                        .sum(dim=-1)
                        .clamp_min(1.0e-12)
                    )
                    scales = (
                        numerator / denominator
                    ).clamp_min(0.0)
                packed_band = _pack_band_codes(codes, bits)
                code_offset = int(
                    packed_index["code_offsets"][
                        batch_index, head_index, band_index
                    ].item()
                )
                scale_offset = int(
                    packed_index["scale_offsets"][
                        batch_index, head_index, band_index
                    ].item()
                )
                head_codes[
                    start:stop,
                    code_offset : code_offset + 2 * bits,
                ].copy_(packed_band)
                head_scales[start:stop, scale_offset].copy_(
                    scales.to(head_scales.dtype)
                )
                if bias_values is not None:
                    proxy_mean_band = proxy_query_mean[
                        batch_index,
                        head_index,
                        16 * band_index : 16 * (band_index + 1),
                    ].float()
                    bias_values.sub_(
                        (codes.float() * scales.unsqueeze(-1))
                        @ proxy_mean_band
                    )
            if bias_values is not None:
                packed_index["score_bias"][
                    batch_index,
                    head_index,
                    start:stop,
                ].copy_(
                    (float(bias_shrinkage) * bias_values).to(
                        packed_index["score_bias"].dtype
                    )
                )
    packed_index["indexed_count"] = stop


def scores(
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    packed_codes: torch.Tensor,
    key_scales: torch.Tensor,
    bit_allocations: torch.Tensor,
    code_offsets: torch.Tensor,
    scale_offsets: torch.Tensor,
    code_bases: torch.Tensor,
    scale_bases: torch.Tensor,
    code_strides: torch.Tensor,
    scale_strides: torch.Tensor,
    history_count: int,
    score_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if score_bias is None:
        score_bias = torch.empty(
            0,
            dtype=key_scales.dtype,
            device=key_scales.device,
        )
    return load_extension().varbit_spectral_scores_forward(
        query_codes,
        query_scales,
        packed_codes,
        key_scales,
        score_bias,
        bit_allocations,
        code_offsets,
        scale_offsets,
        code_bases,
        scale_bases,
        code_strides,
        scale_strides,
        history_count,
    )


def sampled_threshold_compact(
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    packed_codes: torch.Tensor,
    key_scales: torch.Tensor,
    bit_allocations: torch.Tensor,
    code_offsets: torch.Tensor,
    scale_offsets: torch.Tensor,
    code_bases: torch.Tensor,
    scale_bases: torch.Tensor,
    code_strides: torch.Tensor,
    scale_strides: torch.Tensor,
    history_count: int,
    sample_count: int,
    selected_fraction: float,
    candidate_capacity: int,
    score_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    if score_bias is None:
        score_bias = torch.empty(
            0,
            dtype=key_scales.dtype,
            device=key_scales.device,
        )
    return load_extension().varbit_spectral_sampled_compact_forward(
        query_codes,
        query_scales,
        packed_codes,
        key_scales,
        score_bias,
        bit_allocations,
        code_offsets,
        scale_offsets,
        code_bases,
        scale_bases,
        code_strides,
        scale_strides,
        history_count,
        sample_count,
        selected_fraction,
        candidate_capacity,
    )


def sampled_threshold_compact_out(
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    packed_codes: torch.Tensor,
    key_scales: torch.Tensor,
    bit_allocations: torch.Tensor,
    code_offsets: torch.Tensor,
    scale_offsets: torch.Tensor,
    code_bases: torch.Tensor,
    scale_bases: torch.Tensor,
    code_strides: torch.Tensor,
    scale_strides: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_scores: torch.Tensor,
    candidate_counts: torch.Tensor,
    thresholds: torch.Tensor,
    overflow: torch.Tensor,
    history_count: int,
    sample_count: int,
    selected_fraction: float,
    score_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    if score_bias is None:
        score_bias = torch.empty(
            0,
            dtype=key_scales.dtype,
            device=key_scales.device,
        )
    load_extension().varbit_spectral_sampled_compact_out_forward(
        query_codes,
        query_scales,
        packed_codes,
        key_scales,
        score_bias,
        bit_allocations,
        code_offsets,
        scale_offsets,
        code_bases,
        scale_bases,
        code_strides,
        scale_strides,
        candidate_indices,
        candidate_scores,
        candidate_counts,
        thresholds,
        overflow,
        history_count,
        sample_count,
        selected_fraction,
    )
    return (
        candidate_indices,
        candidate_scores,
        candidate_counts,
        thresholds,
        overflow,
    )
