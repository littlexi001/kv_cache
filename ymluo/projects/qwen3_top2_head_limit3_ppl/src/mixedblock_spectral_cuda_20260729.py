from __future__ import annotations

from functools import lru_cache

import torch
from torch.utils.cpp_extension import load_inline


CPP_SOURCE = r"""
#include <torch/extension.h>

void mixedblock_sampled_compact_out_cuda(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    torch::Tensor block_hot_prefix,
    torch::Tensor head_code_bases,
    torch::Tensor head_scale_bases,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_scores,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor overflow,
    int64_t history_count,
    int64_t block_size,
    int64_t sample_count,
    double selected_fraction);

void mixedblock_sampled_compact_gqa4_indices_out_cuda(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    torch::Tensor block_hot_prefix,
    torch::Tensor head_code_bases,
    torch::Tensor head_scale_bases,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor overflow,
    int64_t history_count,
    int64_t block_size,
    int64_t sample_count,
    double selected_fraction);

void plain_sampled_compact_gqa4_indices_out_cuda(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor overflow,
    int64_t history_count,
    int64_t sample_count,
    double selected_fraction);

void plain_sampled_compact_gqa4_mass_out_cuda(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor overflow,
    torch::Tensor selected_denominator,
    torch::Tensor tail_denominator,
    int64_t history_count,
    int64_t sample_count,
    double selected_fraction,
    double attention_scale);

void plain_mass_ladder_thresholds_out_cuda(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    torch::Tensor ladder_thresholds,
    torch::Tensor mass_bins,
    torch::Tensor chosen_thresholds,
    torch::Tensor chosen_rungs,
    torch::Tensor chosen_mass,
    int64_t history_count,
    int64_t sample_count,
    double minimum_fraction,
    double growth,
    int64_t rung_count,
    double target_mass,
    double attention_scale);

void plain_sampled_compact_gqa4_condtail_out_cuda(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor overflow,
    torch::Tensor selected_denominator,
    torch::Tensor tail_denominator,
    torch::Tensor tail_block_denominator,
    torch::Tensor tail_weighted_x,
    int64_t history_count,
    int64_t sample_count,
    double selected_fraction,
    int64_t moment_block_size,
    double attention_scale);

void plain_sampled_compact_gqa4_valuesketch_out_cuda(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    torch::Tensor packed_value_codes,
    torch::Tensor value_minimum,
    torch::Tensor value_scale,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor overflow,
    torch::Tensor selected_denominator,
    torch::Tensor tail_denominator,
    torch::Tensor tail_coefficients,
    int64_t history_count,
    int64_t sample_count,
    double selected_fraction,
    int64_t value_rank,
    int64_t value_block_size,
    double attention_scale);

void plain_sampled_compact_gqa4_valuesketch_deterministic_out_cuda(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    torch::Tensor packed_value_codes,
    torch::Tensor value_minimum,
    torch::Tensor value_scale,
    torch::Tensor selection_masks,
    torch::Tensor tail_partials,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor overflow,
    torch::Tensor selected_denominator,
    torch::Tensor tail_denominator,
    torch::Tensor tail_coefficients,
    int64_t history_count,
    int64_t sample_count,
    double selected_fraction,
    int64_t value_rank,
    int64_t value_block_size,
    double attention_scale);

void plain_sampled_compact_gqa4_valuesketch_progressive_out_cuda(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    torch::Tensor packed_value_codes,
    torch::Tensor value_minimum,
    torch::Tensor value_scale,
    torch::Tensor value_rank8_residual,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor overflow,
    torch::Tensor selected_denominator,
    torch::Tensor tail_denominator,
    torch::Tensor tail_coefficients,
    torch::Tensor refinement_flags,
    int64_t history_count,
    int64_t sample_count,
    double selected_fraction,
    int64_t value_block_size,
    double attention_scale,
    double refinement_tolerance);

void mixedblock_sampled_fused_attention_gqa4_out_cuda(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    torch::Tensor block_hot_prefix,
    torch::Tensor head_code_bases,
    torch::Tensor head_scale_bases,
    torch::Tensor output,
    torch::Tensor partial_output,
    torch::Tensor partial_max,
    torch::Tensor partial_sum,
    torch::Tensor partial_counts,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor overflow,
    int64_t history_count,
    int64_t block_size,
    int64_t sample_count,
    double selected_fraction,
    double scaling,
    int64_t split_count,
    int64_t max_local_candidates);

void sortedblock_sampled_compact_out_cuda(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    torch::Tensor head_code_bases,
    torch::Tensor head_scale_bases,
    torch::Tensor original_blocks,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_scores,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor overflow,
    int64_t history_count,
    int64_t block_size,
    int64_t hot_block_count,
    int64_t sample_count,
    double selected_fraction);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "sampled_compact_out",
      &mixedblock_sampled_compact_out_cuda,
      "Mixed-block variable-bit sampled threshold and compaction");
  m.def(
      "sampled_compact_gqa4_indices_out",
      &mixedblock_sampled_compact_gqa4_indices_out_cuda,
      "Mixed-block GQA4 sampled threshold and index-only compaction");
  m.def(
      "plain_sampled_compact_gqa4_indices_out",
      &plain_sampled_compact_gqa4_indices_out_cuda,
      "Plain-layout GQA4 sampled threshold and index-only compaction");
  m.def(
      "plain_sampled_compact_gqa4_mass_out",
      &plain_sampled_compact_gqa4_mass_out_cuda,
      "Plain-layout GQA4 compaction with exact proxy partition sums");
  m.def(
      "plain_mass_ladder_thresholds_out",
      &plain_mass_ladder_thresholds_out_cuda,
      "Plain-layout GQA4 measured proxy-mass ladder thresholds");
  m.def(
      "plain_sampled_compact_gqa4_condtail_out",
      &plain_sampled_compact_gqa4_condtail_out_cuda,
      "Plain-layout GQA4 compaction with shared-map conditional tail moments");
  m.def(
      "plain_sampled_compact_gqa4_valuesketch_out",
      &plain_sampled_compact_gqa4_valuesketch_out_cuda,
      "Plain-layout GQA4 compaction with fused INT4 Value-sketch tail scan");
  m.def(
      "plain_sampled_compact_gqa4_valuesketch_deterministic_out",
      &plain_sampled_compact_gqa4_valuesketch_deterministic_out_cuda,
      "Deterministic GQA4 Value-sketch compaction and tail reduction");
  m.def(
      "plain_sampled_compact_gqa4_valuesketch_progressive_out",
      &plain_sampled_compact_gqa4_valuesketch_progressive_out_cuda,
      "Plain-layout GQA4 rank-8 scan with certified rank-32 refinement");
  m.def(
      "sampled_fused_attention_gqa4_out",
      &mixedblock_sampled_fused_attention_gqa4_out_cuda,
      "Mixed-block GQA4 sampled threshold and fused exact attention");
  m.def(
      "sortedblock_sampled_compact_out",
      &sortedblock_sampled_compact_out_cuda,
      "Frequency-sorted block variable-bit sampled threshold and compaction");
}
"""


CUDA_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <type_traits>

#define MIXEDBLOCK_MAX_SAMPLE_COUNT 8192

inline int mixedblock_sample_shared_bytes(int sample_count) {
  int sort_count = 1;
  while (sort_count < sample_count) {
    sort_count <<= 1;
  }
  return sort_count * static_cast<int>(sizeof(float));
}

__device__ __forceinline__ int pack_signed_bytes(
    int a, int b, int c, int d) {
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

__device__ __forceinline__ void band_dot_gqa4(
    const uint8_t* __restrict__ packed,
    const int8_t* __restrict__ query_base,
    int query_band_offset,
    int bits,
    int* __restrict__ dots) {
  if (bits == 8) {
#pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
      const uint8_t* code = packed + chunk * 4;
      int word = pack_signed_bytes(
          static_cast<int8_t>(code[0]),
          static_cast<int8_t>(code[1]),
          static_cast<int8_t>(code[2]),
          static_cast<int8_t>(code[3]));
#pragma unroll
      for (int group = 0; group < 4; ++group) {
        const int8_t* query =
            query_base + group * 128 + query_band_offset;
        dots[group] = __dp4a(
            word,
            reinterpret_cast<const int*>(query)[chunk],
            dots[group]);
      }
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
#pragma unroll
      for (int group = 0; group < 4; ++group) {
        const int8_t* query =
            query_base + group * 128 + query_band_offset;
        dots[group] = __dp4a(
            word,
            reinterpret_cast<const int*>(query)[chunk],
            dots[group]);
      }
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
#pragma unroll
      for (int group = 0; group < 4; ++group) {
        const int8_t* query =
            query_base + group * 128 + query_band_offset;
        dots[group] = __dp4a(
            word,
            reinterpret_cast<const int*>(query)[chunk],
            dots[group]);
      }
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
#pragma unroll
      for (int group = 0; group < 4; ++group) {
        const int8_t* query =
            query_base + group * 128 + query_band_offset;
        dots[group] = __dp4a(
            word,
            reinterpret_cast<const int*>(query)[chunk],
            dots[group]);
      }
    }
  }
}

template <typename scale_t>
__device__ __forceinline__ float mixedblock_score_one(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    const int16_t* __restrict__ block_hot_prefix,
    const int64_t* __restrict__ head_code_bases,
    const int64_t* __restrict__ head_scale_bases,
    int batch_kv,
    int query_group,
    int query_groups,
    int token,
    int block_size,
    int block_count,
    int batch_kv_count) {
  int block = token / block_size;
  int local_token = token - block * block_size;
  int prefix_offset =
      batch_kv * (block_count + 1) + block;
  int hot_before =
      static_cast<int>(block_hot_prefix[prefix_offset]);
  int profile =
      block_hot_prefix[prefix_offset + 1]
          != block_hot_prefix[prefix_offset]
      ? 1
      : 0;
  int profile_head = profile * batch_kv_count + batch_kv;
  int64_t query_row =
      static_cast<int64_t>(batch_kv) * query_groups + query_group;
  const int8_t* query = query_codes + query_row * 128;
  const scale_t* qscale = query_scales + query_row * 8;
  const int8_t* allocations = bit_allocations + profile_head * 8;
  const int16_t* offsets = code_offsets + profile_head * 8;
  const int8_t* scale_index = scale_offsets + profile_head * 8;
  int low_code_stride = code_strides[batch_kv];
  int high_code_stride = code_strides[batch_kv_count + batch_kv];
  int low_scale_stride = scale_strides[batch_kv];
  int high_scale_stride = scale_strides[batch_kv_count + batch_kv];
  int64_t code_position = static_cast<int64_t>(block_size)
      * (static_cast<int64_t>(block) * low_code_stride
         + static_cast<int64_t>(hot_before)
             * (high_code_stride - low_code_stride))
      + static_cast<int64_t>(local_token)
          * code_strides[profile_head];
  int64_t scale_position = static_cast<int64_t>(block_size)
      * (static_cast<int64_t>(block) * low_scale_stride
         + static_cast<int64_t>(hot_before)
             * (high_scale_stride - low_scale_stride))
      + static_cast<int64_t>(local_token)
          * scale_strides[profile_head];
  const uint8_t* token_codes =
      packed_codes + head_code_bases[batch_kv] + code_position;
  const scale_t* token_scales =
      key_scales + head_scale_bases[batch_kv] + scale_position;
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
        static_cast<float>(dot0) * static_cast<float>(qscale[0])
            * static_cast<float>(token_scales[0])
        + static_cast<float>(dot1) * static_cast<float>(qscale[1])
            * static_cast<float>(token_scales[1]);
  } else if (
      zero_tail && allocations[0] == 8 && allocations[1] == 1
      && allocations[2] == 1 && allocations[3] == 1) {
    int dot0 = band_dot(token_codes, query, 8);
    int dot1 = band_dot(token_codes + 16, query + 16, 1);
    int dot2 = band_dot(token_codes + 18, query + 32, 1);
    int dot3 = band_dot(token_codes + 20, query + 48, 1);
    score =
        static_cast<float>(dot0) * static_cast<float>(qscale[0])
            * static_cast<float>(token_scales[0])
        + static_cast<float>(dot1) * static_cast<float>(qscale[1])
            * static_cast<float>(token_scales[1])
        + static_cast<float>(dot2) * static_cast<float>(qscale[2])
            * static_cast<float>(token_scales[2])
        + static_cast<float>(dot3) * static_cast<float>(qscale[3])
            * static_cast<float>(token_scales[3]);
  } else if (
      zero_tail && allocations[0] == 4 && allocations[1] == 4
      && allocations[2] == 4 && allocations[3] == 0) {
    int dot0 = band_dot(token_codes, query, 4);
    int dot1 = band_dot(token_codes + 8, query + 16, 4);
    int dot2 = band_dot(token_codes + 16, query + 32, 4);
    score =
        static_cast<float>(dot0) * static_cast<float>(qscale[0])
            * static_cast<float>(token_scales[0])
        + static_cast<float>(dot1) * static_cast<float>(qscale[1])
            * static_cast<float>(token_scales[1])
        + static_cast<float>(dot2) * static_cast<float>(qscale[2])
            * static_cast<float>(token_scales[2]);
  } else if (
      zero_tail && allocations[0] == 4 && allocations[1] == 4
      && allocations[2] == 1 && allocations[3] == 0) {
    int dot0 = band_dot(token_codes, query, 4);
    int dot1 = band_dot(token_codes + 8, query + 16, 4);
    int dot2 = band_dot(token_codes + 16, query + 32, 1);
    score =
        static_cast<float>(dot0) * static_cast<float>(qscale[0])
            * static_cast<float>(token_scales[0])
        + static_cast<float>(dot1) * static_cast<float>(qscale[1])
            * static_cast<float>(token_scales[1])
        + static_cast<float>(dot2) * static_cast<float>(qscale[2])
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
  return score;
}

template <typename scale_t>
__device__ __forceinline__ void mixedblock_score_gqa4_one(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    const int16_t* __restrict__ block_hot_prefix,
    const int64_t* __restrict__ head_code_bases,
    const int64_t* __restrict__ head_scale_bases,
    int batch_kv,
    int token,
    int block_size,
    int block_count,
    int batch_kv_count,
    float scores[4]) {
  constexpr int query_groups = 4;
  int block = token / block_size;
  int local_token = token - block * block_size;
  int prefix_offset =
      batch_kv * (block_count + 1) + block;
  int hot_before =
      static_cast<int>(block_hot_prefix[prefix_offset]);
  int profile =
      block_hot_prefix[prefix_offset + 1]
          != block_hot_prefix[prefix_offset]
      ? 1
      : 0;
  int profile_head = profile * batch_kv_count + batch_kv;
  const int8_t* allocations = bit_allocations + profile_head * 8;
  const int16_t* offsets = code_offsets + profile_head * 8;
  const int8_t* scale_index = scale_offsets + profile_head * 8;
  int low_code_stride = code_strides[batch_kv];
  int high_code_stride = code_strides[batch_kv_count + batch_kv];
  int low_scale_stride = scale_strides[batch_kv];
  int high_scale_stride = scale_strides[batch_kv_count + batch_kv];
  int64_t code_position = static_cast<int64_t>(block_size)
      * (static_cast<int64_t>(block) * low_code_stride
         + static_cast<int64_t>(hot_before)
             * (high_code_stride - low_code_stride))
      + static_cast<int64_t>(local_token)
          * code_strides[profile_head];
  int64_t scale_position = static_cast<int64_t>(block_size)
      * (static_cast<int64_t>(block) * low_scale_stride
         + static_cast<int64_t>(hot_before)
             * (high_scale_stride - low_scale_stride))
      + static_cast<int64_t>(local_token)
          * scale_strides[profile_head];
  const uint8_t* token_codes =
      packed_codes + head_code_bases[batch_kv] + code_position;
  const scale_t* token_scales =
      key_scales + head_scale_bases[batch_kv] + scale_position;
  const int8_t* query_base =
      query_codes + static_cast<int64_t>(batch_kv) * query_groups * 128;
  const scale_t* qscale_base =
      query_scales + static_cast<int64_t>(batch_kv) * query_groups * 8;

#pragma unroll
  for (int group = 0; group < query_groups; ++group) {
    scores[group] = 0.0f;
  }
  bool zero_tail =
      allocations[4] == 0 && allocations[5] == 0
      && allocations[6] == 0 && allocations[7] == 0;
  if (
      zero_tail && allocations[0] == 8 && allocations[1] == 4
      && allocations[2] == 0 && allocations[3] == 0) {
    int dot0[query_groups] = {0, 0, 0, 0};
    int dot1[query_groups] = {0, 0, 0, 0};
    band_dot_gqa4(token_codes, query_base, 0, 8, dot0);
    band_dot_gqa4(token_codes + 16, query_base, 16, 4, dot1);
    float key_scale0 = static_cast<float>(token_scales[0]);
    float key_scale1 = static_cast<float>(token_scales[1]);
#pragma unroll
    for (int group = 0; group < query_groups; ++group) {
      const scale_t* qscale = qscale_base + group * 8;
      scores[group] =
          static_cast<float>(dot0[group])
              * static_cast<float>(qscale[0]) * key_scale0
          + static_cast<float>(dot1[group])
              * static_cast<float>(qscale[1]) * key_scale1;
    }
  } else {
#pragma unroll
    for (int band = 0; band < 8; ++band) {
      int bits = static_cast<int>(allocations[band]);
      if (bits == 0) {
        continue;
      }
      int dots[query_groups] = {0, 0, 0, 0};
      band_dot_gqa4(
          token_codes + offsets[band],
          query_base,
          16 * band,
          bits,
          dots);
      float key_scale =
          static_cast<float>(token_scales[scale_index[band]]);
#pragma unroll
      for (int group = 0; group < query_groups; ++group) {
        const scale_t* qscale = qscale_base + group * 8;
        scores[group] +=
            static_cast<float>(dots[group])
            * static_cast<float>(qscale[band])
            * key_scale;
      }
    }
  }
}

template <typename scale_t>
__device__ __forceinline__ float plain_score_one(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
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
    int token) {
  int64_t query_row =
      static_cast<int64_t>(batch_kv) * query_groups + query_group;
  const int8_t* query = query_codes + query_row * 128;
  const scale_t* qscale = query_scales + query_row * 8;
  const int8_t* allocations = bit_allocations + batch_kv * 8;
  const int16_t* offsets = code_offsets + batch_kv * 8;
  const int8_t* scale_index = scale_offsets + batch_kv * 8;
  const uint8_t* token_codes =
      packed_codes + code_bases[batch_kv]
      + static_cast<int64_t>(token) * code_strides[batch_kv];
  const scale_t* token_scales =
      key_scales + scale_bases[batch_kv]
      + static_cast<int64_t>(token) * scale_strides[batch_kv];
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
        static_cast<float>(dot0) * static_cast<float>(qscale[0])
            * static_cast<float>(token_scales[0])
        + static_cast<float>(dot1) * static_cast<float>(qscale[1])
            * static_cast<float>(token_scales[1]);
  } else if (
      zero_tail && allocations[0] == 8 && allocations[1] == 1
      && allocations[2] == 1 && allocations[3] == 1) {
    int dot0 = band_dot(token_codes, query, 8);
    int dot1 = band_dot(token_codes + 16, query + 16, 1);
    int dot2 = band_dot(token_codes + 18, query + 32, 1);
    int dot3 = band_dot(token_codes + 20, query + 48, 1);
    score =
        static_cast<float>(dot0) * static_cast<float>(qscale[0])
            * static_cast<float>(token_scales[0])
        + static_cast<float>(dot1) * static_cast<float>(qscale[1])
            * static_cast<float>(token_scales[1])
        + static_cast<float>(dot2) * static_cast<float>(qscale[2])
            * static_cast<float>(token_scales[2])
        + static_cast<float>(dot3) * static_cast<float>(qscale[3])
            * static_cast<float>(token_scales[3]);
  } else if (
      zero_tail && allocations[0] == 4 && allocations[1] == 4
      && allocations[2] == 4 && allocations[3] == 0) {
    int dot0 = band_dot(token_codes, query, 4);
    int dot1 = band_dot(token_codes + 8, query + 16, 4);
    int dot2 = band_dot(token_codes + 16, query + 32, 4);
    score =
        static_cast<float>(dot0) * static_cast<float>(qscale[0])
            * static_cast<float>(token_scales[0])
        + static_cast<float>(dot1) * static_cast<float>(qscale[1])
            * static_cast<float>(token_scales[1])
        + static_cast<float>(dot2) * static_cast<float>(qscale[2])
            * static_cast<float>(token_scales[2]);
  } else if (
      zero_tail && allocations[0] == 4 && allocations[1] == 4
      && allocations[2] == 1 && allocations[3] == 0) {
    int dot0 = band_dot(token_codes, query, 4);
    int dot1 = band_dot(token_codes + 8, query + 16, 4);
    int dot2 = band_dot(token_codes + 16, query + 32, 1);
    score =
        static_cast<float>(dot0) * static_cast<float>(qscale[0])
            * static_cast<float>(token_scales[0])
        + static_cast<float>(dot1) * static_cast<float>(qscale[1])
            * static_cast<float>(token_scales[1])
        + static_cast<float>(dot2) * static_cast<float>(qscale[2])
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
  return score;
}

template <int bits, typename scale_t>
__device__ __forceinline__ void plain_accumulate_gqa4_band(
    const uint8_t* __restrict__ token_codes,
    const int8_t* __restrict__ query_base,
    const scale_t* __restrict__ qscale_base,
    const scale_t* __restrict__ token_scales,
    const int16_t* __restrict__ offsets,
    const int8_t* __restrict__ scale_index,
    int band,
    float scores[4]) {
  constexpr int query_groups = 4;
  int dots[query_groups] = {0, 0, 0, 0};
  band_dot_gqa4(
      token_codes + offsets[band],
      query_base,
      16 * band,
      bits,
      dots);
  float key_scale =
      static_cast<float>(token_scales[scale_index[band]]);
#pragma unroll
  for (int group = 0; group < query_groups; ++group) {
    const scale_t* qscale = qscale_base + group * 8;
    scores[group] +=
        static_cast<float>(dots[group])
        * static_cast<float>(qscale[band])
        * key_scale;
  }
}

template <typename scale_t>
__device__ __forceinline__ void plain_score_gqa4_one(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int64_t* __restrict__ code_bases,
    const int64_t* __restrict__ scale_bases,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    int batch_kv,
    int token,
    float scores[4]) {
  constexpr int query_groups = 4;
  const int8_t* allocations = bit_allocations + batch_kv * 8;
  const int16_t* offsets = code_offsets + batch_kv * 8;
  const int8_t* scale_index = scale_offsets + batch_kv * 8;
  const uint8_t* token_codes =
      packed_codes + code_bases[batch_kv]
      + static_cast<int64_t>(token) * code_strides[batch_kv];
  const scale_t* token_scales =
      key_scales + scale_bases[batch_kv]
      + static_cast<int64_t>(token) * scale_strides[batch_kv];
  const int8_t* query_base =
      query_codes + static_cast<int64_t>(batch_kv) * query_groups * 128;
  const scale_t* qscale_base =
      query_scales + static_cast<int64_t>(batch_kv) * query_groups * 8;
#pragma unroll
  for (int group = 0; group < query_groups; ++group) {
    scores[group] = 0.0f;
  }
  bool zero_tail =
      allocations[4] == 0 && allocations[5] == 0
      && allocations[6] == 0 && allocations[7] == 0;
  if (
      zero_tail && allocations[0] == 8 && allocations[1] == 4
      && allocations[2] == 0 && allocations[3] == 0) {
    int dot0[query_groups] = {0, 0, 0, 0};
    int dot1[query_groups] = {0, 0, 0, 0};
    band_dot_gqa4(token_codes, query_base, 0, 8, dot0);
    band_dot_gqa4(token_codes + 16, query_base, 16, 4, dot1);
    float key_scale0 = static_cast<float>(token_scales[0]);
    float key_scale1 = static_cast<float>(token_scales[1]);
#pragma unroll
    for (int group = 0; group < query_groups; ++group) {
      const scale_t* qscale = qscale_base + group * 8;
      scores[group] =
          static_cast<float>(dot0[group])
              * static_cast<float>(qscale[0]) * key_scale0
          + static_cast<float>(dot1[group])
              * static_cast<float>(qscale[1]) * key_scale1;
    }
  } else if (
      zero_tail && allocations[0] == 4 && allocations[1] == 4
      && allocations[2] == 4 && allocations[3] == 0) {
    plain_accumulate_gqa4_band<4>(
        token_codes, query_base, qscale_base, token_scales,
        offsets, scale_index, 0, scores);
    plain_accumulate_gqa4_band<4>(
        token_codes, query_base, qscale_base, token_scales,
        offsets, scale_index, 1, scores);
    plain_accumulate_gqa4_band<4>(
        token_codes, query_base, qscale_base, token_scales,
        offsets, scale_index, 2, scores);
  } else if (
      zero_tail && allocations[0] == 4 && allocations[1] == 4
      && allocations[2] == 2 && allocations[3] == 1) {
    plain_accumulate_gqa4_band<4>(
        token_codes, query_base, qscale_base, token_scales,
        offsets, scale_index, 0, scores);
    plain_accumulate_gqa4_band<4>(
        token_codes, query_base, qscale_base, token_scales,
        offsets, scale_index, 1, scores);
    plain_accumulate_gqa4_band<2>(
        token_codes, query_base, qscale_base, token_scales,
        offsets, scale_index, 2, scores);
    plain_accumulate_gqa4_band<1>(
        token_codes, query_base, qscale_base, token_scales,
        offsets, scale_index, 3, scores);
  } else if (
      zero_tail && allocations[0] == 4 && allocations[1] == 4
      && allocations[2] == 1 && allocations[3] == 1) {
    plain_accumulate_gqa4_band<4>(
        token_codes, query_base, qscale_base, token_scales,
        offsets, scale_index, 0, scores);
    plain_accumulate_gqa4_band<4>(
        token_codes, query_base, qscale_base, token_scales,
        offsets, scale_index, 1, scores);
    plain_accumulate_gqa4_band<1>(
        token_codes, query_base, qscale_base, token_scales,
        offsets, scale_index, 2, scores);
    plain_accumulate_gqa4_band<1>(
        token_codes, query_base, qscale_base, token_scales,
        offsets, scale_index, 3, scores);
  } else if (
      zero_tail && allocations[0] == 4 && allocations[1] == 4
      && allocations[2] == 1 && allocations[3] == 2) {
    plain_accumulate_gqa4_band<4>(
        token_codes, query_base, qscale_base, token_scales,
        offsets, scale_index, 0, scores);
    plain_accumulate_gqa4_band<4>(
        token_codes, query_base, qscale_base, token_scales,
        offsets, scale_index, 1, scores);
    plain_accumulate_gqa4_band<1>(
        token_codes, query_base, qscale_base, token_scales,
        offsets, scale_index, 2, scores);
    plain_accumulate_gqa4_band<2>(
        token_codes, query_base, qscale_base, token_scales,
        offsets, scale_index, 3, scores);
  } else if (
      zero_tail && allocations[0] == 4 && allocations[1] == 4
      && allocations[2] == 2 && allocations[3] == 0) {
    plain_accumulate_gqa4_band<4>(
        token_codes, query_base, qscale_base, token_scales,
        offsets, scale_index, 0, scores);
    plain_accumulate_gqa4_band<4>(
        token_codes, query_base, qscale_base, token_scales,
        offsets, scale_index, 1, scores);
    plain_accumulate_gqa4_band<2>(
        token_codes, query_base, qscale_base, token_scales,
        offsets, scale_index, 2, scores);
  } else if (
      allocations[0] == 4 && allocations[1] == 1
      && allocations[2] == 1 && allocations[3] == 1
      && allocations[4] == 1 && allocations[5] == 1
      && allocations[6] == 0 && allocations[7] == 0) {
    plain_accumulate_gqa4_band<4>(
        token_codes, query_base, qscale_base, token_scales,
        offsets, scale_index, 0, scores);
    plain_accumulate_gqa4_band<1>(
        token_codes, query_base, qscale_base, token_scales,
        offsets, scale_index, 1, scores);
    plain_accumulate_gqa4_band<1>(
        token_codes, query_base, qscale_base, token_scales,
        offsets, scale_index, 2, scores);
    plain_accumulate_gqa4_band<1>(
        token_codes, query_base, qscale_base, token_scales,
        offsets, scale_index, 3, scores);
    plain_accumulate_gqa4_band<1>(
        token_codes, query_base, qscale_base, token_scales,
        offsets, scale_index, 4, scores);
    plain_accumulate_gqa4_band<1>(
        token_codes, query_base, qscale_base, token_scales,
        offsets, scale_index, 5, scores);
  } else {
#pragma unroll
    for (int band = 0; band < 8; ++band) {
      int bits = static_cast<int>(allocations[band]);
      if (bits == 0) {
        continue;
      }
      int dots[query_groups] = {0, 0, 0, 0};
      band_dot_gqa4(
          token_codes + offsets[band],
          query_base,
          16 * band,
          bits,
          dots);
      float key_scale =
          static_cast<float>(token_scales[scale_index[band]]);
#pragma unroll
      for (int group = 0; group < query_groups; ++group) {
        const scale_t* qscale = qscale_base + group * 8;
        scores[group] +=
            static_cast<float>(dots[group])
            * static_cast<float>(qscale[band])
            * key_scale;
      }
    }
  }
}

template <typename scale_t>
__device__ __forceinline__ float sortedblock_score_one(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    const int64_t* __restrict__ head_code_bases,
    const int64_t* __restrict__ head_scale_bases,
    int batch_kv,
    int query_group,
    int query_groups,
    int token,
    int block_size,
    int hot_block_count,
    int batch_kv_count) {
  int block = token / block_size;
  int local_token = token - block * block_size;
  int profile = block < hot_block_count ? 1 : 0;
  int profile_head = profile * batch_kv_count + batch_kv;
  int64_t query_row =
      static_cast<int64_t>(batch_kv) * query_groups + query_group;
  const int8_t* query = query_codes + query_row * 128;
  const scale_t* qscale = query_scales + query_row * 8;
  const int8_t* allocations = bit_allocations + profile_head * 8;
  const int16_t* offsets = code_offsets + profile_head * 8;
  const int8_t* scale_index = scale_offsets + profile_head * 8;
  int high_code_stride = code_strides[batch_kv_count + batch_kv];
  int low_code_stride = code_strides[batch_kv];
  int high_scale_stride = scale_strides[batch_kv_count + batch_kv];
  int low_scale_stride = scale_strides[batch_kv];
  int64_t code_position;
  int64_t scale_position;
  if (profile == 1) {
    code_position =
        static_cast<int64_t>(token) * high_code_stride;
    scale_position =
        static_cast<int64_t>(token) * high_scale_stride;
  } else {
    int cold_token = token - hot_block_count * block_size;
    code_position =
        static_cast<int64_t>(hot_block_count) * block_size
            * high_code_stride
        + static_cast<int64_t>(cold_token) * low_code_stride;
    scale_position =
        static_cast<int64_t>(hot_block_count) * block_size
            * high_scale_stride
        + static_cast<int64_t>(cold_token) * low_scale_stride;
  }
  const uint8_t* token_codes =
      packed_codes + head_code_bases[batch_kv] + code_position;
  const scale_t* token_scales =
      key_scales + head_scale_bases[batch_kv] + scale_position;
  float score = 0.0f;
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
  return score;
}

template <typename scale_t, int warp_count, int retained_per_warp = 32>
__global__ void plain_warpselect_sample_threshold_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int64_t* __restrict__ code_bases,
    const int64_t* __restrict__ scale_bases,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    float* __restrict__ thresholds,
    float* __restrict__ selected_masses,
    float attention_scale,
    int kv_head_count,
    int query_groups,
    int history_count,
    int sample_count,
    int selected_keep) {
  constexpr int warp_width = 32;
  constexpr int samples_per_warp = 256;
  constexpr int retained_count = warp_count * retained_per_warp;
  int row = blockIdx.x;
  int warp = threadIdx.x / warp_width;
  int lane = threadIdx.x % warp_width;
  int query_head_count = kv_head_count * query_groups;
  int batch = row / query_head_count;
  int query_head = row - batch * query_head_count;
  int kv_head = query_head / query_groups;
  int query_group = query_head - kv_head * query_groups;
  int batch_kv = batch * kv_head_count + kv_head;
  extern __shared__ float shared[];
  float* all_samples = shared;
  float* warp_samples = all_samples + warp * samples_per_warp;
  float* retained = shared + warp_count * samples_per_warp;

  for (int slot = lane; slot < samples_per_warp; slot += warp_width) {
    int sample = warp * samples_per_warp + slot;
    float score = -INFINITY;
    if (sample < sample_count) {
      int segment = max(1, history_count / sample_count);
      int phase = (row * 131 + 17) % segment;
      int64_t centered =
          (static_cast<int64_t>(2 * sample + 1) * history_count)
          / (2 * sample_count);
      int token = static_cast<int>((centered + phase) % history_count);
      score = plain_score_one(
          query_codes, query_scales, packed_codes, key_scales,
          bit_allocations, code_offsets, scale_offsets,
          code_bases, scale_bases, code_strides, scale_strides,
          batch_kv, query_group, query_groups, token);
    }
    warp_samples[slot] = score;
  }
  __syncwarp();

  for (int size = 2; size <= samples_per_warp; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int slot = lane; slot < samples_per_warp;
           slot += warp_width) {
        int other = slot ^ stride;
        if (other > slot) {
          bool ascending = (slot & size) == 0;
          float left = warp_samples[slot];
          float right = warp_samples[other];
          if ((left > right) == ascending) {
            warp_samples[slot] = right;
            warp_samples[other] = left;
          }
        }
      }
      __syncwarp();
    }
  }
  for (int retained_index = lane; retained_index < retained_per_warp;
       retained_index += warp_width) {
    retained[warp * retained_per_warp + retained_index] =
        warp_samples[samples_per_warp - 1 - retained_index];
  }
  __syncthreads();

  for (int size = 2; size <= retained_count; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int slot = threadIdx.x; slot < retained_count;
           slot += blockDim.x) {
        int other = slot ^ stride;
        if (other > slot) {
          bool ascending = (slot & size) == 0;
          float left = retained[slot];
          float right = retained[other];
          if ((left > right) == ascending) {
            retained[slot] = right;
            retained[other] = left;
          }
        }
      }
      __syncthreads();
    }
  }
  if (threadIdx.x == 0) {
    thresholds[row] = retained[retained_count - selected_keep];
  }
  if (selected_masses != nullptr) {
    __syncthreads();
    float maximum = retained[retained_count - 1];
    float local_total = 0.0f;
    for (int sample = threadIdx.x; sample < sample_count;
         sample += blockDim.x) {
      local_total += expf(
          (all_samples[sample] - maximum) * attention_scale);
    }
    float local_selected = 0.0f;
    for (int index = threadIdx.x; index < selected_keep;
         index += blockDim.x) {
      local_selected += expf(
          (retained[retained_count - 1 - index] - maximum)
          * attention_scale);
    }
    __syncthreads();
    retained[threadIdx.x] = local_total;
    all_samples[threadIdx.x] = local_selected;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        retained[threadIdx.x] += retained[threadIdx.x + stride];
        all_samples[threadIdx.x] +=
            all_samples[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      selected_masses[row] = fminf(
          1.0f,
          fmaxf(
              0.0f,
              all_samples[0] / fmaxf(retained[0], 1.0e-20f)));
    }
  }
}

template <typename scale_t, int warp_count, int retained_per_warp = 32>
__global__ void plain_warpmerge_sample_threshold_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int64_t* __restrict__ code_bases,
    const int64_t* __restrict__ scale_bases,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    float* __restrict__ thresholds,
    float* __restrict__ selected_masses,
    float attention_scale,
    int kv_head_count,
    int query_groups,
    int history_count,
    int sample_count,
    int selected_keep) {
  constexpr int warp_width = 32;
  constexpr int samples_per_warp = 256;
  int row = blockIdx.x;
  int warp = threadIdx.x / warp_width;
  int lane = threadIdx.x % warp_width;
  int query_head_count = kv_head_count * query_groups;
  int batch = row / query_head_count;
  int query_head = row - batch * query_head_count;
  int kv_head = query_head / query_groups;
  int query_group = query_head - kv_head * query_groups;
  int batch_kv = batch * kv_head_count + kv_head;
  extern __shared__ float shared[];
  float* all_samples = shared;
  float* warp_samples = all_samples + warp * samples_per_warp;
  float* retained = shared + warp_count * samples_per_warp;

  for (int slot = lane; slot < samples_per_warp; slot += warp_width) {
    int sample = warp * samples_per_warp + slot;
    float score = -INFINITY;
    if (sample < sample_count) {
      int segment = max(1, history_count / sample_count);
      int phase = (row * 131 + 17) % segment;
      int64_t centered =
          (static_cast<int64_t>(2 * sample + 1) * history_count)
          / (2 * sample_count);
      int token = static_cast<int>((centered + phase) % history_count);
      score = plain_score_one(
          query_codes, query_scales, packed_codes, key_scales,
          bit_allocations, code_offsets, scale_offsets,
          code_bases, scale_bases, code_strides, scale_strides,
          batch_kv, query_group, query_groups, token);
    }
    warp_samples[slot] = score;
  }
  __syncwarp();

  // Each warp sorts only its 256 local samples.
  for (int size = 2; size <= samples_per_warp; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int slot = lane; slot < samples_per_warp;
           slot += warp_width) {
        int other = slot ^ stride;
        if (other > slot) {
          bool ascending = (slot & size) == 0;
          float left = warp_samples[slot];
          float right = warp_samples[other];
          if ((left > right) == ascending) {
            warp_samples[slot] = right;
            warp_samples[other] = left;
          }
        }
      }
      __syncwarp();
    }
  }
  for (int retained_index = lane; retained_index < retained_per_warp;
       retained_index += warp_width) {
    retained[warp * retained_per_warp + retained_index] =
        warp_samples[samples_per_warp - 1 - retained_index];
  }
  __syncthreads();

  // The retained lists are already descending.  Exact k-way merge needs only
  // selected_keep * warp_count comparisons instead of sorting all retained
  // values with a second global bitonic network.
  if (threadIdx.x == 0) {
    float sample_maximum = -INFINITY;
    if (selected_masses != nullptr) {
#pragma unroll
      for (int source_warp = 0; source_warp < warp_count; ++source_warp) {
        sample_maximum = fmaxf(
            sample_maximum,
            retained[source_warp * retained_per_warp]);
      }
    }
    int cursors[warp_count];
#pragma unroll
    for (int source_warp = 0; source_warp < warp_count; ++source_warp) {
      cursors[source_warp] = 0;
    }
    float threshold = -INFINITY;
    float selected_total = 0.0f;
    for (int rank = 0; rank < selected_keep; ++rank) {
      float best = -INFINITY;
      int best_warp = 0;
#pragma unroll
      for (int source_warp = 0; source_warp < warp_count; ++source_warp) {
        int cursor = cursors[source_warp];
        float value = cursor < retained_per_warp
            ? retained[source_warp * retained_per_warp + cursor]
            : -INFINITY;
        if (value > best) {
          best = value;
          best_warp = source_warp;
        }
      }
      threshold = best;
      ++cursors[best_warp];
      if (selected_masses != nullptr) {
        selected_total += expf(
            (best - sample_maximum) * attention_scale);
      }
    }
    thresholds[row] = threshold;
    if (selected_masses != nullptr) {
      float sample_total = 0.0f;
      for (int sample = 0; sample < sample_count; ++sample) {
        sample_total += expf(
            (all_samples[sample] - sample_maximum) * attention_scale);
      }
      selected_masses[row] = fminf(
          1.0f,
          fmaxf(0.0f, selected_total / fmaxf(sample_total, 1.0e-20f)));
    }
  }
}

template <typename scale_t>
__global__ void plain_sample_threshold_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int64_t* __restrict__ code_bases,
    const int64_t* __restrict__ scale_bases,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    float* __restrict__ thresholds,
    float* __restrict__ selected_masses,
    float attention_scale,
    int kv_head_count,
    int query_groups,
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
  for (int sample = thread; sample < sort_count; sample += blockDim.x) {
    float score = -INFINITY;
    if (sample < sample_count) {
      int segment = max(1, history_count / sample_count);
      int phase = (row * 131 + 17) % segment;
      int64_t centered =
          (static_cast<int64_t>(2 * sample + 1) * history_count)
          / (2 * sample_count);
      int token = static_cast<int>((centered + phase) % history_count);
      score = plain_score_one(
          query_codes, query_scales, packed_codes, key_scales,
          bit_allocations, code_offsets, scale_offsets,
          code_bases, scale_bases, code_strides, scale_strides,
          batch_kv, query_group, query_groups, token);
    }
    samples[sample] = score;
  }
  __syncthreads();
  for (int size = 2; size <= sort_count; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int sample = thread; sample < sort_count;
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
    if (selected_masses != nullptr) {
      float maximum = samples[sort_count - 1];
      float total_partition = 0.0f;
      float selected_partition = 0.0f;
      int first_valid = sort_count - sample_count;
      int first_selected = sort_count - selected_keep;
      for (int index = first_valid; index < sort_count; ++index) {
        float weight = expf(
            (samples[index] - maximum) * attention_scale);
        total_partition += weight;
        if (index >= first_selected) {
          selected_partition += weight;
        }
      }
      selected_masses[row] = fminf(
          1.0f,
          fmaxf(
              0.0f,
              selected_partition
                  / fmaxf(total_partition, 1.0e-20f)));
    }
  }
}

template <typename scale_t, int query_groups>
__global__ void plain_threshold_compact_gqa4_indices_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int64_t* __restrict__ code_bases,
    const int64_t* __restrict__ scale_bases,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    const float* __restrict__ thresholds,
    int64_t* __restrict__ candidate_indices,
    int64_t* __restrict__ candidate_counts,
    bool* __restrict__ overflow,
    int kv_head_count,
    int history_count,
    int candidate_capacity) {
  int token = blockIdx.y * blockDim.x + threadIdx.x;
  if (token >= history_count) {
    return;
  }
  int batch_kv = blockIdx.x;
  int batch = batch_kv / kv_head_count;
  int kv_head = batch_kv - batch * kv_head_count;
  float scores[query_groups];
  if constexpr (query_groups == 4) {
    plain_score_gqa4_one(
        query_codes, query_scales, packed_codes, key_scales,
        bit_allocations, code_offsets, scale_offsets,
        code_bases, scale_bases, code_strides, scale_strides,
        batch_kv, token, scores);
  } else {
    scores[0] = plain_score_one(
        query_codes, query_scales, packed_codes, key_scales,
        bit_allocations, code_offsets, scale_offsets,
        code_bases, scale_bases, code_strides, scale_strides,
        batch_kv, 0, 1, token);
  }
#pragma unroll
  for (int query_group = 0; query_group < query_groups; ++query_group) {
    int query_head = kv_head * query_groups + query_group;
    int row = batch * kv_head_count * query_groups + query_head;
    if (scores[query_group] < thresholds[row]) {
      continue;
    }
    unsigned long long* count =
        reinterpret_cast<unsigned long long*>(candidate_counts + row);
    unsigned long long slot = atomicAdd(count, 1ULL);
    if (slot < static_cast<unsigned long long>(candidate_capacity)) {
      candidate_indices[
          static_cast<int64_t>(row) * candidate_capacity + slot] = token;
    } else {
      overflow[row] = true;
    }
  }
}

__device__ __forceinline__ float qksieve_warp_sum(float value);

template <typename scale_t>
__global__ void plain_threshold_compact_gqa4_mass_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int64_t* __restrict__ code_bases,
    const int64_t* __restrict__ scale_bases,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    const float* __restrict__ thresholds,
    int64_t* __restrict__ candidate_indices,
    int64_t* __restrict__ candidate_counts,
    bool* __restrict__ overflow,
    float* __restrict__ selected_denominator,
    float* __restrict__ tail_denominator,
    int kv_head_count,
    int history_count,
    int candidate_capacity,
    float attention_scale) {
  constexpr int query_groups = 4;
  constexpr int warp_count = 8;
  int token = blockIdx.y * blockDim.x + threadIdx.x;
  bool valid = token < history_count;
  int batch_kv = blockIdx.x;
  int batch = batch_kv / kv_head_count;
  int kv_head = batch_kv - batch * kv_head_count;
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  float scores[query_groups] = {
      -INFINITY, -INFINITY, -INFINITY, -INFINITY};
  if (valid) {
    plain_score_gqa4_one(
        query_codes, query_scales, packed_codes, key_scales,
        bit_allocations, code_offsets, scale_offsets,
        code_bases, scale_bases, code_strides, scale_strides,
        batch_kv, token, scores);
  }

  __shared__ float warp_partials[warp_count * query_groups * 2];
  for (int group = 0; group < query_groups; ++group) {
    int query_head = kv_head * query_groups + group;
    int row = batch * kv_head_count * query_groups + query_head;
    bool keep = valid && scores[group] >= thresholds[row];
    if (keep) {
      unsigned long long* count =
          reinterpret_cast<unsigned long long*>(candidate_counts + row);
      unsigned long long slot = atomicAdd(count, 1ULL);
      if (slot < static_cast<unsigned long long>(candidate_capacity)) {
        candidate_indices[
            static_cast<int64_t>(row) * candidate_capacity + slot] = token;
      } else {
        overflow[row] = true;
      }
    }
    float selected_weight = keep
        ? expf(fminf(
            70.0f,
            (scores[group] - thresholds[row]) * attention_scale))
        : 0.0f;
    float tail_weight = valid && !keep
        ? expf((scores[group] - thresholds[row]) * attention_scale)
        : 0.0f;
    float reduced = qksieve_warp_sum(selected_weight);
    if (lane == 0) {
      warp_partials[(warp * query_groups + group) * 2] = reduced;
    }
    reduced = qksieve_warp_sum(tail_weight);
    if (lane == 0) {
      warp_partials[(warp * query_groups + group) * 2 + 1] = reduced;
    }
  }
  __syncthreads();

  if (warp == 0) {
    constexpr int component_count = query_groups * 2;
    for (int component = lane; component < component_count;
         component += 32) {
      int group = component / 2;
      int local_component = component - group * 2;
      float total = 0.0f;
#pragma unroll
      for (int source_warp = 0; source_warp < warp_count; ++source_warp) {
        total += warp_partials[
            (source_warp * query_groups + group) * 2 + local_component];
      }
      int query_head = kv_head * query_groups + group;
      int row = batch * kv_head_count * query_groups + query_head;
      if (local_component == 0) {
        atomicAdd(selected_denominator + row, total);
      } else {
        atomicAdd(tail_denominator + row, total);
      }
    }
  }
}

constexpr int QKSIEVE_MAX_MASS_LADDER_RUNGS = 16;

template <typename scale_t>
__global__ void plain_mass_ladder_bins_gqa4_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int64_t* __restrict__ code_bases,
    const int64_t* __restrict__ scale_bases,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    const float* __restrict__ ladder_thresholds,
    float* __restrict__ mass_bins,
    int row_count,
    int kv_head_count,
    int history_count,
    int rung_count,
    float attention_scale) {
  constexpr int query_groups = 4;
  constexpr int bin_stride = QKSIEVE_MAX_MASS_LADDER_RUNGS + 1;
  __shared__ float block_bins[query_groups * bin_stride];
  int active_components = query_groups * (rung_count + 1);
  for (int component = threadIdx.x; component < query_groups * bin_stride;
       component += blockDim.x) {
    block_bins[component] = 0.0f;
  }
  __syncthreads();

  int token = blockIdx.y * blockDim.x + threadIdx.x;
  bool valid = token < history_count;
  int batch_kv = blockIdx.x;
  int batch = batch_kv / kv_head_count;
  int kv_head = batch_kv - batch * kv_head_count;
  float scores[query_groups] = {
      -INFINITY, -INFINITY, -INFINITY, -INFINITY};
  if (valid) {
    plain_score_gqa4_one(
        query_codes, query_scales, packed_codes, key_scales,
        bit_allocations, code_offsets, scale_offsets,
        code_bases, scale_bases, code_strides, scale_strides,
        batch_kv, token, scores);
  }
  if (valid) {
#pragma unroll
    for (int group = 0; group < query_groups; ++group) {
      int query_head = kv_head * query_groups + group;
      int row = batch * kv_head_count * query_groups + query_head;
      int bin = rung_count;
      for (int rung = 0; rung < rung_count; ++rung) {
        if (scores[group] >= ladder_thresholds[rung * row_count + row]) {
          bin = rung;
          break;
        }
      }
      // The first rung is the highest threshold and is much closer to the
      // row maximum than the final rung. It keeps exp(score-reference)
      // numerically stable even for adversarially wide synthetic logits.
      float reference = ladder_thresholds[row];
      float weight = expf(fminf(
          70.0f, (scores[group] - reference) * attention_scale));
      atomicAdd(block_bins + group * bin_stride + bin, weight);
    }
  }
  __syncthreads();

  for (int component = threadIdx.x; component < active_components;
       component += blockDim.x) {
    int group = component / (rung_count + 1);
    int bin = component - group * (rung_count + 1);
    int query_head = kv_head * query_groups + group;
    int row = batch * kv_head_count * query_groups + query_head;
    atomicAdd(
        mass_bins + bin * row_count + row,
        block_bins[group * bin_stride + bin]);
  }
}

__global__ void choose_mass_ladder_threshold_kernel(
    const float* __restrict__ ladder_thresholds,
    const float* __restrict__ mass_bins,
    float* __restrict__ chosen_thresholds,
    int64_t* __restrict__ chosen_rungs,
    float* __restrict__ chosen_mass,
    int row_count,
    int rung_count,
    float target_mass) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) {
    return;
  }
  float total = 0.0f;
  for (int bin = 0; bin <= rung_count; ++bin) {
    total += mass_bins[bin * row_count + row];
  }
  float cumulative = 0.0f;
  int chosen = rung_count - 1;
  for (int rung = 0; rung < rung_count; ++rung) {
    cumulative += mass_bins[rung * row_count + row];
    if (cumulative >= target_mass * fmaxf(total, 1.0e-20f)) {
      chosen = rung;
      break;
    }
  }
  chosen_thresholds[row] = ladder_thresholds[chosen * row_count + row];
  chosen_rungs[row] = static_cast<int64_t>(chosen);
  chosen_mass[row] = fminf(
      1.0f, fmaxf(0.0f, cumulative / fmaxf(total, 1.0e-20f)));
}

__device__ __forceinline__ float qksieve_warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

template <typename scale_t>
__global__ void plain_threshold_compact_gqa4_condtail_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int64_t* __restrict__ code_bases,
    const int64_t* __restrict__ scale_bases,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    const float* __restrict__ thresholds,
    int64_t* __restrict__ candidate_indices,
    int64_t* __restrict__ candidate_counts,
    bool* __restrict__ overflow,
    float* __restrict__ selected_denominator,
    float* __restrict__ tail_denominator,
    float* __restrict__ tail_block_denominator,
    float* __restrict__ tail_weighted_x,
    int kv_head_count,
    int history_count,
    int candidate_capacity,
    int moment_block_size,
    int moment_block_count,
    float attention_scale) {
  constexpr int query_groups = 4;
  constexpr int moment_rank = 8;
  constexpr int warp_count = 8;
  constexpr int component_count = 2 + moment_rank;
  int token = blockIdx.y * blockDim.x + threadIdx.x;
  bool valid = token < history_count;
  int batch_kv = blockIdx.x;
  int batch = batch_kv / kv_head_count;
  int kv_head = batch_kv - batch * kv_head_count;
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  float scores[query_groups] = {
      -INFINITY, -INFINITY, -INFINITY, -INFINITY};
  float coordinates[moment_rank] = {
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  if (valid) {
    plain_score_gqa4_one(
        query_codes, query_scales, packed_codes, key_scales,
        bit_allocations, code_offsets, scale_offsets,
        code_bases, scale_bases, code_strides, scale_strides,
        batch_kv, token, scores);
    const uint8_t* token_codes = packed_codes + code_bases[batch_kv]
        + static_cast<int64_t>(token) * code_strides[batch_kv]
        + code_offsets[batch_kv * 8];
    const scale_t* token_scales = key_scales + scale_bases[batch_kv]
        + static_cast<int64_t>(token) * scale_strides[batch_kv];
    float scale = static_cast<float>(
        token_scales[scale_offsets[batch_kv * 8]]);
    int first_band_bits = static_cast<int>(
        bit_allocations[batch_kv * 8]);
#pragma unroll
    for (int rank = 0; rank < moment_rank; ++rank) {
      int code;
      if (first_band_bits == 8) {
        code = static_cast<int8_t>(token_codes[rank]);
      } else {
        uint8_t packed = token_codes[rank >> 1];
        code = (rank & 1) == 0
            ? signed_int4(static_cast<int>(packed & 0x0f))
            : signed_int4(static_cast<int>(packed >> 4));
      }
      coordinates[rank] = static_cast<float>(code) * scale;
    }
  }

  __shared__ float warp_partials[
      warp_count * query_groups * component_count];
  for (int group = 0; group < query_groups; ++group) {
    int query_head = kv_head * query_groups + group;
    int row = batch * kv_head_count * query_groups + query_head;
    bool keep = valid && scores[group] >= thresholds[row];
    if (keep) {
      unsigned long long* count =
          reinterpret_cast<unsigned long long*>(candidate_counts + row);
      unsigned long long slot = atomicAdd(count, 1ULL);
      if (slot < static_cast<unsigned long long>(candidate_capacity)) {
        candidate_indices[
            static_cast<int64_t>(row) * candidate_capacity + slot] = token;
      } else {
        overflow[row] = true;
      }
    }
    float selected_weight = keep
        ? expf(fminf(
            70.0f,
            (scores[group] - thresholds[row]) * attention_scale))
        : 0.0f;
    float tail_weight = valid && !keep
        ? expf((scores[group] - thresholds[row]) * attention_scale)
        : 0.0f;
    float reduced = qksieve_warp_sum(selected_weight);
    if (lane == 0) {
      warp_partials[
          (warp * query_groups + group) * component_count] = reduced;
    }
    reduced = qksieve_warp_sum(tail_weight);
    if (lane == 0) {
      warp_partials[
          (warp * query_groups + group) * component_count + 1] = reduced;
    }
#pragma unroll
    for (int rank = 0; rank < moment_rank; ++rank) {
      reduced = qksieve_warp_sum(tail_weight * coordinates[rank]);
      if (lane == 0) {
        warp_partials[
            (warp * query_groups + group) * component_count + 2 + rank] =
            reduced;
      }
    }
  }
  __syncthreads();

  if (warp == 0) {
    for (int component = lane;
         component < query_groups * component_count;
         component += 32) {
      int group = component / component_count;
      int local_component = component - group * component_count;
      float total = 0.0f;
#pragma unroll
      for (int source_warp = 0; source_warp < warp_count; ++source_warp) {
        total += warp_partials[
            (source_warp * query_groups + group) * component_count
            + local_component];
      }
      int query_head = kv_head * query_groups + group;
      int row = batch * kv_head_count * query_groups + query_head;
      if (local_component == 0) {
        atomicAdd(selected_denominator + row, total);
      } else if (local_component == 1) {
        atomicAdd(tail_denominator + row, total);
        int moment_block = (blockIdx.y * blockDim.x) / moment_block_size;
        moment_block = min(moment_block, moment_block_count - 1);
        atomicAdd(
            tail_block_denominator
                + static_cast<int64_t>(row) * moment_block_count
                + moment_block,
            total);
      } else {
        int rank = local_component - 2;
        atomicAdd(
            tail_weighted_x
                + static_cast<int64_t>(row) * moment_rank + rank,
            total);
      }
    }
  }
}

template <typename scale_t, int value_rank, int query_groups>
__global__ void plain_threshold_compact_gqa4_valuesketch_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int64_t* __restrict__ code_bases,
    const int64_t* __restrict__ scale_bases,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    const uint8_t* __restrict__ packed_value_codes,
    const scale_t* __restrict__ value_minimum,
    const scale_t* __restrict__ value_scale,
    const float* __restrict__ thresholds,
    int64_t* __restrict__ candidate_indices,
    int64_t* __restrict__ candidate_counts,
    bool* __restrict__ overflow,
    float* __restrict__ selected_denominator,
    float* __restrict__ tail_denominator,
    float* __restrict__ tail_coefficients,
    int kv_head_count,
    int history_count,
    int candidate_capacity,
    int value_token_stride,
    int value_block_size,
    int value_block_count,
    float attention_scale) {
  constexpr int warp_count = 8;
  int token = blockIdx.y * blockDim.x + threadIdx.x;
  bool valid = token < history_count;
  int batch_kv = blockIdx.x;
  int batch = batch_kv / kv_head_count;
  int kv_head = batch_kv - batch * kv_head_count;
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  float scores[query_groups];
#pragma unroll
  for (int group = 0; group < query_groups; ++group) {
    scores[group] = -INFINITY;
  }
  if (valid) {
    if constexpr (query_groups == 4) {
      plain_score_gqa4_one(
          query_codes, query_scales, packed_codes, key_scales,
          bit_allocations, code_offsets, scale_offsets,
          code_bases, scale_bases, code_strides, scale_strides,
          batch_kv, token, scores);
    } else {
      scores[0] = plain_score_one(
          query_codes, query_scales, packed_codes, key_scales,
          bit_allocations, code_offsets, scale_offsets,
          code_bases, scale_bases, code_strides, scale_strides,
          batch_kv, 0, 1, token);
    }
  }

  float coefficients[value_rank];
#pragma unroll
  for (int rank = 0; rank < value_rank; ++rank) {
    coefficients[rank] = 0.0f;
  }
  __shared__ float shared_value_minimum[value_rank];
  __shared__ float shared_value_scale[value_rank];
  if (value_block_size == blockDim.x) {
    if (threadIdx.x < value_rank) {
      const scale_t* block_minimum = value_minimum
          + (static_cast<int64_t>(batch_kv) * value_block_count
             + blockIdx.y) * value_rank;
      const scale_t* block_scale = value_scale
          + (static_cast<int64_t>(batch_kv) * value_block_count
             + blockIdx.y) * value_rank;
      shared_value_minimum[threadIdx.x] =
          static_cast<float>(block_minimum[threadIdx.x]);
      shared_value_scale[threadIdx.x] =
          static_cast<float>(block_scale[threadIdx.x]);
    }
    __syncthreads();
  }
  if (valid) {
    int value_block = token / value_block_size;
    int packed_rank = value_rank / 2;
    const uint8_t* token_codes = packed_value_codes
        + (static_cast<int64_t>(batch_kv) * value_token_stride + token)
            * packed_rank;
    const scale_t* token_minimum = value_minimum
        + (static_cast<int64_t>(batch_kv) * value_block_count
           + value_block) * value_rank;
    const scale_t* token_scale = value_scale
        + (static_cast<int64_t>(batch_kv) * value_block_count
           + value_block) * value_rank;
#pragma unroll
    for (int packed_component = 0; packed_component < packed_rank;
         ++packed_component) {
      uint8_t packed = token_codes[packed_component];
      int even_rank = 2 * packed_component;
      int odd_rank = even_rank + 1;
      float even_code = static_cast<float>(packed & 0x0f);
      float odd_code = static_cast<float>(packed >> 4);
      if (value_block_size == blockDim.x) {
        coefficients[even_rank] = even_code;
        coefficients[odd_rank] = odd_code;
      } else {
        coefficients[even_rank] =
            static_cast<float>(token_minimum[even_rank])
            + static_cast<float>(token_scale[even_rank]) * even_code;
        coefficients[odd_rank] =
            static_cast<float>(token_minimum[odd_rank])
            + static_cast<float>(token_scale[odd_rank]) * odd_code;
      }
    }
  }

  __shared__ float warp_partials[
      warp_count * query_groups * (value_rank + 2)];
  for (int group = 0; group < query_groups; ++group) {
    int query_head = kv_head * query_groups + group;
    int row = batch * kv_head_count * query_groups + query_head;
    float threshold = thresholds[row];
    bool keep = valid && scores[group] >= threshold;
    bool tail = valid && !keep;
    if (keep) {
      unsigned long long* count =
          reinterpret_cast<unsigned long long*>(candidate_counts + row);
      unsigned long long slot = atomicAdd(count, 1ULL);
      if (slot < static_cast<unsigned long long>(candidate_capacity)) {
        candidate_indices[
            static_cast<int64_t>(row) * candidate_capacity + slot] = token;
      } else {
        overflow[row] = true;
      }
    }
    float weight = valid
        ? expf(fminf(
            70.0f, (scores[group] - threshold) * attention_scale))
        : 0.0f;
    float tail_weight = tail ? weight : 0.0f;
    float selected_weight = keep ? weight : 0.0f;
    float reduced = qksieve_warp_sum(selected_weight);
    if (lane == 0) {
      warp_partials[
          (warp * query_groups + group) * (value_rank + 2)] =
          reduced;
    }
    reduced = qksieve_warp_sum(tail_weight);
    if (lane == 0) {
      warp_partials[
          (warp * query_groups + group) * (value_rank + 2) + 1] =
          reduced;
    }
#pragma unroll
    for (int rank = 0; rank < value_rank; ++rank) {
      float weighted = tail_weight * coefficients[rank];
      reduced = qksieve_warp_sum(weighted);
      if (lane == 0) {
        warp_partials[
            (warp * query_groups + group)
                * (value_rank + 2) + rank + 2] = reduced;
      }
    }
  }
  __syncthreads();

  if (warp == 0) {
    __shared__ float block_tail_masses[query_groups];
    if (lane < query_groups) {
      float block_tail_mass = 0.0f;
#pragma unroll
      for (int source_warp = 0; source_warp < warp_count; ++source_warp) {
        block_tail_mass += warp_partials[
            (source_warp * query_groups + lane) * (value_rank + 2) + 1];
      }
      block_tail_masses[lane] = block_tail_mass;
    }
    __syncwarp();
    int component_count = query_groups * (value_rank + 2);
    for (int component = lane; component < component_count;
         component += 32) {
      int group = component / (value_rank + 2);
      int local_component = component - group * (value_rank + 2);
      float total = 0.0f;
#pragma unroll
      for (int source_warp = 0; source_warp < warp_count; ++source_warp) {
        total += warp_partials[
            (source_warp * query_groups + group)
                * (value_rank + 2) + local_component];
      }
      int query_head = kv_head * query_groups + group;
      int row = batch * kv_head_count * query_groups + query_head;
      if (local_component == 0) {
        atomicAdd(selected_denominator + row, total);
      } else if (local_component == 1) {
        atomicAdd(tail_denominator + row, total);
      } else {
        int rank = local_component - 2;
        if (value_block_size == blockDim.x) {
          total = shared_value_minimum[rank] * block_tail_masses[group]
              + shared_value_scale[rank] * total;
        }
        atomicAdd(
            tail_coefficients
                + static_cast<int64_t>(row) * value_rank
                + rank,
            total);
      }
    }
  }
}

template <typename scale_t, int value_rank>
__global__ void plain_threshold_mask_gqa4_valuesketch_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int64_t* __restrict__ code_bases,
    const int64_t* __restrict__ scale_bases,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    const uint8_t* __restrict__ packed_value_codes,
    const scale_t* __restrict__ value_minimum,
    const scale_t* __restrict__ value_scale,
    const float* __restrict__ thresholds,
    int32_t* __restrict__ selection_masks,
    float* __restrict__ tail_partials,
    int kv_head_count,
    int history_count,
    int scan_block_count,
    int value_token_stride,
    int value_block_size,
    int value_block_count,
    float attention_scale) {
  constexpr int query_groups = 4;
  constexpr int warp_count = 8;
  constexpr int component_count = value_rank + 2;
  int token = blockIdx.y * blockDim.x + threadIdx.x;
  bool valid = token < history_count;
  int batch_kv = blockIdx.x;
  int batch = batch_kv / kv_head_count;
  int kv_head = batch_kv - batch * kv_head_count;
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  float scores[query_groups] = {
      -INFINITY, -INFINITY, -INFINITY, -INFINITY};
  if (valid) {
    plain_score_gqa4_one(
        query_codes, query_scales, packed_codes, key_scales,
        bit_allocations, code_offsets, scale_offsets,
        code_bases, scale_bases, code_strides, scale_strides,
        batch_kv, token, scores);
  }

  float coefficients[value_rank];
#pragma unroll
  for (int rank = 0; rank < value_rank; ++rank) {
    coefficients[rank] = 0.0f;
  }
  __shared__ float shared_value_minimum[value_rank];
  __shared__ float shared_value_scale[value_rank];
  if (value_block_size == blockDim.x) {
    if (threadIdx.x < value_rank) {
      const scale_t* block_minimum = value_minimum
          + (static_cast<int64_t>(batch_kv) * value_block_count
             + blockIdx.y) * value_rank;
      const scale_t* block_scale = value_scale
          + (static_cast<int64_t>(batch_kv) * value_block_count
             + blockIdx.y) * value_rank;
      shared_value_minimum[threadIdx.x] =
          static_cast<float>(block_minimum[threadIdx.x]);
      shared_value_scale[threadIdx.x] =
          static_cast<float>(block_scale[threadIdx.x]);
    }
    __syncthreads();
  }
  if (valid) {
    int value_block = token / value_block_size;
    int packed_rank = value_rank / 2;
    const uint8_t* token_codes = packed_value_codes
        + (static_cast<int64_t>(batch_kv) * value_token_stride + token)
            * packed_rank;
    const scale_t* token_minimum = value_minimum
        + (static_cast<int64_t>(batch_kv) * value_block_count
           + value_block) * value_rank;
    const scale_t* token_scale = value_scale
        + (static_cast<int64_t>(batch_kv) * value_block_count
           + value_block) * value_rank;
#pragma unroll
    for (int packed_component = 0; packed_component < packed_rank;
         ++packed_component) {
      uint8_t packed = token_codes[packed_component];
      int even_rank = 2 * packed_component;
      int odd_rank = even_rank + 1;
      float even_code = static_cast<float>(packed & 0x0f);
      float odd_code = static_cast<float>(packed >> 4);
      if (value_block_size == blockDim.x) {
        coefficients[even_rank] = even_code;
        coefficients[odd_rank] = odd_code;
      } else {
        coefficients[even_rank] =
            static_cast<float>(token_minimum[even_rank])
            + static_cast<float>(token_scale[even_rank]) * even_code;
        coefficients[odd_rank] =
            static_cast<float>(token_minimum[odd_rank])
            + static_cast<float>(token_scale[odd_rank]) * odd_code;
      }
    }
  }

  __shared__ float warp_partials[
      warp_count * query_groups * component_count];
  for (int group = 0; group < query_groups; ++group) {
    int query_head = kv_head * query_groups + group;
    int row = batch * kv_head_count * query_groups + query_head;
    float threshold = thresholds[row];
    bool keep = valid && scores[group] >= threshold;
    uint32_t mask = __ballot_sync(0xffffffffu, keep);
    if (lane == 0) {
      int64_t mask_offset =
          (static_cast<int64_t>(row) * scan_block_count + blockIdx.y)
              * warp_count
          + warp;
      selection_masks[mask_offset] = static_cast<int32_t>(mask);
    }
    float weight = valid
        ? expf(fminf(
            70.0f, (scores[group] - threshold) * attention_scale))
        : 0.0f;
    float tail_weight = keep ? 0.0f : weight;
    float selected_weight = keep ? weight : 0.0f;
    float reduced = qksieve_warp_sum(selected_weight);
    if (lane == 0) {
      warp_partials[(warp * query_groups + group) * component_count] =
          reduced;
    }
    reduced = qksieve_warp_sum(tail_weight);
    if (lane == 0) {
      warp_partials[
          (warp * query_groups + group) * component_count + 1] = reduced;
    }
#pragma unroll
    for (int rank = 0; rank < value_rank; ++rank) {
      reduced = qksieve_warp_sum(tail_weight * coefficients[rank]);
      if (lane == 0) {
        warp_partials[
            (warp * query_groups + group) * component_count
                + rank + 2] = reduced;
      }
    }
  }
  __syncthreads();

  if (warp == 0) {
    __shared__ float block_tail_masses[query_groups];
    if (lane < query_groups) {
      float block_tail_mass = 0.0f;
#pragma unroll
      for (int source_warp = 0; source_warp < warp_count; ++source_warp) {
        block_tail_mass += warp_partials[
            (source_warp * query_groups + lane) * component_count + 1];
      }
      block_tail_masses[lane] = block_tail_mass;
    }
    __syncwarp();
    int all_components = query_groups * component_count;
    for (int component = lane; component < all_components;
         component += 32) {
      int group = component / component_count;
      int local_component = component - group * component_count;
      float total = 0.0f;
#pragma unroll
      for (int source_warp = 0; source_warp < warp_count; ++source_warp) {
        total += warp_partials[
            (source_warp * query_groups + group) * component_count
                + local_component];
      }
      if (local_component >= 2 && value_block_size == blockDim.x) {
        int rank = local_component - 2;
        total = shared_value_minimum[rank] * block_tail_masses[group]
            + shared_value_scale[rank] * total;
      }
      int query_head = kv_head * query_groups + group;
      int row = batch * kv_head_count * query_groups + query_head;
      int64_t partial_offset =
          (static_cast<int64_t>(row) * scan_block_count + blockIdx.y)
              * component_count
          + local_component;
      tail_partials[partial_offset] = total;
    }
  }
}

template <typename scale_t>
__global__ void plain_threshold_mask_gqa4_valuesketch_wmma_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int64_t* __restrict__ code_bases,
    const int64_t* __restrict__ scale_bases,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    const uint8_t* __restrict__ packed_value_codes,
    const scale_t* __restrict__ value_minimum,
    const scale_t* __restrict__ value_scale,
    const float* __restrict__ thresholds,
    int32_t* __restrict__ selection_masks,
    float* __restrict__ tail_partials,
    int kv_head_count,
    int history_count,
    int scan_block_count,
    int value_token_stride,
    int value_block_count,
    float attention_scale) {
  constexpr int query_groups = 4;
  constexpr int warp_count = 8;
  constexpr int value_rank = 16;
  constexpr int matrix_elements = 16 * 16;
  constexpr int component_count = value_rank + 2;
  int token = blockIdx.y * blockDim.x + threadIdx.x;
  bool valid = token < history_count;
  int batch_kv = blockIdx.x;
  int batch = batch_kv / kv_head_count;
  int kv_head = batch_kv - batch * kv_head_count;
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;

  float scores[query_groups] = {
      -INFINITY, -INFINITY, -INFINITY, -INFINITY};
  if (valid) {
    plain_score_gqa4_one(
        query_codes, query_scales, packed_codes, key_scales,
        bit_allocations, code_offsets, scale_offsets,
        code_bases, scale_bases, code_strides, scale_strides,
        batch_kv, token, scores);
  }

  float coefficients[value_rank];
#pragma unroll
  for (int rank = 0; rank < value_rank; ++rank) {
    coefficients[rank] = 0.0f;
  }
  if (valid) {
    int packed_rank = value_rank / 2;
    const uint8_t* token_codes = packed_value_codes
        + (static_cast<int64_t>(batch_kv) * value_token_stride + token)
            * packed_rank;
#pragma unroll
    for (int packed_component = 0; packed_component < packed_rank;
         ++packed_component) {
      uint8_t packed = token_codes[packed_component];
      coefficients[2 * packed_component] =
          static_cast<float>(packed & 0x0f);
      coefficients[2 * packed_component + 1] =
          static_cast<float>(packed >> 4);
    }
  }

  float tail_weights[query_groups];
  float selected_weights[query_groups];
#pragma unroll
  for (int group = 0; group < query_groups; ++group) {
    int query_head = kv_head * query_groups + group;
    int row = batch * kv_head_count * query_groups + query_head;
    float threshold = thresholds[row];
    bool keep = valid && scores[group] >= threshold;
    uint32_t mask = __ballot_sync(0xffffffffu, keep);
    if (lane == 0) {
      int64_t mask_offset =
          (static_cast<int64_t>(row) * scan_block_count + blockIdx.y)
              * warp_count
          + warp;
      selection_masks[mask_offset] = static_cast<int32_t>(mask);
    }
    float weight = valid
        ? expf(fminf(
            70.0f, (scores[group] - threshold) * attention_scale))
        : 0.0f;
    tail_weights[group] = keep ? 0.0f : weight;
    selected_weights[group] = keep ? weight : 0.0f;
  }

  __shared__ __half matrix_a[warp_count * matrix_elements];
  __shared__ __half matrix_b[warp_count * matrix_elements];
  __shared__ float matrix_c[warp_count * matrix_elements];
  __shared__ float warp_masses[warp_count * query_groups * 2];
  __shared__ float shared_value_minimum[value_rank];
  __shared__ float shared_value_scale[value_rank];
  if (threadIdx.x < value_rank) {
    const scale_t* block_minimum = value_minimum
        + (static_cast<int64_t>(batch_kv) * value_block_count + blockIdx.y)
            * value_rank;
    const scale_t* block_scale = value_scale
        + (static_cast<int64_t>(batch_kv) * value_block_count + blockIdx.y)
            * value_rank;
    shared_value_minimum[threadIdx.x] =
        static_cast<float>(block_minimum[threadIdx.x]);
    shared_value_scale[threadIdx.x] =
        static_cast<float>(block_scale[threadIdx.x]);
  }
  __syncthreads();

  __half* warp_a = matrix_a + warp * matrix_elements;
  __half* warp_b = matrix_b + warp * matrix_elements;
  float* warp_c = matrix_c + warp * matrix_elements;
  using namespace nvcuda;
  wmma::fragment<
      wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_fragment;
  wmma::fragment<
      wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_fragment;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
  wmma::fill_fragment(accumulator, 0.0f);
  for (int half = 0; half < 2; ++half) {
    for (int index = lane; index < matrix_elements; index += 32) {
      warp_a[index] = __float2half_rn(0.0f);
    }
    if (lane / 16 == half) {
      int matrix_row = lane & 15;
#pragma unroll
      for (int group = 0; group < query_groups; ++group) {
        warp_a[group * 16 + matrix_row] =
            __float2half_rn(tail_weights[group]);
      }
#pragma unroll
      for (int rank = 0; rank < value_rank; ++rank) {
        warp_b[matrix_row * value_rank + rank] =
            __float2half_rn(coefficients[rank]);
      }
    }
    __syncwarp();
    wmma::load_matrix_sync(a_fragment, warp_a, 16);
    wmma::load_matrix_sync(b_fragment, warp_b, 16);
    wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
    __syncwarp();
  }
  wmma::store_matrix_sync(
      warp_c, accumulator, 16, wmma::mem_row_major);

#pragma unroll
  for (int group = 0; group < query_groups; ++group) {
    float selected = qksieve_warp_sum(selected_weights[group]);
    float tail = qksieve_warp_sum(tail_weights[group]);
    if (lane == 0) {
      warp_masses[(warp * query_groups + group) * 2] = selected;
      warp_masses[(warp * query_groups + group) * 2 + 1] = tail;
    }
  }
  __syncthreads();

  if (warp == 0) {
    int all_components = query_groups * component_count;
    for (int component = lane; component < all_components;
         component += 32) {
      int group = component / component_count;
      int local_component = component - group * component_count;
      float total = 0.0f;
      if (local_component < 2) {
#pragma unroll
        for (int source_warp = 0; source_warp < warp_count; ++source_warp) {
          total += warp_masses[
              (source_warp * query_groups + group) * 2
              + local_component];
        }
      } else {
        int rank = local_component - 2;
#pragma unroll
        for (int source_warp = 0; source_warp < warp_count; ++source_warp) {
          total += matrix_c[
              source_warp * matrix_elements + group * value_rank + rank];
        }
        float tail_mass = 0.0f;
#pragma unroll
        for (int source_warp = 0; source_warp < warp_count; ++source_warp) {
          tail_mass += warp_masses[
              (source_warp * query_groups + group) * 2 + 1];
        }
        total = shared_value_minimum[rank] * tail_mass
            + shared_value_scale[rank] * total;
      }
      int query_head = kv_head * query_groups + group;
      int row = batch * kv_head_count * query_groups + query_head;
      int64_t partial_offset =
          (static_cast<int64_t>(row) * scan_block_count + blockIdx.y)
              * component_count
          + local_component;
      tail_partials[partial_offset] = total;
    }
  }
}

__global__ void compact_selection_masks_kernel(
    const int32_t* __restrict__ selection_masks,
    int64_t* __restrict__ candidate_indices,
    int64_t* __restrict__ candidate_counts,
    bool* __restrict__ overflow,
    int row_count,
    int words_per_row,
    int history_count,
    int candidate_capacity) {
  int row = blockIdx.x;
  int thread = threadIdx.x;
  __shared__ int scan_values[256];
  __shared__ int running_count;
  __shared__ int chunk_base;
  if (thread == 0) {
    running_count = 0;
  }
  __syncthreads();
  for (int base = 0; base < words_per_row; base += blockDim.x) {
    int word_index = base + thread;
    uint32_t mask = word_index < words_per_row
        ? static_cast<uint32_t>(selection_masks[
            static_cast<int64_t>(row) * words_per_row + word_index])
        : 0u;
    int local_count = __popc(mask);
    scan_values[thread] = local_count;
    __syncthreads();
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
      int add = thread >= offset ? scan_values[thread - offset] : 0;
      __syncthreads();
      if (thread >= offset) {
        scan_values[thread] += add;
      }
      __syncthreads();
    }
    if (thread == 0) {
      chunk_base = running_count;
    }
    __syncthreads();
    int prefix = scan_values[thread] - local_count;
    int local_rank = 0;
    while (mask != 0u) {
      int bit = __ffs(static_cast<int>(mask)) - 1;
      int token = word_index * 32 + bit;
      int slot = chunk_base + prefix + local_rank;
      if (token < history_count && slot < candidate_capacity) {
        candidate_indices[
            static_cast<int64_t>(row) * candidate_capacity + slot] = token;
      }
      mask &= mask - 1u;
      ++local_rank;
    }
    __syncthreads();
    if (thread == 0) {
      running_count += scan_values[blockDim.x - 1];
    }
    __syncthreads();
  }
  if (thread == 0) {
    int kept = min(running_count, candidate_capacity);
    candidate_counts[row] = static_cast<int64_t>(kept);
    overflow[row] = running_count > candidate_capacity;
  }
}

template <int value_rank>
__global__ void reduce_valuesketch_tail_partials_kernel(
    const float* __restrict__ tail_partials,
    float* __restrict__ selected_denominator,
    float* __restrict__ tail_denominator,
    float* __restrict__ tail_coefficients,
    int row_count,
    int scan_block_count) {
  constexpr int component_count = value_rank + 2;
  int row = blockIdx.x;
  int component = blockIdx.y;
  int thread = threadIdx.x;
  float total = 0.0f;
  for (int block = thread; block < scan_block_count; block += blockDim.x) {
    total += tail_partials[
        (static_cast<int64_t>(row) * scan_block_count + block)
            * component_count
        + component];
  }
  __shared__ float partials[256];
  partials[thread] = total;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (thread < offset) {
      partials[thread] += partials[thread + offset];
    }
    __syncthreads();
  }
  if (thread == 0) {
    if (component == 0) {
      selected_denominator[row] = partials[0];
    } else if (component == 1) {
      tail_denominator[row] = partials[0];
    } else {
      tail_coefficients[
          static_cast<int64_t>(row) * value_rank + component - 2] =
          partials[0];
    }
  }
}

template <typename scale_t>
__global__ void plain_threshold_compact_gqa4_valuesketch_base8_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int64_t* __restrict__ code_bases,
    const int64_t* __restrict__ scale_bases,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    const uint8_t* __restrict__ packed_value_codes,
    const scale_t* __restrict__ value_minimum,
    const scale_t* __restrict__ value_scale,
    const float* __restrict__ thresholds,
    int64_t* __restrict__ candidate_indices,
    int64_t* __restrict__ candidate_counts,
    bool* __restrict__ overflow,
    float* __restrict__ selected_denominator,
    float* __restrict__ tail_denominator,
    float* __restrict__ tail_coefficients,
    int kv_head_count,
    int history_count,
    int candidate_capacity,
    int value_token_stride,
    int value_block_size,
    int value_block_count,
    float attention_scale) {
  constexpr int query_groups = 4;
  constexpr int warp_count = 8;
  constexpr int stored_rank = 32;
  constexpr int active_rank = 8;
  constexpr int component_stride = active_rank + 2;
  int token = blockIdx.y * blockDim.x + threadIdx.x;
  bool valid = token < history_count;
  int batch_kv = blockIdx.x;
  int batch = batch_kv / kv_head_count;
  int kv_head = batch_kv - batch * kv_head_count;
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  float scores[query_groups] = {
      -INFINITY, -INFINITY, -INFINITY, -INFINITY};
  if (valid) {
    plain_score_gqa4_one(
        query_codes, query_scales, packed_codes, key_scales,
        bit_allocations, code_offsets, scale_offsets,
        code_bases, scale_bases, code_strides, scale_strides,
        batch_kv, token, scores);
  }

  float coefficients[active_rank];
#pragma unroll
  for (int rank = 0; rank < active_rank; ++rank) {
    coefficients[rank] = 0.0f;
  }
  if (valid) {
    int value_block = token / value_block_size;
    const uint8_t* token_codes = packed_value_codes
        + (static_cast<int64_t>(batch_kv) * value_token_stride + token)
            * (stored_rank / 2);
    const scale_t* token_minimum = value_minimum
        + (static_cast<int64_t>(batch_kv) * value_block_count
           + value_block) * stored_rank;
    const scale_t* token_scale = value_scale
        + (static_cast<int64_t>(batch_kv) * value_block_count
           + value_block) * stored_rank;
#pragma unroll
    for (int rank = 0; rank < active_rank; ++rank) {
      uint8_t packed = token_codes[rank >> 1];
      int code = (rank & 1) == 0
          ? static_cast<int>(packed & 0x0f)
          : static_cast<int>(packed >> 4);
      coefficients[rank] = static_cast<float>(token_minimum[rank])
          + static_cast<float>(token_scale[rank])
              * static_cast<float>(code);
    }
  }

  for (int group = 0; group < query_groups; ++group) {
    int query_head = kv_head * query_groups + group;
    int row = batch * kv_head_count * query_groups + query_head;
    bool keep = valid && scores[group] >= thresholds[row];
    if (keep) {
      unsigned long long* count =
          reinterpret_cast<unsigned long long*>(candidate_counts + row);
      unsigned long long slot = atomicAdd(count, 1ULL);
      if (slot < static_cast<unsigned long long>(candidate_capacity)) {
        candidate_indices[
            static_cast<int64_t>(row) * candidate_capacity + slot] = token;
      } else {
        overflow[row] = true;
      }
    }
  }

  __shared__ float warp_partials[
      warp_count * query_groups * component_stride];
  for (int group = 0; group < query_groups; ++group) {
    int query_head = kv_head * query_groups + group;
    int row = batch * kv_head_count * query_groups + query_head;
    bool keep = valid && scores[group] >= thresholds[row];
    bool tail = valid && !keep;
    float tail_weight = tail
        ? expf((scores[group] - thresholds[row]) * attention_scale)
        : 0.0f;
    float selected_weight = keep
        ? expf(fminf(
            70.0f,
            (scores[group] - thresholds[row]) * attention_scale))
        : 0.0f;
    float reduced = qksieve_warp_sum(tail_weight);
    if (lane == 0) {
      warp_partials[
          (warp * query_groups + group) * component_stride] = reduced;
    }
    reduced = qksieve_warp_sum(selected_weight);
    if (lane == 0) {
      warp_partials[
          (warp * query_groups + group) * component_stride + 1] = reduced;
    }
#pragma unroll
    for (int rank = 0; rank < active_rank; ++rank) {
      reduced = qksieve_warp_sum(tail_weight * coefficients[rank]);
      if (lane == 0) {
        warp_partials[
            (warp * query_groups + group) * component_stride
                + rank + 2] = reduced;
      }
    }
  }
  __syncthreads();

  if (warp == 0) {
    constexpr int component_count = query_groups * component_stride;
    for (int component = lane; component < component_count;
         component += 32) {
      int group = component / component_stride;
      int local_component = component - group * component_stride;
      float total = 0.0f;
#pragma unroll
      for (int source_warp = 0; source_warp < warp_count; ++source_warp) {
        total += warp_partials[
            (source_warp * query_groups + group) * component_stride
                + local_component];
      }
      int query_head = kv_head * query_groups + group;
      int row = batch * kv_head_count * query_groups + query_head;
      if (local_component == 0) {
        atomicAdd(tail_denominator + row, total);
      } else if (local_component == 1) {
        atomicAdd(selected_denominator + row, total);
      } else {
        atomicAdd(
            tail_coefficients
                + static_cast<int64_t>(row) * stored_rank
                + local_component - 2,
            total);
      }
    }
  }
}

__global__ void qksieve_mark_value_refinement_kernel(
    const float* __restrict__ selected_denominator,
    const float* __restrict__ tail_denominator,
    const float* __restrict__ rank8_residual,
    bool* __restrict__ refinement_flags,
    int query_head_count,
    int kv_head_count,
    int row_count,
    float tolerance) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) {
    return;
  }
  int query_head = row % query_head_count;
  int kv_head = query_head / 4;
  int batch = row / query_head_count;
  int batch_kv = batch * kv_head_count + kv_head;
  float selected = fmaxf(0.0f, selected_denominator[row]);
  float tail = fmaxf(0.0f, tail_denominator[row]);
  float omitted_mass = tail / fmaxf(selected + tail, 1.0e-20f);
  refinement_flags[row] =
      omitted_mass * fmaxf(rank8_residual[batch_kv], 0.0f) > tolerance;
}

template <typename scale_t>
__global__ void plain_gqa4_valuesketch_refine32_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int64_t* __restrict__ code_bases,
    const int64_t* __restrict__ scale_bases,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    const uint8_t* __restrict__ packed_value_codes,
    const scale_t* __restrict__ value_minimum,
    const scale_t* __restrict__ value_scale,
    const float* __restrict__ thresholds,
    const bool* __restrict__ refinement_flags,
    float* __restrict__ tail_coefficients,
    int kv_head_count,
    int history_count,
    int value_token_stride,
    int value_block_size,
    int value_block_count,
    float attention_scale) {
  constexpr int query_groups = 4;
  constexpr int warp_count = 8;
  constexpr int stored_rank = 32;
  constexpr int base_rank = 8;
  constexpr int extra_rank = stored_rank - base_rank;
  int batch_kv = blockIdx.x;
  int batch = batch_kv / kv_head_count;
  int kv_head = batch_kv - batch * kv_head_count;
  int row_base = batch * kv_head_count * query_groups
      + kv_head * query_groups;
  bool any_refinement = false;
#pragma unroll
  for (int group = 0; group < query_groups; ++group) {
    any_refinement = any_refinement || refinement_flags[row_base + group];
  }
  if (!any_refinement) {
    return;
  }

  int token = blockIdx.y * blockDim.x + threadIdx.x;
  bool valid = token < history_count;
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  float scores[query_groups] = {
      -INFINITY, -INFINITY, -INFINITY, -INFINITY};
  if (valid) {
    plain_score_gqa4_one(
        query_codes, query_scales, packed_codes, key_scales,
        bit_allocations, code_offsets, scale_offsets,
        code_bases, scale_bases, code_strides, scale_strides,
        batch_kv, token, scores);
  }

  float coefficients[extra_rank];
#pragma unroll
  for (int component = 0; component < extra_rank; ++component) {
    coefficients[component] = 0.0f;
  }
  if (valid) {
    int value_block = token / value_block_size;
    const uint8_t* token_codes = packed_value_codes
        + (static_cast<int64_t>(batch_kv) * value_token_stride + token)
            * (stored_rank / 2);
    const scale_t* token_minimum = value_minimum
        + (static_cast<int64_t>(batch_kv) * value_block_count
           + value_block) * stored_rank;
    const scale_t* token_scale = value_scale
        + (static_cast<int64_t>(batch_kv) * value_block_count
           + value_block) * stored_rank;
#pragma unroll
    for (int component = 0; component < extra_rank; ++component) {
      int rank = base_rank + component;
      uint8_t packed = token_codes[rank >> 1];
      int code = (rank & 1) == 0
          ? static_cast<int>(packed & 0x0f)
          : static_cast<int>(packed >> 4);
      coefficients[component] = static_cast<float>(token_minimum[rank])
          + static_cast<float>(token_scale[rank])
              * static_cast<float>(code);
    }
  }

  __shared__ float warp_partials[
      warp_count * query_groups * extra_rank];
  for (int group = 0; group < query_groups; ++group) {
    int row = row_base + group;
    bool tail = refinement_flags[row] && valid
        && scores[group] < thresholds[row];
    float weight = tail
        ? expf((scores[group] - thresholds[row]) * attention_scale)
        : 0.0f;
#pragma unroll
    for (int component = 0; component < extra_rank; ++component) {
      float reduced = qksieve_warp_sum(weight * coefficients[component]);
      if (lane == 0) {
        warp_partials[
            (warp * query_groups + group) * extra_rank + component] =
            reduced;
      }
    }
  }
  __syncthreads();

  if (warp == 0) {
    constexpr int component_count = query_groups * extra_rank;
    for (int component = lane; component < component_count;
         component += 32) {
      int group = component / extra_rank;
      int local_component = component - group * extra_rank;
      float total = 0.0f;
#pragma unroll
      for (int source_warp = 0; source_warp < warp_count; ++source_warp) {
        total += warp_partials[
            (source_warp * query_groups + group) * extra_rank
                + local_component];
      }
      int row = row_base + group;
      atomicAdd(
          tail_coefficients
              + static_cast<int64_t>(row) * stored_rank
              + base_rank + local_component,
          total);
    }
  }
}

template <typename scale_t>
__global__ void mixedblock_sample_threshold_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    const int16_t* __restrict__ block_hot_prefix,
    const int64_t* __restrict__ head_code_bases,
    const int64_t* __restrict__ head_scale_bases,
    float* __restrict__ thresholds,
    int kv_head_count,
    int query_groups,
    int history_count,
    int block_size,
    int block_count,
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
  int batch_kv_count = gridDim.x / query_groups;
  int sort_count = 1;
  while (sort_count < sample_count) {
    sort_count <<= 1;
  }
  extern __shared__ float samples[];
  for (int sample = thread; sample < sort_count; sample += blockDim.x) {
    float score = -INFINITY;
    if (sample < sample_count) {
      int segment = max(1, history_count / sample_count);
      int phase = (row * 131 + 17) % segment;
      int64_t centered =
          (static_cast<int64_t>(2 * sample + 1) * history_count)
          / (2 * sample_count);
      int token = static_cast<int>((centered + phase) % history_count);
      score = mixedblock_score_one(
          query_codes, query_scales, packed_codes, key_scales,
          bit_allocations, code_offsets, scale_offsets,
          code_strides, scale_strides, block_hot_prefix,
          head_code_bases, head_scale_bases,
          batch_kv, query_group, query_groups, token,
          block_size, block_count, batch_kv_count);
    }
    samples[sample] = score;
  }
  __syncthreads();
  for (int size = 2; size <= sort_count; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int sample = thread; sample < sort_count; sample += blockDim.x) {
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
__global__ void mixedblock_threshold_compact_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    const int16_t* __restrict__ block_hot_prefix,
    const int64_t* __restrict__ head_code_bases,
    const int64_t* __restrict__ head_scale_bases,
    const float* __restrict__ thresholds,
    int64_t* __restrict__ candidate_indices,
    float* __restrict__ candidate_scores,
    int64_t* __restrict__ candidate_counts,
    bool* __restrict__ overflow,
    int kv_head_count,
    int query_groups,
    int history_count,
    int block_size,
    int block_count,
    int candidate_capacity,
    int batch_kv_count) {
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
    float score = mixedblock_score_one(
        query_codes, query_scales, packed_codes, key_scales,
        bit_allocations, code_offsets, scale_offsets,
        code_strides, scale_strides, block_hot_prefix,
        head_code_bases, head_scale_bases,
        batch_kv, query_group, query_groups, token,
        block_size, block_count, batch_kv_count);
    if (score < thresholds[row]) {
      continue;
    }
    unsigned long long* count =
        reinterpret_cast<unsigned long long*>(candidate_counts + row);
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
__global__ void mixedblock_threshold_compact_gqa4_indices_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    const int16_t* __restrict__ block_hot_prefix,
    const int64_t* __restrict__ head_code_bases,
    const int64_t* __restrict__ head_scale_bases,
    const float* __restrict__ thresholds,
    int64_t* __restrict__ candidate_indices,
    int64_t* __restrict__ candidate_counts,
    bool* __restrict__ overflow,
    int kv_head_count,
    int history_count,
    int block_size,
    int block_count,
    int candidate_capacity,
    int batch_kv_count) {
  int token = blockIdx.y * blockDim.x + threadIdx.x;
  if (token >= history_count) {
    return;
  }
  constexpr int query_groups = 4;
  int batch_kv = blockIdx.x;
  int batch = batch_kv / kv_head_count;
  int kv_head = batch_kv - batch * kv_head_count;
  int block = token / block_size;
  int local_token = token - block * block_size;
  int prefix_offset =
      batch_kv * (block_count + 1) + block;
  int hot_before =
      static_cast<int>(block_hot_prefix[prefix_offset]);
  int profile =
      block_hot_prefix[prefix_offset + 1]
          != block_hot_prefix[prefix_offset]
      ? 1
      : 0;
  int profile_head = profile * batch_kv_count + batch_kv;
  const int8_t* allocations = bit_allocations + profile_head * 8;
  const int16_t* offsets = code_offsets + profile_head * 8;
  const int8_t* scale_index = scale_offsets + profile_head * 8;
  int low_code_stride = code_strides[batch_kv];
  int high_code_stride = code_strides[batch_kv_count + batch_kv];
  int low_scale_stride = scale_strides[batch_kv];
  int high_scale_stride = scale_strides[batch_kv_count + batch_kv];
  int64_t code_position = static_cast<int64_t>(block_size)
      * (static_cast<int64_t>(block) * low_code_stride
         + static_cast<int64_t>(hot_before)
             * (high_code_stride - low_code_stride))
      + static_cast<int64_t>(local_token)
          * code_strides[profile_head];
  int64_t scale_position = static_cast<int64_t>(block_size)
      * (static_cast<int64_t>(block) * low_scale_stride
         + static_cast<int64_t>(hot_before)
             * (high_scale_stride - low_scale_stride))
      + static_cast<int64_t>(local_token)
          * scale_strides[profile_head];
  const uint8_t* token_codes =
      packed_codes + head_code_bases[batch_kv] + code_position;
  const scale_t* token_scales =
      key_scales + head_scale_bases[batch_kv] + scale_position;
  const int8_t* query_base =
      query_codes + static_cast<int64_t>(batch_kv) * query_groups * 128;
  const scale_t* qscale_base =
      query_scales + static_cast<int64_t>(batch_kv) * query_groups * 8;

  float scores[query_groups] = {0.0f, 0.0f, 0.0f, 0.0f};
  bool zero_tail =
      allocations[4] == 0 && allocations[5] == 0
      && allocations[6] == 0 && allocations[7] == 0;
  if (
      zero_tail && allocations[0] == 8 && allocations[1] == 4
      && allocations[2] == 0 && allocations[3] == 0) {
    int dot0[query_groups] = {0, 0, 0, 0};
    int dot1[query_groups] = {0, 0, 0, 0};
    band_dot_gqa4(token_codes, query_base, 0, 8, dot0);
    band_dot_gqa4(token_codes + 16, query_base, 16, 4, dot1);
    float key_scale0 = static_cast<float>(token_scales[0]);
    float key_scale1 = static_cast<float>(token_scales[1]);
#pragma unroll
    for (int group = 0; group < query_groups; ++group) {
      const scale_t* qscale = qscale_base + group * 8;
      scores[group] =
          static_cast<float>(dot0[group])
              * static_cast<float>(qscale[0]) * key_scale0
          + static_cast<float>(dot1[group])
              * static_cast<float>(qscale[1]) * key_scale1;
    }
  } else {
#pragma unroll
    for (int band = 0; band < 8; ++band) {
      int bits = static_cast<int>(allocations[band]);
      if (bits == 0) {
        continue;
      }
      int dots[query_groups] = {0, 0, 0, 0};
      band_dot_gqa4(
          token_codes + offsets[band],
          query_base,
          16 * band,
          bits,
          dots);
      float key_scale =
          static_cast<float>(token_scales[scale_index[band]]);
#pragma unroll
      for (int group = 0; group < query_groups; ++group) {
        const scale_t* qscale = qscale_base + group * 8;
        scores[group] +=
            static_cast<float>(dots[group])
            * static_cast<float>(qscale[band])
            * key_scale;
      }
    }
  }

#pragma unroll
  for (int query_group = 0; query_group < query_groups; ++query_group) {
    int query_head = kv_head * query_groups + query_group;
    int row = batch * kv_head_count * query_groups + query_head;
    if (scores[query_group] < thresholds[row]) {
      continue;
    }
    unsigned long long* count =
        reinterpret_cast<unsigned long long*>(candidate_counts + row);
    unsigned long long slot = atomicAdd(count, 1ULL);
    if (slot < static_cast<unsigned long long>(candidate_capacity)) {
      candidate_indices[
          static_cast<int64_t>(row) * candidate_capacity + slot] = token;
    } else {
      overflow[row] = true;
    }
  }
}

template <typename scalar_t>
__global__ void mixedblock_fused_attention_gqa4_kernel(
    const int8_t* __restrict__ query_codes,
    const scalar_t* __restrict__ query_scales,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const uint8_t* __restrict__ packed_codes,
    const scalar_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    const int16_t* __restrict__ block_hot_prefix,
    const int64_t* __restrict__ head_code_bases,
    const int64_t* __restrict__ head_scale_bases,
    const float* __restrict__ thresholds,
    float* __restrict__ partial_output,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    int32_t* __restrict__ partial_counts,
    bool* __restrict__ overflow,
    int kv_head_count,
    int history_count,
    int key_count,
    int block_size,
    int block_count,
    int split_count,
    int max_local_candidates,
    int batch_kv_count,
    int64_t key_stride_b,
    int64_t key_stride_h,
    int64_t key_stride_k,
    int64_t key_stride_d,
    int64_t value_stride_b,
    int64_t value_stride_h,
    int64_t value_stride_k,
    int64_t value_stride_d,
    float scaling) {
  constexpr int query_groups = 4;
  constexpr int head_dim = 128;
  int fused_block = blockIdx.x;
  int batch_kv = fused_block / split_count;
  int split = fused_block - batch_kv * split_count;
  int batch = batch_kv / kv_head_count;
  int kv_head = batch_kv - batch * kv_head_count;
  int query_head_count = kv_head_count * query_groups;
  int chunk = (history_count + split_count - 1) / split_count;
  int start = min(split * chunk, history_count);
  int end = min(start + chunk, history_count);
  int tid = threadIdx.x;

  extern __shared__ unsigned char shared_raw[];
  int* local_counts = reinterpret_cast<int*>(shared_raw);
  int* local_indices = local_counts + query_groups;
  float* local_scores = reinterpret_cast<float*>(
      local_indices + query_groups * max_local_candidates);
  float* reduction =
      local_scores + query_groups * max_local_candidates;
  float* self_scores = reduction + blockDim.x;

  if (tid < query_groups) {
    local_counts[tid] = 0;
  }
  __syncthreads();

  for (int token = start + tid; token < end; token += blockDim.x) {
    float proxy_scores[query_groups];
    mixedblock_score_gqa4_one(
        query_codes, query_scales, packed_codes, key_scales,
        bit_allocations, code_offsets, scale_offsets,
        code_strides, scale_strides, block_hot_prefix,
        head_code_bases, head_scale_bases,
        batch_kv, token, block_size, block_count,
        batch_kv_count, proxy_scores);
#pragma unroll
    for (int group = 0; group < query_groups; ++group) {
      int row =
          batch * query_head_count + kv_head * query_groups + group;
      if (proxy_scores[group] < thresholds[row]) {
        continue;
      }
      int slot = atomicAdd(local_counts + group, 1);
      if (slot < max_local_candidates) {
        local_indices[group * max_local_candidates + slot] = token;
      } else {
        overflow[row] = true;
      }
    }
  }
  __syncthreads();

  const scalar_t* key_base =
      key + batch * key_stride_b + kv_head * key_stride_h;
  const scalar_t* value_base =
      value + batch * value_stride_b + kv_head * value_stride_h;
  int self_token = key_count - 1;

#pragma unroll
  for (int group = 0; group < query_groups; ++group) {
    int row =
        batch * query_head_count + kv_head * query_groups + group;
    int partial_row = row * split_count + split;
    int local_count = min(local_counts[group], max_local_candidates);
    int* indices = local_indices + group * max_local_candidates;
    float* scores = local_scores + group * max_local_candidates;
    const scalar_t* query_row = query + row * head_dim;

    if (tid == 0) {
      partial_counts[partial_row] = local_count;
    }
    for (int local = tid; local < local_count; local += blockDim.x) {
      int token = indices[local];
      const scalar_t* key_row = key_base + token * key_stride_k;
      float score = 0.0f;
#pragma unroll 4
      for (int dim = 0; dim < head_dim; ++dim) {
        score += static_cast<float>(query_row[dim])
            * static_cast<float>(key_row[dim * key_stride_d]);
      }
      scores[local] = score * scaling;
    }
    if (tid == 0) {
      float self_score = -INFINITY;
      if (split == 0) {
        const scalar_t* self_key =
            key_base + self_token * key_stride_k;
        self_score = 0.0f;
#pragma unroll 4
        for (int dim = 0; dim < head_dim; ++dim) {
          self_score += static_cast<float>(query_row[dim])
              * static_cast<float>(
                  self_key[dim * key_stride_d]);
        }
        self_score *= scaling;
      }
      self_scores[group] = self_score;
    }
    __syncthreads();

    float thread_max = -INFINITY;
    for (int local = tid; local < local_count; local += blockDim.x) {
      thread_max = fmaxf(thread_max, scores[local]);
    }
    if (tid == 0 && split == 0) {
      thread_max = fmaxf(thread_max, self_scores[group]);
    }
    reduction[tid] = thread_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduction[tid] = fmaxf(
            reduction[tid], reduction[tid + stride]);
      }
      __syncthreads();
    }
    float block_max =
        isfinite(reduction[0]) ? reduction[0] : 0.0f;

    float thread_sum = 0.0f;
    for (int local = tid; local < local_count; local += blockDim.x) {
      float weight = expf(scores[local] - block_max);
      scores[local] = weight;
      thread_sum += weight;
    }
    if (tid == 0 && split == 0) {
      thread_sum += expf(self_scores[group] - block_max);
    }
    reduction[tid] = thread_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduction[tid] += reduction[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      partial_max[partial_row] = block_max;
      partial_sum[partial_row] = reduction[0];
    }

    float* partial_row_output =
        partial_output + static_cast<int64_t>(partial_row) * head_dim;
    for (int dim = tid; dim < head_dim; dim += blockDim.x) {
      float accumulator = 0.0f;
      for (int local = 0; local < local_count; ++local) {
        int token = indices[local];
        accumulator += scores[local]
            * static_cast<float>(
                value_base[
                    token * value_stride_k
                    + dim * value_stride_d]);
      }
      if (split == 0) {
        accumulator += expf(self_scores[group] - block_max)
            * static_cast<float>(
                value_base[
                    self_token * value_stride_k
                    + dim * value_stride_d]);
      }
      partial_row_output[dim] = accumulator;
    }
    __syncthreads();
  }
}

template <typename scalar_t>
__global__ void mixedblock_reduce_fused_attention_kernel(
    const float* __restrict__ partial_output,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    const int32_t* __restrict__ partial_counts,
    scalar_t* __restrict__ output,
    int64_t* __restrict__ candidate_counts,
    int split_count) {
  constexpr int head_dim = 128;
  __shared__ float global_max;
  __shared__ float inverse_denom;
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (tid == 0) {
    float maximum = -INFINITY;
    int64_t selected = 0;
    for (int split = 0; split < split_count; ++split) {
      int partial_row = row * split_count + split;
      if (partial_sum[partial_row] > 0.0f) {
        maximum = fmaxf(maximum, partial_max[partial_row]);
      }
      selected += static_cast<int64_t>(partial_counts[partial_row]);
    }
    maximum = isfinite(maximum) ? maximum : 0.0f;
    float denominator = 0.0f;
    for (int split = 0; split < split_count; ++split) {
      int partial_row = row * split_count + split;
      if (partial_sum[partial_row] > 0.0f) {
        denominator += partial_sum[partial_row]
            * expf(partial_max[partial_row] - maximum);
      }
    }
    global_max = maximum;
    inverse_denom = 1.0f / fmaxf(denominator, 1.0e-20f);
    candidate_counts[row] = selected;
  }
  __syncthreads();

  const float* partial_row =
      partial_output + static_cast<int64_t>(row)
          * split_count * head_dim;
  scalar_t* output_row = output + row * head_dim;
  for (int dim = tid; dim < head_dim; dim += blockDim.x) {
    float accumulator = 0.0f;
    for (int split = 0; split < split_count; ++split) {
      int partial_index = row * split_count + split;
      if (partial_sum[partial_index] > 0.0f) {
        float scale = expf(
            partial_max[partial_index] - global_max);
        accumulator += scale
            * partial_row[split * head_dim + dim];
      }
    }
    output_row[dim] =
        static_cast<scalar_t>(accumulator * inverse_denom);
  }
}

template <typename scale_t>
__global__ void sortedblock_sample_threshold_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    const int64_t* __restrict__ head_code_bases,
    const int64_t* __restrict__ head_scale_bases,
    float* __restrict__ thresholds,
    int kv_head_count,
    int query_groups,
    int history_count,
    int block_size,
    int hot_block_count,
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
  int batch_kv_count = gridDim.x / query_groups;
  int sort_count = 1;
  while (sort_count < sample_count) {
    sort_count <<= 1;
  }
  extern __shared__ float samples[];
  for (int sample = thread; sample < sort_count; sample += blockDim.x) {
    float score = -INFINITY;
    if (sample < sample_count) {
      int segment = max(1, history_count / sample_count);
      int phase = (row * 131 + 17) % segment;
      int64_t centered =
          (static_cast<int64_t>(2 * sample + 1) * history_count)
          / (2 * sample_count);
      int token = static_cast<int>((centered + phase) % history_count);
      score = sortedblock_score_one(
          query_codes, query_scales, packed_codes, key_scales,
          bit_allocations, code_offsets, scale_offsets,
          code_strides, scale_strides,
          head_code_bases, head_scale_bases,
          batch_kv, query_group, query_groups, token,
          block_size, hot_block_count, batch_kv_count);
    }
    samples[sample] = score;
  }
  __syncthreads();
  for (int size = 2; size <= sort_count; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int sample = thread; sample < sort_count; sample += blockDim.x) {
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
__global__ void sortedblock_threshold_compact_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    const int64_t* __restrict__ head_code_bases,
    const int64_t* __restrict__ head_scale_bases,
    const int32_t* __restrict__ original_blocks,
    const float* __restrict__ thresholds,
    int64_t* __restrict__ candidate_indices,
    float* __restrict__ candidate_scores,
    int64_t* __restrict__ candidate_counts,
    bool* __restrict__ overflow,
    int kv_head_count,
    int query_groups,
    int history_count,
    int block_size,
    int block_count,
    int hot_block_count,
    int candidate_capacity) {
  int token = blockIdx.y * blockDim.x + threadIdx.x;
  if (token >= history_count) {
    return;
  }
  int batch_kv = blockIdx.x;
  int batch = batch_kv / kv_head_count;
  int kv_head = batch_kv - batch * kv_head_count;
  int query_head_count = kv_head_count * query_groups;
  int physical_block = token / block_size;
  int local_token = token - physical_block * block_size;
  int original_block =
      original_blocks[batch_kv * block_count + physical_block];
  int original_token = original_block * block_size + local_token;
  if (original_token >= history_count) {
    return;
  }
  for (int query_group = 0; query_group < query_groups; ++query_group) {
    int query_head = kv_head * query_groups + query_group;
    int row = batch * query_head_count + query_head;
    float score = sortedblock_score_one(
        query_codes, query_scales, packed_codes, key_scales,
        bit_allocations, code_offsets, scale_offsets,
        code_strides, scale_strides,
        head_code_bases, head_scale_bases,
        batch_kv, query_group, query_groups, token,
        block_size, hot_block_count, gridDim.x);
    if (score < thresholds[row]) {
      continue;
    }
    unsigned long long* count =
        reinterpret_cast<unsigned long long*>(candidate_counts + row);
    unsigned long long slot = atomicAdd(count, 1ULL);
    if (slot < static_cast<unsigned long long>(candidate_capacity)) {
      int64_t output_offset =
          static_cast<int64_t>(row) * candidate_capacity + slot;
      candidate_indices[output_offset] = original_token;
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

void mixedblock_sampled_compact_out_cuda(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    torch::Tensor block_hot_prefix,
    torch::Tensor head_code_bases,
    torch::Tensor head_scale_bases,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_scores,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor overflow,
    int64_t history_count,
    int64_t block_size,
    int64_t sample_count,
    double selected_fraction) {
  TORCH_CHECK(query_codes.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(query_codes.scalar_type() == at::kChar, "query codes must be int8");
  TORCH_CHECK(packed_codes.scalar_type() == at::kByte, "codes must be uint8");
  TORCH_CHECK(bit_allocations.scalar_type() == at::kChar, "allocations must be int8");
  TORCH_CHECK(block_hot_prefix.scalar_type() == at::kShort,
              "hot prefix must be int16");
  TORCH_CHECK(head_code_bases.scalar_type() == at::kLong,
              "bases must be int64");
  TORCH_CHECK(sample_count > 0 && sample_count <= MIXEDBLOCK_MAX_SAMPLE_COUNT,
              "invalid sample count");
  TORCH_CHECK(block_size > 0, "block size must be positive");
  int batch_count = static_cast<int>(query_codes.size(0));
  int kv_head_count = static_cast<int>(query_codes.size(1));
  int query_groups = static_cast<int>(query_codes.size(2));
  int query_head_count = kv_head_count * query_groups;
  int batch_kv_count = batch_count * kv_head_count;
  int row_count = batch_count * query_head_count;
  int block_count =
      static_cast<int>(block_hot_prefix.size(1)) - 1;
  int candidate_capacity = static_cast<int>(candidate_indices.size(2));
  int selected_keep = std::max(
      1, static_cast<int>(ceil(selected_fraction * sample_count)));
  c10::cuda::CUDAGuard device_guard(query_codes.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(
      candidate_counts.data_ptr<int64_t>(), 0,
      candidate_counts.numel() * sizeof(int64_t), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      overflow.data_ptr<bool>(), 0,
      overflow.numel() * sizeof(bool), stream));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_scales.scalar_type(),
      "mixedblock_sampled_compact_out",
      [&] {
        mixedblock_sample_threshold_kernel<scalar_t><<<
            row_count, 256,
            mixedblock_sample_shared_bytes(sample_count), stream>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                bit_allocations.data_ptr<int8_t>(),
                code_offsets.data_ptr<int16_t>(),
                scale_offsets.data_ptr<int8_t>(),
                code_strides.data_ptr<int16_t>(),
                scale_strides.data_ptr<int8_t>(),
                block_hot_prefix.data_ptr<int16_t>(),
                head_code_bases.data_ptr<int64_t>(),
                head_scale_bases.data_ptr<int64_t>(),
                thresholds.data_ptr<float>(),
                kv_head_count, query_groups,
                static_cast<int>(history_count),
                static_cast<int>(block_size), block_count,
                static_cast<int>(sample_count), selected_keep);
        dim3 blocks(
            batch_kv_count,
            (history_count + 255) / 256);
        mixedblock_threshold_compact_kernel<scalar_t><<<
            blocks, 256, 0, stream>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                bit_allocations.data_ptr<int8_t>(),
                code_offsets.data_ptr<int16_t>(),
                scale_offsets.data_ptr<int8_t>(),
                code_strides.data_ptr<int16_t>(),
                scale_strides.data_ptr<int8_t>(),
                block_hot_prefix.data_ptr<int16_t>(),
                head_code_bases.data_ptr<int64_t>(),
                head_scale_bases.data_ptr<int64_t>(),
                thresholds.data_ptr<float>(),
                candidate_indices.data_ptr<int64_t>(),
                candidate_scores.data_ptr<float>(),
                candidate_counts.data_ptr<int64_t>(),
                overflow.data_ptr<bool>(),
                kv_head_count, query_groups,
                static_cast<int>(history_count),
                static_cast<int>(block_size), block_count,
                candidate_capacity, batch_kv_count);
      });
  finalize_counts_kernel<<<
      (row_count + 255) / 256, 256, 0, stream>>>(
          candidate_counts.data_ptr<int64_t>(),
          overflow.data_ptr<bool>(),
          row_count,
          candidate_capacity);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void plain_sampled_compact_gqa4_indices_out_cuda(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor overflow,
    int64_t history_count,
    int64_t sample_count,
    double selected_fraction) {
  TORCH_CHECK(query_codes.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(query_codes.scalar_type() == at::kChar,
              "query codes must be int8");
  TORCH_CHECK(packed_codes.scalar_type() == at::kByte,
              "codes must be uint8");
  TORCH_CHECK(bit_allocations.scalar_type() == at::kChar,
              "allocations must be int8");
  TORCH_CHECK(code_bases.scalar_type() == at::kLong,
              "code bases must be int64");
  TORCH_CHECK(scale_bases.scalar_type() == at::kLong,
              "scale bases must be int64");
  TORCH_CHECK(query_codes.size(2) == 1 || query_codes.size(2) == 4,
              "plain index scan requires one or four Query groups");
  TORCH_CHECK(
      sample_count > 0
          && sample_count <= MIXEDBLOCK_MAX_SAMPLE_COUNT,
      "invalid sample count");
  TORCH_CHECK(
      code_bases.is_contiguous() && scale_bases.is_contiguous()
          && code_strides.is_contiguous()
          && scale_strides.is_contiguous(),
      "plain-layout metadata must be contiguous");
  int batch_count = static_cast<int>(query_codes.size(0));
  int kv_head_count = static_cast<int>(query_codes.size(1));
  int query_groups = static_cast<int>(query_codes.size(2));
  int query_head_count = kv_head_count * query_groups;
  int batch_kv_count = batch_count * kv_head_count;
  int row_count = batch_count * query_head_count;
  int candidate_capacity = static_cast<int>(candidate_indices.size(2));
  int selected_keep = std::max(
      1, static_cast<int>(ceil(selected_fraction * sample_count)));
  bool collect_selected_mass = thresholds.numel() >= 2 * row_count;
  float* selected_masses = collect_selected_mass
      ? thresholds.data_ptr<float>() + row_count
      : nullptr;
  c10::cuda::CUDAGuard device_guard(query_codes.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(
      candidate_counts.data_ptr<int64_t>(), 0,
      candidate_counts.numel() * sizeof(int64_t), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      overflow.data_ptr<bool>(), 0,
      overflow.numel() * sizeof(bool), stream));
  if (collect_selected_mass) {
    C10_CUDA_CHECK(cudaMemsetAsync(
        selected_masses, 0,
        row_count * sizeof(float), stream));
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_scales.scalar_type(),
      "plain_sampled_compact_gqa4_indices_out",
      [&] {
        if (selected_keep <= 32 && sample_count > 1024) {
          constexpr int warp_count = 8;
          constexpr int shared_floats =
              warp_count * (256 + 32);
          plain_warpselect_sample_threshold_kernel<
              scalar_t, warp_count><<<
              row_count, warp_count * 32,
              shared_floats * sizeof(float), stream>>>(
                  query_codes.data_ptr<int8_t>(),
                  query_scales.data_ptr<scalar_t>(),
                  packed_codes.data_ptr<uint8_t>(),
                  key_scales.data_ptr<scalar_t>(),
                  bit_allocations.data_ptr<int8_t>(),
                  code_offsets.data_ptr<int16_t>(),
                  scale_offsets.data_ptr<int8_t>(),
                  code_bases.data_ptr<int64_t>(),
                  scale_bases.data_ptr<int64_t>(),
                  code_strides.data_ptr<int16_t>(),
                  scale_strides.data_ptr<int8_t>(),
                  thresholds.data_ptr<float>(),
                  selected_masses,
                  0.08838834764831845f,
                  kv_head_count, query_groups,
                  static_cast<int>(history_count),
                  static_cast<int>(sample_count), selected_keep);
        } else {
          plain_sample_threshold_kernel<scalar_t><<<
              row_count, 256,
              mixedblock_sample_shared_bytes(sample_count), stream>>>(
                  query_codes.data_ptr<int8_t>(),
                  query_scales.data_ptr<scalar_t>(),
                  packed_codes.data_ptr<uint8_t>(),
                  key_scales.data_ptr<scalar_t>(),
                  bit_allocations.data_ptr<int8_t>(),
                  code_offsets.data_ptr<int16_t>(),
                  scale_offsets.data_ptr<int8_t>(),
                  code_bases.data_ptr<int64_t>(),
                  scale_bases.data_ptr<int64_t>(),
                  code_strides.data_ptr<int16_t>(),
                  scale_strides.data_ptr<int8_t>(),
                  thresholds.data_ptr<float>(),
                  selected_masses,
                  0.08838834764831845f,
                  kv_head_count, query_groups,
                  static_cast<int>(history_count),
                  static_cast<int>(sample_count), selected_keep);
        }
        dim3 blocks(
            batch_kv_count,
            (history_count + 255) / 256);
        auto launch_compaction = [&](auto group_constant) {
          constexpr int group_value = decltype(group_constant)::value;
          plain_threshold_compact_gqa4_indices_kernel<
              scalar_t, group_value><<<blocks, 256, 0, stream>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                bit_allocations.data_ptr<int8_t>(),
                code_offsets.data_ptr<int16_t>(),
                scale_offsets.data_ptr<int8_t>(),
                code_bases.data_ptr<int64_t>(),
                scale_bases.data_ptr<int64_t>(),
                code_strides.data_ptr<int16_t>(),
                scale_strides.data_ptr<int8_t>(),
                thresholds.data_ptr<float>(),
                candidate_indices.data_ptr<int64_t>(),
                candidate_counts.data_ptr<int64_t>(),
                overflow.data_ptr<bool>(),
                kv_head_count,
                static_cast<int>(history_count),
                candidate_capacity);
        };
        if (query_groups == 1) {
          launch_compaction(std::integral_constant<int, 1>{});
        } else {
          launch_compaction(std::integral_constant<int, 4>{});
        }
      });
  finalize_counts_kernel<<<
      (row_count + 255) / 256, 256, 0, stream>>>(
          candidate_counts.data_ptr<int64_t>(),
          overflow.data_ptr<bool>(),
          row_count,
          candidate_capacity);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void plain_sampled_compact_gqa4_mass_out_cuda(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor overflow,
    torch::Tensor selected_denominator,
    torch::Tensor tail_denominator,
    int64_t history_count,
    int64_t sample_count,
    double selected_fraction,
    double attention_scale) {
  TORCH_CHECK(query_codes.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(query_codes.scalar_type() == at::kChar,
              "query codes must be int8");
  TORCH_CHECK(packed_codes.scalar_type() == at::kByte,
              "codes must be uint8");
  TORCH_CHECK(bit_allocations.scalar_type() == at::kChar,
              "allocations must be int8");
  TORCH_CHECK(query_codes.size(2) == 4,
              "GQA4 kernel requires exactly four Query groups");
  TORCH_CHECK(sample_count >= 0
                  && sample_count <= MIXEDBLOCK_MAX_SAMPLE_COUNT,
              "invalid sample count");
  TORCH_CHECK(code_bases.is_contiguous() && scale_bases.is_contiguous()
                  && code_strides.is_contiguous()
                  && scale_strides.is_contiguous(),
              "plain-layout metadata must be contiguous");
  int batch_count = static_cast<int>(query_codes.size(0));
  int kv_head_count = static_cast<int>(query_codes.size(1));
  constexpr int query_groups = 4;
  int query_head_count = kv_head_count * query_groups;
  int batch_kv_count = batch_count * kv_head_count;
  int row_count = batch_count * query_head_count;
  int candidate_capacity = static_cast<int>(candidate_indices.size(2));
  int selected_keep = sample_count > 0
      ? std::max(1, static_cast<int>(ceil(selected_fraction * sample_count)))
      : 0;
  TORCH_CHECK(thresholds.scalar_type() == at::kFloat
                  && thresholds.numel() >= row_count,
              "thresholds must contain one float per Query head");
  TORCH_CHECK(selected_denominator.scalar_type() == at::kFloat
                  && selected_denominator.numel() == row_count
                  && tail_denominator.scalar_type() == at::kFloat
                  && tail_denominator.numel() == row_count,
              "partition-sum output shapes are invalid");
  c10::cuda::CUDAGuard device_guard(query_codes.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(
      candidate_counts.data_ptr<int64_t>(), 0,
      candidate_counts.numel() * sizeof(int64_t), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      overflow.data_ptr<bool>(), 0,
      overflow.numel() * sizeof(bool), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      selected_denominator.data_ptr<float>(), 0,
      selected_denominator.numel() * sizeof(float), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      tail_denominator.data_ptr<float>(), 0,
      tail_denominator.numel() * sizeof(float), stream));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_scales.scalar_type(),
      "plain_sampled_compact_gqa4_mass_out",
      [&] {
        if (sample_count == 0) {
          // The caller supplied one measured-mass threshold per Query head.
        } else if (selected_keep <= 32 && sample_count > 1024) {
          constexpr int warp_count = 8;
          constexpr int shared_floats = warp_count * (256 + 32);
          plain_warpselect_sample_threshold_kernel<
              scalar_t, warp_count><<<
              row_count, warp_count * 32,
              shared_floats * sizeof(float), stream>>>(
                  query_codes.data_ptr<int8_t>(),
                  query_scales.data_ptr<scalar_t>(),
                  packed_codes.data_ptr<uint8_t>(),
                  key_scales.data_ptr<scalar_t>(),
                  bit_allocations.data_ptr<int8_t>(),
                  code_offsets.data_ptr<int16_t>(),
                  scale_offsets.data_ptr<int8_t>(),
                  code_bases.data_ptr<int64_t>(),
                  scale_bases.data_ptr<int64_t>(),
                  code_strides.data_ptr<int16_t>(),
                  scale_strides.data_ptr<int8_t>(),
                  thresholds.data_ptr<float>(),
                  nullptr,
                  static_cast<float>(attention_scale),
                  kv_head_count, query_groups,
                  static_cast<int>(history_count),
                  static_cast<int>(sample_count), selected_keep);
        } else {
          plain_sample_threshold_kernel<scalar_t><<<
              row_count, 256,
              mixedblock_sample_shared_bytes(sample_count), stream>>>(
                  query_codes.data_ptr<int8_t>(),
                  query_scales.data_ptr<scalar_t>(),
                  packed_codes.data_ptr<uint8_t>(),
                  key_scales.data_ptr<scalar_t>(),
                  bit_allocations.data_ptr<int8_t>(),
                  code_offsets.data_ptr<int16_t>(),
                  scale_offsets.data_ptr<int8_t>(),
                  code_bases.data_ptr<int64_t>(),
                  scale_bases.data_ptr<int64_t>(),
                  code_strides.data_ptr<int16_t>(),
                  scale_strides.data_ptr<int8_t>(),
                  thresholds.data_ptr<float>(),
                  nullptr,
                  static_cast<float>(attention_scale),
                  kv_head_count, query_groups,
                  static_cast<int>(history_count),
                  static_cast<int>(sample_count), selected_keep);
        }
        dim3 blocks(
            batch_kv_count,
            (history_count + 255) / 256);
        plain_threshold_compact_gqa4_mass_kernel<scalar_t><<<
            blocks, 256, 0, stream>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                bit_allocations.data_ptr<int8_t>(),
                code_offsets.data_ptr<int16_t>(),
                scale_offsets.data_ptr<int8_t>(),
                code_bases.data_ptr<int64_t>(),
                scale_bases.data_ptr<int64_t>(),
                code_strides.data_ptr<int16_t>(),
                scale_strides.data_ptr<int8_t>(),
                thresholds.data_ptr<float>(),
                candidate_indices.data_ptr<int64_t>(),
                candidate_counts.data_ptr<int64_t>(),
                overflow.data_ptr<bool>(),
                selected_denominator.data_ptr<float>(),
                tail_denominator.data_ptr<float>(),
                kv_head_count,
                static_cast<int>(history_count),
                candidate_capacity,
                static_cast<float>(attention_scale));
      });
  finalize_counts_kernel<<<
      (row_count + 255) / 256, 256, 0, stream>>>(
          candidate_counts.data_ptr<int64_t>(),
          overflow.data_ptr<bool>(),
          row_count,
          candidate_capacity);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void plain_mass_ladder_thresholds_out_cuda(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    torch::Tensor ladder_thresholds,
    torch::Tensor mass_bins,
    torch::Tensor chosen_thresholds,
    torch::Tensor chosen_rungs,
    torch::Tensor chosen_mass,
    int64_t history_count,
    int64_t sample_count,
    double minimum_fraction,
    double growth,
    int64_t rung_count,
    double target_mass,
    double attention_scale) {
  TORCH_CHECK(query_codes.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(query_codes.scalar_type() == at::kChar,
              "query codes must be int8");
  TORCH_CHECK(packed_codes.scalar_type() == at::kByte,
              "codes must be uint8");
  TORCH_CHECK(bit_allocations.scalar_type() == at::kChar,
              "allocations must be int8");
  TORCH_CHECK(query_codes.size(2) == 4,
              "mass ladder requires exactly four Query groups");
  TORCH_CHECK(sample_count > 1
                  && sample_count <= MIXEDBLOCK_MAX_SAMPLE_COUNT,
              "invalid sample count");
  TORCH_CHECK(minimum_fraction > 0.0 && minimum_fraction <= 1.0,
              "minimum fraction must lie in (0, 1]");
  TORCH_CHECK(growth > 1.0, "mass ladder growth must exceed one");
  TORCH_CHECK(rung_count > 0
                  && rung_count <= QKSIEVE_MAX_MASS_LADDER_RUNGS,
              "invalid mass ladder rung count");
  TORCH_CHECK(target_mass > 0.0 && target_mass < 1.0,
              "target mass must lie in (0, 1)");
  int batch_count = static_cast<int>(query_codes.size(0));
  int kv_head_count = static_cast<int>(query_codes.size(1));
  constexpr int query_groups = 4;
  int query_head_count = kv_head_count * query_groups;
  int batch_kv_count = batch_count * kv_head_count;
  int row_count = batch_count * query_head_count;
  TORCH_CHECK(ladder_thresholds.scalar_type() == at::kFloat
                  && ladder_thresholds.numel() == rung_count * row_count,
              "ladder thresholds have the wrong shape");
  TORCH_CHECK(mass_bins.scalar_type() == at::kFloat
                  && mass_bins.numel() == (rung_count + 1) * row_count,
              "mass bins have the wrong shape");
  TORCH_CHECK(chosen_thresholds.scalar_type() == at::kFloat
                  && chosen_thresholds.numel() == row_count,
              "chosen thresholds have the wrong shape");
  TORCH_CHECK(chosen_rungs.scalar_type() == at::kLong
                  && chosen_rungs.numel() == row_count,
              "chosen rungs have the wrong shape");
  TORCH_CHECK(chosen_mass.scalar_type() == at::kFloat
                  && chosen_mass.numel() == row_count,
              "chosen mass has the wrong shape");

  c10::cuda::CUDAGuard device_guard(query_codes.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(
      mass_bins.data_ptr<float>(), 0,
      mass_bins.numel() * sizeof(float), stream));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_scales.scalar_type(),
      "plain_mass_ladder_thresholds_out",
      [&] {
        double fraction = minimum_fraction;
        for (int rung = 0; rung < rung_count; ++rung) {
          int selected_keep = std::max(
              1,
              std::min(
                  static_cast<int>(sample_count),
                  static_cast<int>(ceil(fraction * sample_count))));
          plain_sample_threshold_kernel<scalar_t><<<
              row_count, 256,
              mixedblock_sample_shared_bytes(sample_count), stream>>>(
                  query_codes.data_ptr<int8_t>(),
                  query_scales.data_ptr<scalar_t>(),
                  packed_codes.data_ptr<uint8_t>(),
                  key_scales.data_ptr<scalar_t>(),
                  bit_allocations.data_ptr<int8_t>(),
                  code_offsets.data_ptr<int16_t>(),
                  scale_offsets.data_ptr<int8_t>(),
                  code_bases.data_ptr<int64_t>(),
                  scale_bases.data_ptr<int64_t>(),
                  code_strides.data_ptr<int16_t>(),
                  scale_strides.data_ptr<int8_t>(),
                  ladder_thresholds.data_ptr<float>() + rung * row_count,
                  nullptr,
                  static_cast<float>(attention_scale),
                  kv_head_count,
                  query_groups,
                  static_cast<int>(history_count),
                  static_cast<int>(sample_count),
                  selected_keep);
          fraction = std::min(1.0, fraction * growth);
        }
        dim3 blocks(
            batch_kv_count,
            (history_count + 255) / 256);
        plain_mass_ladder_bins_gqa4_kernel<scalar_t><<<
            blocks, 256, 0, stream>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                bit_allocations.data_ptr<int8_t>(),
                code_offsets.data_ptr<int16_t>(),
                scale_offsets.data_ptr<int8_t>(),
                code_bases.data_ptr<int64_t>(),
                scale_bases.data_ptr<int64_t>(),
                code_strides.data_ptr<int16_t>(),
                scale_strides.data_ptr<int8_t>(),
                ladder_thresholds.data_ptr<float>(),
                mass_bins.data_ptr<float>(),
                row_count,
                kv_head_count,
                static_cast<int>(history_count),
                static_cast<int>(rung_count),
                static_cast<float>(attention_scale));
      });
  choose_mass_ladder_threshold_kernel<<<
      (row_count + 255) / 256, 256, 0, stream>>>(
          ladder_thresholds.data_ptr<float>(),
          mass_bins.data_ptr<float>(),
          chosen_thresholds.data_ptr<float>(),
          chosen_rungs.data_ptr<int64_t>(),
          chosen_mass.data_ptr<float>(),
          row_count,
          static_cast<int>(rung_count),
          static_cast<float>(target_mass));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void plain_sampled_compact_gqa4_condtail_out_cuda(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor overflow,
    torch::Tensor selected_denominator,
    torch::Tensor tail_denominator,
    torch::Tensor tail_block_denominator,
    torch::Tensor tail_weighted_x,
    int64_t history_count,
    int64_t sample_count,
    double selected_fraction,
    int64_t moment_block_size,
    double attention_scale) {
  TORCH_CHECK(query_codes.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(query_codes.scalar_type() == at::kChar,
              "query codes must be int8");
  TORCH_CHECK(packed_codes.scalar_type() == at::kByte,
              "codes must be uint8");
  TORCH_CHECK(bit_allocations.scalar_type() == at::kChar,
              "allocations must be int8");
  TORCH_CHECK(query_codes.size(2) == 4,
              "conditional-tail kernel requires exactly four Query groups");
  TORCH_CHECK(sample_count >= 0
                  && sample_count <= MIXEDBLOCK_MAX_SAMPLE_COUNT,
              "invalid sample count");
  TORCH_CHECK(moment_block_size >= 256 && moment_block_size % 256 == 0,
              "moment block size must be a multiple of 256");
  TORCH_CHECK(code_bases.is_contiguous() && scale_bases.is_contiguous()
                  && code_strides.is_contiguous()
                  && scale_strides.is_contiguous(),
              "plain-layout metadata must be contiguous");
  int batch_count = static_cast<int>(query_codes.size(0));
  int kv_head_count = static_cast<int>(query_codes.size(1));
  constexpr int query_groups = 4;
  constexpr int moment_rank = 8;
  int query_head_count = kv_head_count * query_groups;
  int batch_kv_count = batch_count * kv_head_count;
  int row_count = batch_count * query_head_count;
  int candidate_capacity = static_cast<int>(candidate_indices.size(2));
  int selected_keep = sample_count > 0
      ? std::max(1, static_cast<int>(ceil(selected_fraction * sample_count)))
      : 0;
  int moment_block_count = static_cast<int>(tail_block_denominator.size(2));
  TORCH_CHECK(thresholds.scalar_type() == at::kFloat
                  && thresholds.numel() >= row_count,
              "thresholds must contain one float per Query head");
  TORCH_CHECK(selected_denominator.scalar_type() == at::kFloat
                  && selected_denominator.numel() == row_count
                  && tail_denominator.scalar_type() == at::kFloat
                  && tail_denominator.numel() == row_count,
              "partition-sum output shapes are invalid");
  TORCH_CHECK(tail_block_denominator.scalar_type() == at::kFloat
                  && tail_block_denominator.dim() == 3
                  && tail_block_denominator.size(0) == batch_count
                  && tail_block_denominator.size(1) == query_head_count
                  && moment_block_count
                      >= (history_count + moment_block_size - 1)
                          / moment_block_size,
              "block-tail output shape is invalid");
  TORCH_CHECK(tail_weighted_x.scalar_type() == at::kFloat
                  && tail_weighted_x.numel()
                      == static_cast<int64_t>(row_count) * moment_rank,
              "weighted-coordinate output shape is invalid");
  auto first_band_allocations = bit_allocations.select(2, 0);
  TORCH_CHECK(
      first_band_allocations.eq(4).logical_or(
          first_band_allocations.eq(8)).all().item<bool>(),
      "conditional-tail rank-8 scan requires a four- or eight-bit first band");
  c10::cuda::CUDAGuard device_guard(query_codes.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(
      candidate_counts.data_ptr<int64_t>(), 0,
      candidate_counts.numel() * sizeof(int64_t), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      overflow.data_ptr<bool>(), 0,
      overflow.numel() * sizeof(bool), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      selected_denominator.data_ptr<float>(), 0,
      selected_denominator.numel() * sizeof(float), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      tail_denominator.data_ptr<float>(), 0,
      tail_denominator.numel() * sizeof(float), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      tail_block_denominator.data_ptr<float>(), 0,
      tail_block_denominator.numel() * sizeof(float), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      tail_weighted_x.data_ptr<float>(), 0,
      tail_weighted_x.numel() * sizeof(float), stream));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_scales.scalar_type(),
      "plain_sampled_compact_gqa4_condtail_out",
      [&] {
        if (sample_count == 0) {
          // The caller supplied one measured-mass threshold per Query head.
        } else if (selected_keep <= 32 && sample_count > 1024) {
          constexpr int warp_count = 8;
          constexpr int shared_floats = warp_count * (256 + 32);
          plain_warpselect_sample_threshold_kernel<
              scalar_t, warp_count><<<
              row_count, warp_count * 32,
              shared_floats * sizeof(float), stream>>>(
                  query_codes.data_ptr<int8_t>(),
                  query_scales.data_ptr<scalar_t>(),
                  packed_codes.data_ptr<uint8_t>(),
                  key_scales.data_ptr<scalar_t>(),
                  bit_allocations.data_ptr<int8_t>(),
                  code_offsets.data_ptr<int16_t>(),
                  scale_offsets.data_ptr<int8_t>(),
                  code_bases.data_ptr<int64_t>(),
                  scale_bases.data_ptr<int64_t>(),
                  code_strides.data_ptr<int16_t>(),
                  scale_strides.data_ptr<int8_t>(),
                  thresholds.data_ptr<float>(),
                  nullptr,
                  static_cast<float>(attention_scale),
                  kv_head_count, query_groups,
                  static_cast<int>(history_count),
                  static_cast<int>(sample_count), selected_keep);
        } else {
          plain_sample_threshold_kernel<scalar_t><<<
              row_count, 256,
              mixedblock_sample_shared_bytes(sample_count), stream>>>(
                  query_codes.data_ptr<int8_t>(),
                  query_scales.data_ptr<scalar_t>(),
                  packed_codes.data_ptr<uint8_t>(),
                  key_scales.data_ptr<scalar_t>(),
                  bit_allocations.data_ptr<int8_t>(),
                  code_offsets.data_ptr<int16_t>(),
                  scale_offsets.data_ptr<int8_t>(),
                  code_bases.data_ptr<int64_t>(),
                  scale_bases.data_ptr<int64_t>(),
                  code_strides.data_ptr<int16_t>(),
                  scale_strides.data_ptr<int8_t>(),
                  thresholds.data_ptr<float>(),
                  nullptr,
                  static_cast<float>(attention_scale),
                  kv_head_count, query_groups,
                  static_cast<int>(history_count),
                  static_cast<int>(sample_count), selected_keep);
        }
        dim3 blocks(
            batch_kv_count,
            (history_count + 255) / 256);
        plain_threshold_compact_gqa4_condtail_kernel<scalar_t><<<
            blocks, 256, 0, stream>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                bit_allocations.data_ptr<int8_t>(),
                code_offsets.data_ptr<int16_t>(),
                scale_offsets.data_ptr<int8_t>(),
                code_bases.data_ptr<int64_t>(),
                scale_bases.data_ptr<int64_t>(),
                code_strides.data_ptr<int16_t>(),
                scale_strides.data_ptr<int8_t>(),
                thresholds.data_ptr<float>(),
                candidate_indices.data_ptr<int64_t>(),
                candidate_counts.data_ptr<int64_t>(),
                overflow.data_ptr<bool>(),
                selected_denominator.data_ptr<float>(),
                tail_denominator.data_ptr<float>(),
                tail_block_denominator.data_ptr<float>(),
                tail_weighted_x.data_ptr<float>(),
                kv_head_count,
                static_cast<int>(history_count),
                candidate_capacity,
                static_cast<int>(moment_block_size),
                moment_block_count,
                static_cast<float>(attention_scale));
      });
  finalize_counts_kernel<<<
      (row_count + 255) / 256, 256, 0, stream>>>(
          candidate_counts.data_ptr<int64_t>(),
          overflow.data_ptr<bool>(),
          row_count,
          candidate_capacity);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void plain_sampled_compact_gqa4_valuesketch_out_cuda(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    torch::Tensor packed_value_codes,
    torch::Tensor value_minimum,
    torch::Tensor value_scale,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor overflow,
    torch::Tensor selected_denominator,
    torch::Tensor tail_denominator,
    torch::Tensor tail_coefficients,
    int64_t history_count,
    int64_t sample_count,
    double selected_fraction,
    int64_t value_rank,
    int64_t value_block_size,
    double attention_scale) {
  TORCH_CHECK(query_codes.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(query_codes.scalar_type() == at::kChar,
              "query codes must be int8");
  TORCH_CHECK(packed_codes.scalar_type() == at::kByte
                  && packed_value_codes.scalar_type() == at::kByte,
              "Key and Value codes must be uint8");
  TORCH_CHECK(bit_allocations.scalar_type() == at::kChar,
              "allocations must be int8");
  TORCH_CHECK(query_codes.size(2) == 1 || query_codes.size(2) == 4,
              "ValueSketch scan requires one or four Query groups");
  TORCH_CHECK(
      value_rank == 8 || value_rank == 12 || value_rank == 16
          || value_rank == 32,
      "Value-sketch rank must be 8, 12, 16, or 32");
  TORCH_CHECK(value_block_size > 0,
              "Value-sketch block size must be positive");
  TORCH_CHECK(sample_count >= 0
                  && sample_count <= MIXEDBLOCK_MAX_SAMPLE_COUNT,
              "invalid sample count");
  TORCH_CHECK(query_scales.scalar_type() == value_minimum.scalar_type()
                  && query_scales.scalar_type() == value_scale.scalar_type(),
              "Query scales and Value metadata must share a dtype");
  TORCH_CHECK(packed_value_codes.is_contiguous()
                  && value_minimum.is_contiguous()
                  && value_scale.is_contiguous(),
              "Value-sketch tensors must be contiguous");
  TORCH_CHECK(packed_value_codes.dim() == 4
                  && packed_value_codes.size(3) == value_rank / 2,
              "packed Value codes must be [B,KVH,N,rank/2]");
  TORCH_CHECK(value_minimum.sizes() == value_scale.sizes()
                  && value_minimum.dim() == 4
                  && value_minimum.size(3) == value_rank,
              "Value metadata must be [B,KVH,blocks,rank]");
  int batch_count = static_cast<int>(query_codes.size(0));
  int kv_head_count = static_cast<int>(query_codes.size(1));
  int query_groups = static_cast<int>(query_codes.size(2));
  int query_head_count = kv_head_count * query_groups;
  int batch_kv_count = batch_count * kv_head_count;
  int row_count = batch_count * query_head_count;
  int candidate_capacity = static_cast<int>(candidate_indices.size(2));
  int selected_keep = sample_count > 0
      ? std::max(1, static_cast<int>(ceil(selected_fraction * sample_count)))
      : 0;
  int value_block_count = static_cast<int>(value_minimum.size(2));
  TORCH_CHECK(packed_value_codes.size(0) == batch_count
                  && packed_value_codes.size(1) == kv_head_count
                  && packed_value_codes.size(2) >= history_count,
              "packed Value-code shape does not match the query");
  TORCH_CHECK(value_minimum.size(0) == batch_count
                  && value_minimum.size(1) == kv_head_count
                  && value_block_count
                      >= (history_count + value_block_size - 1)
                          / value_block_size,
              "Value metadata does not cover the history");
  TORCH_CHECK(selected_denominator.numel() == row_count
                  && tail_denominator.numel() == row_count
                  && tail_coefficients.numel() == row_count * value_rank,
              "tail output shapes are invalid");
  bool collect_selected_mass = thresholds.numel() >= 2 * row_count;
  float* selected_masses = collect_selected_mass
      ? thresholds.data_ptr<float>() + row_count
      : nullptr;
  c10::cuda::CUDAGuard device_guard(query_codes.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(
      candidate_counts.data_ptr<int64_t>(), 0,
      candidate_counts.numel() * sizeof(int64_t), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      overflow.data_ptr<bool>(), 0,
      overflow.numel() * sizeof(bool), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      selected_denominator.data_ptr<float>(), 0,
      selected_denominator.numel() * sizeof(float), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      tail_denominator.data_ptr<float>(), 0,
      tail_denominator.numel() * sizeof(float), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      tail_coefficients.data_ptr<float>(), 0,
      tail_coefficients.numel() * sizeof(float), stream));
  if (collect_selected_mass) {
    C10_CUDA_CHECK(cudaMemsetAsync(
        selected_masses, 0, row_count * sizeof(float), stream));
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_scales.scalar_type(),
      "plain_sampled_compact_gqa4_valuesketch_out",
      [&] {
        if (sample_count == 0) {
          // The caller supplied one measured-mass threshold per Query head.
        } else if (selected_keep <= 32 && sample_count > 1024) {
          constexpr int warp_count = 8;
          constexpr int shared_floats = warp_count * (256 + 32);
          plain_warpselect_sample_threshold_kernel<
              scalar_t, warp_count><<<
              row_count, warp_count * 32,
              shared_floats * sizeof(float), stream>>>(
                  query_codes.data_ptr<int8_t>(),
                  query_scales.data_ptr<scalar_t>(),
                  packed_codes.data_ptr<uint8_t>(),
                  key_scales.data_ptr<scalar_t>(),
                  bit_allocations.data_ptr<int8_t>(),
                  code_offsets.data_ptr<int16_t>(),
                  scale_offsets.data_ptr<int8_t>(),
                  code_bases.data_ptr<int64_t>(),
                  scale_bases.data_ptr<int64_t>(),
                  code_strides.data_ptr<int16_t>(),
                  scale_strides.data_ptr<int8_t>(),
                  thresholds.data_ptr<float>(),
                  selected_masses,
                  static_cast<float>(attention_scale),
                  kv_head_count, query_groups,
                  static_cast<int>(history_count),
                  static_cast<int>(sample_count), selected_keep);
        } else {
          plain_sample_threshold_kernel<scalar_t><<<
              row_count, 256,
              mixedblock_sample_shared_bytes(sample_count), stream>>>(
                  query_codes.data_ptr<int8_t>(),
                  query_scales.data_ptr<scalar_t>(),
                  packed_codes.data_ptr<uint8_t>(),
                  key_scales.data_ptr<scalar_t>(),
                  bit_allocations.data_ptr<int8_t>(),
                  code_offsets.data_ptr<int16_t>(),
                  scale_offsets.data_ptr<int8_t>(),
                  code_bases.data_ptr<int64_t>(),
                  scale_bases.data_ptr<int64_t>(),
                  code_strides.data_ptr<int16_t>(),
                  scale_strides.data_ptr<int8_t>(),
                  thresholds.data_ptr<float>(),
                  selected_masses,
                  static_cast<float>(attention_scale),
                  kv_head_count, query_groups,
                  static_cast<int>(history_count),
                  static_cast<int>(sample_count), selected_keep);
        }
        dim3 blocks(
            batch_kv_count,
            (history_count + 255) / 256);
        auto launch_value_sketch = [&](auto rank_constant, auto group_constant) {
          constexpr int rank_value = decltype(rank_constant)::value;
          constexpr int group_value = decltype(group_constant)::value;
          plain_threshold_compact_gqa4_valuesketch_kernel<
              scalar_t, rank_value, group_value><<<blocks, 256, 0, stream>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                bit_allocations.data_ptr<int8_t>(),
                code_offsets.data_ptr<int16_t>(),
                scale_offsets.data_ptr<int8_t>(),
                code_bases.data_ptr<int64_t>(),
                scale_bases.data_ptr<int64_t>(),
                code_strides.data_ptr<int16_t>(),
                scale_strides.data_ptr<int8_t>(),
                packed_value_codes.data_ptr<uint8_t>(),
                value_minimum.data_ptr<scalar_t>(),
                value_scale.data_ptr<scalar_t>(),
                thresholds.data_ptr<float>(),
                candidate_indices.data_ptr<int64_t>(),
                candidate_counts.data_ptr<int64_t>(),
                overflow.data_ptr<bool>(),
                selected_denominator.data_ptr<float>(),
                tail_denominator.data_ptr<float>(),
                tail_coefficients.data_ptr<float>(),
                kv_head_count,
                static_cast<int>(history_count),
                candidate_capacity,
                static_cast<int>(packed_value_codes.size(2)),
                static_cast<int>(value_block_size),
                value_block_count,
                static_cast<float>(attention_scale));
        };
        auto launch_for_group = [&](auto group_constant) {
          if (value_rank == 8) {
            launch_value_sketch(
                std::integral_constant<int, 8>{}, group_constant);
          } else if (value_rank == 12) {
            launch_value_sketch(
                std::integral_constant<int, 12>{}, group_constant);
          } else if (value_rank == 16) {
            launch_value_sketch(
                std::integral_constant<int, 16>{}, group_constant);
          } else {
            launch_value_sketch(
                std::integral_constant<int, 32>{}, group_constant);
          }
        };
        if (query_groups == 1) {
          launch_for_group(std::integral_constant<int, 1>{});
        } else {
          launch_for_group(std::integral_constant<int, 4>{});
        }
      });
  finalize_counts_kernel<<<
      (row_count + 255) / 256, 256, 0, stream>>>(
          candidate_counts.data_ptr<int64_t>(),
          overflow.data_ptr<bool>(),
          row_count,
          candidate_capacity);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void plain_sampled_compact_gqa4_valuesketch_deterministic_out_cuda(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    torch::Tensor packed_value_codes,
    torch::Tensor value_minimum,
    torch::Tensor value_scale,
    torch::Tensor selection_masks,
    torch::Tensor tail_partials,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor overflow,
    torch::Tensor selected_denominator,
    torch::Tensor tail_denominator,
    torch::Tensor tail_coefficients,
    int64_t history_count,
    int64_t sample_count,
    double selected_fraction,
    int64_t value_rank,
    int64_t value_block_size,
    double attention_scale) {
  TORCH_CHECK(query_codes.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(query_codes.scalar_type() == at::kChar,
              "query codes must be int8");
  TORCH_CHECK(packed_codes.scalar_type() == at::kByte
                  && packed_value_codes.scalar_type() == at::kByte,
              "Key and Value codes must be uint8");
  TORCH_CHECK(bit_allocations.scalar_type() == at::kChar,
              "allocations must be int8");
  TORCH_CHECK(query_codes.size(2) == 4,
              "GQA4 kernel requires exactly four Query groups");
  TORCH_CHECK(value_rank == 16,
              "deterministic Value-sketch scan currently requires rank 16");
  TORCH_CHECK(value_block_size == 256,
              "deterministic Value-sketch scan requires 256-token blocks");
  TORCH_CHECK(sample_count >= 0
                  && sample_count <= MIXEDBLOCK_MAX_SAMPLE_COUNT,
              "invalid sample count");
  TORCH_CHECK(query_scales.scalar_type() == value_minimum.scalar_type()
                  && query_scales.scalar_type() == value_scale.scalar_type(),
              "Query scales and Value metadata must share a dtype");
  TORCH_CHECK(packed_value_codes.is_contiguous()
                  && value_minimum.is_contiguous()
                  && value_scale.is_contiguous()
                  && selection_masks.is_contiguous()
                  && tail_partials.is_contiguous(),
              "Value-sketch tensors and workspaces must be contiguous");
  TORCH_CHECK(selection_masks.scalar_type() == at::kInt,
              "selection-mask workspace must be int32");
  TORCH_CHECK(tail_partials.scalar_type() == at::kFloat,
              "tail-partial workspace must be float32");
  TORCH_CHECK(packed_value_codes.dim() == 4
                  && packed_value_codes.size(3) == value_rank / 2,
              "packed Value codes must be [B,KVH,N,rank/2]");
  TORCH_CHECK(value_minimum.sizes() == value_scale.sizes()
                  && value_minimum.dim() == 4
                  && value_minimum.size(3) == value_rank,
              "Value metadata must be [B,KVH,blocks,rank]");
  int batch_count = static_cast<int>(query_codes.size(0));
  int kv_head_count = static_cast<int>(query_codes.size(1));
  constexpr int query_groups = 4;
  constexpr int warp_count = 8;
  constexpr int component_count = 18;
  int query_head_count = kv_head_count * query_groups;
  int batch_kv_count = batch_count * kv_head_count;
  int row_count = batch_count * query_head_count;
  int candidate_capacity = static_cast<int>(candidate_indices.size(2));
  int selected_keep = sample_count > 0
      ? std::max(1, static_cast<int>(ceil(selected_fraction * sample_count)))
      : 0;
  int value_block_count = static_cast<int>(value_minimum.size(2));
  int scan_block_count =
      (static_cast<int>(history_count) + 255) / 256;
  int words_per_row = scan_block_count * warp_count;
  TORCH_CHECK(packed_value_codes.size(0) == batch_count
                  && packed_value_codes.size(1) == kv_head_count
                  && packed_value_codes.size(2) >= history_count,
              "packed Value-code shape does not match the query");
  TORCH_CHECK(value_minimum.size(0) == batch_count
                  && value_minimum.size(1) == kv_head_count
                  && value_block_count >= scan_block_count,
              "Value metadata does not cover the history");
  TORCH_CHECK(selection_masks.numel()
                  >= static_cast<int64_t>(row_count) * words_per_row,
              "selection-mask workspace is too small");
  TORCH_CHECK(tail_partials.numel()
                  >= static_cast<int64_t>(row_count) * scan_block_count
                      * component_count,
              "tail-partial workspace is too small");
  TORCH_CHECK(selected_denominator.numel() == row_count
                  && tail_denominator.numel() == row_count
                  && tail_coefficients.numel() == row_count * value_rank,
              "tail output shapes are invalid");
  bool collect_selected_mass = thresholds.numel() >= 2 * row_count;
  float* selected_masses = collect_selected_mass
      ? thresholds.data_ptr<float>() + row_count
      : nullptr;
  c10::cuda::CUDAGuard device_guard(query_codes.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (collect_selected_mass) {
    C10_CUDA_CHECK(cudaMemsetAsync(
        selected_masses, 0, row_count * sizeof(float), stream));
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_scales.scalar_type(),
      "plain_sampled_compact_gqa4_valuesketch_deterministic_out",
      [&] {
        if (sample_count == 0) {
          // The caller supplied one measured-mass threshold per Query head.
        } else if (
            selected_keep <= 32
            && sample_count > 1024
            && sample_count <= 2048) {
          constexpr int threshold_warp_count = 8;
          constexpr int shared_floats =
              threshold_warp_count * (256 + 32);
          plain_warpmerge_sample_threshold_kernel<
              scalar_t, threshold_warp_count><<<
              row_count, threshold_warp_count * 32,
              shared_floats * sizeof(float), stream>>>(
                  query_codes.data_ptr<int8_t>(),
                  query_scales.data_ptr<scalar_t>(),
                  packed_codes.data_ptr<uint8_t>(),
                  key_scales.data_ptr<scalar_t>(),
                  bit_allocations.data_ptr<int8_t>(),
                  code_offsets.data_ptr<int16_t>(),
                  scale_offsets.data_ptr<int8_t>(),
                  code_bases.data_ptr<int64_t>(),
                  scale_bases.data_ptr<int64_t>(),
                  code_strides.data_ptr<int16_t>(),
                  scale_strides.data_ptr<int8_t>(),
                  thresholds.data_ptr<float>(),
                  selected_masses,
                  static_cast<float>(attention_scale),
                  kv_head_count, query_groups,
                  static_cast<int>(history_count),
                  static_cast<int>(sample_count), selected_keep);
        } else if (selected_keep <= 128 && sample_count > 1024) {
          if (sample_count <= 2048) {
            constexpr int threshold_warp_count = 8;
            constexpr int retained_per_warp = 128;
            constexpr int shared_floats = threshold_warp_count
                * (256 + retained_per_warp);
            plain_warpmerge_sample_threshold_kernel<
                scalar_t,
                threshold_warp_count,
                retained_per_warp><<<
                row_count, threshold_warp_count * 32,
                shared_floats * sizeof(float), stream>>>(
                    query_codes.data_ptr<int8_t>(),
                    query_scales.data_ptr<scalar_t>(),
                    packed_codes.data_ptr<uint8_t>(),
                    key_scales.data_ptr<scalar_t>(),
                    bit_allocations.data_ptr<int8_t>(),
                    code_offsets.data_ptr<int16_t>(),
                    scale_offsets.data_ptr<int8_t>(),
                    code_bases.data_ptr<int64_t>(),
                    scale_bases.data_ptr<int64_t>(),
                    code_strides.data_ptr<int16_t>(),
                    scale_strides.data_ptr<int8_t>(),
                    thresholds.data_ptr<float>(),
                    selected_masses,
                    static_cast<float>(attention_scale),
                    kv_head_count, query_groups,
                    static_cast<int>(history_count),
                    static_cast<int>(sample_count), selected_keep);
          } else {
            constexpr int threshold_warp_count = 32;
            constexpr int retained_per_warp = 128;
            constexpr int shared_floats = threshold_warp_count
                * (256 + retained_per_warp);
            plain_warpmerge_sample_threshold_kernel<
                scalar_t,
                threshold_warp_count,
                retained_per_warp><<<
                row_count, threshold_warp_count * 32,
                shared_floats * sizeof(float), stream>>>(
                    query_codes.data_ptr<int8_t>(),
                    query_scales.data_ptr<scalar_t>(),
                    packed_codes.data_ptr<uint8_t>(),
                    key_scales.data_ptr<scalar_t>(),
                    bit_allocations.data_ptr<int8_t>(),
                    code_offsets.data_ptr<int16_t>(),
                    scale_offsets.data_ptr<int8_t>(),
                    code_bases.data_ptr<int64_t>(),
                    scale_bases.data_ptr<int64_t>(),
                    code_strides.data_ptr<int16_t>(),
                    scale_strides.data_ptr<int8_t>(),
                    thresholds.data_ptr<float>(),
                    selected_masses,
                    static_cast<float>(attention_scale),
                    kv_head_count, query_groups,
                    static_cast<int>(history_count),
                    static_cast<int>(sample_count), selected_keep);
          }
        } else {
          plain_sample_threshold_kernel<scalar_t><<<
              row_count, 256,
              mixedblock_sample_shared_bytes(sample_count), stream>>>(
                  query_codes.data_ptr<int8_t>(),
                  query_scales.data_ptr<scalar_t>(),
                  packed_codes.data_ptr<uint8_t>(),
                  key_scales.data_ptr<scalar_t>(),
                  bit_allocations.data_ptr<int8_t>(),
                  code_offsets.data_ptr<int16_t>(),
                  scale_offsets.data_ptr<int8_t>(),
                  code_bases.data_ptr<int64_t>(),
                  scale_bases.data_ptr<int64_t>(),
                  code_strides.data_ptr<int16_t>(),
                  scale_strides.data_ptr<int8_t>(),
                  thresholds.data_ptr<float>(),
                  selected_masses,
                  static_cast<float>(attention_scale),
                  kv_head_count, query_groups,
                  static_cast<int>(history_count),
                  static_cast<int>(sample_count), selected_keep);
        }
        dim3 blocks(batch_kv_count, scan_block_count);
        plain_threshold_mask_gqa4_valuesketch_wmma_kernel<
            scalar_t><<<blocks, 256, 0, stream>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                bit_allocations.data_ptr<int8_t>(),
                code_offsets.data_ptr<int16_t>(),
                scale_offsets.data_ptr<int8_t>(),
                code_bases.data_ptr<int64_t>(),
                scale_bases.data_ptr<int64_t>(),
                code_strides.data_ptr<int16_t>(),
                scale_strides.data_ptr<int8_t>(),
                packed_value_codes.data_ptr<uint8_t>(),
                value_minimum.data_ptr<scalar_t>(),
                value_scale.data_ptr<scalar_t>(),
                thresholds.data_ptr<float>(),
                selection_masks.data_ptr<int32_t>(),
                tail_partials.data_ptr<float>(),
                kv_head_count,
                static_cast<int>(history_count),
                scan_block_count,
                static_cast<int>(packed_value_codes.size(2)),
                value_block_count,
                static_cast<float>(attention_scale));
      });
  compact_selection_masks_kernel<<<row_count, 256, 0, stream>>>(
      selection_masks.data_ptr<int32_t>(),
      candidate_indices.data_ptr<int64_t>(),
      candidate_counts.data_ptr<int64_t>(),
      overflow.data_ptr<bool>(),
      row_count,
      words_per_row,
      static_cast<int>(history_count),
      candidate_capacity);
  dim3 reduction_blocks(row_count, component_count);
  reduce_valuesketch_tail_partials_kernel<16><<<
      reduction_blocks, 256, 0, stream>>>(
          tail_partials.data_ptr<float>(),
          selected_denominator.data_ptr<float>(),
          tail_denominator.data_ptr<float>(),
          tail_coefficients.data_ptr<float>(),
          row_count,
          scan_block_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void plain_sampled_compact_gqa4_valuesketch_progressive_out_cuda(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    torch::Tensor packed_value_codes,
    torch::Tensor value_minimum,
    torch::Tensor value_scale,
    torch::Tensor value_rank8_residual,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor overflow,
    torch::Tensor selected_denominator,
    torch::Tensor tail_denominator,
    torch::Tensor tail_coefficients,
    torch::Tensor refinement_flags,
    int64_t history_count,
    int64_t sample_count,
    double selected_fraction,
    int64_t value_block_size,
    double attention_scale,
    double refinement_tolerance) {
  constexpr int stored_rank = 32;
  TORCH_CHECK(query_codes.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(query_codes.scalar_type() == at::kChar,
              "query codes must be int8");
  TORCH_CHECK(packed_codes.scalar_type() == at::kByte
                  && packed_value_codes.scalar_type() == at::kByte,
              "Key and Value codes must be uint8");
  TORCH_CHECK(bit_allocations.scalar_type() == at::kChar,
              "allocations must be int8");
  TORCH_CHECK(query_codes.size(2) == 4,
              "progressive Value sketch requires GQA4");
  TORCH_CHECK(value_block_size > 0,
              "Value-sketch block size must be positive");
  TORCH_CHECK(sample_count > 0
                  && sample_count <= MIXEDBLOCK_MAX_SAMPLE_COUNT,
              "invalid sample count");
  TORCH_CHECK(refinement_tolerance >= 0.0,
              "refinement tolerance must be non-negative");
  TORCH_CHECK(query_scales.scalar_type() == value_minimum.scalar_type()
                  && query_scales.scalar_type() == value_scale.scalar_type(),
              "Query scales and Value metadata must share a dtype");
  TORCH_CHECK(packed_value_codes.is_contiguous()
                  && value_minimum.is_contiguous()
                  && value_scale.is_contiguous(),
              "Value-sketch tensors must be contiguous");
  TORCH_CHECK(packed_value_codes.dim() == 4
                  && packed_value_codes.size(3) == stored_rank / 2,
              "progressive Value codes must store rank 32 INT4");
  TORCH_CHECK(value_minimum.sizes() == value_scale.sizes()
                  && value_minimum.dim() == 4
                  && value_minimum.size(3) == stored_rank,
              "progressive Value metadata must have rank 32");
  int batch_count = static_cast<int>(query_codes.size(0));
  int kv_head_count = static_cast<int>(query_codes.size(1));
  constexpr int query_groups = 4;
  int query_head_count = kv_head_count * query_groups;
  int batch_kv_count = batch_count * kv_head_count;
  int row_count = batch_count * query_head_count;
  int candidate_capacity = static_cast<int>(candidate_indices.size(2));
  int selected_keep = std::max(
      1, static_cast<int>(ceil(selected_fraction * sample_count)));
  int value_block_count = static_cast<int>(value_minimum.size(2));
  int value_token_stride = static_cast<int>(packed_value_codes.size(2));
  TORCH_CHECK(value_token_stride >= history_count,
              "packed Value codes do not cover the history");
  TORCH_CHECK(value_minimum.size(0) == batch_count
                  && value_minimum.size(1) == kv_head_count
                  && value_block_count
                      >= (history_count + value_block_size - 1)
                          / value_block_size,
              "Value metadata does not cover the history");
  TORCH_CHECK(value_rank8_residual.scalar_type() == at::kFloat
                  && value_rank8_residual.numel() == batch_kv_count,
              "rank-8 residual must be one float per KV head");
  TORCH_CHECK(selected_denominator.scalar_type() == at::kFloat
                  && selected_denominator.numel() == row_count
                  && tail_denominator.scalar_type() == at::kFloat
                  && tail_denominator.numel() == row_count,
              "mass output shapes are invalid");
  TORCH_CHECK(tail_coefficients.scalar_type() == at::kFloat
                  && tail_coefficients.numel() == row_count * stored_rank,
              "tail coefficients must have rank 32");
  TORCH_CHECK(refinement_flags.scalar_type() == at::kBool
                  && refinement_flags.numel() == row_count,
              "refinement flags must be bool per Query head");
  TORCH_CHECK(thresholds.scalar_type() == at::kFloat
                  && thresholds.numel() >= row_count,
              "thresholds must contain one float per Query head");

  c10::cuda::CUDAGuard device_guard(query_codes.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(
      candidate_counts.data_ptr<int64_t>(), 0,
      candidate_counts.numel() * sizeof(int64_t), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      overflow.data_ptr<bool>(), 0,
      overflow.numel() * sizeof(bool), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      selected_denominator.data_ptr<float>(), 0,
      selected_denominator.numel() * sizeof(float), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      tail_denominator.data_ptr<float>(), 0,
      tail_denominator.numel() * sizeof(float), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      tail_coefficients.data_ptr<float>(), 0,
      tail_coefficients.numel() * sizeof(float), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      refinement_flags.data_ptr<bool>(), 0,
      refinement_flags.numel() * sizeof(bool), stream));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_scales.scalar_type(),
      "plain_sampled_compact_gqa4_valuesketch_progressive_out",
      [&] {
        if (selected_keep <= 32 && sample_count > 1024) {
          constexpr int warp_count = 8;
          constexpr int shared_floats = warp_count * (256 + 32);
          plain_warpselect_sample_threshold_kernel<
              scalar_t, warp_count><<<
              row_count, warp_count * 32,
              shared_floats * sizeof(float), stream>>>(
                  query_codes.data_ptr<int8_t>(),
                  query_scales.data_ptr<scalar_t>(),
                  packed_codes.data_ptr<uint8_t>(),
                  key_scales.data_ptr<scalar_t>(),
                  bit_allocations.data_ptr<int8_t>(),
                  code_offsets.data_ptr<int16_t>(),
                  scale_offsets.data_ptr<int8_t>(),
                  code_bases.data_ptr<int64_t>(),
                  scale_bases.data_ptr<int64_t>(),
                  code_strides.data_ptr<int16_t>(),
                  scale_strides.data_ptr<int8_t>(),
                  thresholds.data_ptr<float>(),
                  nullptr,
                  static_cast<float>(attention_scale),
                  kv_head_count, query_groups,
                  static_cast<int>(history_count),
                  static_cast<int>(sample_count), selected_keep);
        } else {
          plain_sample_threshold_kernel<scalar_t><<<
              row_count, 256,
              mixedblock_sample_shared_bytes(sample_count), stream>>>(
                  query_codes.data_ptr<int8_t>(),
                  query_scales.data_ptr<scalar_t>(),
                  packed_codes.data_ptr<uint8_t>(),
                  key_scales.data_ptr<scalar_t>(),
                  bit_allocations.data_ptr<int8_t>(),
                  code_offsets.data_ptr<int16_t>(),
                  scale_offsets.data_ptr<int8_t>(),
                  code_bases.data_ptr<int64_t>(),
                  scale_bases.data_ptr<int64_t>(),
                  code_strides.data_ptr<int16_t>(),
                  scale_strides.data_ptr<int8_t>(),
                  thresholds.data_ptr<float>(),
                  nullptr,
                  static_cast<float>(attention_scale),
                  kv_head_count, query_groups,
                  static_cast<int>(history_count),
                  static_cast<int>(sample_count), selected_keep);
        }
        dim3 blocks(
            batch_kv_count,
            (history_count + 255) / 256);
        plain_threshold_compact_gqa4_valuesketch_base8_kernel<
            scalar_t><<<blocks, 256, 0, stream>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                bit_allocations.data_ptr<int8_t>(),
                code_offsets.data_ptr<int16_t>(),
                scale_offsets.data_ptr<int8_t>(),
                code_bases.data_ptr<int64_t>(),
                scale_bases.data_ptr<int64_t>(),
                code_strides.data_ptr<int16_t>(),
                scale_strides.data_ptr<int8_t>(),
                packed_value_codes.data_ptr<uint8_t>(),
                value_minimum.data_ptr<scalar_t>(),
                value_scale.data_ptr<scalar_t>(),
                thresholds.data_ptr<float>(),
                candidate_indices.data_ptr<int64_t>(),
                candidate_counts.data_ptr<int64_t>(),
                overflow.data_ptr<bool>(),
                selected_denominator.data_ptr<float>(),
                tail_denominator.data_ptr<float>(),
                tail_coefficients.data_ptr<float>(),
                kv_head_count,
                static_cast<int>(history_count),
                candidate_capacity,
                value_token_stride,
                static_cast<int>(value_block_size),
                value_block_count,
                static_cast<float>(attention_scale));
        qksieve_mark_value_refinement_kernel<<<
            (row_count + 255) / 256, 256, 0, stream>>>(
                selected_denominator.data_ptr<float>(),
                tail_denominator.data_ptr<float>(),
                value_rank8_residual.data_ptr<float>(),
                refinement_flags.data_ptr<bool>(),
                query_head_count,
                kv_head_count,
                row_count,
                static_cast<float>(refinement_tolerance));
        plain_gqa4_valuesketch_refine32_kernel<scalar_t><<<
            blocks, 256, 0, stream>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                bit_allocations.data_ptr<int8_t>(),
                code_offsets.data_ptr<int16_t>(),
                scale_offsets.data_ptr<int8_t>(),
                code_bases.data_ptr<int64_t>(),
                scale_bases.data_ptr<int64_t>(),
                code_strides.data_ptr<int16_t>(),
                scale_strides.data_ptr<int8_t>(),
                packed_value_codes.data_ptr<uint8_t>(),
                value_minimum.data_ptr<scalar_t>(),
                value_scale.data_ptr<scalar_t>(),
                thresholds.data_ptr<float>(),
                refinement_flags.data_ptr<bool>(),
                tail_coefficients.data_ptr<float>(),
                kv_head_count,
                static_cast<int>(history_count),
                value_token_stride,
                static_cast<int>(value_block_size),
                value_block_count,
                static_cast<float>(attention_scale));
      });
  finalize_counts_kernel<<<
      (row_count + 255) / 256, 256, 0, stream>>>(
          candidate_counts.data_ptr<int64_t>(),
          overflow.data_ptr<bool>(),
          row_count,
          candidate_capacity);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void mixedblock_sampled_compact_gqa4_indices_out_cuda(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    torch::Tensor block_hot_prefix,
    torch::Tensor head_code_bases,
    torch::Tensor head_scale_bases,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor overflow,
    int64_t history_count,
    int64_t block_size,
    int64_t sample_count,
    double selected_fraction) {
  TORCH_CHECK(query_codes.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(query_codes.scalar_type() == at::kChar, "query codes must be int8");
  TORCH_CHECK(packed_codes.scalar_type() == at::kByte, "codes must be uint8");
  TORCH_CHECK(bit_allocations.scalar_type() == at::kChar, "allocations must be int8");
  TORCH_CHECK(block_hot_prefix.scalar_type() == at::kShort,
              "hot prefix must be int16");
  TORCH_CHECK(head_code_bases.scalar_type() == at::kLong,
              "bases must be int64");
  TORCH_CHECK(query_codes.size(2) == 4,
              "GQA4 kernel requires exactly four Query groups");
  TORCH_CHECK(sample_count > 0 && sample_count <= MIXEDBLOCK_MAX_SAMPLE_COUNT,
              "invalid sample count");
  TORCH_CHECK(block_size > 0, "block size must be positive");
  int batch_count = static_cast<int>(query_codes.size(0));
  int kv_head_count = static_cast<int>(query_codes.size(1));
  constexpr int query_groups = 4;
  int query_head_count = kv_head_count * query_groups;
  int batch_kv_count = batch_count * kv_head_count;
  int row_count = batch_count * query_head_count;
  int block_count =
      static_cast<int>(block_hot_prefix.size(1)) - 1;
  int candidate_capacity = static_cast<int>(candidate_indices.size(2));
  int selected_keep = std::max(
      1, static_cast<int>(ceil(selected_fraction * sample_count)));
  c10::cuda::CUDAGuard device_guard(query_codes.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(
      candidate_counts.data_ptr<int64_t>(), 0,
      candidate_counts.numel() * sizeof(int64_t), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      overflow.data_ptr<bool>(), 0,
      overflow.numel() * sizeof(bool), stream));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_scales.scalar_type(),
      "mixedblock_sampled_compact_gqa4_indices_out",
      [&] {
        mixedblock_sample_threshold_kernel<scalar_t><<<
            row_count, 256,
            mixedblock_sample_shared_bytes(sample_count), stream>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                bit_allocations.data_ptr<int8_t>(),
                code_offsets.data_ptr<int16_t>(),
                scale_offsets.data_ptr<int8_t>(),
                code_strides.data_ptr<int16_t>(),
                scale_strides.data_ptr<int8_t>(),
                block_hot_prefix.data_ptr<int16_t>(),
                head_code_bases.data_ptr<int64_t>(),
                head_scale_bases.data_ptr<int64_t>(),
                thresholds.data_ptr<float>(),
                kv_head_count, query_groups,
                static_cast<int>(history_count),
                static_cast<int>(block_size), block_count,
                static_cast<int>(sample_count), selected_keep);
        dim3 blocks(
            batch_kv_count,
            (history_count + 255) / 256);
        mixedblock_threshold_compact_gqa4_indices_kernel<scalar_t><<<
            blocks, 256, 0, stream>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                bit_allocations.data_ptr<int8_t>(),
                code_offsets.data_ptr<int16_t>(),
                scale_offsets.data_ptr<int8_t>(),
                code_strides.data_ptr<int16_t>(),
                scale_strides.data_ptr<int8_t>(),
                block_hot_prefix.data_ptr<int16_t>(),
                head_code_bases.data_ptr<int64_t>(),
                head_scale_bases.data_ptr<int64_t>(),
                thresholds.data_ptr<float>(),
                candidate_indices.data_ptr<int64_t>(),
                candidate_counts.data_ptr<int64_t>(),
                overflow.data_ptr<bool>(),
                kv_head_count,
                static_cast<int>(history_count),
                static_cast<int>(block_size), block_count,
                candidate_capacity, batch_kv_count);
      });
  finalize_counts_kernel<<<
      (row_count + 255) / 256, 256, 0, stream>>>(
          candidate_counts.data_ptr<int64_t>(),
          overflow.data_ptr<bool>(),
          row_count,
          candidate_capacity);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void mixedblock_sampled_fused_attention_gqa4_out_cuda(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    torch::Tensor block_hot_prefix,
    torch::Tensor head_code_bases,
    torch::Tensor head_scale_bases,
    torch::Tensor output,
    torch::Tensor partial_output,
    torch::Tensor partial_max,
    torch::Tensor partial_sum,
    torch::Tensor partial_counts,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor overflow,
    int64_t history_count,
    int64_t block_size,
    int64_t sample_count,
    double selected_fraction,
    double scaling,
    int64_t split_count,
    int64_t max_local_candidates) {
  TORCH_CHECK(
      query_codes.is_cuda() && query_scales.is_cuda()
          && query.is_cuda() && key.is_cuda() && value.is_cuda(),
      "inputs must be CUDA tensors");
  TORCH_CHECK(query_codes.scalar_type() == at::kChar,
              "query codes must be int8");
  TORCH_CHECK(packed_codes.scalar_type() == at::kByte,
              "codes must be uint8");
  TORCH_CHECK(bit_allocations.scalar_type() == at::kChar,
              "allocations must be int8");
  TORCH_CHECK(block_hot_prefix.scalar_type() == at::kShort,
              "hot prefix must be int16");
  TORCH_CHECK(head_code_bases.scalar_type() == at::kLong,
              "bases must be int64");
  TORCH_CHECK(query_codes.size(2) == 4,
              "GQA4 kernel requires exactly four Query groups");
  TORCH_CHECK(
      query.scalar_type() == key.scalar_type()
          && query.scalar_type() == value.scalar_type()
          && query.scalar_type() == query_scales.scalar_type()
          && query.scalar_type() == key_scales.scalar_type(),
      "query, K/V, and scales must share a dtype");
  TORCH_CHECK(
      query.scalar_type() == at::kHalf
          || query.scalar_type() == at::kBFloat16,
      "fused attention supports FP16 and BF16");
  TORCH_CHECK(
      query.dim() == 3 && query.size(2) == 128,
      "query must have shape [batch, query_heads, 128]");
  TORCH_CHECK(
      key.dim() == 4 && key.size(3) == 128
          && value.sizes() == key.sizes(),
      "K/V must have matching [batch, kv_heads, tokens, 128] shapes");
  TORCH_CHECK(
      history_count > 0 && history_count < key.size(2),
      "history count must exclude one implicit self token");
  TORCH_CHECK(
      sample_count > 0
          && sample_count <= MIXEDBLOCK_MAX_SAMPLE_COUNT,
      "invalid sample count");
  TORCH_CHECK(block_size > 0, "block size must be positive");
  TORCH_CHECK(
      split_count >= 1 && split_count <= 64,
      "split count must be in [1, 64]");
  TORCH_CHECK(
      max_local_candidates > 0 && max_local_candidates <= 1400,
      "local candidate capacity must be in [1, 1400]");

  auto query_c = query.contiguous();
  int batch_count = static_cast<int>(query_codes.size(0));
  int kv_head_count = static_cast<int>(query_codes.size(1));
  constexpr int query_groups = 4;
  int query_head_count = kv_head_count * query_groups;
  int batch_kv_count = batch_count * kv_head_count;
  int row_count = batch_count * query_head_count;
  int block_count =
      static_cast<int>(block_hot_prefix.size(1)) - 1;
  int selected_keep = std::max(
      1, static_cast<int>(ceil(selected_fraction * sample_count)));
  TORCH_CHECK(
      query_c.size(0) == batch_count
          && query_c.size(1) == query_head_count,
      "query head layout does not match GQA4 codes");
  TORCH_CHECK(
      key.size(0) == batch_count
          && key.size(1) == kv_head_count,
      "K/V head layout does not match GQA4 codes");
  TORCH_CHECK(
      output.is_contiguous()
          && output.sizes() == query_c.sizes()
          && output.scalar_type() == query_c.scalar_type(),
      "output must be a contiguous query-shaped tensor");
  TORCH_CHECK(
      partial_output.scalar_type() == at::kFloat
          && partial_max.scalar_type() == at::kFloat
          && partial_sum.scalar_type() == at::kFloat,
      "partial attention buffers must be float32");
  TORCH_CHECK(
      partial_counts.scalar_type() == at::kInt,
      "partial counts must be int32");
  TORCH_CHECK(
      candidate_counts.scalar_type() == at::kLong
          && thresholds.scalar_type() == at::kFloat
          && overflow.scalar_type() == at::kBool,
      "count, threshold, and overflow dtypes are invalid");

  c10::cuda::CUDAGuard device_guard(query_c.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(
      overflow.data_ptr<bool>(), 0,
      overflow.numel() * sizeof(bool), stream));
  size_t fused_shared_bytes =
      query_groups * sizeof(int)
      + static_cast<size_t>(query_groups)
          * static_cast<size_t>(max_local_candidates)
          * (sizeof(int) + sizeof(float))
      + (256 + query_groups) * sizeof(float);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "mixedblock_sampled_fused_attention_gqa4_out",
      [&] {
        mixedblock_sample_threshold_kernel<scalar_t><<<
            row_count, 256,
            mixedblock_sample_shared_bytes(sample_count), stream>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                bit_allocations.data_ptr<int8_t>(),
                code_offsets.data_ptr<int16_t>(),
                scale_offsets.data_ptr<int8_t>(),
                code_strides.data_ptr<int16_t>(),
                scale_strides.data_ptr<int8_t>(),
                block_hot_prefix.data_ptr<int16_t>(),
                head_code_bases.data_ptr<int64_t>(),
                head_scale_bases.data_ptr<int64_t>(),
                thresholds.data_ptr<float>(),
                kv_head_count, query_groups,
                static_cast<int>(history_count),
                static_cast<int>(block_size), block_count,
                static_cast<int>(sample_count), selected_keep);
        mixedblock_fused_attention_gqa4_kernel<scalar_t><<<
            batch_kv_count * split_count,
            256,
            fused_shared_bytes,
            stream>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                query_c.data_ptr<scalar_t>(),
                key.data_ptr<scalar_t>(),
                value.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                bit_allocations.data_ptr<int8_t>(),
                code_offsets.data_ptr<int16_t>(),
                scale_offsets.data_ptr<int8_t>(),
                code_strides.data_ptr<int16_t>(),
                scale_strides.data_ptr<int8_t>(),
                block_hot_prefix.data_ptr<int16_t>(),
                head_code_bases.data_ptr<int64_t>(),
                head_scale_bases.data_ptr<int64_t>(),
                thresholds.data_ptr<float>(),
                partial_output.data_ptr<float>(),
                partial_max.data_ptr<float>(),
                partial_sum.data_ptr<float>(),
                partial_counts.data_ptr<int32_t>(),
                overflow.data_ptr<bool>(),
                kv_head_count,
                static_cast<int>(history_count),
                static_cast<int>(key.size(2)),
                static_cast<int>(block_size),
                block_count,
                static_cast<int>(split_count),
                static_cast<int>(max_local_candidates),
                batch_kv_count,
                key.stride(0),
                key.stride(1),
                key.stride(2),
                key.stride(3),
                value.stride(0),
                value.stride(1),
                value.stride(2),
                value.stride(3),
                static_cast<float>(scaling));
        mixedblock_reduce_fused_attention_kernel<scalar_t><<<
            row_count, 128, 0, stream>>>(
                partial_output.data_ptr<float>(),
                partial_max.data_ptr<float>(),
                partial_sum.data_ptr<float>(),
                partial_counts.data_ptr<int32_t>(),
                output.data_ptr<scalar_t>(),
                candidate_counts.data_ptr<int64_t>(),
                static_cast<int>(split_count));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sortedblock_sampled_compact_out_cuda(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    torch::Tensor head_code_bases,
    torch::Tensor head_scale_bases,
    torch::Tensor original_blocks,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_scores,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor overflow,
    int64_t history_count,
    int64_t block_size,
    int64_t hot_block_count,
    int64_t sample_count,
    double selected_fraction) {
  TORCH_CHECK(query_codes.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(query_codes.scalar_type() == at::kChar, "query codes must be int8");
  TORCH_CHECK(packed_codes.scalar_type() == at::kByte, "codes must be uint8");
  TORCH_CHECK(bit_allocations.scalar_type() == at::kChar, "allocations must be int8");
  TORCH_CHECK(head_code_bases.scalar_type() == at::kLong, "bases must be int64");
  TORCH_CHECK(original_blocks.scalar_type() == at::kInt,
              "original block IDs must be int32");
  TORCH_CHECK(sample_count > 0 && sample_count <= MIXEDBLOCK_MAX_SAMPLE_COUNT,
              "invalid sample count");
  TORCH_CHECK(block_size > 0, "block size must be positive");
  int batch_count = static_cast<int>(query_codes.size(0));
  int kv_head_count = static_cast<int>(query_codes.size(1));
  int query_groups = static_cast<int>(query_codes.size(2));
  int query_head_count = kv_head_count * query_groups;
  int batch_kv_count = batch_count * kv_head_count;
  int row_count = batch_count * query_head_count;
  int block_count = static_cast<int>(original_blocks.size(1));
  int candidate_capacity = static_cast<int>(candidate_indices.size(2));
  int selected_keep = std::max(
      1, static_cast<int>(ceil(selected_fraction * sample_count)));
  c10::cuda::CUDAGuard device_guard(query_codes.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(
      candidate_counts.data_ptr<int64_t>(), 0,
      candidate_counts.numel() * sizeof(int64_t), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      overflow.data_ptr<bool>(), 0,
      overflow.numel() * sizeof(bool), stream));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_scales.scalar_type(),
      "sortedblock_sampled_compact_out",
      [&] {
        sortedblock_sample_threshold_kernel<scalar_t><<<
            row_count, 256,
            mixedblock_sample_shared_bytes(sample_count), stream>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                bit_allocations.data_ptr<int8_t>(),
                code_offsets.data_ptr<int16_t>(),
                scale_offsets.data_ptr<int8_t>(),
                code_strides.data_ptr<int16_t>(),
                scale_strides.data_ptr<int8_t>(),
                head_code_bases.data_ptr<int64_t>(),
                head_scale_bases.data_ptr<int64_t>(),
                thresholds.data_ptr<float>(),
                kv_head_count, query_groups,
                static_cast<int>(history_count),
                static_cast<int>(block_size),
                static_cast<int>(hot_block_count),
                static_cast<int>(sample_count), selected_keep);
        dim3 blocks(
            batch_kv_count,
            (history_count + 255) / 256);
        sortedblock_threshold_compact_kernel<scalar_t><<<
            blocks, 256, 0, stream>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                bit_allocations.data_ptr<int8_t>(),
                code_offsets.data_ptr<int16_t>(),
                scale_offsets.data_ptr<int8_t>(),
                code_strides.data_ptr<int16_t>(),
                scale_strides.data_ptr<int8_t>(),
                head_code_bases.data_ptr<int64_t>(),
                head_scale_bases.data_ptr<int64_t>(),
                original_blocks.data_ptr<int32_t>(),
                thresholds.data_ptr<float>(),
                candidate_indices.data_ptr<int64_t>(),
                candidate_scores.data_ptr<float>(),
                candidate_counts.data_ptr<int64_t>(),
                overflow.data_ptr<bool>(),
                kv_head_count, query_groups,
                static_cast<int>(history_count),
                static_cast<int>(block_size), block_count,
                static_cast<int>(hot_block_count),
                candidate_capacity);
      });
  finalize_counts_kernel<<<
      (row_count + 255) / 256, 256, 0, stream>>>(
          candidate_counts.data_ptr<int64_t>(),
          overflow.data_ptr<bool>(),
          row_count,
          candidate_capacity);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


@lru_cache(maxsize=1)
def load_extension() -> object:
    return load_inline(
        name="qksieve_mixedblock_spectral_20260809_v46_mha_valuesketch_ab",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=None,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )


def sampled_threshold_compact_out(
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    packed_codes: torch.Tensor,
    key_scales: torch.Tensor,
    metadata: dict[str, torch.Tensor | int],
    candidate_indices: torch.Tensor,
    candidate_scores: torch.Tensor,
    candidate_counts: torch.Tensor,
    thresholds: torch.Tensor,
    overflow: torch.Tensor,
    history_count: int,
    sample_count: int,
    selected_fraction: float,
) -> tuple[torch.Tensor, ...]:
    load_extension().sampled_compact_out(
        query_codes,
        query_scales,
        packed_codes,
        key_scales,
        metadata["bit_allocations"],
        metadata["code_offsets"],
        metadata["scale_offsets"],
        metadata["code_strides"],
        metadata["scale_strides"],
        metadata["block_hot_prefix"],
        metadata["head_code_bases"],
        metadata["head_scale_bases"],
        candidate_indices,
        candidate_scores,
        candidate_counts,
        thresholds,
        overflow,
        history_count,
        int(metadata["block_size"]),
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


def sampled_threshold_compact_gqa4_indices_out(
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    packed_codes: torch.Tensor,
    key_scales: torch.Tensor,
    metadata: dict[str, torch.Tensor | int],
    candidate_indices: torch.Tensor,
    candidate_counts: torch.Tensor,
    thresholds: torch.Tensor,
    overflow: torch.Tensor,
    history_count: int,
    sample_count: int,
    selected_fraction: float,
) -> tuple[torch.Tensor, ...]:
    load_extension().sampled_compact_gqa4_indices_out(
        query_codes,
        query_scales,
        packed_codes,
        key_scales,
        metadata["bit_allocations"],
        metadata["code_offsets"],
        metadata["scale_offsets"],
        metadata["code_strides"],
        metadata["scale_strides"],
        metadata["block_hot_prefix"],
        metadata["head_code_bases"],
        metadata["head_scale_bases"],
        candidate_indices,
        candidate_counts,
        thresholds,
        overflow,
        history_count,
        int(metadata["block_size"]),
        sample_count,
        selected_fraction,
    )
    return candidate_indices, candidate_counts, thresholds, overflow


def plain_sampled_threshold_compact_gqa4_indices_out(
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    packed_index: dict[str, torch.Tensor | int],
    candidate_indices: torch.Tensor,
    candidate_counts: torch.Tensor,
    thresholds: torch.Tensor,
    overflow: torch.Tensor,
    history_count: int,
    sample_count: int,
    selected_fraction: float,
) -> tuple[torch.Tensor, ...]:
    load_extension().plain_sampled_compact_gqa4_indices_out(
        query_codes,
        query_scales,
        packed_index["packed_codes"],
        packed_index["key_scales"],
        packed_index["bit_allocations"],
        packed_index["code_offsets"],
        packed_index["scale_offsets"],
        packed_index["code_bases"],
        packed_index["scale_bases"],
        packed_index["code_strides"],
        packed_index["scale_strides"],
        candidate_indices,
        candidate_counts,
        thresholds,
        overflow,
        history_count,
        sample_count,
        selected_fraction,
    )
    return candidate_indices, candidate_counts, thresholds, overflow


def plain_sampled_threshold_compact_gqa4_mass_out(
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    packed_index: dict[str, torch.Tensor | int],
    candidate_indices: torch.Tensor,
    candidate_counts: torch.Tensor,
    thresholds: torch.Tensor,
    overflow: torch.Tensor,
    selected_denominator: torch.Tensor,
    tail_denominator: torch.Tensor,
    history_count: int,
    sample_count: int,
    selected_fraction: float,
    attention_scale: float,
) -> tuple[torch.Tensor, ...]:
    load_extension().plain_sampled_compact_gqa4_mass_out(
        query_codes,
        query_scales,
        packed_index["packed_codes"],
        packed_index["key_scales"],
        packed_index["bit_allocations"],
        packed_index["code_offsets"],
        packed_index["scale_offsets"],
        packed_index["code_bases"],
        packed_index["scale_bases"],
        packed_index["code_strides"],
        packed_index["scale_strides"],
        candidate_indices,
        candidate_counts,
        thresholds,
        overflow,
        selected_denominator,
        tail_denominator,
        history_count,
        sample_count,
        selected_fraction,
        attention_scale,
    )
    return (
        candidate_indices,
        candidate_counts,
        thresholds,
        overflow,
        selected_denominator,
        tail_denominator,
    )


def plain_mass_ladder_thresholds_out(
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    packed_index: dict[str, torch.Tensor | int],
    ladder_thresholds: torch.Tensor,
    mass_bins: torch.Tensor,
    chosen_thresholds: torch.Tensor,
    chosen_rungs: torch.Tensor,
    chosen_mass: torch.Tensor,
    history_count: int,
    sample_count: int,
    minimum_fraction: float,
    growth: float,
    target_mass: float,
    attention_scale: float,
) -> tuple[torch.Tensor, ...]:
    rung_count = int(ladder_thresholds.shape[0])
    load_extension().plain_mass_ladder_thresholds_out(
        query_codes,
        query_scales,
        packed_index["packed_codes"],
        packed_index["key_scales"],
        packed_index["bit_allocations"],
        packed_index["code_offsets"],
        packed_index["scale_offsets"],
        packed_index["code_bases"],
        packed_index["scale_bases"],
        packed_index["code_strides"],
        packed_index["scale_strides"],
        ladder_thresholds,
        mass_bins,
        chosen_thresholds,
        chosen_rungs,
        chosen_mass,
        history_count,
        sample_count,
        minimum_fraction,
        growth,
        rung_count,
        target_mass,
        attention_scale,
    )
    return (
        ladder_thresholds,
        mass_bins,
        chosen_thresholds,
        chosen_rungs,
        chosen_mass,
    )


def plain_sampled_threshold_compact_gqa4_condtail_out(
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    packed_index: dict[str, torch.Tensor | int],
    candidate_indices: torch.Tensor,
    candidate_counts: torch.Tensor,
    thresholds: torch.Tensor,
    overflow: torch.Tensor,
    selected_denominator: torch.Tensor,
    tail_denominator: torch.Tensor,
    tail_block_denominator: torch.Tensor,
    tail_weighted_x: torch.Tensor,
    history_count: int,
    sample_count: int,
    selected_fraction: float,
    moment_block_size: int,
    attention_scale: float,
) -> tuple[torch.Tensor, ...]:
    load_extension().plain_sampled_compact_gqa4_condtail_out(
        query_codes,
        query_scales,
        packed_index["packed_codes"],
        packed_index["key_scales"],
        packed_index["bit_allocations"],
        packed_index["code_offsets"],
        packed_index["scale_offsets"],
        packed_index["code_bases"],
        packed_index["scale_bases"],
        packed_index["code_strides"],
        packed_index["scale_strides"],
        candidate_indices,
        candidate_counts,
        thresholds,
        overflow,
        selected_denominator,
        tail_denominator,
        tail_block_denominator,
        tail_weighted_x,
        history_count,
        sample_count,
        selected_fraction,
        moment_block_size,
        attention_scale,
    )
    return (
        candidate_indices,
        candidate_counts,
        thresholds,
        overflow,
        selected_denominator,
        tail_denominator,
        tail_block_denominator,
        tail_weighted_x,
    )


def plain_sampled_threshold_compact_gqa4_valuesketch_out(
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    packed_index: dict[str, torch.Tensor | int],
    packed_value_codes: torch.Tensor,
    value_minimum: torch.Tensor,
    value_scale: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_counts: torch.Tensor,
    thresholds: torch.Tensor,
    overflow: torch.Tensor,
    selected_denominator: torch.Tensor,
    tail_denominator: torch.Tensor,
    tail_coefficients: torch.Tensor,
    history_count: int,
    sample_count: int,
    selected_fraction: float,
    value_block_size: int,
    attention_scale: float,
) -> tuple[torch.Tensor, ...]:
    value_rank = int(tail_coefficients.shape[-1])
    load_extension().plain_sampled_compact_gqa4_valuesketch_out(
        query_codes,
        query_scales,
        packed_index["packed_codes"],
        packed_index["key_scales"],
        packed_index["bit_allocations"],
        packed_index["code_offsets"],
        packed_index["scale_offsets"],
        packed_index["code_bases"],
        packed_index["scale_bases"],
        packed_index["code_strides"],
        packed_index["scale_strides"],
        packed_value_codes,
        value_minimum,
        value_scale,
        candidate_indices,
        candidate_counts,
        thresholds,
        overflow,
        selected_denominator,
        tail_denominator,
        tail_coefficients,
        history_count,
        sample_count,
        selected_fraction,
        value_rank,
        value_block_size,
        attention_scale,
    )
    return (
        candidate_indices,
        candidate_counts,
        thresholds,
        overflow,
        selected_denominator,
        tail_denominator,
        tail_coefficients,
    )


def plain_sampled_threshold_compact_gqa4_valuesketch_deterministic_out(
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    packed_index: dict[str, torch.Tensor | int],
    packed_value_codes: torch.Tensor,
    value_minimum: torch.Tensor,
    value_scale: torch.Tensor,
    selection_masks: torch.Tensor,
    tail_partials: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_counts: torch.Tensor,
    thresholds: torch.Tensor,
    overflow: torch.Tensor,
    selected_denominator: torch.Tensor,
    tail_denominator: torch.Tensor,
    tail_coefficients: torch.Tensor,
    history_count: int,
    sample_count: int,
    selected_fraction: float,
    value_block_size: int,
    attention_scale: float,
) -> tuple[torch.Tensor, ...]:
    value_rank = int(tail_coefficients.shape[-1])
    load_extension().plain_sampled_compact_gqa4_valuesketch_deterministic_out(
        query_codes,
        query_scales,
        packed_index["packed_codes"],
        packed_index["key_scales"],
        packed_index["bit_allocations"],
        packed_index["code_offsets"],
        packed_index["scale_offsets"],
        packed_index["code_bases"],
        packed_index["scale_bases"],
        packed_index["code_strides"],
        packed_index["scale_strides"],
        packed_value_codes,
        value_minimum,
        value_scale,
        selection_masks,
        tail_partials,
        candidate_indices,
        candidate_counts,
        thresholds,
        overflow,
        selected_denominator,
        tail_denominator,
        tail_coefficients,
        history_count,
        sample_count,
        selected_fraction,
        value_rank,
        value_block_size,
        attention_scale,
    )
    return (
        candidate_indices,
        candidate_counts,
        thresholds,
        overflow,
        selected_denominator,
        tail_denominator,
        tail_coefficients,
    )


def plain_threshold_compact_gqa4_valuesketch_out(
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    packed_index: dict[str, torch.Tensor | int],
    packed_value_codes: torch.Tensor,
    value_minimum: torch.Tensor,
    value_scale: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_counts: torch.Tensor,
    thresholds: torch.Tensor,
    overflow: torch.Tensor,
    selected_denominator: torch.Tensor,
    tail_denominator: torch.Tensor,
    tail_coefficients: torch.Tensor,
    history_count: int,
    value_block_size: int,
    attention_scale: float,
) -> tuple[torch.Tensor, ...]:
    """Compact candidates using externally measured per-head thresholds."""
    return plain_sampled_threshold_compact_gqa4_valuesketch_out(
        query_codes,
        query_scales,
        packed_index,
        packed_value_codes,
        value_minimum,
        value_scale,
        candidate_indices,
        candidate_counts,
        thresholds,
        overflow,
        selected_denominator,
        tail_denominator,
        tail_coefficients,
        history_count,
        0,
        0.0,
        value_block_size,
        attention_scale,
    )


def plain_sampled_threshold_compact_gqa4_valuesketch_progressive_out(
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    packed_index: dict[str, torch.Tensor | int],
    packed_value_codes: torch.Tensor,
    value_minimum: torch.Tensor,
    value_scale: torch.Tensor,
    value_rank8_residual: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_counts: torch.Tensor,
    thresholds: torch.Tensor,
    overflow: torch.Tensor,
    selected_denominator: torch.Tensor,
    tail_denominator: torch.Tensor,
    tail_coefficients: torch.Tensor,
    refinement_flags: torch.Tensor,
    history_count: int,
    sample_count: int,
    selected_fraction: float,
    value_block_size: int,
    attention_scale: float,
    refinement_tolerance: float,
) -> tuple[torch.Tensor, ...]:
    """Run rank 8 for every head and refine certified-risk heads to rank 32."""
    if tail_coefficients.shape[-1] != 32:
        raise ValueError("progressive Value-sketch storage must have rank 32")
    load_extension().plain_sampled_compact_gqa4_valuesketch_progressive_out(
        query_codes,
        query_scales,
        packed_index["packed_codes"],
        packed_index["key_scales"],
        packed_index["bit_allocations"],
        packed_index["code_offsets"],
        packed_index["scale_offsets"],
        packed_index["code_bases"],
        packed_index["scale_bases"],
        packed_index["code_strides"],
        packed_index["scale_strides"],
        packed_value_codes,
        value_minimum,
        value_scale,
        value_rank8_residual.float().contiguous(),
        candidate_indices,
        candidate_counts,
        thresholds,
        overflow,
        selected_denominator,
        tail_denominator,
        tail_coefficients,
        refinement_flags,
        history_count,
        sample_count,
        selected_fraction,
        value_block_size,
        attention_scale,
        refinement_tolerance,
    )
    return (
        candidate_indices,
        candidate_counts,
        thresholds,
        overflow,
        selected_denominator,
        tail_denominator,
        tail_coefficients,
        refinement_flags,
    )


def allocate_fused_attention_workspace(
    query: torch.Tensor,
    split_count: int,
) -> dict[str, torch.Tensor]:
    if query.ndim != 3 or query.shape[-1] != 128:
        raise ValueError("query must have shape [batch, query_heads, 128]")
    if not 1 <= split_count <= 64:
        raise ValueError("split_count must be in [1, 64]")
    batch_count, query_head_count, head_dim = query.shape
    partial_shape = (
        batch_count,
        query_head_count,
        split_count,
    )
    return {
        "output": torch.empty_like(query),
        "partial_output": torch.empty(
            (*partial_shape, head_dim),
            dtype=torch.float32,
            device=query.device,
        ),
        "partial_max": torch.empty(
            partial_shape,
            dtype=torch.float32,
            device=query.device,
        ),
        "partial_sum": torch.empty(
            partial_shape,
            dtype=torch.float32,
            device=query.device,
        ),
        "partial_counts": torch.empty(
            partial_shape,
            dtype=torch.int32,
            device=query.device,
        ),
        "candidate_counts": torch.empty(
            batch_count,
            query_head_count,
            dtype=torch.long,
            device=query.device,
        ),
        "thresholds": torch.empty(
            batch_count,
            query_head_count,
            dtype=torch.float32,
            device=query.device,
        ),
        "overflow": torch.empty(
            batch_count,
            query_head_count,
            dtype=torch.bool,
            device=query.device,
        ),
    }


def sampled_threshold_fused_attention_gqa4_out(
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    packed_codes: torch.Tensor,
    key_scales: torch.Tensor,
    metadata: dict[str, torch.Tensor | int],
    workspace: dict[str, torch.Tensor],
    history_count: int,
    sample_count: int,
    selected_fraction: float,
    scaling: float,
    split_count: int,
    max_local_candidates: int = 1024,
) -> tuple[torch.Tensor, ...]:
    load_extension().sampled_fused_attention_gqa4_out(
        query_codes,
        query_scales,
        query,
        key,
        value,
        packed_codes,
        key_scales,
        metadata["bit_allocations"],
        metadata["code_offsets"],
        metadata["scale_offsets"],
        metadata["code_strides"],
        metadata["scale_strides"],
        metadata["block_hot_prefix"],
        metadata["head_code_bases"],
        metadata["head_scale_bases"],
        workspace["output"],
        workspace["partial_output"],
        workspace["partial_max"],
        workspace["partial_sum"],
        workspace["partial_counts"],
        workspace["candidate_counts"],
        workspace["thresholds"],
        workspace["overflow"],
        history_count,
        int(metadata["block_size"]),
        sample_count,
        selected_fraction,
        scaling,
        split_count,
        max_local_candidates,
    )
    return (
        workspace["output"],
        workspace["candidate_counts"],
        workspace["thresholds"],
        workspace["overflow"],
    )


def sortedblock_sampled_threshold_compact_out(
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    packed_codes: torch.Tensor,
    key_scales: torch.Tensor,
    metadata: dict[str, torch.Tensor | int],
    candidate_indices: torch.Tensor,
    candidate_scores: torch.Tensor,
    candidate_counts: torch.Tensor,
    thresholds: torch.Tensor,
    overflow: torch.Tensor,
    history_count: int,
    sample_count: int,
    selected_fraction: float,
) -> tuple[torch.Tensor, ...]:
    load_extension().sortedblock_sampled_compact_out(
        query_codes,
        query_scales,
        packed_codes,
        key_scales,
        metadata["bit_allocations"],
        metadata["code_offsets"],
        metadata["scale_offsets"],
        metadata["code_strides"],
        metadata["scale_strides"],
        metadata["head_code_bases"],
        metadata["head_scale_bases"],
        metadata["original_blocks"],
        candidate_indices,
        candidate_scores,
        candidate_counts,
        thresholds,
        overflow,
        history_count,
        int(metadata["block_size"]),
        int(metadata["hot_block_count"]),
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
