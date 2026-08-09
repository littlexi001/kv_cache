from __future__ import annotations

from functools import lru_cache

import torch
from torch.utils.cpp_extension import load_inline


CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor progressive_masked_scores_forward(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor band_mask,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    int64_t history_count);

torch::Tensor progressive_candidate_scores_forward(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor band_mask,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    torch::Tensor candidate_indices);

torch::Tensor progressive_token_masked_scores_forward(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor band_mask,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    torch::Tensor candidate_mask);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "progressive_masked_scores_forward",
      &progressive_masked_scores_forward,
      "Scan selected packed spectral bands");
  module.def(
      "progressive_candidate_scores_forward",
      &progressive_candidate_scores_forward,
      "Read selected packed bands only for candidate tokens");
  module.def(
      "progressive_token_masked_scores_forward",
      &progressive_token_masked_scores_forward,
      "Sequentially scan tokens and decode selected bands for candidates");
}
"""


CUDA_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename integer_t>
__device__ __forceinline__ int band_dot(
    const uint8_t* __restrict__ packed,
    const int8_t* __restrict__ query,
    int bits) {
  int total = 0;
  if (bits == 8) {
#pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
      const uint8_t* code = packed + chunk * 4;
      int8_t value0 = static_cast<int8_t>(code[0]);
      int8_t value1 = static_cast<int8_t>(code[1]);
      int8_t value2 = static_cast<int8_t>(code[2]);
      int8_t value3 = static_cast<int8_t>(code[3]);
      total += static_cast<int>(value0) * static_cast<int>(query[4 * chunk]);
      total += static_cast<int>(value1) * static_cast<int>(query[4 * chunk + 1]);
      total += static_cast<int>(value2) * static_cast<int>(query[4 * chunk + 2]);
      total += static_cast<int>(value3) * static_cast<int>(query[4 * chunk + 3]);
    }
  } else if (bits == 4) {
#pragma unroll
    for (int chunk = 0; chunk < 8; ++chunk) {
      uint8_t byte = packed[chunk];
      int low = static_cast<int>(byte & 0xf);
      int high = static_cast<int>((byte >> 4) & 0xf);
      low = low >= 8 ? low - 16 : low;
      high = high >= 8 ? high - 16 : high;
      total += low * static_cast<int>(query[2 * chunk]);
      total += high * static_cast<int>(query[2 * chunk + 1]);
    }
  } else if (bits == 2) {
#pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
      uint8_t byte = packed[chunk];
#pragma unroll
      for (int lane = 0; lane < 4; ++lane) {
        int value = static_cast<int>((byte >> (2 * lane)) & 0x3);
        value = value >= 2 ? value - 4 : value;
        total += value * static_cast<int>(query[4 * chunk + lane]);
      }
    }
  } else if (bits == 1) {
#pragma unroll
    for (int chunk = 0; chunk < 2; ++chunk) {
      uint8_t byte = packed[chunk];
#pragma unroll
      for (int lane = 0; lane < 8; ++lane) {
        int value = ((byte >> lane) & 1) != 0 ? 1 : -1;
        total += value * static_cast<int>(query[8 * chunk + lane]);
      }
    }
  }
  return total;
}

template <typename scale_t>
__device__ __forceinline__ float masked_score_one(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int8_t* __restrict__ band_mask,
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
  const scale_t* query_scale = query_scales + query_row * 8;
  const uint8_t* token_codes = packed_codes
      + code_bases[batch_kv]
      + static_cast<int64_t>(token) * code_strides[batch_kv];
  const scale_t* token_scales = key_scales
      + scale_bases[batch_kv]
      + static_cast<int64_t>(token) * scale_strides[batch_kv];
  const int8_t* allocations = bit_allocations + batch_kv * 8;
  const int8_t* mask = band_mask + batch_kv * 8;
  const int16_t* offsets = code_offsets + batch_kv * 8;
  const int8_t* scale_index = scale_offsets + batch_kv * 8;
  float score = 0.0f;
#pragma unroll
  for (int band = 0; band < 8; ++band) {
    int bits = static_cast<int>(allocations[band]);
    if (bits == 0 || mask[band] == 0) {
      continue;
    }
    int dot = band_dot<int>(
        token_codes + offsets[band],
        query + 16 * band,
        bits);
    score += static_cast<float>(dot)
        * static_cast<float>(query_scale[band])
        * static_cast<float>(token_scales[scale_index[band]]);
  }
  return score;
}

template <typename scale_t>
__global__ void masked_scores_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int8_t* __restrict__ band_mask,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int64_t* __restrict__ code_bases,
    const int64_t* __restrict__ scale_bases,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    float* __restrict__ output,
    int kv_head_count,
    int query_groups,
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
    float score = masked_score_one(
        query_codes,
        query_scales,
        packed_codes,
        key_scales,
        bit_allocations,
        band_mask,
        code_offsets,
        scale_offsets,
        code_bases,
        scale_bases,
        code_strides,
        scale_strides,
        batch_kv,
        query_group,
        query_groups,
        token);
    int query_head = kv_head * query_groups + query_group;
    output[
        (static_cast<int64_t>(batch) * query_head_count + query_head)
            * history_count
        + token] = score;
  }
}

template <typename scale_t>
__global__ void candidate_scores_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int8_t* __restrict__ band_mask,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int64_t* __restrict__ code_bases,
    const int64_t* __restrict__ scale_bases,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    const int64_t* __restrict__ candidate_indices,
    float* __restrict__ output,
    int batch_count,
    int kv_head_count,
    int query_groups,
    int candidate_count) {
  int flat = blockIdx.x * blockDim.x + threadIdx.x;
  int query_head_count = kv_head_count * query_groups;
  int total = batch_count * query_head_count * candidate_count;
  if (flat >= total) {
    return;
  }
  int candidate_offset = flat % candidate_count;
  int row = flat / candidate_count;
  int batch = row / query_head_count;
  int query_head = row - batch * query_head_count;
  int kv_head = query_head / query_groups;
  int query_group = query_head - kv_head * query_groups;
  int batch_kv = batch * kv_head_count + kv_head;
  int token = static_cast<int>(candidate_indices[flat]);
  output[flat] = masked_score_one(
      query_codes,
      query_scales,
      packed_codes,
      key_scales,
      bit_allocations,
      band_mask,
      code_offsets,
      scale_offsets,
      code_bases,
      scale_bases,
      code_strides,
      scale_strides,
      batch_kv,
      query_group,
      query_groups,
      token);
}

template <typename scale_t>
__global__ void token_masked_scores_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int8_t* __restrict__ band_mask,
    const int16_t* __restrict__ code_offsets,
    const int8_t* __restrict__ scale_offsets,
    const int64_t* __restrict__ code_bases,
    const int64_t* __restrict__ scale_bases,
    const int16_t* __restrict__ code_strides,
    const int8_t* __restrict__ scale_strides,
    const uint8_t* __restrict__ candidate_mask,
    float* __restrict__ output,
    int kv_head_count,
    int query_groups,
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
    int query_head = kv_head * query_groups + query_group;
    int64_t output_offset =
        (static_cast<int64_t>(batch) * query_head_count + query_head)
            * history_count
        + token;
    output[output_offset] =
        candidate_mask[output_offset] != 0
        ? masked_score_one(
              query_codes,
              query_scales,
              packed_codes,
              key_scales,
              bit_allocations,
              band_mask,
              code_offsets,
              scale_offsets,
              code_bases,
              scale_bases,
              code_strides,
              scale_strides,
              batch_kv,
              query_group,
              query_groups,
              token)
        : 0.0f;
  }
}

void check_common(
    const torch::Tensor& query_codes,
    const torch::Tensor& query_scales,
    const torch::Tensor& packed_codes,
    const torch::Tensor& key_scales,
    const torch::Tensor& bit_allocations,
    const torch::Tensor& band_mask,
    const torch::Tensor& code_offsets,
    const torch::Tensor& scale_offsets,
    const torch::Tensor& code_bases,
    const torch::Tensor& scale_bases,
    const torch::Tensor& code_strides,
    const torch::Tensor& scale_strides) {
  TORCH_CHECK(
      query_codes.is_cuda() && query_scales.is_cuda()
          && packed_codes.is_cuda() && key_scales.is_cuda()
          && bit_allocations.is_cuda() && band_mask.is_cuda()
          && code_offsets.is_cuda() && scale_offsets.is_cuda()
          && code_bases.is_cuda() && scale_bases.is_cuda()
          && code_strides.is_cuda() && scale_strides.is_cuda(),
      "all progressive inputs must be CUDA tensors");
  TORCH_CHECK(
      query_codes.scalar_type() == at::kChar
          && bit_allocations.scalar_type() == at::kChar
          && band_mask.scalar_type() == at::kChar
          && scale_offsets.scalar_type() == at::kChar
          && scale_strides.scalar_type() == at::kChar,
      "progressive int8 metadata dtypes are invalid");
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
      "query and key scales must have the same dtype");
  TORCH_CHECK(
      query_codes.dim() == 4 && query_codes.size(3) == 128,
      "query codes must have shape [batch, kv_heads, groups, 128]");
  TORCH_CHECK(
      query_scales.dim() == 4 && query_scales.size(3) == 8,
      "query scales must have shape [batch, kv_heads, groups, 8]");
  TORCH_CHECK(
      bit_allocations.dim() == 3 && bit_allocations.size(2) == 8
          && band_mask.sizes() == bit_allocations.sizes(),
      "allocations and masks must have shape [batch, kv_heads, 8]");
  TORCH_CHECK(
      code_offsets.sizes() == bit_allocations.sizes()
          && scale_offsets.sizes() == bit_allocations.sizes(),
      "band metadata shapes must match allocations");
  TORCH_CHECK(
      code_bases.sizes() == code_strides.sizes()
          && scale_bases.sizes() == scale_strides.sizes()
          && code_bases.dim() == 2,
      "head metadata must have shape [batch, kv_heads]");
}

torch::Tensor progressive_masked_scores_forward(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor band_mask,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    int64_t history_count) {
  check_common(
      query_codes, query_scales, packed_codes, key_scales,
      bit_allocations, band_mask, code_offsets, scale_offsets,
      code_bases, scale_bases, code_strides, scale_strides);
  TORCH_CHECK(history_count > 0, "history count must be positive");
  int batch_count = static_cast<int>(query_codes.size(0));
  int kv_head_count = static_cast<int>(query_codes.size(1));
  int query_groups = static_cast<int>(query_codes.size(2));
  auto output = torch::empty(
      {batch_count, kv_head_count * query_groups, history_count},
      query_codes.options().dtype(at::kFloat));
  c10::cuda::CUDAGuard device_guard(query_codes.device());
  dim3 blocks(
      batch_count * kv_head_count,
      (static_cast<int>(history_count) + 255) / 256);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_scales.scalar_type(),
      "progressive_masked_scores_forward",
      [&] {
        masked_scores_kernel<scalar_t><<<
            blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                bit_allocations.data_ptr<int8_t>(),
                band_mask.data_ptr<int8_t>(),
                code_offsets.data_ptr<int16_t>(),
                scale_offsets.data_ptr<int8_t>(),
                code_bases.data_ptr<int64_t>(),
                scale_bases.data_ptr<int64_t>(),
                code_strides.data_ptr<int16_t>(),
                scale_strides.data_ptr<int8_t>(),
                output.data_ptr<float>(),
                kv_head_count,
                query_groups,
                static_cast<int>(history_count));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor progressive_candidate_scores_forward(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor band_mask,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    torch::Tensor candidate_indices) {
  check_common(
      query_codes, query_scales, packed_codes, key_scales,
      bit_allocations, band_mask, code_offsets, scale_offsets,
      code_bases, scale_bases, code_strides, scale_strides);
  TORCH_CHECK(
      candidate_indices.is_cuda()
          && candidate_indices.scalar_type() == at::kLong
          && candidate_indices.dim() == 3,
      "candidate indices must be CUDA int64 [batch, query_heads, count]");
  int batch_count = static_cast<int>(query_codes.size(0));
  int kv_head_count = static_cast<int>(query_codes.size(1));
  int query_groups = static_cast<int>(query_codes.size(2));
  int query_head_count = kv_head_count * query_groups;
  TORCH_CHECK(
      candidate_indices.size(0) == batch_count
          && candidate_indices.size(1) == query_head_count
          && candidate_indices.size(2) > 0,
      "candidate index shape does not match query heads");
  auto output = torch::empty(
      candidate_indices.sizes(),
      query_codes.options().dtype(at::kFloat));
  int total = static_cast<int>(candidate_indices.numel());
  c10::cuda::CUDAGuard device_guard(query_codes.device());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_scales.scalar_type(),
      "progressive_candidate_scores_forward",
      [&] {
        candidate_scores_kernel<scalar_t><<<
            (total + 255) / 256,
            256,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                bit_allocations.data_ptr<int8_t>(),
                band_mask.data_ptr<int8_t>(),
                code_offsets.data_ptr<int16_t>(),
                scale_offsets.data_ptr<int8_t>(),
                code_bases.data_ptr<int64_t>(),
                scale_bases.data_ptr<int64_t>(),
                code_strides.data_ptr<int16_t>(),
                scale_strides.data_ptr<int8_t>(),
                candidate_indices.data_ptr<int64_t>(),
                output.data_ptr<float>(),
                batch_count,
                kv_head_count,
                query_groups,
                static_cast<int>(candidate_indices.size(2)));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor progressive_token_masked_scores_forward(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor band_mask,
    torch::Tensor code_offsets,
    torch::Tensor scale_offsets,
    torch::Tensor code_bases,
    torch::Tensor scale_bases,
    torch::Tensor code_strides,
    torch::Tensor scale_strides,
    torch::Tensor candidate_mask) {
  check_common(
      query_codes, query_scales, packed_codes, key_scales,
      bit_allocations, band_mask, code_offsets, scale_offsets,
      code_bases, scale_bases, code_strides, scale_strides);
  TORCH_CHECK(
      candidate_mask.is_cuda()
          && candidate_mask.scalar_type() == at::kByte
          && candidate_mask.dim() == 3,
      "candidate mask must be CUDA uint8 [batch, query_heads, history]");
  int batch_count = static_cast<int>(query_codes.size(0));
  int kv_head_count = static_cast<int>(query_codes.size(1));
  int query_groups = static_cast<int>(query_codes.size(2));
  int query_head_count = kv_head_count * query_groups;
  int history_count = static_cast<int>(candidate_mask.size(2));
  TORCH_CHECK(
      candidate_mask.size(0) == batch_count
          && candidate_mask.size(1) == query_head_count
          && history_count > 0,
      "candidate mask shape does not match query heads");
  auto output = torch::empty(
      candidate_mask.sizes(),
      query_codes.options().dtype(at::kFloat));
  c10::cuda::CUDAGuard device_guard(query_codes.device());
  dim3 blocks(
      batch_count * kv_head_count,
      (history_count + 255) / 256);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_scales.scalar_type(),
      "progressive_token_masked_scores_forward",
      [&] {
        token_masked_scores_kernel<scalar_t><<<
            blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                bit_allocations.data_ptr<int8_t>(),
                band_mask.data_ptr<int8_t>(),
                code_offsets.data_ptr<int16_t>(),
                scale_offsets.data_ptr<int8_t>(),
                code_bases.data_ptr<int64_t>(),
                scale_bases.data_ptr<int64_t>(),
                code_strides.data_ptr<int16_t>(),
                scale_strides.data_ptr<int8_t>(),
                candidate_mask.data_ptr<uint8_t>(),
                output.data_ptr<float>(),
                kv_head_count,
                query_groups,
                history_count);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
"""


@lru_cache(maxsize=1)
def load_extension():
    return load_inline(
        name="progressive_variablebit_ext_v2",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=None,
        extra_cuda_cflags=("-O3", "--use_fast_math"),
        extra_cflags=("-O3",),
        with_cuda=True,
        verbose=False,
    )


def _metadata_arguments(index: dict[str, object]) -> tuple[torch.Tensor, ...]:
    return (
        index["packed_codes"],
        index["key_scales"],
        index["bit_allocations"],
        index["code_offsets"],
        index["scale_offsets"],
        index["code_bases"],
        index["scale_bases"],
        index["code_strides"],
        index["scale_strides"],
    )


def masked_scores(
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    index: dict[str, object],
    band_mask: torch.Tensor,
    history_count: int,
) -> torch.Tensor:
    return load_extension().progressive_masked_scores_forward(
        query_codes,
        query_scales,
        *_metadata_arguments(index)[:3],
        band_mask.to(device=query_codes.device, dtype=torch.int8).contiguous(),
        *_metadata_arguments(index)[3:],
        history_count,
    )


def candidate_scores(
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    index: dict[str, object],
    band_mask: torch.Tensor,
    candidate_indices: torch.Tensor,
) -> torch.Tensor:
    return load_extension().progressive_candidate_scores_forward(
        query_codes,
        query_scales,
        *_metadata_arguments(index)[:3],
        band_mask.to(device=query_codes.device, dtype=torch.int8).contiguous(),
        *_metadata_arguments(index)[3:],
        candidate_indices.contiguous(),
    )


def token_masked_scores(
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    index: dict[str, object],
    band_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    return load_extension().progressive_token_masked_scores_forward(
        query_codes,
        query_scales,
        *_metadata_arguments(index)[:3],
        band_mask.to(device=query_codes.device, dtype=torch.int8).contiguous(),
        *_metadata_arguments(index)[3:],
        candidate_mask.to(
            device=query_codes.device,
            dtype=torch.uint8,
        ).contiguous(),
    )
