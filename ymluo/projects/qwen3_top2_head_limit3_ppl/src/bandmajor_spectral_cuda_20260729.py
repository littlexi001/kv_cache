from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from torch.utils.cpp_extension import load_inline


CPP_SOURCE = r"""
#include <torch/extension.h>

void bandmajor_sampled_compact_gqa4_indices_out_cuda(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor code_band_bases,
    torch::Tensor scale_band_bases,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    torch::Tensor thresholds,
    torch::Tensor overflow,
    int64_t history_count,
    int64_t sample_count,
    double selected_fraction);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "sampled_compact_gqa4_indices_out",
      &bandmajor_sampled_compact_gqa4_indices_out_cuda,
      "Band-major GQA4 sampled threshold and compaction");
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
#include <cmath>

#define BANDMAJOR_MAX_SAMPLE_COUNT 2048

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
          word, reinterpret_cast<const int*>(query)[chunk], dot);
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
          word, reinterpret_cast<const int*>(query)[chunk], dot);
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
          word, reinterpret_cast<const int*>(query)[chunk], dot);
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
          word, reinterpret_cast<const int*>(query)[chunk], dot);
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
            word, reinterpret_cast<const int*>(query)[chunk], dots[group]);
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
            word, reinterpret_cast<const int*>(query)[chunk], dots[group]);
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
            word, reinterpret_cast<const int*>(query)[chunk], dots[group]);
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
            word, reinterpret_cast<const int*>(query)[chunk], dots[group]);
      }
    }
  }
}

template <typename scale_t>
__device__ __forceinline__ float score_one(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int64_t* __restrict__ code_band_bases,
    const int64_t* __restrict__ scale_band_bases,
    int batch_kv,
    int query_group,
    int token) {
  constexpr int query_groups = 4;
  int64_t query_row =
      static_cast<int64_t>(batch_kv) * query_groups + query_group;
  const int8_t* query = query_codes + query_row * 128;
  const scale_t* query_scale = query_scales + query_row * 8;
  const int8_t* allocations = bit_allocations + batch_kv * 8;
  float score = 0.0f;
#pragma unroll
  for (int band = 0; band < 8; ++band) {
    int bits = static_cast<int>(allocations[band]);
    if (bits == 0) {
      continue;
    }
    int64_t metadata_index = static_cast<int64_t>(batch_kv) * 8 + band;
    const uint8_t* code = packed_codes
        + code_band_bases[metadata_index]
        + static_cast<int64_t>(token) * (2 * bits);
    float key_scale = static_cast<float>(
        key_scales[scale_band_bases[metadata_index] + token]);
    score += static_cast<float>(
        band_dot(code, query + 16 * band, bits))
        * static_cast<float>(query_scale[band]) * key_scale;
  }
  return score;
}

template <typename scale_t>
__device__ __forceinline__ void score_gqa4_one(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int64_t* __restrict__ code_band_bases,
    const int64_t* __restrict__ scale_band_bases,
    int batch_kv,
    int token,
    float scores[4]) {
  constexpr int query_groups = 4;
  const int8_t* query_base =
      query_codes + static_cast<int64_t>(batch_kv) * query_groups * 128;
  const scale_t* query_scale_base =
      query_scales + static_cast<int64_t>(batch_kv) * query_groups * 8;
  const int8_t* allocations = bit_allocations + batch_kv * 8;
#pragma unroll
  for (int group = 0; group < query_groups; ++group) {
    scores[group] = 0.0f;
  }
#pragma unroll
  for (int band = 0; band < 8; ++band) {
    int bits = static_cast<int>(allocations[band]);
    if (bits == 0) {
      continue;
    }
    int64_t metadata_index = static_cast<int64_t>(batch_kv) * 8 + band;
    const uint8_t* code = packed_codes
        + code_band_bases[metadata_index]
        + static_cast<int64_t>(token) * (2 * bits);
    int dots[query_groups] = {0, 0, 0, 0};
    band_dot_gqa4(code, query_base, 16 * band, bits, dots);
    float key_scale = static_cast<float>(
        key_scales[scale_band_bases[metadata_index] + token]);
#pragma unroll
    for (int group = 0; group < query_groups; ++group) {
      const scale_t* query_scale = query_scale_base + group * 8;
      scores[group] += static_cast<float>(dots[group])
          * static_cast<float>(query_scale[band]) * key_scale;
    }
  }
}

template <typename scale_t>
__global__ void sample_threshold_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int64_t* __restrict__ code_band_bases,
    const int64_t* __restrict__ scale_band_bases,
    float* __restrict__ thresholds,
    int kv_head_count,
    int history_count,
    int sample_count,
    int selected_keep) {
  constexpr int query_groups = 4;
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
      score = score_one(
          query_codes, query_scales, packed_codes, key_scales,
          bit_allocations, code_band_bases, scale_band_bases,
          batch_kv, query_group, token);
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
  }
}

template <typename scale_t>
__global__ void threshold_compact_gqa4_kernel(
    const int8_t* __restrict__ query_codes,
    const scale_t* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_codes,
    const scale_t* __restrict__ key_scales,
    const int8_t* __restrict__ bit_allocations,
    const int64_t* __restrict__ code_band_bases,
    const int64_t* __restrict__ scale_band_bases,
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
  constexpr int query_groups = 4;
  int batch_kv = blockIdx.x;
  int batch = batch_kv / kv_head_count;
  int kv_head = batch_kv - batch * kv_head_count;
  float scores[query_groups];
  score_gqa4_one(
      query_codes, query_scales, packed_codes, key_scales,
      bit_allocations, code_band_bases, scale_band_bases,
      batch_kv, token, scores);
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

__global__ void finalize_counts_kernel(
    int64_t* __restrict__ candidate_counts,
    const bool* __restrict__ overflow,
    int row_count,
    int candidate_capacity) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) {
    return;
  }
  if (overflow[row] || candidate_counts[row] > candidate_capacity) {
    candidate_counts[row] = candidate_capacity;
  }
}

void bandmajor_sampled_compact_gqa4_indices_out_cuda(
    torch::Tensor query_codes,
    torch::Tensor query_scales,
    torch::Tensor packed_codes,
    torch::Tensor key_scales,
    torch::Tensor bit_allocations,
    torch::Tensor code_band_bases,
    torch::Tensor scale_band_bases,
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
              "packed codes must be uint8");
  TORCH_CHECK(bit_allocations.scalar_type() == at::kChar,
              "allocations must be int8");
  TORCH_CHECK(code_band_bases.scalar_type() == at::kLong,
              "code band bases must be int64");
  TORCH_CHECK(scale_band_bases.scalar_type() == at::kLong,
              "scale band bases must be int64");
  TORCH_CHECK(query_codes.size(2) == 4,
              "GQA4 kernel requires four Query groups");
  TORCH_CHECK(sample_count > 0
              && sample_count <= BANDMAJOR_MAX_SAMPLE_COUNT,
              "invalid sample count");
  int batch_count = static_cast<int>(query_codes.size(0));
  int kv_head_count = static_cast<int>(query_codes.size(1));
  constexpr int query_groups = 4;
  int row_count = batch_count * kv_head_count * query_groups;
  int batch_kv_count = batch_count * kv_head_count;
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
      "bandmajor_sampled_compact_gqa4_indices_out",
      [&] {
        sample_threshold_kernel<scalar_t><<<
            row_count, 256,
            BANDMAJOR_MAX_SAMPLE_COUNT * sizeof(float), stream>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                bit_allocations.data_ptr<int8_t>(),
                code_band_bases.data_ptr<int64_t>(),
                scale_band_bases.data_ptr<int64_t>(),
                thresholds.data_ptr<float>(),
                kv_head_count,
                static_cast<int>(history_count),
                static_cast<int>(sample_count),
                selected_keep);
        dim3 blocks(
            batch_kv_count,
            (history_count + 255) / 256);
        threshold_compact_gqa4_kernel<scalar_t><<<
            blocks, 256, 0, stream>>>(
                query_codes.data_ptr<int8_t>(),
                query_scales.data_ptr<scalar_t>(),
                packed_codes.data_ptr<uint8_t>(),
                key_scales.data_ptr<scalar_t>(),
                bit_allocations.data_ptr<int8_t>(),
                code_band_bases.data_ptr<int64_t>(),
                scale_band_bases.data_ptr<int64_t>(),
                thresholds.data_ptr<float>(),
                candidate_indices.data_ptr<int64_t>(),
                candidate_counts.data_ptr<int64_t>(),
                overflow.data_ptr<bool>(),
                kv_head_count,
                static_cast<int>(history_count),
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
def load_extension() -> Any:
    return load_inline(
        name="qksieve_bandmajor_spectral_20260729_v1",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=None,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )


def repack_bandmajor(
    packed_index: dict[str, Any],
) -> dict[str, Any]:
    """Transpose a fixed per-head token-major index into band-major storage."""
    capacity = int(packed_index["capacity"])
    allocations_host = packed_index["bit_allocations_host"]
    batch_count, kv_head_count, _ = allocations_host.shape
    device = packed_index["packed_codes"].device
    code_band_bases = torch.full(
        (batch_count, kv_head_count, 8),
        -1,
        dtype=torch.int64,
        device="cpu",
    )
    scale_band_bases = torch.full_like(code_band_bases, -1)
    code_cursor = 0
    scale_cursor = 0
    for batch_index in range(batch_count):
        for head_index in range(kv_head_count):
            for band_index in range(8):
                bits = int(
                    allocations_host[
                        batch_index, head_index, band_index
                    ].item()
                )
                if bits == 0:
                    continue
                code_band_bases[
                    batch_index, head_index, band_index
                ] = code_cursor
                scale_band_bases[
                    batch_index, head_index, band_index
                ] = scale_cursor
                code_cursor += capacity * (2 * bits)
                scale_cursor += capacity

    codes = torch.empty(
        code_cursor, dtype=torch.uint8, device=device
    )
    scales = torch.empty(
        scale_cursor,
        dtype=packed_index["key_scales"].dtype,
        device=device,
    )
    for batch_index in range(batch_count):
        for head_index in range(kv_head_count):
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
            source_codes = packed_index["packed_codes"][
                code_base : code_base + capacity * code_stride
            ].reshape(capacity, code_stride)
            source_scales = packed_index["key_scales"][
                scale_base : scale_base + capacity * scale_stride
            ].reshape(capacity, scale_stride)
            for band_index in range(8):
                bits = int(
                    allocations_host[
                        batch_index, head_index, band_index
                    ].item()
                )
                if bits == 0:
                    continue
                width = 2 * bits
                source_code_offset = int(
                    packed_index["code_offsets"][
                        batch_index, head_index, band_index
                    ].item()
                )
                source_scale_offset = int(
                    packed_index["scale_offsets"][
                        batch_index, head_index, band_index
                    ].item()
                )
                target_code_base = int(
                    code_band_bases[
                        batch_index, head_index, band_index
                    ].item()
                )
                target_scale_base = int(
                    scale_band_bases[
                        batch_index, head_index, band_index
                    ].item()
                )
                codes[
                    target_code_base : target_code_base + capacity * width
                ] = (
                    source_codes[
                        :, source_code_offset : source_code_offset + width
                    ]
                    .contiguous()
                    .reshape(-1)
                )
                scales[
                    target_scale_base : target_scale_base + capacity
                ] = source_scales[:, source_scale_offset]
    return {
        "packed_codes": codes,
        "key_scales": scales,
        "bit_allocations": packed_index["bit_allocations"].contiguous(),
        "code_band_bases": code_band_bases.to(device=device),
        "scale_band_bases": scale_band_bases.to(device=device),
        "capacity": capacity,
    }


def sampled_threshold_compact_gqa4_indices_out(
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    bandmajor_index: dict[str, Any],
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
        bandmajor_index["packed_codes"],
        bandmajor_index["key_scales"],
        bandmajor_index["bit_allocations"],
        bandmajor_index["code_band_bases"],
        bandmajor_index["scale_band_bases"],
        candidate_indices,
        candidate_counts,
        thresholds,
        overflow,
        history_count,
        sample_count,
        selected_fraction,
    )
    return candidate_indices, candidate_counts, thresholds, overflow
