from __future__ import annotations

from functools import lru_cache

import torch
from torch.utils.cpp_extension import load_inline


CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor jointkv_base_priority_forward(
    torch::Tensor codes,
    torch::Tensor joint_ids,
    torch::Tensor risk_codes,
    torch::Tensor packed_query,
    torch::Tensor risk_lut,
    int64_t bits,
    int64_t probe_offset,
    int64_t joint_offset);

torch::Tensor jointkv_tail_cluster_mass_forward(
    torch::Tensor codes,
    torch::Tensor joint_ids,
    torch::Tensor packed_query,
    torch::Tensor selected_mask,
    torch::Tensor references,
    int64_t bits,
    int64_t probe_offset,
    int64_t joint_offset,
    int64_t blocks_per_query);

std::vector<torch::Tensor> jointkv_base_priority_mass_forward(
    torch::Tensor codes,
    torch::Tensor joint_ids,
    torch::Tensor risk_codes,
    torch::Tensor packed_query,
    torch::Tensor risk_lut,
    torch::Tensor references,
    int64_t bits,
    int64_t probe_offset,
    int64_t joint_offset);

std::vector<torch::Tensor> jointkv_base_local_select_mass_forward(
    torch::Tensor codes,
    torch::Tensor joint_ids,
    torch::Tensor risk_codes,
    torch::Tensor packed_query,
    torch::Tensor risk_lut,
    torch::Tensor references,
    int64_t bits,
    int64_t probe_offset,
    int64_t joint_offset,
    int64_t keep_per_warp);

std::vector<torch::Tensor> jointkv_residual_local_shortlist_forward(
    torch::Tensor residual_codes,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_base_scores,
    torch::Tensor packed_query,
    int64_t bits,
    int64_t probe_offset,
    int64_t keep_per_warp);

torch::Tensor jointkv_build_query_lut_forward(
    torch::Tensor packed_query,
    int64_t base_bits,
    int64_t residual_bits,
    int64_t base_offset,
    int64_t residual_offset);

std::vector<torch::Tensor> jointkv_base_lut_local_select_mass_forward(
    torch::Tensor codes,
    torch::Tensor joint_ids,
    torch::Tensor risk_codes,
    torch::Tensor packed_query,
    torch::Tensor query_lut,
    torch::Tensor risk_lut,
    torch::Tensor references,
    int64_t base_chunks,
    int64_t joint_offset,
    int64_t keep_per_warp);

std::vector<torch::Tensor> jointkv_residual_lut_local_shortlist_forward(
    torch::Tensor residual_codes,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_base_scores,
    torch::Tensor query_lut,
    int64_t residual_chunk_offset,
    int64_t residual_chunks,
    int64_t keep_per_warp);

torch::Tensor jointkv_subtract_selected_mass_forward(
    torch::Tensor cluster_mass,
    torch::Tensor codes,
    torch::Tensor joint_ids,
    torch::Tensor packed_query,
    torch::Tensor selected_indices,
    torch::Tensor references,
    int64_t bits,
    int64_t probe_offset,
    int64_t joint_offset);

torch::Tensor jointkv_subtract_selected_mass_lut_forward(
    torch::Tensor cluster_mass,
    torch::Tensor codes,
    torch::Tensor joint_ids,
    torch::Tensor packed_query,
    torch::Tensor query_lut,
    torch::Tensor selected_indices,
    torch::Tensor references,
    int64_t base_chunks,
    int64_t joint_offset);

torch::Tensor jointkv_tail_blend_forward(
    torch::Tensor sparse_output,
    torch::Tensor cluster_mass,
    torch::Tensor value_centroids,
    double selected_fraction);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "jointkv_base_priority_forward",
      &jointkv_base_priority_forward,
      "JointKV fused binary score, joint-ID correction, and risk lookup");
  m.def(
      "jointkv_tail_cluster_mass_forward",
      &jointkv_tail_cluster_mass_forward,
      "JointKV omitted-token cluster-mass reduction");
  m.def(
      "jointkv_base_priority_mass_forward",
      &jointkv_base_priority_mass_forward,
      "JointKV fused priority scan and all-token cluster masses");
  m.def(
      "jointkv_base_local_select_mass_forward",
      &jointkv_base_local_select_mass_forward,
      "JointKV fused scan, warp-local candidate selection, and cluster masses");
  m.def(
      "jointkv_residual_local_shortlist_forward",
      &jointkv_residual_local_shortlist_forward,
      "JointKV fused residual gather, scan, and warp-local shortlist");
  m.def(
      "jointkv_build_query_lut_forward",
      &jointkv_build_query_lut_forward,
      "JointKV build 8-bit signed-dot query lookup tables");
  m.def(
      "jointkv_base_lut_local_select_mass_forward",
      &jointkv_base_lut_local_select_mass_forward,
      "JointKV LUT scan, warp-local selection, and cluster masses");
  m.def(
      "jointkv_residual_lut_local_shortlist_forward",
      &jointkv_residual_lut_local_shortlist_forward,
      "JointKV LUT residual scan and warp-local shortlist");
  m.def(
      "jointkv_subtract_selected_mass_forward",
      &jointkv_subtract_selected_mass_forward,
      "JointKV selected-token subtraction from cluster masses");
  m.def(
      "jointkv_subtract_selected_mass_lut_forward",
      &jointkv_subtract_selected_mass_lut_forward,
      "JointKV query-LUT selected-token subtraction from cluster masses");
  m.def(
      "jointkv_tail_blend_forward",
      &jointkv_tail_blend_forward,
      "JointKV fused centroid reduction and sparse/tail blend");
}
"""


CUDA_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

template <typename scalar_t>
__global__ void jointkv_base_priority_kernel(
    const int64_t* __restrict__ codes,
    const uint8_t* __restrict__ joint_ids,
    const uint8_t* __restrict__ risk_codes,
    const scalar_t* __restrict__ packed_query,
    const float* __restrict__ risk_lut,
    float* __restrict__ output,
    int batch_count,
    int kv_head_count,
    int query_groups,
    int token_count,
    int query_width,
    int bits,
    int probe_offset,
    int joint_offset) {
  int token = blockIdx.x * blockDim.x + threadIdx.x;
  int kv_head = blockIdx.y;
  int batch = blockIdx.z;
  if (token >= token_count) {
    return;
  }

  int token_offset = (batch * kv_head_count + kv_head) * token_count + token;
  unsigned long long code = static_cast<unsigned long long>(codes[token_offset]);
  int joint_id = static_cast<int>(joint_ids[token_offset] & 63u);
  int risk_code = static_cast<int>(risk_codes[token_offset]);
  float maximum = -3.402823466e+38F;
  for (int group = 0; group < query_groups; ++group) {
    int query_base =
        ((batch * kv_head_count + kv_head) * query_groups + group) * query_width;
    float score = 0.0f;
    for (int bit = 0; bit < bits; ++bit) {
      float probe = static_cast<float>(
          packed_query[query_base + probe_offset + bit]);
      score += ((code >> bit) & 1ull) ? probe : -probe;
    }
    if (joint_offset >= 0) {
      score += static_cast<float>(
          packed_query[query_base + joint_offset + joint_id]);
    }
    maximum = fmaxf(maximum, score);
  }
  int risk_offset = (kv_head * 64 + joint_id) * 256 + risk_code;
  output[token_offset] = maximum + risk_lut[risk_offset];
}

template <typename scalar_t>
__global__ void jointkv_tail_cluster_mass_kernel(
    const int64_t* __restrict__ codes,
    const uint8_t* __restrict__ joint_ids,
    const scalar_t* __restrict__ packed_query,
    const uint8_t* __restrict__ selected_mask,
    const float* __restrict__ references,
    float* __restrict__ output,
    int batch_count,
    int kv_head_count,
    int query_groups,
    int token_count,
    int query_width,
    int bits,
    int probe_offset,
    int joint_offset,
    int blocks_per_query) {
  __shared__ float local_mass[64];
  if (threadIdx.x < 64) {
    local_mass[threadIdx.x] = 0.0f;
  }
  __syncthreads();

  int query_index = blockIdx.x / blocks_per_query;
  int local_block = blockIdx.x - query_index * blocks_per_query;
  int group = query_index % query_groups;
  int kv_head = (query_index / query_groups) % kv_head_count;
  int batch = query_index / (query_groups * kv_head_count);
  if (batch >= batch_count) {
    return;
  }
  int query_base =
      ((batch * kv_head_count + kv_head) * query_groups + group) * query_width;
  int token_base = (batch * kv_head_count + kv_head) * token_count;
  float reference = references[
      (batch * kv_head_count + kv_head) * query_groups + group];

  for (int token = local_block * blockDim.x + threadIdx.x;
       token < token_count;
       token += blocks_per_query * blockDim.x) {
    int token_offset = token_base + token;
    if (selected_mask[token_offset] != 0) {
      continue;
    }
    unsigned long long code =
        static_cast<unsigned long long>(codes[token_offset]);
    int joint_id = static_cast<int>(joint_ids[token_offset] & 63u);
    float score = 0.0f;
    for (int bit = 0; bit < bits; ++bit) {
      float probe = static_cast<float>(
          packed_query[query_base + probe_offset + bit]);
      score += ((code >> bit) & 1ull) ? probe : -probe;
    }
    if (joint_offset >= 0) {
      score += static_cast<float>(
          packed_query[query_base + joint_offset + joint_id]);
    }
    float exponent = fminf(20.0f, fmaxf(-80.0f, score - reference));
    atomicAdd(&local_mass[joint_id], __expf(exponent));
  }
  __syncthreads();

  if (threadIdx.x < 64) {
    int output_offset =
        (((batch * kv_head_count + kv_head) * query_groups + group) * 64)
        + threadIdx.x;
    atomicAdd(&output[output_offset], local_mass[threadIdx.x]);
  }
}

template <typename scalar_t>
__global__ void jointkv_base_priority_mass_kernel(
    const int64_t* __restrict__ codes,
    const uint8_t* __restrict__ joint_ids,
    const uint8_t* __restrict__ risk_codes,
    const scalar_t* __restrict__ packed_query,
    const float* __restrict__ risk_lut,
    const float* __restrict__ references,
    float* __restrict__ priority,
    float* __restrict__ cluster_mass,
    int batch_count,
    int kv_head_count,
    int query_groups,
    int token_count,
    int query_width,
    int bits,
    int probe_offset,
    int joint_offset) {
  __shared__ float local_mass[256];
  int mass_count = query_groups * 64;
  if (threadIdx.x < mass_count) {
    local_mass[threadIdx.x] = 0.0f;
  }
  __syncthreads();

  int token = blockIdx.x * blockDim.x + threadIdx.x;
  int kv_head = blockIdx.y;
  int batch = blockIdx.z;
  if (token < token_count) {
    int token_offset =
        (batch * kv_head_count + kv_head) * token_count + token;
    unsigned long long code =
        static_cast<unsigned long long>(codes[token_offset]);
    int joint_id = static_cast<int>(joint_ids[token_offset] & 63u);
    int risk_code = static_cast<int>(risk_codes[token_offset]);
    float maximum = -3.402823466e+38F;
    for (int group = 0; group < query_groups; ++group) {
      int query_base =
          ((batch * kv_head_count + kv_head) * query_groups + group)
          * query_width;
      float score = 0.0f;
      for (int bit = 0; bit < bits; ++bit) {
        float probe = static_cast<float>(
            packed_query[query_base + probe_offset + bit]);
        score += ((code >> bit) & 1ull) ? probe : -probe;
      }
      if (joint_offset >= 0) {
        score += static_cast<float>(
            packed_query[query_base + joint_offset + joint_id]);
      }
      float reference = references[
          (batch * kv_head_count + kv_head) * query_groups + group];
      float exponent = fminf(20.0f, fmaxf(-80.0f, score - reference));
      atomicAdd(&local_mass[group * 64 + joint_id], __expf(exponent));
      maximum = fmaxf(maximum, score);
    }
    int risk_offset = (kv_head * 64 + joint_id) * 256 + risk_code;
    priority[token_offset] = maximum + risk_lut[risk_offset];
  }
  __syncthreads();

  if (threadIdx.x < mass_count) {
    int group = threadIdx.x / 64;
    int cluster = threadIdx.x - group * 64;
    int output_offset =
        (((batch * kv_head_count + kv_head) * query_groups + group) * 64)
        + cluster;
    atomicAdd(&cluster_mass[output_offset], local_mass[threadIdx.x]);
  }
}

__device__ __forceinline__ bool jointkv_score_precedes(
    float left_score,
    int left_index,
    float right_score,
    int right_index) {
  return left_score > right_score
      || (left_score == right_score && left_index < right_index);
}

__device__ __forceinline__ void jointkv_warp_sort_descending(
    float& score,
    int& index) {
  int lane = threadIdx.x & 31;
  for (int width = 2; width <= 32; width <<= 1) {
    // The first bitonic run is descending, so the final width-32 run is also
    // descending and its first lanes hold the retained candidates.
    bool ascending = (lane & width) != 0;
    for (int stride = width >> 1; stride > 0; stride >>= 1) {
      float peer_score = __shfl_xor_sync(0xffffffffu, score, stride);
      int peer_index = __shfl_xor_sync(0xffffffffu, index, stride);
      bool peer_precedes = jointkv_score_precedes(
          peer_score, peer_index, score, index);
      bool self_precedes = jointkv_score_precedes(
          score, index, peer_score, peer_index);
      bool lower_lane = (lane & stride) == 0;
      bool want_better = (!ascending && lower_lane)
          || (ascending && !lower_lane);
      bool take_peer = want_better ? peer_precedes : self_precedes;
      if (take_peer) {
        score = peer_score;
        index = peer_index;
      }
    }
  }
}

template <typename scalar_t>
__global__ void jointkv_base_local_select_mass_kernel(
    const int64_t* __restrict__ codes,
    const uint8_t* __restrict__ joint_ids,
    const uint8_t* __restrict__ risk_codes,
    const scalar_t* __restrict__ packed_query,
    const float* __restrict__ risk_lut,
    const float* __restrict__ references,
    int64_t* __restrict__ candidate_indices,
    float* __restrict__ candidate_scores,
    float* __restrict__ cluster_mass,
    int batch_count,
    int kv_head_count,
    int query_groups,
    int token_count,
    int query_width,
    int bits,
    int probe_offset,
    int joint_offset,
    int keep_per_warp,
    int warp_count,
    int candidate_count) {
  __shared__ float local_mass[256];
  int mass_count = query_groups * 64;
  if (threadIdx.x < mass_count) {
    local_mass[threadIdx.x] = 0.0f;
  }
  __syncthreads();

  int token = blockIdx.x * blockDim.x + threadIdx.x;
  int kv_head = blockIdx.y;
  int batch = blockIdx.z;
  int token_base = (batch * kv_head_count + kv_head) * token_count;
  float priority = -3.402823466e+38F;
  int sorted_index = 0x7fffffff;
  if (token < token_count) {
    int token_offset = token_base + token;
    unsigned long long code =
        static_cast<unsigned long long>(codes[token_offset]);
    int joint_id = static_cast<int>(joint_ids[token_offset] & 63u);
    int risk_code = static_cast<int>(risk_codes[token_offset]);
    float maximum = -3.402823466e+38F;
    for (int group = 0; group < query_groups; ++group) {
      int query_base =
          ((batch * kv_head_count + kv_head) * query_groups + group)
          * query_width;
      float score = 0.0f;
      for (int bit = 0; bit < bits; ++bit) {
        float probe = static_cast<float>(
            packed_query[query_base + probe_offset + bit]);
        score += ((code >> bit) & 1ull) ? probe : -probe;
      }
      if (joint_offset >= 0) {
        score += static_cast<float>(
            packed_query[query_base + joint_offset + joint_id]);
      }
      float reference = references[
          (batch * kv_head_count + kv_head) * query_groups + group];
      float exponent = fminf(20.0f, fmaxf(-80.0f, score - reference));
      atomicAdd(&local_mass[group * 64 + joint_id], __expf(exponent));
      maximum = fmaxf(maximum, score);
    }
    int risk_offset = (kv_head * 64 + joint_id) * 256 + risk_code;
    priority = maximum + risk_lut[risk_offset];
    sorted_index = token;
  }

  jointkv_warp_sort_descending(priority, sorted_index);
  int lane = threadIdx.x & 31;
  int warp_in_block = threadIdx.x >> 5;
  int warp = blockIdx.x * 8 + warp_in_block;
  if (warp < warp_count && lane < keep_per_warp) {
    int output_offset =
        (batch * kv_head_count + kv_head) * candidate_count
        + warp * keep_per_warp + lane;
    candidate_indices[output_offset] =
        sorted_index == 0x7fffffff ? 0 : static_cast<int64_t>(sorted_index);
    candidate_scores[output_offset] = priority;
  }
  __syncthreads();

  if (threadIdx.x < mass_count) {
    int group = threadIdx.x / 64;
    int cluster = threadIdx.x - group * 64;
    int output_offset =
        (((batch * kv_head_count + kv_head) * query_groups + group) * 64)
        + cluster;
    atomicAdd(&cluster_mass[output_offset], local_mass[threadIdx.x]);
  }
}

template <typename scalar_t>
__global__ void jointkv_residual_local_shortlist_kernel(
    const int64_t* __restrict__ residual_codes,
    const int64_t* __restrict__ candidate_indices,
    const float* __restrict__ candidate_base_scores,
    const scalar_t* __restrict__ packed_query,
    int64_t* __restrict__ shortlist_indices,
    float* __restrict__ shortlist_scores,
    int batch_count,
    int kv_head_count,
    int query_groups,
    int token_count,
    int candidate_count,
    int query_width,
    int bits,
    int probe_offset,
    int keep_per_warp,
    int candidate_warp_count,
    int shortlist_count) {
  int candidate_position = blockIdx.x * blockDim.x + threadIdx.x;
  int kv_head = blockIdx.y;
  int batch = blockIdx.z;
  int candidate_base =
      (batch * kv_head_count + kv_head) * candidate_count;
  float refined_score = -3.402823466e+38F;
  int sorted_index = 0x7fffffff;
  if (candidate_position < candidate_count) {
    float base_score = candidate_base_scores[candidate_base + candidate_position];
    int token = static_cast<int>(
        candidate_indices[candidate_base + candidate_position]);
    if (base_score > -3.0e+38F && token >= 0 && token < token_count) {
      unsigned long long code = static_cast<unsigned long long>(
          residual_codes[
              (batch * kv_head_count + kv_head) * token_count + token]);
      float maximum = -3.402823466e+38F;
      for (int group = 0; group < query_groups; ++group) {
        int query_base =
            ((batch * kv_head_count + kv_head) * query_groups + group)
            * query_width;
        float score = 0.0f;
        for (int bit = 0; bit < bits; ++bit) {
          float probe = static_cast<float>(
              packed_query[query_base + probe_offset + bit]);
          score += ((code >> bit) & 1ull) ? probe : -probe;
        }
        maximum = fmaxf(maximum, score);
      }
      refined_score = base_score + maximum;
      sorted_index = token;
    }
  }

  jointkv_warp_sort_descending(refined_score, sorted_index);
  int lane = threadIdx.x & 31;
  int warp_in_block = threadIdx.x >> 5;
  int warp = blockIdx.x * 8 + warp_in_block;
  if (warp < candidate_warp_count && lane < keep_per_warp) {
    int output_offset =
        (batch * kv_head_count + kv_head) * shortlist_count
        + warp * keep_per_warp + lane;
    shortlist_indices[output_offset] =
        sorted_index == 0x7fffffff ? 0 : static_cast<int64_t>(sorted_index);
    shortlist_scores[output_offset] = refined_score;
  }
}

template <typename scalar_t>
__global__ void jointkv_build_query_lut_kernel(
    const scalar_t* __restrict__ packed_query,
    float* __restrict__ query_lut,
    int query_width,
    int base_chunks,
    int residual_chunks,
    int base_offset,
    int residual_offset,
    int element_count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  int pattern = index & 255;
  int packed_index = index >> 8;
  int total_chunks = base_chunks + residual_chunks;
  int chunk = packed_index % total_chunks;
  int query_index = packed_index / total_chunks;
  int source_offset = chunk < base_chunks
      ? base_offset + chunk * 8
      : residual_offset + (chunk - base_chunks) * 8;
  int query_base = query_index * query_width + source_offset;
  float score = 0.0f;
#pragma unroll
  for (int bit = 0; bit < 8; ++bit) {
    float probe = static_cast<float>(packed_query[query_base + bit]);
    score += ((pattern >> bit) & 1) ? probe : -probe;
  }
  query_lut[index] = score;
}

template <typename scalar_t>
__global__ void jointkv_base_lut_local_select_mass_kernel(
    const int64_t* __restrict__ codes,
    const uint8_t* __restrict__ joint_ids,
    const uint8_t* __restrict__ risk_codes,
    const scalar_t* __restrict__ packed_query,
    const float* __restrict__ query_lut,
    const float* __restrict__ risk_lut,
    const float* __restrict__ references,
    int64_t* __restrict__ candidate_indices,
    float* __restrict__ candidate_scores,
    float* __restrict__ cluster_mass,
    int batch_count,
    int kv_head_count,
    int query_groups,
    int token_count,
    int query_width,
    int total_chunks,
    int base_chunks,
    int joint_offset,
    int keep_per_warp,
    int warp_count,
    int candidate_count) {
  __shared__ float local_mass[256];
  int mass_count = query_groups * 64;
  if (threadIdx.x < mass_count) {
    local_mass[threadIdx.x] = 0.0f;
  }
  __syncthreads();

  int token = blockIdx.x * blockDim.x + threadIdx.x;
  int kv_head = blockIdx.y;
  int batch = blockIdx.z;
  int token_base = (batch * kv_head_count + kv_head) * token_count;
  float priority = -3.402823466e+38F;
  int sorted_index = 0x7fffffff;
  if (token < token_count) {
    int token_offset = token_base + token;
    unsigned long long code =
        static_cast<unsigned long long>(codes[token_offset]);
    int joint_id = static_cast<int>(joint_ids[token_offset] & 63u);
    int risk_code = static_cast<int>(risk_codes[token_offset]);
    float maximum = -3.402823466e+38F;
    for (int group = 0; group < query_groups; ++group) {
      int query_index =
          (batch * kv_head_count + kv_head) * query_groups + group;
      int lut_base = query_index * total_chunks * 256;
      float score = 0.0f;
      for (int chunk = 0; chunk < base_chunks; ++chunk) {
        int pattern = static_cast<int>((code >> (chunk * 8)) & 255ull);
        score += query_lut[lut_base + chunk * 256 + pattern];
      }
      score += static_cast<float>(
          packed_query[query_index * query_width + joint_offset + joint_id]);
      float reference = references[query_index];
      float exponent = fminf(20.0f, fmaxf(-80.0f, score - reference));
      atomicAdd(&local_mass[group * 64 + joint_id], __expf(exponent));
      maximum = fmaxf(maximum, score);
    }
    int risk_offset = (kv_head * 64 + joint_id) * 256 + risk_code;
    priority = maximum + risk_lut[risk_offset];
    sorted_index = token;
  }

  jointkv_warp_sort_descending(priority, sorted_index);
  int lane = threadIdx.x & 31;
  int warp_in_block = threadIdx.x >> 5;
  int warp = blockIdx.x * 8 + warp_in_block;
  if (warp < warp_count && lane < keep_per_warp) {
    int output_offset =
        (batch * kv_head_count + kv_head) * candidate_count
        + warp * keep_per_warp + lane;
    candidate_indices[output_offset] =
        sorted_index == 0x7fffffff ? 0 : static_cast<int64_t>(sorted_index);
    candidate_scores[output_offset] = priority;
  }
  __syncthreads();

  if (threadIdx.x < mass_count) {
    int group = threadIdx.x / 64;
    int cluster = threadIdx.x - group * 64;
    int output_offset =
        (((batch * kv_head_count + kv_head) * query_groups + group) * 64)
        + cluster;
    atomicAdd(&cluster_mass[output_offset], local_mass[threadIdx.x]);
  }
}

__global__ void jointkv_residual_lut_local_shortlist_kernel(
    const int64_t* __restrict__ residual_codes,
    const int64_t* __restrict__ candidate_indices,
    const float* __restrict__ candidate_base_scores,
    const float* __restrict__ query_lut,
    int64_t* __restrict__ shortlist_indices,
    float* __restrict__ shortlist_scores,
    int batch_count,
    int kv_head_count,
    int query_groups,
    int token_count,
    int candidate_count,
    int total_chunks,
    int residual_chunk_offset,
    int residual_chunks,
    int keep_per_warp,
    int candidate_warp_count,
    int shortlist_count) {
  int candidate_position = blockIdx.x * blockDim.x + threadIdx.x;
  int kv_head = blockIdx.y;
  int batch = blockIdx.z;
  int candidate_base =
      (batch * kv_head_count + kv_head) * candidate_count;
  float refined_score = -3.402823466e+38F;
  int sorted_index = 0x7fffffff;
  if (candidate_position < candidate_count) {
    float base_score = candidate_base_scores[candidate_base + candidate_position];
    int token = static_cast<int>(
        candidate_indices[candidate_base + candidate_position]);
    if (base_score > -3.0e+38F && token >= 0 && token < token_count) {
      unsigned long long code = static_cast<unsigned long long>(
          residual_codes[
              (batch * kv_head_count + kv_head) * token_count + token]);
      float maximum = -3.402823466e+38F;
      for (int group = 0; group < query_groups; ++group) {
        int query_index =
            (batch * kv_head_count + kv_head) * query_groups + group;
        int lut_base = query_index * total_chunks * 256;
        float score = 0.0f;
        for (int chunk = 0; chunk < residual_chunks; ++chunk) {
          int pattern = static_cast<int>((code >> (chunk * 8)) & 255ull);
          score += query_lut[
              lut_base + (residual_chunk_offset + chunk) * 256 + pattern];
        }
        maximum = fmaxf(maximum, score);
      }
      refined_score = base_score + maximum;
      sorted_index = token;
    }
  }

  jointkv_warp_sort_descending(refined_score, sorted_index);
  int lane = threadIdx.x & 31;
  int warp_in_block = threadIdx.x >> 5;
  int warp = blockIdx.x * 8 + warp_in_block;
  if (warp < candidate_warp_count && lane < keep_per_warp) {
    int output_offset =
        (batch * kv_head_count + kv_head) * shortlist_count
        + warp * keep_per_warp + lane;
    shortlist_indices[output_offset] =
        sorted_index == 0x7fffffff ? 0 : static_cast<int64_t>(sorted_index);
    shortlist_scores[output_offset] = refined_score;
  }
}

template <typename scalar_t>
__global__ void jointkv_subtract_selected_mass_kernel(
    float* __restrict__ cluster_mass,
    const int64_t* __restrict__ codes,
    const uint8_t* __restrict__ joint_ids,
    const scalar_t* __restrict__ packed_query,
    const int64_t* __restrict__ selected_indices,
    const float* __restrict__ references,
    int kv_head_count,
    int query_groups,
    int token_count,
    int selected_count,
    int query_width,
    int bits,
    int probe_offset,
    int joint_offset) {
  int selected_position = blockIdx.x * blockDim.x + threadIdx.x;
  int kv_head = blockIdx.y;
  int batch = blockIdx.z;
  if (selected_position >= selected_count) {
    return;
  }
  int selected_offset =
      (batch * kv_head_count + kv_head) * selected_count + selected_position;
  int token = static_cast<int>(selected_indices[selected_offset]);
  if (token < 0 || token >= token_count) {
    return;
  }
  int token_offset = (batch * kv_head_count + kv_head) * token_count + token;
  unsigned long long code = static_cast<unsigned long long>(codes[token_offset]);
  int joint_id = static_cast<int>(joint_ids[token_offset] & 63u);
  for (int group = 0; group < query_groups; ++group) {
    int query_base =
        ((batch * kv_head_count + kv_head) * query_groups + group) * query_width;
    float score = 0.0f;
    for (int bit = 0; bit < bits; ++bit) {
      float probe = static_cast<float>(
          packed_query[query_base + probe_offset + bit]);
      score += ((code >> bit) & 1ull) ? probe : -probe;
    }
    if (joint_offset >= 0) {
      score += static_cast<float>(
          packed_query[query_base + joint_offset + joint_id]);
    }
    float reference = references[
        (batch * kv_head_count + kv_head) * query_groups + group];
    float exponent = fminf(20.0f, fmaxf(-80.0f, score - reference));
    int mass_offset =
        (((batch * kv_head_count + kv_head) * query_groups + group) * 64)
        + joint_id;
    atomicAdd(&cluster_mass[mass_offset], -__expf(exponent));
  }
}

template <typename scalar_t>
__global__ void jointkv_subtract_selected_mass_lut_kernel(
    float* __restrict__ cluster_mass,
    const int64_t* __restrict__ codes,
    const uint8_t* __restrict__ joint_ids,
    const scalar_t* __restrict__ packed_query,
    const float* __restrict__ query_lut,
    const int64_t* __restrict__ selected_indices,
    const float* __restrict__ references,
    int kv_head_count,
    int query_groups,
    int token_count,
    int selected_count,
    int query_width,
    int total_chunks,
    int base_chunks,
    int joint_offset) {
  int selected_position = blockIdx.x * blockDim.x + threadIdx.x;
  int kv_head = blockIdx.y;
  int batch = blockIdx.z;
  if (selected_position >= selected_count) {
    return;
  }
  int selected_offset =
      (batch * kv_head_count + kv_head) * selected_count + selected_position;
  int token = static_cast<int>(selected_indices[selected_offset]);
  if (token < 0 || token >= token_count) {
    return;
  }
  int token_offset = (batch * kv_head_count + kv_head) * token_count + token;
  unsigned long long code = static_cast<unsigned long long>(codes[token_offset]);
  int joint_id = static_cast<int>(joint_ids[token_offset] & 63u);
  for (int group = 0; group < query_groups; ++group) {
    int query_index =
        (batch * kv_head_count + kv_head) * query_groups + group;
    int lut_base = query_index * total_chunks * 256;
    float score = 0.0f;
    for (int chunk = 0; chunk < base_chunks; ++chunk) {
      int pattern = static_cast<int>((code >> (chunk * 8)) & 255ull);
      score += query_lut[lut_base + chunk * 256 + pattern];
    }
    score += static_cast<float>(
        packed_query[query_index * query_width + joint_offset + joint_id]);
    float reference = references[query_index];
    float exponent = fminf(20.0f, fmaxf(-80.0f, score - reference));
    int mass_offset =
        (((batch * kv_head_count + kv_head) * query_groups + group) * 64)
        + joint_id;
    atomicAdd(&cluster_mass[mass_offset], -__expf(exponent));
  }
}

template <typename scalar_t>
__global__ void jointkv_tail_blend_kernel(
    const scalar_t* __restrict__ sparse_output,
    const float* __restrict__ cluster_mass,
    const float* __restrict__ value_centroids,
    scalar_t* __restrict__ output,
    int batch_count,
    int query_head_count,
    int kv_head_count,
    int query_groups,
    int head_dim,
    float selected_fraction,
    int element_count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  int dimension = index % head_dim;
  int query_head = (index / head_dim) % query_head_count;
  int batch = index / (query_head_count * head_dim);
  int kv_head = query_head / query_groups;
  int group = query_head - kv_head * query_groups;
  int mass_base =
      ((batch * kv_head_count + kv_head) * query_groups + group) * 64;
  int centroid_base = kv_head * 64 * head_dim + dimension;
  float mass_sum = 0.0f;
  float numerator = 0.0f;
  for (int cluster = 0; cluster < 64; ++cluster) {
    float mass = fmaxf(0.0f, cluster_mass[mass_base + cluster]);
    mass_sum += mass;
    numerator += mass * value_centroids[
        centroid_base + cluster * head_dim];
  }
  float tail = numerator / fmaxf(1.0e-8f, mass_sum);
  float sparse = static_cast<float>(sparse_output[index]);
  output[index] = static_cast<scalar_t>(
      selected_fraction * sparse + (1.0f - selected_fraction) * tail);
}

torch::Tensor jointkv_base_priority_forward(
    torch::Tensor codes,
    torch::Tensor joint_ids,
    torch::Tensor risk_codes,
    torch::Tensor packed_query,
    torch::Tensor risk_lut,
    int64_t bits,
    int64_t probe_offset,
    int64_t joint_offset) {
  TORCH_CHECK(codes.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(
      joint_ids.is_cuda() && risk_codes.is_cuda() && packed_query.is_cuda()
          && risk_lut.is_cuda(),
      "inputs must share a CUDA device");
  TORCH_CHECK(codes.dim() == 3, "codes must be [B,KVH,N]");
  TORCH_CHECK(codes.scalar_type() == at::kLong, "codes must be int64");
  TORCH_CHECK(
      joint_ids.sizes() == codes.sizes() && risk_codes.sizes() == codes.sizes(),
      "metadata must align with codes");
  TORCH_CHECK(
      joint_ids.scalar_type() == at::kByte
          && risk_codes.scalar_type() == at::kByte,
      "metadata must be uint8");
  TORCH_CHECK(packed_query.dim() == 4, "packed query must be [B,KVH,G,W]");
  TORCH_CHECK(
      packed_query.size(0) == codes.size(0)
          && packed_query.size(1) == codes.size(1),
      "query and code dimensions must align");
  TORCH_CHECK(
      risk_lut.dim() == 3 && risk_lut.size(0) == codes.size(1)
          && risk_lut.size(1) == 64 && risk_lut.size(2) == 256,
      "risk LUT must be [KVH,64,256]");
  TORCH_CHECK(risk_lut.scalar_type() == at::kFloat, "risk LUT must be float32");
  TORCH_CHECK(bits > 0 && bits <= 64, "bits must lie in [1,64]");
  TORCH_CHECK(
      probe_offset >= 0 && probe_offset + bits <= packed_query.size(3),
      "probe range leaves packed query");
  TORCH_CHECK(
      joint_offset < 0 || joint_offset + 64 <= packed_query.size(3),
      "joint-score range leaves packed query");

  auto codes_c = codes.contiguous();
  auto ids_c = joint_ids.contiguous();
  auto risk_c = risk_codes.contiguous();
  auto query_c = packed_query.contiguous();
  auto lut_c = risk_lut.contiguous();
  auto output = torch::empty(
      codes_c.sizes(), codes_c.options().dtype(at::kFloat));
  int batch_count = static_cast<int>(codes_c.size(0));
  int kv_head_count = static_cast<int>(codes_c.size(1));
  int token_count = static_cast<int>(codes_c.size(2));
  int query_groups = static_cast<int>(query_c.size(2));
  int query_width = static_cast<int>(query_c.size(3));

  c10::cuda::CUDAGuard device_guard(codes_c.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((token_count + 255) / 256, kv_head_count, batch_count);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "jointkv_base_priority_forward",
      [&] {
        jointkv_base_priority_kernel<scalar_t><<<grid, 256, 0, stream>>>(
            codes_c.data_ptr<int64_t>(),
            ids_c.data_ptr<uint8_t>(),
            risk_c.data_ptr<uint8_t>(),
            query_c.data_ptr<scalar_t>(),
            lut_c.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_count,
            kv_head_count,
            query_groups,
            token_count,
            query_width,
            static_cast<int>(bits),
            static_cast<int>(probe_offset),
            static_cast<int>(joint_offset));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor jointkv_tail_cluster_mass_forward(
    torch::Tensor codes,
    torch::Tensor joint_ids,
    torch::Tensor packed_query,
    torch::Tensor selected_mask,
    torch::Tensor references,
    int64_t bits,
    int64_t probe_offset,
    int64_t joint_offset,
    int64_t blocks_per_query) {
  TORCH_CHECK(codes.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(
      joint_ids.is_cuda() && packed_query.is_cuda() && selected_mask.is_cuda()
          && references.is_cuda(),
      "inputs must share a CUDA device");
  TORCH_CHECK(codes.dim() == 3 && codes.scalar_type() == at::kLong,
              "codes must be int64 [B,KVH,N]");
  TORCH_CHECK(
      joint_ids.sizes() == codes.sizes()
          && selected_mask.sizes() == codes.sizes(),
      "metadata and selected mask must align with codes");
  TORCH_CHECK(
      joint_ids.scalar_type() == at::kByte
          && selected_mask.scalar_type() == at::kByte,
      "joint IDs and mask must be uint8");
  TORCH_CHECK(packed_query.dim() == 4, "packed query must be [B,KVH,G,W]");
  TORCH_CHECK(
      references.dim() == 3
          && references.size(0) == packed_query.size(0)
          && references.size(1) == packed_query.size(1)
          && references.size(2) == packed_query.size(2),
      "references must be [B,KVH,G]");
  TORCH_CHECK(references.scalar_type() == at::kFloat,
              "references must be float32");
  TORCH_CHECK(bits > 0 && bits <= 64, "bits must lie in [1,64]");
  TORCH_CHECK(blocks_per_query > 0, "blocks per query must be positive");

  auto codes_c = codes.contiguous();
  auto ids_c = joint_ids.contiguous();
  auto query_c = packed_query.contiguous();
  auto mask_c = selected_mask.contiguous();
  auto references_c = references.contiguous();
  int batch_count = static_cast<int>(codes_c.size(0));
  int kv_head_count = static_cast<int>(codes_c.size(1));
  int token_count = static_cast<int>(codes_c.size(2));
  int query_groups = static_cast<int>(query_c.size(2));
  int query_width = static_cast<int>(query_c.size(3));
  auto output = torch::zeros(
      {batch_count, kv_head_count, query_groups, 64},
      codes_c.options().dtype(at::kFloat));

  c10::cuda::CUDAGuard device_guard(codes_c.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  int query_count = batch_count * kv_head_count * query_groups;
  int block_count = query_count * static_cast<int>(blocks_per_query);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "jointkv_tail_cluster_mass_forward",
      [&] {
        jointkv_tail_cluster_mass_kernel<scalar_t><<<
            block_count, 256, 0, stream>>>(
                codes_c.data_ptr<int64_t>(),
                ids_c.data_ptr<uint8_t>(),
                query_c.data_ptr<scalar_t>(),
                mask_c.data_ptr<uint8_t>(),
                references_c.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_count,
                kv_head_count,
                query_groups,
                token_count,
                query_width,
                static_cast<int>(bits),
                static_cast<int>(probe_offset),
                static_cast<int>(joint_offset),
                static_cast<int>(blocks_per_query));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> jointkv_base_priority_mass_forward(
    torch::Tensor codes,
    torch::Tensor joint_ids,
    torch::Tensor risk_codes,
    torch::Tensor packed_query,
    torch::Tensor risk_lut,
    torch::Tensor references,
    int64_t bits,
    int64_t probe_offset,
    int64_t joint_offset) {
  TORCH_CHECK(codes.is_cuda() && packed_query.is_cuda(),
              "inputs must be CUDA tensors");
  TORCH_CHECK(codes.dim() == 3 && codes.scalar_type() == at::kLong,
              "codes must be int64 [B,KVH,N]");
  TORCH_CHECK(joint_ids.sizes() == codes.sizes()
                  && risk_codes.sizes() == codes.sizes(),
              "metadata must align with codes");
  TORCH_CHECK(joint_ids.scalar_type() == at::kByte
                  && risk_codes.scalar_type() == at::kByte,
              "metadata must be uint8");
  TORCH_CHECK(packed_query.dim() == 4, "packed query must be [B,KVH,G,W]");
  TORCH_CHECK(references.dim() == 3
                  && references.size(0) == packed_query.size(0)
                  && references.size(1) == packed_query.size(1)
                  && references.size(2) == packed_query.size(2),
              "references must be [B,KVH,G]");
  TORCH_CHECK(risk_lut.scalar_type() == at::kFloat
                  && references.scalar_type() == at::kFloat,
              "risk LUT and references must be float32");
  TORCH_CHECK(bits > 0 && bits <= 64, "bits must lie in [1,64]");

  auto codes_c = codes.contiguous();
  auto ids_c = joint_ids.contiguous();
  auto risk_c = risk_codes.contiguous();
  auto query_c = packed_query.contiguous();
  auto lut_c = risk_lut.contiguous();
  auto references_c = references.contiguous();
  int batch_count = static_cast<int>(codes_c.size(0));
  int kv_head_count = static_cast<int>(codes_c.size(1));
  int token_count = static_cast<int>(codes_c.size(2));
  int query_groups = static_cast<int>(query_c.size(2));
  int query_width = static_cast<int>(query_c.size(3));
  TORCH_CHECK(query_groups <= 4,
              "fused mass kernel supports at most four GQA groups");
  auto priority = torch::empty(
      codes_c.sizes(), codes_c.options().dtype(at::kFloat));
  auto mass = torch::zeros(
      {batch_count, kv_head_count, query_groups, 64},
      codes_c.options().dtype(at::kFloat));

  c10::cuda::CUDAGuard device_guard(codes_c.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((token_count + 255) / 256, kv_head_count, batch_count);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "jointkv_base_priority_mass_forward",
      [&] {
        jointkv_base_priority_mass_kernel<scalar_t><<<grid, 256, 0, stream>>>(
            codes_c.data_ptr<int64_t>(),
            ids_c.data_ptr<uint8_t>(),
            risk_c.data_ptr<uint8_t>(),
            query_c.data_ptr<scalar_t>(),
            lut_c.data_ptr<float>(),
            references_c.data_ptr<float>(),
            priority.data_ptr<float>(),
            mass.data_ptr<float>(),
            batch_count,
            kv_head_count,
            query_groups,
            token_count,
            query_width,
            static_cast<int>(bits),
            static_cast<int>(probe_offset),
            static_cast<int>(joint_offset));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {priority, mass};
}

std::vector<torch::Tensor> jointkv_base_local_select_mass_forward(
    torch::Tensor codes,
    torch::Tensor joint_ids,
    torch::Tensor risk_codes,
    torch::Tensor packed_query,
    torch::Tensor risk_lut,
    torch::Tensor references,
    int64_t bits,
    int64_t probe_offset,
    int64_t joint_offset,
    int64_t keep_per_warp) {
  TORCH_CHECK(codes.is_cuda() && packed_query.is_cuda(),
              "inputs must be CUDA tensors");
  TORCH_CHECK(codes.dim() == 3 && codes.scalar_type() == at::kLong,
              "codes must be int64 [B,KVH,N]");
  TORCH_CHECK(joint_ids.sizes() == codes.sizes()
                  && risk_codes.sizes() == codes.sizes(),
              "metadata must align with codes");
  TORCH_CHECK(joint_ids.scalar_type() == at::kByte
                  && risk_codes.scalar_type() == at::kByte,
              "metadata must be uint8");
  TORCH_CHECK(packed_query.dim() == 4,
              "packed query must be [B,KVH,G,W]");
  TORCH_CHECK(references.dim() == 3
                  && references.size(0) == packed_query.size(0)
                  && references.size(1) == packed_query.size(1)
                  && references.size(2) == packed_query.size(2),
              "references must be [B,KVH,G]");
  TORCH_CHECK(risk_lut.scalar_type() == at::kFloat
                  && references.scalar_type() == at::kFloat,
              "risk LUT and references must be float32");
  TORCH_CHECK(bits > 0 && bits <= 64, "bits must lie in [1,64]");
  TORCH_CHECK(keep_per_warp > 0 && keep_per_warp <= 32,
              "keep per warp must lie in [1,32]");

  auto codes_c = codes.contiguous();
  auto ids_c = joint_ids.contiguous();
  auto risk_c = risk_codes.contiguous();
  auto query_c = packed_query.contiguous();
  auto lut_c = risk_lut.contiguous();
  auto references_c = references.contiguous();
  int batch_count = static_cast<int>(codes_c.size(0));
  int kv_head_count = static_cast<int>(codes_c.size(1));
  int token_count = static_cast<int>(codes_c.size(2));
  int query_groups = static_cast<int>(query_c.size(2));
  int query_width = static_cast<int>(query_c.size(3));
  TORCH_CHECK(query_groups <= 4,
              "fused local selector supports at most four GQA groups");
  int warp_count = (token_count + 31) / 32;
  int candidate_count = warp_count * static_cast<int>(keep_per_warp);
  auto candidate_indices = torch::empty(
      {batch_count, kv_head_count, candidate_count}, codes_c.options());
  auto candidate_scores = torch::empty(
      {batch_count, kv_head_count, candidate_count},
      codes_c.options().dtype(at::kFloat));
  auto mass = torch::zeros(
      {batch_count, kv_head_count, query_groups, 64},
      codes_c.options().dtype(at::kFloat));

  c10::cuda::CUDAGuard device_guard(codes_c.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((token_count + 255) / 256, kv_head_count, batch_count);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "jointkv_base_local_select_mass_forward",
      [&] {
        jointkv_base_local_select_mass_kernel<scalar_t><<<
            grid, 256, 0, stream>>>(
                codes_c.data_ptr<int64_t>(),
                ids_c.data_ptr<uint8_t>(),
                risk_c.data_ptr<uint8_t>(),
                query_c.data_ptr<scalar_t>(),
                lut_c.data_ptr<float>(),
                references_c.data_ptr<float>(),
                candidate_indices.data_ptr<int64_t>(),
                candidate_scores.data_ptr<float>(),
                mass.data_ptr<float>(),
                batch_count,
                kv_head_count,
                query_groups,
                token_count,
                query_width,
                static_cast<int>(bits),
                static_cast<int>(probe_offset),
                static_cast<int>(joint_offset),
                static_cast<int>(keep_per_warp),
                warp_count,
                candidate_count);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {candidate_indices, candidate_scores, mass};
}

std::vector<torch::Tensor> jointkv_residual_local_shortlist_forward(
    torch::Tensor residual_codes,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_base_scores,
    torch::Tensor packed_query,
    int64_t bits,
    int64_t probe_offset,
    int64_t keep_per_warp) {
  TORCH_CHECK(residual_codes.is_cuda() && packed_query.is_cuda(),
              "inputs must be CUDA tensors");
  TORCH_CHECK(residual_codes.dim() == 3
                  && residual_codes.scalar_type() == at::kLong,
              "residual codes must be int64 [B,KVH,N]");
  TORCH_CHECK(candidate_indices.dim() == 3
                  && candidate_indices.scalar_type() == at::kLong,
              "candidate indices must be int64 [B,KVH,C]");
  TORCH_CHECK(candidate_base_scores.dim() == 3
                  && candidate_base_scores.scalar_type() == at::kFloat,
              "candidate scores must be float32 [B,KVH,C]");
  TORCH_CHECK(candidate_indices.sizes() == candidate_base_scores.sizes(),
              "candidate indices and scores must align");
  TORCH_CHECK(candidate_indices.size(0) == residual_codes.size(0)
                  && candidate_indices.size(1) == residual_codes.size(1),
              "candidate and residual-code dimensions must align");
  TORCH_CHECK(bits > 0 && bits <= 64, "bits must lie in [1,64]");
  TORCH_CHECK(keep_per_warp > 0 && keep_per_warp <= 32,
              "keep per warp must lie in [1,32]");

  auto residual_c = residual_codes.contiguous();
  auto indices_c = candidate_indices.contiguous();
  auto scores_c = candidate_base_scores.contiguous();
  auto query_c = packed_query.contiguous();
  int batch_count = static_cast<int>(residual_c.size(0));
  int kv_head_count = static_cast<int>(residual_c.size(1));
  int token_count = static_cast<int>(residual_c.size(2));
  int candidate_count = static_cast<int>(indices_c.size(2));
  int query_groups = static_cast<int>(query_c.size(2));
  int query_width = static_cast<int>(query_c.size(3));
  int candidate_warp_count = (candidate_count + 31) / 32;
  int shortlist_count =
      candidate_warp_count * static_cast<int>(keep_per_warp);
  auto shortlist_indices = torch::empty(
      {batch_count, kv_head_count, shortlist_count}, residual_c.options());
  auto shortlist_scores = torch::empty(
      {batch_count, kv_head_count, shortlist_count},
      residual_c.options().dtype(at::kFloat));

  c10::cuda::CUDAGuard device_guard(residual_c.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((candidate_count + 255) / 256, kv_head_count, batch_count);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "jointkv_residual_local_shortlist_forward",
      [&] {
        jointkv_residual_local_shortlist_kernel<scalar_t><<<
            grid, 256, 0, stream>>>(
                residual_c.data_ptr<int64_t>(),
                indices_c.data_ptr<int64_t>(),
                scores_c.data_ptr<float>(),
                query_c.data_ptr<scalar_t>(),
                shortlist_indices.data_ptr<int64_t>(),
                shortlist_scores.data_ptr<float>(),
                batch_count,
                kv_head_count,
                query_groups,
                token_count,
                candidate_count,
                query_width,
                static_cast<int>(bits),
                static_cast<int>(probe_offset),
                static_cast<int>(keep_per_warp),
                candidate_warp_count,
                shortlist_count);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {shortlist_indices, shortlist_scores};
}

torch::Tensor jointkv_build_query_lut_forward(
    torch::Tensor packed_query,
    int64_t base_bits,
    int64_t residual_bits,
    int64_t base_offset,
    int64_t residual_offset) {
  TORCH_CHECK(packed_query.is_cuda() && packed_query.dim() == 4,
              "packed query must be CUDA [B,KVH,G,W]");
  TORCH_CHECK(base_bits > 0 && base_bits <= 64 && base_bits % 8 == 0,
              "base bits must be a positive multiple of eight up to 64");
  TORCH_CHECK(residual_bits > 0 && residual_bits <= 64
                  && residual_bits % 8 == 0,
              "residual bits must be a positive multiple of eight up to 64");
  TORCH_CHECK(base_offset >= 0
                  && base_offset + base_bits <= packed_query.size(3),
              "base probe range leaves packed query");
  TORCH_CHECK(residual_offset >= 0
                  && residual_offset + residual_bits <= packed_query.size(3),
              "residual probe range leaves packed query");
  auto query_c = packed_query.contiguous();
  int base_chunks = static_cast<int>(base_bits / 8);
  int residual_chunks = static_cast<int>(residual_bits / 8);
  int total_chunks = base_chunks + residual_chunks;
  auto output = torch::empty(
      {query_c.size(0), query_c.size(1), query_c.size(2), total_chunks, 256},
      query_c.options().dtype(at::kFloat));
  int element_count = static_cast<int>(output.numel());
  int query_width = static_cast<int>(query_c.size(3));

  c10::cuda::CUDAGuard device_guard(query_c.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "jointkv_build_query_lut_forward",
      [&] {
        jointkv_build_query_lut_kernel<scalar_t><<<
            (element_count + 255) / 256, 256, 0, stream>>>(
                query_c.data_ptr<scalar_t>(),
                output.data_ptr<float>(),
                query_width,
                base_chunks,
                residual_chunks,
                static_cast<int>(base_offset),
                static_cast<int>(residual_offset),
                element_count);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> jointkv_base_lut_local_select_mass_forward(
    torch::Tensor codes,
    torch::Tensor joint_ids,
    torch::Tensor risk_codes,
    torch::Tensor packed_query,
    torch::Tensor query_lut,
    torch::Tensor risk_lut,
    torch::Tensor references,
    int64_t base_chunks,
    int64_t joint_offset,
    int64_t keep_per_warp) {
  TORCH_CHECK(codes.is_cuda() && packed_query.is_cuda()
                  && query_lut.is_cuda(),
              "inputs must be CUDA tensors");
  TORCH_CHECK(codes.dim() == 3 && codes.scalar_type() == at::kLong,
              "codes must be int64 [B,KVH,N]");
  TORCH_CHECK(joint_ids.sizes() == codes.sizes()
                  && risk_codes.sizes() == codes.sizes(),
              "metadata must align with codes");
  TORCH_CHECK(joint_ids.scalar_type() == at::kByte
                  && risk_codes.scalar_type() == at::kByte,
              "metadata must be uint8");
  TORCH_CHECK(query_lut.dim() == 5 && query_lut.size(4) == 256
                  && query_lut.scalar_type() == at::kFloat,
              "query LUT must be float32 [B,KVH,G,C,256]");
  TORCH_CHECK(query_lut.size(0) == packed_query.size(0)
                  && query_lut.size(1) == packed_query.size(1)
                  && query_lut.size(2) == packed_query.size(2),
              "query LUT and packed query must align");
  TORCH_CHECK(references.dim() == 3
                  && references.size(0) == packed_query.size(0)
                  && references.size(1) == packed_query.size(1)
                  && references.size(2) == packed_query.size(2),
              "references must be [B,KVH,G]");
  TORCH_CHECK(base_chunks > 0 && base_chunks <= query_lut.size(3),
              "base chunk count leaves query LUT");
  TORCH_CHECK(joint_offset >= 0 && joint_offset + 64 <= packed_query.size(3),
              "joint-score range leaves packed query");
  TORCH_CHECK(keep_per_warp > 0 && keep_per_warp <= 32,
              "keep per warp must lie in [1,32]");

  auto codes_c = codes.contiguous();
  auto ids_c = joint_ids.contiguous();
  auto risk_c = risk_codes.contiguous();
  auto query_c = packed_query.contiguous();
  auto query_lut_c = query_lut.contiguous();
  auto risk_lut_c = risk_lut.contiguous();
  auto references_c = references.contiguous();
  int batch_count = static_cast<int>(codes_c.size(0));
  int kv_head_count = static_cast<int>(codes_c.size(1));
  int token_count = static_cast<int>(codes_c.size(2));
  int query_groups = static_cast<int>(query_c.size(2));
  int query_width = static_cast<int>(query_c.size(3));
  int total_chunks = static_cast<int>(query_lut_c.size(3));
  TORCH_CHECK(query_groups <= 4,
              "LUT local selector supports at most four GQA groups");
  int warp_count = (token_count + 31) / 32;
  int candidate_count = warp_count * static_cast<int>(keep_per_warp);
  auto candidate_indices = torch::empty(
      {batch_count, kv_head_count, candidate_count}, codes_c.options());
  auto candidate_scores = torch::empty(
      {batch_count, kv_head_count, candidate_count},
      codes_c.options().dtype(at::kFloat));
  auto mass = torch::zeros(
      {batch_count, kv_head_count, query_groups, 64},
      codes_c.options().dtype(at::kFloat));

  c10::cuda::CUDAGuard device_guard(codes_c.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((token_count + 255) / 256, kv_head_count, batch_count);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "jointkv_base_lut_local_select_mass_forward",
      [&] {
        jointkv_base_lut_local_select_mass_kernel<scalar_t><<<
            grid, 256, 0, stream>>>(
                codes_c.data_ptr<int64_t>(),
                ids_c.data_ptr<uint8_t>(),
                risk_c.data_ptr<uint8_t>(),
                query_c.data_ptr<scalar_t>(),
                query_lut_c.data_ptr<float>(),
                risk_lut_c.data_ptr<float>(),
                references_c.data_ptr<float>(),
                candidate_indices.data_ptr<int64_t>(),
                candidate_scores.data_ptr<float>(),
                mass.data_ptr<float>(),
                batch_count,
                kv_head_count,
                query_groups,
                token_count,
                query_width,
                total_chunks,
                static_cast<int>(base_chunks),
                static_cast<int>(joint_offset),
                static_cast<int>(keep_per_warp),
                warp_count,
                candidate_count);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {candidate_indices, candidate_scores, mass};
}

std::vector<torch::Tensor> jointkv_residual_lut_local_shortlist_forward(
    torch::Tensor residual_codes,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_base_scores,
    torch::Tensor query_lut,
    int64_t residual_chunk_offset,
    int64_t residual_chunks,
    int64_t keep_per_warp) {
  TORCH_CHECK(residual_codes.is_cuda() && query_lut.is_cuda(),
              "inputs must be CUDA tensors");
  TORCH_CHECK(residual_codes.dim() == 3
                  && residual_codes.scalar_type() == at::kLong,
              "residual codes must be int64 [B,KVH,N]");
  TORCH_CHECK(candidate_indices.dim() == 3
                  && candidate_indices.scalar_type() == at::kLong,
              "candidate indices must be int64 [B,KVH,C]");
  TORCH_CHECK(candidate_base_scores.dim() == 3
                  && candidate_base_scores.scalar_type() == at::kFloat
                  && candidate_indices.sizes() == candidate_base_scores.sizes(),
              "candidate indices and float32 scores must align");
  TORCH_CHECK(query_lut.dim() == 5 && query_lut.size(4) == 256
                  && query_lut.scalar_type() == at::kFloat,
              "query LUT must be float32 [B,KVH,G,C,256]");
  TORCH_CHECK(residual_chunk_offset >= 0 && residual_chunks > 0
                  && residual_chunk_offset + residual_chunks
                      <= query_lut.size(3),
              "residual chunk range leaves query LUT");
  TORCH_CHECK(keep_per_warp > 0 && keep_per_warp <= 32,
              "keep per warp must lie in [1,32]");

  auto residual_c = residual_codes.contiguous();
  auto indices_c = candidate_indices.contiguous();
  auto scores_c = candidate_base_scores.contiguous();
  auto query_lut_c = query_lut.contiguous();
  int batch_count = static_cast<int>(residual_c.size(0));
  int kv_head_count = static_cast<int>(residual_c.size(1));
  int token_count = static_cast<int>(residual_c.size(2));
  int candidate_count = static_cast<int>(indices_c.size(2));
  int query_groups = static_cast<int>(query_lut_c.size(2));
  int total_chunks = static_cast<int>(query_lut_c.size(3));
  int candidate_warp_count = (candidate_count + 31) / 32;
  int shortlist_count =
      candidate_warp_count * static_cast<int>(keep_per_warp);
  auto shortlist_indices = torch::empty(
      {batch_count, kv_head_count, shortlist_count}, residual_c.options());
  auto shortlist_scores = torch::empty(
      {batch_count, kv_head_count, shortlist_count},
      residual_c.options().dtype(at::kFloat));

  c10::cuda::CUDAGuard device_guard(residual_c.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((candidate_count + 255) / 256, kv_head_count, batch_count);
  jointkv_residual_lut_local_shortlist_kernel<<<grid, 256, 0, stream>>>(
      residual_c.data_ptr<int64_t>(),
      indices_c.data_ptr<int64_t>(),
      scores_c.data_ptr<float>(),
      query_lut_c.data_ptr<float>(),
      shortlist_indices.data_ptr<int64_t>(),
      shortlist_scores.data_ptr<float>(),
      batch_count,
      kv_head_count,
      query_groups,
      token_count,
      candidate_count,
      total_chunks,
      static_cast<int>(residual_chunk_offset),
      static_cast<int>(residual_chunks),
      static_cast<int>(keep_per_warp),
      candidate_warp_count,
      shortlist_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {shortlist_indices, shortlist_scores};
}

torch::Tensor jointkv_subtract_selected_mass_forward(
    torch::Tensor cluster_mass,
    torch::Tensor codes,
    torch::Tensor joint_ids,
    torch::Tensor packed_query,
    torch::Tensor selected_indices,
    torch::Tensor references,
    int64_t bits,
    int64_t probe_offset,
    int64_t joint_offset) {
  TORCH_CHECK(cluster_mass.is_cuda() && codes.is_cuda(),
              "inputs must be CUDA tensors");
  TORCH_CHECK(cluster_mass.scalar_type() == at::kFloat,
              "cluster mass must be float32");
  TORCH_CHECK(selected_indices.scalar_type() == at::kLong
                  && selected_indices.dim() == 3,
              "selected indices must be int64 [B,KVH,K]");
  auto mass_c = cluster_mass.contiguous();
  auto codes_c = codes.contiguous();
  auto ids_c = joint_ids.contiguous();
  auto query_c = packed_query.contiguous();
  auto indices_c = selected_indices.contiguous();
  auto references_c = references.contiguous();
  int batch_count = static_cast<int>(codes_c.size(0));
  int kv_head_count = static_cast<int>(codes_c.size(1));
  int token_count = static_cast<int>(codes_c.size(2));
  int selected_count = static_cast<int>(indices_c.size(2));
  int query_groups = static_cast<int>(query_c.size(2));
  int query_width = static_cast<int>(query_c.size(3));

  c10::cuda::CUDAGuard device_guard(codes_c.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((selected_count + 255) / 256, kv_head_count, batch_count);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "jointkv_subtract_selected_mass_forward",
      [&] {
        jointkv_subtract_selected_mass_kernel<scalar_t><<<grid, 256, 0, stream>>>(
            mass_c.data_ptr<float>(),
            codes_c.data_ptr<int64_t>(),
            ids_c.data_ptr<uint8_t>(),
            query_c.data_ptr<scalar_t>(),
            indices_c.data_ptr<int64_t>(),
            references_c.data_ptr<float>(),
            kv_head_count,
            query_groups,
            token_count,
            selected_count,
            query_width,
            static_cast<int>(bits),
            static_cast<int>(probe_offset),
            static_cast<int>(joint_offset));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return mass_c;
}

torch::Tensor jointkv_subtract_selected_mass_lut_forward(
    torch::Tensor cluster_mass,
    torch::Tensor codes,
    torch::Tensor joint_ids,
    torch::Tensor packed_query,
    torch::Tensor query_lut,
    torch::Tensor selected_indices,
    torch::Tensor references,
    int64_t base_chunks,
    int64_t joint_offset) {
  TORCH_CHECK(cluster_mass.is_cuda() && codes.is_cuda()
                  && packed_query.is_cuda() && query_lut.is_cuda(),
              "inputs must be CUDA tensors");
  TORCH_CHECK(cluster_mass.scalar_type() == at::kFloat,
              "cluster mass must be float32");
  TORCH_CHECK(query_lut.dim() == 5 && query_lut.size(4) == 256
                  && query_lut.scalar_type() == at::kFloat,
              "query LUT must be float32 [B,KVH,G,C,256]");
  TORCH_CHECK(selected_indices.scalar_type() == at::kLong
                  && selected_indices.dim() == 3,
              "selected indices must be int64 [B,KVH,K]");
  TORCH_CHECK(base_chunks > 0 && base_chunks <= query_lut.size(3),
              "base chunk count leaves query LUT");
  TORCH_CHECK(joint_offset >= 0 && joint_offset + 64 <= packed_query.size(3),
              "joint-score range leaves packed query");
  auto mass_c = cluster_mass.contiguous();
  auto codes_c = codes.contiguous();
  auto ids_c = joint_ids.contiguous();
  auto query_c = packed_query.contiguous();
  auto query_lut_c = query_lut.contiguous();
  auto indices_c = selected_indices.contiguous();
  auto references_c = references.contiguous();
  int batch_count = static_cast<int>(codes_c.size(0));
  int kv_head_count = static_cast<int>(codes_c.size(1));
  int token_count = static_cast<int>(codes_c.size(2));
  int selected_count = static_cast<int>(indices_c.size(2));
  int query_groups = static_cast<int>(query_c.size(2));
  int query_width = static_cast<int>(query_c.size(3));
  int total_chunks = static_cast<int>(query_lut_c.size(3));

  c10::cuda::CUDAGuard device_guard(codes_c.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((selected_count + 255) / 256, kv_head_count, batch_count);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "jointkv_subtract_selected_mass_lut_forward",
      [&] {
        jointkv_subtract_selected_mass_lut_kernel<scalar_t><<<
            grid, 256, 0, stream>>>(
                mass_c.data_ptr<float>(),
                codes_c.data_ptr<int64_t>(),
                ids_c.data_ptr<uint8_t>(),
                query_c.data_ptr<scalar_t>(),
                query_lut_c.data_ptr<float>(),
                indices_c.data_ptr<int64_t>(),
                references_c.data_ptr<float>(),
                kv_head_count,
                query_groups,
                token_count,
                selected_count,
                query_width,
                total_chunks,
                static_cast<int>(base_chunks),
                static_cast<int>(joint_offset));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return mass_c;
}

torch::Tensor jointkv_tail_blend_forward(
    torch::Tensor sparse_output,
    torch::Tensor cluster_mass,
    torch::Tensor value_centroids,
    double selected_fraction) {
  TORCH_CHECK(sparse_output.is_cuda() && cluster_mass.is_cuda()
                  && value_centroids.is_cuda(),
              "inputs must be CUDA tensors");
  TORCH_CHECK(sparse_output.dim() == 4 && sparse_output.size(1) == 1,
              "sparse output must be [B,1,QH,D]");
  TORCH_CHECK(cluster_mass.dim() == 4 && cluster_mass.size(3) == 64,
              "cluster mass must be [B,KVH,G,64]");
  TORCH_CHECK(value_centroids.dim() == 3 && value_centroids.size(1) == 64,
              "centroids must be [KVH,64,D]");
  TORCH_CHECK(cluster_mass.scalar_type() == at::kFloat
                  && value_centroids.scalar_type() == at::kFloat,
              "mass and centroids must be float32");
  TORCH_CHECK(selected_fraction >= 0.0 && selected_fraction <= 1.0,
              "selected fraction must lie in [0,1]");
  auto sparse_c = sparse_output.contiguous();
  auto mass_c = cluster_mass.contiguous();
  auto centroids_c = value_centroids.contiguous();
  auto output = torch::empty_like(sparse_c);
  int batch_count = static_cast<int>(sparse_c.size(0));
  int query_head_count = static_cast<int>(sparse_c.size(2));
  int kv_head_count = static_cast<int>(mass_c.size(1));
  int query_groups = static_cast<int>(mass_c.size(2));
  int head_dim = static_cast<int>(sparse_c.size(3));
  int element_count = static_cast<int>(sparse_c.numel());
  TORCH_CHECK(query_head_count == kv_head_count * query_groups,
              "GQA dimensions do not align");

  c10::cuda::CUDAGuard device_guard(sparse_c.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      sparse_c.scalar_type(),
      "jointkv_tail_blend_forward",
      [&] {
        jointkv_tail_blend_kernel<scalar_t><<<
            (element_count + 255) / 256, 256, 0, stream>>>(
                sparse_c.data_ptr<scalar_t>(),
                mass_c.data_ptr<float>(),
                centroids_c.data_ptr<float>(),
                output.data_ptr<scalar_t>(),
                batch_count,
                query_head_count,
                kv_head_count,
                query_groups,
                head_dim,
                static_cast<float>(selected_fraction),
                element_count);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
"""


@lru_cache(maxsize=1)
def load_extension():
    return load_inline(
        name="jointkv_sieve_cuda_ext_v4",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )


def base_priority(
    codes: torch.Tensor,
    joint_ids: torch.Tensor,
    risk_codes: torch.Tensor,
    packed_query: torch.Tensor,
    risk_lut: torch.Tensor,
    *,
    bits: int,
    probe_offset: int,
    joint_offset: int = -1,
) -> torch.Tensor:
    return load_extension().jointkv_base_priority_forward(
        codes,
        joint_ids,
        risk_codes,
        packed_query,
        risk_lut,
        bits,
        probe_offset,
        joint_offset,
    )


def tail_cluster_mass(
    codes: torch.Tensor,
    joint_ids: torch.Tensor,
    packed_query: torch.Tensor,
    selected_mask: torch.Tensor,
    references: torch.Tensor,
    *,
    bits: int,
    probe_offset: int,
    joint_offset: int,
    blocks_per_query: int,
) -> torch.Tensor:
    return load_extension().jointkv_tail_cluster_mass_forward(
        codes,
        joint_ids,
        packed_query,
        selected_mask,
        references,
        bits,
        probe_offset,
        joint_offset,
        blocks_per_query,
    )


def base_priority_and_mass(
    codes: torch.Tensor,
    joint_ids: torch.Tensor,
    risk_codes: torch.Tensor,
    packed_query: torch.Tensor,
    risk_lut: torch.Tensor,
    references: torch.Tensor,
    *,
    bits: int,
    probe_offset: int,
    joint_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    priority, mass = load_extension().jointkv_base_priority_mass_forward(
        codes,
        joint_ids,
        risk_codes,
        packed_query,
        risk_lut,
        references,
        bits,
        probe_offset,
        joint_offset,
    )
    return priority, mass


def base_local_select_and_mass(
    codes: torch.Tensor,
    joint_ids: torch.Tensor,
    risk_codes: torch.Tensor,
    packed_query: torch.Tensor,
    risk_lut: torch.Tensor,
    references: torch.Tensor,
    *,
    bits: int,
    probe_offset: int,
    joint_offset: int,
    keep_per_warp: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices, scores, mass = (
        load_extension().jointkv_base_local_select_mass_forward(
            codes,
            joint_ids,
            risk_codes,
            packed_query,
            risk_lut,
            references,
            bits,
            probe_offset,
            joint_offset,
            keep_per_warp,
        )
    )
    return indices, scores, mass


def residual_local_shortlist(
    residual_codes: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_base_scores: torch.Tensor,
    packed_query: torch.Tensor,
    *,
    bits: int,
    probe_offset: int,
    keep_per_warp: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices, scores = load_extension().jointkv_residual_local_shortlist_forward(
        residual_codes,
        candidate_indices,
        candidate_base_scores,
        packed_query,
        bits,
        probe_offset,
        keep_per_warp,
    )
    return indices, scores


def build_query_lut(
    packed_query: torch.Tensor,
    *,
    base_bits: int,
    residual_bits: int,
    base_offset: int,
    residual_offset: int,
) -> torch.Tensor:
    return load_extension().jointkv_build_query_lut_forward(
        packed_query,
        base_bits,
        residual_bits,
        base_offset,
        residual_offset,
    )


def base_lut_local_select_and_mass(
    codes: torch.Tensor,
    joint_ids: torch.Tensor,
    risk_codes: torch.Tensor,
    packed_query: torch.Tensor,
    query_lut: torch.Tensor,
    risk_lut: torch.Tensor,
    references: torch.Tensor,
    *,
    base_chunks: int,
    joint_offset: int,
    keep_per_warp: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices, scores, mass = (
        load_extension().jointkv_base_lut_local_select_mass_forward(
            codes,
            joint_ids,
            risk_codes,
            packed_query,
            query_lut,
            risk_lut,
            references,
            base_chunks,
            joint_offset,
            keep_per_warp,
        )
    )
    return indices, scores, mass


def residual_lut_local_shortlist(
    residual_codes: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_base_scores: torch.Tensor,
    query_lut: torch.Tensor,
    *,
    residual_chunk_offset: int,
    residual_chunks: int,
    keep_per_warp: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices, scores = (
        load_extension().jointkv_residual_lut_local_shortlist_forward(
            residual_codes,
            candidate_indices,
            candidate_base_scores,
            query_lut,
            residual_chunk_offset,
            residual_chunks,
            keep_per_warp,
        )
    )
    return indices, scores


def subtract_selected_mass(
    cluster_mass: torch.Tensor,
    codes: torch.Tensor,
    joint_ids: torch.Tensor,
    packed_query: torch.Tensor,
    selected_indices: torch.Tensor,
    references: torch.Tensor,
    *,
    bits: int,
    probe_offset: int,
    joint_offset: int,
) -> torch.Tensor:
    return load_extension().jointkv_subtract_selected_mass_forward(
        cluster_mass,
        codes,
        joint_ids,
        packed_query,
        selected_indices,
        references,
        bits,
        probe_offset,
        joint_offset,
    )


def subtract_selected_mass_lut(
    cluster_mass: torch.Tensor,
    codes: torch.Tensor,
    joint_ids: torch.Tensor,
    packed_query: torch.Tensor,
    query_lut: torch.Tensor,
    selected_indices: torch.Tensor,
    references: torch.Tensor,
    *,
    base_chunks: int,
    joint_offset: int,
) -> torch.Tensor:
    return load_extension().jointkv_subtract_selected_mass_lut_forward(
        cluster_mass,
        codes,
        joint_ids,
        packed_query,
        query_lut,
        selected_indices,
        references,
        base_chunks,
        joint_offset,
    )


def tail_blend(
    sparse_output: torch.Tensor,
    cluster_mass: torch.Tensor,
    value_centroids: torch.Tensor,
    selected_fraction: float,
) -> torch.Tensor:
    return load_extension().jointkv_tail_blend_forward(
        sparse_output,
        cluster_mass,
        value_centroids,
        selected_fraction,
    )
