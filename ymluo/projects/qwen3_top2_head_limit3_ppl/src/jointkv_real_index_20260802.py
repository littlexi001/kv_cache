#!/usr/bin/env python
"""Build the JointKV CUDA index from real post-RoPE K/V tensors.

This module is deliberately separate from the synthetic CUDA benchmark.  It
implements the data contract that benchmark previously populated with random
tensors: packed 64-bit principal codes, packed 48-bit residual codes, joint
K/V IDs, and an 8-bit first-stage risk code.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch


BASE_BITS = 64
RESIDUAL_BITS = 48
JOINT_CLUSTERS = 64
BASE_OFFSET = 0
RESIDUAL_OFFSET = 64
JOINT_OFFSET = 128
QUERY_WIDTH = 192


@dataclass
class JointKVRealIndex:
    base_codes: torch.Tensor
    residual_codes: torch.Tensor
    joint_ids: torch.Tensor
    risk_codes: torch.Tensor
    query_matrix: torch.Tensor
    risk_lut: torch.Tensor
    value_centroids: torch.Tensor
    base_error_levels: torch.Tensor
    value_error_levels: torch.Tensor
    build_seconds: float
    logical_bits_per_token_head: float
    physical_bytes_per_token_head: int

    @property
    def token_count(self) -> int:
        return int(self.base_codes.shape[-1])


def _symmetric_row_quantize(values: torch.Tensor, bits: int) -> torch.Tensor:
    if bits < 2:
        raise ValueError("signed centroid quantization needs at least two bits")
    maximum_code = float((1 << (bits - 1)) - 1)
    scale = values.float().abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8)
    codes = torch.round(values.float() / scale * maximum_code).clamp(
        -maximum_code, maximum_code
    )
    return codes * scale / maximum_code


def _cluster_means(
    values: torch.Tensor,
    assignments: torch.Tensor,
    clusters: int,
    fallback: torch.Tensor,
) -> torch.Tensor:
    sums = torch.zeros(
        clusters,
        values.shape[-1],
        dtype=torch.float32,
        device=values.device,
    )
    sums.index_add_(0, assignments, values.float())
    counts = torch.bincount(assignments, minlength=clusters).to(torch.float32)
    means = sums / counts[:, None].clamp_min(1.0)
    empty = counts == 0
    if bool(empty.any()):
        means[empty] = fallback.float()[empty]
    return means


def encode_binary_packed(
    coordinates: torch.Tensor,
    projection: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedily encode coordinates and pack one sign decision per bit."""
    if projection.shape[-1] != coordinates.shape[-1]:
        raise ValueError("projection and coordinates do not share a dimension")
    if projection.shape[0] > 64:
        raise ValueError("one int64 can hold at most 64 binary decisions")
    residual = coordinates.float().clone()
    packed = torch.zeros(
        coordinates.shape[0], dtype=torch.int64, device=coordinates.device
    )
    for bit, vector in enumerate(projection.float()):
        positive = (residual @ vector) >= 0
        sign = positive.to(torch.float32).mul_(2.0).sub_(1.0)
        residual.sub_(sign[:, None] * vector[None, :])
        packed.bitwise_or_(positive.to(torch.int64) << bit)
    return packed, residual


def _assign_joint_ids(
    key_residual: torch.Tensor,
    values: torch.Tensor,
    joint_model: dict[str, Any],
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    codebook = joint_model["codebook"].to(key_residual.device).float()
    dimension = int(joint_model["head_dim"])
    key_scale = float(joint_model["key_scale"])
    value_scale = float(joint_model["value_scale"])
    value_weight = float(joint_model["value_weight"])
    value_mean = joint_model["value_mean"].to(values.device).float()
    codebook_norm = codebook.square().sum(dim=-1)
    assignments = []
    for start in range(0, key_residual.shape[0], chunk_size):
        stop = min(key_residual.shape[0], start + chunk_size)
        normalized = torch.cat(
            (
                key_residual[start:stop].float() / key_scale,
                value_weight
                * (values[start:stop].float() - value_mean)
                / value_scale,
            ),
            dim=-1,
        )
        distances = (
            normalized.square().sum(dim=-1, keepdim=True)
            + codebook_norm[None, :]
            - 2.0 * normalized @ codebook.T
        )
        assignments.append(distances.argmin(dim=-1))
    assignment = torch.cat(assignments)
    offline_key_centroids = codebook[:, :dimension] * key_scale
    offline_value_centroids = (
        codebook[:, dimension:] * value_scale / value_weight + value_mean
    )
    return assignment, offline_key_centroids, offline_value_centroids


def _quantize_log_error_global(
    errors: torch.Tensor,
    bits: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return integer log-error codes and their dequantized lookup levels."""
    levels = 1 << bits
    logarithm = errors.float().clamp_min(1.0e-12).log2()
    lower = logarithm.amin()
    upper = logarithm.amax()
    if float(upper - lower) <= 1.0e-12:
        codes = torch.zeros_like(logarithm, dtype=torch.uint8)
        reconstruction = lower.exp2().repeat(levels)
        return codes, reconstruction
    step = (upper - lower) / float(levels - 1)
    codes = torch.round((logarithm - lower) / step).clamp(0, levels - 1)
    reconstruction = (lower + torch.arange(
        levels, device=errors.device, dtype=torch.float32
    ) * step).exp2()
    return codes.to(torch.uint8), reconstruction


def _build_risk_lut(
    base_error_levels: torch.Tensor,
    value_error_levels: torch.Tensor,
    value_centroids: torch.Tensor,
    head_dim: int,
    mode: str,
    risk_lambda: float,
) -> torch.Tensor:
    kv_heads = value_centroids.shape[0]
    codes = torch.arange(256, device=value_centroids.device)
    key_codes = codes & 15
    value_codes = codes >> 4
    output = torch.empty(
        kv_heads, JOINT_CLUSTERS, 256,
        dtype=torch.float32,
        device=value_centroids.device,
    )
    # Qwen's per-head q_norm makes ||q|| close to sqrt(D), so this table is
    # stable across decode positions and can remain part of the frozen index.
    normalized_query_norm = math.sqrt(float(head_dim))
    for head in range(kv_heads):
        key_uncertainty = (
            base_error_levels[head].index_select(0, key_codes)
            * normalized_query_norm
            / float(head_dim)
        )
        if mode == "qk_risk":
            output[head] = risk_lambda * key_uncertainty[None, :]
            continue
        if mode != "output_bound":
            raise ValueError(f"unknown risk mode: {mode}")
        value_error = value_error_levels[head].index_select(0, value_codes)
        centroid_norm = value_centroids[head].float().norm(dim=-1)
        sensitivity = value_error[None, :] + key_uncertainty[None, :] * (
            centroid_norm[:, None] + value_error[None, :]
        )
        output[head] = risk_lambda * sensitivity.clamp_min(1.0e-8).log()
    return output


@torch.no_grad()
def build_real_index(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_codebooks: list[dict[str, Any]],
    *,
    key_centroid_bits: int = 8,
    value_centroid_bits: int = 4,
    assignment_chunk_size: int = 4096,
    risk_mode: str = "output_bound",
    risk_lambda: float = 1.0,
) -> JointKVRealIndex:
    """Build one layer's index from [B=1, KVH, N, D] post-RoPE K/V."""
    if key.shape != value.shape or key.ndim != 4 or key.shape[0] != 1:
        raise ValueError("real index expects aligned [1,KVH,N,D] K/V")
    if len(layer_codebooks) != key.shape[1]:
        raise ValueError("one codebook is required for each KV head")
    torch.cuda.synchronize(key.device)
    start = time.perf_counter()
    base_rows = []
    residual_rows = []
    id_rows = []
    risk_rows = []
    query_matrices = []
    value_centroid_rows = []
    base_level_rows = []
    value_level_rows = []
    scale = key.shape[-1] ** -0.5
    for head, state in enumerate(layer_codebooks):
        head_key = key[0, head].float()
        head_value = value[0, head].float()
        key_factor = state["key_factor"].to(key.device).float()
        query_factor = state["query_factor"].to(key.device).float()
        projection = state["projection"].to(key.device).float()
        residual_projection = state["residual_projection"].to(key.device).float()
        coordinates = head_key @ key_factor
        base_codes, base_residual = encode_binary_packed(coordinates, projection)
        assignments, offline_key_centroids, offline_value_centroids = (
            _assign_joint_ids(
                base_residual,
                head_value,
                state["joint_model"],
                assignment_chunk_size,
            )
        )
        key_centroids = _cluster_means(
            base_residual,
            assignments,
            JOINT_CLUSTERS,
            offline_key_centroids,
        )
        value_centroids = _cluster_means(
            head_value,
            assignments,
            JOINT_CLUSTERS,
            offline_value_centroids,
        )
        key_centroids = _symmetric_row_quantize(key_centroids, key_centroid_bits)
        value_centroids = _symmetric_row_quantize(
            value_centroids, value_centroid_bits
        )
        second_residual = base_residual - key_centroids.index_select(
            0, assignments
        )
        residual_codes, final_residual = encode_binary_packed(
            second_residual, residual_projection
        )
        base_errors = second_residual.norm(dim=-1)
        value_errors = (
            head_value - value_centroids.index_select(0, assignments)
        ).norm(dim=-1)
        base_risk, base_levels = _quantize_log_error_global(base_errors)
        value_risk, value_levels = _quantize_log_error_global(value_errors)
        packed_risk = base_risk.bitwise_or(value_risk << 4)

        matrix = torch.zeros(
            key.shape[-1], QUERY_WIDTH,
            dtype=torch.float32,
            device=key.device,
        )
        matrix[:, BASE_OFFSET : BASE_OFFSET + BASE_BITS] = (
            query_factor @ projection.T * scale
        )
        matrix[:, RESIDUAL_OFFSET : RESIDUAL_OFFSET + RESIDUAL_BITS] = (
            query_factor @ residual_projection.T * scale
        )
        matrix[:, JOINT_OFFSET : JOINT_OFFSET + JOINT_CLUSTERS] = (
            query_factor @ key_centroids.T * scale
        )

        base_rows.append(base_codes)
        residual_rows.append(residual_codes)
        id_rows.append(assignments.to(torch.uint8))
        risk_rows.append(packed_risk)
        query_matrices.append(matrix)
        value_centroid_rows.append(value_centroids)
        base_level_rows.append(base_levels)
        value_level_rows.append(value_levels)
    value_centroid_tensor = torch.stack(value_centroid_rows)
    base_level_tensor = torch.stack(base_level_rows)
    value_level_tensor = torch.stack(value_level_rows)
    risk_lut = _build_risk_lut(
        base_level_tensor,
        value_level_tensor,
        value_centroid_tensor,
        key.shape[-1],
        risk_mode,
        risk_lambda,
    )
    torch.cuda.synchronize(key.device)
    elapsed = time.perf_counter() - start
    return JointKVRealIndex(
        base_codes=torch.stack(base_rows).unsqueeze(0).contiguous(),
        residual_codes=torch.stack(residual_rows).unsqueeze(0).contiguous(),
        joint_ids=torch.stack(id_rows).unsqueeze(0).contiguous(),
        risk_codes=torch.stack(risk_rows).unsqueeze(0).contiguous(),
        query_matrix=torch.stack(query_matrices).contiguous(),
        risk_lut=risk_lut.contiguous(),
        value_centroids=value_centroid_tensor.contiguous(),
        base_error_levels=base_level_tensor,
        value_error_levels=value_level_tensor,
        build_seconds=elapsed,
        # 64 base + 48 residual + 6 ID + 4 base risk + 4 value risk.
        logical_bits_per_token_head=126.0,
        physical_bytes_per_token_head=18,
    )


def prepare_packed_query(
    query: torch.Tensor,
    index: JointKVRealIndex,
) -> torch.Tensor:
    """Project [B,QH,D] post-RoPE queries to [B,KVH,G,192]."""
    if query.ndim != 3 or query.shape[0] != 1:
        raise ValueError("query must be [1,QH,D]")
    kv_heads = index.base_codes.shape[1]
    if query.shape[1] % kv_heads:
        raise ValueError("query heads are not divisible by KV heads")
    groups = query.shape[1] // kv_heads
    grouped = query.reshape(1, kv_heads, groups, query.shape[-1])
    return torch.einsum(
        "bhgd,hdw->bhgw", grouped, index.query_matrix.to(query.dtype)
    ).contiguous()

