#!/usr/bin/env python
"""Direct CUDA timing for the first physical JointKV-Sieve prototype.

The benchmark executes every stage in the measured complete path.  It is a
systems timing probe, not a quality claim: the current quality runner and this
CUDA prototype share the index/action contract, while exact numerical parity
will be established after the Pareto-worthy layout is selected.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

import jointkv_sieve_cuda_20260802 as joint_cuda
import qabs_cuda_kernels as sparse_cuda


BASE_OFFSET = 0
RESIDUAL_OFFSET = 64
JOINT_OFFSET = 128
QUERY_WIDTH = 192


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--base_bits", type=int, choices=(48, 64), default=64)
    parser.add_argument("--residual_bits", type=int, choices=(48, 64), default=48)
    parser.add_argument("--refine_fraction", type=float, default=0.20)
    parser.add_argument("--local_base_keep", type=int, default=8)
    parser.add_argument("--local_residual_keep", type=int, default=8)
    parser.add_argument(
        "--sparse_split_count", type=int, choices=(0, 1, 2, 4, 8, 16), default=0
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--paired_iterations", type=int, default=50)
    parser.add_argument("--paired_repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def selected_count(history_tokens: int) -> int:
    return min(history_tokens, 1280, max(256, math.ceil(0.06 * history_tokens)))


def measure_ms(function: Callable[[], object], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    stop.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(stop)) / iterations


def sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_indices: torch.Tensor,
    split_count_override: int = 0,
) -> torch.Tensor:
    query_indices = kv_indices.repeat_interleave(4, dim=1).contiguous()
    counts = torch.full(
        (query.shape[0], query.shape[1]),
        query_indices.shape[-1],
        dtype=torch.long,
        device=query.device,
    )
    split_count = (
        split_count_override
        if split_count_override > 0
        else (8 if query_indices.shape[-1] >= 900 else 2)
    )
    return sparse_cuda.final_attention_ragged_self_split(
        query,
        key,
        value,
        query_indices,
        counts,
        128.0**-0.5,
        split_count,
    )


def validate_cuda_primitives(dtype: torch.dtype, bits: int) -> None:
    batch, heads, groups, tokens = 1, 2, 2, 256
    generator = torch.Generator(device="cuda").manual_seed(11)
    codes = torch.randint(
        -(2**63),
        2**63 - 1,
        (batch, heads, tokens),
        dtype=torch.int64,
        device="cuda",
        generator=generator,
    )
    joint_ids = torch.randint(
        0, 64, codes.shape, dtype=torch.uint8, device="cuda", generator=generator
    )
    risk_codes = torch.randint(
        0, 256, codes.shape, dtype=torch.uint8, device="cuda", generator=generator
    )
    packed_query = (
        torch.randn(
            batch,
            heads,
            groups,
            QUERY_WIDTH,
            device="cuda",
            generator=generator,
        )
        * 0.02
    ).to(dtype)
    risk_lut = torch.randn(
        heads, 64, 256, device="cuda", generator=generator
    ).float() * 0.01
    observed = joint_cuda.base_priority(
        codes,
        joint_ids,
        risk_codes,
        packed_query,
        risk_lut,
        bits=bits,
        probe_offset=BASE_OFFSET,
        joint_offset=JOINT_OFFSET,
    )
    bit_ids = torch.arange(bits, device="cuda", dtype=torch.int64)
    signs = torch.where(
        ((codes[..., None] >> bit_ids) & 1) != 0,
        1.0,
        -1.0,
    )
    probe = packed_query[..., BASE_OFFSET : BASE_OFFSET + bits].float()
    reference = torch.einsum("bhni,bhgi->bhgn", signs, probe)
    joint = packed_query[..., JOINT_OFFSET : JOINT_OFFSET + 64].float()
    gathered_joint = torch.gather(
        joint,
        3,
        joint_ids[:, :, None, :].long().expand(-1, -1, groups, -1),
    )
    reference = (reference + gathered_joint).amax(dim=2)
    risk_reference = risk_lut[
        torch.arange(heads, device="cuda")[None, :, None],
        joint_ids.long(),
        risk_codes.long(),
    ]
    reference = reference + risk_reference
    maximum_error = float((observed - reference).abs().max().item())
    if maximum_error > 2.0e-3:
        raise RuntimeError(f"base-priority CUDA mismatch: {maximum_error}")

    fused_priority, fused_all_mass = joint_cuda.base_priority_and_mass(
        codes,
        joint_ids,
        risk_codes,
        packed_query,
        risk_lut,
        torch.zeros(batch, heads, groups, device="cuda"),
        bits=bits,
        probe_offset=BASE_OFFSET,
        joint_offset=JOINT_OFFSET,
    )
    fused_priority_error = float((fused_priority - reference).abs().max().item())
    if fused_priority_error > 2.0e-3:
        raise RuntimeError(
            f"fused base-priority CUDA mismatch: {fused_priority_error}"
        )

    local_indices, local_scores, local_mass = joint_cuda.base_local_select_and_mass(
        codes,
        joint_ids,
        risk_codes,
        packed_query,
        risk_lut,
        torch.zeros(batch, heads, groups, device="cuda"),
        bits=bits,
        probe_offset=BASE_OFFSET,
        joint_offset=JOINT_OFFSET,
        keep_per_warp=8,
    )
    padded_tokens = math.ceil(tokens / 32) * 32
    padded_reference = F.pad(
        reference, (0, padded_tokens - tokens), value=-float("inf")
    )
    warp_reference = padded_reference.reshape(batch, heads, -1, 32)
    expected_local_scores, expected_local_lanes = torch.topk(
        warp_reference, k=8, dim=-1, sorted=True
    )
    warp_offsets = (
        torch.arange(padded_tokens // 32, device="cuda", dtype=torch.long)
        .reshape(1, 1, -1, 1)
        .mul(32)
    )
    expected_local_indices = expected_local_lanes + warp_offsets
    expected_local_scores = expected_local_scores.flatten(2)
    expected_local_indices = expected_local_indices.flatten(2)
    valid_local = torch.isfinite(expected_local_scores)
    local_score_error = float(
        (local_scores[valid_local] - expected_local_scores[valid_local])
        .abs()
        .max()
        .item()
    )
    if local_score_error > 2.0e-3 or not torch.equal(
        local_indices[valid_local], expected_local_indices[valid_local]
    ):
        local_delta = (local_scores - expected_local_scores).abs().masked_fill(
            ~valid_local, 0.0
        )
        mismatch_flat = int(local_delta.argmax().item())
        mismatch_head = (mismatch_flat // local_scores.shape[-1]) % heads
        mismatch_position = mismatch_flat % local_scores.shape[-1]
        raise RuntimeError(
            "warp-local base selector mismatch: "
            f"score_error={local_score_error}; "
            f"observed_scores={local_scores[0, 0, :8].tolist()}; "
            f"expected_scores={expected_local_scores[0, 0, :8].tolist()}; "
            f"observed_indices={local_indices[0, 0, :8].tolist()}; "
            f"expected_indices={expected_local_indices[0, 0, :8].tolist()}; "
            f"max_at=({mismatch_head},{mismatch_position}); "
            f"observed_at_max={local_scores[0, mismatch_head, mismatch_position].item()}; "
            f"expected_at_max={expected_local_scores[0, mismatch_head, mismatch_position].item()}; "
            f"observed_index_at_max={local_indices[0, mismatch_head, mismatch_position].item()}; "
            f"expected_index_at_max={expected_local_indices[0, mismatch_head, mismatch_position].item()}"
        )

    residual_codes = torch.randint(
        -(2**63),
        2**63 - 1,
        codes.shape,
        dtype=torch.int64,
        device="cuda",
        generator=generator,
    )
    shortlist_indices, shortlist_scores = joint_cuda.residual_local_shortlist(
        residual_codes,
        local_indices,
        local_scores,
        packed_query,
        bits=min(48, bits),
        probe_offset=RESIDUAL_OFFSET,
        keep_per_warp=8,
    )
    residual_bits = min(48, bits)
    residual_bit_ids = torch.arange(
        residual_bits, device="cuda", dtype=torch.int64
    )
    gathered_residual = torch.gather(residual_codes, 2, local_indices)
    residual_signs = torch.where(
        ((gathered_residual[..., None] >> residual_bit_ids) & 1) != 0,
        1.0,
        -1.0,
    )
    residual_probe = packed_query[
        ..., RESIDUAL_OFFSET : RESIDUAL_OFFSET + residual_bits
    ].float()
    residual_reference = torch.einsum(
        "bhci,bhgi->bhgc", residual_signs, residual_probe
    ).amax(dim=2)
    refined_reference = local_scores + residual_reference
    candidate_count = refined_reference.shape[-1]
    padded_candidates = math.ceil(candidate_count / 32) * 32
    refined_reference = F.pad(
        refined_reference,
        (0, padded_candidates - candidate_count),
        value=-float("inf"),
    )
    padded_local_indices = F.pad(
        local_indices, (0, padded_candidates - candidate_count), value=0
    )
    refined_warps = refined_reference.reshape(batch, heads, -1, 32)
    index_warps = padded_local_indices.reshape(batch, heads, -1, 32)
    expected_short_scores, expected_short_lanes = torch.topk(
        refined_warps, k=8, dim=-1, sorted=True
    )
    expected_short_indices = torch.gather(
        index_warps, 3, expected_short_lanes
    ).flatten(2)
    expected_short_scores = expected_short_scores.flatten(2)
    valid_short = torch.isfinite(expected_short_scores)
    short_score_error = float(
        (shortlist_scores[valid_short] - expected_short_scores[valid_short])
        .abs()
        .max()
        .item()
    )
    if short_score_error > 2.0e-3 or not torch.equal(
        shortlist_indices[valid_short], expected_short_indices[valid_short]
    ):
        raise RuntimeError(
            f"warp-local residual selector mismatch: score_error={short_score_error}"
        )

    query_lut = joint_cuda.build_query_lut(
        packed_query,
        base_bits=bits,
        residual_bits=residual_bits,
        base_offset=BASE_OFFSET,
        residual_offset=RESIDUAL_OFFSET,
    )
    lut_indices, lut_scores, lut_mass = (
        joint_cuda.base_lut_local_select_and_mass(
            codes,
            joint_ids,
            risk_codes,
            packed_query,
            query_lut,
            risk_lut,
            torch.zeros(batch, heads, groups, device="cuda"),
            base_chunks=bits // 8,
            joint_offset=JOINT_OFFSET,
            keep_per_warp=8,
        )
    )
    lut_base_score_error = float((lut_scores - local_scores).abs().max().item())
    lut_mass_error = float(
        ((lut_mass - local_mass).abs() / local_mass.clamp_min(1.0))
        .max()
        .item()
    )
    lut_base_index_overlap = float(
        torch.stack(
            [
                torch.isin(lut_indices[0, head], local_indices[0, head])
                .float()
                .mean()
                for head in range(heads)
            ]
        )
        .mean()
        .item()
    )
    if (
        lut_base_score_error > 2.0e-3
        or lut_mass_error > 2.0e-3
        or lut_base_index_overlap < 0.98
    ):
        raise RuntimeError(
            "query-LUT base selector mismatch: "
            f"score_error={lut_base_score_error}, mass_error={lut_mass_error}, "
            f"index_overlap={lut_base_index_overlap}"
        )
    lut_short_indices, lut_short_scores = (
        joint_cuda.residual_lut_local_shortlist(
            residual_codes,
            lut_indices,
            lut_scores,
            query_lut,
            residual_chunk_offset=bits // 8,
            residual_chunks=residual_bits // 8,
            keep_per_warp=8,
        )
    )
    lut_residual_score_error = float(
        (lut_short_scores - shortlist_scores).abs().max().item()
    )
    lut_residual_index_overlap = float(
        torch.stack(
            [
                torch.isin(
                    lut_short_indices[0, head], shortlist_indices[0, head]
                )
                .float()
                .mean()
                for head in range(heads)
            ]
        )
        .mean()
        .item()
    )
    if (
        lut_residual_score_error > 2.0e-3
        or lut_residual_index_overlap < 0.98
    ):
        raise RuntimeError(
            "query-LUT residual selector mismatch: "
            f"score_error={lut_residual_score_error}, "
            f"index_overlap={lut_residual_index_overlap}"
        )

    mask = torch.zeros_like(joint_ids)
    mask[..., :13] = 1
    references = torch.zeros(batch, heads, groups, device="cuda")
    observed_mass = joint_cuda.tail_cluster_mass(
        codes,
        joint_ids,
        packed_query,
        mask,
        references,
        bits=bits,
        probe_offset=BASE_OFFSET,
        joint_offset=JOINT_OFFSET,
        blocks_per_query=3,
    )
    per_query_score = torch.einsum("bhni,bhgi->bhgn", signs, probe)
    per_query_score = per_query_score + gathered_joint
    weights = per_query_score.exp() * (1.0 - mask[:, :, None, :].float())
    expected_mass = torch.zeros_like(observed_mass)
    expected_mass.scatter_add_(
        3,
        joint_ids[:, :, None, :].long().expand(-1, -1, groups, -1),
        weights,
    )
    mass_error = float(
        ((observed_mass - expected_mass).abs() / expected_mass.clamp_min(1.0)).max().item()
    )
    if mass_error > 2.0e-3:
        raise RuntimeError(f"tail-mass CUDA mismatch: {mass_error}")

    all_weights = per_query_score.exp()
    expected_all_mass = torch.zeros_like(fused_all_mass)
    expected_all_mass.scatter_add_(
        3,
        joint_ids[:, :, None, :].long().expand(-1, -1, groups, -1),
        all_weights,
    )
    fused_mass_error = float(
        (
            (fused_all_mass - expected_all_mass).abs()
            / expected_all_mass.clamp_min(1.0)
        ).max().item()
    )
    if fused_mass_error > 2.0e-3:
        raise RuntimeError(f"fused all-mass CUDA mismatch: {fused_mass_error}")
    local_mass_error = float(
        ((local_mass - expected_all_mass).abs() / expected_all_mass.clamp_min(1.0))
        .max()
        .item()
    )
    if local_mass_error > 2.0e-3:
        raise RuntimeError(f"local-select all-mass CUDA mismatch: {local_mass_error}")

    selected_indices = torch.arange(13, device="cuda").reshape(1, 1, 13)
    selected_indices = selected_indices.expand(batch, heads, -1).contiguous()
    omitted_mass = joint_cuda.subtract_selected_mass(
        fused_all_mass.clone(),
        codes,
        joint_ids,
        packed_query,
        selected_indices,
        references,
        bits=bits,
        probe_offset=BASE_OFFSET,
        joint_offset=JOINT_OFFSET,
    )
    omitted_mass_lut = joint_cuda.subtract_selected_mass_lut(
        fused_all_mass.clone(),
        codes,
        joint_ids,
        packed_query,
        query_lut,
        selected_indices,
        references,
        base_chunks=bits // 8,
        joint_offset=JOINT_OFFSET,
    )
    subtract_error = float(
        ((omitted_mass - expected_mass).abs() / expected_mass.clamp_min(1.0))
        .max()
        .item()
    )
    if subtract_error > 2.0e-3:
        raise RuntimeError(f"selected-mass subtraction mismatch: {subtract_error}")
    subtract_lut_error = float(
        ((omitted_mass_lut - expected_mass).abs() / expected_mass.clamp_min(1.0))
        .max()
        .item()
    )
    if subtract_lut_error > 2.0e-3:
        raise RuntimeError(
            f"query-LUT selected-mass subtraction mismatch: {subtract_lut_error}"
        )

    sparse = torch.randn(
        batch, 1, heads * groups, 128, device="cuda", generator=generator
    ).to(dtype)
    centroids = torch.randn(
        heads, 64, 128, device="cuda", generator=generator
    ).float()
    alpha = 0.2
    blended = joint_cuda.tail_blend(sparse, omitted_mass, centroids, alpha)
    tail_numerator = torch.einsum("bhgc,hcd->bhgd", omitted_mass, centroids)
    tail_reference = tail_numerator / omitted_mass.sum(
        dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    blend_reference = alpha * sparse.float() + (1.0 - alpha) * tail_reference.reshape(
        batch, 1, heads * groups, 128
    )
    blend_error = float((blended.float() - blend_reference).abs().max().item())
    if blend_error > 2.0e-2:
        raise RuntimeError(f"fused tail-blend mismatch: {blend_error}")


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.length <= 0:
        raise ValueError("length must be positive")
    if not 0.0 < args.refine_fraction <= 1.0:
        raise ValueError("refine fraction must lie in (0,1]")
    if not 1 <= args.local_base_keep <= 32:
        raise ValueError("local base keep must lie in [1,32]")
    if not 1 <= args.local_residual_keep <= 32:
        raise ValueError("local residual keep must lie in [1,32]")
    if args.paired_iterations <= 0 or args.paired_repeats <= 0:
        raise ValueError("paired iterations and repeats must be positive")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    torch.manual_seed(args.seed)
    validate_cuda_primitives(dtype, args.base_bits)

    history_tokens = args.length
    exact_count = selected_count(history_tokens)
    refine_count = min(
        history_tokens,
        max(exact_count, math.ceil(args.refine_fraction * history_tokens / 256) * 256),
    )
    iterations = min(args.iterations, 12 if history_tokens >= 131072 else args.iterations)
    warmup = min(args.warmup, iterations)

    query = torch.randn(1, 32, 128, dtype=dtype, device="cuda")
    grouped_query = query.reshape(1, 8, 4, 128)
    query_matrix = torch.randn(8, 128, QUERY_WIDTH, dtype=dtype, device="cuda") * 0.02
    key = torch.randn(1, 8, history_tokens, 128, dtype=dtype, device="cuda")
    value = torch.randn_like(key)
    base_codes = torch.randint(
        -(2**63), 2**63 - 1, (1, 8, history_tokens), dtype=torch.int64, device="cuda"
    )
    residual_codes = torch.randint(
        -(2**63), 2**63 - 1, (1, 8, history_tokens), dtype=torch.int64, device="cuda"
    )
    joint_ids = torch.randint(
        0, 64, base_codes.shape, dtype=torch.uint8, device="cuda"
    )
    risk_codes = torch.randint(
        0, 256, base_codes.shape, dtype=torch.uint8, device="cuda"
    )
    residual_risk_codes = torch.zeros_like(risk_codes)
    risk_lut = torch.rand(8, 64, 256, dtype=torch.float32, device="cuda") * 0.1
    zero_risk_lut = torch.zeros_like(risk_lut)
    value_centroids = torch.randn(8, 64, 128, dtype=torch.float32, device="cuda")
    selected_mask = torch.zeros_like(joint_ids)
    references = torch.zeros(1, 8, 4, dtype=torch.float32, device="cuda")

    def query_prepare() -> torch.Tensor:
        return torch.einsum("bhgd,hdw->bhgw", grouped_query, query_matrix).contiguous()

    packed_query = query_prepare()

    def query_lut_build(
        current_query: torch.Tensor = packed_query,
    ) -> torch.Tensor:
        return joint_cuda.build_query_lut(
            current_query,
            base_bits=args.base_bits,
            residual_bits=args.residual_bits,
            base_offset=BASE_OFFSET,
            residual_offset=RESIDUAL_OFFSET,
        )

    query_lut = query_lut_build()

    def base_scan(current_query: torch.Tensor = packed_query) -> torch.Tensor:
        return joint_cuda.base_priority(
            base_codes,
            joint_ids,
            risk_codes,
            current_query,
            risk_lut,
            bits=args.base_bits,
            probe_offset=BASE_OFFSET,
            joint_offset=JOINT_OFFSET,
        )

    base_scores = base_scan()

    def refine_select() -> torch.Tensor:
        return torch.topk(base_scores, k=refine_count, dim=-1, sorted=False).indices

    refine_indices = refine_select()

    def residual_gather() -> torch.Tensor:
        return torch.gather(residual_codes, 2, refine_indices)

    gathered_residual_codes = residual_gather()
    gathered_ids = torch.gather(joint_ids, 2, refine_indices)
    gathered_risks = torch.gather(residual_risk_codes, 2, refine_indices)

    def residual_scan(current_query: torch.Tensor = packed_query) -> torch.Tensor:
        return joint_cuda.base_priority(
            gathered_residual_codes,
            gathered_ids,
            gathered_risks,
            current_query,
            zero_risk_lut,
            bits=args.residual_bits,
            probe_offset=RESIDUAL_OFFSET,
            joint_offset=-1,
        )

    residual_scores = residual_scan()
    base_refined_scores = torch.gather(base_scores, 2, refine_indices)
    refined_scores = base_refined_scores + residual_scores

    def exact_rerank() -> torch.Tensor:
        local = torch.topk(
            refined_scores, k=exact_count, dim=-1, sorted=False
        ).indices
        return torch.gather(refine_indices, 2, local)

    exact_indices = exact_rerank()

    def exact_sparse_attention() -> torch.Tensor:
        return sparse_attention(
            query, key, value, exact_indices, args.sparse_split_count
        )

    sparse_output = exact_sparse_attention()

    def tail_mass() -> torch.Tensor:
        selected_mask.zero_()
        selected_mask.scatter_(2, exact_indices, 1)
        return joint_cuda.tail_cluster_mass(
            base_codes,
            joint_ids,
            packed_query,
            selected_mask,
            references,
            bits=args.base_bits,
            probe_offset=BASE_OFFSET,
            joint_offset=JOINT_OFFSET,
            blocks_per_query=min(64, max(1, math.ceil(history_tokens / 2048))),
        )

    cluster_mass = tail_mass()

    def fused_base_scan_mass(
        current_query: torch.Tensor = packed_query,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return joint_cuda.base_priority_and_mass(
            base_codes,
            joint_ids,
            risk_codes,
            current_query,
            risk_lut,
            references,
            bits=args.base_bits,
            probe_offset=BASE_OFFSET,
            joint_offset=JOINT_OFFSET,
        )

    fused_base_scores, all_cluster_mass = fused_base_scan_mass()

    def selected_mass_subtract() -> torch.Tensor:
        return joint_cuda.subtract_selected_mass(
            all_cluster_mass.clone(),
            base_codes,
            joint_ids,
            packed_query,
            exact_indices,
            references,
            bits=args.base_bits,
            probe_offset=BASE_OFFSET,
            joint_offset=JOINT_OFFSET,
        )

    omitted_cluster_mass = selected_mass_subtract()

    def local_base_select_mass(
        current_query: torch.Tensor = packed_query,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return joint_cuda.base_local_select_and_mass(
            base_codes,
            joint_ids,
            risk_codes,
            current_query,
            risk_lut,
            references,
            bits=args.base_bits,
            probe_offset=BASE_OFFSET,
            joint_offset=JOINT_OFFSET,
            keep_per_warp=args.local_base_keep,
        )

    local_base_indices, local_base_scores, local_all_mass = local_base_select_mass()

    def local_residual_shortlist(
        current_query: torch.Tensor = packed_query,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return joint_cuda.residual_local_shortlist(
            residual_codes,
            local_base_indices,
            local_base_scores,
            current_query,
            bits=args.residual_bits,
            probe_offset=RESIDUAL_OFFSET,
            keep_per_warp=args.local_residual_keep,
        )

    local_shortlist_indices, local_shortlist_scores = local_residual_shortlist()
    if local_shortlist_scores.shape[-1] < exact_count:
        raise ValueError(
            "local shortlist is smaller than the requested exact-token budget: "
            f"{local_shortlist_scores.shape[-1]} < {exact_count}"
        )

    def local_final_select() -> torch.Tensor:
        local = torch.topk(
            local_shortlist_scores, k=exact_count, dim=-1, sorted=False
        ).indices
        return torch.gather(local_shortlist_indices, 2, local)

    local_exact_indices = local_final_select()
    local_selection_overlap = float(
        torch.stack(
            [
                torch.isin(local_exact_indices[0, head], exact_indices[0, head])
                .float()
                .mean()
                for head in range(local_exact_indices.shape[1])
            ]
        )
        .mean()
        .item()
    )

    def lut_base_select_mass(
        current_query: torch.Tensor = packed_query,
        current_lut: torch.Tensor = query_lut,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return joint_cuda.base_lut_local_select_and_mass(
            base_codes,
            joint_ids,
            risk_codes,
            current_query,
            current_lut,
            risk_lut,
            references,
            base_chunks=args.base_bits // 8,
            joint_offset=JOINT_OFFSET,
            keep_per_warp=args.local_base_keep,
        )

    lut_base_indices, lut_base_scores, lut_all_mass = lut_base_select_mass()

    def lut_residual_shortlist(
        current_lut: torch.Tensor = query_lut,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return joint_cuda.residual_lut_local_shortlist(
            residual_codes,
            lut_base_indices,
            lut_base_scores,
            current_lut,
            residual_chunk_offset=args.base_bits // 8,
            residual_chunks=args.residual_bits // 8,
            keep_per_warp=args.local_residual_keep,
        )

    lut_shortlist_indices, lut_shortlist_scores = lut_residual_shortlist()

    def lut_final_select() -> torch.Tensor:
        local = torch.topk(
            lut_shortlist_scores, k=exact_count, dim=-1, sorted=False
        ).indices
        return torch.gather(lut_shortlist_indices, 2, local)

    lut_exact_indices = lut_final_select()

    def lut_selected_mass_subtract() -> torch.Tensor:
        return joint_cuda.subtract_selected_mass_lut(
            lut_all_mass.clone(),
            base_codes,
            joint_ids,
            packed_query,
            query_lut,
            lut_exact_indices,
            references,
            base_chunks=args.base_bits // 8,
            joint_offset=JOINT_OFFSET,
        )

    lut_omitted_mass = lut_selected_mass_subtract()
    lut_selection_overlap = float(
        torch.stack(
            [
                torch.isin(lut_exact_indices[0, head], exact_indices[0, head])
                .float()
                .mean()
                for head in range(lut_exact_indices.shape[1])
            ]
        )
        .mean()
        .item()
    )
    lut_selection_overlap_vs_local = float(
        torch.stack(
            [
                torch.isin(
                    lut_exact_indices[0, head], local_exact_indices[0, head]
                )
                .float()
                .mean()
                for head in range(lut_exact_indices.shape[1])
            ]
        )
        .mean()
        .item()
    )

    def tail_reduce(current_mass: torch.Tensor = cluster_mass) -> torch.Tensor:
        numerator = torch.einsum("bhgc,hcd->bhgd", current_mass, value_centroids)
        return numerator / current_mass.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)

    tail_output = tail_reduce()

    def blend_outputs(
        current_sparse: torch.Tensor = sparse_output,
        current_tail: torch.Tensor = tail_output,
    ) -> torch.Tensor:
        alpha = exact_count / history_tokens
        tail_q = current_tail.reshape(1, 1, 32, 128).to(current_sparse.dtype)
        return alpha * current_sparse + (1.0 - alpha) * tail_q

    def fused_tail_blend() -> torch.Tensor:
        return joint_cuda.tail_blend(
            sparse_output,
            omitted_cluster_mass,
            value_centroids,
            exact_count / history_tokens,
        )

    def selection_complete() -> tuple[torch.Tensor, torch.Tensor]:
        current_query = query_prepare()
        current_base = base_scan(current_query)
        current_refine_indices = torch.topk(
            current_base, k=refine_count, dim=-1, sorted=False
        ).indices
        current_residual_codes = torch.gather(
            residual_codes, 2, current_refine_indices
        )
        current_ids = torch.gather(joint_ids, 2, current_refine_indices)
        current_risks = torch.gather(
            residual_risk_codes, 2, current_refine_indices
        )
        current_residual = joint_cuda.base_priority(
            current_residual_codes,
            current_ids,
            current_risks,
            current_query,
            zero_risk_lut,
            bits=args.residual_bits,
            probe_offset=RESIDUAL_OFFSET,
            joint_offset=-1,
        )
        current_refined = torch.gather(
            current_base, 2, current_refine_indices
        ) + current_residual
        local = torch.topk(
            current_refined, k=exact_count, dim=-1, sorted=False
        ).indices
        return torch.gather(current_refine_indices, 2, local), current_query

    def attention_complete() -> torch.Tensor:
        current_indices, current_query = selection_complete()
        current_sparse = sparse_attention(
            query, key, value, current_indices, args.sparse_split_count
        )
        selected_mask.zero_()
        selected_mask.scatter_(2, current_indices, 1)
        current_mass = joint_cuda.tail_cluster_mass(
            base_codes,
            joint_ids,
            current_query,
            selected_mask,
            references,
            bits=args.base_bits,
            probe_offset=BASE_OFFSET,
            joint_offset=JOINT_OFFSET,
            blocks_per_query=min(64, max(1, math.ceil(history_tokens / 2048))),
        )
        current_tail = tail_reduce(current_mass)
        alpha = exact_count / history_tokens
        return alpha * current_sparse + (1.0 - alpha) * current_tail.reshape(
            1, 1, 32, 128
        ).to(current_sparse.dtype)

    def attention_complete_fused_tail() -> torch.Tensor:
        current_query = query_prepare()
        current_base, current_mass = joint_cuda.base_priority_and_mass(
            base_codes,
            joint_ids,
            risk_codes,
            current_query,
            risk_lut,
            references,
            bits=args.base_bits,
            probe_offset=BASE_OFFSET,
            joint_offset=JOINT_OFFSET,
        )
        current_refine_indices = torch.topk(
            current_base, k=refine_count, dim=-1, sorted=False
        ).indices
        current_residual_codes = torch.gather(
            residual_codes, 2, current_refine_indices
        )
        current_ids = torch.gather(joint_ids, 2, current_refine_indices)
        current_risks = torch.gather(
            residual_risk_codes, 2, current_refine_indices
        )
        current_residual = joint_cuda.base_priority(
            current_residual_codes,
            current_ids,
            current_risks,
            current_query,
            zero_risk_lut,
            bits=args.residual_bits,
            probe_offset=RESIDUAL_OFFSET,
            joint_offset=-1,
        )
        current_refined = torch.gather(
            current_base, 2, current_refine_indices
        ) + current_residual
        local = torch.topk(
            current_refined, k=exact_count, dim=-1, sorted=False
        ).indices
        current_indices = torch.gather(current_refine_indices, 2, local)
        current_sparse = sparse_attention(
            query, key, value, current_indices, args.sparse_split_count
        )
        current_mass = joint_cuda.subtract_selected_mass(
            current_mass,
            base_codes,
            joint_ids,
            current_query,
            current_indices,
            references,
            bits=args.base_bits,
            probe_offset=BASE_OFFSET,
            joint_offset=JOINT_OFFSET,
        )
        return joint_cuda.tail_blend(
            current_sparse,
            current_mass,
            value_centroids,
            exact_count / history_tokens,
        )

    def selection_complete_local() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current_query = query_prepare()
        current_base_indices, current_base_scores, current_mass = (
            joint_cuda.base_local_select_and_mass(
                base_codes,
                joint_ids,
                risk_codes,
                current_query,
                risk_lut,
                references,
                bits=args.base_bits,
                probe_offset=BASE_OFFSET,
                joint_offset=JOINT_OFFSET,
                keep_per_warp=args.local_base_keep,
            )
        )
        current_short_indices, current_short_scores = (
            joint_cuda.residual_local_shortlist(
                residual_codes,
                current_base_indices,
                current_base_scores,
                current_query,
                bits=args.residual_bits,
                probe_offset=RESIDUAL_OFFSET,
                keep_per_warp=args.local_residual_keep,
            )
        )
        local = torch.topk(
            current_short_scores, k=exact_count, dim=-1, sorted=False
        ).indices
        current_indices = torch.gather(current_short_indices, 2, local)
        return current_indices, current_query, current_mass

    def attention_complete_local() -> torch.Tensor:
        current_indices, current_query, current_mass = selection_complete_local()
        current_sparse = sparse_attention(
            query, key, value, current_indices, args.sparse_split_count
        )
        current_mass = joint_cuda.subtract_selected_mass(
            current_mass,
            base_codes,
            joint_ids,
            current_query,
            current_indices,
            references,
            bits=args.base_bits,
            probe_offset=BASE_OFFSET,
            joint_offset=JOINT_OFFSET,
        )
        return joint_cuda.tail_blend(
            current_sparse,
            current_mass,
            value_centroids,
            exact_count / history_tokens,
        )

    def selection_complete_lut() -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        current_query = query_prepare()
        current_lut = query_lut_build(current_query)
        current_base_indices, current_base_scores, current_mass = (
            joint_cuda.base_lut_local_select_and_mass(
                base_codes,
                joint_ids,
                risk_codes,
                current_query,
                current_lut,
                risk_lut,
                references,
                base_chunks=args.base_bits // 8,
                joint_offset=JOINT_OFFSET,
                keep_per_warp=args.local_base_keep,
            )
        )
        current_short_indices, current_short_scores = (
            joint_cuda.residual_lut_local_shortlist(
                residual_codes,
                current_base_indices,
                current_base_scores,
                current_lut,
                residual_chunk_offset=args.base_bits // 8,
                residual_chunks=args.residual_bits // 8,
                keep_per_warp=args.local_residual_keep,
            )
        )
        local = torch.topk(
            current_short_scores, k=exact_count, dim=-1, sorted=False
        ).indices
        current_indices = torch.gather(current_short_indices, 2, local)
        return current_indices, current_query, current_lut, current_mass

    def attention_complete_lut() -> torch.Tensor:
        current_indices, current_query, current_lut, current_mass = (
            selection_complete_lut()
        )
        current_sparse = sparse_attention(
            query, key, value, current_indices, args.sparse_split_count
        )
        current_mass = joint_cuda.subtract_selected_mass(
            current_mass,
            base_codes,
            joint_ids,
            current_query,
            current_indices,
            references,
            bits=args.base_bits,
            probe_offset=BASE_OFFSET,
            joint_offset=JOINT_OFFSET,
        )
        return joint_cuda.tail_blend(
            current_sparse,
            current_mass,
            value_centroids,
            exact_count / history_tokens,
        )

    full_key = key.repeat_interleave(4, dim=1)
    full_value = value.repeat_interleave(4, dim=1)

    def full_attention() -> torch.Tensor:
        return F.scaled_dot_product_attention(
            query.unsqueeze(2), full_key, full_value, is_causal=False
        )

    dense_iterations = min(8, iterations)
    dense_warmup = min(4, warmup)
    full_ms = measure_ms(full_attention, dense_warmup, dense_iterations)
    query_ms = measure_ms(query_prepare, warmup, iterations)
    base_ms = measure_ms(base_scan, warmup, iterations)
    refine_topk_ms = measure_ms(refine_select, warmup, iterations)
    residual_gather_ms = measure_ms(residual_gather, warmup, iterations)
    residual_scan_ms = measure_ms(residual_scan, warmup, iterations)
    exact_rerank_ms = measure_ms(exact_rerank, warmup, iterations)
    exact_attention_ms = measure_ms(exact_sparse_attention, warmup, iterations)
    tail_mass_ms = measure_ms(tail_mass, warmup, iterations)
    tail_reduce_ms = measure_ms(tail_reduce, warmup, iterations)
    blend_ms = measure_ms(blend_outputs, warmup, iterations)
    fused_base_mass_ms = measure_ms(fused_base_scan_mass, warmup, iterations)
    selected_subtract_ms = measure_ms(selected_mass_subtract, warmup, iterations)
    fused_tail_blend_ms = measure_ms(fused_tail_blend, warmup, iterations)
    selection_ms = measure_ms(selection_complete, warmup, iterations)
    complete_ms = measure_ms(attention_complete, warmup, iterations)
    fused_complete_ms = measure_ms(
        attention_complete_fused_tail, warmup, iterations
    )
    local_base_mass_ms = measure_ms(local_base_select_mass, warmup, iterations)
    local_residual_ms = measure_ms(local_residual_shortlist, warmup, iterations)
    local_final_select_ms = measure_ms(local_final_select, warmup, iterations)
    local_selection_ms = measure_ms(
        selection_complete_local, warmup, iterations
    )
    local_complete_ms = measure_ms(attention_complete_local, warmup, iterations)
    query_lut_ms = measure_ms(query_lut_build, warmup, iterations)
    lut_base_mass_ms = measure_ms(lut_base_select_mass, warmup, iterations)
    lut_residual_ms = measure_ms(lut_residual_shortlist, warmup, iterations)
    lut_final_select_ms = measure_ms(lut_final_select, warmup, iterations)
    lut_selected_subtract_ms = measure_ms(
        lut_selected_mass_subtract, warmup, iterations
    )
    lut_selection_ms = measure_ms(selection_complete_lut, warmup, iterations)
    lut_complete_ms = measure_ms(attention_complete_lut, warmup, iterations)
    paired_full_ms: list[float] = []
    paired_lut_ms: list[float] = []
    paired_warmup = min(6, max(1, args.paired_iterations // 5))
    for repeat in range(args.paired_repeats):
        if repeat % 2 == 0:
            paired_full_ms.append(
                measure_ms(full_attention, paired_warmup, args.paired_iterations)
            )
            paired_lut_ms.append(
                measure_ms(
                    attention_complete_lut,
                    paired_warmup,
                    args.paired_iterations,
                )
            )
        else:
            paired_lut_ms.append(
                measure_ms(
                    attention_complete_lut,
                    paired_warmup,
                    args.paired_iterations,
                )
            )
            paired_full_ms.append(
                measure_ms(full_attention, paired_warmup, args.paired_iterations)
            )
    paired_speedups = [
        full / sparse for full, sparse in zip(paired_full_ms, paired_lut_ms)
    ]

    row = {
        "history_tokens": history_tokens,
        "base_bits": args.base_bits,
        "residual_bits": args.residual_bits,
        "refine_tokens_per_kv_head": refine_count,
        "refine_fraction": refine_count / history_tokens,
        "exact_tokens_per_kv_head": exact_count,
        "exact_fraction": exact_count / history_tokens,
        "sparse_split_count": (
            args.sparse_split_count
            if args.sparse_split_count > 0
            else (8 if exact_count >= 900 else 2)
        ),
        "full_preexpanded_sdpa_direct_ms": full_ms,
        "query_prepare_direct_ms": query_ms,
        "base_priority_scan_direct_ms": base_ms,
        "refine_topk_direct_ms": refine_topk_ms,
        "conditional_residual_gather_direct_ms": residual_gather_ms,
        "conditional_residual_scan_direct_ms": residual_scan_ms,
        "fixed_cost_exact_rerank_direct_ms": exact_rerank_ms,
        "selection_complete_direct_ms": selection_ms,
        "exact_sparse_attention_direct_ms": exact_attention_ms,
        "tail_cluster_mass_direct_ms": tail_mass_ms,
        "tail_centroid_reduce_direct_ms": tail_reduce_ms,
        "tail_blend_direct_ms": blend_ms,
        "attention_complete_direct_ms": complete_ms,
        "fused_base_priority_and_mass_direct_ms": fused_base_mass_ms,
        "selected_mass_subtract_direct_ms": selected_subtract_ms,
        "fused_tail_blend_direct_ms": fused_tail_blend_ms,
        "attention_complete_fused_tail_direct_ms": fused_complete_ms,
        "local_base_candidates_per_kv_head": local_base_indices.shape[-1],
        "local_shortlist_per_kv_head": local_shortlist_indices.shape[-1],
        "local_selection_overlap_vs_global_path": local_selection_overlap,
        "local_base_select_and_mass_direct_ms": local_base_mass_ms,
        "local_residual_shortlist_direct_ms": local_residual_ms,
        "local_final_topk_direct_ms": local_final_select_ms,
        "local_selection_complete_direct_ms": local_selection_ms,
        "attention_complete_local_direct_ms": local_complete_ms,
        "query_lut_build_direct_ms": query_lut_ms,
        "lut_base_select_and_mass_direct_ms": lut_base_mass_ms,
        "lut_residual_shortlist_direct_ms": lut_residual_ms,
        "lut_final_topk_direct_ms": lut_final_select_ms,
        "lut_selected_mass_subtract_direct_ms": lut_selected_subtract_ms,
        "lut_selection_overlap_vs_global_path": lut_selection_overlap,
        "lut_selection_overlap_vs_fp32_local_path": (
            lut_selection_overlap_vs_local
        ),
        "lut_selection_complete_direct_ms": lut_selection_ms,
        "attention_complete_lut_direct_ms": lut_complete_ms,
        "selection_speedup_vs_full": full_ms / selection_ms,
        "attention_speedup_vs_full": full_ms / complete_ms,
        "attention_fused_tail_speedup_vs_full": full_ms / fused_complete_ms,
        "attention_local_speedup_vs_full": full_ms / local_complete_ms,
        "attention_lut_speedup_vs_full": full_ms / lut_complete_ms,
        "paired_full_direct_ms": paired_full_ms,
        "paired_lut_direct_ms": paired_lut_ms,
        "paired_lut_speedups": paired_speedups,
        "paired_lut_speedup_median": statistics.median(paired_speedups),
        "paired_lut_speedup_mean": statistics.fmean(paired_speedups),
    }
    result = {
        "schema": "jointkv_sieve_direct_cuda_stages_v2",
        "hardware": torch.cuda.get_device_name(0),
        "contract": {
            "batch": 1,
            "query_heads": 32,
            "kv_heads": 8,
            "gqa_group_size": 4,
            "head_dimension": 128,
            "dtype": args.dtype,
            "base_index": "Q-aware signed principal code plus joint-ID/risk LUT",
            "action_policy": "base -> conditional residual -> fixed-cost exact rerank",
            "tail": "fused omitted-token mass by 6-bit joint K/V ID",
            "fused_tail_path": (
                "base priority and all-token mass in one scan; subtract "
                "selected proxy mass; fused centroid reduction and blend"
            ),
            "candidate_schedule": "min(N,1280,max(256,ceil(0.06*N)))",
            "refinement_policy": "fixed measured 20% prototype capacity",
            "local_selection_policy": (
                f"keep {args.local_base_keep}/32 base candidates per warp; "
                f"keep {args.local_residual_keep}/32 refined candidates per "
                "candidate warp; global top-k only on the compact shortlist"
            ),
            "query_lut_policy": (
                "rebuild one 256-entry signed-dot LUT per query/head/group/"
                "8-bit code chunk on every measured decode step"
            ),
            "quality_claim": False,
            "stage_sums_used_for_speedup": False,
        },
        "row": row,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
