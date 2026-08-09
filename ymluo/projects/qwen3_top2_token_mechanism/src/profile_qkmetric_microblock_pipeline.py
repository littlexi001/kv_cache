from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import torch

import qabs_cuda_kernels
from run_head_top2_targeted_ppl_20260714 import (
    _pca_int4_candidate_range_scores,
    _pca_int4_qkmetric_microblock_candidates,
)


def timed_ms(callback: Callable[[], object], warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        callback()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        callback()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / repeats)


def extend_history(tensor: torch.Tensor, target_count: int) -> torch.Tensor:
    repeats = math.ceil(target_count / tensor.shape[-2])
    return tensor.repeat(1, 1, repeats, 1)[..., :target_count, :].contiguous()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--history_tokens", type=int, default=127999)
    parser.add_argument("--projection_dim", type=int, default=48)
    parser.add_argument("--candidate_fraction", type=float, default=0.06)
    parser.add_argument("--outer_fraction", type=float, default=0.24)
    parser.add_argument("--block_size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    payload = torch.load(args.trace, map_location="cpu", weights_only=False)
    records = [
        row for row in payload["records"] if int(row["layer"]) == args.layer
    ]
    first = next(row for row in records if row["key"] is not None)
    key = extend_history(first["key"].cuda().half(), args.history_tokens)
    queries = [row["query"].cuda().half().contiguous() for row in records]
    candidate_count = math.ceil(args.candidate_fraction * args.history_tokens)
    state: dict[str, object] = {}
    for query in queries[:4]:
        _pca_int4_qkmetric_microblock_candidates(
            query[:, :, 0],
            key,
            state,
            args.projection_dim,
            candidate_count,
            args.outer_fraction,
            args.block_size,
        )

    query = queries[8][:, :, 0]
    # Refresh query codes/state without changing the index geometry.
    _pca_int4_qkmetric_microblock_candidates(
        query,
        key,
        state,
        args.projection_dim,
        candidate_count,
        args.outer_fraction,
        args.block_size,
    )
    projected_query_native = state["last_projected_query"]
    projected_query = projected_query_native.float()
    batch_count, kv_head_count, group_count, _ = projected_query.shape
    block_count = math.ceil(args.history_tokens / args.block_size)
    mean_native = state["microblock_mean"][..., :block_count, :]
    variance_native = state["microblock_variance"][..., :block_count, :]
    mean = mean_native.float()
    variance = variance_native.float()
    lengths = torch.full(
        (block_count,), args.block_size, dtype=torch.float32, device="cuda"
    )
    lengths[-1] = args.history_tokens - (block_count - 1) * args.block_size
    multiplier = torch.sqrt(2.0 * lengths.clamp_min(2.0).log()).view(
        1, 1, 1, -1
    )

    def block_scores() -> torch.Tensor:
        center = torch.einsum("bhnr,bhgr->bhgn", mean, projected_query)
        sigma = torch.sqrt(
            torch.einsum(
                "bhnr,bhgr->bhgn", variance, projected_query.square()
            ).clamp_min(0.0)
        )
        return (center + sigma * multiplier).reshape(
            batch_count, kv_head_count * group_count, block_count
        )

    multiplier_native = multiplier.to(projected_query_native.dtype)

    def block_scores_native_einsum() -> torch.Tensor:
        center = torch.einsum(
            "bhnr,bhgr->bhgn", mean_native, projected_query_native
        )
        sigma = torch.sqrt(
            torch.einsum(
                "bhnr,bhgr->bhgn",
                variance_native,
                projected_query_native.square(),
            ).clamp_min(0.0)
        )
        return (center + sigma * multiplier_native).reshape(
            batch_count, kv_head_count * group_count, block_count
        )

    def block_scores_native_bmm() -> torch.Tensor:
        rows = batch_count * kv_head_count
        mean_matrix = mean_native.reshape(rows, block_count, -1)
        variance_matrix = variance_native.reshape(rows, block_count, -1)
        query_matrix = projected_query_native.reshape(rows, group_count, -1)
        center = torch.bmm(mean_matrix, query_matrix.transpose(1, 2))
        sigma = torch.sqrt(
            torch.bmm(
                variance_matrix, query_matrix.square().transpose(1, 2)
            ).clamp_min(0.0)
        )
        scores_native = center + sigma * multiplier_native.reshape(1, block_count, 1)
        return scores_native.transpose(1, 2).reshape(
            batch_count, kv_head_count * group_count, block_count
        )

    last_block_size = args.history_tokens - (block_count - 1) * args.block_size

    def block_scores_fused() -> torch.Tensor:
        return qabs_cuda_kernels.microblock_expected_max_scores(
            state["microblock_mean"],
            state["microblock_variance"],
            projected_query_native,
            block_count,
            args.block_size,
            last_block_size,
        )

    scores = block_scores()
    outer_block_count = math.ceil(
        args.outer_fraction * args.history_tokens / args.block_size
    )

    def select_blocks() -> torch.Tensor:
        return torch.topk(
            scores, outer_block_count, dim=-1, sorted=False
        ).indices

    scores_native_einsum = block_scores_native_einsum()
    scores_native_bmm = block_scores_native_bmm()
    scores_fused = block_scores_fused()

    def select_blocks_native_einsum() -> torch.Tensor:
        return torch.topk(
            scores_native_einsum, outer_block_count, dim=-1, sorted=False
        ).indices

    def select_blocks_native_bmm() -> torch.Tensor:
        return torch.topk(
            scores_native_bmm, outer_block_count, dim=-1, sorted=False
        ).indices

    def select_blocks_fused() -> torch.Tensor:
        return torch.topk(
            scores_fused, outer_block_count, dim=-1, sorted=False
        ).indices

    selected_blocks = select_blocks()
    selected_blocks_native_bmm = select_blocks_native_bmm()
    selected_blocks_fused = select_blocks_fused()
    block_mask = torch.zeros_like(scores, dtype=torch.bool)
    block_mask_native = torch.zeros_like(scores, dtype=torch.bool)
    block_mask.scatter_(-1, selected_blocks, True)
    block_mask_native.scatter_(-1, selected_blocks_native_bmm, True)
    native_block_overlap = float(
        (block_mask & block_mask_native).sum().item()
        / max(1, block_mask.sum().item())
    )
    block_mask_fused = torch.zeros_like(scores, dtype=torch.bool)
    block_mask_fused.scatter_(-1, selected_blocks_fused, True)
    fused_block_overlap = float(
        (block_mask & block_mask_fused).sum().item()
        / max(1, block_mask.sum().item())
    )

    offsets = torch.arange(args.block_size, device="cuda").view(1, 1, 1, -1)

    def expand_blocks() -> torch.Tensor:
        return (
            selected_blocks.unsqueeze(-1) * args.block_size + offsets
        ).flatten(-2).clamp_max(args.history_tokens - 1)

    outer_indices = expand_blocks()

    def score_outer() -> torch.Tensor:
        return _pca_int4_candidate_range_scores(
            state,
            outer_indices,
            args.history_tokens,
            0,
            args.projection_dim,
        )

    outer_scores = score_outer()

    def select_candidates() -> tuple[torch.Tensor, torch.Tensor]:
        values, local = torch.topk(
            outer_scores, candidate_count, dim=-1, sorted=False
        )
        return values, torch.gather(outer_indices, -1, local)

    _, candidate_indices = select_candidates()
    query_codes = state["last_projected_query_codes"]
    query_scale = state["last_projected_query_scale"]

    def score_selected_blocks_fused() -> torch.Tensor:
        return qabs_cuda_kernels.pca_int4_logscale16_selected_block_scores(
            query_codes,
            query_scale,
            state["packed_chunked"],
            state["scales"],
            state["logscale_exponents"],
            selected_blocks,
            args.history_tokens,
            args.block_size,
            0,
            args.projection_dim,
        )

    selected_block_scores_fused = score_selected_blocks_fused()
    _, selected_block_local = torch.topk(
        selected_block_scores_fused,
        candidate_count,
        dim=-1,
        sorted=False,
    )
    selected_block_candidates_fused = (
        qabs_cuda_kernels.microblock_local_to_token_indices(
            selected_blocks,
            selected_block_local,
            args.history_tokens,
            args.block_size,
        )
    )
    candidate_mask = torch.zeros(
        (batch_count, kv_head_count * group_count, args.history_tokens),
        dtype=torch.bool,
        device="cuda",
    )
    candidate_mask_fused = torch.zeros_like(candidate_mask)
    candidate_mask.scatter_(-1, candidate_indices, True)
    candidate_mask_fused.scatter_(-1, selected_block_candidates_fused, True)
    fused_candidate_overlap = float(
        (candidate_mask & candidate_mask_fused).sum().item()
        / max(1, candidate_mask.sum().item())
    )

    def global_proxy() -> torch.Tensor:
        return (
            qabs_cuda_kernels.pca_int4_chunked_logscale16_prefix_scores(
                query_codes,
                state["packed_chunked"],
                state["scales"],
                state["logscale_exponents"],
                args.history_tokens,
                args.projection_dim,
            )
            * query_scale.reshape(batch_count, kv_head_count * group_count, 1)
        )

    global_scores = global_proxy()

    def global_topk() -> torch.Tensor:
        return torch.topk(
            global_scores, candidate_count, dim=-1, sorted=False
        ).indices

    stage_callbacks = {
        "block_scores": block_scores,
        "block_scores_native_einsum": block_scores_native_einsum,
        "block_scores_native_bmm": block_scores_native_bmm,
        "block_scores_fused": block_scores_fused,
        "block_topk": select_blocks,
        "block_topk_native_einsum": select_blocks_native_einsum,
        "block_topk_native_bmm": select_blocks_native_bmm,
        "block_topk_fused": select_blocks_fused,
        "block_expand": expand_blocks,
        "outer_logscale_score": score_outer,
        "selected_block_logscale_score_fused": score_selected_blocks_fused,
        "outer_topk_and_gather": select_candidates,
        "global_logscale_score": global_proxy,
        "global_topk": global_topk,
    }
    timings = {
        name: timed_ms(callback, args.warmup, args.repeats)
        for name, callback in stage_callbacks.items()
    }

    def hierarchical_retrieval() -> tuple[torch.Tensor, torch.Tensor]:
        current_scores = block_scores()
        current_blocks = torch.topk(
            current_scores, outer_block_count, dim=-1, sorted=False
        ).indices
        current_indices = (
            current_blocks.unsqueeze(-1) * args.block_size + offsets
        ).flatten(-2).clamp_max(args.history_tokens - 1)
        current_outer_scores = _pca_int4_candidate_range_scores(
            state,
            current_indices,
            args.history_tokens,
            0,
            args.projection_dim,
        )
        values, local = torch.topk(
            current_outer_scores, candidate_count, dim=-1, sorted=False
        )
        return values, torch.gather(current_indices, -1, local)

    def hierarchical_retrieval_native_bmm() -> tuple[torch.Tensor, torch.Tensor]:
        current_scores = block_scores_native_bmm()
        current_blocks = torch.topk(
            current_scores, outer_block_count, dim=-1, sorted=False
        ).indices
        current_indices = (
            current_blocks.unsqueeze(-1) * args.block_size + offsets
        ).flatten(-2).clamp_max(args.history_tokens - 1)
        current_outer_scores = _pca_int4_candidate_range_scores(
            state,
            current_indices,
            args.history_tokens,
            0,
            args.projection_dim,
        )
        values, local = torch.topk(
            current_outer_scores, candidate_count, dim=-1, sorted=False
        )
        return values, torch.gather(current_indices, -1, local)

    def hierarchical_retrieval_fused() -> tuple[torch.Tensor, torch.Tensor]:
        current_scores = block_scores_fused()
        current_blocks = torch.topk(
            current_scores, outer_block_count, dim=-1, sorted=False
        ).indices
        current_indices = (
            current_blocks.unsqueeze(-1) * args.block_size + offsets
        ).flatten(-2).clamp_max(args.history_tokens - 1)
        current_outer_scores = _pca_int4_candidate_range_scores(
            state,
            current_indices,
            args.history_tokens,
            0,
            args.projection_dim,
        )
        values, local = torch.topk(
            current_outer_scores, candidate_count, dim=-1, sorted=False
        )
        return values, torch.gather(current_indices, -1, local)

    def hierarchical_retrieval_fully_fused() -> tuple[torch.Tensor, torch.Tensor]:
        current_scores = block_scores_fused()
        current_blocks = torch.topk(
            current_scores, outer_block_count, dim=-1, sorted=False
        ).indices
        current_outer_scores = (
            qabs_cuda_kernels.pca_int4_logscale16_selected_block_scores(
                query_codes,
                query_scale,
                state["packed_chunked"],
                state["scales"],
                state["logscale_exponents"],
                current_blocks,
                args.history_tokens,
                args.block_size,
                0,
                args.projection_dim,
            )
        )
        values, local = torch.topk(
            current_outer_scores, candidate_count, dim=-1, sorted=False
        )
        indices = qabs_cuda_kernels.microblock_local_to_token_indices(
            current_blocks,
            local,
            args.history_tokens,
            args.block_size,
        )
        return values, indices

    def global_retrieval() -> tuple[torch.Tensor, torch.Tensor]:
        current = global_proxy()
        return torch.topk(current, candidate_count, dim=-1, sorted=False)

    timings["hierarchical_retrieval"] = timed_ms(
        hierarchical_retrieval, args.warmup, args.repeats
    )
    timings["hierarchical_retrieval_native_bmm"] = timed_ms(
        hierarchical_retrieval_native_bmm, args.warmup, args.repeats
    )
    timings["hierarchical_retrieval_fused"] = timed_ms(
        hierarchical_retrieval_fused, args.warmup, args.repeats
    )
    timings["hierarchical_retrieval_fully_fused"] = timed_ms(
        hierarchical_retrieval_fully_fused, args.warmup, args.repeats
    )
    timings["global_retrieval"] = timed_ms(
        global_retrieval, args.warmup, args.repeats
    )
    report = {
        "config": vars(args)
        | {"trace": str(args.trace), "output": str(args.output)},
        "hardware": torch.cuda.get_device_name(),
        "timings_ms": timings,
        "hierarchical_speedup_vs_global_retrieval": (
            timings["global_retrieval"] / timings["hierarchical_retrieval"]
        ),
        "native_hierarchical_speedup_vs_global_retrieval": (
            timings["global_retrieval"]
            / timings["hierarchical_retrieval_native_bmm"]
        ),
        "fused_hierarchical_speedup_vs_global_retrieval": (
            timings["global_retrieval"]
            / timings["hierarchical_retrieval_fused"]
        ),
        "fully_fused_hierarchical_speedup_vs_global_retrieval": (
            timings["global_retrieval"]
            / timings["hierarchical_retrieval_fully_fused"]
        ),
        "native_block_score_max_abs_error": float(
            (scores_native_bmm.float() - scores).abs().max().item()
        ),
        "native_selected_block_overlap": native_block_overlap,
        "fused_block_score_max_abs_error": float(
            (scores_fused - scores).abs().max().item()
        ),
        "fused_selected_block_overlap": fused_block_overlap,
        "selected_block_score_max_abs_error": float(
            (selected_block_scores_fused - outer_scores).abs().max().item()
        ),
        "selected_block_candidate_overlap": fused_candidate_overlap,
        "candidate_count": candidate_count,
        "outer_token_count": int(outer_indices.shape[-1]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
