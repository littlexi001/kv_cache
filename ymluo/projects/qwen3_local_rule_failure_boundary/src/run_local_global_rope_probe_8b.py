from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import statistics
import time
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

import run_length_causal_mechanism_20260717 as causal
import run_local_rule_failure_boundary as base
import run_rope_retrieval_repair_8b as rope_repair


VARIANTS = (
    "full_rope",
    "rope_top2",
    "semantic_top2_postscore",
    "local_global_raw",
    "local_global_calibrated",
    "local_global_blend25",
    "local_global_rephase02",
    "local_global_rephase05",
    "local_global_rephase10",
    "local_global_rephase15",
    "local_global_rephase25",
    "local_global_rephase50",
    "local_global_rephase75",
    "local_global_rephase100",
    "local_global_blend50",
    "dual_max_blend25",
    "local_global_postscore",
    "dual_max_postscore",
    "lowfreq32_postscore",
    "lowfreq32_int2_postscore",
    "lowfreq32_int4_postscore",
    "lowfreq32_adaptive_postscore",
    "lowfreq64_int2_postscore",
    "prerope_pca32_int4_postscore",
    "prerope_pca64_int2_postscore",
    "prerope_pca64_int4_postscore",
    "post2x_pre_rerank_postscore",
    "post4x_pre_rerank_postscore",
    "post2x_pre_boundary50_postscore",
    "post2x_pre_boundary75_postscore",
    "post2x_pre_rerank_masspreserve25",
    "pre_monotone25",
    "dual_monotone25",
    "dual_monotone50",
    "pre_masspreserve25",
    "dual_masspreserve25",
    "dual_masspreserve50",
)

REPHASE_ALPHAS = {
    "local_global_rephase02": 0.02,
    "local_global_rephase05": 0.05,
    "local_global_rephase10": 0.10,
    "local_global_rephase15": 0.15,
    "local_global_rephase25": 0.25,
    "local_global_rephase50": 0.50,
    "local_global_rephase75": 0.75,
    "local_global_rephase100": 1.00,
}

_ACTIVE_CONTROLLER: "Controller | None" = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe a local-RoPE/global-semantic attention design on Qwen3-8B. "
            "The local branch keeps standard RoPE; the remote branch retrieves "
            "with pre-RoPE QK and optionally calibrates its score distribution."
        )
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lengths", default="8192,16384,32768,65536")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=4)
    parser.add_argument("--ratio", type=float, default=0.02)
    parser.add_argument("--minimum-keep-tokens", type=int, default=0)
    parser.add_argument("--maximum-keep-tokens", type=int, default=0)
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="Comma-separated subset of attention variants.",
    )
    parser.add_argument("--local-window", type=int, default=128)
    parser.add_argument("--sink-tokens", type=int, default=16)
    parser.add_argument("--prefill-chunk-size", type=int, default=128)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--original-max-position-embeddings", type=int, default=40960)
    parser.add_argument("--global-max-position", type=int, default=70000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float16


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def clear_allocator() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def repeat_kv(hidden: torch.Tensor, groups: int) -> torch.Tensor:
    return hidden if groups == 1 else hidden.repeat_interleave(groups, dim=1)


def add_attention_mask(
    scores: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    if attention_mask is None:
        return scores
    return scores + attention_mask[
        :,
        :,
        -scores.shape[-2] :,
        : scores.shape[-1],
    ]


def force_current_topk(scores: torch.Tensor, keep_count: int) -> torch.Tensor:
    if scores.dim() != 2:
        raise ValueError(f"scores must be [heads, keys], got {tuple(scores.shape)}")
    key_count = int(scores.shape[-1])
    keep_count = min(key_count, max(1, int(keep_count)))
    current = key_count - 1
    if keep_count == 1:
        return torch.full(
            (scores.shape[0], 1),
            current,
            dtype=torch.long,
            device=scores.device,
        )
    history_count = keep_count - 1
    chosen = torch.topk(
        scores[:, :current].float(),
        k=history_count,
        dim=-1,
        largest=True,
    ).indices
    current_column = torch.full(
        (scores.shape[0], 1),
        current,
        dtype=torch.long,
        device=scores.device,
    )
    return torch.cat((chosen, current_column), dim=-1)


def local_global_selection(
    pre_scores: torch.Tensor,
    keep_count: int,
    local_window: int,
    sink_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return selected positions and a mask identifying semantic remote slots."""

    if pre_scores.dim() != 2:
        raise ValueError("pre_scores must be [heads, keys]")
    heads, key_count = pre_scores.shape
    current = key_count - 1
    history_budget = min(current, max(0, int(keep_count) - 1))
    local_count = min(int(local_window), history_budget)
    local_start = current - local_count
    remaining = history_budget - local_count
    sink_count = min(int(sink_tokens), remaining, max(0, local_start))
    remaining -= sink_count
    remote_start = sink_count
    remote_end = local_start
    remote_available = max(0, remote_end - remote_start)
    remote_count = min(remaining, remote_available)
    remaining -= remote_count

    pieces: list[torch.Tensor] = []
    remote_masks: list[torch.Tensor] = []
    if sink_count:
        sink = torch.arange(
            sink_count,
            device=pre_scores.device,
            dtype=torch.long,
        ).view(1, -1).expand(heads, -1)
        pieces.append(sink)
        remote_masks.append(torch.zeros_like(sink, dtype=torch.bool))
    if remote_count:
        remote = torch.topk(
            pre_scores[:, remote_start:remote_end].float(),
            k=remote_count,
            dim=-1,
            largest=True,
        ).indices + remote_start
        pieces.append(remote)
        remote_masks.append(torch.ones_like(remote, dtype=torch.bool))
    if remaining:
        # This only occurs when the remote region is shorter than the budget.
        extra_end = max(0, remote_start)
        extra_count = min(remaining, extra_end)
        if extra_count:
            extra = torch.arange(
                extra_end - extra_count,
                extra_end,
                device=pre_scores.device,
                dtype=torch.long,
            ).view(1, -1).expand(heads, -1)
            pieces.append(extra)
            remote_masks.append(torch.zeros_like(extra, dtype=torch.bool))
    if local_count:
        local = torch.arange(
            local_start,
            current,
            device=pre_scores.device,
            dtype=torch.long,
        ).view(1, -1).expand(heads, -1)
        pieces.append(local)
        remote_masks.append(torch.zeros_like(local, dtype=torch.bool))
    current_column = torch.full(
        (heads, 1),
        current,
        dtype=torch.long,
        device=pre_scores.device,
    )
    pieces.append(current_column)
    remote_masks.append(torch.zeros_like(current_column, dtype=torch.bool))
    selected = torch.cat(pieces, dim=-1)
    remote_mask = torch.cat(remote_masks, dim=-1)
    if selected.shape[-1] != min(key_count, keep_count):
        raise RuntimeError(
            f"selection budget mismatch: got={selected.shape[-1]} "
            f"expected={min(key_count, keep_count)}"
        )
    return selected, remote_mask


def post_overfetch_pre_rerank_selection(
    pre_scores: torch.Tensor,
    post_scores: torch.Tensor,
    keep_count: int,
    local_window: int,
    sink_tokens: int,
    overfetch_factor: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Overfetch with post-RoPE scores, then rerank the pool by pre-RoPE scores."""

    if pre_scores.shape != post_scores.shape or pre_scores.dim() != 2:
        raise ValueError("pre/post scores must share [heads, keys] shape")
    if overfetch_factor < 1:
        raise ValueError("overfetch factor must be positive")
    heads, key_count = pre_scores.shape
    current = key_count - 1
    history_budget = min(current, max(0, int(keep_count) - 1))
    local_count = min(int(local_window), history_budget)
    local_start = current - local_count
    remaining = history_budget - local_count
    sink_count = min(int(sink_tokens), remaining, max(0, local_start))
    remaining -= sink_count
    remote_start = sink_count
    remote_end = local_start
    remote_available = max(0, remote_end - remote_start)
    remote_count = min(remaining, remote_available)
    remaining -= remote_count

    pieces: list[torch.Tensor] = []
    remote_masks: list[torch.Tensor] = []
    if sink_count:
        sink = torch.arange(
            sink_count,
            device=pre_scores.device,
            dtype=torch.long,
        ).view(1, -1).expand(heads, -1)
        pieces.append(sink)
        remote_masks.append(torch.zeros_like(sink, dtype=torch.bool))
    if remote_count:
        pool_count = min(
            remote_available,
            max(remote_count, overfetch_factor * remote_count),
        )
        post_pool = torch.topk(
            post_scores[:, remote_start:remote_end].float(),
            k=pool_count,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices + remote_start
        pre_pool_scores = pre_scores.float().gather(1, post_pool)
        selected_in_pool = torch.topk(
            pre_pool_scores,
            k=remote_count,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices
        remote = post_pool.gather(1, selected_in_pool)
        pieces.append(remote)
        remote_masks.append(torch.ones_like(remote, dtype=torch.bool))
    if remaining:
        extra_end = max(0, remote_start)
        extra_count = min(remaining, extra_end)
        if extra_count:
            extra = torch.arange(
                extra_end - extra_count,
                extra_end,
                device=pre_scores.device,
                dtype=torch.long,
            ).view(1, -1).expand(heads, -1)
            pieces.append(extra)
            remote_masks.append(torch.zeros_like(extra, dtype=torch.bool))
    if local_count:
        local = torch.arange(
            local_start,
            current,
            device=pre_scores.device,
            dtype=torch.long,
        ).view(1, -1).expand(heads, -1)
        pieces.append(local)
        remote_masks.append(torch.zeros_like(local, dtype=torch.bool))
    current_column = torch.full(
        (heads, 1),
        current,
        dtype=torch.long,
        device=pre_scores.device,
    )
    pieces.append(current_column)
    remote_masks.append(torch.zeros_like(current_column, dtype=torch.bool))
    selected = torch.cat(pieces, dim=-1)
    remote_mask = torch.cat(remote_masks, dim=-1)
    if selected.shape[-1] != min(key_count, keep_count):
        raise RuntimeError(
            f"overfetch/rerank budget mismatch: got={selected.shape[-1]} "
            f"expected={min(key_count, keep_count)}"
        )
    return selected, remote_mask


def post_overfetch_pre_boundary_rerank_selection(
    pre_scores: torch.Tensor,
    post_scores: torch.Tensor,
    keep_count: int,
    local_window: int,
    sink_tokens: int,
    core_fraction: float,
    overfetch_factor: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the confident post-RoPE core and rerank only its rank boundary."""

    if pre_scores.shape != post_scores.shape or pre_scores.dim() != 2:
        raise ValueError("pre/post scores must share [heads, keys] shape")
    if not 0.0 <= core_fraction < 1.0:
        raise ValueError("core fraction must be in [0, 1)")
    if overfetch_factor < 1:
        raise ValueError("overfetch factor must be positive")
    heads, key_count = pre_scores.shape
    current = key_count - 1
    history_budget = min(current, max(0, int(keep_count) - 1))
    local_count = min(int(local_window), history_budget)
    local_start = current - local_count
    remaining = history_budget - local_count
    sink_count = min(int(sink_tokens), remaining, max(0, local_start))
    remaining -= sink_count
    remote_start = sink_count
    remote_end = local_start
    remote_available = max(0, remote_end - remote_start)
    remote_count = min(remaining, remote_available)
    remaining -= remote_count

    pieces: list[torch.Tensor] = []
    remote_masks: list[torch.Tensor] = []
    if sink_count:
        sink = torch.arange(
            sink_count,
            device=pre_scores.device,
            dtype=torch.long,
        ).view(1, -1).expand(heads, -1)
        pieces.append(sink)
        remote_masks.append(torch.zeros_like(sink, dtype=torch.bool))
    if remote_count:
        pool_count = min(
            remote_available,
            max(remote_count, overfetch_factor * remote_count),
        )
        ranked_pool = torch.topk(
            post_scores[:, remote_start:remote_end].float(),
            k=pool_count,
            dim=-1,
            largest=True,
            sorted=True,
        ).indices + remote_start
        core_count = min(
            remote_count,
            max(0, int(round(remote_count * float(core_fraction)))),
        )
        if core_count:
            core = ranked_pool[:, :core_count]
            pieces.append(core)
            remote_masks.append(torch.ones_like(core, dtype=torch.bool))
        fill_count = remote_count - core_count
        if fill_count:
            boundary_pool = ranked_pool[:, core_count:]
            boundary_scores = pre_scores.float().gather(1, boundary_pool)
            selected_boundary = torch.topk(
                boundary_scores,
                k=fill_count,
                dim=-1,
                largest=True,
                sorted=False,
            ).indices
            boundary = boundary_pool.gather(1, selected_boundary)
            pieces.append(boundary)
            remote_masks.append(torch.ones_like(boundary, dtype=torch.bool))
    if remaining:
        extra_end = max(0, remote_start)
        extra_count = min(remaining, extra_end)
        if extra_count:
            extra = torch.arange(
                extra_end - extra_count,
                extra_end,
                device=pre_scores.device,
                dtype=torch.long,
            ).view(1, -1).expand(heads, -1)
            pieces.append(extra)
            remote_masks.append(torch.zeros_like(extra, dtype=torch.bool))
    if local_count:
        local = torch.arange(
            local_start,
            current,
            device=pre_scores.device,
            dtype=torch.long,
        ).view(1, -1).expand(heads, -1)
        pieces.append(local)
        remote_masks.append(torch.zeros_like(local, dtype=torch.bool))
    current_column = torch.full(
        (heads, 1),
        current,
        dtype=torch.long,
        device=pre_scores.device,
    )
    pieces.append(current_column)
    remote_masks.append(torch.zeros_like(current_column, dtype=torch.bool))
    selected = torch.cat(pieces, dim=-1)
    remote_mask = torch.cat(remote_masks, dim=-1)
    if selected.shape[-1] != min(key_count, keep_count):
        raise RuntimeError(
            f"boundary rerank budget mismatch: got={selected.shape[-1]} "
            f"expected={min(key_count, keep_count)}"
        )
    return selected, remote_mask


def gather_scores(scores: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return scores[0, :, 0, :].gather(1, positions)


def local_global_virtual_positions(
    selected: torch.Tensor,
    remote_mask: torch.Tensor,
    query_position: int,
    local_window: int,
) -> torch.Tensor:
    """Pack only semantic-remote slots before the native recent window.

    Remote tokens keep their original order. Sink, recent, and current slots
    retain their native positions. Floating-point positions are intentional so
    callers can interpolate continuously between native and virtual positions.
    """

    if selected.shape != remote_mask.shape or selected.dim() != 2:
        raise ValueError("selected and remote_mask must share [heads, kept] shape")
    virtual = selected.to(dtype=torch.float32).clone()
    local_count = min(max(0, int(local_window)), max(0, int(query_position)))
    for head in range(selected.shape[0]):
        slots = torch.nonzero(remote_mask[head], as_tuple=False).flatten()
        remote_count = int(slots.numel())
        if remote_count == 0:
            continue
        ordered_slots = slots[
            torch.argsort(selected[head, slots], dim=-1, stable=True)
        ]
        start = max(0, int(query_position) - local_count - remote_count)
        virtual[head, ordered_slots] = torch.arange(
            start,
            start + remote_count,
            device=selected.device,
            dtype=torch.float32,
        )
    return virtual


def rephase_selected_scores(
    query_post: torch.Tensor,
    expanded_key_post: torch.Tensor,
    selected: torch.Tensor,
    remote_mask: torch.Tensor,
    query_position: int,
    local_window: int,
    alpha: float,
    inv_freq: torch.Tensor,
    scaling: float,
    attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return repaired selected scores, virtual positions, and effective positions."""

    gather_index = selected.view(1, selected.shape[0], -1, 1).expand(
        1,
        selected.shape[0],
        selected.shape[1],
        expanded_key_post.shape[-1],
    )
    selected_key = expanded_key_post.gather(2, gather_index)
    virtual = local_global_virtual_positions(
        selected,
        remote_mask,
        query_position,
        local_window,
    )
    original = selected.to(dtype=torch.float32)
    effective = original + float(alpha) * (virtual - original)
    repaired_key = rope_repair.apply_rope_delta(
        selected_key,
        original,
        effective,
        inv_freq,
    )
    repaired = torch.matmul(
        query_post,
        repaired_key.transpose(2, 3),
    ) * float(scaling)
    if attention_mask is not None:
        mask_scores = add_attention_mask(
            torch.zeros(
                (1, query_post.shape[1], 1, expanded_key_post.shape[-2]),
                dtype=repaired.dtype,
                device=repaired.device,
            ),
            attention_mask,
        )
        repaired = repaired + gather_scores(mask_scores, selected).unsqueeze(0).unsqueeze(2)
    return repaired[0, :, 0, :], virtual, effective


def low_frequency_pre_scores(
    query_pre: torch.Tensor,
    key_pre: torch.Tensor,
    *,
    quantization: str,
    frequency_count: int = 16,
    clip_alpha: float = 1.5,
) -> torch.Tensor:
    """Score the 32 slowest RoPE coordinates, matching the deployable rescue index."""

    if quantization not in {"none", "int2", "int4"}:
        raise ValueError(f"unknown low-frequency quantization: {quantization}")
    head_dim = int(query_pre.shape[-1])
    if head_dim % 2 or not 0 < frequency_count <= head_dim // 2:
        raise ValueError("invalid low-frequency RoPE slice")
    half = head_dim // 2
    query_low = torch.cat(
        (
            query_pre[..., half - frequency_count : half],
            query_pre[..., head_dim - frequency_count : head_dim],
        ),
        dim=-1,
    ).float()
    key_low = torch.cat(
        (
            key_pre[..., half - frequency_count : half],
            key_pre[..., head_dim - frequency_count : head_dim],
        ),
        dim=-1,
    ).float()
    normalized_query = torch.nn.functional.normalize(query_low, dim=-1)
    normalized_key = torch.nn.functional.normalize(key_low, dim=-1)
    if quantization == "none":
        return torch.matmul(
            normalized_query,
            normalized_key.transpose(2, 3),
        )

    query_scale = query_low.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8)
    query_codes = torch.round(query_low * (127.0 / query_scale)).clamp(
        -127.0,
        127.0,
    )
    if quantization == "int4":
        key_scale = normalized_key.abs().amax(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0e-8) / 7.0
        quantized_key = (
            torch.round(normalized_key / key_scale).clamp(-7.0, 7.0)
            * key_scale
        )
        return torch.matmul(query_codes, quantized_key.transpose(2, 3))

    normalized_key = (
        normalized_key * (math.sqrt(float(2 * frequency_count)) / clip_alpha)
    ).clamp(-1.0, 1.0)
    key_codes = (
        2.0 * torch.round((normalized_key + 1.0) * 1.5).clamp(0.0, 3.0)
        - 3.0
    )
    return torch.matmul(query_codes, key_codes.transpose(2, 3))


def pca_pre_scores(
    query_pre: torch.Tensor,
    key_pre: torch.Tensor,
    *,
    projection_dim: int,
    quantization: str,
    sample_count: int = 256,
) -> torch.Tensor:
    """Estimate pre-RoPE QK with a request-local per-KV-head PCA index."""

    if quantization not in {"int2", "int4"}:
        raise ValueError(f"unknown PCA quantization: {quantization}")
    batch_count, kv_head_count, key_count, head_dim = key_pre.shape
    query_head_count = int(query_pre.shape[1])
    if query_head_count % kv_head_count:
        raise ValueError("query heads must be divisible by KV heads")
    if not 0 < projection_dim <= head_dim:
        raise ValueError("invalid pre-RoPE PCA dimension")

    actual_samples = min(int(sample_count), key_count)
    sample_ids = (
        (2 * torch.arange(actual_samples, device=key_pre.device) + 1)
        * key_count
    ) // (2 * actual_samples)
    sampled_key = key_pre.index_select(-2, sample_ids).float()
    key_mean = sampled_key.mean(dim=-2, keepdim=True)
    centered_sample = sampled_key - key_mean
    covariance = torch.einsum(
        "bhnd,bhne->bhde",
        centered_sample,
        centered_sample,
    ) / float(actual_samples)
    _, eigenvectors = torch.linalg.eigh(covariance)
    basis = eigenvectors[..., -projection_dim:].contiguous()

    projected_key = torch.einsum(
        "bhnd,bhdm->bhnm",
        key_pre.float() - key_mean,
        basis,
    )
    groups = query_head_count // kv_head_count
    grouped_query = query_pre.reshape(
        batch_count,
        kv_head_count,
        groups,
        1,
        head_dim,
    )
    projected_query = torch.einsum(
        "bhgnd,bhdm->bhgnm",
        grouped_query.float(),
        basis,
    )
    query_scale = projected_query.abs().amax(
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0e-8) / 127.0
    query_codes = torch.round(projected_query / query_scale).clamp(
        -127.0,
        127.0,
    )

    levels = 7.0 if quantization == "int4" else 1.0
    key_scale = projected_key.abs().amax(
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0e-8)
    normalized_key = (projected_key / key_scale).clamp(-1.0, 1.0)
    if quantization == "int4":
        quantized_key = (
            torch.round(normalized_key * levels) / levels
        ) * key_scale
    else:
        quantized_key = (
            (
                2.0
                * torch.round((normalized_key + 1.0) * 1.5).clamp(
                    0.0,
                    3.0,
                )
                - 3.0
            )
            / 3.0
        ) * key_scale
    scores = torch.einsum(
        "bhgnd,bhkd->bhgnk",
        query_codes,
        quantized_key,
    )
    return scores.reshape(batch_count, query_head_count, 1, key_count)


def calibrated_pre_scores(
    pre_scores: torch.Tensor,
    post_scores: torch.Tensor,
    remote_end: int,
    sink_tokens: int,
) -> torch.Tensor:
    """Match pre-RoPE remote score mean/std to post-RoPE per attention head."""

    start = min(max(0, int(sink_tokens)), remote_end)
    if remote_end - start < 2:
        return pre_scores
    pre = pre_scores[..., start:remote_end].float()
    post = post_scores[..., start:remote_end].float()
    pre_mean = pre.mean(dim=-1, keepdim=True)
    post_mean = post.mean(dim=-1, keepdim=True)
    pre_std = pre.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    post_std = post.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (
        (pre_scores.float() - pre_mean) / pre_std * post_std + post_mean
    ).to(dtype=pre_scores.dtype)


@dataclass
class MetricAccumulator:
    head_rows: int = 0
    selected_gold: int = 0
    selectable_gold: int = 0
    line_hits: int = 0
    line_events: int = 0
    chain_hits: int = 0
    evidence_mass_sum: float = 0.0
    remote_slots_sum: int = 0
    selected_slots_sum: int = 0
    semantic_gap_positive_sum: float = 0.0
    semantic_gap_positive_count: int = 0
    semantic_gap_remote_count: int = 0
    semantic_gap_max_sum: float = 0.0
    semantic_gap_head_count: int = 0
    semantic_gap_active_head_count: int = 0
    rephase_remote_count: int = 0
    rephase_position_shift_sum: float = 0.0
    rephase_effective_distance_sum: float = 0.0
    rephase_score_delta_sum: float = 0.0
    rephase_gold_count: int = 0
    rephase_gold_score_delta_sum: float = 0.0
    rephase_nongold_count: int = 0
    rephase_nongold_score_delta_sum: float = 0.0

    def summary(self) -> dict[str, float]:
        return {
            "gold_evidence_token_recall": (
                self.selected_gold / max(1, self.selectable_gold)
            ),
            "gold_evidence_line_hit_rate": self.line_hits / max(1, self.line_events),
            "gold_chain_complete_rate": self.chain_hits / max(1, self.head_rows),
            "gold_evidence_attention_mass": (
                self.evidence_mass_sum / max(1, self.head_rows)
            ),
            "remote_selected_fraction": (
                self.remote_slots_sum / max(1, self.selected_slots_sum)
            ),
            "semantic_positive_gap_fraction": (
                self.semantic_gap_positive_count
                / max(1, self.semantic_gap_remote_count)
            ),
            "semantic_positive_gap_mean": (
                self.semantic_gap_positive_sum
                / max(1, self.semantic_gap_positive_count)
            ),
            "semantic_positive_gap_max_mean": (
                self.semantic_gap_max_sum
                / max(1, self.semantic_gap_head_count)
            ),
            "semantic_rescue_head_fraction": (
                self.semantic_gap_active_head_count
                / max(1, self.semantic_gap_head_count)
            ),
            "rephase_mean_abs_position_shift": (
                self.rephase_position_shift_sum
                / max(1, self.rephase_remote_count)
            ),
            "rephase_mean_effective_distance": (
                self.rephase_effective_distance_sum
                / max(1, self.rephase_remote_count)
            ),
            "rephase_remote_score_delta_mean": (
                self.rephase_score_delta_sum
                / max(1, self.rephase_remote_count)
            ),
            "rephase_gold_score_delta_mean": (
                self.rephase_gold_score_delta_sum
                / max(1, self.rephase_gold_count)
            ),
            "rephase_nongold_score_delta_mean": (
                self.rephase_nongold_score_delta_sum
                / max(1, self.rephase_nongold_count)
            ),
            "rephase_gold_minus_nongold_delta": (
                self.rephase_gold_score_delta_sum
                / max(1, self.rephase_gold_count)
                - self.rephase_nongold_score_delta_sum
                / max(1, self.rephase_nongold_count)
            ),
        }


@dataclass
class Controller:
    variant: str
    ratio: float
    minimum_keep_tokens: int
    maximum_keep_tokens: int
    local_window: int
    sink_tokens: int
    evidence_spans: tuple[tuple[int, int], ...]
    metrics: MetricAccumulator = field(default_factory=MetricAccumulator)

    def evidence_mask(self, key_count: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(key_count, dtype=torch.bool, device=device)
        for start, end in self.evidence_spans:
            mask[max(0, start) : min(key_count, end)] = True
        return mask

    def record(
        self,
        positions: torch.Tensor,
        weights: torch.Tensor,
        key_count: int,
        remote_mask: torch.Tensor | None,
    ) -> None:
        gold = self.evidence_mask(key_count, positions.device)
        selected_gold = gold[positions]
        heads = int(positions.shape[0])
        self.metrics.head_rows += heads
        self.metrics.selected_gold += int(selected_gold.sum().item())
        self.metrics.selectable_gold += int(gold.sum().item()) * heads
        for start, end in self.evidence_spans:
            hits = ((positions >= start) & (positions < end)).any(dim=-1)
            self.metrics.line_hits += int(hits.sum().item())
            self.metrics.line_events += heads
        if self.evidence_spans:
            line_matrix = torch.stack(
                [
                    ((positions >= start) & (positions < end)).any(dim=-1)
                    for start, end in self.evidence_spans
                ],
                dim=-1,
            )
            self.metrics.chain_hits += int(line_matrix.all(dim=-1).sum().item())
        mass = (
            weights[0, :, 0, :]
            .float()
            .masked_fill(~selected_gold, 0.0)
            .sum(dim=-1)
        )
        self.metrics.evidence_mass_sum += float(mass.sum().item())
        if remote_mask is not None:
            self.metrics.remote_slots_sum += int(remote_mask.sum().item())
            self.metrics.selected_slots_sum += int(remote_mask.numel())

    def record_semantic_gap(
        self,
        post_scores: torch.Tensor,
        calibrated_pre_scores: torch.Tensor,
        remote_mask: torch.Tensor,
    ) -> None:
        gap = calibrated_pre_scores.float() - post_scores.float()
        remote_gap = gap[remote_mask]
        positive = remote_gap[remote_gap > 0.0]
        self.metrics.semantic_gap_positive_sum += float(positive.sum().item())
        self.metrics.semantic_gap_positive_count += int(positive.numel())
        self.metrics.semantic_gap_remote_count += int(remote_gap.numel())
        masked_gap = gap.masked_fill(~remote_mask, -torch.inf)
        maxima = masked_gap.amax(dim=-1)
        finite = torch.isfinite(maxima)
        self.metrics.semantic_gap_max_sum += float(
            maxima[finite].clamp_min(0.0).sum().item()
        )
        self.metrics.semantic_gap_head_count += int(finite.sum().item())
        self.metrics.semantic_gap_active_head_count += int(
            (maxima[finite] > 0.0).sum().item()
        )

    def record_rephase(
        self,
        positions: torch.Tensor,
        remote_mask: torch.Tensor,
        post_scores: torch.Tensor,
        repaired_scores: torch.Tensor,
        effective_positions: torch.Tensor,
        query_position: int,
        key_count: int,
    ) -> None:
        if hasattr(self, "collect_metrics") and not bool(self.collect_metrics):
            return
        delta = repaired_scores.float() - post_scores.float()
        remote_delta = delta[remote_mask]
        original = positions.to(dtype=torch.float32)
        shift = (effective_positions.float() - original).abs()[remote_mask]
        distance = (
            float(query_position) - effective_positions.float()
        ).abs()[remote_mask]
        self.metrics.rephase_remote_count += int(remote_delta.numel())
        self.metrics.rephase_position_shift_sum += float(shift.sum().item())
        self.metrics.rephase_effective_distance_sum += float(distance.sum().item())
        self.metrics.rephase_score_delta_sum += float(remote_delta.sum().item())

        gold = self.evidence_mask(key_count, positions.device)[positions]
        gold_remote = gold & remote_mask
        nongold_remote = (~gold) & remote_mask
        gold_delta = delta[gold_remote]
        nongold_delta = delta[nongold_remote]
        self.metrics.rephase_gold_count += int(gold_delta.numel())
        self.metrics.rephase_gold_score_delta_sum += float(gold_delta.sum().item())
        self.metrics.rephase_nongold_count += int(nongold_delta.numel())
        self.metrics.rephase_nongold_score_delta_sum += float(
            nongold_delta.sum().item()
        )


def monotone_semantic_rescue(
    post_scores: torch.Tensor,
    calibrated_pre_scores: torch.Tensor,
    remote_mask: torch.Tensor,
    blend: float,
    preserve_remote_partition: bool,
) -> torch.Tensor:
    boosted = post_scores.float() + blend * torch.relu(
        calibrated_pre_scores.float() - post_scores.float()
    )
    if not preserve_remote_partition:
        return boosted.to(post_scores.dtype)

    negative_infinity = torch.full_like(boosted, -torch.inf)
    post_remote = torch.where(remote_mask, post_scores.float(), negative_infinity)
    boosted_remote = torch.where(remote_mask, boosted, negative_infinity)
    post_partition = torch.logsumexp(post_remote, dim=-1, keepdim=True)
    boosted_partition = torch.logsumexp(
        boosted_remote,
        dim=-1,
        keepdim=True,
    )
    finite = torch.isfinite(post_partition) & torch.isfinite(boosted_partition)
    correction = torch.where(
        finite,
        boosted_partition - post_partition,
        torch.zeros_like(post_partition),
    )
    corrected = torch.where(remote_mask, boosted - correction, boosted)
    return corrected.to(post_scores.dtype)


@contextmanager
def activate(controller: Controller | None):
    global _ACTIVE_CONTROLLER
    previous = _ACTIVE_CONTROLLER
    _ACTIVE_CONTROLLER = controller
    try:
        yield
    finally:
        _ACTIVE_CONTROLLER = previous


def local_global_attention_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    past_key_value: Any | None = None,
    cache_position: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    controller = _ACTIVE_CONTROLLER
    if controller is None or hidden_states.shape[-2] != 1:
        return self._local_global_original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )

    modeling_qwen3 = self._local_global_modeling_qwen3
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_pre = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    current_key_pre = self.k_norm(
        self.k_proj(hidden_states).view(hidden_shape)
    ).transpose(1, 2)
    current_value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_post, current_key_post = modeling_qwen3.apply_rotary_pos_emb(
        query_pre,
        current_key_pre,
        cos.to(query_pre.device),
        sin.to(query_pre.device),
    )
    if past_key_value is not None:
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "cache_position": cache_position,
        }
        key_post, value = past_key_value.update(
            current_key_post,
            current_value,
            self.layer_idx,
            cache_kwargs,
        )
    else:
        key_post, value = current_key_post, current_value

    groups = query_post.shape[1] // key_post.shape[1]
    expanded_key_post = repeat_kv(key_post, groups)
    expanded_value = repeat_kv(value, groups)
    key_count = int(expanded_key_post.shape[-2])
    scale = float(
        getattr(self, "scaling", 1.0 / math.sqrt(query_post.shape[-1]))
    )
    post_scores = torch.matmul(
        query_post,
        expanded_key_post.transpose(2, 3),
    ) * scale
    post_scores = add_attention_mask(post_scores, attention_mask)

    positions_all = torch.arange(key_count, device=key_post.device)
    attention_scale = rope_repair._attention_scaling((cos, sin))
    key_pre = rope_repair.invert_rope(
        key_post,
        positions_all,
        self._local_global_inv_freq,
        attention_scale,
    )
    expanded_key_pre = repeat_kv(key_pre, groups)
    pre_scores = torch.matmul(
        query_pre,
        expanded_key_pre.transpose(2, 3),
    ) * scale
    pre_scores = add_attention_mask(pre_scores, attention_mask)

    keep_count = max(1, int(math.ceil(controller.ratio * key_count)))
    if controller.minimum_keep_tokens > 0:
        keep_count = max(keep_count, controller.minimum_keep_tokens)
    if controller.maximum_keep_tokens > 0:
        keep_count = min(keep_count, controller.maximum_keep_tokens)
    keep_count = min(key_count, keep_count)
    remote_mask: torch.Tensor | None = None
    if controller.variant == "full_rope":
        selected = positions_all.view(1, -1).expand(query_post.shape[1], -1)
        sparse_scores = post_scores
        selected_value = expanded_value
    elif controller.variant in ("rope_top2", "semantic_top2_postscore"):
        retrieval = post_scores if controller.variant == "rope_top2" else pre_scores
        selected = force_current_topk(retrieval[0, :, 0, :], keep_count)
        sparse_scores = gather_scores(post_scores, selected).unsqueeze(0).unsqueeze(2)
        gather_index = selected.view(1, selected.shape[0], -1, 1).expand(
            1,
            selected.shape[0],
            selected.shape[1],
            expanded_value.shape[-1],
        )
        selected_value = expanded_value.gather(2, gather_index)
    else:
        local_count = min(controller.local_window, max(0, keep_count - 1))
        remote_end = max(0, key_count - 1 - local_count)
        calibrated = calibrated_pre_scores(
            pre_scores,
            post_scores,
            remote_end,
            controller.sink_tokens,
        )
        if controller.variant.startswith("dual_"):
            selection_scores = torch.maximum(
                calibrated[0, :, 0, :].float(),
                post_scores[0, :, 0, :].float(),
            )
        elif controller.variant in (
            "lowfreq32_postscore",
            "lowfreq32_int2_postscore",
            "lowfreq32_int4_postscore",
            "lowfreq32_adaptive_postscore",
            "lowfreq64_int2_postscore",
        ):
            quantization = "none"
            if controller.variant in (
                "lowfreq32_int2_postscore",
                "lowfreq64_int2_postscore",
            ):
                quantization = "int2"
            elif controller.variant == "lowfreq32_int4_postscore":
                quantization = "int4"
            elif controller.variant == "lowfreq32_adaptive_postscore":
                quantization = "int2" if key_count <= 32768 else "int4"
            selection_scores = low_frequency_pre_scores(
                query_pre,
                expanded_key_pre,
                quantization=quantization,
                frequency_count=(
                    32
                    if controller.variant == "lowfreq64_int2_postscore"
                    else 16
                ),
            )[0, :, 0, :]
        elif controller.variant in (
            "prerope_pca32_int4_postscore",
            "prerope_pca64_int2_postscore",
            "prerope_pca64_int4_postscore",
        ):
            projection_dim = (
                32
                if controller.variant == "prerope_pca32_int4_postscore"
                else 64
            )
            quantization = (
                "int2"
                if controller.variant == "prerope_pca64_int2_postscore"
                else "int4"
            )
            selection_scores = pca_pre_scores(
                query_pre,
                key_pre,
                projection_dim=projection_dim,
                quantization=quantization,
            )[0, :, 0, :]
        else:
            selection_scores = pre_scores[0, :, 0, :]
        if controller.variant in (
            "post2x_pre_rerank_postscore",
            "post4x_pre_rerank_postscore",
            "post2x_pre_rerank_masspreserve25",
        ):
            selected, remote_mask = post_overfetch_pre_rerank_selection(
                pre_scores[0, :, 0, :],
                post_scores[0, :, 0, :],
                keep_count,
                controller.local_window,
                controller.sink_tokens,
                (
                    4
                    if controller.variant
                    == "post4x_pre_rerank_postscore"
                    else 2
                ),
            )
        elif controller.variant in (
            "post2x_pre_boundary50_postscore",
            "post2x_pre_boundary75_postscore",
        ):
            selected, remote_mask = (
                post_overfetch_pre_boundary_rerank_selection(
                    pre_scores[0, :, 0, :],
                    post_scores[0, :, 0, :],
                    keep_count,
                    controller.local_window,
                    controller.sink_tokens,
                    (
                        0.75
                        if controller.variant
                        == "post2x_pre_boundary75_postscore"
                        else 0.50
                    ),
                )
            )
        else:
            selected, remote_mask = local_global_selection(
                selection_scores,
                keep_count,
                controller.local_window,
                controller.sink_tokens,
            )
        post_selected = gather_scores(post_scores, selected)
        pre_selected = gather_scores(pre_scores, selected)
        calibrated_selected = gather_scores(calibrated, selected)
        controller.record_semantic_gap(
            post_selected,
            calibrated_selected,
            remote_mask,
        )
        if controller.variant == "local_global_raw":
            remote_selected = pre_selected
        else:
            if controller.variant == "local_global_calibrated":
                remote_selected = calibrated_selected
            elif controller.variant in (
                "local_global_postscore",
                "dual_max_postscore",
                "lowfreq32_postscore",
                "lowfreq32_int2_postscore",
                "lowfreq32_int4_postscore",
                "lowfreq32_adaptive_postscore",
                "lowfreq64_int2_postscore",
                "prerope_pca32_int4_postscore",
                "prerope_pca64_int2_postscore",
                "prerope_pca64_int4_postscore",
                "post2x_pre_rerank_postscore",
                "post4x_pre_rerank_postscore",
                "post2x_pre_boundary50_postscore",
                "post2x_pre_boundary75_postscore",
            ):
                remote_selected = post_selected
            elif controller.variant in REPHASE_ALPHAS:
                remote_selected, virtual_positions, effective_positions = (
                    rephase_selected_scores(
                        query_post,
                        expanded_key_post,
                        selected,
                        remote_mask,
                        key_count - 1,
                        controller.local_window,
                        REPHASE_ALPHAS[controller.variant],
                        self._local_global_inv_freq,
                        scale,
                        attention_mask,
                    )
                )
                controller.record_rephase(
                    selected,
                    remote_mask,
                    post_selected,
                    remote_selected,
                    effective_positions,
                    key_count - 1,
                    key_count,
                )
            elif controller.variant in (
                "local_global_blend25",
                "dual_max_blend25",
            ):
                remote_selected = (
                    0.75 * post_selected.float()
                    + 0.25 * calibrated_selected.float()
                ).to(post_selected.dtype)
            elif controller.variant == "local_global_blend50":
                remote_selected = 0.5 * (
                    calibrated_selected.float() + post_selected.float()
                )
                remote_selected = remote_selected.to(post_selected.dtype)
            elif controller.variant in (
                "pre_monotone25",
                "dual_monotone25",
            ):
                remote_selected = monotone_semantic_rescue(
                    post_selected,
                    calibrated_selected,
                    remote_mask,
                    blend=0.25,
                    preserve_remote_partition=False,
                )
            elif controller.variant == "dual_monotone50":
                remote_selected = monotone_semantic_rescue(
                    post_selected,
                    calibrated_selected,
                    remote_mask,
                    blend=0.50,
                    preserve_remote_partition=False,
                )
            elif controller.variant in (
                "pre_masspreserve25",
                "dual_masspreserve25",
                "post2x_pre_rerank_masspreserve25",
            ):
                remote_selected = monotone_semantic_rescue(
                    post_selected,
                    calibrated_selected,
                    remote_mask,
                    blend=0.25,
                    preserve_remote_partition=True,
                )
            elif controller.variant == "dual_masspreserve50":
                remote_selected = monotone_semantic_rescue(
                    post_selected,
                    calibrated_selected,
                    remote_mask,
                    blend=0.50,
                    preserve_remote_partition=True,
                )
            else:
                raise ValueError(f"unknown variant: {controller.variant}")
        merged = torch.where(remote_mask, remote_selected, post_selected)
        sparse_scores = merged.unsqueeze(0).unsqueeze(2)
        gather_index = selected.view(1, selected.shape[0], -1, 1).expand(
            1,
            selected.shape[0],
            selected.shape[1],
            expanded_value.shape[-1],
        )
        selected_value = expanded_value.gather(2, gather_index)

    weights = F.softmax(sparse_scores.float(), dim=-1).to(query_post.dtype)
    controller.record(selected, weights, key_count, remote_mask)
    attention_output = torch.matmul(weights, selected_value)
    attention_output = attention_output.transpose(1, 2).contiguous()
    attention_output = attention_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attention_output), weights


def patch_model(model: Any) -> None:
    import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3

    inv_freq = model.model.rotary_emb.inv_freq.detach().float()
    found = 0
    for module in model.modules():
        if module.__class__.__name__ != "Qwen3Attention":
            continue
        module._local_global_original_forward = module.forward
        module._local_global_modeling_qwen3 = modeling_qwen3
        module._local_global_inv_freq = inv_freq.to(next(module.parameters()).device)
        module.forward = types.MethodType(local_global_attention_forward, module)
        found += 1
    if found == 0:
        raise RuntimeError("no Qwen3Attention modules found")


def load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    factor = base.rope_factor_for_length(
        args.global_max_position,
        args.original_max_position_embeddings,
    )
    if factor > 1.0:
        config.max_position_embeddings = args.global_max_position
        config.rope_scaling = {
            "type": "yarn",
            "factor": float(factor),
            "original_max_position_embeddings": (
                args.original_max_position_embeddings
            ),
        }
    dtype = resolve_dtype(args.dtype)
    kwargs: dict[str, Any] = {
        "config": config,
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "device_map": getattr(args, "device_map", "auto"),
        "attn_implementation": args.attn_implementation,
    }
    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **kwargs,
    )
    model.eval()
    model.config.use_cache = True
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def seeded_case(
    tokenizer: Any,
    length: int,
    seed: int,
) -> dict[str, Any]:
    words = causal.build_english_single_token_code_pool(tokenizer, required=64)
    rng = random.Random(2026073001 + seed * 1009)
    rng.shuffle(words)
    codes = words[:3]
    events = [
        causal.make_event(
            "relevant",
            f"T{step}",
            "VERIFIED RULE",
            codes[step],
            codes[step + 1],
            step,
        )
        for step in range(2)
    ]
    encoded, _ = causal.encode_event_block(tokenizer, events, 0)
    body = base.build_filler_ids(tokenizer, length, 2_300_000 + seed)
    preferred = min(256, length - len(encoded))
    placed: list[base.RuleEvent] = []
    occupied: list[tuple[int, int]] = []
    causal.insert_event_block(
        body,
        tokenizer,
        events,
        preferred,
        occupied,
        placed,
    )
    placed.sort(key=lambda event: event.start_token)
    suffix = causal.build_suffix("chat_concise", codes[0], 2, "full2")
    wrapper_prefix, wrapper_suffix = causal.chat_wrapper_ids(tokenizer)
    prompt_ids = (
        wrapper_prefix
        + body
        + base.token_ids(tokenizer, suffix)
        + wrapper_suffix
    )
    offset = len(wrapper_prefix)
    return {
        "prompt": torch.tensor(prompt_ids, dtype=torch.long).view(1, -1),
        "codes": codes,
        "evidence_spans": tuple(
            (
                offset + int(event.start_token),
                offset + int(event.end_token),
            )
            for event in placed
        ),
    }


def answer_metrics(
    tokenizer: Any,
    logits: torch.Tensor,
    gold: str,
) -> dict[str, Any]:
    ids = tokenizer(f" {gold}", add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise RuntimeError(f"gold answer is not one token: {gold!r} -> {ids}")
    gold_id = int(ids[0])
    log_probs = F.log_softmax(logits[:, -1, :].float(), dim=-1)
    nll = -float(log_probs[0, gold_id].item())
    prediction_id = int(logits[:, -1, :].argmax(dim=-1).item())
    return {
        "gold_nll": nll,
        "gold_ppl": math.exp(nll),
        "gold_probability": math.exp(-nll),
        "prediction_token_id": prediction_id,
        "prediction_text": tokenizer.decode(
            [prediction_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ).replace("\n", "\\n"),
        "next_token_correct": int(prediction_id == gold_id),
    }


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = sorted(
        {(int(row["target_context_tokens"]), str(row["variant"])) for row in rows}
    )
    for length, variant in keys:
        selected = [
            row
            for row in rows
            if int(row["target_context_tokens"]) == length
            and str(row["variant"]) == variant
        ]
        mean_nll = statistics.fmean(float(row["gold_nll"]) for row in selected)
        output.append(
            {
                "target_context_tokens": length,
                "variant": variant,
                "sample_count": len(selected),
                "gold_evidence_token_recall": statistics.fmean(
                    float(row["gold_evidence_token_recall"]) for row in selected
                ),
                "gold_evidence_line_hit_rate": statistics.fmean(
                    float(row["gold_evidence_line_hit_rate"]) for row in selected
                ),
                "gold_chain_complete_rate": statistics.fmean(
                    float(row["gold_chain_complete_rate"]) for row in selected
                ),
                "gold_evidence_attention_mass": statistics.fmean(
                    float(row["gold_evidence_attention_mass"]) for row in selected
                ),
                "mean_gold_nll": mean_nll,
                "gold_ppl": math.exp(mean_nll),
                "next_token_accuracy": statistics.fmean(
                    int(row["next_token_correct"]) for row in selected
                ),
                "mean_query_seconds": statistics.fmean(
                    float(row["query_seconds"]) for row in selected
                ),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    if not 0.0 < args.ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    if args.minimum_keep_tokens < 0 or args.maximum_keep_tokens < 0:
        raise ValueError("keep-token bounds must be non-negative")
    if (
        args.minimum_keep_tokens > 0
        and args.maximum_keep_tokens > 0
        and args.minimum_keep_tokens > args.maximum_keep_tokens
    ):
        raise ValueError("minimum keep tokens cannot exceed maximum")
    lengths = sorted(
        {int(item.strip()) for item in args.lengths.split(",") if item.strip()}
    )
    if not lengths:
        raise ValueError("no lengths supplied")
    variants = [
        item.strip() for item in args.variants.split(",") if item.strip()
    ]
    unknown_variants = sorted(set(variants) - set(VARIANTS))
    if not variants or unknown_variants:
        raise ValueError(f"unknown variants: {unknown_variants}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "resolved_lengths": lengths,
        "variants": variants,
        "cuda_visible_devices": __import__("os").environ.get(
            "CUDA_VISIBLE_DEVICES",
            "",
        ),
    }
    write_json(output_dir / "config.json", config)
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return

    model, tokenizer = load_model(args)
    patch_model(model)
    rows_path = output_dir / "rows.jsonl"
    completed: set[tuple[int, int]] = set()
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                completed.add(
                    (int(item["target_context_tokens"]), int(item["seed"]))
                )

    for length in lengths:
        for seed in range(args.seed_start, args.seed_start + args.num_seeds):
            if (length, seed) in completed:
                print(f"length={length} seed={seed} already complete", flush=True)
                continue
            case = seeded_case(tokenizer, length, seed)
            prompt = case["prompt"]
            prefix_length = int(prompt.shape[1]) - 1
            base.synchronize()
            started = time.perf_counter()
            legacy, prefill_seconds = base.prefill_sequence(
                model,
                prompt[:, :-1],
                args.prefill_chunk_size,
            )
            cache = base.cache_from_legacy(legacy)
            del legacy
            seed_rows: list[dict[str, Any]] = []
            for variant in variants:
                controller = Controller(
                    variant=variant,
                    ratio=args.ratio,
                    minimum_keep_tokens=args.minimum_keep_tokens,
                    maximum_keep_tokens=args.maximum_keep_tokens,
                    local_window=args.local_window,
                    sink_tokens=args.sink_tokens,
                    evidence_spans=case["evidence_spans"],
                )
                base.synchronize()
                query_started = time.perf_counter()
                with activate(controller), torch.inference_mode():
                    output = base.forward_with_cache(
                        model,
                        prompt[:, -1:].to(base.input_device(model)),
                        cache,
                        prefix_length,
                    )
                base.synchronize()
                query_seconds = time.perf_counter() - query_started
                metrics = answer_metrics(tokenizer, output.logits, case["codes"][-1])
                seed_rows.append(
                    {
                        "target_context_tokens": length,
                        "prompt_tokens": int(prompt.shape[1]),
                        "seed": seed,
                        "variant": variant,
                        "gold_chain": " -> ".join(case["codes"]),
                        **controller.metrics.summary(),
                        **metrics,
                        "prefill_seconds": prefill_seconds,
                        "query_seconds": query_seconds,
                    }
                )
                del output
                rope_repair.reset_dynamic_cache(cache, prefix_length)
            with rows_path.open("a", encoding="utf-8") as handle:
                for row in seed_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            all_rows = [
                json.loads(line)
                for line in rows_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            write_csv(output_dir / "rows.csv", all_rows)
            write_csv(output_dir / "summary.csv", summarize(all_rows))
            print(
                f"length={length} seed={seed} prompt={prompt.shape[1]} "
                f"prefill={prefill_seconds:.2f}s total={time.perf_counter() - started:.2f}s",
                flush=True,
            )
            for row in seed_rows:
                print(
                    f"  {row['variant']}: recall={row['gold_evidence_token_recall']:.4f} "
                    f"mass={row['gold_evidence_attention_mass']:.6f} "
                    f"ppl={row['gold_ppl']:.4f} correct={row['next_token_correct']}",
                    flush=True,
                )
            del cache, prompt
            clear_allocator()

    all_rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = summarize(all_rows)
    write_csv(output_dir / "rows.csv", all_rows)
    write_csv(output_dir / "summary.csv", summary)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "done.txt").write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
