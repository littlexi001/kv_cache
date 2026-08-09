from __future__ import annotations

import math
import json
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType, SimpleNamespace
from typing import Any, Iterator

import torch
from transformers.cache_utils import Cache


@dataclass
class DirectoryUpdate:
    final_slots: torch.Tensor
    misses: list[tuple[int, int, torch.Tensor, torch.Tensor]]
    hit_rate: float


def pack_projected_int4(
    projected: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if projected.shape[-1] % 2 != 0:
        raise ValueError("INT4 projection dimension must be even")
    scales = projected.float().abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / 7.0
    codes = torch.round(projected.float() / scales).clamp(-7, 7).to(torch.int16) + 7
    packed = codes[..., 0::2].to(torch.uint8) | (
        codes[..., 1::2].to(torch.uint8) << 4
    )
    return packed, scales.to(projected.dtype)


def exact_cache_capacity(
    sequence_length: int,
    max_new_tokens: int,
    exact_cache_fraction: float,
    candidate_fraction: float,
    stream_group_size: int,
    candidate_min_tokens: int = 1,
    candidate_max_tokens: int | None = None,
) -> int:
    """Allocate enough exact slots for the largest allowed decode candidate set."""
    capacity = int(sequence_length) + int(max_new_tokens)
    target_count = max(1, math.floor(exact_cache_fraction * int(sequence_length)))
    largest_candidate_count = bounded_fraction_count(
        capacity - 1,
        candidate_fraction,
        minimum_tokens=candidate_min_tokens,
        maximum_tokens=candidate_max_tokens,
    )
    required_count = largest_candidate_count * int(stream_group_size)
    return max(target_count, required_count)


def bounded_fraction_count(
    sequence_length: int,
    fraction: float,
    *,
    minimum_tokens: int = 1,
    maximum_tokens: int | None = None,
) -> int:
    """Convert a fractional budget to a bounded integer token count."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if minimum_tokens <= 0:
        raise ValueError("minimum_tokens must be positive")
    if maximum_tokens is not None:
        if maximum_tokens <= 0:
            raise ValueError("maximum_tokens must be positive when provided")
        if maximum_tokens < minimum_tokens:
            raise ValueError("maximum_tokens cannot be smaller than minimum_tokens")
    count = max(minimum_tokens, math.ceil(fraction * sequence_length))
    if maximum_tokens is not None:
        count = min(count, maximum_tokens)
    return min(sequence_length, count)


def update_sorted_directory(
    candidates: torch.Tensor,
    resident_ids: torch.Tensor,
    resident_slots: torch.Tensor,
    resident_ages: torch.Tensor,
) -> DirectoryUpdate:
    """Exact candidate lookup with an 8-bit recency cache policy.

    This is the clarity-first integration implementation. The production path
    replaces it with the fused hash/histogram CUDA kernel benchmarked separately.
    """
    if candidates.dim() != 3:
        raise ValueError("candidates must have shape [batch, kv_heads, selected]")
    if resident_ids.shape != resident_slots.shape or resident_ids.shape != resident_ages.shape:
        raise ValueError("resident directory tensors must have identical shapes")
    if candidates.shape[:2] != resident_ids.shape[:2]:
        raise ValueError("candidate and resident batch/head dimensions must match")
    if candidates.shape[-1] > resident_ids.shape[-1]:
        raise ValueError("selected candidate count cannot exceed exact cache capacity")
    if candidates.dtype != torch.int32 or resident_ids.dtype != torch.int32:
        raise ValueError("candidate and resident token ids must be int32")
    if resident_slots.dtype != torch.int32 or resident_ages.dtype != torch.uint8:
        raise ValueError("resident slots must be int32 and ages must be uint8")

    resident_ages.copy_(
        torch.where(resident_ages > 0, resident_ages - 1, resident_ages)
    )
    positions = torch.searchsorted(resident_ids, candidates)
    safe_positions = positions.clamp_max(resident_ids.shape[-1] - 1)
    hit = torch.gather(resident_ids, -1, safe_positions).eq(candidates)
    final_slots = torch.full_like(candidates, -1, dtype=torch.int32)
    misses: list[tuple[int, int, torch.Tensor, torch.Tensor]] = []

    for batch in range(candidates.shape[0]):
        for head in range(candidates.shape[1]):
            head_hit = hit[batch, head]
            hit_positions = safe_positions[batch, head, head_hit]
            if hit_positions.numel():
                final_slots[batch, head, head_hit] = resident_slots[
                    batch, head, hit_positions
                ]
                resident_ages[batch, head, hit_positions] = 255

            miss_positions = torch.nonzero(~head_hit, as_tuple=False).flatten()
            miss_count = int(miss_positions.numel())
            if miss_count == 0:
                continue
            eligible_ages = resident_ages[batch, head].to(torch.int16)
            if hit_positions.numel():
                eligible_ages[hit_positions] = 256
            eviction_positions = torch.topk(
                eligible_ages,
                k=miss_count,
                largest=False,
                sorted=False,
            ).indices
            destination_slots = resident_slots[batch, head, eviction_positions]
            miss_tokens = candidates[batch, head, miss_positions]
            final_slots[batch, head, miss_positions] = destination_slots
            resident_ids[batch, head, eviction_positions] = miss_tokens
            resident_ages[batch, head, eviction_positions] = 255

            sorted_ids, order = torch.sort(resident_ids[batch, head])
            resident_ids[batch, head] = sorted_ids
            resident_slots[batch, head] = resident_slots[batch, head, order]
            resident_ages[batch, head] = resident_ages[batch, head, order]
            misses.append((batch, head, miss_tokens, destination_slots))

    return DirectoryUpdate(
        final_slots=final_slots,
        misses=misses,
        hit_rate=float(hit.float().mean().item()),
    )


def head_balanced_topk_offsets(
    group_scores: torch.Tensor,
    keep_count: int,
) -> torch.Tensor:
    """Reserve a per-query-head quota, then fill unused slots by shared score."""
    if group_scores.dim() != 4:
        raise ValueError("group_scores must have shape [batch, kv_heads, groups, items]")
    item_count = int(group_scores.shape[-1])
    groups = int(group_scores.shape[-2])
    if groups <= 0 or not 0 < keep_count <= item_count:
        raise ValueError("keep_count must be within the item count")

    quota = keep_count // groups
    shared_scores = group_scores.sum(dim=2)
    if quota == 0:
        return torch.topk(
            shared_scores, k=keep_count, dim=-1, sorted=False
        ).indices

    per_head = torch.topk(
        group_scores, k=quota, dim=-1, sorted=False
    ).indices
    protected = torch.zeros_like(shared_scores, dtype=torch.bool)
    protected.scatter_(2, per_head.flatten(start_dim=2), True)
    priority = torch.where(
        protected,
        torch.full_like(shared_scores, torch.inf),
        shared_scores,
    )
    return torch.topk(priority, k=keep_count, dim=-1, sorted=False).indices


def exact_shared_rerank_slots(
    grouped_query: torch.Tensor,
    key_cache: torch.Tensor,
    candidate_slots: torch.Tensor,
    keep_count: int,
    selection_mode: str = "shared_sum",
) -> torch.Tensor:
    """Rerank a coarse KV-head shortlist with exact grouped-query QK scores."""
    if grouped_query.dim() != 4:
        raise ValueError("grouped_query must have shape [batch, kv_heads, groups, dim]")
    if key_cache.dim() != 4:
        raise ValueError("key_cache must have shape [batch, kv_heads, slots, dim]")
    if candidate_slots.dim() != 3:
        raise ValueError("candidate_slots must have shape [batch, kv_heads, selected]")
    if grouped_query.shape[:2] != key_cache.shape[:2]:
        raise ValueError("query and key cache batch/head dimensions must match")
    if candidate_slots.shape[:2] != key_cache.shape[:2]:
        raise ValueError("candidate and key cache batch/head dimensions must match")
    if grouped_query.shape[-1] != key_cache.shape[-1]:
        raise ValueError("query and key dimensions must match")
    if not 0 < keep_count <= candidate_slots.shape[-1]:
        raise ValueError("keep_count must be within the candidate count")

    slots = candidate_slots.to(torch.long)
    candidate_keys = torch.gather(
        key_cache,
        2,
        slots.unsqueeze(-1).expand(-1, -1, -1, key_cache.shape[-1]),
    )
    exact_scores = torch.einsum("bhgd,bhkd->bhgk", grouped_query, candidate_keys)
    if selection_mode == "head_balanced":
        selected_offsets = head_balanced_topk_offsets(exact_scores, keep_count)
    elif selection_mode == "shared_sum":
        selected_offsets = torch.topk(
            exact_scores.sum(dim=2), k=keep_count, dim=-1, sorted=False
        ).indices
    elif selection_mode == "shared_max":
        selected_offsets = torch.topk(
            exact_scores.max(dim=2).values, k=keep_count, dim=-1, sorted=False
        ).indices
    else:
        raise ValueError(
            "selection_mode must be 'shared_sum', 'shared_max', or 'head_balanced'"
        )
    return torch.gather(slots, -1, selected_offsets)


def exact_per_head_rerank_slots(
    grouped_query: torch.Tensor,
    key_cache: torch.Tensor,
    candidate_slots: torch.Tensor,
    keep_count: int,
) -> torch.Tensor:
    """Rerank each query head's virtual candidates without sharing its slots."""
    if grouped_query.dim() != 4 or key_cache.dim() != 4:
        raise ValueError("grouped_query and key_cache must both be rank four")
    if candidate_slots.dim() != 4:
        raise ValueError(
            "candidate_slots must have shape [batch, kv_heads, groups, selected]"
        )
    if grouped_query.shape[:3] != candidate_slots.shape[:3]:
        raise ValueError("query and candidate batch/head/group dimensions must match")
    if grouped_query.shape[:2] != key_cache.shape[:2]:
        raise ValueError("query and key cache batch/head dimensions must match")
    if grouped_query.shape[-1] != key_cache.shape[-1]:
        raise ValueError("query and key dimensions must match")
    if not 0 < keep_count <= candidate_slots.shape[-1]:
        raise ValueError("keep_count must be within the candidate count")

    slots = candidate_slots.to(torch.long)
    expanded_keys = key_cache.unsqueeze(2).expand(
        -1, -1, grouped_query.shape[2], -1, -1
    )
    candidate_keys = torch.gather(
        expanded_keys,
        3,
        slots.unsqueeze(-1).expand(-1, -1, -1, -1, key_cache.shape[-1]),
    )
    exact_scores = torch.einsum(
        "bhgd,bhgkd->bhgk", grouped_query, candidate_keys
    )
    selected_offsets = torch.topk(
        exact_scores, k=keep_count, dim=-1, sorted=False
    ).indices
    return torch.gather(slots, -1, selected_offsets)


def merge_streamed_gqa_outputs(
    group_outputs: list[torch.Tensor],
    query_heads: int,
    kv_heads: int,
) -> torch.Tensor:
    if not group_outputs:
        raise ValueError("group_outputs cannot be empty")
    reference_shape = group_outputs[0].shape
    if len(reference_shape) != 4 or reference_shape[1] != 1:
        raise ValueError(
            "streamed outputs must have shape [batch, query_length=1, kv_heads, dim]"
        )
    if any(output.shape != reference_shape for output in group_outputs):
        raise ValueError("all streamed outputs must have identical shapes")
    batch, _, chunk_query_heads, head_dim = reference_shape
    if chunk_query_heads % kv_heads != 0:
        raise ValueError("each streamed chunk must contain whole GQA groups")
    groups_per_chunk = chunk_query_heads // kv_heads
    if kv_heads * groups_per_chunk * len(group_outputs) != query_heads:
        raise ValueError("streamed KV-head/group count does not match query_heads")
    grouped_chunks = [
        output.reshape(batch, 1, kv_heads, groups_per_chunk, head_dim)
        for output in group_outputs
    ]
    return torch.cat(grouped_chunks, dim=3).reshape(
        batch, 1, query_heads, head_dim
    )


@dataclass
class HierarchicalLayerState:
    host_kv: torch.Tensor
    basis: torch.Tensor
    quantized: torch.Tensor
    scales: torch.Tensor
    padded_query: torch.Tensor
    device_cache: torch.Tensor
    resident_ids: torch.Tensor
    resident_slots: torch.Tensor
    resident_ages: torch.Tensor
    table_keys: torch.Tensor
    table_slots: torch.Tensor
    length: int
    initial_length: int
    capacity: int
    exact_cache_count: int
    last_shared_candidates: torch.Tensor | None = None
    last_attention_output: torch.Tensor | None = None
    cached_retrieved_candidates: torch.Tensor | None = None
    candidate_cache_age: int = 0
    last_retrieval_probe: torch.Tensor | None = None
    last_retrieval_score_spread: torch.Tensor | None = None
    last_retrieval_candidate_stability: torch.Tensor | None = None
    last_retrieval_refreshed: torch.Tensor | None = None
    query_basis: torch.Tensor | None = None
    packed_index: dict[str, Any] | None = None
    packed_scale_metrics: torch.Tensor | None = None
    packed_indexed_count: int = 0


class HierarchicalPCACache(Cache):
    """Pinned-host exact KV plus GPU PCA index and token-granular hot cache."""

    is_compileable = False

    def __init__(
        self,
        states: list[HierarchicalLayerState],
        *,
        projection_dim: int,
        index_bits: int,
        candidate_fraction: float,
        attention_fraction: float,
        candidate_selection_mode: str,
        rerank_selection_mode: str,
        original_gpu_bytes: int,
        directory_backend: str,
        record_traces: bool,
        recent_fraction: float,
        debug_directory: bool,
        stream_group_size: int,
        candidate_refresh_interval: int = 1,
        async_host_append: bool = True,
        collect_retrieval_features: bool = False,
        index_mode: str = "pca_fixed",
        qk_metric_query_shrinkage: float = 0.75,
        variable_rate_budget: int = 15,
        fixed_bit_allocation: tuple[int, ...] | None = None,
        candidate_min_tokens: int = 1,
        candidate_max_tokens: int | None = None,
        retrieval_backend: str = "full_topk",
        sampled_candidate_multiplier: float = 1.5,
    ) -> None:
        super().__init__()
        self.states = states
        self.projection_dim = int(projection_dim)
        self.index_bits = int(index_bits)
        self.candidate_fraction = float(candidate_fraction)
        self.attention_fraction = float(attention_fraction)
        self.max_candidate_fraction = float(candidate_fraction)
        self.candidate_selection_mode = candidate_selection_mode
        self.rerank_selection_mode = rerank_selection_mode
        self.original_gpu_bytes = int(original_gpu_bytes)
        self.directory_backend = directory_backend
        self.record_traces = bool(record_traces)
        self.recent_fraction = float(recent_fraction)
        self.debug_directory = bool(debug_directory)
        self.stream_group_size = int(stream_group_size)
        self.candidate_refresh_interval = int(candidate_refresh_interval)
        self.async_host_append = bool(async_host_append)
        self.collect_retrieval_features = bool(collect_retrieval_features)
        self.index_mode = str(index_mode)
        self.qk_metric_query_shrinkage = float(qk_metric_query_shrinkage)
        self.variable_rate_budget = int(variable_rate_budget)
        self.fixed_bit_allocation = fixed_bit_allocation
        self.candidate_min_tokens = int(candidate_min_tokens)
        self.candidate_max_tokens = (
            None
            if candidate_max_tokens is None
            else int(candidate_max_tokens)
        )
        self.retrieval_backend = str(retrieval_backend)
        self.sampled_candidate_multiplier = float(sampled_candidate_multiplier)
        self.sampled_workspaces: dict[
            tuple[str, int | None, int, int], dict[str, torch.Tensor]
        ] = {}
        self.sampled_candidate_count_tensors: list[torch.Tensor] = []
        self.sampled_overflow_tensors: list[torch.Tensor] = []
        self.sampled_clipped_tensors: list[torch.Tensor] = []
        if self.candidate_refresh_interval <= 0:
            raise ValueError("candidate_refresh_interval must be positive")
        self.hit_rates: list[float] = []
        self.hit_rate_tensors: list[torch.Tensor] = []
        self.candidate_union_mean_tensors: list[torch.Tensor] = []
        self.candidate_union_max_tensors: list[torch.Tensor] = []

    def retrieval_diagnostic_checkpoint(
        self,
    ) -> list[
        tuple[
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
        ]
    ]:
        """Snapshot causal retrieval diagnostics for counterfactual probes."""
        return [
            (
                state.last_retrieval_probe,
                state.last_retrieval_score_spread,
                state.last_retrieval_candidate_stability,
                state.last_retrieval_refreshed,
            )
            for state in self.states
        ]

    def restore_retrieval_diagnostic(
        self,
        checkpoint: list[
            tuple[
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor | None,
            ]
        ],
    ) -> None:
        if len(checkpoint) != len(self.states):
            raise ValueError("retrieval diagnostic checkpoint has the wrong layer count")
        for state, values in zip(self.states, checkpoint):
            (
                state.last_retrieval_probe,
                state.last_retrieval_score_spread,
                state.last_retrieval_candidate_stability,
                state.last_retrieval_refreshed,
            ) = values

    def retrieval_features(self) -> dict[str, float]:
        """Aggregate diagnostics produced by the preceding causal forward."""
        valid_states = [
            state
            for state in self.states
            if state.last_retrieval_score_spread is not None
            and state.last_retrieval_candidate_stability is not None
            and state.last_retrieval_refreshed is not None
        ]
        if not valid_states:
            return {
                "retrieval_feature_valid": 0.0,
                "retrieval_score_spread": 0.0,
                "retrieval_candidate_stability": 0.0,
                "retrieval_refreshed_fraction": 0.0,
            }

        by_device: dict[torch.device, list[torch.Tensor]] = {}
        for state in valid_states:
            vector = torch.stack(
                (
                    state.last_retrieval_score_spread,
                    state.last_retrieval_candidate_stability,
                    state.last_retrieval_refreshed,
                )
            )
            by_device.setdefault(vector.device, []).append(vector)
        totals = [0.0, 0.0, 0.0]
        for vectors in by_device.values():
            device_sum = torch.stack(vectors).sum(dim=0).tolist()
            totals = [left + float(right) for left, right in zip(totals, device_sum)]
        means = [value / len(valid_states) for value in totals]

        return {
            "retrieval_feature_valid": 1.0,
            "retrieval_score_spread": means[0],
            "retrieval_candidate_stability": means[1],
            "retrieval_refreshed_fraction": means[2],
        }

    def set_runtime_action(
        self,
        *,
        candidate_fraction: float,
        attention_fraction: float | None = None,
        stream_group_size: int | None = None,
    ) -> None:
        if attention_fraction is None:
            attention_fraction = candidate_fraction
        if not (
            0.0
            < attention_fraction
            <= candidate_fraction
            <= self.max_candidate_fraction
        ):
            raise ValueError(
                "runtime action must satisfy 0 < attention <= candidate <= "
                "the construction-time maximum"
            )
        if stream_group_size is None:
            stream_group_size = self.stream_group_size
        if stream_group_size <= 0:
            raise ValueError("stream_group_size must be positive")
        if self.candidate_selection_mode != "per_head_stream" and stream_group_size != 1:
            raise ValueError("stream_group_size only applies to per_head_stream")
        candidate_budget_changed = not math.isclose(
            float(candidate_fraction),
            self.candidate_fraction,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        for state in self.states:
            candidate_count = math.ceil(
                candidate_fraction * (state.capacity - 1)
            )
            if candidate_count * stream_group_size > state.exact_cache_count:
                raise ValueError(
                    "runtime action exceeds the physical exact-cache capacity"
                )
        self.candidate_fraction = float(candidate_fraction)
        self.attention_fraction = float(attention_fraction)
        self.stream_group_size = int(stream_group_size)
        if candidate_budget_changed:
            for state in self.states:
                state.cached_retrieved_candidates = None
                state.candidate_cache_age = 0

    def sequence_checkpoint(self) -> int:
        """Return the shared logical sequence length for counterfactual probes."""
        if not self.states:
            return 0
        lengths = {state.length for state in self.states}
        if len(lengths) != 1:
            raise RuntimeError("cache layer lengths are not synchronized")
        return lengths.pop()

    def restore_sequence_length(self, length: int) -> None:
        """Rewind appended tokens while retaining the warmed exact-KV directory."""
        for state in self.states:
            if not state.initial_length <= length <= state.length:
                raise ValueError(
                    "restored length must be between the initial and current length"
                )
        for state in self.states:
            state.length = int(length)

    @classmethod
    @torch.inference_mode()
    def from_dynamic_cache(
        cls,
        cache: Any,
        *,
        projection_dim: int = 32,
        index_bits: int = 8,
        index_mode: str = "pca_fixed",
        query_tail_by_layer: dict[int, torch.Tensor] | None = None,
        qk_metric_query_shrinkage: float = 0.75,
        variable_rate_budget: int = 15,
        fixed_bit_allocation: tuple[int, ...] | None = None,
        candidate_fraction: float = 0.02,
        candidate_min_tokens: int = 1,
        candidate_max_tokens: int | None = None,
        retrieval_backend: str = "full_topk",
        sampled_candidate_multiplier: float = 1.5,
        attention_fraction: float | None = None,
        exact_cache_fraction: float = 0.032,
        max_new_tokens: int = 512,
        directory_backend: str = "sorted",
        record_traces: bool = False,
        recent_fraction: float = 0.0,
        debug_directory: bool = False,
        selection_mode: str = "shared_sum",
        candidate_selection_mode: str | None = None,
        rerank_selection_mode: str | None = None,
        stream_group_size: int = 1,
        candidate_refresh_interval: int = 1,
        async_host_append: bool = True,
        async_conversion: bool = True,
        collect_retrieval_features: bool = False,
        _reuse_host_kv_by_layer: dict[int, torch.Tensor] | None = None,
    ) -> "HierarchicalPCACache":
        if index_mode not in {"pca_fixed", "qk_variable"}:
            raise ValueError("index_mode must be 'pca_fixed' or 'qk_variable'")
        if index_mode == "qk_variable":
            projection_dim = 128
        if projection_dim <= 0 or projection_dim % 4 != 0:
            raise ValueError("projection_dim must be a positive multiple of four")
        if index_mode == "pca_fixed" and index_bits not in {4, 8}:
            raise ValueError("index_bits must be 4 or 8")
        if index_mode == "qk_variable":
            if query_tail_by_layer is None:
                raise ValueError(
                    "qk_variable requires per-layer captured prefill queries"
                )
            if not 0.0 <= qk_metric_query_shrinkage <= 1.0:
                raise ValueError(
                    "qk_metric_query_shrinkage must be in [0, 1]"
                )
            if variable_rate_budget <= 0:
                raise ValueError("variable_rate_budget must be positive")
            if fixed_bit_allocation is not None:
                from run_head_top2_targeted_ppl_20260714 import (
                    _expand_packed_qmse_fixed_allocation,
                )

                _expand_packed_qmse_fixed_allocation(
                    fixed_bit_allocation,
                    1,
                    1,
                    torch.device("cpu"),
                )
        elif fixed_bit_allocation is not None:
            raise ValueError(
                "fixed_bit_allocation is only supported by qk_variable"
            )
        if retrieval_backend not in {"full_topk", "sampled_compact"}:
            raise ValueError(
                "retrieval_backend must be 'full_topk' or 'sampled_compact'"
            )
        if retrieval_backend == "sampled_compact":
            if index_mode != "qk_variable":
                raise ValueError(
                    "sampled_compact currently requires qk_variable"
                )
            if sampled_candidate_multiplier < 1.0:
                raise ValueError(
                    "sampled_candidate_multiplier must be at least one"
                )
        if attention_fraction is None:
            attention_fraction = candidate_fraction
        if candidate_min_tokens <= 0:
            raise ValueError("candidate_min_tokens must be positive")
        if candidate_max_tokens is not None:
            if candidate_max_tokens <= 0:
                raise ValueError(
                    "candidate_max_tokens must be positive when provided"
                )
            if candidate_max_tokens < candidate_min_tokens:
                raise ValueError(
                    "candidate_max_tokens cannot be smaller than "
                    "candidate_min_tokens"
                )
        if not (
            0.0 < attention_fraction <= candidate_fraction <= 1.0
            and 0.0 < exact_cache_fraction < 1.0
        ):
            raise ValueError(
                "expected 0 < attention_fraction <= candidate_fraction <= 1 "
                "and 0 < exact_cache_fraction < 1"
            )
        if directory_backend not in {"sorted", "fused"}:
            raise ValueError("directory_backend must be 'sorted' or 'fused'")
        candidate_selection_mode = candidate_selection_mode or selection_mode
        rerank_selection_mode = rerank_selection_mode or selection_mode
        valid_selection_modes = {"shared_sum", "shared_max", "head_balanced"}
        valid_candidate_selection_modes = valid_selection_modes | {
            "per_head",
            "per_head_union",
            "per_head_stream",
        }
        if candidate_selection_mode not in valid_candidate_selection_modes:
            raise ValueError(
                "candidate_selection_mode must be 'shared_sum', 'shared_max', "
                "'head_balanced', "
                "'per_head', 'per_head_union', or 'per_head_stream'"
            )
        if rerank_selection_mode not in valid_selection_modes:
            raise ValueError(
                "rerank_selection_mode must be 'shared_sum', 'shared_max', or "
                "'head_balanced'"
            )
        if not 0.0 <= recent_fraction < candidate_fraction:
            raise ValueError("recent_fraction must be in [0, candidate_fraction)")
        if (
            candidate_selection_mode not in {"shared_sum", "shared_max"}
            and recent_fraction > 0
        ):
            raise ValueError(
                "non-shared candidate selection does not yet support recent_fraction"
            )
        if candidate_selection_mode == "per_head_union" and directory_backend != "fused":
            raise ValueError("per_head_union requires the fused directory backend")
        if candidate_selection_mode == "per_head_stream" and directory_backend != "fused":
            raise ValueError("per_head_stream requires the fused directory backend")
        if stream_group_size <= 0:
            raise ValueError("stream_group_size must be positive")
        if candidate_refresh_interval <= 0:
            raise ValueError("candidate_refresh_interval must be positive")
        if candidate_refresh_interval > 1 and candidate_selection_mode != "per_head_stream":
            raise ValueError(
                "candidate refresh reuse currently requires per_head_stream"
            )
        if candidate_selection_mode != "per_head_stream" and stream_group_size != 1:
            raise ValueError("stream_group_size only applies to per_head_stream")
        if retrieval_backend == "sampled_compact":
            if candidate_selection_mode != "per_head_stream":
                raise ValueError(
                    "sampled_compact requires per_head_stream selection"
                )
            if stream_group_size != 1:
                raise ValueError(
                    "sampled_compact currently requires stream_group_size=1"
                )
            if candidate_refresh_interval != 1:
                raise ValueError(
                    "sampled_compact does not support temporal candidate reuse"
                )
            if recent_fraction != 0.0:
                raise ValueError(
                    "sampled_compact does not support recent_fraction"
                )
        if not hasattr(cache, "key_cache") or not hasattr(cache, "value_cache"):
            raise TypeError("expected a DynamicCache-like object")

        states: list[HierarchicalLayerState] = []
        original_gpu_bytes = 0
        conversion_devices: set[torch.device] = set()
        for layer, (key, value) in enumerate(zip(cache.key_cache, cache.value_cache)):
            if not key.is_cuda or not value.is_cuda:
                raise ValueError("source cache must be on CUDA")
            if key.shape != value.shape or key.dim() != 4:
                raise ValueError("expected matching [batch, kv_heads, seq, dim] K/V")
            batch, kv_heads, sequence, head_dim = key.shape
            conversion_devices.add(key.device)
            if projection_dim > head_dim:
                raise ValueError("projection_dim cannot exceed head_dim")
            original_gpu_bytes += (key.numel() + value.numel()) * key.element_size()
            capacity = int(sequence) + int(max_new_tokens)
            exact_cache_count = exact_cache_capacity(
                sequence_length=int(sequence),
                max_new_tokens=max_new_tokens,
                exact_cache_fraction=exact_cache_fraction,
                candidate_fraction=candidate_fraction,
                stream_group_size=stream_group_size,
                candidate_min_tokens=candidate_min_tokens,
                candidate_max_tokens=candidate_max_tokens,
            )

            sampled_key = key[..., ::32, :].float()
            second_moment = torch.einsum(
                "bhkd,bhke->bhde", sampled_key, sampled_key
            ) / float(sampled_key.shape[2])
            packed_index: dict[str, Any] | None = None
            packed_scale_metrics: torch.Tensor | None = None
            query_basis: torch.Tensor | None = None
            initial_codes: torch.Tensor | None = None
            initial_scales: torch.Tensor | None = None
            if index_mode == "qk_variable":
                import variablebit_spectral_cuda_20260727 as variablebit_cuda
                from run_head_top2_targeted_ppl_20260714 import (
                    _expand_packed_qmse_fixed_allocation,
                    _hierarchical_qmse_rate_allocation,
                    _qk_metric_projection_factors,
                )

                query_tail = query_tail_by_layer.get(layer)
                if query_tail is None:
                    raise ValueError(
                        f"missing captured prefill queries for layer {layer}"
                    )
                query_tail = query_tail.to(key.device)
                if (
                    query_tail.dim() != 4
                    or query_tail.shape[0] != batch
                    or query_tail.shape[-1] != head_dim
                    or query_tail.shape[1] % kv_heads != 0
                ):
                    raise ValueError(
                        f"invalid captured-query shape at layer {layer}: "
                        f"{tuple(query_tail.shape)}"
                    )
                query_groups = int(query_tail.shape[1]) // kv_heads
                grouped_queries = (
                    query_tail.reshape(
                        batch,
                        kv_heads,
                        query_groups,
                        query_tail.shape[-2],
                        head_dim,
                    )
                    .permute(0, 1, 3, 2, 4)
                    .reshape(
                        batch,
                        kv_heads,
                        query_tail.shape[-2] * query_groups,
                        head_dim,
                    )
                )
                query_second_moment = torch.einsum(
                    "bhnd,bhne->bhde",
                    grouped_queries.float(),
                    grouped_queries.float(),
                ) / float(grouped_queries.shape[-2])
                query_factor, key_factor = _qk_metric_projection_factors(
                    second_moment,
                    query_second_moment,
                    projection_dim=128,
                    query_shrinkage=qk_metric_query_shrinkage,
                )
                basis = key_factor.to(key.dtype)
                query_basis = query_factor.to(key.dtype)
                projected_sample = torch.einsum(
                    "bhkd,bhdm->bhkm",
                    sampled_key.to(key.dtype),
                    basis,
                )
                projected_queries = torch.einsum(
                    "bhnd,bhdm->bhnm",
                    grouped_queries.to(query_basis.dtype),
                    query_basis,
                )
                packed_scale_metrics = torch.einsum(
                    "bhqgd,bhqge->bhgde",
                    projected_queries.float().reshape(
                        batch,
                        kv_heads,
                        projected_queries.shape[-2],
                        8,
                        16,
                    ),
                    projected_queries.float().reshape(
                        batch,
                        kv_heads,
                        projected_queries.shape[-2],
                        8,
                        16,
                    ),
                ) / float(projected_queries.shape[-2])
                if fixed_bit_allocation is None:
                    allocation = _hierarchical_qmse_rate_allocation(
                        projected_sample,
                        projected_queries,
                        bit_budget_per_coordinate=variable_rate_budget,
                        allow_zero_bits=True,
                        include_scale_metadata=True,
                        query_covariance_shrinkage="none",
                        metric_scale_quantization=True,
                    )
                else:
                    allocation = _expand_packed_qmse_fixed_allocation(
                        fixed_bit_allocation,
                        batch,
                        kv_heads,
                        key.device,
                    )
                packed_index = variablebit_cuda.allocate_packed_index(
                    allocation.to(dtype=torch.int8),
                    capacity,
                    key.dtype,
                )
                for chunk_start in range(0, int(sequence), 4096):
                    chunk_stop = min(int(sequence), chunk_start + 4096)
                    projected_chunk = torch.einsum(
                        "bhkd,bhdm->bhkm",
                        key[..., chunk_start:chunk_stop, :],
                        basis,
                    )
                    variablebit_cuda.encode_projected_keys_into(
                        projected_chunk.contiguous(),
                        packed_index,
                        chunk_start,
                        scale_metrics=packed_scale_metrics,
                    )
                quantized = torch.empty(
                    (batch, kv_heads, 0, 0),
                    dtype=torch.uint8,
                    device=key.device,
                )
                scales = torch.empty(
                    (batch, kv_heads, 0, 0),
                    dtype=key.dtype,
                    device=key.device,
                )
            else:
                _, eigenvectors = torch.linalg.eigh(second_moment)
                basis = eigenvectors[..., -projection_dim:].to(key.dtype)
                query_basis = basis
                projected = torch.einsum("bhkd,bhdm->bhkm", key, basis)
                if index_bits == 4:
                    initial_codes, initial_scales = pack_projected_int4(
                        projected
                    )
                    code_dim = projection_dim // 2
                    code_dtype = torch.uint8
                else:
                    initial_scales = (
                        projected.float()
                        .abs()
                        .amax(dim=-1, keepdim=True)
                        .clamp_min(1.0e-8)
                        / 127.0
                    )
                    initial_codes = (
                        torch.round(projected.float() / initial_scales)
                        .clamp(-127, 127)
                        .to(torch.int8)
                    )
                    code_dim = projection_dim
                    code_dtype = torch.int8
                quantized = torch.empty(
                    (batch, kv_heads, capacity, code_dim),
                    dtype=code_dtype,
                    device=key.device,
                )
                scales = torch.empty(
                    (batch, kv_heads, capacity, 1),
                    dtype=key.dtype,
                    device=key.device,
                )
                quantized[..., :sequence, :] = initial_codes
                scales[..., :sequence, :] = initial_scales.to(key.dtype)

            if _reuse_host_kv_by_layer is None:
                host_kv = torch.empty(
                    (2, batch, kv_heads, capacity, head_dim),
                    dtype=key.dtype,
                    pin_memory=True,
                )
                host_kv[0, ..., :sequence, :].copy_(
                    key, non_blocking=async_conversion
                )
                host_kv[1, ..., :sequence, :].copy_(
                    value, non_blocking=async_conversion
                )
            else:
                host_kv = _reuse_host_kv_by_layer.get(layer)
                if host_kv is None:
                    raise ValueError(
                        f"missing reusable host K/V for layer {layer}"
                    )
                expected_prefix = (2, batch, kv_heads)
                if (
                    host_kv.device.type != "cpu"
                    or not host_kv.is_pinned()
                    or host_kv.dtype != key.dtype
                    or tuple(host_kv.shape[:3]) != expected_prefix
                    or int(host_kv.shape[-2]) < capacity
                    or int(host_kv.shape[-1]) != head_dim
                ):
                    raise ValueError(
                        "reusable host K/V must be pinned CPU storage with "
                        "matching dtype, batch, heads, capacity, and head dim"
                    )
            device_cache = torch.empty(
                (2, batch, kv_heads, exact_cache_count + 1, head_dim),
                dtype=key.dtype,
                device=key.device,
            )
            sentinel = (
                torch.iinfo(torch.int32).max
                if directory_backend == "sorted"
                else -1
            )
            resident_ids = torch.full(
                (batch, kv_heads, exact_cache_count),
                sentinel,
                dtype=torch.int32,
                device=key.device,
            )
            if directory_backend == "sorted":
                resident_slots = (
                    torch.arange(
                        exact_cache_count, dtype=torch.int32, device=key.device
                    )
                    .reshape(1, 1, -1)
                    .expand(batch, kv_heads, -1)
                    .clone()
                )
                table_keys = torch.empty(0, dtype=torch.int32, device=key.device)
                table_slots = torch.empty(0, dtype=torch.int32, device=key.device)
            else:
                resident_slots = torch.empty(0, dtype=torch.int32, device=key.device)
                table_size = 1 << math.ceil(math.log2(exact_cache_count * 1.6))
                table_keys = torch.full(
                    (batch, kv_heads, table_size),
                    -1,
                    dtype=torch.int32,
                    device=key.device,
                )
                table_slots = torch.empty_like(table_keys)
            resident_ages = torch.zeros_like(resident_ids, dtype=torch.uint8)
            padded_query = torch.zeros(
                (batch, kv_heads, 16, projection_dim),
                dtype=torch.int8,
                device=key.device,
            )
            states.append(
                HierarchicalLayerState(
                    host_kv=host_kv,
                    basis=basis,
                    quantized=quantized,
                    scales=scales,
                    padded_query=padded_query,
                    device_cache=device_cache,
                    resident_ids=resident_ids,
                    resident_slots=resident_slots,
                    resident_ages=resident_ages,
                    table_keys=table_keys,
                    table_slots=table_slots,
                    length=int(sequence),
                    initial_length=int(sequence),
                    capacity=capacity,
                    exact_cache_count=exact_cache_count,
                    query_basis=query_basis,
                    packed_index=packed_index,
                    packed_scale_metrics=packed_scale_metrics,
                    packed_indexed_count=(
                        int(sequence)
                        if packed_index is not None
                        else 0
                    ),
                )
            )
            cache.key_cache[layer] = torch.empty(0, dtype=key.dtype, device=key.device)
            cache.value_cache[layer] = torch.empty(0, dtype=value.dtype, device=value.device)
            del sampled_key, second_moment, initial_scales, initial_codes

        for device in conversion_devices:
            torch.cuda.synchronize(device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return cls(
            states,
            projection_dim=projection_dim,
            index_bits=index_bits,
            candidate_fraction=candidate_fraction,
            attention_fraction=attention_fraction,
            candidate_selection_mode=candidate_selection_mode,
            rerank_selection_mode=rerank_selection_mode,
            original_gpu_bytes=original_gpu_bytes,
            directory_backend=directory_backend,
            record_traces=record_traces,
            recent_fraction=recent_fraction,
            debug_directory=debug_directory,
            stream_group_size=stream_group_size,
            candidate_refresh_interval=candidate_refresh_interval,
            async_host_append=async_host_append,
            collect_retrieval_features=collect_retrieval_features,
            index_mode=index_mode,
            qk_metric_query_shrinkage=qk_metric_query_shrinkage,
            variable_rate_budget=variable_rate_budget,
            fixed_bit_allocation=fixed_bit_allocation,
            candidate_min_tokens=candidate_min_tokens,
            candidate_max_tokens=candidate_max_tokens,
            retrieval_backend=retrieval_backend,
            sampled_candidate_multiplier=sampled_candidate_multiplier,
        )

    @classmethod
    @torch.inference_mode()
    def from_offloaded_prefill_cache(
        cls,
        cache: Any,
        *,
        use_transient_quantized_key: bool = False,
        **conversion_kwargs: Any,
    ) -> "HierarchicalPCACache":
        """Convert exact host-resident prefill K/V one layer at a time.

        Only one full layer is materialized on GPU during conversion. This avoids
        overlapping the complete model-wide GPU KV cache with the hierarchical
        state, while reusing the validated dynamic-cache conversion path.
        """
        if not hasattr(cache, "completed_states") or not hasattr(cache, "synchronize"):
            raise TypeError("expected an OffloadedExactPrefillCache-like object")
        cache.synchronize()
        source_states = cache.completed_states()
        if not source_states:
            raise ValueError("offloaded prefill cache is empty")

        states: list[HierarchicalLayerState] = []
        prototype: HierarchicalPCACache | None = None
        original_gpu_bytes = 0
        all_query_tails = conversion_kwargs.get("query_tail_by_layer")
        for layer, source in enumerate(source_states):
            sequence = int(source.length)
            if sequence <= 0:
                raise ValueError("offloaded prefill layer is empty")
            if use_transient_quantized_key:
                if not (
                    hasattr(source, "quantized_kv")
                    and hasattr(source, "scales")
                    and hasattr(cache, "_dequantize")
                ):
                    raise ValueError(
                        "transient quantized-key conversion requires a "
                        "quantized offloaded prefill cache"
                    )
                key = cache._dequantize(
                    source.quantized_kv[0, ..., :sequence, :],
                    source.scales[0, ..., :sequence, :],
                    source.host_kv.dtype,
                    int(source.head_dim),
                )
            else:
                key = source.host_kv[0, ..., :sequence, :].to(
                    source.device, non_blocking=False
                )
            # Value contents are never consumed while constructing the index.
            # Reuse the key view to preserve the DynamicCache shape contract.
            value = key
            original_gpu_bytes += 2 * key.numel() * key.element_size()
            temporary = SimpleNamespace(key_cache=[key], value_cache=[value])
            layer_kwargs = dict(conversion_kwargs)
            if all_query_tails is not None:
                if layer not in all_query_tails:
                    raise ValueError(
                        f"missing captured prefill queries for layer {layer}"
                    )
                layer_kwargs["query_tail_by_layer"] = {
                    0: all_query_tails[layer]
                }
            layer_kwargs["_reuse_host_kv_by_layer"] = {0: source.host_kv}
            converted = cls.from_dynamic_cache(temporary, **layer_kwargs)
            state = converted.states[0]
            if source.host_kv.shape[-2] < state.capacity:
                raise ValueError(
                    "offloaded prefill host capacity is smaller than decode capacity"
                )
            if state.host_kv.data_ptr() != source.host_kv.data_ptr():
                raise RuntimeError("offloaded conversion did not reuse host K/V")
            del key, value, temporary
            states.append(state)
            prototype = converted

        assert prototype is not None
        return cls(
            states,
            projection_dim=prototype.projection_dim,
            index_bits=prototype.index_bits,
            candidate_fraction=prototype.candidate_fraction,
            attention_fraction=prototype.attention_fraction,
            candidate_selection_mode=prototype.candidate_selection_mode,
            rerank_selection_mode=prototype.rerank_selection_mode,
            original_gpu_bytes=original_gpu_bytes,
            directory_backend=prototype.directory_backend,
            record_traces=prototype.record_traces,
            recent_fraction=prototype.recent_fraction,
            debug_directory=prototype.debug_directory,
            stream_group_size=prototype.stream_group_size,
            candidate_refresh_interval=prototype.candidate_refresh_interval,
            async_host_append=prototype.async_host_append,
            collect_retrieval_features=prototype.collect_retrieval_features,
            index_mode=prototype.index_mode,
            qk_metric_query_shrinkage=(
                prototype.qk_metric_query_shrinkage
            ),
            variable_rate_budget=prototype.variable_rate_budget,
            fixed_bit_allocation=prototype.fixed_bit_allocation,
            candidate_min_tokens=prototype.candidate_min_tokens,
            candidate_max_tokens=prototype.candidate_max_tokens,
            retrieval_backend=prototype.retrieval_backend,
            sampled_candidate_multiplier=(
                prototype.sampled_candidate_multiplier
            ),
        )

    @torch.inference_mode()
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        if key_states.shape[-2] != 1 or value_states.shape[-2] != 1:
            raise ValueError("hierarchical cache decode requires one token at a time")
        state = self.states[layer_idx]
        position = state.length
        if position >= state.capacity:
            raise RuntimeError("hierarchical cache capacity exceeded")
        state.host_kv[0, ..., position : position + 1, :].copy_(
            key_states, non_blocking=self.async_host_append
        )
        state.host_kv[1, ..., position : position + 1, :].copy_(
            value_states, non_blocking=self.async_host_append
        )
        projected = torch.einsum("bhkd,bhdm->bhkm", key_states, state.basis)
        if self.index_mode == "qk_variable":
            if state.packed_index is None:
                raise RuntimeError("qk_variable state has no packed index")
            import variablebit_spectral_cuda_20260727 as variablebit_cuda

            variablebit_cuda.encode_projected_keys_into(
                projected.contiguous(),
                state.packed_index,
                position,
                scale_metrics=state.packed_scale_metrics,
            )
            state.packed_indexed_count = position + 1
        elif self.index_bits == 4:
            codes, scales = pack_projected_int4(projected)
            state.quantized[..., position : position + 1, :] = codes
            state.scales[..., position : position + 1, :] = scales.to(
                key_states.dtype
            )
        else:
            scales = (
                projected.float()
                .abs()
                .amax(dim=-1, keepdim=True)
                .clamp_min(1.0e-8)
                / 127.0
            )
            codes = (
                torch.round(projected.float() / scales)
                .clamp(-127, 127)
                .to(torch.int8)
            )
            state.quantized[..., position : position + 1, :] = codes
            state.scales[..., position : position + 1, :] = scales.to(
                key_states.dtype
            )
        self_slot = state.exact_cache_count
        state.device_cache[0, ..., self_slot : self_slot + 1, :].copy_(key_states)
        state.device_cache[1, ..., self_slot : self_slot + 1, :].copy_(value_states)
        state.length += 1
        return key_states, value_states

    @torch.inference_mode()
    def _resolve_fused_candidate_slots(
        self,
        state: HierarchicalLayerState,
        candidates: torch.Tensor,
        candidate_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from hierarchical_cache_cuda_20260715 import (
            load_hierarchical_cache_extension,
        )

        batch, kv_heads, candidate_count = candidates.shape
        module = load_hierarchical_cache_extension()
        flat_candidates = candidates.reshape(batch * kv_heads, candidate_count).contiguous()
        flat_resident_ids = state.resident_ids.reshape(
            batch * kv_heads, state.exact_cache_count
        )
        flat_resident_ages = state.resident_ages.reshape_as(flat_resident_ids)
        flat_table_keys = state.table_keys.reshape(
            batch * kv_heads, state.table_keys.shape[-1]
        )
        flat_table_slots = state.table_slots.reshape_as(flat_table_keys)
        flat_counts = (
            None
            if candidate_counts is None
            else candidate_counts.reshape(batch * kv_heads).contiguous()
        )
        if flat_counts is None:
            lookup_slots = module.hash_lookup_forward(
                flat_candidates, flat_table_keys, flat_table_slots
            )
            final_slots = module.variable_lru_update_forward(
                flat_resident_ids,
                flat_resident_ages,
                flat_candidates,
                lookup_slots,
            )
            self.hit_rate_tensors.append(lookup_slots.ge(0).float().mean())
        else:
            lookup_slots = module.hash_lookup_ragged_forward(
                flat_candidates,
                flat_counts,
                flat_table_keys,
                flat_table_slots,
            )
            final_slots = module.variable_lru_update_ragged_forward(
                flat_resident_ids,
                flat_resident_ages,
                flat_candidates,
                flat_counts,
                lookup_slots,
            )
            valid_count = flat_counts.sum().clamp_min(1)
            self.hit_rate_tensors.append(
                lookup_slots.ge(0).sum().float() / valid_count.float()
            )
        flat_table_keys.fill_(-1)
        module.hash_rebuild_forward(
            flat_resident_ids, flat_table_keys, flat_table_slots
        )
        batched_lookup_slots = lookup_slots.reshape(batch, kv_heads, candidate_count)
        batched_final_slots = final_slots.reshape(batch, kv_heads, candidate_count)
        for batch_index in range(batch):
            if candidate_counts is None:
                module.mapped_host_fill_variable_forward(
                    state.host_kv[:, batch_index],
                    state.device_cache[:, batch_index],
                    candidates[batch_index].contiguous(),
                    batched_lookup_slots[batch_index].contiguous(),
                    batched_final_slots[batch_index].contiguous(),
                )
            else:
                module.mapped_host_fill_variable_ragged_forward(
                    state.host_kv[:, batch_index],
                    state.device_cache[:, batch_index],
                    candidates[batch_index].contiguous(),
                    candidate_counts[batch_index].contiguous(),
                    batched_lookup_slots[batch_index].contiguous(),
                    batched_final_slots[batch_index].contiguous(),
                )
        return final_slots.reshape(batch, kv_heads, candidate_count)

    def _sampled_workspace(
        self,
        *,
        device: torch.device,
        batch: int,
        query_heads: int,
        candidate_capacity: int,
    ) -> dict[str, torch.Tensor]:
        key = (device.type, device.index, batch, query_heads)
        workspace = self.sampled_workspaces.get(key)
        if (
            workspace is not None
            and workspace["candidate_indices"].shape[-1]
            >= candidate_capacity
        ):
            return workspace
        workspace = {
            "candidate_indices": torch.empty(
                (batch, query_heads, candidate_capacity),
                dtype=torch.long,
                device=device,
            ),
            "candidate_scores": torch.empty(
                (batch, query_heads, candidate_capacity),
                dtype=torch.float32,
                device=device,
            ),
            "candidate_counts": torch.empty(
                (batch, query_heads),
                dtype=torch.long,
                device=device,
            ),
            "thresholds": torch.empty(
                (batch, query_heads),
                dtype=torch.float32,
                device=device,
            ),
            "overflow": torch.empty(
                (batch, query_heads),
                dtype=torch.bool,
                device=device,
            ),
        }
        self.sampled_workspaces[key] = workspace
        return workspace

    @torch.inference_mode()
    def _attend_qk_sampled(
        self,
        *,
        state: HierarchicalLayerState,
        grouped_query: torch.Tensor,
        query_codes: torch.Tensor,
        query_scales: torch.Tensor,
        scaling: float,
        history_count: int,
        candidate_count: int,
    ) -> torch.Tensor:
        import qabs_cuda_kernels as kernels
        import variablebit_spectral_cuda_20260727 as variablebit_cuda

        if state.packed_index is None:
            raise RuntimeError("qk_variable state has no packed index")
        batch, kv_heads, groups, head_dim = grouped_query.shape
        query_heads = kv_heads * groups
        selected_fraction = candidate_count / float(history_count)
        sample_count = min(
            2048,
            max(256, math.ceil(16.0 / selected_fraction)),
        )
        quantile_fraction_std = math.sqrt(
            selected_fraction
            * (1.0 - selected_fraction)
            / float(sample_count)
        )
        capacity_fraction = min(
            1.0,
            max(
                0.06,
                selected_fraction + 6.0 * quantile_fraction_std,
            ),
        )
        maximum_history = max(history_count, state.capacity - 1)
        candidate_capacity = min(
            maximum_history,
            max(
                1,
                math.ceil(capacity_fraction * maximum_history),
            ),
        )
        workspace = self._sampled_workspace(
            device=grouped_query.device,
            batch=batch,
            query_heads=query_heads,
            candidate_capacity=candidate_capacity,
        )
        packed = state.packed_index
        (
            candidate_indices,
            candidate_scores,
            raw_candidate_counts,
            _,
            overflow,
        ) = variablebit_cuda.sampled_threshold_compact_out(
            query_codes,
            query_scales,
            packed["packed_codes"],
            packed["key_scales"],
            packed["bit_allocations"],
            packed["code_offsets"],
            packed["scale_offsets"],
            packed["code_bases"],
            packed["scale_bases"],
            packed["code_strides"],
            packed["scale_strides"],
            workspace["candidate_indices"],
            workspace["candidate_scores"],
            workspace["candidate_counts"],
            workspace["thresholds"],
            workspace["overflow"],
            history_count,
            sample_count,
            selected_fraction,
            score_bias=packed.get("score_bias"),
        )
        buffer_capacity = int(candidate_indices.shape[-1])
        width = min(
            state.exact_cache_count,
            buffer_capacity,
            max(
                1,
                math.ceil(
                    self.sampled_candidate_multiplier * candidate_count
                ),
            ),
        )
        raw_counts = raw_candidate_counts.clamp(0, buffer_capacity)
        valid = (
            torch.arange(
                buffer_capacity,
                device=grouped_query.device,
            ).reshape(1, 1, -1)
            < raw_counts.unsqueeze(-1)
        )
        masked_scores = candidate_scores.masked_fill(~valid, -torch.inf)
        _, selected_positions = torch.topk(
            masked_scores,
            k=width,
            dim=-1,
            largest=True,
            sorted=True,
        )
        selected_candidates = torch.gather(
            candidate_indices,
            -1,
            selected_positions,
        ).reshape(batch, kv_heads, groups, width)
        selected_counts = raw_counts.clamp_max(width).reshape(
            batch, kv_heads, groups
        )
        selected_valid = (
            torch.arange(
                width,
                device=grouped_query.device,
            ).reshape(1, 1, 1, -1)
            < selected_counts.unsqueeze(-1)
        )
        selected_candidates = torch.where(
            selected_valid,
            selected_candidates,
            torch.full_like(selected_candidates, history_count - 1),
        ).to(torch.int32)
        self.sampled_candidate_count_tensors.append(
            selected_counts.float().mean()
        )
        self.sampled_overflow_tensors.append(overflow.float().mean())
        self.sampled_clipped_tensors.append(
            raw_counts.gt(width).float().mean()
        )

        group_outputs: list[torch.Tensor] = []
        for group in range(groups):
            group_candidates = selected_candidates[:, :, group, :].contiguous()
            group_counts = selected_counts[:, :, group].contiguous()
            group_slots = self._resolve_fused_candidate_slots(
                state,
                group_candidates,
                group_counts,
            )
            final_indices = torch.zeros(
                (batch, kv_heads, width + 1),
                dtype=torch.long,
                device=grouped_query.device,
            )
            final_indices[..., :width] = group_slots.to(torch.long)
            final_indices.scatter_(
                -1,
                group_counts.unsqueeze(-1),
                torch.full(
                    (batch, kv_heads, 1),
                    state.exact_cache_count,
                    dtype=torch.long,
                    device=grouped_query.device,
                ),
            )
            final_counts = group_counts + 1
            group_outputs.append(
                kernels.final_attention_ragged(
                    grouped_query[:, :, group, :].contiguous(),
                    state.device_cache[0],
                    state.device_cache[1],
                    final_indices.contiguous(),
                    final_counts.contiguous(),
                    float(scaling),
                )
            )
        return merge_streamed_gqa_outputs(
            group_outputs,
            query_heads,
            kv_heads,
        )

    @torch.inference_mode()
    def attend(
        self,
        layer_idx: int,
        query: torch.Tensor,
        scaling: float,
    ) -> torch.Tensor:
        import qabs_cuda_kernels as kernels

        state = self.states[layer_idx]
        history_count = state.length - 1
        batch, query_heads, query_length, head_dim = query.shape
        if query_length != 1:
            raise ValueError("hierarchical attention requires one query token")
        kv_heads = state.quantized.shape[1]
        if query_heads % kv_heads != 0:
            raise ValueError("query heads must be divisible by KV heads")
        groups = query_heads // kv_heads
        grouped_query = query.squeeze(2).reshape(batch, kv_heads, groups, head_dim)
        candidate_count = bounded_fraction_count(
            history_count,
            self.candidate_fraction,
            minimum_tokens=self.candidate_min_tokens,
            maximum_tokens=self.candidate_max_tokens,
        )
        if self.attention_fraction == self.candidate_fraction:
            attention_count = candidate_count
        else:
            attention_count = min(
                candidate_count,
                max(1, math.ceil(self.attention_fraction * history_count)),
            )
        recent_count = min(
            candidate_count - 1,
            math.floor(self.recent_fraction * history_count),
        )
        refresh_recent_count = (
            min(
                candidate_count - 1,
                self.candidate_refresh_interval - 1,
                history_count,
            )
            if self.candidate_selection_mode == "per_head_stream"
            else 0
        )
        retrieved_count = candidate_count - refresh_recent_count
        cached_retrieved = state.cached_retrieved_candidates
        reuse_temporal_candidates = (
            self.candidate_selection_mode == "per_head_stream"
            and self.candidate_refresh_interval > 1
            and cached_retrieved is not None
            and cached_retrieved.shape[-1] == retrieved_count
            and state.candidate_cache_age < self.candidate_refresh_interval - 1
        )
        group_scores: torch.Tensor | None = None
        shared_scores: torch.Tensor | None = None
        if not reuse_temporal_candidates:
            active_query_basis = (
                state.query_basis
                if state.query_basis is not None
                else state.basis
            )
            projected_query = torch.einsum(
                "bhgd,bhdm->bhgm", grouped_query, active_query_basis
            )
            if self.index_mode == "qk_variable":
                if state.packed_index is None:
                    raise RuntimeError(
                        "qk_variable state has no packed index"
                    )
                import variablebit_spectral_cuda_20260727 as variablebit_cuda

                query_codes, query_scales = (
                    variablebit_cuda.quantize_projected_query(
                        projected_query
                    )
                )
                packed = state.packed_index
                if self.retrieval_backend == "sampled_compact":
                    return self._attend_qk_sampled(
                        state=state,
                        grouped_query=grouped_query,
                        query_codes=query_codes,
                        query_scales=query_scales,
                        scaling=scaling,
                        history_count=history_count,
                        candidate_count=candidate_count,
                    )
                scores = variablebit_cuda.scores(
                    query_codes,
                    query_scales,
                    packed["packed_codes"],
                    packed["key_scales"],
                    packed["bit_allocations"],
                    packed["code_offsets"],
                    packed["scale_offsets"],
                    packed["code_bases"],
                    packed["scale_bases"],
                    packed["code_strides"],
                    packed["scale_strides"],
                    history_count,
                    score_bias=packed.get("score_bias"),
                )
            else:
                query_scales = (
                    projected_query.float()
                    .abs()
                    .amax(dim=-1, keepdim=True)
                    .clamp_min(1.0e-8)
                    / 127.0
                )
                query_codes = (
                    torch.round(projected_query.float() / query_scales)
                    .clamp(-127, 127)
                    .to(torch.int8)
                )
            if self.index_mode == "pca_fixed" and self.index_bits == 4:
                scores = kernels.pca_int4_scores(
                    query_codes.contiguous(),
                    state.quantized,
                    state.scales,
                    history_count,
                )
            elif self.index_mode == "pca_fixed":
                state.padded_query.zero_()
                state.padded_query[..., :groups, :].copy_(query_codes)
                scores = kernels.pca_int8_wmma_scores(
                    state.padded_query,
                    state.quantized,
                    state.scales,
                    history_count,
                    groups,
                )
            group_scores = scores.reshape(
                batch, kv_heads, groups, history_count
            )
            if self.candidate_selection_mode == "shared_max":
                shared_scores = group_scores.max(dim=2).values
            else:
                shared_scores = group_scores.sum(dim=2)
        per_head_candidates: torch.Tensor | None = None
        if self.candidate_selection_mode in {
            "per_head",
            "per_head_union",
            "per_head_stream",
        }:
            if reuse_temporal_candidates:
                assert cached_retrieved is not None
                retrieved_candidates = cached_retrieved
                state.candidate_cache_age += 1
                if (
                    self.collect_retrieval_features
                    and state.last_retrieval_score_spread is not None
                ):
                    state.last_retrieval_candidate_stability = torch.ones(
                        (), dtype=torch.float32, device=query.device
                    )
                    state.last_retrieval_refreshed = torch.zeros(
                        (), dtype=torch.float32, device=query.device
                    )
            else:
                assert group_scores is not None
                if refresh_recent_count:
                    group_scores[..., history_count - refresh_recent_count :] = -torch.inf
                retrieved_values, retrieved_candidates = torch.topk(
                    group_scores,
                    k=retrieved_count,
                    dim=-1,
                    sorted=False,
                )
                retrieved_candidates = retrieved_candidates.to(torch.int32)
                if self.collect_retrieval_features:
                    float_values = retrieved_values.float()
                    top_score = float_values.amax(dim=-1)
                    boundary_score = float_values.amin(dim=-1)
                    state.last_retrieval_score_spread = (
                        (top_score - boundary_score)
                        / (
                            top_score.abs()
                            + boundary_score.abs()
                            + torch.finfo(torch.float32).eps
                        )
                    ).mean()
                    probe_count = min(16, retrieved_count)
                    probe_positions = torch.topk(
                        retrieved_values,
                        k=probe_count,
                        dim=-1,
                        sorted=False,
                    ).indices
                    current_probe = torch.gather(
                        retrieved_candidates,
                        dim=-1,
                        index=probe_positions,
                    )
                    previous_probe = state.last_retrieval_probe
                    if (
                        previous_probe is None
                        or previous_probe.shape != current_probe.shape
                    ):
                        stability = torch.zeros(
                            (), dtype=torch.float32, device=query.device
                        )
                    else:
                        stability = (
                            current_probe.unsqueeze(-1)
                            .eq(previous_probe.unsqueeze(-2))
                            .any(dim=-1)
                            .float()
                            .mean()
                        )
                    state.last_retrieval_probe = current_probe.detach()
                    state.last_retrieval_candidate_stability = stability
                    state.last_retrieval_refreshed = torch.ones(
                        (), dtype=torch.float32, device=query.device
                    )
                if (
                    self.candidate_selection_mode == "per_head_stream"
                    and self.candidate_refresh_interval > 1
                ):
                    state.cached_retrieved_candidates = retrieved_candidates.detach()
                    state.candidate_cache_age = 0
            if refresh_recent_count:
                recent_candidates = torch.arange(
                    history_count - refresh_recent_count,
                    history_count,
                    dtype=torch.int32,
                    device=query.device,
                ).reshape(1, 1, 1, -1).expand(
                    batch, kv_heads, groups, -1
                )
                per_head_candidates = torch.cat(
                    (retrieved_candidates, recent_candidates), dim=-1
                )
            else:
                per_head_candidates = retrieved_candidates
            if self.candidate_selection_mode == "per_head_union":
                virtual_candidates = per_head_candidates.reshape(
                    batch, query_heads, candidate_count
                ).to(torch.long)
                compact_union, union_counts = kernels.candidate_union_compact(
                    virtual_candidates,
                    history_count,
                    groups,
                    state.exact_cache_count,
                )
                maximum_union_count = int(union_counts.max().item())
                if maximum_union_count > state.exact_cache_count:
                    raise RuntimeError(
                        "per-head candidate union exceeds exact cache capacity"
                    )
                shared_candidates = compact_union[..., :maximum_union_count]
                positions = torch.arange(
                    maximum_union_count,
                    device=query.device,
                    dtype=union_counts.dtype,
                ).reshape(1, 1, -1)
                padding = positions >= union_counts.unsqueeze(-1)
                shared_candidates = torch.where(
                    padding,
                    shared_candidates[..., :1],
                    shared_candidates,
                )
                union_fractions = union_counts.float() / float(history_count)
                self.candidate_union_mean_tensors.append(union_fractions.mean())
                self.candidate_union_max_tensors.append(union_fractions.max())
            else:
                shared_candidates = per_head_candidates.flatten(start_dim=2)
        elif recent_count > 0:
            assert shared_scores is not None
            shared_scores[..., history_count - recent_count :] = -torch.inf
            retrieved = torch.topk(
                shared_scores,
                k=candidate_count - recent_count,
                dim=-1,
                sorted=False,
            ).indices
            recent = torch.arange(
                history_count - recent_count,
                history_count,
                dtype=torch.long,
                device=query.device,
            ).reshape(1, 1, -1).expand(batch, kv_heads, -1)
            shared_candidates = torch.cat((retrieved, recent), dim=-1).to(
                torch.int32
            )
        elif self.candidate_selection_mode == "head_balanced":
            assert group_scores is not None
            shared_candidates = head_balanced_topk_offsets(
                group_scores, candidate_count
            ).to(torch.int32)
        else:
            assert shared_scores is not None
            shared_candidates = torch.topk(
                shared_scores,
                k=candidate_count,
                dim=-1,
                sorted=False,
            ).indices.to(torch.int32)
        physical_candidate_count = (
            candidate_count * min(self.stream_group_size, groups)
            if self.candidate_selection_mode == "per_head_stream"
            else shared_candidates.shape[-1]
        )
        if physical_candidate_count > state.exact_cache_count:
            raise RuntimeError(
                "physical candidate count exceeds the exact cache capacity; "
                "increase exact_cache_fraction"
            )
        if self.candidate_selection_mode == "per_head_stream":
            if groups % self.stream_group_size != 0:
                raise RuntimeError(
                    "stream_group_size must divide the number of GQA groups"
                )
            group_outputs = []
            for group_start in range(0, groups, self.stream_group_size):
                group_end = group_start + self.stream_group_size
                chunk_candidates = per_head_candidates[
                    :, :, group_start:group_end, :
                ]
                if self.record_traces:
                    virtual_chunk = chunk_candidates.reshape(
                        batch,
                        kv_heads * self.stream_group_size,
                        candidate_count,
                    ).to(torch.long)
                    union_counts = kernels.candidate_union_counts(
                        virtual_chunk,
                        history_count,
                        self.stream_group_size,
                    )
                    union_fractions = union_counts.float() / float(history_count)
                    self.candidate_union_mean_tensors.append(
                        union_fractions.mean()
                    )
                    self.candidate_union_max_tensors.append(
                        union_fractions.max()
                    )
                if self.stream_group_size == 1:
                    group_slots = self._resolve_fused_candidate_slots(
                        state, chunk_candidates.squeeze(2)
                    ).unsqueeze(2)
                else:
                    group_slots = self._resolve_fused_candidate_slots(
                        state,
                        chunk_candidates.flatten(start_dim=2),
                    ).reshape(
                        batch,
                        kv_heads,
                        self.stream_group_size,
                        candidate_count,
                    )
                if attention_count < candidate_count:
                    group_slots = exact_per_head_rerank_slots(
                        grouped_query[:, :, group_start:group_end, :],
                        state.device_cache[0],
                        group_slots,
                        attention_count,
                    )
                chunk_query_heads = kv_heads * self.stream_group_size
                self_indices = torch.full(
                    (batch, chunk_query_heads, 1),
                    state.exact_cache_count,
                    dtype=torch.long,
                    device=query.device,
                )
                final_indices = torch.cat(
                    (
                        group_slots.reshape(
                            batch, chunk_query_heads, attention_count
                        ).to(torch.long),
                        self_indices,
                    ),
                    dim=-1,
                )
                counts = torch.full(
                    (batch, chunk_query_heads),
                    attention_count + 1,
                    dtype=torch.long,
                    device=query.device,
                )
                group_outputs.append(
                    kernels.final_attention_ragged(
                        grouped_query[:, :, group_start:group_end, :]
                        .reshape(batch, chunk_query_heads, head_dim)
                        .contiguous(),
                        state.device_cache[0],
                        state.device_cache[1],
                        final_indices.contiguous(),
                        counts,
                        float(scaling),
                    )
                )
            output = merge_streamed_gqa_outputs(
                group_outputs, query_heads, kv_heads
            )
            if self.record_traces:
                state.last_attention_output = output.detach().clone()
            return output
        previous_candidates = state.last_shared_candidates
        if self.record_traces or (self.debug_directory and layer_idx == 0):
            state.last_shared_candidates = shared_candidates.detach().clone()
        per_head_final_slots: torch.Tensor | None = None
        if self.directory_backend == "sorted":
            directory = update_sorted_directory(
                shared_candidates,
                state.resident_ids,
                state.resident_slots,
                state.resident_ages,
            )
            self.hit_rates.append(directory.hit_rate)
            final_slots = directory.final_slots
            for batch_index, head, miss_tokens, destinations in directory.misses:
                host_tokens = miss_tokens.detach().cpu().to(torch.long)
                exact_kv = state.host_kv[
                    :, batch_index, head, host_tokens, :
                ].to(query.device, non_blocking=True)
                state.device_cache[
                    :, batch_index, head, destinations.to(torch.long), :
                ] = exact_kv
        else:
            from hierarchical_cache_cuda_20260715 import (
                load_hierarchical_cache_extension,
            )

            module = load_hierarchical_cache_extension()
            flat_candidates = shared_candidates.reshape(
                batch * kv_heads, shared_candidates.shape[-1]
            ).contiguous()
            flat_resident_ids = state.resident_ids.reshape(
                batch * kv_heads, state.exact_cache_count
            )
            flat_resident_ages = state.resident_ages.reshape_as(flat_resident_ids)
            flat_table_keys = state.table_keys.reshape(
                batch * kv_heads, state.table_keys.shape[-1]
            )
            flat_table_slots = state.table_slots.reshape_as(flat_table_keys)
            lookup_slots = module.hash_lookup_forward(
                flat_candidates, flat_table_keys, flat_table_slots
            )
            flat_final_slots = module.variable_lru_update_forward(
                flat_resident_ids,
                flat_resident_ages,
                flat_candidates,
                lookup_slots,
            )
            self.hit_rate_tensors.append(lookup_slots.ge(0).float().mean())
            flat_table_keys.fill_(-1)
            module.hash_rebuild_forward(
                flat_resident_ids, flat_table_keys, flat_table_slots
            )
            if self.candidate_selection_mode == "per_head_union":
                virtual_candidates = per_head_candidates.reshape(
                    batch * kv_heads, groups * candidate_count
                ).contiguous()
                virtual_slots = module.hash_lookup_forward(
                    virtual_candidates, flat_table_keys, flat_table_slots
                )
                if bool(virtual_slots.lt(0).any().item()):
                    raise RuntimeError("a virtual per-head candidate is absent from the union")
                per_head_final_slots = virtual_slots.reshape(
                    batch, kv_heads, groups, candidate_count
                )
            batched_candidates = flat_candidates.reshape(
                batch, kv_heads, shared_candidates.shape[-1]
            )
            batched_lookup_slots = lookup_slots.reshape_as(batched_candidates)
            batched_final_slots = flat_final_slots.reshape_as(batched_candidates)
            for batch_index in range(batch):
                module.mapped_host_fill_variable_forward(
                    state.host_kv[:, batch_index],
                    state.device_cache[:, batch_index],
                    batched_candidates[batch_index].contiguous(),
                    batched_lookup_slots[batch_index].contiguous(),
                    batched_final_slots[batch_index].contiguous(),
                )
            final_slots = flat_final_slots.reshape_as(shared_candidates)
            if self.debug_directory and layer_idx == 0:
                resident_at_slots = torch.gather(
                    flat_resident_ids, 1, flat_final_slots.to(torch.long)
                )
                previous_overlap = 0.0
                if previous_candidates is not None:
                    previous_flat = previous_candidates.reshape(
                        batch * kv_heads, -1
                    )
                    previous_overlap = float(
                        flat_candidates.unsqueeze(-1)
                        .eq(previous_flat.unsqueeze(-2))
                        .any(dim=-1)
                        .float()
                        .mean()
                        .item()
                    )
                print(
                    json.dumps(
                        {
                            "directory_debug_layer": layer_idx,
                            "history_count": history_count,
                            "lookup_hit_rate": float(
                                lookup_slots.ge(0).float().mean().item()
                            ),
                            "candidate_overlap_previous": previous_overlap,
                            "resident_key_match": float(
                                resident_at_slots.eq(flat_candidates).float().mean().item()
                            ),
                            "table_occupancy": int(flat_table_keys.ge(0).sum().item()),
                            "slot_min": int(flat_final_slots.min().item()),
                            "slot_max": int(flat_final_slots.max().item()),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        if per_head_candidates is not None:
            per_head_slots = per_head_final_slots
            if per_head_slots is None:
                per_head_slots = final_slots.reshape(
                    batch, kv_heads, groups, candidate_count
                )
            if attention_count < candidate_count:
                per_head_slots = exact_per_head_rerank_slots(
                    grouped_query,
                    state.device_cache[0],
                    per_head_slots,
                    attention_count,
                )
            expanded_slots = per_head_slots.reshape(
                batch, query_heads, attention_count
            ).to(torch.long)
        else:
            attention_slots = final_slots
            if attention_count < candidate_count:
                attention_slots = exact_shared_rerank_slots(
                    grouped_query,
                    state.device_cache[0],
                    final_slots,
                    attention_count,
                    self.rerank_selection_mode,
                )

            expanded_slots = (
                attention_slots.unsqueeze(2)
                .expand(-1, -1, groups, -1)
                .reshape(batch, query_heads, attention_count)
                .to(torch.long)
            )
        self_indices = torch.full(
            (batch, query_heads, 1),
            state.exact_cache_count,
            dtype=torch.long,
            device=query.device,
        )
        final_indices = torch.cat((expanded_slots, self_indices), dim=-1)
        counts = torch.full(
            (batch, query_heads),
            attention_count + 1,
            dtype=torch.long,
            device=query.device,
        )
        output = kernels.final_attention_ragged(
            query.squeeze(2).contiguous(),
            state.device_cache[0],
            state.device_cache[1],
            final_indices.contiguous(),
            counts,
            float(scaling),
        )
        if self.record_traces:
            state.last_attention_output = output.detach().clone()
        return output

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        if not self.states:
            return 0
        return int(self.states[0 if layer_idx is None else layer_idx].length)

    def get_max_cache_shape(self) -> int | None:
        return None

    def __len__(self) -> int:
        return len(self.states)

    def persistent_gpu_bytes(self) -> int:
        tensors: list[torch.Tensor] = []
        for state in self.states:
            tensors.extend(
                [
                    state.basis,
                    state.quantized,
                    state.scales,
                    state.padded_query,
                    state.device_cache,
                    state.resident_ids,
                    state.resident_slots,
                    state.resident_ages,
                    state.table_keys,
                    state.table_slots,
                ]
            )
            if (
                state.query_basis is not None
                and state.query_basis.data_ptr() != state.basis.data_ptr()
            ):
                tensors.append(state.query_basis)
            if state.packed_scale_metrics is not None:
                tensors.append(state.packed_scale_metrics)
            if state.packed_index is not None:
                tensors.extend(
                    value
                    for value in state.packed_index.values()
                    if isinstance(value, torch.Tensor)
                )
            if state.cached_retrieved_candidates is not None:
                tensors.append(state.cached_retrieved_candidates)
        for workspace in self.sampled_workspaces.values():
            tensors.extend(workspace.values())
        storage_bytes: dict[tuple[str, int | None, int], int] = {}
        for tensor in tensors:
            if not tensor.is_cuda or tensor.numel() == 0:
                continue
            storage = tensor.untyped_storage()
            key = (
                tensor.device.type,
                tensor.device.index,
                int(storage.data_ptr()),
            )
            storage_bytes[key] = max(storage_bytes.get(key, 0), storage.nbytes())
        return sum(storage_bytes.values())

    def mean_cache_hit_rate(self) -> float:
        values = list(self.hit_rates)
        if self.hit_rate_tensors:
            values.extend(float(value.item()) for value in self.hit_rate_tensors)
        return sum(values) / max(1, len(values))

    def mean_sampled_candidate_count(self) -> float | None:
        if not self.sampled_candidate_count_tensors:
            return None
        return sum(
            float(value.item())
            for value in self.sampled_candidate_count_tensors
        ) / len(self.sampled_candidate_count_tensors)

    def mean_sampled_overflow_rate(self) -> float | None:
        if not self.sampled_overflow_tensors:
            return None
        return sum(
            float(value.item()) for value in self.sampled_overflow_tensors
        ) / len(self.sampled_overflow_tensors)

    def mean_sampled_clipped_fraction(self) -> float | None:
        if not self.sampled_clipped_tensors:
            return None
        return sum(
            float(value.item()) for value in self.sampled_clipped_tensors
        ) / len(self.sampled_clipped_tensors)

    def mean_candidate_union_fraction(self) -> float | None:
        if not self.candidate_union_mean_tensors:
            return None
        return sum(
            float(value.item()) for value in self.candidate_union_mean_tensors
        ) / len(self.candidate_union_mean_tensors)

    def max_candidate_union_fraction(self) -> float | None:
        if not self.candidate_union_max_tensors:
            return None
        return max(float(value.item()) for value in self.candidate_union_max_tensors)

    def pinned_host_bytes(self) -> int:
        return sum(
            state.host_kv.numel() * state.host_kv.element_size()
            for state in self.states
        )


_ORIGINAL_LLAMA_ATTENTION_FORWARD: Any | None = None
_ORIGINAL_QWEN3_ATTENTION_FORWARD: Any | None = None


def _hierarchical_llama_attention_forward(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_value: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not isinstance(past_key_value, HierarchicalPCACache):
        if _ORIGINAL_LLAMA_ATTENTION_FORWARD is None:
            raise RuntimeError("original LlamaAttention.forward was not installed")
        return _ORIGINAL_LLAMA_ATTENTION_FORWARD(
            module,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )

    del attention_mask, cache_position, kwargs
    import transformers.models.llama.modeling_llama as modeling_llama

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    query_states = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = modeling_llama.apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )
    past_key_value.update(
        key_states,
        value_states,
        int(module.layer_idx),
        {"sin": sin, "cos": cos},
    )
    attention_output = past_key_value.attend(
        int(module.layer_idx), query_states, float(module.scaling)
    )
    attention_output = attention_output.reshape(*input_shape, -1).contiguous()
    attention_output = module.o_proj(attention_output)
    return attention_output, None


@contextmanager
def hierarchical_llama_attention_mode(
    model: torch.nn.Module | None = None,
) -> Iterator[None]:
    global _ORIGINAL_LLAMA_ATTENTION_FORWARD
    import transformers.models.llama.modeling_llama as modeling_llama

    previous = modeling_llama.LlamaAttention.forward
    if _ORIGINAL_LLAMA_ATTENTION_FORWARD is None:
        _ORIGINAL_LLAMA_ATTENTION_FORWARD = previous
    modeling_llama.LlamaAttention.forward = _hierarchical_llama_attention_forward
    accelerated_instances: list[tuple[torch.nn.Module, Any]] = []
    if model is not None:
        for module in model.modules():
            if isinstance(module, modeling_llama.LlamaAttention) and hasattr(
                module, "_old_forward"
            ):
                accelerated_instances.append((module, module._old_forward))
                module._old_forward = MethodType(
                    _hierarchical_llama_attention_forward, module
                )
    try:
        yield
    finally:
        for module, original in accelerated_instances:
            module._old_forward = original
        modeling_llama.LlamaAttention.forward = previous


def _hierarchical_qwen3_attention_forward(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_value: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not isinstance(past_key_value, HierarchicalPCACache):
        if _ORIGINAL_QWEN3_ATTENTION_FORWARD is None:
            raise RuntimeError("original Qwen3Attention.forward was not installed")
        return _ORIGINAL_QWEN3_ATTENTION_FORWARD(
            module,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )

    del attention_mask, cache_position, kwargs
    import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    query_states = module.q_norm(
        module.q_proj(hidden_states).view(hidden_shape)
    ).transpose(1, 2)
    key_states = module.k_norm(
        module.k_proj(hidden_states).view(hidden_shape)
    ).transpose(1, 2)
    value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = modeling_qwen3.apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )
    past_key_value.update(
        key_states,
        value_states,
        int(module.layer_idx),
        {"sin": sin, "cos": cos},
    )
    attention_output = past_key_value.attend(
        int(module.layer_idx), query_states, float(module.scaling)
    )
    attention_output = attention_output.reshape(*input_shape, -1).contiguous()
    attention_output = module.o_proj(attention_output)
    return attention_output, None


@contextmanager
def hierarchical_qwen3_attention_mode(
    model: torch.nn.Module | None = None,
) -> Iterator[None]:
    global _ORIGINAL_QWEN3_ATTENTION_FORWARD
    import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3

    previous = modeling_qwen3.Qwen3Attention.forward
    if _ORIGINAL_QWEN3_ATTENTION_FORWARD is None:
        _ORIGINAL_QWEN3_ATTENTION_FORWARD = previous
    modeling_qwen3.Qwen3Attention.forward = _hierarchical_qwen3_attention_forward
    accelerated_instances: list[tuple[torch.nn.Module, Any]] = []
    if model is not None:
        for module in model.modules():
            if isinstance(module, modeling_qwen3.Qwen3Attention) and hasattr(
                module, "_old_forward"
            ):
                accelerated_instances.append((module, module._old_forward))
                module._old_forward = MethodType(
                    _hierarchical_qwen3_attention_forward, module
                )
    try:
        yield
    finally:
        for module, original in accelerated_instances:
            module._old_forward = original
        modeling_qwen3.Qwen3Attention.forward = previous


@contextmanager
def hierarchical_attention_mode(
    model: torch.nn.Module | None = None,
) -> Iterator[None]:
    """Install hierarchical attention handlers for supported decoder families."""
    with hierarchical_llama_attention_mode(model), hierarchical_qwen3_attention_mode(
        model
    ):
        yield
