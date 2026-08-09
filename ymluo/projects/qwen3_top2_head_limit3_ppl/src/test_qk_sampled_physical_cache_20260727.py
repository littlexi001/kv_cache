from __future__ import annotations

import math
from types import MethodType, SimpleNamespace

import torch

from hierarchical_pca_cache_20260715 import HierarchicalPCACache


@torch.inference_mode()
def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(20260728)
    device = torch.device("cuda")
    dtype = torch.float16
    batch = 1
    kv_heads = 2
    query_heads = 8
    groups = query_heads // kv_heads
    history = 2048
    head_dim = 128
    candidate_fraction = 0.0625
    scaling = 1.0 / math.sqrt(head_dim)

    key = torch.randn(
        batch,
        kv_heads,
        history,
        head_dim,
        device=device,
        dtype=dtype,
    )
    value = torch.randn_like(key)
    query_tail = torch.randn(
        batch,
        query_heads,
        8,
        head_dim,
        device=device,
        dtype=dtype,
    )
    source = SimpleNamespace(
        key_cache=[key.clone()],
        value_cache=[value.clone()],
    )
    cache = HierarchicalPCACache.from_dynamic_cache(
        source,
        index_mode="qk_variable",
        query_tail_by_layer={0: query_tail},
        qk_metric_query_shrinkage=0.75,
        variable_rate_budget=15,
        candidate_fraction=candidate_fraction,
        attention_fraction=candidate_fraction,
        exact_cache_fraction=0.30,
        max_new_tokens=4,
        directory_backend="fused",
        candidate_selection_mode="per_head_stream",
        rerank_selection_mode="shared_sum",
        stream_group_size=1,
        retrieval_backend="sampled_compact",
        sampled_candidate_multiplier=1.5,
        async_conversion=False,
        async_host_append=False,
    )
    state = cache.states[0]
    new_key = torch.randn(
        batch,
        kv_heads,
        1,
        head_dim,
        device=device,
        dtype=dtype,
    )
    new_value = torch.randn_like(new_key)
    query = torch.randn(
        batch,
        query_heads,
        1,
        head_dim,
        device=device,
        dtype=dtype,
    )
    cache.update(new_key, new_value, 0)

    resolved_candidates: list[tuple[torch.Tensor, torch.Tensor]] = []
    original_resolve = cache._resolve_fused_candidate_slots

    def recording_resolve(
        self: HierarchicalPCACache,
        active_state,
        candidates: torch.Tensor,
        candidate_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if candidate_counts is None:
            raise AssertionError("sampled physical path must use ragged counts")
        resolved_candidates.append(
            (candidates.detach().clone(), candidate_counts.detach().clone())
        )
        return original_resolve(
            active_state,
            candidates,
            candidate_counts,
        )

    cache._resolve_fused_candidate_slots = MethodType(
        recording_resolve,
        cache,
    )
    actual = cache.attend(0, query, scaling).float()
    if len(resolved_candidates) != groups:
        raise AssertionError("one ragged directory pass is required per GQA group")

    expected_heads = []
    for query_head in range(query_heads):
        kv_head = query_head // groups
        group = query_head % groups
        group_candidates, group_counts = resolved_candidates[group]
        count = int(group_counts[0, kv_head].item())
        indices = group_candidates[0, kv_head, :count].to(torch.long)
        selected_key = torch.cat(
            (key[0, kv_head, indices], new_key[0, kv_head]),
            dim=0,
        )
        selected_value = torch.cat(
            (value[0, kv_head, indices], new_value[0, kv_head]),
            dim=0,
        )
        exact_scores = (
            query[0, query_head, 0].float()
            @ selected_key.float().transpose(0, 1)
        ) * scaling
        expected_heads.append(
            torch.softmax(exact_scores, dim=-1)
            @ selected_value.float()
        )
    expected = torch.stack(expected_heads).reshape(
        batch,
        1,
        query_heads,
        head_dim,
    )
    max_error = float((actual - expected).abs().max().item())
    mean_error = float((actual - expected).abs().mean().item())
    if max_error > 2.0e-2:
        raise AssertionError(
            f"sampled physical attention mismatch: max={max_error}"
        )
    candidate_counts = torch.cat(
        [counts.reshape(-1) for _, counts in resolved_candidates]
    )
    maximum_width = max(
        candidates.shape[-1]
        for candidates, _ in resolved_candidates
    )
    if bool((candidate_counts > maximum_width).any().item()):
        raise AssertionError("ragged count exceeds the physical candidate width")
    if state.packed_indexed_count != history + 1:
        raise AssertionError("appended key was not encoded")
    if cache.mean_sampled_candidate_count() is None:
        raise AssertionError("sampled candidate statistics were not recorded")
    print(
        {
            "max_abs_error": max_error,
            "mean_abs_error": mean_error,
            "candidate_count_mean": float(candidate_counts.float().mean().item()),
            "candidate_count_min": int(candidate_counts.min().item()),
            "candidate_count_max": int(candidate_counts.max().item()),
            "candidate_width": maximum_width,
            "overflow_rate": cache.mean_sampled_overflow_rate(),
            "clipped_fraction": cache.mean_sampled_clipped_fraction(),
            "persistent_ratio": (
                cache.persistent_gpu_bytes()
                / cache.original_gpu_bytes
            ),
        }
    )


if __name__ == "__main__":
    main()
