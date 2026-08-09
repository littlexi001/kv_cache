from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from hierarchical_pca_cache_20260715 import HierarchicalPCACache
import variablebit_spectral_cuda_20260727 as variablebit_cuda


@torch.inference_mode()
def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(20260727)
    device = torch.device("cuda")
    dtype = torch.float16
    batch = 1
    kv_heads = 2
    query_heads = 8
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

    groups = query_heads // kv_heads
    grouped_query = query.squeeze(2).reshape(
        batch,
        kv_heads,
        groups,
        head_dim,
    )
    projected_query = torch.einsum(
        "bhgd,bhdm->bhgm",
        grouped_query,
        state.query_basis,
    )
    query_codes, query_scales = variablebit_cuda.quantize_projected_query(
        projected_query
    )
    packed = state.packed_index
    assert packed is not None
    approximate_scores = variablebit_cuda.scores(
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
        history,
    ).reshape(batch, kv_heads, groups, history)
    candidate_count = math.ceil(candidate_fraction * history)
    candidates = torch.topk(
        approximate_scores,
        k=candidate_count,
        dim=-1,
        sorted=False,
    ).indices

    expected_heads = []
    for query_head in range(query_heads):
        kv_head = query_head // groups
        group = query_head % groups
        indices = candidates[0, kv_head, group]
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
    actual = cache.attend(0, query, scaling).float()
    max_error = float((actual - expected).abs().max().item())
    mean_error = float((actual - expected).abs().mean().item())
    if max_error > 2.0e-2:
        raise AssertionError(
            f"physical packed attention mismatch: max={max_error}"
        )
    if state.packed_indexed_count != history + 1:
        raise AssertionError("appended key was not encoded")
    print(
        {
            "max_abs_error": max_error,
            "mean_abs_error": mean_error,
            "candidate_count": candidate_count,
            "persistent_ratio": (
                cache.persistent_gpu_bytes()
                / cache.original_gpu_bytes
            ),
        }
    )


if __name__ == "__main__":
    main()
