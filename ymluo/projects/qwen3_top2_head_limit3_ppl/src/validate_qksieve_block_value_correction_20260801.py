"""Deterministic validation for QKSieve block-Value tail correction."""

from __future__ import annotations

import torch

import run_head_top2_targeted_ppl_20260714 as sparse


def exact_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: torch.Tensor | None,
    scaling: float,
) -> torch.Tensor:
    groups = query.shape[1] // key.shape[1]
    expanded_key = key.repeat_interleave(groups, dim=1)
    expanded_value = value.repeat_interleave(groups, dim=1)
    if indices is not None:
        current = torch.full(
            (*indices.shape[:-1], 1),
            key.shape[2] - 1,
            dtype=torch.long,
        )
        selected = torch.cat((indices, current), dim=-1)
        expanded_key = torch.gather(
            expanded_key,
            2,
            selected.unsqueeze(-1).expand(*selected.shape, key.shape[-1]),
        )
        expanded_value = torch.gather(
            expanded_value,
            2,
            selected.unsqueeze(-1).expand(*selected.shape, value.shape[-1]),
        )
    scores = torch.einsum("bhd,bhnd->bhn", query, expanded_key) * scaling
    output = torch.einsum(
        "bhn,bhnd->bhd",
        torch.softmax(scores.float(), dim=-1),
        expanded_value.float(),
    )
    return output.unsqueeze(1)


def main() -> None:
    torch.manual_seed(20260801)
    batch, kv_heads, groups, tokens, dim = 1, 2, 2, 11, 8
    query = torch.randn(batch, kv_heads * groups, dim)
    key = torch.randn(batch, kv_heads, tokens, dim)
    value = torch.randn(batch, kv_heads, tokens, dim)
    scaling = dim**-0.5
    proxy_scores = torch.einsum(
        "bhgd,bhnd->bhgn",
        query.reshape(batch, kv_heads, groups, dim),
        key[..., :-1, :],
    )
    candidates = torch.tensor(
        [[[0, 3, 7], [1, 4, 8], [2, 5, 6], [0, 6, 9]]],
        dtype=torch.long,
    )
    counts = torch.full((batch, kv_heads * groups), 3, dtype=torch.long)
    selected_output = exact_attention(
        query,
        key,
        value,
        candidates,
        scaling,
    )
    corrected = sparse._block_mean_value_corrected_output(
        selected_output,
        key[..., :-1, :],
        value,
        proxy_scores,
        query,
        key[..., -1, :],
        candidates,
        counts,
        scaling,
        block_size=1,
        state={},
    )
    full_output = exact_attention(query, key, value, None, scaling)
    torch.testing.assert_close(corrected, full_output, atol=2.0e-6, rtol=2.0e-6)

    shared_corrected = sparse._block_mean_value_corrected_output(
        selected_output,
        key[..., :-1, :],
        value,
        proxy_scores,
        query,
        key[..., -1, :],
        candidates,
        counts,
        scaling,
        block_size=1,
        shared_normalizer=True,
        state={},
    )
    torch.testing.assert_close(
        shared_corrected,
        full_output,
        atol=2.0e-6,
        rtol=2.0e-6,
    )

    all_candidates = (
        torch.arange(tokens - 1)
        .reshape(1, 1, -1)
        .expand(batch, kv_heads * groups, -1)
        .contiguous()
    )
    all_counts = torch.full(
        (batch, kv_heads * groups),
        tokens - 1,
        dtype=torch.long,
    )
    value_sketch_output = sparse._value_sketch_residual_output(
        full_output,
        key[..., :-1, :],
        value,
        proxy_scores,
        query,
        key[..., -1, :],
        all_candidates,
        all_counts,
        scaling,
        rank=4,
        bits=4,
        state={},
    )
    torch.testing.assert_close(
        value_sketch_output,
        full_output,
        atol=2.0e-6,
        rtol=2.0e-6,
    )

    state: dict[str, object] = {}
    first_sums, first_counts = sparse._running_block_value_sums(
        value[..., :10, :],
        state,
        block_size=4,
    )
    updated_sums, updated_counts = sparse._running_block_value_sums(
        value,
        state,
        block_size=4,
    )
    reference_sums, reference_counts = sparse._running_block_value_sums(
        value,
        {},
        block_size=4,
    )
    assert first_sums.shape[2] == 3
    assert first_counts.tolist() == [4.0, 4.0, 2.0]
    torch.testing.assert_close(updated_sums, reference_sums)
    torch.testing.assert_close(updated_counts, reference_counts)
    print("QKSIEVE_BLOCK_VALUE_VALIDATION_OK")


if __name__ == "__main__":
    main()
