from __future__ import annotations

import json

import torch

import qabs_cuda_kernels

from run_head_top2_targeted_ppl_20260714 import (
    _pca_int4_candidate_range_scores,
    _pca_int4_partial_scores,
    _pca_int4_progressive_candidates,
)


def unpack_projected_key(
    state: dict[str, object], history_count: int
) -> torch.Tensor:
    packed = state["packed"][..., :history_count, :]
    low = (packed & 0x0F).float() - 7
    high = (packed >> 4).float() - 7
    codes = torch.stack((low, high), dim=-1).flatten(-2)
    return codes * state["scales"][..., :history_count, :].float()


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(20260717)
    device = torch.device("cuda")
    batch_count = 1
    kv_heads = 2
    groups = 4
    query_heads = kv_heads * groups
    history_count = 4096
    head_dim = 64
    query = torch.randn(
        batch_count, query_heads, head_dim, device=device, dtype=torch.float16
    )
    key = torch.randn(
        batch_count,
        kv_heads,
        history_count + 1,
        head_dim,
        device=device,
        dtype=torch.float16,
    )
    state: dict[str, object] = {}
    prefix_scores = _pca_int4_partial_scores(
        query,
        key[..., :history_count, :],
        state,
        64,
        basis_descending=True,
        score_prefix_dim=16,
    )
    projected_key = unpack_projected_key(state, history_count)
    query_codes = state["last_projected_query_codes"].float()
    query_scale = state["last_projected_query_scale"].float()
    quantized_query = query_codes * query_scale
    prefix_reference = torch.einsum(
        "bhgm,bhkm->bhgk",
        quantized_query[..., :16],
        projected_key[..., :16],
    ).reshape(batch_count, query_heads, history_count)

    candidate_indices = torch.randint(
        0,
        history_count,
        (batch_count, query_heads, 257),
        device=device,
    )
    range_scores = _pca_int4_candidate_range_scores(
        state, candidate_indices, history_count, 16, 32
    )
    grouped_indices = candidate_indices.reshape(
        batch_count, kv_heads, groups, 257
    )
    candidate_key = torch.gather(
        projected_key[..., 16:32].unsqueeze(2).expand(-1, -1, groups, -1, -1),
        dim=3,
        index=grouped_indices.unsqueeze(-1).expand(-1, -1, -1, -1, 16),
    )
    range_reference = torch.einsum(
        "bhgcm,bhgm->bhgc", candidate_key, quantized_query[..., 16:32]
    ).reshape(batch_count, query_heads, 257)

    cascade_state: dict[str, object] = {
        "cascade_stage1_fraction": 0.30,
        "cascade_stage2_fraction": 0.12,
    }
    cascade_scores, cascade_indices = _pca_int4_progressive_candidates(
        query,
        key[..., :history_count, :],
        cascade_state,
        projection_dim=64,
        candidate_count=328,
    )
    scaling = head_dim**-0.5
    exact_candidates = qabs_cuda_kernels.candidate_compact_scores(
        query,
        key[..., :history_count, :],
        cascade_indices,
        scaling,
    )
    selected_count = 82
    selected_scores, selected_local = torch.topk(
        exact_candidates, k=selected_count, dim=-1, sorted=True
    )
    selected_indices = torch.gather(
        cascade_indices, dim=-1, index=selected_local
    )
    packed_indices = torch.cat(
        (
            selected_indices,
            torch.full(
                (batch_count, query_heads, 1),
                history_count,
                dtype=torch.long,
                device=device,
            ),
        ),
        dim=-1,
    )
    selected_counts = torch.full(
        (batch_count, query_heads),
        selected_count + 1,
        dtype=torch.long,
        device=device,
    )
    current_key = key[..., -1, :].repeat_interleave(groups, dim=1)
    self_scores = (
        query.float() * current_key.float()
    ).sum(dim=-1, keepdim=True) * scaling
    packed_scores = torch.cat((selected_scores, self_scores), dim=-1)
    value = torch.randn_like(key)
    old_attention = qabs_cuda_kernels.final_attention_ragged(
        query,
        key,
        value,
        packed_indices,
        selected_counts,
        scaling,
    )
    reused_attention = qabs_cuda_kernels.final_attention_from_scores_ragged(
        value,
        packed_indices,
        packed_scores,
        selected_counts,
    )
    payload = {
        "prefix_max_abs_error": float(
            (prefix_scores - prefix_reference).abs().max().item()
        ),
        "prefix_mean_abs_error": float(
            (prefix_scores - prefix_reference).abs().mean().item()
        ),
        "range_max_abs_error": float(
            (range_scores - range_reference).abs().max().item()
        ),
        "range_mean_abs_error": float(
            (range_scores - range_reference).abs().mean().item()
        ),
        "cascade_shape": list(cascade_scores.shape),
        "cascade_unique_per_row": bool(
            all(
                torch.unique(row).numel() == row.numel()
                for row in cascade_indices.reshape(-1, cascade_indices.shape[-1])
            )
        ),
        "cascade_counts": list(cascade_state["last_cascade_counts"]),
        "reused_attention_max_abs_error": float(
            (old_attention - reused_attention).abs().max().item()
        ),
        "reused_attention_mean_abs_error": float(
            (old_attention - reused_attention).abs().mean().item()
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["prefix_max_abs_error"] > 2.0e-3:
        raise RuntimeError("prefix CUDA kernel does not match the reference")
    if payload["range_max_abs_error"] > 2.0e-3:
        raise RuntimeError("candidate-range CUDA kernel does not match the reference")
    if not payload["cascade_unique_per_row"]:
        raise RuntimeError("cascade returned duplicate candidates")
    if payload["reused_attention_max_abs_error"] > 5.0e-4:
        raise RuntimeError("score-reuse attention does not match the reference")


if __name__ == "__main__":
    main()
