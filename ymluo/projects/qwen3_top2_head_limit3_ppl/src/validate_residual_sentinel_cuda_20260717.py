from __future__ import annotations

import json

import torch

from run_head_top2_targeted_ppl_20260714 import (
    qabs_sampled_head_adaptive_attention,
)


def main() -> None:
    torch.manual_seed(20260717)
    device = torch.device("cuda")
    head_dim = 128
    history_count = 2048
    query = torch.randn(1, 4, 1, head_dim, dtype=torch.float16, device=device)
    key = torch.randn(
        1, 2, history_count + 1, head_dim, dtype=torch.float16, device=device
    )
    value = torch.randn_like(key)
    state: dict[str, object] = {}
    diagnostics: dict[str, object] = {}

    output, selected = qabs_sampled_head_adaptive_attention(
        query,
        key,
        value,
        attention_mask=None,
        scaling=head_dim**-0.5,
        mass_threshold=0.9,
        budget_fractions=(0.02,),
        sample_fraction=0.01,
        qabs_dim_count=8,
        candidate_fraction=0.03,
        use_cuda_kernels=True,
        diagnostics=diagnostics,
        score_mode="pca_int4_residual_sentinel",
        projection_dim=64,
        pca_state=state,
    )

    primary = state["last_primary_indices"]
    rescue = state["last_rescue_indices"]
    upper = state["last_residual_upper_scores"]
    assert isinstance(primary, torch.Tensor)
    assert isinstance(rescue, torch.Tensor)
    assert isinstance(upper, torch.Tensor)
    candidates = torch.cat((primary, rescue), dim=-1)

    group_count = query.shape[1] // key.shape[1]
    expanded_key = key[:, :, :history_count].repeat_interleave(group_count, dim=1)
    expanded_value = value.repeat_interleave(group_count, dim=1)
    exact_history_scores = torch.einsum(
        "bhd,bhkd->bhk", query[:, :, 0].float(), expanded_key.float()
    )
    candidate_scores = torch.gather(exact_history_scores, -1, candidates)
    keep_count = primary.shape[-1]
    exact_positions = torch.topk(candidate_scores, k=keep_count, dim=-1).indices
    expected_history = torch.gather(candidates, -1, exact_positions)
    actual_history = selected[:, :, 0, :keep_count]

    selected_with_self = torch.cat(
        (
            expected_history,
            torch.full(
                (*expected_history.shape[:-1], 1),
                history_count,
                dtype=torch.long,
                device=device,
            ),
        ),
        dim=-1,
    )
    expected_output = torch.empty(
        1, query.shape[1], head_dim, dtype=torch.float32, device=device
    )
    for head in range(query.shape[1]):
        indices = selected_with_self[0, head]
        selected_key = expanded_key[0, head].new_empty(
            keep_count + 1, head_dim
        )
        selected_key[:-1] = expanded_key[0, head].index_select(
            0, expected_history[0, head]
        )
        selected_key[-1] = key[0, head // group_count, history_count]
        selected_value = expanded_value[0, head].index_select(0, indices)
        scores = (
            selected_key.float() * query[0, head, 0].float().unsqueeze(0)
        ).sum(dim=-1) * (head_dim**-0.5)
        expected_output[0, head] = torch.matmul(
            torch.softmax(scores, dim=-1), selected_value.float()
        )

    primary_membership = (
        candidates.unsqueeze(-1) == primary.unsqueeze(-2)
    ).any(dim=-2)
    upper_violation = (exact_history_scores - upper).clamp_min(0.0)
    payload = {
        "candidate_count": int(candidates.shape[-1]),
        "primary_count": int(primary.shape[-1]),
        "rescue_count": int(rescue.shape[-1]),
        "primary_candidates_preserved": bool(primary_membership.all()),
        "exact_rerank_index_match": bool(
            torch.equal(
                torch.sort(actual_history, dim=-1).values,
                torch.sort(expected_history, dim=-1).values,
            )
        ),
        "upper_bound_violation_count": int((upper_violation > 2.0e-3).sum()),
        "upper_bound_max_violation": float(upper_violation.max()),
        "attention_output_max_abs_error": float(
            (output[:, 0].float() - expected_output).abs().max()
        ),
        "attention_output_mean_abs_error": float(
            (output[:, 0].float() - expected_output).abs().mean()
        ),
        "radius_dtype": str(state["error_radius_codes"].dtype),
        "diagnostic_candidate_fraction": float(diagnostics["candidate_fraction"]),
        "diagnostic_selected_fraction": float(
            diagnostics["selected_history_fraction"].float().mean()
        ),
    }
    print(json.dumps(payload, sort_keys=True))

    if not payload["primary_candidates_preserved"]:
        raise RuntimeError("primary candidates were dropped")
    if not payload["exact_rerank_index_match"]:
        raise RuntimeError("compact exact rerank disagrees with PyTorch")
    if payload["upper_bound_violation_count"]:
        raise RuntimeError("residual upper bound was violated")
    if payload["attention_output_max_abs_error"] > 2.0e-3:
        raise RuntimeError("sparse attention output disagrees with PyTorch")


if __name__ == "__main__":
    main()
