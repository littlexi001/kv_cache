import torch

import run_head_top2_targeted_ppl_20260714 as runner
from run_head_top2_targeted_ppl_20260714 import (
    _partition_global_sample_budget_ladder,
    _partition_proxy_ucb_budget_ladder,
    _partition_ucb_budget_ladder,
    _partition_ucb_progressive_exact_ladder,
    _partition_ucb_progressive_ladder,
    _sample_calibrated_candidate_counts,
    _sample_uncertainty_band_mask,
    _temporal_partition_reuse_attention,
    _topk_contribution_candidates,
    _value_norm_candidate_priority,
)


def test_inactive_attention_fallback_dispatches_by_architecture(monkeypatch):
    monkeypatch.setattr(
        runner,
        "_ORIGINAL_LLAMA_EAGER_ATTENTION_FORWARD",
        lambda *args, **kwargs: ("llama", None),
    )
    monkeypatch.setattr(
        runner,
        "_ORIGINAL_QWEN3_EAGER_ATTENTION_FORWARD",
        lambda *args, **kwargs: ("qwen3", None),
    )
    monkeypatch.setattr(
        runner,
        "_ORIGINAL_QWEN2_EAGER_ATTENTION_FORWARD",
        lambda *args, **kwargs: ("qwen2", None),
    )
    monkeypatch.setattr(
        runner,
        "_ORIGINAL_MISTRAL_EAGER_ATTENTION_FORWARD",
        lambda *args, **kwargs: ("mistral", None),
    )
    llama_type = type("LlamaAttention", (torch.nn.Module,), {})
    llama_type.__module__ = "transformers.models.llama.modeling_llama"
    qwen2_type = type("Qwen2Attention", (torch.nn.Module,), {})
    qwen2_type.__module__ = "transformers.models.qwen2.modeling_qwen2"
    qwen3_type = type("Qwen3Attention", (torch.nn.Module,), {})
    qwen3_type.__module__ = "transformers.models.qwen3.modeling_qwen3"
    mistral_type = type("MistralAttention", (torch.nn.Module,), {})
    mistral_type.__module__ = "transformers.models.mistral.modeling_mistral"
    query = torch.zeros(1, 1, 1, 2)

    llama_output, _ = runner._patched_llama_eager_attention_forward(
        llama_type(), query, query, query, None, 1.0
    )
    qwen2_output, _ = runner._patched_llama_eager_attention_forward(
        qwen2_type(), query, query, query, None, 1.0
    )
    qwen3_output, _ = runner._patched_llama_eager_attention_forward(
        qwen3_type(), query, query, query, None, 1.0
    )
    mistral_output, _ = runner._patched_llama_eager_attention_forward(
        mistral_type(), query, query, query, None, 1.0
    )

    assert llama_output == "llama"
    assert qwen2_output == "qwen2"
    assert qwen3_output == "qwen3"
    assert mistral_output == "mistral"


def test_value_norm_priority_is_incremental_and_shared_across_gqa_heads():
    state = {"capacity": 5}
    partial = torch.zeros(1, 2, 3)
    values = torch.tensor([[[[1.0, 0.0], [2.0, 0.0], [4.0, 0.0]]]])

    priority = _value_norm_candidate_priority(partial, values, state, scaling=1.0)

    expected = torch.tensor([0.0, torch.log(torch.tensor(2.0)), torch.log(torch.tensor(4.0))])
    assert torch.allclose(priority[0, 0], expected, atol=5.0e-4, rtol=0.0)
    assert torch.allclose(priority[0, 1], expected, atol=5.0e-4, rtol=0.0)
    extended_values = torch.cat((values, torch.tensor([[[[8.0, 0.0]]]])), dim=2)
    extended = _value_norm_candidate_priority(
        torch.zeros(1, 2, 4), extended_values, state, scaling=1.0
    )
    assert state["value_norm_indexed_count"] == 4
    assert torch.allclose(
        extended[0, 0, -1], torch.log(torch.tensor(8.0)), atol=5.0e-4, rtol=0.0
    )


def test_contribution_rerank_keeps_exact_scores_for_attention():
    exact_scores = torch.tensor([[[3.0, 2.0, 1.0]]])
    candidate_indices = torch.tensor([[[0, 1, 2]]])
    value_log_norms = torch.tensor([[[0.0, 0.0, 3.0]]])

    selected_scores, selected_indices = _topk_contribution_candidates(
        exact_scores, candidate_indices, value_log_norms, keep_count=2
    )

    assert selected_indices.tolist() == [[[2, 0]]]
    assert selected_scores.tolist() == [[[1.0, 3.0]]]


def test_partition_ucb_ladder_is_exact_when_proxy_is_exact():
    scores = torch.tensor([[[3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]]])
    candidate_scores, candidate_indices = torch.topk(scores, k=4, dim=-1, sorted=True)
    sample_indices = torch.arange(scores.shape[-1])
    self_scores = torch.tensor([[0.5]])
    keep_counts = (1, 2, 4)
    rung, estimated_mass, _ = _partition_ucb_budget_ladder(
        scores,
        candidate_scores,
        candidate_indices,
        scores,
        sample_indices,
        self_scores,
        keep_counts,
        target_mass=0.8,
        ucb_z=0.0,
    )

    chosen_count = keep_counts[int(rung.item())]
    exact_partition = torch.exp(scores).sum() + torch.exp(self_scores)
    selected_partition = (
        torch.exp(candidate_scores[..., :chosen_count]).sum() + torch.exp(self_scores)
    )
    exact_mass = selected_partition / exact_partition
    assert torch.allclose(estimated_mass, exact_mass, atol=1.0e-6, rtol=1.0e-6)


def test_higher_partition_confidence_never_selects_less_budget():
    torch.manual_seed(20260719)
    exact_scores = torch.randn(1, 4, 200)
    proxy_scores = exact_scores + 0.4 * torch.randn_like(exact_scores)
    _, candidate_indices = torch.topk(proxy_scores, k=40, dim=-1, sorted=True)
    candidate_scores = torch.gather(exact_scores, -1, candidate_indices)
    candidate_scores, rerank = torch.sort(candidate_scores, dim=-1, descending=True)
    candidate_indices = torch.gather(candidate_indices, -1, rerank)
    sample_indices = torch.arange(0, 200, 5)
    sample_scores = exact_scores.index_select(-1, sample_indices)
    self_scores = torch.randn(1, 4)
    keep_counts = (2, 4, 8, 16, 40)
    narrow, _, _ = _partition_ucb_budget_ladder(
        proxy_scores,
        candidate_scores,
        candidate_indices,
        sample_scores,
        sample_indices,
        self_scores,
        keep_counts,
        target_mass=0.8,
        ucb_z=0.0,
    )
    wide, _, _ = _partition_ucb_budget_ladder(
        proxy_scores,
        candidate_scores,
        candidate_indices,
        sample_scores,
        sample_indices,
        self_scores,
        keep_counts,
        target_mass=0.8,
        ucb_z=1.64,
    )
    assert torch.all(wide >= narrow)


def test_proxy_partition_confidence_never_selects_less_budget():
    torch.manual_seed(20260722)
    exact_scores = torch.randn(1, 4, 200)
    proxy_scores = exact_scores + 0.4 * torch.randn_like(exact_scores)
    _, candidate_indices = torch.topk(proxy_scores, k=40, dim=-1, sorted=True)
    sample_indices = torch.arange(0, 200, 5)
    sample_scores = exact_scores.index_select(-1, sample_indices)
    self_scores = torch.randn(1, 4)
    keep_counts = (2, 4, 8, 16, 40)
    narrow, _, _ = _partition_proxy_ucb_budget_ladder(
        proxy_scores,
        candidate_indices,
        sample_scores,
        sample_indices,
        self_scores,
        keep_counts,
        target_mass=0.8,
        ucb_z=0.0,
    )
    wide, _, _ = _partition_proxy_ucb_budget_ladder(
        proxy_scores,
        candidate_indices,
        sample_scores,
        sample_indices,
        self_scores,
        keep_counts,
        target_mass=0.8,
        ucb_z=1.64,
    )
    assert torch.all(wide >= narrow)


def test_global_sample_ladder_is_exact_when_proxy_is_exact():
    scores = torch.tensor([[[3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]]])
    _, candidate_indices = torch.topk(scores, k=4, dim=-1, sorted=True)
    sample_indices = torch.arange(scores.shape[-1])
    self_scores = torch.tensor([[0.5]])
    keep_counts = (1, 2, 4)
    rung, estimated_mass, _ = _partition_global_sample_budget_ladder(
        scores,
        candidate_indices,
        scores,
        sample_indices,
        self_scores,
        keep_counts,
        target_mass=0.8,
        ucb_z=0.0,
    )

    chosen_count = keep_counts[int(rung.item())]
    candidate_scores = torch.gather(scores, -1, candidate_indices)
    exact_partition = torch.exp(scores).sum() + torch.exp(self_scores)
    selected_partition = (
        torch.exp(candidate_scores[..., :chosen_count]).sum() + torch.exp(self_scores)
    )
    assert torch.allclose(
        estimated_mass,
        selected_partition / exact_partition,
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_progressive_partition_ladder_returns_consistent_rerank():
    torch.manual_seed(20260720)
    exact_scores = torch.randn(1, 3, 100)
    proxy_scores = exact_scores + 0.3 * torch.randn_like(exact_scores)
    _, proxy_indices = torch.topk(proxy_scores, 20, dim=-1, sorted=True)
    proxy_candidate_exact = torch.gather(exact_scores, -1, proxy_indices)
    sample_indices = torch.arange(0, 100, 4)
    keep_counts = (2, 5, 10)
    rung, _, _, selected_scores, selected_indices = _partition_ucb_progressive_ladder(
        proxy_scores,
        proxy_candidate_exact,
        proxy_indices,
        exact_scores.index_select(-1, sample_indices),
        sample_indices,
        torch.randn(1, 3),
        keep_counts,
        target_mass=0.8,
        ucb_z=1.64,
        overfetch_factor=2,
    )
    assert selected_scores.shape == (1, 3, keep_counts[-1])
    assert selected_indices.shape == selected_scores.shape
    for head in range(3):
        selected_count = keep_counts[int(rung[0, head])]
        expected_scores = exact_scores[0, head].gather(
            0, selected_indices[0, head, :selected_count]
        )
        assert torch.allclose(
            selected_scores[0, head, :selected_count], expected_scores
        )
        assert torch.isneginf(selected_scores[0, head, selected_count:]).all()


class _ReferenceCandidateScores:
    @staticmethod
    def candidate_compact_scores(query, key, indices, scaling):
        groups = query.shape[1] // key.shape[1]
        expanded_key = key.repeat_interleave(groups, dim=1)
        gathered = torch.gather(
            expanded_key,
            dim=2,
            index=indices.unsqueeze(-1).expand(-1, -1, -1, key.shape[-1]),
        )
        return torch.einsum("bhd,bhkd->bhk", query, gathered) * scaling

    @classmethod
    def candidate_compact_scores_range(
        cls, query, key, indices, start_counts, end_counts, output, scaling
    ):
        full_scores = cls.candidate_compact_scores(query, key, indices, scaling)
        positions = torch.arange(indices.shape[-1], device=indices.device).view(1, 1, -1)
        update = (positions >= start_counts.unsqueeze(-1)) & (
            positions < end_counts.unsqueeze(-1)
        )
        output.copy_(torch.where(update, full_scores, output))
        return output

    @staticmethod
    def final_attention_from_scores_ragged(
        value, indices, scores, counts, value_mass_threshold
    ):
        del value_mass_threshold
        groups = indices.shape[1] // value.shape[1]
        expanded_value = value.repeat_interleave(groups, dim=1)
        gathered = torch.gather(
            expanded_value,
            dim=2,
            index=indices.unsqueeze(-1).expand(-1, -1, -1, value.shape[-1]),
        )
        positions = torch.arange(indices.shape[-1]).view(1, 1, -1)
        valid = positions < counts.unsqueeze(-1)
        weights = torch.softmax(scores.masked_fill(~valid, -torch.inf), dim=-1)
        return torch.einsum("bhk,bhkd->bhd", weights, gathered).unsqueeze(1)


def test_temporal_partition_reuses_candidates_and_adds_new_history_token():
    torch.manual_seed(20260723)
    query = torch.ones(1, 4, 1, 8)
    key = torch.randn(1, 2, 11, 8)
    key[..., 9, :] = 100.0
    value = torch.randn_like(key)
    basis = torch.eye(8, 4).view(1, 1, 8, 4).expand(1, 2, -1, -1)
    grouped_query = query[..., 0, :].reshape(1, 2, 2, 8)
    previous_projected = torch.einsum("bhgd,bhdm->bhgm", grouped_query, basis)
    candidate_indices = torch.tensor(
        [[[0, 2, 4, 6], [1, 3, 5, 7], [0, 3, 6, 8], [1, 4, 7, 8]]]
    )
    state = {
        "basis": basis,
        "last_projected_query": previous_projected,
        "temporal_partition_candidate_indices": candidate_indices,
        "temporal_partition_exact_counts": torch.full((1, 4), 3),
        "temporal_partition_keep_counts": torch.full((1, 4), 2),
        "temporal_partition_budget_fractions": torch.full((1, 4), 0.2),
        "temporal_partition_estimated_mass": torch.full((1, 4), 0.8),
        "temporal_partition_history_count": 9,
    }
    diagnostics = {}

    result = _temporal_partition_reuse_attention(
        query,
        key,
        value,
        scaling=8**-0.5,
        history_count=10,
        max_keep_count=3,
        pca_state=state,
        cuda_kernels=_ReferenceCandidateScores,
        value_mass_threshold=1.0,
        diagnostics=diagnostics,
    )

    assert result is not None
    output, packed_indices = result
    assert output.shape == (1, 1, 4, 8)
    assert packed_indices.shape == (1, 4, 1, 4)
    assert torch.all(diagnostics["temporal_reuse_rate"] == 1)
    assert torch.allclose(
        diagnostics["selected_budget_fraction"], torch.full((1, 4), 0.2)
    )
    # Index 9 was not present at refresh time and is eligible on the reuse step.
    assert torch.any(packed_indices[..., :3] == 9)


def test_progressive_exact_ladder_matches_full_pool_simulation():
    torch.manual_seed(20260721)
    query = torch.randn(1, 4, 16)
    key = torch.randn(1, 2, 100, 16)
    scaling = 16**-0.5
    exact_scores = torch.einsum(
        "bhd,bhkd->bhk", query, key.repeat_interleave(2, dim=1)
    ) * scaling
    proxy_scores = exact_scores + 0.25 * torch.randn_like(exact_scores)
    _, candidate_indices = torch.topk(proxy_scores, 20, dim=-1, sorted=True)
    candidate_exact_scores = torch.gather(exact_scores, -1, candidate_indices)
    sample_indices = torch.arange(0, 100, 4)
    sample_scores = exact_scores.index_select(-1, sample_indices)
    self_scores = torch.randn(1, 4)
    keep_counts = (2, 5, 10)
    expected = _partition_ucb_progressive_ladder(
        proxy_scores,
        candidate_exact_scores,
        candidate_indices,
        sample_scores,
        sample_indices,
        self_scores,
        keep_counts,
        target_mass=0.8,
        ucb_z=1.64,
        overfetch_factor=2,
    )
    actual = _partition_ucb_progressive_exact_ladder(
        proxy_scores,
        candidate_indices,
        query,
        key,
        _ReferenceCandidateScores,
        scaling,
        sample_scores,
        sample_indices,
        self_scores,
        keep_counts,
        target_mass=0.8,
        ucb_z=1.64,
        overfetch_factor=2,
    )

    assert torch.equal(actual[0], expected[0])
    assert torch.allclose(actual[1], expected[1], atol=2.0e-6, rtol=2.0e-6)
    assert torch.allclose(actual[2], expected[2], atol=2.0e-6, rtol=2.0e-6)
    assert torch.equal(actual[4], expected[4])
    assert torch.allclose(actual[3], expected[3], atol=1.0e-6, rtol=1.0e-6)
    assert torch.all(actual[5] <= candidate_indices.shape[-1])
    selected_keep = torch.tensor(keep_counts)[actual[0]]
    assert torch.all(actual[5] >= 2 * selected_keep)


def test_higher_confidence_never_uses_fewer_candidates():
    torch.manual_seed(20260717)
    batch_count = 1
    query_heads = 4
    kv_heads = 2
    history_count = 1000
    head_dim = 16
    query = torch.randn(batch_count, query_heads, head_dim)
    key = torch.randn(batch_count, kv_heads, history_count, head_dim)
    groups = query_heads // kv_heads
    exact_scores = torch.einsum(
        "bhd,bhkd->bhk", query, key.repeat_interleave(groups, dim=1)
    ) * (head_dim**-0.5)
    partial_scores = exact_scores / (head_dim**-0.5)
    partial_scores = partial_scores + 0.2 * torch.randn_like(partial_scores)
    candidate_count = 80
    candidate_scores, _ = torch.topk(
        partial_scores, k=candidate_count, dim=-1, sorted=True
    )

    low_counts, low_sigma, low_buffers = _sample_calibrated_candidate_counts(
        partial_scores,
        candidate_scores,
        query,
        key,
        _ReferenceCandidateScores,
        head_dim**-0.5,
        final_count=20,
        sample_fraction=0.02,
        sample_offset=0,
        confidence_threshold=0.25,
    )
    high_counts, high_sigma, high_buffers = _sample_calibrated_candidate_counts(
        partial_scores,
        candidate_scores,
        query,
        key,
        _ReferenceCandidateScores,
        head_dim**-0.5,
        final_count=20,
        sample_fraction=0.02,
        sample_offset=0,
        confidence_threshold=1.0,
    )

    assert low_counts.shape == (batch_count, query_heads)
    assert torch.all(high_counts >= low_counts)
    assert set(low_counts.flatten().tolist()).issubset({20, 30, 40, 60, 80})
    assert torch.equal(low_sigma, high_sigma)
    assert torch.equal(low_buffers, high_buffers)
    assert torch.all(low_sigma > 0)


def test_wider_uncertainty_band_never_removes_candidates():
    torch.manual_seed(20260718)
    query = torch.randn(1, 4, 16)
    key = torch.randn(1, 2, 1000, 16)
    exact_scores = torch.einsum(
        "bhd,bhkd->bhk", query, key.repeat_interleave(2, dim=1)
    )
    partial_scores = exact_scores + 0.5 * torch.randn_like(exact_scores)
    candidate_scores, _ = torch.topk(
        partial_scores, k=80, dim=-1, sorted=False
    )
    narrow_mask, narrow_counts, narrow_sigma = _sample_uncertainty_band_mask(
        partial_scores,
        candidate_scores,
        query,
        key,
        _ReferenceCandidateScores,
        1.0,
        final_count=20,
        sample_fraction=0.02,
        sample_offset=0,
        confidence_width=0.5,
    )
    wide_mask, wide_counts, wide_sigma = _sample_uncertainty_band_mask(
        partial_scores,
        candidate_scores,
        query,
        key,
        _ReferenceCandidateScores,
        1.0,
        final_count=20,
        sample_fraction=0.02,
        sample_offset=0,
        confidence_width=1.0,
    )
    assert torch.all(wide_counts >= narrow_counts)
    assert torch.all(wide_mask | ~narrow_mask)
    assert torch.equal(narrow_sigma, wide_sigma)
    assert torch.all(narrow_counts >= 20)
