import torch

from run_head_top2_targeted_ppl_20260714 import (
    _pca_int4_partial_scores,
    _pca_int4_progressive_candidates,
    _pca_int4_two_stage_candidates,
)


def test_descending_pca_basis_preserves_full_score() -> None:
    torch.manual_seed(7)
    query = torch.randn(1, 4, 64)
    key = torch.randn(1, 1, 128, 64)
    ascending = _pca_int4_partial_scores(query, key, {}, 64)
    descending = _pca_int4_partial_scores(
        query, key, {}, 64, basis_descending=True
    )
    torch.testing.assert_close(ascending, descending, atol=2.0e-5, rtol=2.0e-5)


def test_progressive_candidates_are_valid_and_compact() -> None:
    torch.manual_seed(9)
    query = torch.randn(1, 4, 64)
    key = torch.randn(1, 1, 128, 64)
    scores, indices = _pca_int4_progressive_candidates(
        query, key, {}, projection_dim=64, candidate_count=11
    )
    assert scores.shape == (1, 4, 11)
    assert indices.shape == (1, 4, 11)
    assert int(indices.min()) >= 0
    assert int(indices.max()) < 128
    for row in indices.reshape(-1, 11):
        assert torch.unique(row).numel() == 11


def test_two_stage_runtime_candidates_are_valid() -> None:
    torch.manual_seed(11)
    query = torch.randn(1, 4, 64)
    key = torch.randn(1, 1, 128, 64)
    for prefix_dim, stage_fraction in ((16, 0.30), (32, 0.20), (48, 0.08)):
        scores, indices = _pca_int4_two_stage_candidates(
            query,
            key,
            {},
            projection_dim=64,
            candidate_count=11,
            prefix_dim=prefix_dim,
            stage_fraction=stage_fraction,
        )
        assert scores.shape == (1, 4, 11)
        assert indices.shape == (1, 4, 11)
        assert int(indices.min()) >= 0
        assert int(indices.max()) < 128
