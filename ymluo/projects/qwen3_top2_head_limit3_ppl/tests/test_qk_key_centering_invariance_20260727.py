from __future__ import annotations

import torch


def test_key_centering_preserves_ranking_and_softmax() -> None:
    generator = torch.Generator().manual_seed(20260727)
    key = torch.randn(257, 128, generator=generator, dtype=torch.float64)
    query = torch.randn(7, 128, generator=generator, dtype=torch.float64)
    key_mean = key.mean(dim=0, keepdim=True)

    full_scores = query @ key.transpose(0, 1)
    centered_scores = query @ (key - key_mean).transpose(0, 1)
    removed_offset = full_scores - centered_scores

    assert torch.allclose(
        removed_offset,
        removed_offset[:, :1].expand_as(removed_offset),
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    assert torch.equal(
        torch.argsort(full_scores, dim=-1),
        torch.argsort(centered_scores, dim=-1),
    )
    assert torch.allclose(
        torch.softmax(full_scores, dim=-1),
        torch.softmax(centered_scores, dim=-1),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
