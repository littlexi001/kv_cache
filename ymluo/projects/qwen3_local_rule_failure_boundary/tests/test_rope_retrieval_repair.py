from __future__ import annotations

import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import run_rope_retrieval_repair_8b as target  # noqa: E402


def test_invert_rope_recovers_split_half_vectors() -> None:
    torch.manual_seed(0)
    pre = torch.randn(1, 2, 7, 8)
    positions = torch.arange(7)
    inv_freq = torch.tensor([1.0, 0.3, 0.07, 0.01])
    cos, sin = target.rope_angles(positions, inv_freq, 8, pre.dtype)
    scale = 1.17
    post = scale * (pre * cos.view(1, 1, 7, 8) + target.rotate_half(pre) * sin.view(1, 1, 7, 8))
    recovered = target.invert_rope(post, positions, inv_freq, scale)
    torch.testing.assert_close(recovered, pre, atol=1e-5, rtol=1e-5)


def test_rope_delta_matches_direct_virtual_rotation() -> None:
    torch.manual_seed(1)
    pre = torch.randn(1, 2, 5, 8)
    old = torch.tensor([[2, 8, 11, 17, 20], [1, 5, 9, 18, 20]])
    new = torch.tensor([[16, 17, 18, 19, 20], [16, 17, 18, 19, 20]])
    inv_freq = torch.tensor([1.0, 0.3, 0.07, 0.01])

    def rotate(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        frequencies = positions.float().unsqueeze(-1) * inv_freq.view(1, 1, -1)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return values * embedding.cos() + target.rotate_half(values) * embedding.sin()

    old_post = rotate(pre, old)
    expected = rotate(pre, new)
    actual = target.apply_rope_delta(old_post, old, new, inv_freq)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_force_current_topk_preserves_budget_and_self() -> None:
    scores = torch.tensor([[9.0, 1.0, 8.0, 2.0, -5.0], [0.0, 4.0, 3.0, 2.0, -9.0]])
    selected = target.force_current_topk(scores, 3)
    assert selected.tolist() == [[0, 2, 4], [1, 2, 4]]


def test_virtual_positions_preserve_order_and_current() -> None:
    selected = torch.tensor([[2, 7, 11, 20], [1, 3, 19, 20]])
    virtual = target.virtual_positions(selected, 20)
    assert virtual.tolist() == [[17, 18, 19, 20], [17, 18, 19, 20]]


def test_envelope_is_invariant_to_relative_phase() -> None:
    query = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    key = torch.tensor([[[[4.0, 3.0, 2.0, 1.0]]]])
    base_score = target.envelope_scores(query, key)
    angle = 0.73
    emb = torch.tensor([angle, angle, angle, angle])
    rotated_key = key * emb.cos() + target.rotate_half(key) * emb.sin()
    rotated_score = target.envelope_scores(query, rotated_key)
    torch.testing.assert_close(rotated_score, base_score, atol=1e-6, rtol=1e-6)


def test_generation_answer_uses_last_valid_code_mention() -> None:
    result = target.extract_generation_answer(
        "Let us follow river to window, therefore the final answer is basket.",
        ["river", "window", "basket", "train"],
    )
    assert result["generation_mentions"] == ["river", "window", "basket"]
    assert result["generation_final_answer"] == "basket"
