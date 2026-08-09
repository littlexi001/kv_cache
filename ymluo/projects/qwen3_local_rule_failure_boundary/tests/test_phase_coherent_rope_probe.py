from __future__ import annotations

import math
import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import run_phase_coherent_rope_probe_8b as target  # noqa: E402
import run_rope_retrieval_repair_8b as rope_repair  # noqa: E402


def _rotate(
    values: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    attention_scale: float,
) -> torch.Tensor:
    phase = positions.float().unsqueeze(-1) * inv_freq.float().unsqueeze(0)
    embedding = torch.cat((phase, phase), dim=-1).to(values.dtype)
    cos = embedding.cos().view(1, 1, positions.numel(), values.shape[-1])
    sin = embedding.sin().view(1, 1, positions.numel(), values.shape[-1])
    return attention_scale * (
        values * cos + rope_repair.rotate_half(values) * sin
    )


def test_relative_rotary_scores_reconstruct_standard_rope() -> None:
    """The identity phase map must exactly reconstruct ordinary relative RoPE."""

    torch.manual_seed(20260801)
    heads, keys, head_dim = 3, 7, 8
    query_position = 41
    key_positions = torch.tensor([0, 2, 9, 17, 28, 39, 41])
    delta = query_position - key_positions
    inv_freq = torch.tensor([1.0, 0.27, 0.051, 0.006])
    attention_scale = 1.13
    score_scale = 1.0 / math.sqrt(head_dim)
    query_pre = torch.randn(1, heads, 1, head_dim, dtype=torch.float64)
    key_pre = torch.randn(1, heads, keys, head_dim, dtype=torch.float64)

    query_post = _rotate(
        query_pre,
        torch.tensor([query_position]),
        inv_freq,
        attention_scale,
    )
    key_post = _rotate(
        key_pre,
        key_positions,
        inv_freq,
        attention_scale,
    )
    expected = torch.matmul(query_post, key_post.transpose(2, 3)) * score_scale
    actual = target.relative_rotary_scores(
        query_pre,
        key_pre,
        delta,
        inv_freq,
        rope_repair.rotate_half,
        attention_scale,
        score_scale,
        lambda phase: phase,
    )

    # Qwen stores ``inv_freq`` in float32, so subtracting two large absolute
    # phases is expected to differ from forming their relative phase directly
    # by roughly one float32 ULP.
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_phase_coherent_is_standard_at_zero_distance() -> None:
    torch.manual_seed(3)
    query = torch.randn(1, 2, 1, 8, dtype=torch.float64)
    key = torch.randn(1, 2, 1, 8, dtype=torch.float64)
    inv_freq = torch.tensor([1.0, 0.3, 0.07, 0.01])
    scale = 1.19
    score_scale = 1.0 / math.sqrt(8)

    actual = target.phase_coherent_scores(
        query,
        key,
        torch.zeros(1, dtype=torch.long),
        inv_freq,
        rope_repair.rotate_half,
        scale,
        score_scale,
        cutoff=2.0,
    )
    expected = torch.matmul(query, key.transpose(2, 3)) * score_scale * scale**2
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_phase_coherent_converges_to_nope_remotely() -> None:
    torch.manual_seed(4)
    query = torch.randn(1, 2, 1, 8, dtype=torch.float64)
    key = torch.randn(1, 2, 1, 8, dtype=torch.float64)
    inv_freq = torch.tensor([1.0, 0.3, 0.07, 0.01])
    score_scale = 1.0 / math.sqrt(8)

    actual = target.phase_coherent_scores(
        query,
        key,
        torch.tensor([1_000_000]),
        inv_freq,
        rope_repair.rotate_half,
        attention_scale=1.0,
        score_scale=score_scale,
        cutoff=2.0,
    )
    expected = torch.matmul(query, key.transpose(2, 3)) * score_scale
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_phase_coherent_releases_high_frequencies_first() -> None:
    delta = torch.tensor(128.0)
    inv_freq = torch.tensor([1.0, 0.1, 0.01, 0.001])
    kappa = torch.exp(-torch.square(delta * inv_freq / 2.0))

    assert torch.all(kappa[:-1] < kappa[1:])
    assert float(kappa[0]) < 1e-8
    assert float(kappa[-1]) > 0.99


def test_phase_coherent_onset_preserves_standard_rope() -> None:
    torch.manual_seed(5)
    query = torch.randn(1, 2, 1, 8, dtype=torch.float64)
    key = torch.randn(1, 2, 3, 8, dtype=torch.float64)
    delta = torch.tensor([17, 513, 4096])
    inv_freq = torch.tensor([1.0, 0.3, 0.07, 0.01])
    score_scale = 1.0 / math.sqrt(8)

    expected = target.relative_rotary_scores(
        query,
        key,
        delta,
        inv_freq,
        rope_repair.rotate_half,
        attention_scale=1.0,
        score_scale=score_scale,
        phase_map=lambda phase: phase,
    )
    actual = target.phase_coherent_scores(
        query,
        key,
        delta,
        inv_freq,
        rope_repair.rotate_half,
        attention_scale=1.0,
        score_scale=score_scale,
        cutoff=4.0,
        onset=4096.0,
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_phase_return_is_local_rope_and_remote_nope() -> None:
    cutoff = 2.0
    phase = torch.tensor([0.0, 0.01, 1.0, 1000.0])
    mapped = phase / (1.0 + torch.square(torch.abs(phase) / cutoff))

    assert mapped[0].item() == 0.0
    assert abs(float(mapped[1] - phase[1])) < 1e-6
    assert abs(float(mapped[-1])) < 0.005


def test_minimal_phase_rescue_is_exact_noop_without_trigger() -> None:
    query = torch.tensor([[[[1.0, 0.0]]]])
    key = torch.tensor([[[[1.0, 0.0]]]])
    post = torch.tensor([[0.37]])
    corrected, stats = target.minimal_phase_rescue_scores(
        query,
        key,
        torch.tensor([[3]]),
        post,
        torch.tensor([[False]]),
        torch.tensor([1.0]),
        attention_scale=1.0,
        score_scale=1.0,
        boundary=0.0,
        lift_fraction=0.5,
    )

    torch.testing.assert_close(corrected, post, atol=0.0, rtol=0.0)
    assert not bool(stats["trigger"].item())


def test_minimal_phase_rescue_lifts_counterfactually_suppressed_pair() -> None:
    query = torch.tensor([[[[1.0, 0.0]]]])
    key = torch.tensor([[[[1.0, 0.0]]]])
    native = torch.tensor([[math.cos(3.0)]])
    corrected, stats = target.minimal_phase_rescue_scores(
        query,
        key,
        torch.tensor([[3]]),
        native,
        torch.tensor([[True]]),
        torch.tensor([1.0]),
        attention_scale=1.0,
        score_scale=1.0,
        boundary=0.0,
        lift_fraction=0.5,
    )

    assert bool(stats["trigger"].item())
    assert float(corrected.item()) > float(native.item())
    assert float(stats["exact_lift"].item()) > 0.0
    assert float(stats["phase_shift_rms"].item()) <= 0.25


def test_sparse_minimal_phase_rescue_obeys_frequency_budget() -> None:
    torch.manual_seed(9)
    query = torch.randn(1, 1, 1, 16)
    key = query.clone()
    inv_freq = torch.tensor([1.0, 0.7, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01])
    delta = torch.tensor([[31]])
    native = target.relative_rotary_scores(
        query,
        key,
        delta[0],
        inv_freq,
        rope_repair.rotate_half,
        attention_scale=1.0,
        score_scale=1.0,
        phase_map=lambda phase: phase,
    )[0, :, 0, :]
    _, stats = target.minimal_phase_rescue_scores(
        query,
        key,
        delta,
        native,
        torch.tensor([[True]]),
        inv_freq,
        attention_scale=1.0,
        score_scale=1.0,
        boundary=0.0,
        lift_fraction=0.25,
        minimum_counterfactual_gap=-1e9,
        frequency_budget=3,
    )

    assert int(stats["active_plane_count"].item()) <= 3
