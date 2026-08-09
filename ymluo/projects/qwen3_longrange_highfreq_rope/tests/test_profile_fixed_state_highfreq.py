import importlib.util
from pathlib import Path

import torch


PATH = Path(__file__).parents[1] / "src" / "profile_fixed_state_highfreq.py"
SPEC = importlib.util.spec_from_file_location("profile_fixed_state_highfreq", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_subsequence_starts_finds_overlapping_matches() -> None:
    assert MODULE.subsequence_starts([1, 2, 1, 2, 1], [1, 2, 1]) == [0, 2]


def test_rope_inverse_reconstructs_input() -> None:
    torch.manual_seed(0)
    value = torch.randn(2, 7, 8)
    positions = torch.arange(7)
    inv_freq = torch.tensor([1.0, 0.3, 0.1, 0.03])
    scaling = torch.tensor(1.2)
    pre = MODULE.invert_rope(value, positions, inv_freq, scaling)
    reconstructed = MODULE.apply_rope(pre, positions, inv_freq, scaling)
    assert torch.allclose(reconstructed, value, atol=2e-6, rtol=2e-6)


def test_mass_and_rank() -> None:
    logits = torch.tensor([0.0, 3.0, 2.0, 1.0])
    mass, rank = MODULE.mass_and_rank(logits, torch.tensor([2, 3]))
    assert rank == 2
    assert 0.0 < mass < 1.0


def test_counterfactual_nope_matches_explicit_zero_relative_phase() -> None:
    torch.manual_seed(3)
    head_dim = 8
    half = head_dim // 2
    query_pre = torch.randn(head_dim)
    keys_pre = torch.randn(6, head_dim)
    inv_freq = torch.tensor([1.0, 0.3, 0.1, 0.03])
    query_position = 5
    query_post = MODULE.apply_rope(
        query_pre[None, None], torch.tensor([query_position]), inv_freq, torch.tensor(1.0)
    )[0, 0]
    key_post = MODULE.apply_rope(
        keys_pre, torch.arange(6), inv_freq, torch.tensor(1.0)
    )
    native, counterfactual = MODULE.counterfactual_high_nope_logits(
        query_post, key_post, query_position, inv_freq, high_end=2, attention_scale=1.0
    )
    selected = list(range(2)) + list(range(half, half + 2))
    explicit = native.clone()
    explicit -= torch.matmul(key_post[:, selected], query_post[selected])
    explicit += torch.matmul(keys_pre[:, selected], query_pre[selected])
    assert torch.allclose(counterfactual, explicit, atol=2e-5, rtol=2e-5)
