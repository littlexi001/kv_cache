import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modeling_pe import Attention, ModelConfig, apply_rope, rope_pair_scales
from train_mixed_pe import is_synthetic


def test_optimized_scale_contracts() -> None:
    config = ModelConfig()
    device = torch.device("cpu")
    native = rope_pair_scales(config, "native", 11, device, config.num_heads)
    taper_shallow = rope_pair_scales(config, "deep_highfreq_taper", 0, device, config.num_heads)
    taper_deep = rope_pair_scales(config, "deep_highfreq_taper", 11, device, config.num_heads)
    slow_shallow = rope_pair_scales(config, "layerwise_slow_rope", 0, device, config.num_heads)
    slow_deep = rope_pair_scales(config, "layerwise_slow_rope", 11, device, config.num_heads)
    complementary_q = rope_pair_scales(
        config, "complementary_smooth", 11, device, config.num_heads
    )
    complementary_k = rope_pair_scales(
        config, "complementary_smooth", 11, device, config.num_kv_heads
    )
    assert torch.allclose(native, torch.ones_like(native))
    assert taper_shallow[0] > 0.98
    assert taper_deep[0] < 0.30
    assert taper_deep[32] > 0.99
    assert slow_shallow[0] > 0.98
    assert 0.50 < slow_deep[0] < 0.52
    assert complementary_q.shape == (6, 64)
    assert complementary_k.shape == (2, 64)
    assert torch.allclose(complementary_q[:3], torch.ones_like(complementary_q[:3]))
    assert torch.allclose(complementary_k[:1], torch.ones_like(complementary_k[:1]))
    assert complementary_q[3, 0] < 0.30
    assert complementary_k[1, 0] < 0.30


def test_rope_accepts_head_specific_scales() -> None:
    config = ModelConfig()
    values = torch.randn(2, config.num_heads, 32, config.head_dim)
    positions = torch.arange(32)
    inv_freq = config.rope_theta ** (
        -torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim
    )
    scales = rope_pair_scales(
        config, "complementary_smooth", 11, torch.device("cpu"), config.num_heads
    )
    output = apply_rope(values, positions, inv_freq, scales)
    assert output.shape == values.shape
    assert torch.isfinite(output).all()


def test_mixture_fraction_is_deterministic() -> None:
    flags_a = [is_synthetic(index, 0.05, 1701) for index in range(100_000)]
    flags_b = [is_synthetic(index, 0.05, 1701) for index in range(100_000)]
    assert flags_a == flags_b
    fraction = sum(flags_a) / len(flags_a)
    assert math.isclose(fraction, 0.05, abs_tol=0.002)


def test_band8_native_matches_standard_rope() -> None:
    torch.manual_seed(7)
    config = ModelConfig()
    standard = Attention(config, 0, "native")
    reference = Attention(config, 0, "native_band8_reference")
    reference.load_state_dict(standard.state_dict())
    hidden = torch.randn(1, 16, config.hidden_size)
    standard_output = standard(hidden)
    reference_output = reference(hidden)
    assert torch.allclose(standard_output, reference_output, atol=2e-5, rtol=2e-5)


def test_fade_rope_tau_receives_gradient() -> None:
    torch.manual_seed(11)
    config = ModelConfig()
    attention = Attention(config, 0, "fade_rope_band8")
    hidden = torch.randn(1, 16, config.hidden_size, requires_grad=True)
    loss = attention(hidden).square().mean()
    loss.backward()
    assert attention.phase_log_scale is not None
    assert attention.phase_log_scale.grad is not None
    assert torch.isfinite(attention.phase_log_scale.grad).all()


if __name__ == "__main__":
    test_optimized_scale_contracts()
    test_rope_accepts_head_specific_scales()
    test_mixture_fraction_is_deterministic()
    test_band8_native_matches_standard_rope()
    test_fade_rope_tau_receives_gradient()
    print("contract tests passed")
