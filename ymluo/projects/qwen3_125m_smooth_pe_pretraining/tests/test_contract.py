import math


def estimated_parameter_count() -> int:
    vocab = 32000
    hidden = 768
    layers = 12
    heads = 6
    kv_heads = 2
    head_dim = 128
    intermediate = 3072
    embedding = vocab * hidden
    attention = hidden * heads * head_dim + 2 * hidden * kv_heads * head_dim + heads * head_dim * hidden
    mlp = 3 * hidden * intermediate
    norms = layers * (2 * hidden + heads * head_dim + kv_heads * head_dim) + hidden
    return embedding + layers * (attention + mlp) + norms


def smooth_scale(layer: int, pair: int) -> float:
    alpha_min = 0.25
    layer_gate = 1.0 / (1.0 + math.exp(-(layer - 7.5) / 1.5))
    frequency_gate = 1.0 / (1.0 + math.exp(-(7.5 - pair) / 1.5))
    return 1.0 - (1.0 - alpha_min) * layer_gate * frequency_gate


def test_parameter_budget() -> None:
    assert 115_000_000 <= estimated_parameter_count() <= 140_000_000


def test_smooth_scaling_contract() -> None:
    assert smooth_scale(0, 0) > 0.99
    assert smooth_scale(11, 0) < 0.35
    assert smooth_scale(11, 40) > 0.99
    assert smooth_scale(5, 0) > smooth_scale(11, 0)
    assert smooth_scale(11, 16) > smooth_scale(11, 0)


def test_rope_period_matches_qwen_head_dimension() -> None:
    head_dim = 128
    theta = 1_000_000.0
    period_f40 = 2 * math.pi * theta ** (2 * 40 / head_dim)
    assert 35_000 < period_f40 < 36_000

