import importlib.util
from pathlib import Path

import torch


PATH = Path(__file__).parents[1] / "src" / "head_frequency_intervention.py"
SPEC = importlib.util.spec_from_file_location("head_frequency_intervention", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_split_half_frequency_dimensions() -> None:
    assert MODULE.frequency_dimensions([0, 7], 128) == (0, 7, 64, 71)


def test_normalize_spec_merges_atoms() -> None:
    spec = {
        "atoms": [
            {"layers": [30], "head_groups": [2], "frequency_pairs": [4, 5]},
            {"layers": [30], "head_groups": [2], "frequency_pairs": [5, 6]},
        ]
    }
    assert MODULE.normalize_spec(spec, 36, 8, 128) == {
        30: {2: {4: 0.0, 5: 0.0, 6: 0.0}}
    }


def test_normalize_spec_supports_continuous_frequency_scale() -> None:
    spec = {
        "atoms": [{
            "layers": [25],
            "head_groups": [3],
            "frequency_pairs": [46],
            "frequency_scale": 0.5,
        }]
    }
    assert MODULE.normalize_spec(spec, 36, 8, 128) == {25: {3: {46: 0.5}}}


def test_native_spec_is_empty() -> None:
    assert MODULE.normalize_spec({"name": "native", "atoms": []}, 36, 8, 128) == {}


def test_normalize_piecewise_position_warp_start() -> None:
    spec = {
        "atoms": [{
            "layers": [25],
            "head_groups": [3],
            "frequency_pairs": [46],
            "frequency_scale": 0.5,
            "position_warp_start": 16384,
        }]
    }
    assert MODULE.normalize_spec(spec, 36, 8, 128) == {25: {3: {46: 0.5}}}
    assert MODULE.normalize_warp_starts(spec, 36, 8, 128) == {
        25: {3: {46: 16384.0}}
    }


def test_normalize_relative_distance_warp_is_not_an_absolute_warp() -> None:
    spec = {
        "atoms": [{
            "layers": [18, 19],
            "head_groups": [4],
            "frequency_pairs": [47],
            "frequency_scale": 0.25,
            "position_warp_start": 16384,
            "warp_mode": "relative_distance",
        }]
    }
    assert MODULE.normalize_spec(spec, 36, 8, 128) == {}
    assert MODULE.normalize_warp_starts(spec, 36, 8, 128) == {}
    assert MODULE.normalize_relative_warps(spec, 36, 8, 128) == {
        18: {4: {47: (0.25, 16384.0, 1.0)}},
        19: {4: {47: (0.25, 16384.0, 1.0)}},
    }


def test_normalize_remote_concentration_gate() -> None:
    spec = {
        "atoms": [{
            "layers": [25],
            "head_groups": [3],
            "frequency_pairs": [46],
            "frequency_scale": 0.25,
            "position_warp_start": 8192,
            "warp_mode": "relative_distance",
            "adaptive_gate": "remote_concentration",
            "adaptive_remote_mass_scale": 0.2,
            "adaptive_topk": 8,
            "adaptive_topk_mass_scale": 0.6,
        }]
    }
    assert MODULE.normalize_relative_gates(spec, 36, 8, 128) == {
        25: {3: {46: {
            "mode": "remote_concentration",
            "remote_mass_scale": 0.2,
            "topk": 8,
            "topk_mass_scale": 0.6,
        }}}
    }
    assert MODULE.normalize_relative_warps(spec, 36, 8, 128) == {
        25: {3: {46: (0.25, 8192.0, 1.0)}},
    }


def test_normalize_semantic_topk_gate() -> None:
    spec = {
        "atoms": [{
            "layers": [25],
            "head_groups": [3],
            "frequency_pairs": [46],
            "frequency_scale": 0.25,
            "position_warp_start": 8192,
            "warp_mode": "relative_distance",
            "adaptive_gate": "semantic_topk",
            "adaptive_topk_fraction": 0.02,
            "adaptive_minimum_topk": 2,
        }]
    }
    assert MODULE.normalize_relative_gates(spec, 36, 8, 128) == {
        25: {3: {46: {
            "mode": "semantic_topk",
            "topk_fraction": 0.02,
            "minimum_topk": 2,
            "replace_full_score": 0.0,
            "semantic_score_blend": 1.0,
        }}}
    }


def test_normalize_semantic_full_score_replacement() -> None:
    spec = {
        "atoms": [{
            "layers": [25],
            "head_groups": [3],
            "frequency_pairs": [46],
            "frequency_scale": 0.25,
            "position_warp_start": 8192,
            "warp_mode": "relative_distance",
            "adaptive_gate": "semantic_topk",
            "adaptive_topk_fraction": 0.01,
            "adaptive_replace_full_score": True,
            "adaptive_semantic_score_blend": 0.25,
        }]
    }
    config = MODULE.normalize_relative_gates(spec, 36, 8, 128)[25][3][46]
    assert config["replace_full_score"] == 1.0
    assert config["semantic_score_blend"] == 0.25


def rotate_half(values: torch.Tensor) -> torch.Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def test_semantic_topk_gate_selects_pre_rope_content_match() -> None:
    controller = object.__new__(MODULE.HeadFrequencyIntervention)
    controller.head_dim = 4
    controller.rope_theta = 10000.0
    controller.rotary_embedding = type(
        "FakeRotary",
        (),
        {"inv_freq": torch.tensor([1.0, 0.2]), "attention_scaling": 1.0},
    )()
    query_position = torch.tensor([10])
    query_pre = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
    key_pre = torch.zeros((1, 1, 11, 4))
    key_pre[0, 0, 3, 0] = 5.0
    key_pre[0, 0, 0, 0] = 2.0
    key_pre[0, 0, 8, 0] = 20.0  # Strongest semantic key, but not remote.

    def rotate_at(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        angles = positions.float()[:, None] * torch.tensor([1.0, 0.2])[None, :]
        doubled = torch.cat((angles, angles), dim=-1)
        cosine = doubled.cos()[None, None]
        sine = doubled.sin()[None, None]
        return values * cosine + rotate_half(values) * sine

    query_post = rotate_at(query_pre, query_position)
    key_post = rotate_at(key_pre, torch.arange(11))
    gate = controller._semantic_topk_gate(
        query_post,
        key_post,
        query_position,
        2.0,
        {"mode": "semantic_topk", "topk_fraction": 0.1, "minimum_topk": 1},
        None,
        0,
        1,
    )
    selected = torch.nonzero(gate[0, 0, 0], as_tuple=False).flatten().tolist()
    assert selected == [3]


def test_zero_scale_keeps_content_on_selected_split_half_pair() -> None:
    controller = object.__new__(MODULE.HeadFrequencyIntervention)
    controller.head_dim = 4
    controller.num_query_heads = 4
    controller.num_kv_heads = 1
    controller.query_heads_per_group = 4
    controller.rope_theta = 10000.0
    controller.active = {0: {0: {1: 0.0}}}
    controller._device_scales = {}
    q = torch.arange(1, 1 + 1 * 4 * 2 * 4, dtype=torch.float32).reshape(1, 4, 2, 4)
    k = torch.arange(1, 1 + 1 * 1 * 2 * 4, dtype=torch.float32).reshape(1, 1, 2, 4)
    positions = torch.tensor([0, 3])
    inv = 10000.0 ** (-torch.arange(0, 4, 2, dtype=torch.float32) / 4)
    angles = positions[:, None].float() * inv[None, :]
    doubled = torch.cat((angles, angles), dim=-1)
    cos, sin = doubled.cos()[None, None], doubled.sin()[None, None]
    native_q = q * cos + rotate_half(q) * sin
    native_k = k * cos + rotate_half(k) * sin
    fake_qwen3 = type("Fake", (), {"rotate_half": staticmethod(rotate_half)})
    actual_q, actual_k = controller._scaled_rotation(
        fake_qwen3, q, k, native_q, native_k, 0, positions
    )
    assert torch.equal(actual_q[..., [1, 3]], q[..., [1, 3]])
    assert torch.equal(actual_k[..., [1, 3]], k[..., [1, 3]])
    assert torch.equal(actual_q[..., [0, 2]], native_q[..., [0, 2]])
    assert torch.equal(actual_k[..., [0, 2]], native_k[..., [0, 2]])


def test_piecewise_warp_preserves_prefix_and_compresses_suffix() -> None:
    controller = object.__new__(MODULE.HeadFrequencyIntervention)
    controller.head_dim = 4
    controller.num_query_heads = 4
    controller.num_kv_heads = 1
    controller.query_heads_per_group = 4
    controller.rope_theta = 10000.0
    controller.active = {0: {0: {1: 0.5}}}
    controller.active_warp_starts = {0: {0: {1: 10.0}}}
    controller._device_scales = {}
    q = torch.arange(1, 1 + 1 * 4 * 2 * 4, dtype=torch.float32).reshape(1, 4, 2, 4)
    k = torch.arange(1, 1 + 1 * 1 * 2 * 4, dtype=torch.float32).reshape(1, 1, 2, 4)
    positions = torch.tensor([5, 14])
    inv = 10000.0 ** (-torch.arange(0, 4, 2, dtype=torch.float32) / 4)
    native_angles = positions[:, None].float() * inv[None, :]
    native_doubled = torch.cat((native_angles, native_angles), dim=-1)
    native_cos, native_sin = native_doubled.cos()[None, None], native_doubled.sin()[None, None]
    native_q = q * native_cos + rotate_half(q) * native_sin
    native_k = k * native_cos + rotate_half(k) * native_sin
    fake_qwen3 = type("Fake", (), {"rotate_half": staticmethod(rotate_half)})
    actual_q, actual_k = controller._scaled_rotation(
        fake_qwen3, q, k, native_q, native_k, 0, positions
    )
    warped_positions = torch.tensor([5.0, 12.0])
    warped_angles = warped_positions[:, None] * inv[None, :]
    warped_doubled = torch.cat((warped_angles, warped_angles), dim=-1)
    warped_cos, warped_sin = warped_doubled.cos()[None, None], warped_doubled.sin()[None, None]
    expected_q = q * warped_cos + rotate_half(q) * warped_sin
    expected_k = k * warped_cos + rotate_half(k) * warped_sin
    assert torch.equal(actual_q[..., [1, 3]], expected_q[..., [1, 3]])
    assert torch.equal(actual_k[..., [1, 3]], expected_k[..., [1, 3]])
    assert torch.equal(actual_q[:, :, 0, [1, 3]], native_q[:, :, 0, [1, 3]])
    assert torch.equal(actual_k[:, :, 0, [1, 3]], native_k[:, :, 0, [1, 3]])


def test_piecewise_prefix_reuses_native_tensor_without_recomputation() -> None:
    controller = object.__new__(MODULE.HeadFrequencyIntervention)
    controller.head_dim = 4
    controller.num_query_heads = 4
    controller.num_kv_heads = 1
    controller.query_heads_per_group = 4
    controller.rope_theta = 10000.0
    controller.active = {0: {0: {1: 0.25}}}
    controller.active_warp_starts = {0: {0: {1: 10.0}}}
    controller._device_scales = {}
    q = torch.ones((1, 4, 2, 4), dtype=torch.float32)
    k = torch.ones((1, 1, 2, 4), dtype=torch.float32)
    native_q = torch.full_like(q, 7.0)
    native_k = torch.full_like(k, 11.0)
    fake_qwen3 = type("Fake", (), {"rotate_half": staticmethod(rotate_half)})
    actual_q, actual_k = controller._scaled_rotation(
        fake_qwen3,
        q,
        k,
        native_q,
        native_k,
        0,
        torch.tensor([5, 14]),
    )
    assert torch.equal(actual_q[:, :, 0], native_q[:, :, 0])
    assert torch.equal(actual_k[:, :, 0], native_k[:, :, 0])
    assert not torch.equal(actual_q[:, :, 1, [1, 3]], native_q[:, :, 1, [1, 3]])
    assert not torch.equal(actual_k[:, :, 1, [1, 3]], native_k[:, :, 1, [1, 3]])


def test_piecewise_warp_uses_live_extended_rope_parameters() -> None:
    controller = object.__new__(MODULE.HeadFrequencyIntervention)
    controller.head_dim = 4
    controller.num_query_heads = 4
    controller.num_kv_heads = 1
    controller.query_heads_per_group = 4
    controller.rope_theta = 10000.0
    controller.rotary_embedding = type(
        "FakeRotary",
        (),
        {
            "inv_freq": torch.tensor([1.0, 0.2]),
            "attention_scaling": 1.5,
        },
    )()
    controller.active = {0: {0: {1: 0.5}}}
    controller.active_warp_starts = {0: {0: {1: 10.0}}}
    controller._device_scales = {}
    q = torch.arange(1, 1 + 1 * 4 * 1 * 4, dtype=torch.float32).reshape(1, 4, 1, 4)
    k = torch.arange(1, 1 + 1 * 1 * 1 * 4, dtype=torch.float32).reshape(1, 1, 1, 4)
    native_q = torch.zeros_like(q)
    native_k = torch.zeros_like(k)
    fake_qwen3 = type("Fake", (), {"rotate_half": staticmethod(rotate_half)})

    actual_q, actual_k = controller._scaled_rotation(
        fake_qwen3, q, k, native_q, native_k, 0, torch.tensor([14])
    )

    warped_position = 12.0
    angle = warped_position * 0.2
    expected_cos = torch.cos(torch.tensor(angle)) * 1.5
    expected_sin = torch.sin(torch.tensor(angle)) * 1.5
    expected_q_x = q[..., 1] * expected_cos - q[..., 3] * expected_sin
    expected_q_y = q[..., 3] * expected_cos + q[..., 1] * expected_sin
    expected_k_x = k[..., 1] * expected_cos - k[..., 3] * expected_sin
    expected_k_y = k[..., 3] * expected_cos + k[..., 1] * expected_sin
    assert torch.allclose(actual_q[..., 1], expected_q_x)
    assert torch.allclose(actual_q[..., 3], expected_q_y)
    assert torch.allclose(actual_k[..., 1], expected_k_x)
    assert torch.allclose(actual_k[..., 3], expected_k_y)


def test_relative_score_correction_matches_direct_relative_rotation() -> None:
    controller = object.__new__(MODULE.HeadFrequencyIntervention)
    controller.head_dim = 4
    controller.rope_theta = 10000.0
    controller.rotary_embedding = type(
        "FakeRotary",
        (),
        {"inv_freq": torch.tensor([1.0, 0.2]), "attention_scaling": 1.0},
    )()
    query_position = torch.tensor([14])
    key_positions = torch.tensor([0, 5, 12])
    query_pre = torch.tensor([[[[0.0, 2.0, 0.0, -1.0]]]])
    key_pre = torch.zeros((1, 1, 15, 4))
    key_pre[0, 0, 0] = torch.tensor([0.0, 1.0, 0.0, 3.0])
    key_pre[0, 0, 5] = torch.tensor([0.0, -2.0, 0.0, 1.0])
    key_pre[0, 0, 12] = torch.tensor([0.0, 4.0, 0.0, 2.0])

    def rotate_at(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        angles = positions.float()[:, None] * torch.tensor([1.0, 0.2])[None, :]
        doubled = torch.cat((angles, angles), dim=-1)
        cosine = doubled.cos()[None, None]
        sine = doubled.sin()[None, None]
        return values * cosine + rotate_half(values) * sine

    query_post = rotate_at(query_pre, query_position)
    key_post = rotate_at(key_pre, torch.arange(15))
    correction = controller._relative_score_correction(
        query_post,
        key_post,
        {1: (0.5, 10.0, 1.0)},
        query_position,
    )[0, 0, 0, key_positions]

    native = (
        query_post[..., 1, None] * key_post[..., 1]
        + query_post[..., 3, None] * key_post[..., 3]
    )[0, 0, 0, key_positions]
    expected = []
    for index, key_position in enumerate(key_positions.tolist()):
        distance = 14.0 - key_position
        warped_distance = distance if distance <= 10.0 else 10.0 + 0.5 * (distance - 10.0)
        angle = -warped_distance * 0.2
        kx = key_pre[0, 0, key_position, 1]
        ky = key_pre[0, 0, key_position, 3]
        rotated_kx = kx * torch.cos(torch.tensor(angle)) - ky * torch.sin(torch.tensor(angle))
        rotated_ky = kx * torch.sin(torch.tensor(angle)) + ky * torch.cos(torch.tensor(angle))
        desired = query_pre[0, 0, 0, 1] * rotated_kx + query_pre[0, 0, 0, 3] * rotated_ky
        expected.append(desired - native[index])
    assert torch.allclose(correction, torch.stack(expected), atol=1e-5)

    half_correction = controller._relative_score_correction(
        query_post,
        key_post,
        {1: (0.5, 10.0, 0.5)},
        query_position,
    )[0, 0, 0, key_positions]
    assert torch.allclose(half_correction, 0.5 * correction, atol=1e-5)


def test_remote_concentration_gate_prefers_focused_remote_attention() -> None:
    controller = object.__new__(MODULE.HeadFrequencyIntervention)
    controller.head_dim = 2
    focused_query = torch.tensor([[[[1.0, 0.0]]]])
    focused_keys = torch.tensor([[[[6.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]]])
    local_query = torch.tensor([[[[1.0, 0.0]]]])
    local_keys = torch.tensor([[[[-6.0, 0.0], [-6.0, 0.0], [6.0, 0.0], [0.0, 0.0]]]])
    config = {
        "mode": "remote_concentration",
        "remote_mass_scale": 0.2,
        "topk": 1,
        "topk_mass_scale": 0.8,
    }
    focused = controller._remote_concentration_gate(
        focused_query,
        focused_keys,
        torch.tensor([3]),
        1.0,
        config,
        1.0,
        None,
        0,
        1,
    )
    local = controller._remote_concentration_gate(
        local_query,
        local_keys,
        torch.tensor([3]),
        1.0,
        config,
        1.0,
        None,
        0,
        1,
    )
    assert focused.shape == (1, 1, 1, 1)
    assert focused.item() > 0.99
    assert local.item() < 0.05
