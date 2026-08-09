from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from offloaded_prefill_cache_20260716 import (
    OffloadedExactPrefillCache,
    QuantizedOffloadedExactPrefillCache,
)


def test_offloaded_prefill_cache_restores_exact_history() -> None:
    cache = OffloadedExactPrefillCache(capacity=5, pin_memory=False)
    first_key = torch.tensor([[[[1.0], [2.0]]]])
    first_value = first_key + 10.0
    full_key, full_value = cache.update(first_key, first_value, layer_idx=0)

    assert full_key.flatten().tolist() == [1.0, 2.0]
    assert full_value.flatten().tolist() == [11.0, 12.0]
    assert cache.get_seq_length() == 2

    second_key = torch.tensor([[[[3.0], [4.0]]]])
    second_value = second_key + 10.0
    full_key, full_value = cache.update(second_key, second_value, layer_idx=0)

    assert full_key.flatten().tolist() == [1.0, 2.0, 3.0, 4.0]
    assert full_value.flatten().tolist() == [11.0, 12.0, 13.0, 14.0]
    assert cache.get_seq_length() == 4
    assert cache.completed_states()[0].host_kv[0, 0, 0, :4, 0].tolist() == [
        1.0,
        2.0,
        3.0,
        4.0,
    ]


def test_offloaded_prefill_cache_rejects_capacity_overflow() -> None:
    cache = OffloadedExactPrefillCache(capacity=2, pin_memory=False)
    cache.update(torch.ones(1, 1, 2, 1), torch.ones(1, 1, 2, 1), 0)

    with pytest.raises(ValueError, match="capacity"):
        cache.update(torch.ones(1, 1, 1, 1), torch.ones(1, 1, 1, 1), 0)


def test_offloaded_prefill_cache_requires_synchronized_layers() -> None:
    cache = OffloadedExactPrefillCache(capacity=4, pin_memory=False)
    cache.update(torch.ones(1, 1, 2, 1), torch.ones(1, 1, 2, 1), 0)
    cache.update(torch.ones(1, 1, 1, 1), torch.ones(1, 1, 1, 1), 1)

    with pytest.raises(RuntimeError, match="not synchronized"):
        cache.completed_states()


def test_offloaded_prefill_matches_dynamic_cache_on_tiny_llama() -> None:
    torch.manual_seed(7)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            attention_dropout=0.0,
        )
    ).eval()
    input_ids = torch.randint(0, 128, (1, 12))

    def forward_chunks(cache: object | None) -> tuple[object, torch.Tensor]:
        logits = None
        for start in range(0, input_ids.shape[-1], 4):
            end = start + 4
            outputs = model(
                input_ids=input_ids[:, start:end],
                past_key_values=cache,
                cache_position=torch.arange(start, end),
                use_cache=True,
                return_dict=True,
            )
            cache = outputs.past_key_values
            logits = outputs.logits
        assert logits is not None
        return cache, logits

    dynamic_cache, dynamic_logits = forward_chunks(None)
    offloaded_cache, offloaded_logits = forward_chunks(
        OffloadedExactPrefillCache(capacity=12, pin_memory=False)
    )

    assert isinstance(offloaded_cache, OffloadedExactPrefillCache)
    assert torch.allclose(offloaded_logits, dynamic_logits, atol=1.0e-6, rtol=1.0e-5)
    for layer, state in enumerate(offloaded_cache.completed_states()):
        assert torch.allclose(
            state.host_kv[0, ..., :12, :],
            dynamic_cache.key_cache[layer],
            atol=1.0e-6,
            rtol=1.0e-5,
        )
        assert torch.allclose(
            state.host_kv[1, ..., :12, :],
            dynamic_cache.value_cache[layer],
            atol=1.0e-6,
            rtol=1.0e-5,
        )

    quantized_cache, quantized_logits = forward_chunks(
        QuantizedOffloadedExactPrefillCache(
            capacity=12,
            bits=4,
            pin_memory=False,
        )
    )
    assert isinstance(
        quantized_cache,
        QuantizedOffloadedExactPrefillCache,
    )
    assert torch.isfinite(quantized_logits).all()
    assert quantized_logits.shape == dynamic_logits.shape
    assert all(
        state.length == 12
        for state in quantized_cache.completed_states()
    )


@pytest.mark.parametrize(
    ("bits", "maximum_error"),
    ((4, 0.08), (8, 0.005)),
)
def test_quantized_offloaded_prefill_keeps_exact_host_and_current_chunk(
    bits: int,
    maximum_error: float,
) -> None:
    cache = QuantizedOffloadedExactPrefillCache(
        capacity=6,
        bits=bits,
        pin_memory=False,
    )
    first_key = torch.tensor(
        [[[[1.0, -0.75, 0.5, -0.25], [0.25, -0.5, 0.75, -1.0]]]]
    )
    first_value = first_key * 1.5
    cache.update(first_key, first_value, layer_idx=0)

    second_key = torch.tensor(
        [[[[0.2, 0.4, 0.6, 0.8], [-0.8, -0.6, -0.4, -0.2]]]]
    )
    second_value = second_key * 1.5
    full_key, full_value = cache.update(
        second_key,
        second_value,
        layer_idx=0,
    )

    assert torch.allclose(
        full_key[..., :2, :],
        first_key,
        atol=maximum_error,
        rtol=0.0,
    )
    assert torch.allclose(
        full_value[..., :2, :],
        first_value,
        atol=1.5 * maximum_error,
        rtol=0.0,
    )
    assert torch.equal(full_key[..., 2:, :], second_key)
    assert torch.equal(full_value[..., 2:, :], second_value)
    state = cache.completed_states()[0]
    assert torch.equal(
        state.host_kv[0, ..., :4, :],
        torch.cat((first_key, second_key), dim=-2),
    )
    assert torch.equal(
        state.host_kv[1, ..., :4, :],
        torch.cat((first_value, second_value), dim=-2),
    )
    assert cache.transient_gpu_bytes() < state.host_kv.numel() * 4


def test_quantized_offloaded_prefill_rejects_invalid_bits() -> None:
    with pytest.raises(ValueError, match="bits"):
        QuantizedOffloadedExactPrefillCache(
            capacity=4,
            bits=2,
            pin_memory=False,
        )


def test_groupwise_int4_reduces_outlier_induced_error() -> None:
    values = torch.tensor(
        [[[[20.0, 0.25, -0.5, 0.75, 1.0, -1.25, 1.5, -1.75]]]]
    )
    per_head = QuantizedOffloadedExactPrefillCache(
        capacity=1,
        bits=4,
        pin_memory=False,
    )
    groupwise = QuantizedOffloadedExactPrefillCache(
        capacity=1,
        bits=4,
        group_size=2,
        pin_memory=False,
    )

    per_head_codes, per_head_scales = per_head._quantize(values)
    group_codes, group_scales = groupwise._quantize(values)
    per_head_values = per_head._dequantize(
        per_head_codes,
        per_head_scales,
        values.dtype,
        values.shape[-1],
    )
    group_values = groupwise._dequantize(
        group_codes,
        group_scales,
        values.dtype,
        values.shape[-1],
    )

    per_head_mse = torch.mean((per_head_values - values) ** 2)
    group_mse = torch.mean((group_values - values) ** 2)
    assert group_mse < per_head_mse * 0.1
    assert group_scales.shape[-1] == 4


def test_quantized_prefill_rejects_nondivisible_group_size() -> None:
    cache = QuantizedOffloadedExactPrefillCache(
        capacity=1,
        bits=4,
        group_size=3,
        pin_memory=False,
    )

    with pytest.raises(ValueError, match="divide"):
        cache.update(
            torch.ones(1, 1, 1, 8),
            torch.ones(1, 1, 1, 8),
            layer_idx=0,
        )
