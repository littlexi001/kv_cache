import torch

from fier_rtn1_cuda_20260728 import (
    allocate_packed_index,
    allocated_bytes,
    reconstruct_keys,
    scores,
    update_packed_index,
)
from run_head_top2_targeted_ppl_20260714 import (
    _fier_rtn1_packed_attention,
    _fier_rtn1_reconstruct_groups,
)


def test_packed_fier_matches_reference_reconstruction():
    generator = torch.Generator().manual_seed(17)
    keys = torch.randn(
        1, 2, 37, 128, generator=generator, dtype=torch.float16
    )
    packed = allocate_packed_index(1, 2, 64, keys.device)

    update_packed_index(keys, packed, 37)

    expected = _fier_rtn1_reconstruct_groups(keys, group_size=32)
    actual = reconstruct_keys(packed, 37, dtype=keys.dtype)
    assert torch.equal(actual, expected)
    assert packed["indexed_count"] == 37


def test_packed_fier_incrementally_rebuilds_only_open_group():
    generator = torch.Generator().manual_seed(29)
    keys = torch.randn(
        1, 1, 45, 128, generator=generator, dtype=torch.float16
    )
    packed = allocate_packed_index(1, 1, 64, keys.device)

    update_packed_index(keys, packed, 31)
    update_packed_index(keys, packed, 32)
    closed_codes = packed["packed_codes"][..., 0, :].clone()
    closed_lower = packed["lower"][..., 0, :].clone()
    closed_upper = packed["upper"][..., 0, :].clone()
    update_packed_index(keys, packed, 45)

    assert torch.equal(packed["packed_codes"][..., 0, :], closed_codes)
    assert torch.equal(packed["lower"][..., 0, :], closed_lower)
    assert torch.equal(packed["upper"][..., 0, :], closed_upper)
    expected = _fier_rtn1_reconstruct_groups(keys, group_size=32)
    assert torch.equal(
        reconstruct_keys(packed, 45, dtype=keys.dtype), expected
    )


def test_packed_fier_scores_support_gqa():
    generator = torch.Generator().manual_seed(43)
    keys = torch.randn(
        2, 2, 35, 128, generator=generator, dtype=torch.float16
    )
    query = torch.randn(
        2, 8, 128, generator=generator, dtype=torch.float16
    )
    packed = allocate_packed_index(2, 2, 64, keys.device)
    update_packed_index(keys, packed, 35)

    actual = scores(query, packed, 35)
    reconstructed = reconstruct_keys(packed, 35)
    expected = torch.einsum(
        "bhgd,bhkd->bhgk",
        query.float().reshape(2, 2, 4, 128),
        reconstructed,
    ).reshape(2, 8, 35)
    assert torch.allclose(actual, expected, atol=1.0e-5, rtol=1.0e-5)


def test_packed_fier_uses_paper_spec_bytes():
    packed = allocate_packed_index(
        batch_count=2,
        kv_head_count=3,
        capacity_tokens=64,
        device=torch.device("cpu"),
    )

    assert allocated_bytes(packed) == 2 * 3 * 64 * 32
    assert packed["logical_bits_per_head_token"] == 256


def test_packed_fier_attention_selects_proxy_topk_and_records_contract():
    class FakeKernels:
        def final_attention_ragged_self(
            self,
            query,
            key,
            value,
            candidate_indices,
            candidate_counts,
            scaling,
        ):
            self.indices = candidate_indices
            self.counts = candidate_counts
            return torch.zeros_like(query)

    generator = torch.Generator().manual_seed(71)
    keys = torch.randn(
        1, 1, 5, 128, generator=generator, dtype=torch.float16
    )
    query = torch.randn(
        1, 2, 128, generator=generator, dtype=torch.float16
    )
    value = torch.randn(
        1, 1, 5, 128, generator=generator, dtype=torch.float16
    )
    packed = allocate_packed_index(1, 1, 32, keys.device)
    update_packed_index(keys, packed, 5)
    expected_scores = scores(query, packed, 5)
    expected = torch.topk(
        expected_scores, k=2, dim=-1, sorted=False
    ).indices
    state = {}
    kernels = FakeKernels()

    output, indices = _fier_rtn1_packed_attention(
        query,
        keys,
        value,
        history_count=5,
        scaling=1.0,
        selected_fraction=0.4,
        state=state,
        cuda_kernels=kernels,
        diagnostics={},
    )

    assert output.shape == query.shape
    assert torch.equal(indices.sort().values, expected.sort().values)
    assert torch.equal(kernels.counts, torch.full((1, 2), 2))
    assert state["packed_qmse_transform"] == "fier_rtn1_g32_packed"
    assert state["hierarchical_logical_bits_per_token"] == 256.0
    assert state["fier_packed_allocated_bytes"] > 0
