from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import torch


SOURCE = Path(__file__).resolve().parents[1] / "src" / "run_per_head_hierarchical_memory.py"
SPEC = importlib.util.spec_from_file_location("per_head_hierarchical_memory", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

PPL_SOURCE = Path(__file__).resolve().parents[1] / "src" / "run_sparse_memory_ppl.py"
PPL_SPEC = importlib.util.spec_from_file_location("sparse_memory_ppl", PPL_SOURCE)
assert PPL_SPEC is not None and PPL_SPEC.loader is not None
PPL_MODULE = importlib.util.module_from_spec(PPL_SPEC)
PPL_SPEC.loader.exec_module(PPL_MODULE)


def make_retriever(tokens: int = 800):
    token_ids = torch.tensor([(index * 7) % 31 for index in range(tokens)], dtype=torch.long)
    embeddings = torch.stack(
        [torch.sin(torch.arange(12, dtype=torch.float32) + index) for index in range(tokens)]
    )
    decoded = ["\n" if index % 23 == 0 else str(int(token_ids[index])) for index in range(tokens)]
    return MODULE.RetrieverBank(
        token_ids,
        embeddings,
        decoded,
        ratio=0.02,
        query_window=8,
        block_size=16,
        repeat_max_n=3,
        sink_tokens=2,
        hybrid_position_fraction=0.5,
        random_seed=7,
    )


def make_memory(*, heads: int = 4, l0: int = 20):
    weights = torch.tensor(
        [
            [0.80, 0.05, 0.05, 0.05, 0.05],
            [0.05, 0.80, 0.05, 0.05, 0.05],
            [0.05, 0.05, 0.80, 0.05, 0.05],
            [0.05, 0.05, 0.05, 0.80, 0.05],
        ][:heads],
        dtype=torch.float32,
    )
    return MODULE.PerHeadHierarchicalMemory(
        make_retriever(),
        weights,
        l0_capacity=l0,
        l0_recent_tokens=min(8, l0 - 2),
        l1_capacity=max(40, l0),
        l2_block_size=16,
        l2_block_budget=4,
        sink_tokens=2,
        l1_retention_bonus=0.05,
        l0_retention_bonus=0.05,
    )


def test_every_policy_respects_unique_historical_l0_capacity() -> None:
    memory = make_memory()
    selections = memory.selections(200)
    assert tuple(selections) == MODULE.POLICIES
    for selected in selections.values():
        assert selected.shape == (4, 20)
        assert int(selected.min()) >= 0
        assert int(selected.max()) < 200
        for row in selected:
            assert len(torch.unique(row)) == 20


def test_hierarchy_keeps_sink_and_recent_reservation() -> None:
    memory = make_memory()
    selected = memory.selections(200)["hier_function_500"]
    mandatory = set(range(2)) | set(range(192, 200))
    for row in selected:
        assert mandatory.issubset(set(int(value) for value in row.tolist()))


def test_head_specific_weights_create_distinct_hot_memories() -> None:
    memory = make_memory()
    selected = memory.selections(200)["hier_function_500"]
    assert len(torch.unique(selected, dim=0)) > 1


def test_state_is_persistent_and_chunk_cache_supports_old_layers() -> None:
    memory = make_memory()
    first = memory.selections(200)["hier_function_500"].clone()
    memory.selections(201)
    replay = memory.selections(200)["hier_function_500"]
    assert torch.equal(first, replay)
    assert memory.warm_positions.shape[1] <= memory.l1_capacity
    assert memory.hot_positions.shape[1] <= memory.l0_capacity


def test_hard_500_token_cap() -> None:
    memory = make_memory(l0=500)
    selected = memory.selections(700)["hier_function_500"]
    assert selected.shape == (4, 500)
    assert max(int(row["l0_max_tokens"]) for row in memory.diagnostic_rows) == 500


def test_functional_weights_are_normalized_and_cover_all_heads() -> None:
    atlas_path = (
        Path(__file__).resolve().parents[2]
        / "qwen3_head_function_atlas"
        / "outputs"
        / "head_function_atlas.csv"
    )
    with atlas_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    weights, routes = MODULE.functional_method_weights(rows)
    assert weights.shape == (448, 5)
    assert len(routes) == 448
    assert torch.allclose(weights.sum(dim=-1), torch.ones(448))
    assert len(torch.unique(weights, dim=0)) > 100


def test_sparse_mask_allows_l0_and_current_chunk_only() -> None:
    memory = make_memory()
    controller = PPL_MODULE.SparseMemoryController(
        memory,
        policy="hier_function_500",
        layer_count=1,
        head_count=4,
    )
    allowed = controller.allowed_mask(
        layer=0,
        query_count=2,
        key_count=202,
        device=torch.device("cpu"),
    )
    assert allowed.shape == (4, 2, 202)
    assert allowed[:, 0, 200].all()
    assert allowed[:, 1, 200:202].all()
    assert not allowed[:, 0, 201].any()
    assert controller.max_allowed_history == 20
    assert controller.min_allowed_history == 20


def test_confidence_gated_promotions_only_activate_supported_heads() -> None:
    rows = [
        {
            "layer": "0",
            "head": "0",
            "conservative_function": "semantic_evidence",
            "confidence": "高",
        },
        {
            "layer": "0",
            "head": "1",
            "conservative_function": "lexical_copy",
            "confidence": "中",
        },
        {
            "layer": "0",
            "head": "2",
            "conservative_function": "syntactic_dependency",
            "confidence": "高",
        },
        {
            "layer": "0",
            "head": "3",
            "conservative_function": "mixed_or_common",
            "confidence": "低",
        },
    ]
    slots = MODULE.promotion_slots_from_atlas(
        rows,
        l0_capacity=500,
        l0_recent_tokens=448,
        sink_tokens=4,
        policy="confidence_gated",
        medium_slots=20,
    )
    assert slots.tolist() == [48, 20, 0, 0]

    semantic_only = MODULE.promotion_slots_from_atlas(
        rows,
        l0_capacity=500,
        l0_recent_tokens=448,
        sink_tokens=4,
        policy="confidence_gated",
        medium_slots=20,
        active_categories=["semantic_evidence"],
    )
    assert semantic_only.tolist() == [48, 0, 0, 0]
