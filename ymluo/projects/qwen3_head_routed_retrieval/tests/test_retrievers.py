from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


SOURCE = Path(__file__).resolve().parents[1] / "src" / "run_head_retriever_imitation.py"
SPEC = importlib.util.spec_from_file_location("head_retriever_imitation", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_bank() -> object:
    token_ids = torch.tensor([1, 2, 3, 1, 2, 4, 5, 1, 2, 3, 6, 7], dtype=torch.long)
    embeddings = torch.eye(12, 8, dtype=torch.float32)
    decoded = ["a", "b", ".", "a", "b", "c", "\n", "a", "b", ".", "1", "D"]
    return MODULE.RetrieverBank(
        token_ids,
        embeddings,
        decoded,
        ratio=0.25,
        query_window=3,
        block_size=4,
        repeat_max_n=3,
        sink_tokens=1,
        hybrid_position_fraction=0.5,
        random_seed=7,
    )


def test_all_retrievers_respect_exact_budget_and_history() -> None:
    bank = make_bank()
    current = 9
    expected = MODULE.historical_budget(current, 0.25)
    selected = bank.selections(current)
    assert tuple(selected) == MODULE.METHODS
    for indices in selected.values():
        assert len(indices) == expected
        assert int(indices.min()) >= 0
        assert int(indices.max()) < current
        assert len(torch.unique(indices)) == expected


def test_retrieval_is_deterministic() -> None:
    bank = make_bank()
    first = bank.selections(9)
    second = bank.selections(9)
    for method in MODULE.METHODS:
        assert torch.equal(first[method], second[method])


def test_position_retriever_keeps_sink() -> None:
    bank = make_bank()
    selected = bank.selections(9)["position"]
    assert 0 in selected.tolist()


def test_hybrids_keep_exact_budget_without_duplicates() -> None:
    bank = make_bank()
    current = 9
    expected = MODULE.historical_budget(current, 0.25)
    for method in ("hybrid_lexical", "hybrid_semantic", "hybrid_format", "hybrid_repeat"):
        selected = bank.selections(current)[method]
        assert len(selected) == expected
        assert len(torch.unique(selected)) == expected
        assert 0 in selected.tolist()


def test_format_categories_are_distinct() -> None:
    assert MODULE.token_format_category("\n") != MODULE.token_format_category("word")
    assert MODULE.token_format_category(".") != MODULE.token_format_category("word")
    assert MODULE.token_format_category("123") != MODULE.token_format_category("word")
