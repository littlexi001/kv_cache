from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_pg19_causal_echo_ppl_20260714 import parse_indices, tokenize_book_prefix


class FakeTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        assert not add_special_tokens
        return {"input_ids": list(range(len(text)))}


def test_parse_indices_preserves_requested_order() -> None:
    assert parse_indices("4, 1,9") == [4, 1, 9]


def test_tokenize_book_prefix_stops_after_required_tokens() -> None:
    ids = tokenize_book_prefix(FakeTokenizer(), "x" * 500_000, 32_256)
    assert len(ids) >= 32_256
    assert len(ids) < 500_000
