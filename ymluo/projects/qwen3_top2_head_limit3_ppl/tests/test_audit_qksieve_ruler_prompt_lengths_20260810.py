from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "audit_qksieve_ruler_prompt_lengths_20260810.py"
)
SPEC = importlib.util.spec_from_file_location("ruler_prompt_audit", MODULE_PATH)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class CharacterTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False):
        assert not add_special_tokens
        return {"input_ids": list(text)}


def row(length: int, max_new_tokens: int = 2) -> dict[str, object]:
    return {
        "task": f"niah_single_1_{length}",
        "sample_id": f"sample-{length}",
        "context": "abcd",
        "query": "question",
        "prefix_template": "p",
        "suffix_template": "-{input}",
        "max_new_tokens": max_new_tokens,
        "length": length,
    }


def test_none_wrapper_matches_three_region_tokenization() -> None:
    observed = audit.prompt_token_count(
        CharacterTokenizer(), row(16), "none", set()
    )
    assert observed == len("p") + len("abcd") + len("-question")


def test_no_chat_disables_llama_wrapper() -> None:
    item = row(16)
    item["no_chat"] = True
    assert audit.prompt_token_count(
        CharacterTokenizer(), item, "llama3", set()
    ) == audit.prompt_token_count(CharacterTokenizer(), item, "none", set())


def test_summary_reports_generation_reserve_overflow() -> None:
    rows = [(Path("examples.jsonl"), row(16, max_new_tokens=8))]
    per_length, total = audit.summarize_lengths(
        rows,
        CharacterTokenizer(),
        "none",
        set(),
        max_sequence_tokens=20,
    )
    assert total == 1
    assert per_length["16"]["overflow_rows"] == 1
    assert per_length["16"]["max_prompt_plus_generation_tokens"] == 22
