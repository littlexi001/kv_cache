import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "src" / "run_hotpot_oracle_pilot.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("oracle_runner", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_official_qa_f1_and_exact_match():
    assert MODULE.official_score("The Nile", ["Nile"]) == 1.0
    assert MODULE.normalized_exact_match("the Nile!", ["Nile"])
    assert MODULE.official_score("blue river", ["blue ocean"]) == 0.5


def test_parse_hf_dict_of_lists():
    value = {"title": ["A", "B"], "sent_id": [1, 2]}
    assert MODULE.parse_struct_pairs(value, "title", "sent_id") == [("A", 1), ("B", 2)]


def test_evidence_position_bins():
    assert MODULE.position_bin(0.1) == "early"
    assert MODULE.position_bin(0.5) == "middle"
    assert MODULE.position_bin(0.9) == "late"
    position = MODULE.evidence_position("alpha beta gamma delta", ["gamma"])
    assert 0.4 < position < 0.9


def test_answer_boundary_check():
    assert MODULE.contains_answer("The answer is New York.", ["New York"])
    assert not MODULE.contains_answer("A yorkshire terrier", ["York"])


def test_parse_longbench_passages_and_offsets():
    context = "Passage 1:\nAlpha_Title\nFirst body.\n\nPassage 2:\nBeta\nSecond body."
    documents = MODULE.parse_longbench_passages(context)
    assert [document.title for document in documents] == ["Alpha_Title", "Beta"]
    assert [document.sentences for document in documents] == [
        ("First body.",),
        ("Second body.",),
    ]
    assert documents[0].char_start < documents[1].char_start
    assert context[documents[0].char_start : documents[0].char_end].startswith(
        "Passage 1:"
    )


def test_title_normalization_handles_unicode_case_and_underscores():
    assert MODULE.normalize_title("  Miller_v.  California ") == MODULE.normalize_title(
        "MILLER V. CALIFORNIA"
    )


def test_random_prefix_matches_token_budget():
    class CharacterTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(character) for character in text]}

        def decode(self, ids, **_kwargs):
            return "".join(chr(token_id) for token_id in ids)

    context, actual, repetitions = MODULE.token_budget_prefix(
        CharacterTokenizer(), "abcdefgh", 5
    )
    assert context == "abcde"
    assert actual == 5
    assert repetitions == 1


def test_random_prefix_repeats_short_pool_to_match_budget():
    class CharacterTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(character) for character in text]}

        def decode(self, ids, **_kwargs):
            return "".join(chr(token_id) for token_id in ids)

    context, actual, repetitions = MODULE.token_budget_prefix(
        CharacterTokenizer(), "abc", 8
    )
    assert context == "abc\n\nabc"
    assert actual == 8
    assert repetitions == 3
