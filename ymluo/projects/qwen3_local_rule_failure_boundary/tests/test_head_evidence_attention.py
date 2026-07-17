from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from run_head_evidence_attention import make_nonconflict_variant  # noqa: E402
from run_local_rule_failure_boundary import RuleEvent  # noqa: E402
from summarize_head_evidence_attention import aggregate, paired_effects  # noqa: E402


class CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}


def test_nonconflict_pair_changes_only_conflict_antecedent() -> None:
    tokenizer = CharacterTokenizer()
    text = "DECOY RULE D0: AB12-345 -> ZZ99-999\n"
    event = RuleEvent("conflict", "D0", text, 3, 3 + len(text), "AB12-345", "ZZ99-999", -1)
    prefix = [1, 2, 3]
    suffix = [4, 5]
    prompt = torch.tensor([prefix + tokenizer.encode(text) + suffix])
    changed, events, difference_count = make_nonconflict_variant(tokenizer, prompt, [event], 17)

    assert changed.shape == prompt.shape
    assert torch.equal(changed[0, :3], prompt[0, :3])
    assert torch.equal(changed[0, event.end_token :], prompt[0, event.end_token :])
    assert events[0].kind == "decoy"
    assert events[0].antecedent != event.antecedent
    assert events[0].consequent == event.consequent
    assert difference_count > 0


def test_paired_effect_is_conflict_minus_nonconflict() -> None:
    base = {
        "pair_id": "p0",
        "competitor_count": "0",
        "layer": "1",
        "head": "2",
    }
    nonconflict = {**base, "condition": "nonconflict"}
    conflict = {**base, "condition": "conflict"}
    for metric in (
        "gold_rule_mass",
        "decoy_rule_mass",
        "competitor_rule_mass",
        "non_gold_rule_mass",
        "gold_rule_selectivity",
        "gold_uniform_enrichment",
        "gold_vs_decoy_log2_density_ratio",
        "gold_vs_background_density_ratio",
        "gold_top2_token_recall",
        "gold_top2_token_precision",
        "gold_top2_mass_recall",
        "gold_best_token_rank",
        "gold_mean_token_rank",
        "gold_step_0_mass",
        "gold_step_1_mass",
    ):
        nonconflict[metric] = "1.0"
        conflict[metric] = "1.25"
    effects = paired_effects([nonconflict, conflict])
    exact = [row for row in effects if row["competitor_count"] == "0"][0]
    assert exact["sample_count"] == 1
    assert exact["mean_delta_gold_rule_mass"] == 0.25


def test_aggregate_ignores_nonfinite_values() -> None:
    rows = [
        {"condition": "x", "value": "1"},
        {"condition": "x", "value": "nan"},
        {"condition": "x", "value": "3"},
    ]
    result = aggregate(rows, ("condition",), ["value"])
    assert result[0]["mean_value"] == 2.0
