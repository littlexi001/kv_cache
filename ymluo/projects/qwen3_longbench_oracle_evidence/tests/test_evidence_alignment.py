import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "src" / "evidence_alignment.py"
SPEC = importlib.util.spec_from_file_location("evidence_alignment", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_canonical_exact_ignores_html_and_punctuation_but_returns_raw_span():
    source = "Spencer & Plaza(Tamil: name) is a shopping mall in India."
    passage = "Background. Spencer &amp; Plaza (Tamil: name) is a shopping mall in India! More text."

    result = MODULE.align_support_sentence(source, passage)

    assert result.matched
    assert result.match_type == MODULE.MATCH_CANONICAL_EXACT
    assert result.failure_code is None
    assert result.matched_text == "Spencer &amp; Plaza (Tamil: name) is a shopping mall in India"
    assert passage[result.matched_start : result.matched_end] == result.matched_text


def test_fuzzy_local_accepts_one_non_anchor_lexical_edit():
    source = (
        "We Found Love is a song recorded by Barbadian singer Rihanna from her sixth "
        "studio album Talk That Talk released during the year 2011 in November worldwide."
    )
    passage = (
        "Unrelated introduction. We Found Love is a song performed by Barbadian singer "
        "Rihanna from her sixth studio album Talk That Talk released during the year "
        "2011 in November worldwide. Unrelated ending."
    )

    result = MODULE.align_support_sentence(source, passage)

    assert result.matched
    assert result.match_type == MODULE.MATCH_FUZZY_LOCAL
    assert result.sequence_similarity >= 0.95
    assert result.bag_f1 >= 0.95
    assert result.source_recall >= 0.95
    assert not result.anchor_differences


def test_fuzzy_candidate_rejects_changed_number_even_when_similarity_is_high():
    source = (
        "Major General Walter Robert Dornberger born 6 September 1895 and deceased "
        "27 June 1980 was a German Army artillery officer whose long career spanned "
        "both World War I and World War II."
    )
    passage = (
        "Major General Walter Robert Dornberger born 6 September 1895 and deceased "
        "26 June 1980 was a German Army artillery officer whose long career spanned "
        "both World War I and World War II."
    )

    result = MODULE.align_support_sentence(source, passage)

    assert not result.matched
    assert result.failure_code == MODULE.FAIL_PROTECTED_ANCHOR_CHANGED
    assert result.sequence_similarity >= 0.95
    assert result.anchor_differences["numbers"] == {
        "source": ("6", "1895", "27", "1980"),
        "candidate": ("6", "1895", "26", "1980"),
    }


def test_fuzzy_candidate_rejects_changed_month_with_same_day_and_year():
    source = (
        "The carefully maintained public record says the ceremony happened on 12 June "
        "2014 after officials completed every required review for the regional event."
    )
    passage = (
        "The carefully maintained public record says the ceremony happened on 12 July "
        "2014 after officials completed every required review for the regional event."
    )

    result = MODULE.align_support_sentence(source, passage)

    assert not result.matched
    assert result.failure_code == MODULE.FAIL_PROTECTED_ANCHOR_CHANGED
    assert result.anchor_differences["months"] == {
        "source": ("june",),
        "candidate": ("july",),
    }


def test_fuzzy_candidate_rejects_added_negation():
    source = (
        "The university research archive confirms the actor did perform the central role "
        "in the production and later received recognition from the national theatre board."
    )
    passage = (
        "The university research archive confirms the actor did not perform the central "
        "role in the production and later received recognition from the national theatre board."
    )

    result = MODULE.align_support_sentence(source, passage)

    assert not result.matched
    assert result.failure_code == MODULE.FAIL_PROTECTED_ANCHOR_CHANGED
    assert result.anchor_differences["negations"] == {
        "source": (),
        "candidate": ("not",),
    }


def test_fuzzy_candidate_rejects_reversed_comparison_direction():
    source = (
        "The official chronology states that the Miller case reached the court earlier "
        "than the Collier case according to the complete judicial history kept by archivists."
    )
    passage = (
        "The official chronology states that the Miller case reached the court later "
        "than the Collier case according to the complete judicial history kept by archivists."
    )

    result = MODULE.align_support_sentence(source, passage)

    assert not result.matched
    assert result.failure_code == MODULE.FAIL_PROTECTED_ANCHOR_CHANGED
    assert result.anchor_differences["comparisons"] == {
        "source": ("time:-",),
        "candidate": ("time:+",),
    }


def test_fuzzy_candidate_rejects_numbers_rebound_to_different_entities():
    source = (
        "The complete municipal archive records that Alice was born in 1980 and later "
        "worked at the northern institute for many productive years, while Bob was born "
        "in 1990 and later worked at the southern institute for many productive years "
        "according to the same carefully preserved public register."
    )
    passage = (
        "The complete municipal archive records that Alice was born in 1990 and later "
        "worked at the northern institute for many productive years, while Bob was born "
        "in 1980 and later worked at the southern institute for many productive years "
        "according to the same carefully preserved public register."
    )

    result = MODULE.align_support_sentence(source, passage)

    assert result.sequence_similarity >= 0.95
    assert result.bag_f1 == 1.0
    assert not result.matched
    assert result.failure_code == MODULE.FAIL_PROTECTED_ANCHOR_CHANGED
    assert result.anchor_differences["numbers"] == {
        "source": ("1980", "1990"),
        "candidate": ("1990", "1980"),
    }


def test_fuzzy_candidate_rejects_comparisons_rebound_to_different_entities():
    source = (
        "The complete biographical archive explains that Alice was older than her local "
        "colleague during their long service at the northern institute, while Bob was "
        "younger than his local colleague during their long service at the southern "
        "institute according to the same carefully preserved public register."
    )
    passage = (
        "The complete biographical archive explains that Alice was younger than her local "
        "colleague during their long service at the northern institute, while Bob was "
        "older than his local colleague during their long service at the southern "
        "institute according to the same carefully preserved public register."
    )

    result = MODULE.align_support_sentence(source, passage)

    assert result.sequence_similarity >= 0.95
    assert result.bag_f1 == 1.0
    assert not result.matched
    assert result.failure_code == MODULE.FAIL_PROTECTED_ANCHOR_CHANGED
    assert result.anchor_differences["comparisons"] == {
        "source": ("age:+", "age:-", "equality:same"),
        "candidate": ("age:-", "age:+", "equality:same"),
    }


def test_fuzzy_candidate_rejects_entity_swap_with_number_order_unchanged():
    source = (
        "The complete municipal archive states that Alice was born in 1980 and served "
        "the northern institute for many years with a distinguished public record, while "
        "Bob was born in 1990 and served the southern institute for many years with a "
        "distinguished public record preserved by regional officials."
    )
    passage = (
        "The complete municipal archive states that Bob was born in 1980 and served "
        "the northern institute for many years with a distinguished public record, while "
        "Alice was born in 1990 and served the southern institute for many years with a "
        "distinguished public record preserved by regional officials."
    )

    result = MODULE.align_support_sentence(source, passage)

    assert result.sequence_similarity >= 0.95
    assert result.bag_f1 == 1.0
    assert result.protected_anchors["numbers"] == result.matched_anchors["numbers"]
    assert not result.matched
    assert result.failure_code == MODULE.FAIL_PROTECTED_ANCHOR_CHANGED
    assert "bindings" in result.anchor_differences


def test_fuzzy_candidate_rejects_entity_swap_with_comparison_order_unchanged():
    source = (
        "The complete biographical archive states that Alice was older than Carol during "
        "the northern expedition that catalogued rare manuscripts and restored the historic "
        "library building, while Bob was younger than David during a southern conference "
        "that reviewed municipal biographies and published the official regional register "
        "for researchers and local historians."
    )
    passage = (
        "The complete biographical archive states that Bob was older than Carol during "
        "the northern expedition that catalogued rare manuscripts and restored the historic "
        "library building, while Alice was younger than David during a southern conference "
        "that reviewed municipal biographies and published the official regional register "
        "for researchers and local historians."
    )

    result = MODULE.align_support_sentence(source, passage)

    assert result.sequence_similarity >= 0.95
    assert result.bag_f1 == 1.0
    assert result.protected_anchors["comparisons"] == result.matched_anchors["comparisons"]
    assert not result.matched
    assert result.failure_code == MODULE.FAIL_PROTECTED_ANCHOR_CHANGED
    assert "bindings" in result.anchor_differences


def test_equal_nonoverlapping_high_scoring_candidates_are_ambiguous():
    source = (
        "The regional archive reports that Alice served as director of the institute "
        "during the annual public science program hosted in the central city library."
    )
    first = (
        "The regional archive reports that Alice worked as director of the institute "
        "during the annual public science program hosted in the central city library."
    )
    second = (
        "The regional archive reports that Alice acted as director of the institute "
        "during the annual public science program hosted in the central city library."
    )

    result = MODULE.align_support_sentence(source, first + " " + second)

    assert not result.matched
    assert result.failure_code == MODULE.FAIL_CANDIDATE_AMBIGUOUS
    assert result.runner_up_candidate is not None
    assert result.candidate_margin == 0.0


def test_adjacent_two_sentence_candidate_can_pass_fuzzy_gate():
    source = (
        "Sault Ste Marie is a city in the American state of Michigan; it remains the "
        "county seat of Chippewa County according to the current officially maintained "
        "regional government record for residents."
    )
    passage = (
        "Sault Ste Marie is a municipality in the American state of Michigan. "
        "It remains the county seat of Chippewa County according to the current officially "
        "maintained regional government record for residents."
    )

    result = MODULE.align_support_sentence(source, passage)

    assert result.matched
    assert result.match_type == MODULE.MATCH_FUZZY_LOCAL
    assert result.best_candidate.sentence_end - result.best_candidate.sentence_start == 1


def test_low_lexical_overlap_gets_similarity_failure_with_best_candidate_audit():
    source = "The old source sentence identifies a specific court decision and its author."
    passage = "This passage discusses a botanical garden and several tropical trees."

    result = MODULE.align_support_sentence(source, passage)

    assert not result.matched
    assert result.failure_code == MODULE.FAIL_SIMILARITY_BELOW_THRESHOLD
    assert result.best_candidate is not None
    assert result.matched_text == passage


def test_time_anchor_normalizes_7pm_and_7_00pm():
    source = MODULE.extract_protected_anchors("The program aired at 7pm on Monday.")
    candidate = MODULE.extract_protected_anchors("The program aired at 7.00 p.m. on Monday.")

    assert source["times"] == ("07:00pm",)
    assert source["times"] == candidate["times"]
    assert MODULE.compare_anchors(source, candidate) == {}


def test_duplicate_canonical_exact_occurrences_are_safe_and_counted():
    source = "The same evidence sentence appears here."
    passage = "The same evidence sentence appears here. Other. The same evidence sentence appears here!"

    result = MODULE.align_support_sentence(source, passage)

    assert result.matched
    assert result.match_type == MODULE.MATCH_CANONICAL_EXACT
    assert result.exact_occurrence_count == 2


def test_empty_inputs_return_explicit_failure_codes():
    empty_source = MODULE.align_support_sentence("...", "A real passage.")
    empty_passage = MODULE.align_support_sentence("A real source sentence.", "---")

    assert empty_source.failure_code == MODULE.FAIL_EMPTY_SOURCE
    assert empty_passage.failure_code == MODULE.FAIL_EMPTY_PASSAGE


def test_to_dict_is_json_serializable_and_exposes_thresholds():
    result = MODULE.align_support_sentence("A useful sentence.", "A useful sentence!")
    serialized = result.to_dict()

    assert serialized["matched"] is True
    assert serialized["thresholds"]["minimum_sequence_similarity"] == 0.95
    json.dumps(serialized)
