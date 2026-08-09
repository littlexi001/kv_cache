#!/usr/bin/env python3
"""Fail-closed alignment of HotpotQA support sentences to LongBench passages.

The module is intentionally independent of the model runner.  It uses only the
annotated source support sentence and the corresponding LongBench passage; a
gold answer string is neither accepted by the API nor used by the algorithm.

Alignment has two acceptance paths:

1. the canonical source-token sequence occurs exactly in the passage; or
2. one LongBench sentence (or two adjacent sentences) clears every lexical,
   length, uniqueness, and protected-anchor gate in :class:`AlignmentConfig`.

Every rejected result retains the best candidate and stage-local audit values
so that a human can tell which gate failed.
"""

from __future__ import annotations

import difflib
import html
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence


FAIL_EMPTY_SOURCE = "empty_source"
FAIL_EMPTY_PASSAGE = "empty_passage"
FAIL_NO_SENTENCE_CANDIDATE = "no_sentence_candidate"
FAIL_NO_LENGTH_ELIGIBLE_CANDIDATE = "no_length_eligible_candidate"
FAIL_SIMILARITY_BELOW_THRESHOLD = "similarity_below_threshold"
FAIL_PROTECTED_ANCHOR_CHANGED = "protected_anchor_changed"
FAIL_CANDIDATE_AMBIGUOUS = "candidate_ambiguous"

MATCH_CANONICAL_EXACT = "canonical_exact"
MATCH_FUZZY_LOCAL = "fuzzy_local"


@dataclass(frozen=True)
class AlignmentConfig:
    """Thresholds for the fail-closed local alignment decision."""

    minimum_sequence_similarity: float = 0.95
    minimum_bag_f1: float = 0.95
    minimum_source_recall: float = 0.95
    minimum_length_ratio: float = 0.75
    maximum_length_ratio: float = 1.33
    minimum_candidate_margin: float = 0.05
    maximum_adjacent_sentences: int = 2

    def __post_init__(self) -> None:
        unit_interval = {
            "minimum_sequence_similarity": self.minimum_sequence_similarity,
            "minimum_bag_f1": self.minimum_bag_f1,
            "minimum_source_recall": self.minimum_source_recall,
            "minimum_candidate_margin": self.minimum_candidate_margin,
        }
        for name, value in unit_interval.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.minimum_length_ratio <= 0.0:
            raise ValueError("minimum_length_ratio must be positive")
        if self.maximum_length_ratio < self.minimum_length_ratio:
            raise ValueError("maximum_length_ratio must be >= minimum_length_ratio")
        if self.maximum_adjacent_sentences not in (1, 2):
            raise ValueError("maximum_adjacent_sentences must be 1 or 2")


DEFAULT_CONFIG = AlignmentConfig()


@dataclass(frozen=True)
class TokenSpan:
    token: str
    start: int
    end: int


@dataclass(frozen=True)
class TextSpan:
    text: str
    start: int
    end: int
    sentence_start: int
    sentence_end: int


@dataclass(frozen=True)
class CandidateAudit:
    text: str
    start: int
    end: int
    sentence_start: int
    sentence_end: int
    token_count: int
    sequence_similarity: float
    bag_f1: float
    source_recall: float
    candidate_precision: float
    length_ratio: float
    rank_score: float
    anchors_passed: bool
    anchor_differences: Mapping[str, Mapping[str, tuple[str, ...]]]


@dataclass(frozen=True)
class AlignmentResult:
    """Result and complete audit record for one support sentence."""

    matched: bool
    match_type: str | None
    failure_code: str | None
    source_text: str
    matched_text: str | None
    matched_start: int | None
    matched_end: int | None
    source_token_count: int
    sequence_similarity: float
    bag_f1: float
    source_recall: float
    candidate_precision: float
    length_ratio: float
    candidate_margin: float | None
    candidate_count: int
    length_eligible_candidate_count: int
    exact_occurrence_count: int
    protected_anchors: Mapping[str, tuple[str, ...]]
    matched_anchors: Mapping[str, tuple[str, ...]]
    anchor_differences: Mapping[str, Mapping[str, tuple[str, ...]]]
    best_candidate: CandidateAudit | None = None
    runner_up_candidate: CandidateAudit | None = None
    thresholds: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable audit dictionary."""

        return asdict(self)


_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
_HTML_ENTITY_RE = re.compile(r"&(?:#[0-9]+|#[xX][0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]+);")
_CLOSING_PUNCTUATION = set('"\'\u2019\u201d)]}')
_OPENING_PUNCTUATION = set('"\'\u2018\u201c([{')
_ABBREVIATIONS = {
    "adj",
    "adm",
    "adv",
    "al",
    "apr",
    "aug",
    "ave",
    "brig",
    "capt",
    "col",
    "dec",
    "dr",
    "e.g",
    "etc",
    "feb",
    "fig",
    "gen",
    "gov",
    "i.e",
    "inc",
    "jan",
    "jr",
    "jul",
    "jun",
    "lt",
    "mar",
    "mr",
    "mrs",
    "ms",
    "mt",
    "no",
    "nov",
    "oct",
    "prof",
    "rev",
    "sen",
    "sep",
    "sept",
    "sr",
    "st",
    "u.k",
    "u.s",
    "vs",
}

_MONTHS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
}

_NUMBER_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "million",
    "billion",
}

_NEGATIONS = {
    "no",
    "not",
    "never",
    "neither",
    "nor",
    "without",
    "cannot",
    "cant",
    "isnt",
    "wasnt",
    "werent",
    "didnt",
    "doesnt",
    "dont",
    "wont",
    "wouldnt",
    "couldnt",
    "shouldnt",
    "hasnt",
    "havent",
    "hadnt",
}

# Values encode direction, not wording, so a near-verbatim synonym can pass but
# a reversal (for example, earlier -> later) cannot.
_COMPARISON_DIRECTIONS = {
    "more": "quantity:+",
    "most": "quantity:+max",
    "greater": "magnitude:+",
    "higher": "magnitude:+",
    "highest": "magnitude:+max",
    "larger": "size:+",
    "largest": "size:+max",
    "longer": "length:+",
    "longest": "length:+max",
    "older": "age:+",
    "oldest": "age:+max",
    "after": "time:+",
    "later": "time:+",
    "latest": "time:+max",
    "above": "vertical:+",
    "over": "threshold:+",
    "exceeds": "threshold:+",
    "increased": "change:+",
    "increases": "change:+",
    "increase": "change:+",
    "less": "quantity:-",
    "least": "quantity:-max",
    "fewer": "quantity:-",
    "lower": "magnitude:-",
    "lowest": "magnitude:-max",
    "smaller": "size:-",
    "smallest": "size:-max",
    "shorter": "length:-",
    "shortest": "length:-max",
    "younger": "age:-",
    "youngest": "age:-max",
    "before": "time:-",
    "earlier": "time:-",
    "earliest": "time:-max",
    "below": "vertical:-",
    "under": "threshold:-",
    "decreased": "change:-",
    "decreases": "change:-",
    "decrease": "change:-",
    "first": "order:first",
    "last": "order:last",
    "same": "equality:same",
    "equal": "equality:same",
    "equally": "equality:same",
    "both": "equality:both",
}

_TIME_RE = re.compile(
    r"(?<!\w)(?P<hour>[0-2]?\d)(?:[.:](?P<minute>[0-5]\d))?\s*"
    r"(?P<period>a\.?m\.?|p\.?m\.?)\b",
    flags=re.IGNORECASE,
)
_NUMBER_RE = re.compile(
    r"(?<!\w)[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"(?:st|nd|rd|th)?%?",
    flags=re.IGNORECASE,
)


def _emit_normalized_character(
    output: list[str], origins: list[tuple[int, int]], value: str, start: int, end: int
) -> None:
    for character in value:
        normalized = unicodedata.normalize("NFKC", character).casefold()
        for item in normalized:
            output.append(item)
            origins.append((start, end))


def _normalize_with_origins(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Normalize text while retaining a conservative mapping to raw offsets."""

    output: list[str] = []
    origins: list[tuple[int, int]] = []
    cursor = 0
    for match in _HTML_ENTITY_RE.finditer(text):
        for index in range(cursor, match.start()):
            _emit_normalized_character(output, origins, text[index], index, index + 1)
        decoded = html.unescape(match.group(0))
        if decoded == match.group(0):
            decoded = match.group(0)
        _emit_normalized_character(output, origins, decoded, match.start(), match.end())
        cursor = match.end()
    for index in range(cursor, len(text)):
        _emit_normalized_character(output, origins, text[index], index, index + 1)
    return "".join(output), origins


def token_spans(text: str) -> tuple[TokenSpan, ...]:
    """Return canonical Unicode alphanumeric tokens and raw-text offsets."""

    normalized, origins = _normalize_with_origins(str(text))
    output: list[TokenSpan] = []
    for match in _TOKEN_RE.finditer(normalized):
        start_origin = origins[match.start()]
        end_origin = origins[match.end() - 1]
        output.append(TokenSpan(match.group(0), start_origin[0], end_origin[1]))
    return tuple(output)


def canonical_tokens(text: str) -> tuple[str, ...]:
    """Canonical tokens used by exact and fuzzy alignment."""

    return tuple(item.token for item in token_spans(text))


def canonical_text(text: str) -> str:
    """Space-joined canonical token representation for audit logs."""

    return " ".join(canonical_tokens(text))


def _is_abbreviation(text: str, period_index: int) -> bool:
    if 0 < period_index < len(text) - 1:
        if text[period_index - 1].isdigit() and text[period_index + 1].isdigit():
            return True
    prefix = text[max(0, period_index - 16) : period_index + 1]
    dotted = re.search(r"(?:\b[A-Za-z]\.){2,}$", prefix)
    if dotted:
        return True
    word_match = re.search(r"([A-Za-z]+)\.$", prefix)
    if not word_match:
        return False
    word = word_match.group(1).casefold()
    return len(word) == 1 or word in _ABBREVIATIONS


def sentence_spans(text: str) -> tuple[TextSpan, ...]:
    """Split a passage into auditable sentence-like raw spans.

    Newlines are boundaries.  Period, question-mark, and exclamation-mark
    boundaries are accepted when the next visible character looks like the
    start of a sentence.  Common abbreviations and decimal points are retained.
    The exact-token path is evaluated before this splitter, so harmless
    splitter mistakes cannot break a canonical exact match.
    """

    text = str(text)
    boundaries = {0, len(text)}
    index = 0
    while index < len(text):
        character = text[index]
        if character == "\n":
            end = index
            while index < len(text) and text[index] == "\n":
                index += 1
            boundaries.add(end)
            boundaries.add(index)
            continue
        if character not in ".?!":
            index += 1
            continue
        if character == "." and _is_abbreviation(text, index):
            index += 1
            continue
        end = index + 1
        while end < len(text) and text[end] in _CLOSING_PUNCTUATION:
            end += 1
        next_index = end
        while next_index < len(text) and text[next_index].isspace():
            next_index += 1
        visible_index = next_index
        while visible_index < len(text) and text[visible_index] in _OPENING_PUNCTUATION:
            visible_index += 1
        if visible_index >= len(text) or text[visible_index].isupper() or text[visible_index].isdigit():
            boundaries.add(end)
            boundaries.add(next_index)
        index = max(index + 1, end)

    ordered = sorted(boundaries)
    output: list[TextSpan] = []
    sentence_index = 0
    for start, end in zip(ordered, ordered[1:]):
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start >= end or not canonical_tokens(text[start:end]):
            continue
        output.append(TextSpan(text[start:end], start, end, sentence_index, sentence_index))
        sentence_index += 1
    return tuple(output)


def sentence_candidates(text: str, maximum_adjacent_sentences: int = 2) -> tuple[TextSpan, ...]:
    """Return single-sentence and adjacent-two-sentence candidates."""

    if maximum_adjacent_sentences not in (1, 2):
        raise ValueError("maximum_adjacent_sentences must be 1 or 2")
    singles = sentence_spans(text)
    output = list(singles)
    if maximum_adjacent_sentences == 2:
        for left, right in zip(singles, singles[1:]):
            output.append(
                TextSpan(
                    text=str(text)[left.start : right.end],
                    start=left.start,
                    end=right.end,
                    sentence_start=left.sentence_start,
                    sentence_end=right.sentence_end,
                )
            )
    return tuple(output)


def _normalized_for_anchors(text: str) -> str:
    normalized, _ = _normalize_with_origins(str(text))
    # Joining apostrophe contractions lets "didn't" and "didnt" share one
    # negation anchor.  Other punctuation remains available to date/time regexes.
    return re.sub(r"(?<=\w)[\u2019'](?=\w)", "", normalized)


def _normalize_time(match: re.Match[str]) -> str:
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    period = match.group("period").replace(".", "").casefold()
    return f"{hour:02d}:{minute:02d}{period}"


def _masked(text: str, spans: Iterable[tuple[int, int]]) -> str:
    characters = list(text)
    for start, end in spans:
        characters[start:end] = " " * (end - start)
    return "".join(characters)


def _anchor_binding_signature(
    *,
    category: str,
    value: str,
    start: int,
    end: int,
    token_matches: Sequence[re.Match[str]],
    radius: int = 5,
) -> str:
    """Bind an anchor value to its nearby canonical lexical context.

    Value order alone is insufficient: swapping Alice and Bob while leaving
    the year sequence ``1980, 1990`` unchanged would preserve the number list.
    A five-token context on each side catches that re-binding while remaining
    independent of the answer string and absolute character position.
    """

    overlapping = [
        index
        for index, match in enumerate(token_matches)
        if match.start() < end and start < match.end()
    ]
    if overlapping:
        first = overlapping[0]
        last = overlapping[-1]
    else:
        first = next(
            (index for index, match in enumerate(token_matches) if match.start() >= end),
            len(token_matches),
        )
        last = first - 1
    left = tuple(
        match.group(0) for match in token_matches[max(0, first - radius) : first]
    )
    right = tuple(
        match.group(0)
        for match in token_matches[last + 1 : min(len(token_matches), last + 1 + radius)]
    )
    return f"{category}={value}|left={' '.join(left)}|right={' '.join(right)}"


def extract_protected_anchors(text: str) -> dict[str, tuple[str, ...]]:
    """Extract answer-agnostic facts that a fuzzy match may not change."""

    normalized = _normalized_for_anchors(text)
    time_matches = list(_TIME_RE.finditer(normalized))
    # Preserve occurrence order.  A sorted multiset would miss a high-similarity
    # rewrite that rebinds two dates (for example, Alice:1980/Bob:1990 becoming
    # Alice:1990/Bob:1980).
    times = tuple(_normalize_time(match) for match in time_matches)
    without_times = _masked(normalized, ((match.start(), match.end()) for match in time_matches))
    number_matches = list(_NUMBER_RE.finditer(without_times))
    numbers = tuple(
        match.group(0).replace(",", "").casefold()
        for match in number_matches
    )
    token_matches = list(_TOKEN_RE.finditer(normalized))
    tokens = tuple(match.group(0) for match in token_matches)
    months = tuple(token for token in tokens if token in _MONTHS)
    number_words = tuple(token for token in tokens if token in _NUMBER_WORDS)
    negations = tuple(token for token in tokens if token in _NEGATIONS)
    comparisons = tuple(
        _COMPARISON_DIRECTIONS[token]
        for token in tokens
        if token in _COMPARISON_DIRECTIONS
    )
    events: list[tuple[int, int, str, str]] = []
    events.extend(
        (match.start(), match.end(), "times", _normalize_time(match))
        for match in time_matches
    )
    events.extend(
        (
            match.start(),
            match.end(),
            "numbers",
            match.group(0).replace(",", "").casefold(),
        )
        for match in number_matches
    )
    for match in token_matches:
        token = match.group(0)
        if token in _MONTHS:
            events.append((match.start(), match.end(), "months", token))
        if token in _NUMBER_WORDS:
            events.append((match.start(), match.end(), "number_words", token))
        if token in _NEGATIONS:
            events.append((match.start(), match.end(), "negations", token))
        if token in _COMPARISON_DIRECTIONS:
            events.append(
                (
                    match.start(),
                    match.end(),
                    "comparisons",
                    _COMPARISON_DIRECTIONS[token],
                )
            )
    bindings = tuple(
        _anchor_binding_signature(
            category=category,
            value=value,
            start=start,
            end=end,
            token_matches=token_matches,
        )
        for start, end, category, value in sorted(
            events, key=lambda item: (item[0], item[1], item[2], item[3])
        )
    )
    return {
        "numbers": numbers,
        "times": times,
        "months": months,
        "number_words": number_words,
        "negations": negations,
        "comparisons": comparisons,
        "bindings": bindings,
    }


def compare_anchors(
    source: Mapping[str, Sequence[str]], candidate: Mapping[str, Sequence[str]]
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Return changed protected-anchor categories and both observed values."""

    differences: dict[str, dict[str, tuple[str, ...]]] = {}
    for name in sorted(set(source) | set(candidate)):
        source_values = tuple(source.get(name, ()))
        candidate_values = tuple(candidate.get(name, ()))
        if source_values != candidate_values:
            differences[name] = {
                "source": source_values,
                "candidate": candidate_values,
            }
    return differences


def _similarity_metrics(
    source_tokens: Sequence[str], candidate_tokens: Sequence[str]
) -> tuple[float, float, float, float, float, float]:
    source_counter = Counter(source_tokens)
    candidate_counter = Counter(candidate_tokens)
    overlap = sum((source_counter & candidate_counter).values())
    source_recall = overlap / max(1, len(source_tokens))
    candidate_precision = overlap / max(1, len(candidate_tokens))
    bag_f1 = 2.0 * overlap / max(1, len(source_tokens) + len(candidate_tokens))
    sequence_similarity = difflib.SequenceMatcher(
        None, source_tokens, candidate_tokens, autojunk=False
    ).ratio()
    length_ratio = len(candidate_tokens) / max(1, len(source_tokens))
    rank_score = (sequence_similarity + bag_f1 + source_recall) / 3.0
    return (
        sequence_similarity,
        bag_f1,
        source_recall,
        candidate_precision,
        length_ratio,
        rank_score,
    )


def _candidate_audit(
    source_tokens: Sequence[str], source_anchors: Mapping[str, Sequence[str]], span: TextSpan
) -> CandidateAudit:
    candidate_tokens = canonical_tokens(span.text)
    (
        sequence_similarity,
        bag_f1,
        source_recall,
        candidate_precision,
        length_ratio,
        rank_score,
    ) = _similarity_metrics(source_tokens, candidate_tokens)
    candidate_anchors = extract_protected_anchors(span.text)
    differences = compare_anchors(source_anchors, candidate_anchors)
    return CandidateAudit(
        text=span.text,
        start=span.start,
        end=span.end,
        sentence_start=span.sentence_start,
        sentence_end=span.sentence_end,
        token_count=len(candidate_tokens),
        sequence_similarity=sequence_similarity,
        bag_f1=bag_f1,
        source_recall=source_recall,
        candidate_precision=candidate_precision,
        length_ratio=length_ratio,
        rank_score=rank_score,
        anchors_passed=not differences,
        anchor_differences=differences,
    )


def _find_exact_occurrences(
    source_tokens: Sequence[str], passage_token_spans: Sequence[TokenSpan]
) -> list[tuple[int, int]]:
    if not source_tokens or len(source_tokens) > len(passage_token_spans):
        return []
    passage_tokens = tuple(item.token for item in passage_token_spans)
    width = len(source_tokens)
    return [
        (index, index + width)
        for index in range(len(passage_tokens) - width + 1)
        if passage_tokens[index : index + width] == tuple(source_tokens)
    ]


def _overlaps(left: CandidateAudit, right: CandidateAudit) -> bool:
    return left.start < right.end and right.start < left.end


def _threshold_dict(config: AlignmentConfig) -> dict[str, Any]:
    return asdict(config)


def _result_from_candidate(
    *,
    matched: bool,
    failure_code: str | None,
    source_text: str,
    source_tokens: Sequence[str],
    source_anchors: Mapping[str, tuple[str, ...]],
    best: CandidateAudit | None,
    runner_up: CandidateAudit | None,
    margin: float | None,
    candidate_count: int,
    length_eligible_count: int,
    config: AlignmentConfig,
) -> AlignmentResult:
    matched_anchors = extract_protected_anchors(best.text) if best else {}
    return AlignmentResult(
        matched=matched,
        match_type=MATCH_FUZZY_LOCAL if matched else None,
        failure_code=failure_code,
        source_text=source_text,
        matched_text=best.text if best else None,
        matched_start=best.start if best else None,
        matched_end=best.end if best else None,
        source_token_count=len(source_tokens),
        sequence_similarity=best.sequence_similarity if best else 0.0,
        bag_f1=best.bag_f1 if best else 0.0,
        source_recall=best.source_recall if best else 0.0,
        candidate_precision=best.candidate_precision if best else 0.0,
        length_ratio=best.length_ratio if best else 0.0,
        candidate_margin=margin,
        candidate_count=candidate_count,
        length_eligible_candidate_count=length_eligible_count,
        exact_occurrence_count=0,
        protected_anchors=source_anchors,
        matched_anchors=matched_anchors,
        anchor_differences=best.anchor_differences if best else {},
        best_candidate=best,
        runner_up_candidate=runner_up,
        thresholds=_threshold_dict(config),
    )


def align_support_sentence(
    source_sentence: str,
    longbench_passage: str,
    config: AlignmentConfig = DEFAULT_CONFIG,
) -> AlignmentResult:
    """Align one annotated support sentence to one LongBench passage.

    Args:
        source_sentence: The HotpotQA sentence selected by ``(title, sent_id)``.
        longbench_passage: The body of the exact-title LongBench passage.
        config: Frozen lexical, length, uniqueness, and candidate-span gates.

    Returns:
        :class:`AlignmentResult`.  A rejected result still contains the best
        local candidate, scores, protected-anchor differences, and failure code.
    """

    source_sentence = str(source_sentence)
    longbench_passage = str(longbench_passage)
    source_tokens = canonical_tokens(source_sentence)
    source_anchors = extract_protected_anchors(source_sentence)
    thresholds = _threshold_dict(config)

    if not source_tokens:
        return AlignmentResult(
            matched=False,
            match_type=None,
            failure_code=FAIL_EMPTY_SOURCE,
            source_text=source_sentence,
            matched_text=None,
            matched_start=None,
            matched_end=None,
            source_token_count=0,
            sequence_similarity=0.0,
            bag_f1=0.0,
            source_recall=0.0,
            candidate_precision=0.0,
            length_ratio=0.0,
            candidate_margin=None,
            candidate_count=0,
            length_eligible_candidate_count=0,
            exact_occurrence_count=0,
            protected_anchors=source_anchors,
            matched_anchors={},
            anchor_differences={},
            thresholds=thresholds,
        )

    passage_tokens = token_spans(longbench_passage)
    if not passage_tokens:
        return AlignmentResult(
            matched=False,
            match_type=None,
            failure_code=FAIL_EMPTY_PASSAGE,
            source_text=source_sentence,
            matched_text=None,
            matched_start=None,
            matched_end=None,
            source_token_count=len(source_tokens),
            sequence_similarity=0.0,
            bag_f1=0.0,
            source_recall=0.0,
            candidate_precision=0.0,
            length_ratio=0.0,
            candidate_margin=None,
            candidate_count=0,
            length_eligible_candidate_count=0,
            exact_occurrence_count=0,
            protected_anchors=source_anchors,
            matched_anchors={},
            anchor_differences={},
            thresholds=thresholds,
        )

    exact_occurrences = _find_exact_occurrences(source_tokens, passage_tokens)
    if exact_occurrences:
        token_start, token_end = exact_occurrences[0]
        raw_start = passage_tokens[token_start].start
        raw_end = passage_tokens[token_end - 1].end
        matched_text = longbench_passage[raw_start:raw_end]
        matched_anchors = extract_protected_anchors(matched_text)
        return AlignmentResult(
            matched=True,
            match_type=MATCH_CANONICAL_EXACT,
            failure_code=None,
            source_text=source_sentence,
            matched_text=matched_text,
            matched_start=raw_start,
            matched_end=raw_end,
            source_token_count=len(source_tokens),
            sequence_similarity=1.0,
            bag_f1=1.0,
            source_recall=1.0,
            candidate_precision=1.0,
            length_ratio=1.0,
            candidate_margin=1.0,
            candidate_count=0,
            length_eligible_candidate_count=0,
            exact_occurrence_count=len(exact_occurrences),
            protected_anchors=source_anchors,
            matched_anchors=matched_anchors,
            anchor_differences=compare_anchors(source_anchors, matched_anchors),
            thresholds=thresholds,
        )

    spans = sentence_candidates(longbench_passage, config.maximum_adjacent_sentences)
    if not spans:
        return _result_from_candidate(
            matched=False,
            failure_code=FAIL_NO_SENTENCE_CANDIDATE,
            source_text=source_sentence,
            source_tokens=source_tokens,
            source_anchors=source_anchors,
            best=None,
            runner_up=None,
            margin=None,
            candidate_count=0,
            length_eligible_count=0,
            config=config,
        )

    audits = [_candidate_audit(source_tokens, source_anchors, span) for span in spans]
    length_eligible = [
        item
        for item in audits
        if config.minimum_length_ratio <= item.length_ratio <= config.maximum_length_ratio
    ]
    if not length_eligible:
        best_any = max(audits, key=lambda item: item.rank_score)
        return _result_from_candidate(
            matched=False,
            failure_code=FAIL_NO_LENGTH_ELIGIBLE_CANDIDATE,
            source_text=source_sentence,
            source_tokens=source_tokens,
            source_anchors=source_anchors,
            best=best_any,
            runner_up=None,
            margin=None,
            candidate_count=len(audits),
            length_eligible_count=0,
            config=config,
        )

    ranked = sorted(
        length_eligible,
        key=lambda item: (
            item.rank_score,
            item.sequence_similarity,
            item.bag_f1,
            -item.start,
            -item.end,
        ),
        reverse=True,
    )
    best = ranked[0]
    runner_up = next((item for item in ranked[1:] if not _overlaps(best, item)), None)
    margin = best.rank_score - runner_up.rank_score if runner_up else 1.0

    similarity_passed = (
        best.sequence_similarity >= config.minimum_sequence_similarity
        and best.bag_f1 >= config.minimum_bag_f1
        and best.source_recall >= config.minimum_source_recall
    )
    if not similarity_passed:
        failure_code = FAIL_SIMILARITY_BELOW_THRESHOLD
    elif not best.anchors_passed:
        failure_code = FAIL_PROTECTED_ANCHOR_CHANGED
    elif margin < config.minimum_candidate_margin:
        failure_code = FAIL_CANDIDATE_AMBIGUOUS
    else:
        failure_code = None

    return _result_from_candidate(
        matched=failure_code is None,
        failure_code=failure_code,
        source_text=source_sentence,
        source_tokens=source_tokens,
        source_anchors=source_anchors,
        best=best,
        runner_up=runner_up,
        margin=margin,
        candidate_count=len(audits),
        length_eligible_count=len(length_eligible),
        config=config,
    )


def align_support_sentences(
    source_sentences: Sequence[str],
    longbench_passage: str,
    config: AlignmentConfig = DEFAULT_CONFIG,
) -> tuple[AlignmentResult, ...]:
    """Convenience wrapper that preserves source-sentence order."""

    return tuple(
        align_support_sentence(source_sentence, longbench_passage, config)
        for source_sentence in source_sentences
    )


__all__ = [
    "AlignmentConfig",
    "AlignmentResult",
    "CandidateAudit",
    "DEFAULT_CONFIG",
    "FAIL_CANDIDATE_AMBIGUOUS",
    "FAIL_EMPTY_PASSAGE",
    "FAIL_EMPTY_SOURCE",
    "FAIL_NO_LENGTH_ELIGIBLE_CANDIDATE",
    "FAIL_NO_SENTENCE_CANDIDATE",
    "FAIL_PROTECTED_ANCHOR_CHANGED",
    "FAIL_SIMILARITY_BELOW_THRESHOLD",
    "MATCH_CANONICAL_EXACT",
    "MATCH_FUZZY_LOCAL",
    "align_support_sentence",
    "align_support_sentences",
    "canonical_text",
    "canonical_tokens",
    "compare_anchors",
    "extract_protected_anchors",
    "sentence_candidates",
    "sentence_spans",
    "token_spans",
]
