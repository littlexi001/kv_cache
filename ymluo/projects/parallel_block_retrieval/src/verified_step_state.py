from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class AtomicFact:
    subject: str
    relation: str
    object: str
    evidence: str = ""


@dataclass(frozen=True)
class StepAction:
    facts: tuple[AtomicFact, ...]
    kind: str
    value: str


def normalize_span(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def parse_step_action(text: str) -> StepAction:
    facts = []
    for match in re.finditer(
        r"(?:^|\n)\s*FACT\s*:\s*([^\n]+)",
        text,
        flags=re.IGNORECASE,
    ):
        parts = [
            item.strip().strip("`*\"' .")
            for item in match.group(1).split("|", maxsplit=3)
        ]
        if len(parts) in {3, 4} and all(parts[:3]):
            facts.append(AtomicFact(*parts))
    actions = []
    for kind in ("FINAL", "SEARCH"):
        match = re.search(
            rf"(?:^|\n)\s*(?:ACTION\s*:\s*)?{kind}\s*:\s*(.+)",
            text,
            flags=re.IGNORECASE,
        )
        if match is not None:
            value = match.group(1).splitlines()[0].strip().strip("`*\"' .")
            if value:
                actions.append((kind.casefold(), value))
    if len(actions) != 1:
        raise ValueError(f"expected exactly one FINAL or SEARCH action: {text!r}")
    kind, value = actions[0]
    return StepAction(tuple(facts), kind, value)


def fact_supported(
    fact: AtomicFact,
    memory: str,
    *,
    max_entity_distance: int = 800,
    require_evidence: bool = False,
) -> bool:
    subject = normalize_span(fact.subject)
    object_text = normalize_span(fact.object)
    if not subject or not object_text:
        return False
    for block_text in re.split(r"\n\s*\n", memory):
        normalized_block = normalize_span(block_text)
        if require_evidence:
            evidence = normalize_span(fact.evidence)
            if (
                not evidence
                or evidence not in normalized_block
                or subject not in evidence
                or object_text not in evidence
            ):
                continue
        subject_positions = [
            item.start() for item in re.finditer(re.escape(subject), normalized_block)
        ]
        object_positions = [
            item.start() for item in re.finditer(re.escape(object_text), normalized_block)
        ]
        if any(
            abs(subject_position - object_position) <= max_entity_distance
            for subject_position in subject_positions
            for object_position in object_positions
        ):
            return True
    return False


def verified_facts(
    action: StepAction,
    memory: str,
    *,
    require_evidence: bool = False,
) -> tuple[AtomicFact, ...]:
    return tuple(
        fact
        for fact in action.facts
        if fact_supported(fact, memory, require_evidence=require_evidence)
    )


def fact_chain_connects(
    facts: Sequence[AtomicFact],
    final_value_text: str,
    question: str,
) -> bool:
    final_value = normalize_span(final_value_text)
    if not final_value or not facts:
        return False
    adjacency: dict[str, set[str]] = {}
    entity_text: dict[str, str] = {}
    for fact in facts:
        subject = normalize_span(fact.subject)
        object_text = normalize_span(fact.object)
        adjacency.setdefault(subject, set()).add(object_text)
        adjacency.setdefault(object_text, set()).add(subject)
        entity_text[subject] = fact.subject
        entity_text[object_text] = fact.object
    final_nodes = {
        node
        for node in adjacency
        if final_value == node or final_value in node or node in final_value
    }
    question_normalized = normalize_span(question)
    frontier = [
        node
        for node, original in entity_text.items()
        if normalize_span(original) in question_normalized
    ]
    visited = set(frontier)
    while frontier:
        node = frontier.pop()
        if node in final_nodes:
            return True
        for neighbor in adjacency.get(node, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
    return False


def final_action_verified(
    action: StepAction,
    memory: str,
    question: str,
) -> bool:
    if action.kind != "final" or not action.facts:
        return False
    facts = verified_facts(action, memory)
    if len(facts) != len(action.facts):
        return False
    final_value = normalize_span(action.value)
    if not final_value or final_value not in normalize_span(memory):
        return False
    return fact_chain_connects(facts, action.value, question)


def verified_step_prompt(
    memory: str,
    question: str,
    previous_queries: Sequence[str],
    fact_ledger: Sequence[AtomicFact] = (),
) -> str:
    previous_text = (
        "\nPrevious lookup queries: " + " | ".join(previous_queries)
        if previous_queries
        else ""
    )
    ledger_text = ""
    if fact_ledger:
        ledger_lines = "\n".join(
            f"FACT: {fact.subject} | {fact.relation} | {fact.object}"
            for fact in fact_ledger
        )
        ledger_text = (
            "\nPreviously verified compact state (trusted; do not invent additions):\n"
            f"{ledger_lines}\n"
        )
    return (
        "Memory:\n"
        f"{memory}\n\n"
        f"Question: {question}"
        f"{previous_text}\n"
        f"{ledger_text}"
        "Write one or more atomic evidence facts using exact entity spans from Memory, "
        "one per line as `FACT: <subject> | <relation> | <object> | <exact short quote "
        "from Memory containing both entities>`. Then output exactly "
        "one action. Use `FINAL: <concise answer>` only if those facts form a complete "
        "chain from an entity in the Question to the answer and every fact is directly "
        "supported by Memory. Otherwise use `SEARCH: <exact bridge entity> | "
        "<still-unresolved relation>`. Do not explain and do not invent missing facts."
    )
