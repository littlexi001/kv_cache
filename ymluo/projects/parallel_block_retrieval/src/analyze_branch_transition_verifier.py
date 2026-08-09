from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

from analyze_stepwise_set_utility import mcnemar_exact_p


STOPWORDS = {
    "a",
    "an",
    "and",
    "entity",
    "for",
    "in",
    "is",
    "linked",
    "memory",
    "of",
    "on",
    "only",
    "the",
    "this",
    "to",
    "with",
}

RELATION_MARKERS = {
    "resolve_director": ("directed by", "director"),
    "resolve_spouse": ("married", "spouse", "wife", "husband"),
    "resolve_father": ("father", "daughter of", "son of"),
    "resolve_death_date": ("died", "death"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select parallel retrieval branches with state-anchor grounding."
    )
    parser.add_argument("--rows_path", required=True)
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = " " + " ".join(re.findall(r"[a-z0-9]+", text.casefold())) + " "
    normalized_phrase = " ".join(re.findall(r"[a-z0-9]+", phrase.casefold()))
    return bool(normalized_phrase) and f" {normalized_phrase} " in normalized_text


def state_anchor(step: dict[str, Any]) -> str:
    if step["step_type"] == "resolve_bridge":
        return str(step["lookup_key"])
    compact_state = [str(item) for item in step["compact_state_before"]]
    if not compact_state or ":" not in compact_state[0]:
        raise ValueError("answer step is missing a typed bridge state")
    return compact_state[0].split(":", maxsplit=1)[1].strip()


def transition_grounding_score(
    step: dict[str, Any], branch: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    anchor = state_anchor(step)
    memory = str(branch["memory_text"])
    generated = str(branch["generated_text"])
    state_terms = terms(str(step["question"]) + " " + " ".join(step["compact_state_before"]))
    memory_terms = terms(memory)
    generated_terms = terms(generated)
    step_query_terms = terms(
        str(step.get("step_question", "")) + " " + str(step.get("question", ""))
    ) - STOPWORDS
    operator = str(step.get("step_operator", ""))
    relation_markers = RELATION_MARKERS.get(operator, ())
    typed_relation = bool(relation_markers)
    anchor_aliases = [anchor]
    anchor_terms = re.findall(r"[a-z0-9]+", anchor.casefold())
    if (
        typed_relation
        and step["step_type"] == "resolve_answer_from_bridge"
        and len(anchor_terms) >= 2
        and len(anchor_terms[-1]) >= 4
    ):
        anchor_aliases.append(anchor_terms[-1])
    anchor_present = any(contains_phrase(memory, alias) for alias in anchor_aliases)
    repeats_anchor = contains_phrase(generated, anchor)
    relation_supported = any(
        contains_phrase(memory, marker) for marker in relation_markers
    )
    output_grounded = contains_phrase(memory, generated)
    query_memory_overlap = len(step_query_terms & memory_terms)
    novel_grounded_terms = (generated_terms & memory_terms) - state_terms - STOPWORDS
    grounding_ratio = len(generated_terms & memory_terms) / max(1, len(generated_terms))
    score = (
        10.0 * float(anchor_present)
        - 10.0
        * float(
            repeats_anchor
            and (
                step["step_type"] == "resolve_bridge"
                or not typed_relation
            )
        )
        + 10.0 * float(relation_supported)
        + 12.0 * float(output_grounded)
        + 0.25 * query_memory_overlap
        + len(novel_grounded_terms)
        + grounding_ratio
    )
    return score, {
        "anchor": anchor,
        "anchor_present": anchor_present,
        "repeats_anchor": repeats_anchor,
        "relation_supported": relation_supported,
        "output_grounded": output_grounded,
        "query_memory_overlap": query_memory_overlap,
        "step_operator": operator,
        "novel_grounded_terms": sorted(novel_grounded_terms),
        "grounding_ratio": grounding_ratio,
    }


def choose_branch(
    step: dict[str, Any], branches: Sequence[dict[str, Any]]
) -> tuple[int, list[dict[str, Any]]]:
    scored = []
    for index, branch in enumerate(branches):
        score, trace = transition_grounding_score(step, branch)
        scored.append({"branch_index": index, "score": score, **trace})
    selected = max(range(len(scored)), key=lambda index: (scored[index]["score"], -index))
    return selected, scored


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.rows_path))
    steps = read_jsonl(Path(args.step_queries_path))
    step_by_key = {
        (int(step["query_id"]), int(step["step_index"])): step for step in steps
    }
    details = []
    for row in rows:
        step = step_by_key[(int(row["query_id"]), int(row["step_index"]))]
        selected, scores = choose_branch(step, row["branches"])
        branch = row["branches"][selected]
        details.append(
            {
                "query_id": int(row["query_id"]),
                "step_index": int(row["step_index"]),
                "step_type": str(row["step_type"]),
                "top1_hit": bool(row["target_hit"]),
                "verifier_selected_index": selected,
                "verifier_hit": bool(branch["target_hit"]),
                "oracle_any_branch_hit": bool(row["any_branch_target_hit"]),
                "score_trace": scores,
            }
        )
    summaries = []
    for step_type in sorted({row["step_type"] for row in details}):
        group = [row for row in details if row["step_type"] == step_type]
        wins = sum(row["verifier_hit"] and not row["top1_hit"] for row in group)
        losses = sum(row["top1_hit"] and not row["verifier_hit"] for row in group)
        summaries.append(
            {
                "step_type": step_type,
                "steps": len(group),
                "top1_hit_rate": sum(row["top1_hit"] for row in group) / len(group),
                "verifier_hit_rate": sum(row["verifier_hit"] for row in group) / len(group),
                "oracle_any_branch_hit_rate": sum(
                    row["oracle_any_branch_hit"] for row in group
                )
                / len(group),
                "verifier_wins_losses_ties": [
                    wins,
                    losses,
                    len(group) - wins - losses,
                ],
                "mcnemar_exact_p": mcnemar_exact_p(wins, losses),
            }
        )
    payload = {
        "source": str(args.rows_path),
        "selection_uses_gold": False,
        "oracle_metric_uses_gold_for_evaluation_only": True,
        "summaries": summaries,
        "details": details,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
