from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure whether step-generation failures copy co-located snippets."
    )
    parser.add_argument("--rows_path", required=True)
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = f" {normalized(text)} "
    normalized_phrase = normalized(phrase)
    return bool(normalized_phrase) and f" {normalized_phrase} " in normalized_text


def main() -> None:
    args = parse_args()
    corpus_dir = Path(args.corpus_dir)
    rows = read_jsonl(Path(args.rows_path))
    queries = read_jsonl(corpus_dir / "queries.jsonl")
    blocks = read_jsonl(corpus_dir / "blocks.jsonl")
    query_by_id = {int(row["query_id"]): row for row in queries}
    block_by_id = {int(row["block_id"]): row for row in blocks}

    details = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        query_id = int(row["query_id"])
        query = query_by_id[query_id]
        co_located_query_ids = sorted(
            {
                int(other_query_id)
                for block_id in row["block_ids"]
                for other_query_id in block_by_id[int(block_id)]["synthetic_query_ids"]
                if int(other_query_id) != query_id
            }
        )
        distractor_entities = [
            str(query_by_id[item]["entity"])
            for item in co_located_query_ids
            if item in query_by_id
        ]
        distractor_answers = [
            str(answer)
            for item in co_located_query_ids
            if item in query_by_id
            for answer in query_by_id[item]["answers"]
        ]
        alias_match = re.search(r"\b[A-Z][A-Za-z]+-\d{4}\b", str(query["question"]))
        lookup_key = alias_match.group(0) if alias_match else ""
        generated = str(row["generated_text"])
        detail = {
            "query_id": query_id,
            "step_index": int(row["step_index"]),
            "step_type": str(row["step_type"]),
            "mode": str(row["mode"]),
            "target_hit": bool(row["target_hit"]),
            "block_ids": [int(item) for item in row["block_ids"]],
            "selected_snippets": sum(
                len(block_by_id[int(block_id)]["synthetic_query_ids"])
                for block_id in row["block_ids"]
            ),
            "co_located_queries": len(co_located_query_ids),
            "repeats_lookup_key": contains_phrase(generated, lookup_key),
            "matches_distractor_entity": any(
                contains_phrase(generated, item) for item in distractor_entities
            ),
            "matches_distractor_answer": any(
                contains_phrase(generated, item) for item in distractor_answers
            ),
            "generated_text": generated,
        }
        details.append(detail)
        grouped[(detail["step_type"], detail["mode"])].append(detail)

    summaries = []
    for (step_type, mode), group in sorted(grouped.items()):
        count = len(group)
        failures = [row for row in group if not row["target_hit"]]
        denominator = max(1, len(failures))
        summaries.append(
            {
                "step_type": step_type,
                "mode": mode,
                "rows": count,
                "failures": len(failures),
                "mean_selected_snippets": sum(
                    row["selected_snippets"] for row in group
                )
                / count,
                "failure_lookup_repeat_rate": sum(
                    row["repeats_lookup_key"] for row in failures
                )
                / denominator,
                "failure_distractor_entity_rate": sum(
                    row["matches_distractor_entity"] for row in failures
                )
                / denominator,
                "failure_distractor_answer_rate": sum(
                    row["matches_distractor_answer"] for row in failures
                )
                / denominator,
            }
        )
    payload = {
        "source": str(args.rows_path),
        "corpus_dir": str(corpus_dir),
        "summaries": summaries,
        "failure_details": [row for row in details if not row["target_hit"]],
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
