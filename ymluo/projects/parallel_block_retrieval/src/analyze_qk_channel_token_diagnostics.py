from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from rerank_sparse_candidate_blocks_svd import rank_ids, target_rank


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "by",
    "does",
    "for",
    "from",
    "in",
    "is",
    "of",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose per-channel QK retrieval and winning query-token types."
    )
    parser.add_argument("--profile_dir", required=True)
    parser.add_argument("--rows_path", required=True)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def token_category(text: str) -> str:
    terms = re.findall(r"[a-z0-9]+", text.casefold())
    if not terms:
        return "punctuation"
    if all(term in STOPWORDS for term in terms):
        return "stopword"
    return "content"


def profile_ranks(row: dict[str, Any], method: str) -> list[int]:
    candidates = [int(item) for item in row["candidate_candidates"]]
    matrix = row[f"{method}_profile_scores"]
    profile_count = len(matrix[0])
    target = int(row["target_block_id"])
    return [
        target_rank(
            rank_ids(candidates, [float(values[profile]) for values in matrix]),
            target,
        )
        for profile in range(profile_count)
    ]


def select_profile(rows: list[dict[str, Any]], method: str, count: int) -> int:
    scores = []
    for profile in range(count):
        ranks = [profile_ranks(row, method)[profile] for row in rows]
        recall = statistics.fmean(0 < rank <= 3 for rank in ranks)
        reachable = [rank for rank in ranks if rank > 0]
        mrr = statistics.fmean(1.0 / rank for rank in reachable) if reachable else 0.0
        scores.append((recall, mrr, -profile))
    return max(range(count), key=lambda profile: scores[profile])


def category_for_candidate(
    row: dict[str, Any], method: str, candidate_id: int
) -> str:
    candidates = [int(item) for item in row["candidate_candidates"]]
    candidate_index = candidates.index(candidate_id)
    profile_scores = row[f"{method}_profile_scores"][candidate_index]
    profile = max(
        range(len(profile_scores)), key=lambda index: (profile_scores[index], -index)
    )
    query_position = int(
        row[f"{method}_winning_query_positions"][candidate_index][profile]
    )
    return token_category(str(row["query_token_texts"][query_position]))


def category_rates(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    return {
        category: counter[category] / total if total else 0.0
        for category in ("content", "stopword", "punctuation")
    }


def evaluate_group(
    rows: list[dict[str, Any]], method: str, selected_profile: int
) -> dict[str, Any]:
    ranks_by_row = [profile_ranks(row, method) for row in rows]
    selected_ranks = [ranks[selected_profile] for ranks in ranks_by_row]
    top_categories: Counter[str] = Counter()
    target_categories: Counter[str] = Counter()
    for row in rows:
        top_id = int(row[f"{method}_candidates"][0])
        top_categories[category_for_candidate(row, method, top_id)] += 1
        target = int(row["target_block_id"])
        if target in row["candidate_candidates"]:
            target_categories[category_for_candidate(row, method, target)] += 1
    return {
        "steps": len(rows),
        "selected_profile": selected_profile,
        "max_over_profiles_recall_at_3": statistics.fmean(
            0 < int(row[f"{method}_rank"]) <= 3 for row in rows
        ),
        "selected_profile_recall_at_3": statistics.fmean(
            0 < rank <= 3 for rank in selected_ranks
        ),
        "oracle_any_profile_recall_at_3": statistics.fmean(
            any(0 < rank <= 3 for rank in ranks) for ranks in ranks_by_row
        ),
        "top1_winning_token_categories": category_rates(top_categories),
        "reachable_target_winning_token_categories": category_rates(target_categories),
    }


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.rows_path))
    profile_summary = json.loads(
        (Path(args.profile_dir) / "summary.json").read_text(encoding="utf-8")
    )
    pair_specs = [dict(item) for item in profile_summary["pair_specs"]]
    profile_count = len(pair_specs)
    payload: dict[str, Any] = {
        "source": "train-selected QK channel and winning query-token diagnostic",
        "selection_uses_gold": False,
        "train_labels_used_for_profile_selection_only": True,
        "pair_specs": pair_specs,
        "methods": {},
    }
    step_types = sorted({str(row["step_type"]) for row in rows})
    for method in ("full128", "svd"):
        method_payload: dict[str, Any] = {}
        for step_type in step_types:
            train = [
                row
                for row in rows
                if str(row["split"]) == "train"
                and str(row["step_type"]) == step_type
            ]
            selected = select_profile(train, method, profile_count)
            evaluations = {}
            for split in ("train", "dev", "test"):
                group = [
                    row
                    for row in rows
                    if str(row["split"]) == split
                    and str(row["step_type"]) == step_type
                ]
                evaluations[split] = evaluate_group(group, method, selected)
            method_payload[step_type] = {
                "selected_profile": selected,
                "selected_pair": pair_specs[selected],
                "evaluations": evaluations,
            }
        payload["methods"][method] = method_payload
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
