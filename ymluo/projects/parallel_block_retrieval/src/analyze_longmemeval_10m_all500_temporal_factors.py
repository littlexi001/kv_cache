from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from analyze_longmemeval_10m_temporal_reader_ablation import (
    METHODS,
    compare,
    quality,
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze date and causal-order factors on all 500 LongMemEval queries."
    )
    parser.add_argument("--baseline_pattern", required=True)
    parser.add_argument("--dates_only_pattern", required=True)
    parser.add_argument("--order_only_pattern", required=True)
    parser.add_argument("--dates_order_pattern", required=True)
    parser.add_argument("--dates_reverse_pattern", required=True)
    parser.add_argument("--partitions", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patterns = {
        "baseline_no_dates_retrieval_order": args.baseline_pattern,
        "dates_only_retrieval_order": args.dates_only_pattern,
        "order_only_old_to_new": args.order_only_pattern,
        "dates_and_old_to_new": args.dates_order_pattern,
        "dates_and_new_to_old": args.dates_reverse_pattern,
    }
    configs: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        name: defaultdict(dict) for name in patterns
    }
    for name, pattern in patterns.items():
        for partition in range(args.partitions):
            reader_dir = Path(pattern.format(partition=partition))
            for row in read_jsonl(reader_dir / "rows.jsonl"):
                if row["method"] in METHODS:
                    configs[name][str(row["method"])][str(row["question_id"])] = row
    ids = sorted(configs["baseline_no_dates_retrieval_order"][METHODS[0]])
    if len(ids) != 500:
        raise RuntimeError("expected all 500 questions")
    for config in configs.values():
        for method in METHODS:
            if set(config[method]) != set(ids):
                raise RuntimeError("all reader configurations must cover the same questions")

    question_types = sorted(
        {
            configs["baseline_no_dates_retrieval_order"][METHODS[0]][qid][
                "question_type"
            ]
            for qid in ids
        }
    )
    output = {
        "protocol": {
            "memory_scope": "eight independent real 10M-token shards",
            "queries": len(ids),
            "same_selected_page_ids_within_method": True,
            "selection_uses_answer": False,
        },
        "quality": {
            name: {
                method: quality(configs[name][method], ids) for method in METHODS
            }
            for name in configs
        },
        "versus_baseline": {
            name: {
                method: compare(
                    configs["baseline_no_dates_retrieval_order"][method],
                    configs[name][method],
                    ids,
                    samples=args.bootstrap_samples,
                    seed=args.seed + config_index * 20 + method_index * 5,
                )
                for method_index, method in enumerate(METHODS)
            }
            for config_index, name in enumerate(patterns)
            if name != "baseline_no_dates_retrieval_order"
        },
        "factor_contrasts": {
            "date_effect_given_old_to_new": {
                method: compare(
                    configs["order_only_old_to_new"][method],
                    configs["dates_and_old_to_new"][method],
                    ids,
                    samples=args.bootstrap_samples,
                    seed=args.seed + 200 + method_index * 5,
                )
                for method_index, method in enumerate(METHODS)
            },
            "order_effect_given_dates": {
                method: compare(
                    configs["dates_only_retrieval_order"][method],
                    configs["dates_and_old_to_new"][method],
                    ids,
                    samples=args.bootstrap_samples,
                    seed=args.seed + 220 + method_index * 5,
                )
                for method_index, method in enumerate(METHODS)
            },
            "old_to_new_vs_new_to_old_with_dates": {
                method: compare(
                    configs["dates_and_new_to_old"][method],
                    configs["dates_and_old_to_new"][method],
                    ids,
                    samples=args.bootstrap_samples,
                    seed=args.seed + 240 + method_index * 5,
                )
                for method_index, method in enumerate(METHODS)
            },
        },
        "by_question_type": {
            question_type: {
                "queries": len(type_ids),
                "versus_baseline": {
                    name: {
                        method: compare(
                            configs["baseline_no_dates_retrieval_order"][method],
                            configs[name][method],
                            type_ids,
                            samples=args.bootstrap_samples,
                            seed=(
                                args.seed
                                + 400
                                + type_index * 100
                                + config_index * 20
                                + method_index * 5
                            ),
                        )
                        for method_index, method in enumerate(METHODS)
                    }
                    for config_index, name in enumerate(patterns)
                    if name != "baseline_no_dates_retrieval_order"
                },
            }
            for type_index, question_type in enumerate(question_types)
            for type_ids in [
                [
                    qid
                    for qid in ids
                    if configs["baseline_no_dates_retrieval_order"][METHODS[0]][qid][
                        "question_type"
                    ]
                    == question_type
                ]
            ]
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
