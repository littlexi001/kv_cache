from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest


METHODS = ("static_top12", "evidence_state_dynamic_top12")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze date-label and page-order LongMemEval reader ablations."
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def paired_continuous(
    baseline: dict[str, dict[str, Any]],
    treatment: dict[str, dict[str, Any]],
    ids: list[str],
    metric: str,
    *,
    samples: int,
    seed: int,
    lower_is_better: bool,
) -> dict[str, Any]:
    base = np.asarray([float(baseline[qid][metric]) for qid in ids])
    new = np.asarray([float(treatment[qid][metric]) for qid in ids])
    difference = new - base
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(samples, len(ids)))
    sampled = difference[draws].mean(axis=1)
    wins = difference < 0 if lower_is_better else difference > 0
    losses = difference > 0 if lower_is_better else difference < 0
    output = {
        "queries": len(ids),
        "baseline_mean": float(base.mean()),
        "treatment_mean": float(new.mean()),
        "treatment_minus_baseline": float(difference.mean()),
        "bootstrap_95_ci": [
            float(np.quantile(sampled, 0.025)),
            float(np.quantile(sampled, 0.975)),
        ],
        "wins": int(wins.sum()),
        "losses": int(losses.sum()),
        "ties": int((difference == 0).sum()),
    }
    if metric == "reference_nll":
        output["perplexity_ratio_exp_mean_delta"] = math.exp(
            float(difference.mean())
        )
    return output


def paired_binary(
    baseline: dict[str, dict[str, Any]],
    treatment: dict[str, dict[str, Any]],
    ids: list[str],
    metric: str,
) -> dict[str, Any]:
    base = np.asarray([bool(baseline[qid][metric]) for qid in ids])
    new = np.asarray([bool(treatment[qid][metric]) for qid in ids])
    wins = int((~base & new).sum())
    losses = int((base & ~new).sum())
    return {
        "queries": len(ids),
        "baseline_rate": float(base.mean()),
        "treatment_rate": float(new.mean()),
        "delta": float(new.mean() - base.mean()),
        "wins": wins,
        "losses": losses,
        "two_sided_binomial_p": float(
            binomtest(wins, wins + losses, 0.5).pvalue
        )
        if wins + losses
        else 1.0,
    }


def quality(rows: dict[str, dict[str, Any]], ids: list[str]) -> dict[str, Any]:
    return {
        "queries": len(ids),
        "mean_reference_nll": float(
            np.mean([rows[qid]["reference_nll"] for qid in ids])
        ),
        "mean_token_f1": float(np.mean([rows[qid]["token_f1"] for qid in ids])),
        "exact_match": float(
            np.mean([rows[qid]["normalized_exact_match"] for qid in ids])
        ),
        "answer_contains": float(
            np.mean([rows[qid]["answer_contains"] for qid in ids])
        ),
    }


def compare(
    baseline: dict[str, dict[str, Any]],
    treatment: dict[str, dict[str, Any]],
    ids: list[str],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "reference_nll": paired_continuous(
            baseline,
            treatment,
            ids,
            "reference_nll",
            samples=samples,
            seed=seed,
            lower_is_better=True,
        ),
        "token_f1": paired_continuous(
            baseline,
            treatment,
            ids,
            "token_f1",
            samples=samples,
            seed=seed + 1,
            lower_is_better=False,
        ),
        "exact_match": paired_binary(
            baseline, treatment, ids, "normalized_exact_match"
        ),
        "answer_contains": paired_binary(
            baseline, treatment, ids, "answer_contains"
        ),
    }


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
                if (
                    row["question_type"] == "knowledge-update"
                    and not row["is_abstention"]
                    and row["method"] in METHODS
                ):
                    configs[name][str(row["method"])][str(row["question_id"])] = row
    ids = sorted(configs["baseline_no_dates_retrieval_order"][METHODS[0]])
    if len(ids) != 72:
        raise RuntimeError("expected 72 answerable knowledge-update questions")
    for config in configs.values():
        for method in METHODS:
            if set(config[method]) != set(ids):
                raise RuntimeError("all reader configurations must cover the same questions")

    output = {
        "protocol": {
            "memory_scope": "eight independent real 10M-token shards",
            "question_type": "knowledge-update",
            "answerable_queries": len(ids),
            "selection_uses_answer": False,
            "same_selected_page_ids_within_method": True,
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
                    seed=args.seed + index * 20 + method_index * 5,
                )
                for method_index, method in enumerate(METHODS)
            }
            for index, name in enumerate(patterns)
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
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
