from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate answer-free static/dynamic page gates with leave-one-10M-shard-out "
            "prediction."
        )
    )
    parser.add_argument("--selection_pattern", required=True)
    parser.add_argument("--reader_pattern", required=True)
    parser.add_argument("--partitions", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def contiguous_components(values: Iterable[int]) -> int:
    ordered = sorted(set(int(value) for value in values))
    if not ordered:
        return 0
    return 1 + sum(b != a + 1 for a, b in zip(ordered, ordered[1:]))


def state_features(
    state: dict[str, Any],
    static: dict[str, Any],
    dynamic: dict[str, Any],
    *,
    include_text_flags: bool,
    include_question_type: bool,
) -> dict[str, float | str]:
    initial = [int(value) for value in state["initial_block_ids"]]
    static_ids = [int(value) for value in static["top_block_ids"]]
    dynamic_ids = [int(value) for value in dynamic["top_block_ids"]]
    refreshes = list(state["refreshes"])
    added_counts = [len(row["added_block_ids"]) for row in refreshes]
    top_sets = [initial] + [row["top_block_ids"] for row in refreshes]
    static_sessions = set(int(value) for value in state["static_session_rows"])
    routed_sessions = [
        set(int(value) for value in row["selected_session_rows"]) for row in refreshes
    ]
    all_routed_sessions = set().union(static_sessions, *routed_sessions)
    new_routed_sessions = all_routed_sessions - static_sessions

    features: dict[str, float | str] = {
        "generated_tokens": float(state["generated_tokens"]),
        "total_added_pages": float(sum(added_counts)),
        "max_added_per_refresh": float(max(added_counts, default=0)),
        "refreshes_with_addition": float(sum(value > 0 for value in added_counts)),
        "final_frontier_pages": float(len(state["dynamic_frontier_block_ids"])),
        "static_dynamic_jaccard": jaccard(static_ids, dynamic_ids),
        "dynamic_only_pages": float(len(set(dynamic_ids) - set(static_ids))),
        "static_only_pages": float(len(set(static_ids) - set(dynamic_ids))),
        "dynamic_working_tokens": float(dynamic["working_set_tokens"]),
        "initial_components": float(contiguous_components(initial)),
        "static_components": float(contiguous_components(static_ids)),
        "dynamic_components": float(contiguous_components(dynamic_ids)),
        "unique_routed_sessions": float(len(all_routed_sessions)),
        "new_routed_sessions": float(len(new_routed_sessions)),
        "route_refreshes_with_novel_session": float(
            sum(bool(sessions - static_sessions) for sessions in routed_sessions)
        ),
        "mean_top_initial_jaccard": float(
            mean(jaccard(initial, values) for values in top_sets[1:]) or 0.0
        ),
        "mean_consecutive_top_jaccard": float(
            mean(jaccard(a, b) for a, b in zip(top_sets, top_sets[1:])) or 0.0
        ),
        "mean_route_static_jaccard": float(
            mean(jaccard(static_sessions, sessions) for sessions in routed_sessions) or 0.0
        ),
        "mean_consecutive_route_jaccard": float(
            mean(jaccard(a, b) for a, b in zip(routed_sessions, routed_sessions[1:]))
            or 0.0
        ),
    }
    static_extra = set(static_ids) - set(initial)
    dynamic_extra = set(dynamic_ids) - set(initial)
    features["dynamic_extra_static_overlap"] = float(len(static_extra & dynamic_extra))
    features["dynamic_extra_novel_to_static"] = float(len(dynamic_extra - static_extra))

    text = str(state["state_text"])
    if include_text_flags:
        lowered = text.lower()
        unresolved_terms = re.findall(
            r"\b(?:unknown|unclear|unresolved|missing|need|needs|needed|"
            r"insufficient|not found|not given|no evidence|no information)\b",
            lowered,
        )
        features.update(
            {
                "state_chars": float(len(text)),
                "state_words": float(len(re.findall(r"\b\w+\b", text))),
                "state_lines": float(len([line for line in text.splitlines() if line.strip()])),
                "state_digits": float(len(re.findall(r"\d", text))),
                "state_quotes": float(text.count('"') + text.count("'")),
                "state_question_marks": float(text.count("?")),
                "state_unresolved_terms": float(len(unresolved_terms)),
                "state_has_unresolved": float(bool(unresolved_terms)),
            }
        )
    if include_question_type:
        features["question_type"] = str(state["question_type"])
        features["is_abstention"] = str(bool(state["is_abstention"]))
    return features


def fit_predict_outer_fold(
    records: list[dict[str, Any]],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    variant: str,
) -> tuple[np.ndarray, float]:
    include_text_flags = variant in {"structured", "structured_plus_state_text"}
    include_question_type = variant in {"structured", "structured_plus_state_text"}
    train_dicts = [
        state_features(
            records[index]["state"],
            records[index]["selection_static"],
            records[index]["selection_dynamic"],
            include_text_flags=include_text_flags,
            include_question_type=include_question_type,
        )
        for index in train_indices
    ]
    test_dicts = [
        state_features(
            records[index]["state"],
            records[index]["selection_static"],
            records[index]["selection_dynamic"],
            include_text_flags=include_text_flags,
            include_question_type=include_question_type,
        )
        for index in test_indices
    ]
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform(train_dicts)
    x_test = vectorizer.transform(test_dicts)

    if variant == "structured_plus_state_text":
        text_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=3,
            max_features=3_000,
            sublinear_tf=True,
        )
        train_text = [str(records[index]["state"]["state_text"]) for index in train_indices]
        test_text = [str(records[index]["state"]["state_text"]) for index in test_indices]
        text_train = text_vectorizer.fit_transform(train_text)
        text_test = text_vectorizer.transform(test_text)
        x_train = sparse.hstack([x_train, text_train], format="csr")
        x_test = sparse.hstack([x_test, text_test], format="csr")

    target = np.asarray(
        [records[index]["nll_delta"] for index in train_indices], dtype=np.float64
    )
    lower, upper = np.quantile(target, [0.025, 0.975])
    clipped = np.clip(target, lower, upper)
    train_groups = np.asarray([records[index]["partition"] for index in train_indices])
    group_splits = list(
        GroupKFold(n_splits=min(7, len(set(train_groups)))).split(
            x_train, clipped, train_groups
        )
    )
    model = RidgeCV(
        alphas=np.asarray([0.1, 1.0, 10.0, 100.0, 1_000.0]),
        cv=group_splits,
        scoring="neg_mean_squared_error",
    )
    model.fit(x_train, clipped)
    return np.asarray(model.predict(x_test), dtype=np.float64), float(model.alpha_)


def paired_bootstrap(
    selected: np.ndarray,
    baseline: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    difference = selected - baseline
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(difference), size=(samples, len(difference)))
    sampled = difference[draws].mean(axis=1)
    return {
        "selected_mean": float(selected.mean()),
        "baseline_mean": float(baseline.mean()),
        "selected_minus_baseline": float(difference.mean()),
        "bootstrap_95_ci": [
            float(np.quantile(sampled, 0.025)),
            float(np.quantile(sampled, 0.975)),
        ],
        "wins": int((difference < 0).sum()),
        "losses": int((difference > 0).sum()),
        "ties": int((difference == 0).sum()),
        "perplexity_ratio_exp_mean_delta": math.exp(float(difference.mean())),
    }


def summarize_variant(
    records: list[dict[str, Any]],
    predictions: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    static_nll = np.asarray([row["reader_static"]["reference_nll"] for row in records])
    dynamic_nll = np.asarray([row["reader_dynamic"]["reference_nll"] for row in records])
    static_f1 = np.asarray([row["reader_static"]["token_f1"] for row in records])
    dynamic_f1 = np.asarray([row["reader_dynamic"]["token_f1"] for row in records])
    use_dynamic = predictions < 0.0
    selected_nll = np.where(use_dynamic, dynamic_nll, static_nll)
    selected_f1 = np.where(use_dynamic, dynamic_f1, static_f1)
    selected_tokens = np.asarray(
        [
            row["reader_dynamic"]["working_set_tokens"]
            if use_dynamic[index]
            else row["reader_static"]["working_set_tokens"]
            for index, row in enumerate(records)
        ],
        dtype=np.float64,
    )
    actual_delta = dynamic_nll - static_nll
    output: dict[str, Any] = {
        "queries": len(records),
        "dynamic_selection_rate": float(use_dynamic.mean()),
        "mean_selected_working_set_tokens": float(selected_tokens.mean()),
        "reference_nll_vs_all_static": paired_bootstrap(
            selected_nll, static_nll, samples=samples, seed=seed
        ),
        "reference_nll_vs_all_dynamic": paired_bootstrap(
            selected_nll, dynamic_nll, samples=samples, seed=seed + 1
        ),
        "mean_token_f1": float(selected_f1.mean()),
        "prediction_delta_spearman": float(spearmanr(predictions, actual_delta).statistic),
    }
    sign = actual_delta < 0
    if len(set(sign)) == 2:
        output["dynamic_better_sign_auc"] = float(roc_auc_score(sign, -predictions))

    by_completeness: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"queries": 0, "dynamic_selected": 0}
    )
    for index, row in enumerate(records):
        if row["is_abstention"]:
            key = "abstention"
        else:
            static_complete = bool(row["selection_static"]["all_evidence_sessions_at_12"])
            dynamic_complete = bool(row["selection_dynamic"]["all_evidence_sessions_at_12"])
            key = {
                (False, True): "rescued",
                (True, False): "lost",
                (True, True): "both_complete",
                (False, False): "both_incomplete",
            }[(static_complete, dynamic_complete)]
        by_completeness[key]["queries"] += 1
        by_completeness[key]["dynamic_selected"] += int(use_dynamic[index])
    output["gate_by_retrieval_completeness_posthoc"] = {
        key: {
            **value,
            "dynamic_selection_rate": value["dynamic_selected"] / value["queries"],
        }
        for key, value in sorted(by_completeness.items())
    }
    return output


def main() -> None:
    args = parse_args()
    records: list[dict[str, Any]] = []
    for partition in range(args.partitions):
        selection_dir = Path(args.selection_pattern.format(partition=partition))
        reader_dir = Path(args.reader_pattern.format(partition=partition))
        states = {
            str(row["question_id"]): row for row in read_jsonl(selection_dir / "states.jsonl")
        }
        selection: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in read_jsonl(selection_dir / "rows.jsonl"):
            if row["method"] in {"static_top12", "evidence_state_dynamic_top12"}:
                selection[str(row["method"])][str(row["question_id"])] = row
        reader: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in read_jsonl(reader_dir / "rows.jsonl"):
            reader[str(row["method"])][str(row["question_id"])] = row
        for question_id, state in states.items():
            static_reader = reader["static_top12"][question_id]
            dynamic_reader = reader["evidence_state_dynamic_top12"][question_id]
            records.append(
                {
                    "question_id": question_id,
                    "partition": partition,
                    "question_type": state["question_type"],
                    "is_abstention": bool(state["is_abstention"]),
                    "state": state,
                    "selection_static": selection["static_top12"][question_id],
                    "selection_dynamic": selection["evidence_state_dynamic_top12"][question_id],
                    "reader_static": static_reader,
                    "reader_dynamic": dynamic_reader,
                    "nll_delta": float(dynamic_reader["reference_nll"])
                    - float(static_reader["reference_nll"]),
                }
            )
    records.sort(key=lambda row: (row["partition"], row["question_id"]))
    if len(records) != 500 or len({row["question_id"] for row in records}) != 500:
        raise RuntimeError("expected 500 unique questions")

    groups = np.asarray([row["partition"] for row in records], dtype=np.int64)
    variants = ["trajectory_only", "structured", "structured_plus_state_text"]
    predictions = {variant: np.zeros(len(records), dtype=np.float64) for variant in variants}
    fold_alphas: dict[str, list[float]] = {variant: [] for variant in variants}
    for held_out in range(args.partitions):
        train_indices = np.flatnonzero(groups != held_out)
        test_indices = np.flatnonzero(groups == held_out)
        for variant in variants:
            fold_prediction, alpha = fit_predict_outer_fold(
                records, train_indices, test_indices, variant=variant
            )
            predictions[variant][test_indices] = fold_prediction
            fold_alphas[variant].append(alpha)

    static_nll = np.asarray([row["reader_static"]["reference_nll"] for row in records])
    dynamic_nll = np.asarray([row["reader_dynamic"]["reference_nll"] for row in records])
    oracle_nll = np.minimum(static_nll, dynamic_nll)
    output = {
        "protocol": {
            "memory_scope": "eight independent real 10M-token shards",
            "outer_validation": "leave-one-10M-shard-out",
            "selection_rule": "use dynamic pages iff OOF predicted dynamic-minus-static NLL < 0",
            "forbidden_features": [
                "reference answer",
                "gold evidence ids or recall labels",
                "reader NLL or generated answer at test time",
                "posthoc state-reference overlap flags",
            ],
            "target": "train-shard dynamic-minus-static teacher-forced reference NLL",
        },
        "queries": len(records),
        "always_static_mean_reference_nll": float(static_nll.mean()),
        "always_dynamic_mean_reference_nll": float(dynamic_nll.mean()),
        "diagnostic_oracle_min_mean_reference_nll": float(oracle_nll.mean()),
        "diagnostic_oracle_vs_static": paired_bootstrap(
            oracle_nll, static_nll, samples=args.bootstrap_samples, seed=args.seed
        ),
        "variants": {
            variant: {
                "fold_selected_alphas": fold_alphas[variant],
                **summarize_variant(
                    records,
                    predictions[variant],
                    samples=args.bootstrap_samples,
                    seed=args.seed + 10 + index * 10,
                ),
            }
            for index, variant in enumerate(variants)
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    rows_path = output_path.with_suffix(".rows.jsonl")
    with rows_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(records):
            compact = {
                "question_id": row["question_id"],
                "partition": row["partition"],
                "question_type": row["question_type"],
                "is_abstention": row["is_abstention"],
                "nll_delta": row["nll_delta"],
            }
            for variant in variants:
                compact[f"{variant}_predicted_nll_delta"] = float(predictions[variant][index])
                compact[f"{variant}_use_dynamic"] = bool(predictions[variant][index] < 0)
            handle.write(json.dumps(compact, ensure_ascii=False) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
