from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, spearmanr, wilcoxon
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from analyze_longmemeval_10m_utility_gate import paired_bootstrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze LongMemEval pairwise conditional set-utility probes."
    )
    parser.add_argument("--probe_pattern", required=True)
    parser.add_argument("--reader_pattern", required=True)
    parser.add_argument("--selection_pattern", required=True)
    parser.add_argument("--partitions", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def retrieval_change(record: dict[str, Any]) -> str:
    if record["is_abstention"]:
        return "abstention"
    static_complete = bool(record["selection_static"]["all_evidence_sessions_at_12"])
    dynamic_complete = bool(record["selection_dynamic"]["all_evidence_sessions_at_12"])
    return {
        (False, True): "rescued",
        (True, False): "lost",
        (True, True): "both_complete",
        (False, False): "both_incomplete",
    }[(static_complete, dynamic_complete)]


def score_diagnostics(records: list[dict[str, Any]], scores: np.ndarray) -> dict[str, Any]:
    delta = np.asarray([record["nll_delta"] for record in records])
    dynamic_better = delta < 0
    output = {
        "score_vs_negative_nll_delta_spearman": float(
            spearmanr(scores, -delta).statistic
        ),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
    }
    if len(set(dynamic_better)) == 2:
        output["dynamic_better_sign_auc"] = float(
            roc_auc_score(dynamic_better, scores)
        )
    by_change = defaultdict(list)
    for index, record in enumerate(records):
        by_change[retrieval_change(record)].append(float(scores[index]))
    output["score_by_retrieval_change_posthoc"] = {
        key: {
            "queries": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
        }
        for key, values in sorted(by_change.items())
    }
    return output


def selection_summary(
    records: list[dict[str, Any]],
    use_dynamic: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    static_nll = np.asarray(
        [record["reader_static"]["reference_nll"] for record in records]
    )
    dynamic_nll = np.asarray(
        [record["reader_dynamic"]["reference_nll"] for record in records]
    )
    selected_nll = np.where(use_dynamic, dynamic_nll, static_nll)
    static_f1 = np.asarray([record["reader_static"]["token_f1"] for record in records])
    dynamic_f1 = np.asarray([record["reader_dynamic"]["token_f1"] for record in records])
    selected_f1 = np.where(use_dynamic, dynamic_f1, static_f1)
    static_exact = np.asarray(
        [bool(record["reader_static"]["normalized_exact_match"]) for record in records]
    )
    dynamic_exact = np.asarray(
        [bool(record["reader_dynamic"]["normalized_exact_match"]) for record in records]
    )
    selected_exact = np.where(use_dynamic, dynamic_exact, static_exact)
    static_contains = np.asarray(
        [bool(record["reader_static"]["answer_contains"]) for record in records]
    )
    dynamic_contains = np.asarray(
        [bool(record["reader_dynamic"]["answer_contains"]) for record in records]
    )
    selected_contains = np.where(use_dynamic, dynamic_contains, static_contains)
    static_refusal = np.asarray(
        [bool(record["reader_static"]["predicted_refusal"]) for record in records]
    )
    dynamic_refusal = np.asarray(
        [bool(record["reader_dynamic"]["predicted_refusal"]) for record in records]
    )
    selected_refusal = np.where(use_dynamic, dynamic_refusal, static_refusal)
    answerable = np.asarray([not record["is_abstention"] for record in records])
    abstention = ~answerable
    selected_tokens = np.asarray(
        [
            record["reader_dynamic"]["working_set_tokens"]
            if use_dynamic[index]
            else record["reader_static"]["working_set_tokens"]
            for index, record in enumerate(records)
        ]
    )
    by_change: dict[str, dict[str, int]] = defaultdict(
        lambda: {"queries": 0, "dynamic_selected": 0}
    )
    for index, record in enumerate(records):
        key = retrieval_change(record)
        by_change[key]["queries"] += 1
        by_change[key]["dynamic_selected"] += int(use_dynamic[index])
    difference = selected_nll - static_nll
    nonzero = difference[difference != 0]
    output = {
        "dynamic_selection_rate": float(use_dynamic.mean()),
        "mean_selected_working_set_tokens": float(selected_tokens.mean()),
        "answerable_token_f1": {
            "static": float(static_f1[answerable].mean()),
            "selected": float(selected_f1[answerable].mean()),
            "delta": float((selected_f1[answerable] - static_f1[answerable]).mean()),
        },
        "answerable_exact_match": {
            "static": float(static_exact[answerable].mean()),
            "selected": float(selected_exact[answerable].mean()),
            "delta": float(
                selected_exact[answerable].mean() - static_exact[answerable].mean()
            ),
        },
        "answerable_answer_contains": {
            "static": float(static_contains[answerable].mean()),
            "selected": float(selected_contains[answerable].mean()),
            "delta": float(
                selected_contains[answerable].mean()
                - static_contains[answerable].mean()
            ),
        },
        "abstention_refusal_accuracy": {
            "static": float(static_refusal[abstention].mean()),
            "selected": float(selected_refusal[abstention].mean()),
            "delta": float(
                selected_refusal[abstention].mean()
                - static_refusal[abstention].mean()
            ),
        },
        "reference_nll_vs_all_static": paired_bootstrap(
            selected_nll, static_nll, samples=samples, seed=seed
        ),
        "reference_nll_vs_all_dynamic": paired_bootstrap(
            selected_nll, dynamic_nll, samples=samples, seed=seed + 1
        ),
        "selection_by_retrieval_change_posthoc": {
            key: {
                **value,
                "dynamic_selection_rate": value["dynamic_selected"] / value["queries"],
            }
            for key, value in sorted(by_change.items())
        },
        "nll_robustness_vs_static": {
            "nonzero_queries": int(len(nonzero)),
            "wins": int((nonzero < 0).sum()),
            "losses": int((nonzero > 0).sum()),
            "nonzero_median": float(np.median(nonzero)) if len(nonzero) else 0.0,
            "two_sided_sign_p": float(
                binomtest(int((nonzero < 0).sum()), len(nonzero), 0.5).pvalue
            )
            if len(nonzero)
            else 1.0,
            "two_sided_wilcoxon_p": float(wilcoxon(nonzero).pvalue)
            if len(nonzero)
            else 1.0,
            "trimmed_mean": {
                str(trim): float(
                    np.sort(difference)[
                        int(trim * len(difference)) : len(difference)
                        - int(trim * len(difference))
                    ].mean()
                )
                for trim in (0.01, 0.025, 0.05, 0.1)
            },
            "clipped_mean": {
                str(bound): float(np.clip(difference, -bound, bound).mean())
                for bound in (1.0, 2.0, 3.0, 5.0)
            },
        },
    }
    output["nll_delta_by_held_out_partition"] = [
        {
            "partition": partition,
            "queries": int(
                sum(record["partition"] == partition for record in records)
            ),
            "dynamic_selection_rate": float(
                use_dynamic[
                    np.asarray(
                        [record["partition"] == partition for record in records]
                    )
                ].mean()
            ),
            "selected_minus_static_nll": float(
                (selected_nll - static_nll)[
                    np.asarray(
                        [record["partition"] == partition for record in records]
                    )
                ].mean()
            ),
        }
        for partition in sorted({record["partition"] for record in records})
    ]
    output["nll_delta_by_question_type"] = [
        {
            "question_type": question_type,
            "queries": int(
                sum(record["question_type"] == question_type for record in records)
            ),
            "dynamic_selection_rate": float(
                use_dynamic[
                    np.asarray(
                        [record["question_type"] == question_type for record in records]
                    )
                ].mean()
            ),
            "selected_minus_static_nll": float(
                (selected_nll - static_nll)[
                    np.asarray(
                        [record["question_type"] == question_type for record in records]
                    )
                ].mean()
            ),
        }
        for question_type in sorted(
            {record["question_type"] for record in records}
        )
    ]
    return output


def train_threshold(scores: np.ndarray, delta: np.ndarray) -> float:
    finite = np.sort(np.unique(scores[np.isfinite(scores)]))
    if not len(finite):
        return float("inf")
    quantiles = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 101)))
    thresholds = np.concatenate(
        ([float("-inf")], quantiles, [float("inf")])
    )
    losses = [float(np.where(scores > threshold, delta, 0.0).mean()) for threshold in thresholds]
    best = int(np.argmin(losses))
    return float(thresholds[best])


def oof_threshold_gate(
    records: list[dict[str, Any]], scores: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, list[float]]:
    delta = np.asarray([record["nll_delta"] for record in records])
    use_dynamic = np.zeros(len(records), dtype=bool)
    thresholds = []
    for held_out in sorted(set(groups)):
        train = groups != held_out
        test = groups == held_out
        threshold = train_threshold(scores[train], delta[train])
        thresholds.append(threshold)
        use_dynamic[test] = scores[test] > threshold
    return use_dynamic, thresholds


def oof_type_threshold_gate(
    records: list[dict[str, Any]], scores: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    delta = np.asarray([record["nll_delta"] for record in records])
    kinds = np.asarray([record["question_type"] for record in records])
    use_dynamic = np.zeros(len(records), dtype=bool)
    thresholds = []
    for held_out in sorted(set(groups)):
        train_outer = groups != held_out
        test_outer = groups == held_out
        global_threshold = train_threshold(scores[train_outer], delta[train_outer])
        for kind in sorted(set(kinds)):
            train = train_outer & (kinds == kind)
            test = test_outer & (kinds == kind)
            threshold = (
                train_threshold(scores[train], delta[train])
                if int(train.sum()) >= 20
                else global_threshold
            )
            use_dynamic[test] = scores[test] > threshold
            thresholds.append(
                {
                    "held_out_partition": int(held_out),
                    "question_type": str(kind),
                    "train_queries": int(train.sum()),
                    "threshold": float(threshold),
                }
            )
    return use_dynamic, thresholds


def probe_feature(record: dict[str, Any], *, include_question_type: bool) -> list[float]:
    probe = record["probe"]
    output = [
        float(probe["pairwise_dynamic_utility_score"]),
        float(probe["completeness_dynamic_utility_score"]),
        float(probe["forward_dynamic_log_odds"]),
        float(probe["reverse_dynamic_log_odds"]),
        abs(
            float(probe["forward_dynamic_log_odds"])
            - float(probe["reverse_dynamic_log_odds"])
        ),
        float(probe["static_completeness_log_odds"]),
        float(probe["dynamic_completeness_log_odds"]),
        float(len(probe["static_extra_block_ids"])),
        float(len(probe["dynamic_extra_block_ids"])),
        float(probe["order_sign_agreement"]),
    ]
    if include_question_type:
        kinds = (
            "knowledge-update",
            "multi-session",
            "single-session-assistant",
            "single-session-preference",
            "single-session-user",
            "temporal-reasoning",
        )
        output.extend(float(record["question_type"] == kind) for kind in kinds)
    return output


def oof_ridge_gate(
    records: list[dict[str, Any]], groups: np.ndarray, *, include_question_type: bool
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    features = np.asarray(
        [
            probe_feature(record, include_question_type=include_question_type)
            for record in records
        ],
        dtype=np.float64,
    )
    delta = np.asarray([record["nll_delta"] for record in records], dtype=np.float64)
    predictions = np.zeros(len(records), dtype=np.float64)
    alphas = []
    for held_out in sorted(set(groups)):
        train = groups != held_out
        test = groups == held_out
        scaler = StandardScaler()
        x_train = scaler.fit_transform(features[train])
        x_test = scaler.transform(features[test])
        lower, upper = np.quantile(delta[train], [0.025, 0.975])
        target = np.clip(delta[train], lower, upper)
        train_groups = groups[train]
        splits = list(
            GroupKFold(n_splits=min(7, len(set(train_groups)))).split(
                x_train, target, train_groups
            )
        )
        model = RidgeCV(
            alphas=np.asarray([0.1, 1.0, 10.0, 100.0, 1_000.0]),
            cv=splits,
            scoring="neg_mean_squared_error",
        )
        model.fit(x_train, target)
        predictions[test] = model.predict(x_test)
        alphas.append(float(model.alpha_))
    return predictions < 0.0, predictions, alphas


def main() -> None:
    args = parse_args()
    records = []
    for partition in range(args.partitions):
        probe_dir = Path(args.probe_pattern.format(partition=partition))
        reader_dir = Path(args.reader_pattern.format(partition=partition))
        selection_dir = Path(args.selection_pattern.format(partition=partition))
        probes = {str(row["question_id"]): row for row in read_jsonl(probe_dir / "rows.jsonl")}
        reader: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in read_jsonl(reader_dir / "rows.jsonl"):
            reader[str(row["method"])][str(row["question_id"])] = row
        selection: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in read_jsonl(selection_dir / "rows.jsonl"):
            if row["method"] in {"static_top12", "evidence_state_dynamic_top12"}:
                selection[str(row["method"])][str(row["question_id"])] = row
        states = {
            str(row["question_id"]): row
            for row in read_jsonl(selection_dir / "states.jsonl")
        }
        for question_id, probe in probes.items():
            static_reader = reader["static_top12"][question_id]
            dynamic_reader = reader["evidence_state_dynamic_top12"][question_id]
            records.append(
                {
                    "question_id": question_id,
                    "partition": partition,
                    "question_type": probe["question_type"],
                    "is_abstention": bool(probe["is_abstention"]),
                    "probe": probe,
                    "reader_static": static_reader,
                    "reader_dynamic": dynamic_reader,
                    "selection_static": selection["static_top12"][question_id],
                    "selection_dynamic": selection["evidence_state_dynamic_top12"][question_id],
                    "state": states[question_id],
                    "nll_delta": float(dynamic_reader["reference_nll"])
                    - float(static_reader["reference_nll"]),
                }
            )
    records.sort(key=lambda row: (row["partition"], row["question_id"]))
    if len(records) != 500 or len({row["question_id"] for row in records}) != 500:
        raise RuntimeError("expected 500 unique questions")
    groups = np.asarray([record["partition"] for record in records])
    pair_scores = np.asarray(
        [record["probe"]["pairwise_dynamic_utility_score"] for record in records]
    )
    complete_scores = np.asarray(
        [record["probe"]["completeness_dynamic_utility_score"] for record in records]
    )
    completeness_zero_use_dynamic = complete_scores > 0

    pair_oof, pair_thresholds = oof_threshold_gate(records, pair_scores, groups)
    complete_oof, complete_thresholds = oof_threshold_gate(
        records, complete_scores, groups
    )
    complete_type_oof, complete_type_thresholds = oof_type_threshold_gate(
        records, complete_scores, groups
    )
    ridge_use_dynamic, ridge_prediction, ridge_alphas = oof_ridge_gate(
        records, groups, include_question_type=False
    )
    type_ridge_use_dynamic, type_ridge_prediction, type_ridge_alphas = oof_ridge_gate(
        records, groups, include_question_type=True
    )
    changed = [record for record in records if not record["probe"]["sets_identical"]]
    output = {
        "protocol": {
            "memory_scope": "eight independent real 10M-token shards",
            "selection_uses_answer_at_test": False,
            "probe_generates_answer": False,
            "outer_validation": "leave-one-10M-shard-out",
            "actions": "keep static rank 9-12 or use state-innovated extra pages",
        },
        "queries": len(records),
        "changed_candidate_sets": len(changed),
        "mean_probe_seconds_changed": float(
            np.mean(
                [
                    sum(
                        float(record["probe"][key])
                        for key in (
                            "forward_seconds",
                            "reverse_seconds",
                            "static_completeness_seconds",
                            "dynamic_completeness_seconds",
                        )
                    )
                    for record in changed
                ]
            )
        ),
        "order_sign_agreement_changed": float(
            np.mean([record["probe"]["order_sign_agreement"] for record in changed])
        ),
        "score_diagnostics": {
            "counterbalanced_pairwise": score_diagnostics(records, pair_scores),
            "independent_completeness_difference": score_diagnostics(
                records, complete_scores
            ),
            "ridge_predicted_negative_nll_delta": score_diagnostics(
                records, -ridge_prediction
            ),
            "type_ridge_predicted_negative_nll_delta": score_diagnostics(
                records, -type_ridge_prediction
            ),
        },
        "state_reference_posthoc_audit": [
            {
                "question_type": question_type,
                "state_mentions_reference_posthoc": mentions_reference,
                "queries": len(indices),
                "mean_completeness_score": float(complete_scores[indices].mean()),
                "mean_raw_dynamic_minus_static_nll": float(
                    np.mean([records[index]["nll_delta"] for index in indices])
                ),
                "completeness_zero_selected_minus_static_nll": float(
                    np.mean(
                        [
                            records[index]["nll_delta"]
                            if completeness_zero_use_dynamic[index]
                            else 0.0
                            for index in indices
                        ]
                    )
                ),
            }
            for question_type in sorted(
                {record["question_type"] for record in records}
            )
            for mentions_reference in (False, True)
            for indices in [
                [
                    index
                    for index, record in enumerate(records)
                    if record["question_type"] == question_type
                    and bool(record["state"]["state_mentions_reference_posthoc"])
                    == mentions_reference
                ]
            ]
            if indices
        ],
        "gates": {
            "pairwise_zero": selection_summary(
                records,
                pair_scores > 0,
                samples=args.bootstrap_samples,
                seed=args.seed,
            ),
            "completeness_zero": selection_summary(
                records,
                completeness_zero_use_dynamic,
                samples=args.bootstrap_samples,
                seed=args.seed + 10,
            ),
            "pairwise_train_threshold": {
                "fold_thresholds": pair_thresholds,
                **selection_summary(
                    records,
                    pair_oof,
                    samples=args.bootstrap_samples,
                    seed=args.seed + 20,
                ),
            },
            "completeness_train_threshold": {
                "fold_thresholds": complete_thresholds,
                **selection_summary(
                    records,
                    complete_oof,
                    samples=args.bootstrap_samples,
                    seed=args.seed + 30,
                ),
            },
            "completeness_type_train_threshold": {
                "fold_type_thresholds": complete_type_thresholds,
                **selection_summary(
                    records,
                    complete_type_oof,
                    samples=args.bootstrap_samples,
                    seed=args.seed + 35,
                ),
            },
            "combined_ridge": {
                "fold_alphas": ridge_alphas,
                **selection_summary(
                    records,
                    ridge_use_dynamic,
                    samples=args.bootstrap_samples,
                    seed=args.seed + 40,
                ),
            },
            "combined_type_ridge": {
                "fold_alphas": type_ridge_alphas,
                **selection_summary(
                    records,
                    type_ridge_use_dynamic,
                    samples=args.bootstrap_samples,
                    seed=args.seed + 50,
                ),
            },
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows_path = output_path.with_suffix(".rows.jsonl")
    with rows_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            handle.write(
                json.dumps(
                    {
                        "question_id": record["question_id"],
                        "partition": record["partition"],
                        "nll_delta": record["nll_delta"],
                        "retrieval_change_posthoc": retrieval_change(record),
                        "pairwise_score": float(pair_scores[index]),
                        "completeness_score": float(complete_scores[index]),
                        "ridge_predicted_nll_delta": float(ridge_prediction[index]),
                        "combined_ridge_use_dynamic": bool(ridge_use_dynamic[index]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
