from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


DEPTHS = (3, 8, 16, 32)
TRANSITIONS = tuple(zip(DEPTHS[:-1], DEPTHS[1:]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relate observed-state retrospective probes to future scope utility."
    )
    parser.add_argument("--probe_rows", required=True)
    parser.add_argument("--retrieval_rows", required=True)
    parser.add_argument("--ppl128_rows", required=True)
    parser.add_argument("--ppl512_rows", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--output_policy_rows", required=True)
    parser.add_argument("--probe_windows", default="8,16,32,64")
    parser.add_argument("--gain_thresholds", default="0,0.0025,0.005,0.01")
    parser.add_argument("--bootstrap_samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_ints(spec: str) -> list[int]:
    return sorted({int(item.strip()) for item in spec.split(",") if item.strip()})


def parse_floats(spec: str) -> list[float]:
    return sorted({float(item.strip()) for item in spec.split(",") if item.strip()})


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def bootstrap_mean_ci(
    values: list[float], *, samples: int, seed: int
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def infer_state_suffix(row: dict[str, Any]) -> int:
    suffix = (
        int(row["model_input_tokens"])
        - int(row["retrieved_tokens"])
        - int(row["target_tokens"])
    )
    if suffix not in (128, 512):
        raise ValueError(f"unexpected state suffix: {suffix}")
    return suffix


def main() -> None:
    args = parse_args()
    windows = parse_ints(args.probe_windows)
    thresholds = parse_floats(args.gain_thresholds)
    probe_rows = read_jsonl(args.probe_rows)
    probe_lookup = {
        (
            int(row["query_id"]),
            int(row["state_suffix_tokens"]),
            int(row["scope_depth"]),
        ): row
        for row in probe_rows
    }

    retrieval_lookup: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in read_jsonl(args.retrieval_rows):
        if int(row["memory_tokens"]) != 100_000_000:
            continue
        method = str(row["method"])
        if not method.startswith("hier_bm25_scope"):
            continue
        depth = int(method.removeprefix("hier_bm25_scope"))
        if depth in DEPTHS and int(row["prefix_tokens"]) in (128, 512):
            retrieval_lookup[
                (int(row["query_id"]), int(row["prefix_tokens"]), depth)
            ] = row

    ppl_lookup: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in read_jsonl(args.ppl128_rows) + read_jsonl(args.ppl512_rows):
        method = str(row["method"])
        if method not in {f"hier_bm25_scope{depth}" for depth in DEPTHS}:
            continue
        depth = int(method.removeprefix("hier_bm25_scope"))
        if depth in DEPTHS:
            ppl_lookup[(int(row["query_id"]), infer_state_suffix(row), depth)] = row

    events = []
    for query_id in range(30):
        for suffix in (128, 512):
            for previous_depth, expanded_depth in TRANSITIONS:
                previous_future = float(
                    ppl_lookup[(query_id, suffix, previous_depth)]["mean_nll"]
                )
                expanded_future = float(
                    ppl_lookup[(query_id, suffix, expanded_depth)]["mean_nll"]
                )
                event = {
                    "query_id": query_id,
                    "state_suffix_tokens": suffix,
                    "previous_depth": previous_depth,
                    "expanded_depth": expanded_depth,
                    "future_marginal_nll_gain": previous_future - expanded_future,
                    "future_expansion_helped": previous_future > expanded_future,
                    "retrospective_gain": {},
                }
                for window in windows:
                    previous_probe = float(
                        probe_lookup[(query_id, suffix, previous_depth)][
                            "probe_mean_nll"
                        ][str(window)]
                    )
                    expanded_probe = float(
                        probe_lookup[(query_id, suffix, expanded_depth)][
                            "probe_mean_nll"
                        ][str(window)]
                    )
                    event["retrospective_gain"][str(window)] = (
                        previous_probe - expanded_probe
                    )
                events.append(event)

    signal_quality = {}
    future_gains = np.asarray(
        [float(row["future_marginal_nll_gain"]) for row in events]
    )
    future_labels = (future_gains > 0).astype(np.int64)
    for window in windows:
        scores = np.asarray(
            [float(row["retrospective_gain"][str(window)]) for row in events]
        )
        correlation = spearmanr(scores, future_gains)
        item: dict[str, Any] = {
            "events": len(events),
            "spearman_with_future_gain": float(correlation.statistic),
            "spearman_pvalue": float(correlation.pvalue),
            "sign_auc": safe_auc(future_labels, scores),
            "sign_agreement": float(np.mean((scores > 0) == (future_gains > 0))),
            "by_state": {},
            "by_transition": {},
        }
        for suffix in (128, 512):
            mask = np.asarray(
                [int(row["state_suffix_tokens"]) == suffix for row in events]
            )
            state_correlation = spearmanr(scores[mask], future_gains[mask])
            item["by_state"][str(suffix)] = {
                "events": int(mask.sum()),
                "spearman": float(state_correlation.statistic),
                "spearman_pvalue": float(state_correlation.pvalue),
                "sign_auc": safe_auc(future_labels[mask], scores[mask]),
                "sign_agreement": float(
                    np.mean((scores[mask] > 0) == (future_gains[mask] > 0))
                ),
            }
        for previous_depth, expanded_depth in TRANSITIONS:
            label = f"{previous_depth}_to_{expanded_depth}"
            mask = np.asarray(
                [
                    int(row["previous_depth"]) == previous_depth
                    and int(row["expanded_depth"]) == expanded_depth
                    for row in events
                ]
            )
            transition_correlation = spearmanr(scores[mask], future_gains[mask])
            item["by_transition"][label] = {
                "events": int(mask.sum()),
                "spearman": float(transition_correlation.statistic),
                "spearman_pvalue": float(transition_correlation.pvalue),
                "sign_auc": safe_auc(future_labels[mask], scores[mask]),
            }
        signal_quality[str(window)] = item

    policy_rows = []
    event_lookup = {
        (
            int(row["query_id"]),
            int(row["state_suffix_tokens"]),
            int(row["previous_depth"]),
            int(row["expanded_depth"]),
        ): row
        for row in events
    }
    for window in windows:
        for threshold in thresholds:
            for query_id in range(30):
                for suffix in (128, 512):
                    chosen_depth = DEPTHS[0]
                    evaluated_depths = [DEPTHS[0]]
                    decisions = []
                    for previous_depth, expanded_depth in TRANSITIONS:
                        if chosen_depth != previous_depth:
                            break
                        evaluated_depths.append(expanded_depth)
                        event = event_lookup[
                            (query_id, suffix, previous_depth, expanded_depth)
                        ]
                        gain = float(event["retrospective_gain"][str(window)])
                        expand = gain > threshold
                        decisions.append(
                            {
                                "previous_depth": previous_depth,
                                "expanded_depth": expanded_depth,
                                "retrospective_nll_gain": gain,
                                "expand": expand,
                            }
                        )
                        if not expand:
                            break
                        chosen_depth = expanded_depth
                    probe_seconds = sum(
                        float(probe_lookup[(query_id, suffix, depth)]["forward_seconds"])
                        for depth in evaluated_depths
                    )
                    policy_rows.append(
                        {
                            "query_id": query_id,
                            "state_suffix_tokens": suffix,
                            "method": (
                                f"retrospective_probe{window}_"
                                f"g{int(round(threshold * 10_000)):03d}"
                            ),
                            "probe_window_tokens": window,
                            "gain_threshold": threshold,
                            "chosen_scope_depth": chosen_depth,
                            "evaluated_scope_depths": evaluated_depths,
                            "probe_forwards": len(evaluated_depths),
                            "probe_seconds": probe_seconds,
                            "decisions": decisions,
                            "probe_uses_future_target": False,
                            "selection_uses_target": False,
                        }
                    )

    policy_quality = []
    for suffix in (128, 512):
        for method in sorted({str(row["method"]) for row in policy_rows}):
            group = [
                row
                for row in policy_rows
                if row["method"] == method
                and int(row["state_suffix_tokens"]) == suffix
            ]
            selected_nll = [
                float(
                    ppl_lookup[
                        (
                            int(row["query_id"]),
                            suffix,
                            int(row["chosen_scope_depth"]),
                        )
                    ]["mean_nll"]
                )
                for row in group
            ]
            selected_retrieval = [
                retrieval_lookup[
                    (
                        int(row["query_id"]),
                        suffix,
                        int(row["chosen_scope_depth"]),
                    )
                ]
                for row in group
            ]
            item = {
                "state_suffix_tokens": suffix,
                "method": method,
                "queries": len(group),
                "mean_chosen_scope_depth": mean(
                    [float(row["chosen_scope_depth"]) for row in group]
                ),
                "mean_probe_forwards": mean(
                    [float(row["probe_forwards"]) for row in group]
                ),
                "mean_probe_seconds": mean(
                    [float(row["probe_seconds"]) for row in group]
                ),
                "ppl": math.exp(mean(selected_nll)),
                "mean_candidate_blocks": mean(
                    [float(row["candidate_blocks"]) for row in selected_retrieval]
                ),
                "same_scope_any_at_8": mean(
                    [float(row["same_scope_any_at_8"]) for row in selected_retrieval]
                ),
                "same_scope_fraction_at_8": mean(
                    [float(row["same_scope_fraction_at_8"]) for row in selected_retrieval]
                ),
                "paired_vs_fixed": {},
            }
            method_window = int(group[0]["probe_window_tokens"])
            for fixed_depth in DEPTHS:
                differences = [
                    selected_nll[index]
                    - float(
                        ppl_lookup[
                            (int(row["query_id"]), suffix, fixed_depth)
                        ]["mean_nll"]
                    )
                    for index, row in enumerate(group)
                ]
                item["paired_vs_fixed"][str(fixed_depth)] = {
                    "meaning": "negative favors retrospective policy",
                    "mean_nll_policy_minus_fixed": mean(differences),
                    "bootstrap95": bootstrap_mean_ci(
                        differences,
                        samples=args.bootstrap_samples,
                        seed=args.seed + suffix + fixed_depth + method_window,
                    ),
                }
            policy_quality.append(item)

    output = {
        "source": "observed-state retrospective probe versus future scope utility",
        "protocol": {
            "queries": 30,
            "states": [128, 512],
            "events": len(events),
            "probe_windows": windows,
            "gain_thresholds": thresholds,
            "retrieval_query_end_offset_tokens": sorted(
                {
                    int(row.get("retrieval_query_end_offset_tokens", 0))
                    for row in probe_rows
                }
            ),
            "retrieval_query_uses_observed_probe_tokens": any(
                bool(row.get("retrieval_query_uses_observed_probe_tokens", True))
                for row in probe_rows
            ),
            "probe_uses_only_already_observed_state": True,
            "probe_uses_future_target": False,
            "selection_uses_target": False,
        },
        "signal_quality": signal_quality,
        "policy_quality": policy_quality,
    }
    output_path = Path(args.output_summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    policy_path = Path(args.output_policy_rows)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    with policy_path.open("w", encoding="utf-8") as handle:
        for row in policy_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
