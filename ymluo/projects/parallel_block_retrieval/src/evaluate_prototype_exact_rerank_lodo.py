from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

from benchmark_selected_head_debiased_retrieval import read_selection, rrf_ranking
from evaluate_query_prototype_full_axis import (
    normalize_rows,
    surrogate_score,
    top_prefix,
    valid_vectors,
)
from run_all_head_prior_debiased_retrieval import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate train-only prototype coarse routing followed by exact-QK "
            "z-score reranking and final LODO RRF gold recall."
        )
    )
    parser.add_argument("--packed_profile_dir", required=True)
    parser.add_argument("--support_index_dir", required=True)
    parser.add_argument("--query_profiles", required=True)
    parser.add_argument("--selection_csv", required=True)
    parser.add_argument("--queries_jsonl", required=True)
    parser.add_argument("--zscore_reference_npz", required=True)
    parser.add_argument("--exact_score_profile_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gate_feature", default="raw_top1_block_diversity")
    parser.add_argument("--heads_per_fold", type=int, default=16)
    parser.add_argument("--prototypes", type=int, default=128)
    parser.add_argument("--candidate_budgets", default="2048,4096,8192,9766")
    parser.add_argument("--top_per_head", type=int, default=16)
    parser.add_argument("--target_blocks", type=int, default=39)
    parser.add_argument("--std_epsilon", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def top_local_indices(
    scores: np.ndarray, block_ids: np.ndarray, count: int
) -> np.ndarray:
    if count >= len(scores):
        selected = np.arange(len(scores), dtype=np.int64)
    else:
        selected = np.argpartition(-scores, count - 1)[:count]
    order = np.lexsort((block_ids[selected], -scores[selected]))
    return selected[order]


def main() -> None:
    args = parse_args()
    candidate_budgets = sorted(
        {int(item) for item in args.candidate_budgets.split(",")}
    )
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = json.loads(
        (Path(args.packed_profile_dir) / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    num_blocks = int(profile["num_blocks"])
    num_query_heads = int(profile["num_query_heads"])
    if min(candidate_budgets) < args.top_per_head or max(candidate_budgets) > num_blocks:
        raise ValueError("candidate budgets must be between per-head Top-K and blocks")

    selected_by_fold = read_selection(
        Path(args.selection_csv), args.gate_feature, args.heads_per_fold
    )
    union_flat_heads = sorted(
        {head for heads in selected_by_fold.values() for head in heads}
    )
    flat_head_to_output = {
        flat_head: index for index, flat_head in enumerate(union_flat_heads)
    }
    query_payload = torch.load(
        Path(args.query_profiles), map_location="cpu", weights_only=False
    )
    q = query_payload["svd_q"].numpy()
    mask = query_payload["mask"].numpy()
    queries = read_jsonl(Path(args.queries_jsonl))
    with np.load(Path(args.zscore_reference_npz)) as reference:
        reference_ids = reference["block_ids"].astype(np.int64)
        reference_scores = reference["scores"].astype(np.float32)
        fold_ids = reference["fold_ids"].astype(np.int64)
        layers = reference["layers"].astype(np.int64)

    exact_dir = Path(args.exact_score_profile_dir)
    exact_raw_scores = np.load(
        exact_dir / "exact_raw_selected_scores.npy", mmap_mode="r"
    )
    exact_means = np.load(exact_dir / "exact_train_mean.npy", mmap_mode="r")
    exact_stds = np.load(exact_dir / "exact_train_std.npy", mmap_mode="r")
    stored_flat_heads = np.load(exact_dir / "selected_flat_heads.npy").astype(np.int64)
    if not np.array_equal(stored_flat_heads, np.asarray(union_flat_heads)):
        raise RuntimeError("exact score profile uses a different selected-head union")
    model_indices: dict[tuple[int, int], int] = {}
    with (exact_dir / "models.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            model_indices[(int(row["fold"]), int(row["flat_head"]))] = int(
                row["model_index"]
            )

    alignment_slots = 0
    alignment_id_mismatches = 0
    alignment_set_mismatches = 0
    alignment_max_score_error = 0.0
    all_block_ids = np.arange(num_blocks, dtype=np.int64)
    for fold, heads in selected_by_fold.items():
        test_queries = np.flatnonzero(fold_ids == fold)
        for flat_head in heads:
            layer_index, query_head = divmod(flat_head, num_query_heads)
            output_head = flat_head_to_output[flat_head]
            exact_model = model_indices[(fold, flat_head)]
            for query_index in test_queries:
                exact_score = (
                    exact_raw_scores[query_index, output_head]
                    - exact_means[exact_model]
                ) / exact_stds[exact_model]
                local = top_local_indices(
                    np.asarray(exact_score), all_block_ids, args.top_per_head
                )
                expected = reference_ids[query_index, layer_index, query_head]
                alignment_slots += args.top_per_head
                alignment_id_mismatches += int(np.sum(local != expected))
                alignment_set_mismatches += int(
                    set(int(item) for item in local)
                    != set(int(item) for item in expected)
                )
                alignment_max_score_error = max(
                    alignment_max_score_error,
                    float(
                        np.max(
                            np.abs(
                                exact_score[expected]
                                - reference_scores[
                                    query_index, layer_index, query_head
                                ]
                            )
                        )
                    ),
                )

    approximate_ids = {
        budget: np.full(
            (len(queries), len(union_flat_heads), args.top_per_head),
            -1,
            dtype=np.int32,
        )
        for budget in candidate_budgets
    }
    head_rows: list[dict[str, Any]] = []
    maximum_budget = max(candidate_budgets)
    support_dir = Path(args.support_index_dir)
    total_started = time.perf_counter()

    for fold in sorted(selected_by_fold):
        train_queries = np.flatnonzero(fold_ids != fold)
        test_queries = np.flatnonzero(fold_ids == fold)
        for flat_head in selected_by_fold[fold]:
            model_started = time.perf_counter()
            layer_index, query_head = divmod(flat_head, num_query_heads)
            train = valid_vectors(q, mask, train_queries, layer_index, query_head)
            train_directions = normalize_rows(train)
            model = MiniBatchKMeans(
                n_clusters=args.prototypes,
                batch_size=min(1024, len(train_directions)),
                max_iter=100,
                n_init=1,
                random_state=args.seed + fold * 1000 + flat_head * 17,
                reassignment_ratio=0.01,
            )
            model.fit(train_directions)
            centers = normalize_rows(model.cluster_centers_.astype(np.float32))
            centers_tensor = torch.from_numpy(centers).to(device=device)
            support = np.load(
                support_dir / f"fold{fold:02d}_head{flat_head:03d}.npy",
                mmap_mode="r",
            )

            proxy_sum = np.zeros(num_blocks, dtype=np.float64)
            proxy_square_sum = np.zeros(num_blocks, dtype=np.float64)
            for query_index in train_queries:
                score = surrogate_score(
                    values=q[query_index, :, layer_index, query_head],
                    valid=mask[query_index].astype(bool),
                    centers_tensor=centers_tensor,
                    support=support,
                    device=device,
                )
                proxy_sum += score
                proxy_square_sum += score.astype(np.float64) ** 2
            proxy_mean = proxy_sum / len(train_queries)
            proxy_variance = np.maximum(
                0.0,
                proxy_square_sum / len(train_queries) - proxy_mean**2,
            )
            proxy_std = np.maximum(np.sqrt(proxy_variance), args.std_epsilon)

            output_head = flat_head_to_output[flat_head]
            exact_model = model_indices[(fold, flat_head)]
            exact_mean = exact_means[exact_model]
            exact_std = exact_stds[exact_model]
            recalls = {budget: [] for budget in candidate_budgets}
            for query_index in test_queries:
                proxy_raw = surrogate_score(
                    values=q[query_index, :, layer_index, query_head],
                    valid=mask[query_index].astype(bool),
                    centers_tensor=centers_tensor,
                    support=support,
                    device=device,
                )
                proxy_zscore = (proxy_raw.astype(np.float64) - proxy_mean) / proxy_std
                candidates = top_prefix(proxy_zscore, maximum_budget)
                exact_reference = reference_ids[
                    query_index, layer_index, query_head
                ]
                exact_reference_set = set(int(item) for item in exact_reference)
                for budget in candidate_budgets:
                    candidate_ids = candidates[:budget]
                    exact_candidate_scores = (
                        exact_raw_scores[query_index, output_head, candidate_ids]
                        - exact_mean[candidate_ids]
                    ) / exact_std[candidate_ids]
                    refined_local = top_local_indices(
                        np.asarray(exact_candidate_scores),
                        candidate_ids,
                        args.top_per_head,
                    )
                    refined_ids = candidate_ids[refined_local]
                    approximate_ids[budget][
                        query_index, output_head
                    ] = refined_ids.astype(np.int32)
                    recalls[budget].append(
                        len(exact_reference_set & set(int(item) for item in refined_ids))
                        / args.top_per_head
                    )

            row: dict[str, Any] = {
                "fold": fold,
                "flat_head": int(flat_head),
                "layer": int(layers[layer_index]),
                "query_head": query_head,
                "test_queries": int(len(test_queries)),
                "seconds": time.perf_counter() - model_started,
            }
            for budget in candidate_budgets:
                row[f"exact_top16_recall_at_{budget}"] = float(
                    np.mean(recalls[budget])
                )
            head_rows.append(row)
            write_csv(output_dir / "head_results.partial.csv", head_rows)
            print(json.dumps(row), flush=True)

    query_rows: list[dict[str, Any]] = []
    reference_hits = np.zeros(len(queries), dtype=bool)
    approximate_hits = {
        budget: np.zeros(len(queries), dtype=bool) for budget in candidate_budgets
    }
    for query_index, query in enumerate(queries):
        fold = int(fold_ids[query_index])
        heads = selected_by_fold[fold]
        output_heads = [flat_head_to_output[head] for head in heads]
        reference_head_ids = np.stack(
            [
                reference_ids[
                    query_index,
                    head // num_query_heads,
                    head % num_query_heads,
                ]
                for head in heads
            ]
        )
        reference_ranking = rrf_ranking(
            reference_head_ids, args.target_blocks, num_blocks
        )
        gold = np.asarray(query.get("gold_block_ids", []), dtype=np.int64)
        reference_hits[query_index] = np.isin(gold, reference_ranking).any()
        row = {
            "query_index": query_index,
            "dataset": str(query.get("dataset", "")),
            "fold": fold,
            "reference_hit": int(reference_hits[query_index]),
        }
        for budget in candidate_budgets:
            ranking = rrf_ranking(
                approximate_ids[budget][query_index, output_heads],
                args.target_blocks,
                num_blocks,
            )
            approximate_hits[budget][query_index] = np.isin(gold, ranking).any()
            row[f"hit_at_{budget}"] = int(approximate_hits[budget][query_index])
        query_rows.append(row)

    summary = {
        "experiment": "prototype_coarse_exact_zscore_rerank_lodo",
        "contains_synthetic_vectors": False,
        "selection_uses_gold": False,
        "selection_uses_heldout_queries": False,
        "gold_used_only_for_final_recall": True,
        "queries": len(queries),
        "blocks": num_blocks,
        "heads_per_fold": args.heads_per_fold,
        "prototypes": args.prototypes,
        "candidate_budgets": candidate_budgets,
        "candidate_fractions": {
            str(budget): budget / num_blocks for budget in candidate_budgets
        },
        "reference_rrf39_gold_recall": float(reference_hits.mean()),
        "approximate_rrf39_gold_recall": {
            str(budget): float(approximate_hits[budget].mean())
            for budget in candidate_budgets
        },
        "hit_disagreements_vs_reference": {
            str(budget): int(
                np.sum(approximate_hits[budget] != reference_hits)
            )
            for budget in candidate_budgets
        },
        "paired_wins_vs_reference": {
            str(budget): int(
                np.sum(approximate_hits[budget] & ~reference_hits)
            )
            for budget in candidate_budgets
        },
        "paired_losses_vs_reference": {
            str(budget): int(
                np.sum(~approximate_hits[budget] & reference_hits)
            )
            for budget in candidate_budgets
        },
        "mean_per_head_exact_top16_recall": {
            str(budget): float(
                np.average(
                    [row[f"exact_top16_recall_at_{budget}"] for row in head_rows],
                    weights=[row["test_queries"] for row in head_rows],
                )
            )
            for budget in candidate_budgets
        },
        "proxy_scoring_is_linear_in_blocks": True,
        "exact_rerank_uses_train_only_exact_prior": True,
        "exact_profile_reference_alignment": {
            "max_score_error": alignment_max_score_error,
            "id_mismatch_slots": alignment_id_mismatches,
            "id_mismatch_fraction": alignment_id_mismatches / alignment_slots,
            "set_mismatch_query_head_pairs": alignment_set_mismatches,
        },
        "total_wall_seconds": time.perf_counter() - total_started,
        "device": str(device),
        "seed": args.seed,
    }
    write_csv(output_dir / "head_results.csv", head_rows)
    write_csv(output_dir / "query_results.csv", query_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
