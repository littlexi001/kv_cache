from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

from benchmark_selected_head_debiased_retrieval import read_selection
from run_all_head_prior_debiased_retrieval import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a train-only query-prototype Lipschitz support bound on "
            "uniformly sampled real 10M blocks."
        )
    )
    parser.add_argument("--packed_profile_dir", required=True)
    parser.add_argument("--query_profiles", required=True)
    parser.add_argument("--selection_csv", required=True)
    parser.add_argument("--queries_jsonl", required=True)
    parser.add_argument("--full_raw_reference_npz", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gate_feature", default="raw_top1_block_diversity")
    parser.add_argument("--heads_per_fold", type=int, default=16)
    parser.add_argument("--prototypes", type=int, default=128)
    parser.add_argument("--candidate_budgets", default="16,32,64,128")
    parser.add_argument("--sample_blocks", type=int, default=512)
    parser.add_argument("--max_test_queries_per_fold", type=int, default=32)
    parser.add_argument("--exclude_block_prefix_tokens", type=int, default=16)
    parser.add_argument("--query_batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def load_sampled_keys(
    profile_dir: Path,
    profile: dict[str, Any],
    layer: int,
    packed_kv_index: int,
    block_ids: np.ndarray,
    prefix: int,
) -> np.ndarray:
    output = np.empty(
        (len(block_ids), 256 - prefix, 32), dtype=np.float16
    )
    filled = np.zeros(len(block_ids), dtype=bool)
    for shard in profile["shards"]:
        start = int(shard["block_start"])
        end = int(shard["block_end"])
        inside = (block_ids >= start) & (block_ids < end)
        if not np.any(inside):
            continue
        source = np.load(
            profile_dir / Path(shard["layer_k_paths"][str(layer)]).name,
            mmap_mode="r",
        )
        local_ids = block_ids[inside] - start
        output[inside] = source[local_ids, prefix:, packed_kv_index]
        filled[inside] = True
    if not np.all(filled):
        raise RuntimeError("failed to load every sampled block")
    return output


def valid_vectors(
    q: np.ndarray,
    mask: np.ndarray,
    query_indices: np.ndarray,
    layer_index: int,
    query_head: int,
) -> np.ndarray:
    values = q[query_indices, :, layer_index, query_head].astype(np.float32)
    valid = mask[query_indices].astype(bool)
    return values[valid]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("prototype support evaluation requires CUDA")
    if args.prototypes <= 0 or args.sample_blocks <= 0 or args.query_batch <= 0:
        raise ValueError("prototype, block, and batch counts must be positive")
    candidate_budgets = sorted(
        {int(item) for item in args.candidate_budgets.split(",")}
    )
    if min(candidate_budgets) <= 0 or max(candidate_budgets) > args.sample_blocks:
        raise ValueError("candidate budgets must fit inside sampled blocks")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = Path(args.packed_profile_dir)
    profile = json.loads((profile_dir / "summary.json").read_text(encoding="utf-8"))
    selected_by_fold = read_selection(
        Path(args.selection_csv), args.gate_feature, args.heads_per_fold
    )
    payload = torch.load(
        Path(args.query_profiles), map_location="cpu", weights_only=False
    )
    q = payload["svd_q"].numpy()
    mask = payload["mask"].numpy()
    queries = read_jsonl(Path(args.queries_jsonl))
    with np.load(Path(args.full_raw_reference_npz)) as reference:
        reference_scores = reference["scores"]
        fold_ids = reference["fold_ids"].astype(np.int64)
        layers = reference["layers"].astype(np.int64)
    datasets = np.asarray([str(query["dataset"]) for query in queries])
    num_blocks = int(profile["num_blocks"])
    num_query_heads = int(profile["num_query_heads"])
    num_kv_heads = int(profile["num_kv_heads"])
    repeat_groups = num_query_heads // num_kv_heads
    rng = np.random.default_rng(args.seed)
    sampled_blocks = np.sort(
        rng.choice(num_blocks, size=args.sample_blocks, replace=False)
    ).astype(np.int64)
    key_cache: dict[tuple[int, int], np.ndarray] = {}
    rows: list[dict[str, Any]] = []

    for fold in sorted(selected_by_fold):
        train_queries = np.flatnonzero(fold_ids != fold)
        test_queries = np.flatnonzero(fold_ids == fold)
        if len(test_queries) > args.max_test_queries_per_fold:
            fold_rng = np.random.default_rng(args.seed + fold * 100003)
            test_queries = np.sort(
                fold_rng.choice(
                    test_queries,
                    size=args.max_test_queries_per_fold,
                    replace=False,
                )
            )
        for flat_head in selected_by_fold[fold]:
            layer_index, query_head = divmod(flat_head, num_query_heads)
            layer = int(layers[layer_index])
            original_kv_head = query_head // repeat_groups
            packed_kv_heads = [
                int(item)
                for item in profile["selected_kv_heads_by_layer"][str(layer)]
            ]
            packed_kv_index = packed_kv_heads.index(original_kv_head)
            cache_key = (layer, original_kv_head)
            if cache_key not in key_cache:
                key_cache[cache_key] = load_sampled_keys(
                    profile_dir,
                    profile,
                    layer,
                    packed_kv_index,
                    sampled_blocks,
                    args.exclude_block_prefix_tokens,
                )
            keys_np = key_cache[cache_key]

            train = valid_vectors(
                q, mask, train_queries, layer_index, query_head
            )
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
            centers_np = normalize_rows(
                model.cluster_centers_.astype(np.float32)
            )
            centers = torch.from_numpy(centers_np).to(device=device)
            keys = torch.from_numpy(keys_np).to(
                device=device, dtype=torch.float32
            )
            block_norm = torch.linalg.vector_norm(keys, dim=-1).amax(dim=1)
            prototype_support = torch.einsum(
                "md,btd->mbt", centers, keys
            ).amax(dim=-1)

            test_q = q[test_queries, :, layer_index, query_head].astype(np.float32)
            test_mask = mask[test_queries].astype(np.float32)
            thresholds = reference_scores[
                test_queries, layer_index, query_head, -1
            ].astype(np.float32)
            candidate_counts = np.zeros(len(test_queries), dtype=np.int32)
            nearest_cosine_sum = np.zeros(len(test_queries), dtype=np.float64)
            nearest_cosine_count = np.zeros(len(test_queries), dtype=np.int32)
            surrogate_spearman = np.zeros(len(test_queries), dtype=np.float64)
            surrogate_top1 = np.zeros(len(test_queries), dtype=bool)
            surrogate_recalls = {
                budget: np.zeros(len(test_queries), dtype=np.float64)
                for budget in candidate_budgets
            }
            for query_start in range(0, len(test_queries), args.query_batch):
                query_end = min(
                    len(test_queries), query_start + args.query_batch
                )
                values = torch.from_numpy(test_q[query_start:query_end]).to(
                    device=device
                )
                query_mask = torch.from_numpy(
                    test_mask[query_start:query_end]
                ).to(device=device)
                norms = torch.linalg.vector_norm(values, dim=-1)
                directions = values / norms.clamp_min(1e-12)[:, :, None]
                cosine = torch.einsum("qtd,md->qtm", directions, centers).clamp(
                    -1.0, 1.0
                )
                distance = torch.sqrt((2.0 - 2.0 * cosine).clamp_min(0))
                upper_unit = (
                    prototype_support[None, None]
                    + distance[:, :, :, None] * block_norm[None, None, None]
                ).amin(dim=2)
                upper_tokens = norms[:, :, None] * upper_unit
                valid = query_mask.sum(dim=1).clamp_min(1)
                upper_score = (
                    upper_tokens * query_mask[:, :, None]
                ).sum(dim=1) / valid[:, None]
                upper_cpu = upper_score.cpu().numpy()
                candidate_counts[query_start:query_end] = np.sum(
                    upper_cpu >= thresholds[query_start:query_end, None],
                    axis=1,
                ).astype(np.int32)
                cosine_cpu = cosine.amax(dim=-1).cpu().numpy()
                mask_cpu = test_mask[query_start:query_end].astype(bool)
                nearest_cosine_sum[query_start:query_end] = (
                    cosine_cpu * mask_cpu
                ).sum(axis=1)
                nearest_cosine_count[query_start:query_end] = mask_cpu.sum(axis=1)
                nearest = cosine.argmax(dim=-1)
                surrogate_tokens = prototype_support.index_select(
                    0, nearest.reshape(-1)
                ).reshape(
                    query_end - query_start,
                    test_q.shape[1],
                    args.sample_blocks,
                )
                surrogate_score = (
                    norms[:, :, None]
                    * surrogate_tokens
                    * query_mask[:, :, None]
                ).sum(dim=1) / valid[:, None]
                exact_tokens = torch.einsum(
                    "qtd,bkd->qtbk", values, keys
                ).amax(dim=-1)
                exact_score = (
                    exact_tokens * query_mask[:, :, None]
                ).sum(dim=1) / valid[:, None]
                surrogate_cpu = surrogate_score.cpu().numpy()
                exact_cpu = exact_score.cpu().numpy()
                for batch_index in range(query_end - query_start):
                    output_index = query_start + batch_index
                    exact_order = np.lexsort(
                        (sampled_blocks, -exact_cpu[batch_index])
                    )
                    surrogate_order = np.lexsort(
                        (sampled_blocks, -surrogate_cpu[batch_index])
                    )
                    surrogate_top1[output_index] = (
                        exact_order[0] == surrogate_order[0]
                    )
                    exact_ranks = np.empty(args.sample_blocks, dtype=np.float64)
                    surrogate_ranks = np.empty(
                        args.sample_blocks, dtype=np.float64
                    )
                    exact_ranks[exact_order] = np.arange(args.sample_blocks)
                    surrogate_ranks[surrogate_order] = np.arange(
                        args.sample_blocks
                    )
                    surrogate_spearman[output_index] = float(
                        np.corrcoef(exact_ranks, surrogate_ranks)[0, 1]
                    )
                    exact_top = set(int(item) for item in exact_order[:16])
                    for budget in candidate_budgets:
                        surrogate_recalls[budget][output_index] = (
                            len(
                                exact_top
                                & set(
                                    int(item)
                                    for item in surrogate_order[:budget]
                                )
                            )
                            / len(exact_top)
                        )

            for output_index, query_index in enumerate(test_queries):
                fraction = candidate_counts[output_index] / args.sample_blocks
                rows.append(
                    {
                        "fold": fold,
                        "heldout_dataset": str(datasets[query_index]),
                        "query_index": int(query_index),
                        "flat_head": flat_head,
                        "layer": layer,
                        "query_head": query_head,
                        "sample_blocks": args.sample_blocks,
                        "candidate_blocks": int(candidate_counts[output_index]),
                        "candidate_fraction": fraction,
                        "pruned_blocks": int(
                            args.sample_blocks - candidate_counts[output_index]
                        ),
                        "mean_nearest_prototype_cosine": float(
                            nearest_cosine_sum[output_index]
                            / max(nearest_cosine_count[output_index], 1)
                        ),
                        "sample_score_spearman": float(
                            surrogate_spearman[output_index]
                        ),
                        "sample_top1_agreement": int(
                            surrogate_top1[output_index]
                        ),
                        **{
                            f"sample_exact_top16_recall_at_{budget}": float(
                                surrogate_recalls[budget][output_index]
                            )
                            for budget in candidate_budgets
                        },
                    }
                )
            print(
                json.dumps(
                    {
                        "fold": fold,
                        "dataset": str(np.unique(datasets[test_queries])[0]),
                        "flat_head": flat_head,
                    }
                ),
                flush=True,
            )

    write_csv(output_dir / "query_head_sample_results.csv", rows)
    fractions = np.asarray(
        [row["candidate_fraction"] for row in rows], dtype=np.float64
    )
    cosine = np.asarray(
        [row["mean_nearest_prototype_cosine"] for row in rows],
        dtype=np.float64,
    )
    spearman = np.asarray(
        [row["sample_score_spearman"] for row in rows], dtype=np.float64
    )
    serving_index_bytes_10m_fp16 = (
        args.heads_per_fold * num_blocks * args.prototypes * 2
    )
    summary: dict[str, Any] = {
        "experiment": "lodo_query_prototype_lipschitz_bound_sample",
        "contains_synthetic_vectors": False,
        "selection_uses_gold": False,
        "selection_uses_heldout_queries": False,
        "threshold_source": "frozen exact raw Top16 score",
        "bound": (
            "h_b(q) <= norm(q) * min_m[h_b(c_m) + "
            "max_norm_K_b * norm(unit(q)-c_m)]"
        ),
        "queries_evaluated": len(set(int(row["query_index"]) for row in rows)),
        "query_head_pairs": len(rows),
        "fold_head_models": len(selected_by_fold) * args.heads_per_fold,
        "prototypes": args.prototypes,
        "uniform_sample_blocks": args.sample_blocks,
        "mean_candidate_fraction": float(fractions.mean()),
        "median_candidate_fraction": float(np.median(fractions)),
        "p05_candidate_fraction": float(np.percentile(fractions, 5)),
        "minimum_candidate_fraction": float(fractions.min()),
        "mean_pruned_blocks": float(
            args.sample_blocks * (1.0 - fractions.mean())
        ),
        "mean_nearest_prototype_cosine": float(cosine.mean()),
        "mean_sample_score_spearman": float(spearman.mean()),
        "median_sample_score_spearman": float(np.median(spearman)),
        "sample_top1_agreement": float(
            np.mean([row["sample_top1_agreement"] for row in rows])
        ),
        "sample_exact_top16_recall": {
            str(budget): float(
                np.mean(
                    [
                        row[f"sample_exact_top16_recall_at_{budget}"]
                        for row in rows
                    ]
                )
            )
            for budget in candidate_budgets
        },
        "candidate_budgets": candidate_budgets,
        "serving_index_bytes_10m_fp16": serving_index_bytes_10m_fp16,
        "serving_index_bytes_1b_fp16_projection": int(
            serving_index_bytes_10m_fp16
            * (1_000_000_000 / (num_blocks * 256))
        ),
        "seed": args.seed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
