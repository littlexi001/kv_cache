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

from benchmark_selected_head_debiased_retrieval import read_selection
from run_all_head_prior_debiased_retrieval import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate train-only query-prototype support scores against frozen "
            "exact Top-K over the complete real 10M block axis."
        )
    )
    parser.add_argument("--packed_profile_dir", required=True)
    parser.add_argument("--query_profiles", required=True)
    parser.add_argument("--selection_csv", required=True)
    parser.add_argument("--queries_jsonl", required=True)
    parser.add_argument("--full_raw_reference_npz", required=True)
    parser.add_argument("--full_zscore_reference_npz", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--support_index_dir")
    parser.add_argument("--exact_prior_dir")
    parser.add_argument("--gate_feature", default="raw_top1_block_diversity")
    parser.add_argument("--heads_per_fold", type=int, default=16)
    parser.add_argument("--prototypes", type=int, default=128)
    parser.add_argument(
        "--candidate_budgets", default="128,256,512,1024,2048,4096,8192,9766"
    )
    parser.add_argument("--exclude_block_prefix_tokens", type=int, default=16)
    parser.add_argument("--block_batch", type=int, default=256)
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


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


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


def build_support_index(
    *,
    profile_dir: Path,
    profile: dict[str, Any],
    layer: int,
    packed_kv_index: int,
    centers: np.ndarray,
    prefix: int,
    block_batch: int,
    output_path: Path,
    device: torch.device,
) -> float:
    started = time.perf_counter()
    num_blocks = int(profile["num_blocks"])
    support = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(centers), num_blocks),
    )
    centers_tensor = torch.from_numpy(centers).to(device=device, dtype=torch.float16)
    for shard in profile["shards"]:
        shard_start = int(shard["block_start"])
        shard_end = int(shard["block_end"])
        source = np.load(
            profile_dir / Path(shard["layer_k_paths"][str(layer)]).name,
            mmap_mode="r",
        )
        for local_start in range(0, shard_end - shard_start, block_batch):
            local_end = min(shard_end - shard_start, local_start + block_batch)
            keys_np = np.asarray(
                source[local_start:local_end, prefix:, packed_kv_index]
            )
            keys = torch.from_numpy(keys_np).to(device=device, dtype=torch.float16)
            flat_keys = keys.reshape(-1, keys.shape[-1]).transpose(0, 1)
            values = torch.matmul(centers_tensor, flat_keys)
            values = values.reshape(
                len(centers), local_end - local_start, keys.shape[1]
            ).amax(dim=-1)
            global_start = shard_start + local_start
            global_end = shard_start + local_end
            support[:, global_start:global_end] = values.cpu().numpy()
    support.flush()
    return time.perf_counter() - started


def top_prefix(scores: np.ndarray, maximum_budget: int) -> np.ndarray:
    if maximum_budget >= len(scores):
        selected = np.arange(len(scores), dtype=np.int64)
    else:
        selected = np.argpartition(-scores, maximum_budget - 1)[:maximum_budget]
    order = np.lexsort((selected, -scores[selected]))
    return selected[order]


def surrogate_score(
    *,
    values: np.ndarray,
    valid: np.ndarray,
    centers_tensor: torch.Tensor,
    support: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    values = values[valid].astype(np.float32)
    norms = np.linalg.norm(values, axis=1)
    directions = values / np.maximum(norms[:, None], 1e-12)
    direction_tensor = torch.from_numpy(directions).to(device=device)
    nearest = (
        torch.matmul(direction_tensor, centers_tensor.transpose(0, 1))
        .argmax(dim=1)
        .cpu()
        .numpy()
    )
    return (
        np.asarray(support[nearest], dtype=np.float32) * norms[:, None]
    ).mean(axis=0)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("full-axis prototype evaluation requires CUDA")
    if args.prototypes <= 0 or args.block_batch <= 0:
        raise ValueError("prototype and block batch counts must be positive")

    candidate_budgets = sorted(
        {int(item) for item in args.candidate_budgets.split(",")}
    )
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    output_dir = Path(args.output_dir)
    support_dir = (
        Path(args.support_index_dir)
        if args.support_index_dir
        else output_dir / "support_indices"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    support_dir.mkdir(parents=True, exist_ok=True)

    profile_dir = Path(args.packed_profile_dir)
    profile = json.loads((profile_dir / "summary.json").read_text(encoding="utf-8"))
    num_blocks = int(profile["num_blocks"])
    if min(candidate_budgets) <= 0 or max(candidate_budgets) > num_blocks:
        raise ValueError("candidate budgets must fit inside the complete block axis")

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
        raw_reference_block_ids = reference["block_ids"].astype(np.int64)
        fold_ids = reference["fold_ids"].astype(np.int64)
        layers = reference["layers"].astype(np.int64)
    with np.load(Path(args.full_zscore_reference_npz)) as reference:
        zscore_reference_block_ids = reference["block_ids"].astype(np.int64)
        zscore_fold_ids = reference["fold_ids"].astype(np.int64)
        zscore_layers = reference["layers"].astype(np.int64)
    if not np.array_equal(fold_ids, zscore_fold_ids) or not np.array_equal(
        layers, zscore_layers
    ):
        raise RuntimeError("raw and z-score references use different folds or layers")

    num_query_heads = int(profile["num_query_heads"])
    num_kv_heads = int(profile["num_kv_heads"])
    repeat_groups = num_query_heads // num_kv_heads
    datasets = np.asarray([str(query["dataset"]) for query in queries])
    exact_prior_means: np.ndarray | None = None
    exact_prior_stds: np.ndarray | None = None
    exact_prior_models: dict[tuple[int, int], int] = {}
    if args.exact_prior_dir:
        exact_prior_dir = Path(args.exact_prior_dir)
        exact_prior_means = np.load(
            exact_prior_dir / "exact_train_mean.npy", mmap_mode="r"
        )
        exact_prior_stds = np.load(
            exact_prior_dir / "exact_train_std.npy", mmap_mode="r"
        )
        with (exact_prior_dir / "models.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                exact_prior_models[(int(row["fold"]), int(row["flat_head"]))] = int(
                    row["model_index"]
                )
    rows: list[dict[str, Any]] = []
    runtimes: list[dict[str, Any]] = []
    maximum_budget = max(candidate_budgets)
    total_started = time.perf_counter()

    for fold in sorted(selected_by_fold):
        train_queries = np.flatnonzero(fold_ids != fold)
        test_queries = np.flatnonzero(fold_ids == fold)
        for flat_head in selected_by_fold[fold]:
            model_started = time.perf_counter()
            layer_index, query_head = divmod(flat_head, num_query_heads)
            layer = int(layers[layer_index])
            original_kv_head = query_head // repeat_groups
            packed_kv_heads = [
                int(item)
                for item in profile["selected_kv_heads_by_layer"][str(layer)]
            ]
            packed_kv_index = packed_kv_heads.index(original_kv_head)

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
            centers = normalize_rows(model.cluster_centers_.astype(np.float32))
            support_path = support_dir / f"fold{fold:02d}_head{flat_head:03d}.npy"
            if support_path.exists():
                existing = np.load(support_path, mmap_mode="r")
                expected = (args.prototypes, num_blocks)
                if existing.shape != expected or existing.dtype != np.float16:
                    raise RuntimeError(f"invalid resumed support index: {support_path}")
                support_seconds = 0.0
                resumed = True
            else:
                partial_path = support_path.with_suffix(".partial.npy")
                partial_path.unlink(missing_ok=True)
                support_seconds = build_support_index(
                    profile_dir=profile_dir,
                    profile=profile,
                    layer=layer,
                    packed_kv_index=packed_kv_index,
                    centers=centers,
                    prefix=args.exclude_block_prefix_tokens,
                    block_batch=args.block_batch,
                    output_path=partial_path,
                    device=device,
                )
                partial_path.replace(support_path)
                resumed = False

            support = np.load(support_path, mmap_mode="r")
            centers_tensor = torch.from_numpy(centers).to(device=device)
            calibration_started = time.perf_counter()
            calibration_sum = np.zeros(num_blocks, dtype=np.float64)
            calibration_square_sum = np.zeros(num_blocks, dtype=np.float64)
            for query_index in train_queries:
                train_score = surrogate_score(
                    values=q[query_index, :, layer_index, query_head],
                    valid=mask[query_index].astype(bool),
                    centers_tensor=centers_tensor,
                    support=support,
                    device=device,
                )
                calibration_sum += train_score
                calibration_square_sum += train_score.astype(np.float64) ** 2
            calibration_mean = calibration_sum / len(train_queries)
            calibration_variance = np.maximum(
                0.0,
                calibration_square_sum / len(train_queries)
                - calibration_mean**2,
            )
            calibration_std = np.maximum(
                np.sqrt(calibration_variance), args.std_epsilon
            )
            calibration_seconds = time.perf_counter() - calibration_started
            exact_prior_index = exact_prior_models.get((fold, int(flat_head)))
            if args.exact_prior_dir and exact_prior_index is None:
                raise RuntimeError(
                    f"exact prior is missing fold={fold}, flat_head={flat_head}"
                )
            score_started = time.perf_counter()
            for query_index in test_queries:
                raw_surrogate = surrogate_score(
                    values=q[query_index, :, layer_index, query_head],
                    valid=mask[query_index].astype(bool),
                    centers_tensor=centers_tensor,
                    support=support,
                    device=device,
                )
                method_scores = {
                    "raw": raw_surrogate,
                    "zscore": (
                        raw_surrogate.astype(np.float64) - calibration_mean
                    )
                    / calibration_std,
                }
                if exact_prior_index is not None:
                    assert exact_prior_means is not None
                    assert exact_prior_stds is not None
                    method_scores["zscore_exact_prior"] = (
                        raw_surrogate.astype(np.float64)
                        - exact_prior_means[exact_prior_index]
                    ) / exact_prior_stds[exact_prior_index]
                exact_by_method = {
                    "raw": raw_reference_block_ids[
                        query_index, layer_index, query_head
                    ],
                    "zscore": zscore_reference_block_ids[
                        query_index, layer_index, query_head
                    ],
                }
                if exact_prior_index is not None:
                    exact_by_method["zscore_exact_prior"] = (
                        zscore_reference_block_ids[
                            query_index, layer_index, query_head
                        ]
                    )
                row: dict[str, Any] = {
                    "fold": fold,
                    "heldout_dataset": str(datasets[query_index]),
                    "query_index": int(query_index),
                    "flat_head": int(flat_head),
                    "layer": layer,
                    "query_head": query_head,
                }
                for method, method_score in method_scores.items():
                    prefix_order = top_prefix(method_score, maximum_budget)
                    exact_top = exact_by_method[method]
                    exact_set = set(int(item) for item in exact_top)
                    row[f"{method}_exact_top1_surrogate_rank"] = (
                        int(np.flatnonzero(prefix_order == exact_top[0])[0] + 1)
                        if exact_top[0] in prefix_order
                        else num_blocks + 1
                    )
                    for budget in candidate_budgets:
                        candidates = set(
                            int(item) for item in prefix_order[:budget]
                        )
                        row[
                            f"{method}_exact_top16_recall_at_{budget}"
                        ] = len(exact_set & candidates) / len(exact_set)
                        row[f"{method}_exact_top1_hit_at_{budget}"] = int(
                            int(exact_top[0]) in candidates
                        )
                rows.append(row)

            score_seconds = time.perf_counter() - score_started
            runtime = {
                "fold": fold,
                "flat_head": int(flat_head),
                "layer": layer,
                "query_head": query_head,
                "test_queries": int(len(test_queries)),
                "support_seconds": support_seconds,
                "calibration_seconds": calibration_seconds,
                "score_seconds": score_seconds,
                "total_seconds": time.perf_counter() - model_started,
                "resumed_support_index": int(resumed),
            }
            runtimes.append(runtime)
            write_csv(output_dir / "query_head_results.partial.csv", rows)
            write_csv(output_dir / "head_runtime.partial.csv", runtimes)
            print(json.dumps(runtime), flush=True)

    methods = ["raw", "zscore"]
    if args.exact_prior_dir:
        methods.append("zscore_exact_prior")
    recall_summary = {
        method: {
            str(budget): float(
                np.mean(
                    [
                        row[f"{method}_exact_top16_recall_at_{budget}"]
                        for row in rows
                    ]
                )
            )
            for budget in candidate_budgets
        }
        for method in methods
    }
    top1_summary = {
        method: {
            str(budget): float(
                np.mean(
                    [
                        row[f"{method}_exact_top1_hit_at_{budget}"]
                        for row in rows
                    ]
                )
            )
            for budget in candidate_budgets
        }
        for method in methods
    }
    summary = {
        "experiment": "lodo_query_prototype_full_10m_block_axis",
        "contains_synthetic_vectors": False,
        "selection_uses_gold": False,
        "selection_uses_heldout_queries": False,
        "reference": "frozen exact raw and z-score per-head Top16 block IDs",
        "queries": int(len(queries)),
        "query_head_pairs": int(len(rows)),
        "fold_head_models": int(len(runtimes)),
        "blocks": num_blocks,
        "prototypes": args.prototypes,
        "candidate_budgets": candidate_budgets,
        "candidate_fractions": {
            str(budget): budget / num_blocks for budget in candidate_budgets
        },
        "exact_top16_recall": recall_summary,
        "exact_top1_candidate_hit": top1_summary,
        "support_index_bytes_per_fold_fp16": (
            args.heads_per_fold * args.prototypes * num_blocks * 2
        ),
        "support_index_bytes_1b_linear_projection_per_fold_fp16": (
            args.heads_per_fold * args.prototypes * 3906250 * 2
        ),
        "scoring_is_linear_in_blocks": True,
        "requires_exact_selected_head_rerank": True,
        "zscore_prior_fit_on_training_queries_only": True,
        "exact_qk_prior_profile_used": bool(args.exact_prior_dir),
        "std_epsilon": args.std_epsilon,
        "total_wall_seconds": time.perf_counter() - total_started,
        "seed": args.seed,
    }
    write_csv(output_dir / "query_head_results.csv", rows)
    write_csv(output_dir / "head_runtime.csv", runtimes)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
