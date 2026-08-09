from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from run_lexical_block_retrieval import (
    descending_ids,
    evaluate_selection,
    group_for_context,
    read_jsonl,
    write_csv,
)
from run_real_qk_retrieval import score_colbert_blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BM25 routing followed by pre-RoPE multi-Q SVD late interaction."
    )
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--profile_dir", required=True)
    parser.add_argument(
        "--query_profiles",
        default="",
        help="Optional query_profiles.pt from a query-only holdout profile.",
    )
    parser.add_argument("--lexical_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_blocks", type=int, default=39)
    parser.add_argument("--record_allocations", default="20,30,39")
    parser.add_argument("--semantic_rank", type=int, default=32)
    parser.add_argument("--global_candidates", type=int, default=782)
    parser.add_argument("--top_record_candidates", type=int, default=5)
    parser.add_argument("--exclude_block_prefix_tokens", type=int, default=16)
    parser.add_argument("--record_margin_threshold", type=float, default=0.04)
    parser.add_argument("--record_routing_csv")
    parser.add_argument("--record_score_csv")
    parser.add_argument(
        "--multi_record_schedules",
        default=(
            "head=20,8,4,2,1,1,1,1,1;"
            "balanced=8,6,4,3,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1;"
            "deep28=28,4,2,1,1,1,1,1;"
            "deep30=30,3,1,1,1,1,1,1;"
            "deep32=32,1,1,1,1,1,1,1"
        ),
    )
    return parser.parse_args()


def parse_record_schedules(spec: str) -> dict[str, list[int]]:
    schedules: dict[str, list[int]] = {}
    for raw_schedule in spec.split(";"):
        raw_schedule = raw_schedule.strip()
        if not raw_schedule:
            continue
        if "=" not in raw_schedule:
            raise ValueError("record schedule must use name=q1,q2,...")
        name, raw_values = raw_schedule.split("=", maxsplit=1)
        quotas = [int(item) for item in raw_values.split(",") if item.strip()]
        if not name.strip() or not quotas or min(quotas) < 0:
            raise ValueError(f"invalid record schedule: {raw_schedule}")
        schedules[name.strip()] = quotas
    if not schedules:
        raise ValueError("at least one multi-record schedule is required")
    return schedules


def allocate_multi_record_blocks(
    record_order: list[int],
    rankings_by_record: dict[int, list[int]],
    quotas: list[int],
    target_blocks: int,
    fallback_ranking: list[int],
) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()
    for record_id, quota in zip(record_order, quotas):
        for block_id in rankings_by_record.get(record_id, [])[:quota]:
            if len(selected) >= target_blocks:
                return selected
            if block_id not in seen:
                selected.append(block_id)
                seen.add(block_id)
    for block_id in fallback_ranking:
        if len(selected) >= target_blocks:
            break
        if block_id not in seen:
            selected.append(block_id)
            seen.add(block_id)
    return selected


def load_svd_index(
    profile_dir: Path,
    summary: dict[str, Any],
    semantic_rank: int,
    device: torch.device,
) -> torch.Tensor:
    block_count = int(summary["num_blocks"])
    block_tokens = int(summary["block_tokens"])
    profile_count = len(summary["pair_specs"])
    stored_rank = int(summary["svd_rank"])
    if semantic_rank <= 0 or semantic_rank > stored_rank:
        raise ValueError(f"semantic_rank must be in [1, {stored_rank}]")
    output = torch.empty(
        block_count,
        block_tokens,
        profile_count,
        semantic_rank,
        dtype=torch.float16,
        device=device,
    )
    for shard in summary["shards"]:
        start = int(shard["block_start"])
        end = int(shard["block_end"])
        array = np.load(profile_dir / Path(shard["svd_k_path"]).name, mmap_mode="r")
        # The profile shards are read-only memmaps. Copy before exposing them to
        # PyTorch so the tensor never aliases non-writable NumPy storage.
        shard = np.array(array[..., :semantic_rank], copy=True)
        output[start:end].copy_(torch.from_numpy(shard))
    return output


def candidate_ids_for_query(
    *,
    block_scores: np.ndarray,
    record_scores: np.ndarray,
    records: list[dict[str, Any]],
    global_candidates: int,
    top_record_candidates: int,
) -> tuple[list[int], list[int]]:
    ranked_blocks = descending_ids(block_scores)
    ranked_records = descending_ids(record_scores)
    candidates = set(ranked_blocks[:global_candidates])
    for record_id in ranked_records[:top_record_candidates]:
        record = records[record_id]
        start = int(record["block_start"])
        count = int(record["block_count"])
        candidates.update(range(start, start + count))
    return sorted(candidates), ranked_records


def rrf_order(
    candidate_ids: list[int],
    lexical_scores: np.ndarray,
    semantic_scores: np.ndarray,
    constant: float = 60.0,
) -> list[int]:
    ids = np.asarray(candidate_ids, dtype=np.int64)
    lexical_order = np.lexsort((ids, -lexical_scores))
    semantic_order = np.lexsort((ids, -semantic_scores))
    lexical_rank = np.empty(len(ids), dtype=np.int64)
    semantic_rank = np.empty(len(ids), dtype=np.int64)
    lexical_rank[lexical_order] = np.arange(len(ids))
    semantic_rank[semantic_order] = np.arange(len(ids))
    scores = 1.0 / (constant + lexical_rank) + 1.0 / (constant + semantic_rank)
    return ids[np.lexsort((ids, -scores))].tolist()


def main() -> None:
    args = parse_args()
    multi_record_schedules = parse_record_schedules(args.multi_record_schedules)
    allocations = sorted(
        {int(item) for item in args.record_allocations.split(",") if item.strip()}
    )
    corpus_dir = Path(args.corpus_dir)
    profile_dir = Path(args.profile_dir)
    lexical_dir = Path(args.lexical_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    queries = read_jsonl(corpus_dir / "queries.jsonl")
    records = read_jsonl(corpus_dir / "records.jsonl")
    corpus_summary = json.loads((corpus_dir / "summary.json").read_text(encoding="utf-8"))
    profile_summary = json.loads((profile_dir / "summary.json").read_text(encoding="utf-8"))
    if profile_summary.get("profile_space") != "pre_rope_record_qk":
        raise ValueError("Hybrid semantic routing requires a pre_rope_record_qk profile")
    if profile_summary.get("query_vector_mode") != "question_content":
        raise ValueError("Hybrid semantic routing requires question_content query vectors")
    block_scores = np.load(lexical_dir / "block_scores.npy", mmap_mode="r")
    record_scores = np.load(lexical_dir / "record_scores.npy", mmap_mode="r")
    if block_scores.shape != (len(queries), int(corpus_summary["num_blocks"])):
        raise ValueError(f"Unexpected block score shape: {block_scores.shape}")
    if record_scores.shape != (len(queries), len(records)):
        raise ValueError(f"Unexpected record score shape: {record_scores.shape}")
    external_routes: dict[int, dict[str, Any]] = {}
    if args.record_routing_csv:
        with Path(args.record_routing_csv).open("r", encoding="utf-8", newline="") as f:
            external_routes = {
                int(row["query_id"]): row for row in csv.DictReader(f)
            }
        if set(external_routes) != set(range(len(queries))):
            raise ValueError("record_routing_csv must contain exactly one row per query")
    record_nll_rows: dict[int, dict[int, dict[str, Any]]] = {}
    if args.record_score_csv:
        with Path(args.record_score_csv).open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                record_nll_rows.setdefault(int(row["query_id"]), {})[
                    int(row["record_id"])
                ] = row

    block_to_record = np.empty(int(corpus_summary["num_blocks"]), dtype=np.int32)
    source_record_by_start: dict[int, int] = {}
    for record_id, record in enumerate(records):
        start = int(record["block_start"])
        end = start + int(record["block_count"])
        block_to_record[start:end] = record_id
        source_record_by_start[start] = record_id

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    load_started = time.perf_counter()
    keys = load_svd_index(profile_dir, profile_summary, args.semantic_rank, device)
    query_profile_path = Path(args.query_profiles) if args.query_profiles else profile_dir / "query_profiles.pt"
    query_payload = torch.load(query_profile_path, map_location="cpu", weights_only=False)
    query_vectors = query_payload["svd_q"][..., : args.semantic_rank].to(device)
    query_mask = query_payload["mask"].to(device)
    torch.cuda.synchronize(device)
    index_load_seconds = time.perf_counter() - load_started

    rows: list[dict[str, Any]] = []
    scoring_started = time.perf_counter()
    candidate_counts: list[int] = []
    for query_index, query in enumerate(queries):
        candidate_ids, ranked_records = candidate_ids_for_query(
            block_scores=block_scores[query_index],
            record_scores=record_scores[query_index],
            records=records,
            global_candidates=args.global_candidates,
            top_record_candidates=args.top_record_candidates,
        )
        candidate_counts.append(len(candidate_ids))
        candidate_tensor = torch.tensor(candidate_ids, dtype=torch.long, device=device)
        candidate_keys = keys.index_select(0, candidate_tensor)
        semantic_scores = score_colbert_blocks(
            candidate_keys,
            query_vectors[query_index : query_index + 1],
            query_mask[query_index : query_index + 1],
            query_batch=1,
            block_chunk=256,
            exclude_block_prefix_tokens=args.exclude_block_prefix_tokens,
        )[0].cpu().numpy()
        lexical_candidate_scores = np.asarray(
            block_scores[query_index, candidate_ids], dtype=np.float32
        )
        semantic_order = np.asarray(candidate_ids)[
            np.lexsort((np.asarray(candidate_ids), -semantic_scores))
        ].tolist()
        rrf_ranked = rrf_order(
            candidate_ids, lexical_candidate_scores, semantic_scores
        )
        predicted_record = ranked_records[0]
        margin = float(
            record_scores[query_index, ranked_records[0]]
            - record_scores[query_index, ranked_records[1]]
        )
        source_record = source_record_by_start[int(query["block_start"])]

        for method, ranked in [
            (f"hybrid_semantic{args.semantic_rank}", semantic_order[: args.target_blocks]),
            (f"hybrid_rrf{args.semantic_rank}", rrf_ranked[: args.target_blocks]),
        ]:
            rows.append(
                evaluate_selection(
                    method=method,
                    query=query,
                    ranked_ids=ranked,
                    context_ids=group_for_context(ranked, block_to_record),
                    predicted_record=predicted_record,
                    source_record=source_record,
                    record_margin=margin,
                )
            )

        predicted = records[predicted_record]
        record_start = int(predicted["block_start"])
        record_ids = list(range(record_start, record_start + int(predicted["block_count"])))
        semantic_by_id = dict(zip(candidate_ids, semantic_scores.tolist()))
        lexical_record_ids = sorted(
            record_ids,
            key=lambda block_id: (-float(block_scores[query_index, block_id]), block_id),
        )
        record_ids.sort(key=lambda block_id: (-semantic_by_id[block_id], block_id))
        global_lexical_order = descending_ids(block_scores[query_index])

        routed_record_pool = ranked_records[: args.top_record_candidates]
        rrf_rankings_by_record: dict[int, list[int]] = {}
        semantic_rankings_by_record: dict[int, list[int]] = {}
        for routed_record_id in routed_record_pool:
            routed_record = records[routed_record_id]
            routed_start = int(routed_record["block_start"])
            routed_ids = list(
                range(routed_start, routed_start + int(routed_record["block_count"]))
            )
            routed_semantic = np.asarray(
                [semantic_by_id[block_id] for block_id in routed_ids], dtype=np.float32
            )
            routed_lexical = np.asarray(
                [block_scores[query_index, block_id] for block_id in routed_ids],
                dtype=np.float32,
            )
            rrf_rankings_by_record[routed_record_id] = rrf_order(
                routed_ids, routed_lexical, routed_semantic
            )
            semantic_rankings_by_record[routed_record_id] = sorted(
                routed_ids,
                key=lambda block_id: (-semantic_by_id[block_id], block_id),
            )

        route_orders = {"bm25": list(routed_record_pool)}
        if query_index in record_nll_rows:
            scored = record_nll_rows[query_index]
            nll_prefix = sorted(
                (record_id for record_id in routed_record_pool if record_id in scored),
                key=lambda record_id: (
                    float(scored[record_id]["question_nll"]),
                    record_id,
                ),
            )
            route_orders["nll5"] = nll_prefix + [
                record_id
                for record_id in routed_record_pool
                if record_id not in set(nll_prefix)
            ]
        for route_name, record_order in route_orders.items():
            for selector_name, rankings_by_record in (
                ("rrf", rrf_rankings_by_record),
                ("semantic", semantic_rankings_by_record),
            ):
                for schedule_name, quotas in multi_record_schedules.items():
                    ranked = allocate_multi_record_blocks(
                        record_order,
                        rankings_by_record,
                        quotas,
                        args.target_blocks,
                        global_lexical_order,
                    )
                    rows.append(
                        evaluate_selection(
                            method=(
                                f"multirecord_{route_name}_{schedule_name}_{selector_name}"
                                f"_svd{args.semantic_rank}"
                            ),
                            query=query,
                            ranked_ids=ranked,
                            context_ids=group_for_context(ranked, block_to_record),
                            predicted_record=record_order[0],
                            source_record=source_record,
                            record_margin=margin,
                        )
                    )

        for allocation in allocations:
            ranked = record_ids[: min(allocation, args.target_blocks)]
            selected = set(ranked)
            for block_id in global_lexical_order:
                if len(ranked) >= args.target_blocks:
                    break
                if block_id not in selected:
                    ranked.append(block_id)
                    selected.add(block_id)
            rows.append(
                evaluate_selection(
                    method=f"hybrid_record{allocation}_svd{args.semantic_rank}",
                    query=query,
                    ranked_ids=ranked,
                    context_ids=group_for_context(ranked, block_to_record),
                    predicted_record=predicted_record,
                    source_record=source_record,
                    record_margin=margin,
                )
            )

        top_record_score = float(record_scores[query_index, ranked_records[0]])
        relative_margin = margin / max(abs(top_record_score), 1.0e-6)
        if relative_margin >= args.record_margin_threshold:
            risk_ranked = record_ids[: args.target_blocks]
        else:
            fallback_allocation = min(30, args.target_blocks)
            risk_ranked = lexical_record_ids[:fallback_allocation]
        risk_selected = set(risk_ranked)
        for block_id in global_lexical_order:
            if len(risk_ranked) >= args.target_blocks:
                break
            if block_id not in risk_selected:
                risk_ranked.append(block_id)
                risk_selected.add(block_id)
        rows.append(
            evaluate_selection(
                method=f"risk_bm25_svd{args.semantic_rank}",
                query=query,
                ranked_ids=risk_ranked,
                context_ids=group_for_context(risk_ranked, block_to_record),
                predicted_record=predicted_record,
                source_record=source_record,
                record_margin=relative_margin,
            )
        )

        if query_index in external_routes:
            external_record = int(external_routes[query_index]["likelihood_record"])
            external = records[external_record]
            external_start = int(external["block_start"])
            external_ids = list(
                range(external_start, external_start + int(external["block_count"]))
            )
            external_ids.sort(key=lambda block_id: (-semantic_by_id[block_id], block_id))
            external_ranked = external_ids[: args.target_blocks]
            external_selected = set(external_ranked)
            for block_id in global_lexical_order:
                if len(external_ranked) >= args.target_blocks:
                    break
                if block_id not in external_selected:
                    external_ranked.append(block_id)
                    external_selected.add(block_id)
            rows.append(
                evaluate_selection(
                    method=f"deep_ql_record{args.target_blocks}_svd{args.semantic_rank}",
                    query=query,
                    ranked_ids=external_ranked,
                    context_ids=group_for_context(external_ranked, block_to_record),
                    predicted_record=external_record,
                    source_record=source_record,
                    record_margin=relative_margin,
                )
            )

        routed_records = ranked_records[: args.top_record_candidates]
        routed_semantic_scores: list[float] = []
        for record_id in routed_records:
            record = records[record_id]
            start = int(record["block_start"])
            values = [
                semantic_by_id[block_id]
                for block_id in range(start, start + int(record["block_count"]))
            ]
            values.sort(reverse=True)
            routed_semantic_scores.append(statistics.fmean(values[: min(3, len(values))]))
        semantic_values = np.asarray(routed_semantic_scores, dtype=np.float32)
        semantic_order = np.lexsort(
            (np.asarray(routed_records, dtype=np.int64), -semantic_values)
        )
        semantic_ranks = np.empty(len(routed_records), dtype=np.int64)
        semantic_ranks[semantic_order] = np.arange(len(routed_records))
        bm25_ranks = np.arange(len(routed_records), dtype=np.int64)
        route_rrf = 1.0 / (60.0 + bm25_ranks) + 1.0 / (60.0 + semantic_ranks)
        bm25_values = np.asarray(
            [record_scores[query_index, record_id] for record_id in routed_records],
            dtype=np.float32,
        )

        def zscore(values: np.ndarray) -> np.ndarray:
            scale = max(float(values.std()), 1.0e-6)
            return (values - values.mean()) / scale

        route_zscore = zscore(bm25_values) + zscore(semantic_values)
        route_choices = {
            "sem3": routed_records[int(semantic_order[0])],
            "rrf3": routed_records[int(np.lexsort((routed_records, -route_rrf))[0])],
            "z3": routed_records[int(np.lexsort((routed_records, -route_zscore))[0])],
        }
        for route_name, routed_record in route_choices.items():
            routed = records[routed_record]
            routed_start = int(routed["block_start"])
            routed_ids = list(
                range(routed_start, routed_start + int(routed["block_count"]))
            )
            routed_ids.sort(key=lambda block_id: (-semantic_by_id[block_id], block_id))
            ranked = routed_ids[: args.target_blocks]
            selected = set(ranked)
            for block_id in global_lexical_order:
                if len(ranked) >= args.target_blocks:
                    break
                if block_id not in selected:
                    ranked.append(block_id)
                    selected.add(block_id)
            rows.append(
                evaluate_selection(
                    method=(
                        f"hybrid_route_{route_name}_record{args.target_blocks}_svd"
                        f"{args.semantic_rank}"
                    ),
                    query=query,
                    ranked_ids=ranked,
                    context_ids=group_for_context(ranked, block_to_record),
                    predicted_record=routed_record,
                    source_record=source_record,
                    record_margin=margin,
                )
            )
    torch.cuda.synchronize(device)
    scoring_seconds = time.perf_counter() - scoring_started

    methods = sorted({row["method"] for row in rows})
    summaries: list[dict[str, Any]] = []
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        summaries.append(
            {
                "method": method,
                "queries": len(group),
                "source_record_recall": statistics.fmean(
                    row["source_record_recall"] for row in group
                ),
                "record_top1_recall": statistics.fmean(
                    row["record_top1_recall"] for row in group
                ),
                "answer_block_recall": statistics.fmean(
                    row["answer_block_recall"] for row in group
                ),
                "answer_block_mrr": statistics.fmean(
                    row["answer_block_mrr"] for row in group
                ),
            }
        )

    write_csv(output_dir / "query_results.csv", rows, list(rows[0]))
    write_csv(output_dir / "method_summary.csv", summaries, list(summaries[0]))
    summary = {
        "source": "BM25 candidates plus pre-RoPE question-content multi-Q SVD",
        "contains_synthetic_vectors": False,
        "num_queries": len(queries),
        "target_blocks": args.target_blocks,
        "semantic_rank": args.semantic_rank,
        "global_candidates": args.global_candidates,
        "top_record_candidates": args.top_record_candidates,
        "record_margin_threshold": args.record_margin_threshold,
        "record_routing_csv": args.record_routing_csv,
        "mean_candidate_blocks": statistics.fmean(candidate_counts),
        "index_load_seconds": index_load_seconds,
        "scoring_seconds": scoring_seconds,
        "methods": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
