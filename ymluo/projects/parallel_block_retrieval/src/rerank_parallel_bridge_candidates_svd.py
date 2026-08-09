from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prepare_verified_chained_answer_steps import rewrite_answer_step_with_generated_state
from profile_real_qk import QKCapture, read_jsonl, resolve_dtype
from profile_step_state_q import step_state_text
from rerank_sparse_candidate_blocks_svd import (
    max_attention_diagnostics,
    rank_ids,
    target_rank,
)
from run_global_dynamic_svd_kv_single import capture_query_ids
from train_pairwise_qk_passage_head import (
    column_zscore,
    runtime_passage_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerank a shared candidate pool with parallel bridge-state SVD Q/K."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--profile_dir", required=True)
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument("--hypothesis_rows_path", required=True)
    parser.add_argument("--passage_head_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--candidate_limit", type=int, default=16)
    parser.add_argument("--query_tokens", type=int, default=16)
    parser.add_argument("--svd_rank", type=int, default=32)
    parser.add_argument("--bonus_lambdas", default="0.25,0.5,1.0,2.0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def aggregate_hypothesis_scores(
    scores: np.ndarray,
    *,
    selected_index: int,
    branch_order: Sequence[int],
    bonus_lambdas: Sequence[float],
) -> dict[str, np.ndarray]:
    if scores.ndim != 2:
        raise ValueError("scores must be [hypotheses, candidates]")
    methods = {
        "selected_passage": scores[selected_index],
        "max_passage": scores.max(axis=0),
        "mean_passage": scores.mean(axis=0),
    }
    score_rank = np.empty(len(branch_order), dtype=np.int64)
    for rank, branch_index in enumerate(branch_order):
        score_rank[int(branch_index)] = rank
    confidence = (len(branch_order) - 1 - score_rank).astype(np.float64)
    for value in bonus_lambdas:
        label = str(value).replace(".", "p")
        methods[f"bonus_{label}"] = (
            scores + float(value) * confidence[:, None]
        ).max(axis=0)
    return methods


def summarize(rows: Sequence[dict[str, Any]], methods: Sequence[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "queries": len(rows),
        "candidate_pool_recall_at_16": statistics.fmean(
            0 < int(row["candidate_pool_rank"]) <= 16 for row in rows
        ),
        "selected_bm25_recall_at_3": statistics.fmean(
            0 < int(row["selected_bm25_rank"]) <= 3 for row in rows
        ),
        "mean_q_capture_ms": 1000.0
        * statistics.fmean(float(row["q_capture_seconds"]) for row in rows),
        "mean_rerank_ms": 1000.0
        * statistics.fmean(float(row["rerank_seconds"]) for row in rows),
    }
    for method in methods:
        for budget in (1, 3, 16):
            payload[f"{method}_recall_at_{budget}"] = statistics.fmean(
                0 < int(row["method_ranks"][method]) <= budget for row in rows
            )
    return payload


def main() -> None:
    args = parse_args()
    if min(args.candidate_limit, args.query_tokens, args.svd_rank) <= 0:
        raise ValueError("candidate/query/rank values must be positive")
    bonus_lambdas = [
        float(item.strip()) for item in args.bonus_lambdas.split(",") if item.strip()
    ]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    profile_dir = Path(args.profile_dir)
    profile_summary = json.loads(
        (profile_dir / "summary.json").read_text(encoding="utf-8")
    )
    pair_specs = [dict(item) for item in profile_summary["pair_specs"]]
    block_ids = np.load(profile_dir / "block_ids.npy", mmap_mode="r")
    offsets = {int(block_id): offset for offset, block_id in enumerate(block_ids)}
    svd = np.load(profile_dir / f"svd{args.svd_rank}_k.npy", mmap_mode="r")
    basis_payload = torch.load(
        profile_dir / "basis.pt", map_location="cpu", weights_only=False
    )
    basis = basis_payload["basis"][..., : args.svd_rank].to(
        device=device, dtype=torch.float32
    )
    head_payload = json.loads(Path(args.passage_head_path).read_text(encoding="utf-8"))
    head_parameters = head_payload["methods"]["svd"][
        "resolve_answer_from_bridge"
    ]["runtime_parameters"]

    answer_steps = {
        int(row["query_id"]): row
        for row in read_jsonl(Path(args.step_queries_path))
        if str(row["split"]) == args.split
        and row["step_type"] == "resolve_answer_from_bridge"
    }
    hypothesis_rows = read_jsonl(Path(args.hypothesis_rows_path))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.dtype) if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    capture = QKCapture(model, sorted({int(item["layer"]) for item in pair_specs}))

    output_rows = []
    method_names: list[str] = []
    for row_index, source in enumerate(hypothesis_rows, start=1):
        query_id = int(source["query_id"])
        answer_step = answer_steps[query_id]
        candidates = [
            int(item) for item in source["round_robin_candidates"][: args.candidate_limit]
        ]
        if any(block_id not in offsets for block_id in candidates):
            raise KeyError("parallel candidate block is missing from the sparse K profile")
        candidate_offsets = [offsets[block_id] for block_id in candidates]
        svd_keys = torch.from_numpy(
            np.asarray(svd[candidate_offsets], dtype=np.float32)
        ).to(device)

        hypothesis_scores = []
        hypothesis_bm25_scores = []
        q_capture_seconds = 0.0
        rerank_seconds = 0.0
        for hypothesis_index, generated_state in enumerate(source["generated_states"]):
            rewritten = rewrite_answer_step_with_generated_state(
                answer_step, str(generated_state)
            )
            state_ids = tokenizer(
                step_state_text(rewritten), add_special_tokens=False
            )["input_ids"]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            query = capture_query_ids(
                model, capture, pair_specs, state_ids, args.query_tokens, device
            )
            projected_query = torch.einsum(
                "qpd,pdr->qpr", query.float(), basis.float()
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            q_capture_seconds += time.perf_counter() - started

            started = time.perf_counter()
            _scores, profile_scores, _winning = max_attention_diagnostics(
                projected_query, svd_keys
            )
            bm25 = np.asarray(
                source["hypothesis_union_bm25_scores"][hypothesis_index][
                    : len(candidates)
                ],
                dtype=np.float64,
            )[:, None]
            hypothesis_bm25_scores.append(bm25[:, 0].copy())
            features = np.concatenate(
                [
                    column_zscore(bm25),
                    column_zscore(
                        np.asarray(profile_scores.cpu().tolist(), dtype=np.float64)
                    ),
                ],
                axis=1,
            )
            hypothesis_scores.append(
                runtime_passage_scores(features, head_parameters)
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            rerank_seconds += time.perf_counter() - started

        score_matrix = np.stack(hypothesis_scores)
        aggregated = aggregate_hypothesis_scores(
            score_matrix,
            selected_index=int(source["selected_index"]),
            branch_order=source["branch_order"],
            bonus_lambdas=bonus_lambdas,
        )
        method_names = list(aggregated)
        method_candidates = {
            method: rank_ids(candidates, scores.tolist())
            for method, scores in aggregated.items()
        }
        target = int(source["target_block_id"])
        output_rows.append(
            {
                "query_id": query_id,
                "split": args.split,
                "selection_uses_gold": False,
                "training_uses_train_labels_only": True,
                "target_block_id": target,
                "selected_bm25_rank": int(source["selected_rank"]),
                "candidate_pool_rank": target_rank(candidates, target),
                "candidate_pool": candidates,
                "selected_index": int(source["selected_index"]),
                "branch_order": [int(item) for item in source["branch_order"]],
                "hypothesis_candidate_ranks": [
                    [
                        (
                            ranking.index(block_id) + 1
                            if block_id in ranking
                            else args.candidate_limit + 1
                        )
                        for block_id in candidates
                    ]
                    for ranking in source["hypothesis_candidates"]
                ],
                "hypothesis_bm25_scores": [
                    values.tolist() for values in hypothesis_bm25_scores
                ],
                "hypothesis_passage_scores": score_matrix.tolist(),
                "method_ranks": {
                    method: target_rank(ranking, target)
                    for method, ranking in method_candidates.items()
                },
                "method_candidates": method_candidates,
                "q_capture_seconds": q_capture_seconds,
                "rerank_seconds": rerank_seconds,
            }
        )
        if row_index % 20 == 0 or row_index == len(hypothesis_rows):
            print(
                json.dumps({"query": row_index, "queries": len(hypothesis_rows)}),
                flush=True,
            )
    capture.close()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source": "shared-pool reranking with parallel bridge Q and train-fitted SVD32 K",
        "selection_uses_gold": False,
        "split": args.split,
        "candidate_limit": args.candidate_limit,
        "hypotheses_per_query": 3,
        "svd_rank": args.svd_rank,
        "methods": method_names,
        **summarize(output_rows, method_names),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
