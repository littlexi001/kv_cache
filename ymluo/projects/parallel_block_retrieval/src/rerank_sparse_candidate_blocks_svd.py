from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from profile_real_qk import QKCapture, read_jsonl, resolve_dtype
from profile_step_state_q import step_state_text
from run_global_dynamic_svd_kv_single import capture_query_ids
from run_global_step_block_retrieval import parse_profile_indices, select_profiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerank lexical candidate blocks with full and low-rank real Q/K."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--profile_dir", required=True)
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument("--candidate_rows_path", required=True)
    parser.add_argument("--candidate_field", default="lexical_candidates")
    parser.add_argument(
        "--candidate_scores_field",
        default="lexical_top_scores",
        help="Optional aligned coarse scores; missing scores fall back to candidate rank.",
    )
    parser.add_argument("--candidate_limit", type=int, default=16)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--splits", default="dev,test")
    parser.add_argument("--task_types", default="multihop")
    parser.add_argument("--query_tokens", type=int, default=16)
    parser.add_argument("--svd_rank", type=int, default=32)
    parser.add_argument("--resolve_bridge_profiles", default="")
    parser.add_argument("--resolve_answer_profiles", default="")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def rank_ids(candidate_ids: Sequence[int], scores: Sequence[float]) -> list[int]:
    return [
        int(candidate_ids[index])
        for index in sorted(
            range(len(candidate_ids)),
            key=lambda index: (-float(scores[index]), int(candidate_ids[index])),
        )
    ]


def target_rank(ranked: Sequence[int], target: int) -> int:
    return ranked.index(target) + 1 if target in ranked else 0


def max_attention_scores(query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    block_scores, _, _ = max_attention_diagnostics(query, keys)
    return block_scores


def max_attention_diagnostics(
    query: torch.Tensor, keys: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if query.ndim != 3 or keys.ndim != 4:
        raise ValueError("expected query [tokens, profiles, dim] and keys [blocks, tokens, profiles, dim]")
    if query.shape[1:] != keys.shape[2:]:
        raise ValueError("query and key profile dimensions do not align")
    similarities = torch.einsum(
        "qpd,btpd->qbpt", query.float(), keys.float()
    )
    scale = math.sqrt(float(query.shape[-1]))
    per_query_profile = similarities.amax(dim=3).permute(1, 2, 0)
    profile_scores, winning_query_positions = per_query_profile.max(dim=2)
    profile_scores = profile_scores / scale
    return profile_scores.amax(dim=1), profile_scores, winning_query_positions


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    groups = sorted({(str(row["split"]), str(row["step_type"])) for row in rows})
    for split, step_type in groups:
        group = [
            row
            for row in rows
            if str(row["split"]) == split and str(row["step_type"]) == step_type
        ]
        item: dict[str, Any] = {
            "split": split,
            "step_type": step_type,
            "steps": len(group),
            "mean_q_capture_seconds": statistics.fmean(
                float(row["q_capture_seconds"]) for row in group
            ),
            "mean_candidate_load_seconds": statistics.fmean(
                float(row["candidate_load_seconds"]) for row in group
            ),
            "mean_full128_score_seconds": statistics.fmean(
                float(row["full128_score_seconds"]) for row in group
            ),
            "mean_svd_score_seconds": statistics.fmean(
                float(row["svd_score_seconds"]) for row in group
            ),
        }
        for method in ("candidate", "full128", "svd"):
            ranks = [int(row[f"{method}_rank"]) for row in group]
            for budget in (1, 3, 16):
                item[f"{method}_recall_at_{budget}"] = statistics.fmean(
                    0 < rank <= budget for rank in ranks
                )
            reachable = [rank for rank in ranks if rank > 0]
            item[f"{method}_conditional_mrr"] = (
                statistics.fmean(1.0 / rank for rank in reachable) if reachable else 0.0
            )
        item["svd_full_top1_agreement"] = statistics.fmean(
            row["svd_candidates"][0] == row["full128_candidates"][0]
            for row in group
        )
        output.append(item)
    return output


def main() -> None:
    args = parse_args()
    if min(args.candidate_limit, args.query_tokens, args.svd_rank) <= 0:
        raise ValueError("candidate/query/rank values must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    profile_dir = Path(args.profile_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_summary = json.loads(
        (profile_dir / "summary.json").read_text(encoding="utf-8")
    )
    pair_specs = [dict(item) for item in profile_summary["pair_specs"]]
    profile_count = len(pair_specs)
    routes = {
        "resolve_bridge": parse_profile_indices(
            args.resolve_bridge_profiles, profile_count
        ),
        "resolve_answer_from_bridge": parse_profile_indices(
            args.resolve_answer_profiles, profile_count
        ),
    }
    raw = np.load(profile_dir / "raw_k.npy", mmap_mode="r")
    svd = np.load(profile_dir / f"svd{args.svd_rank}_k.npy", mmap_mode="r")
    block_ids = np.load(profile_dir / "block_ids.npy", mmap_mode="r")
    offsets = {int(block_id): offset for offset, block_id in enumerate(block_ids)}
    basis_payload = torch.load(
        profile_dir / "basis.pt", map_location="cpu", weights_only=False
    )
    basis = basis_payload["basis"][..., : args.svd_rank].to(
        device=device, dtype=torch.float32
    )

    allowed_splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    allowed_tasks = {item.strip() for item in args.task_types.split(",") if item.strip()}
    steps = [
        row
        for row in read_jsonl(Path(args.step_queries_path))
        if str(row["split"]) in allowed_splits
        and str(row["task_type"]) in allowed_tasks
    ]
    steps.sort(key=lambda row: (int(row["query_id"]), int(row["step_index"])))
    expected = {(int(row["query_id"]), int(row["step_index"])) for row in steps}
    candidate_rows = {
        key: row
        for row in read_jsonl(Path(args.candidate_rows_path))
        if (key := (int(row["query_id"]), int(row["step_index"]))) in expected
    }
    if set(candidate_rows) != expected:
        raise ValueError("candidate rows do not exactly cover requested steps")

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

    rows = []
    for index, step in enumerate(steps, start=1):
        key = (int(step["query_id"]), int(step["step_index"]))
        source = candidate_rows[key]
        candidates = [
            int(item) for item in source[args.candidate_field][: args.candidate_limit]
        ]
        source_scores = source.get(args.candidate_scores_field, [])
        bm25_scores = [float(item) for item in source_scores[: len(candidates)]]
        candidate_score_source = args.candidate_scores_field
        if len(bm25_scores) != len(candidates):
            bm25_scores = [float(len(candidates) - rank) for rank in range(len(candidates))]
            candidate_score_source = "rank_surrogate"
        if any(item not in offsets for item in candidates):
            raise KeyError("candidate block was not included in the sparse K profile")
        route = routes[str(step["step_type"])]
        state_ids = tokenizer(
            step_state_text(step), add_special_tokens=False
        )["input_ids"]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        query = capture_query_ids(
            model, capture, pair_specs, state_ids, args.query_tokens, device
        )
        query = select_profiles(query, route, axis=1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        q_capture_seconds = time.perf_counter() - started

        load_started = time.perf_counter()
        candidate_offsets = [offsets[item] for item in candidates]
        raw_keys = torch.from_numpy(
            np.asarray(raw[candidate_offsets], dtype=np.float32)
        ).to(device)
        svd_keys = torch.from_numpy(
            np.asarray(svd[candidate_offsets], dtype=np.float32)
        ).to(device)
        raw_keys = select_profiles(raw_keys, route, axis=2)
        svd_keys = select_profiles(svd_keys, route, axis=2)
        route_basis = select_profiles(basis, route, axis=0)
        projected_query = torch.einsum(
            "qpd,pdr->qpr", query.float(), route_basis.float()
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        candidate_load_seconds = time.perf_counter() - load_started

        started = time.perf_counter()
        (
            full_scores_tensor,
            full_profile_scores_tensor,
            full_winning_query_tensor,
        ) = max_attention_diagnostics(query, raw_keys)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        full_seconds = time.perf_counter() - started
        started = time.perf_counter()
        (
            svd_scores_tensor,
            svd_profile_scores_tensor,
            svd_winning_query_tensor,
        ) = max_attention_diagnostics(projected_query, svd_keys)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        svd_seconds = time.perf_counter() - started
        full_scores = [float(item) for item in full_scores_tensor.cpu().tolist()]
        svd_scores = [float(item) for item in svd_scores_tensor.cpu().tolist()]
        query_token_ids = state_ids[-int(query.shape[0]) :]
        query_token_texts = [
            tokenizer.decode([int(token_id)], skip_special_tokens=False)
            for token_id in query_token_ids
        ]
        full_ranked = rank_ids(candidates, full_scores)
        svd_ranked = rank_ids(candidates, svd_scores)
        target = int(step["target_block_ids"][0])
        row = {
            "query_id": key[0],
            "step_index": key[1],
            "split": str(step["split"]),
            "step_type": str(step["step_type"]),
            "selection_uses_gold": False,
            "target_block_id": target,
            "profile_indices": route,
            "q_capture_seconds": q_capture_seconds,
            "candidate_load_seconds": candidate_load_seconds,
            "full128_score_seconds": full_seconds,
            "svd_score_seconds": svd_seconds,
            "candidate_rank": target_rank(candidates, target),
            "full128_rank": target_rank(full_ranked, target),
            "svd_rank": target_rank(svd_ranked, target),
            "candidate_candidates": candidates,
            "bm25_scores": bm25_scores,
            "candidate_score_source": candidate_score_source,
            "full128_candidates": full_ranked,
            "svd_candidates": svd_ranked,
            "full128_scores": full_scores,
            "svd_scores": svd_scores,
            "full128_profile_scores": full_profile_scores_tensor.cpu().tolist(),
            "svd_profile_scores": svd_profile_scores_tensor.cpu().tolist(),
            "full128_winning_query_positions": full_winning_query_tensor.cpu().tolist(),
            "svd_winning_query_positions": svd_winning_query_tensor.cpu().tolist(),
            "query_token_ids": [int(item) for item in query_token_ids],
            "query_token_texts": query_token_texts,
        }
        rows.append(row)
        if index % 20 == 0 or index == len(steps):
            print(
                json.dumps(
                    {
                        "step": index,
                        "steps": len(steps),
                        "candidate_rank": row["candidate_rank"],
                        "full128_rank": row["full128_rank"],
                        "svd_rank": row["svd_rank"],
                    }
                ),
                flush=True,
            )
    capture.close()
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    payload = {
        "source": "coarse candidate reranking with real full128 and train-fitted SVD K",
        "contains_synthetic_vectors": False,
        "selection_uses_gold": False,
        "candidate_field": args.candidate_field,
        "candidate_scores_field": args.candidate_scores_field,
        "candidate_limit": args.candidate_limit,
        "svd_rank": args.svd_rank,
        "query_tokens": args.query_tokens,
        "steps": len(rows),
        "summaries": summarize(rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
