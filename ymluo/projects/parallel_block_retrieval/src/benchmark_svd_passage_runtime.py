from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from profile_real_qk import QKCapture, read_jsonl, resolve_dtype
from profile_step_state_q import step_state_text
from rerank_sparse_candidate_blocks_svd import (
    max_attention_diagnostics,
    rank_ids,
    target_rank,
)
from run_global_dynamic_svd_kv_single import capture_query_ids
from train_pairwise_qk_passage_head import candidate_features, runtime_passage_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the deployable SVD-only passage-head reranking path."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--profile_dir", required=True)
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument("--candidate_rows_path", required=True)
    parser.add_argument("--passage_head_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--candidate_limit", type=int, default=16)
    parser.add_argument("--query_tokens", type=int, default=16)
    parser.add_argument("--svd_rank", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": 1000.0 * statistics.fmean(values),
        "median_ms": 1000.0 * statistics.median(values),
        "p95_ms": 1000.0 * percentile(values, 95),
    }


def main() -> None:
    args = parse_args()
    if min(args.candidate_limit, args.query_tokens, args.svd_rank, args.max_steps) <= 0:
        raise ValueError("candidate/query/rank/max_steps values must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    profile_dir = Path(args.profile_dir)
    profile_summary = json.loads(
        (profile_dir / "summary.json").read_text(encoding="utf-8")
    )
    pair_specs = [dict(item) for item in profile_summary["pair_specs"]]
    svd = np.load(profile_dir / f"svd{args.svd_rank}_k.npy", mmap_mode="r")
    block_ids = np.load(profile_dir / "block_ids.npy", mmap_mode="r")
    offsets = {int(block_id): index for index, block_id in enumerate(block_ids)}
    basis_payload = torch.load(
        profile_dir / "basis.pt", map_location="cpu", weights_only=False
    )
    basis = basis_payload["basis"][..., : args.svd_rank].to(
        device=device, dtype=torch.float32
    )
    head_payload = json.loads(Path(args.passage_head_path).read_text(encoding="utf-8"))

    steps = [
        row
        for row in read_jsonl(Path(args.step_queries_path))
        if str(row["split"]) == args.split
    ]
    steps.sort(key=lambda row: (int(row["query_id"]), int(row["step_index"])))
    steps = steps[: args.max_steps]
    expected = {(int(row["query_id"]), int(row["step_index"])) for row in steps}
    candidates_by_key = {
        key: row
        for row in read_jsonl(Path(args.candidate_rows_path))
        if (key := (int(row["query_id"]), int(row["step_index"]))) in expected
    }
    if set(candidates_by_key) != expected:
        raise ValueError("candidate rows do not exactly cover benchmark steps")

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
    for step in steps:
        key = (int(step["query_id"]), int(step["step_index"]))
        source = candidates_by_key[key]
        candidates = [
            int(item) for item in source["lexical_candidates"][: args.candidate_limit]
        ]
        state_ids = tokenizer(step_state_text(step), add_special_tokens=False)["input_ids"]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        query = capture_query_ids(
            model, capture, pair_specs, state_ids, args.query_tokens, device
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        q_seconds = time.perf_counter() - started

        started = time.perf_counter()
        projected_query = torch.einsum(
            "qpd,pdr->qpr", query.float(), basis.float()
        )
        candidate_offsets = [offsets[item] for item in candidates]
        svd_keys = torch.from_numpy(
            np.asarray(svd[candidate_offsets], dtype=np.float32)
        ).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        load_seconds = time.perf_counter() - started

        started = time.perf_counter()
        _, profile_scores, _ = max_attention_diagnostics(
            projected_query, svd_keys
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        qk_seconds = time.perf_counter() - started

        row_for_features = {
            "bm25_scores": [
                float(item) for item in source["lexical_top_scores"][: len(candidates)]
            ],
            "svd_profile_scores": profile_scores.cpu().tolist(),
        }
        parameters = head_payload["methods"]["svd"][str(step["step_type"])][
            "runtime_parameters"
        ]
        started = time.perf_counter()
        learned_scores = runtime_passage_scores(
            candidate_features(row_for_features, "svd"), parameters
        )
        ranked = rank_ids(candidates, learned_scores.tolist())
        head_seconds = time.perf_counter() - started
        target = int(step["target_block_ids"][0])
        rows.append(
            {
                "query_id": key[0],
                "step_index": key[1],
                "step_type": str(step["step_type"]),
                "q_capture_seconds": q_seconds,
                "svd_load_project_seconds": load_seconds,
                "svd_qk_seconds": qk_seconds,
                "passage_head_seconds": head_seconds,
                "total_seconds": q_seconds + load_seconds + qk_seconds + head_seconds,
                "rank": target_rank(ranked, target),
            }
        )
    capture.close()
    payload: dict[str, Any] = {
        "source": "deployable SVD-only internal-state passage-head runtime",
        "selection_uses_gold": False,
        "steps": len(rows),
        "candidate_limit": args.candidate_limit,
        "svd_rank": args.svd_rank,
        "latency": {
            field: latency_summary([float(row[field]) for row in rows])
            for field in (
                "q_capture_seconds",
                "svd_load_project_seconds",
                "svd_qk_seconds",
                "passage_head_seconds",
                "total_seconds",
            )
        },
        "summaries": [
            {
                "step_type": step_type,
                "steps": len(group),
                "recall_at_1": statistics.fmean(int(row["rank"]) == 1 for row in group),
                "recall_at_3": statistics.fmean(0 < int(row["rank"]) <= 3 for row in group),
                "recall_at_16": statistics.fmean(int(row["rank"]) > 0 for row in group),
            }
            for step_type in sorted({str(row["step_type"]) for row in rows})
            if (group := [row for row in rows if str(row["step_type"]) == step_type])
        ],
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
