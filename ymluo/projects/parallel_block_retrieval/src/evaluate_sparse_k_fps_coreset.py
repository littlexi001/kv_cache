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
from rerank_sparse_candidate_blocks_svd import (
    max_attention_diagnostics,
    rank_ids,
    target_rank,
)
from run_global_dynamic_svd_kv_single import capture_query_ids


BUDGETS = (1, 2, 4, 8, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FPS K coresets against exact max-QK on frozen candidates."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--profile_dir", required=True)
    parser.add_argument("--coreset_dir", required=True)
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument("--candidate_rows_path", required=True)
    parser.add_argument("--candidate_field", default="lexical_candidates")
    parser.add_argument("--candidate_limit", type=int, default=16)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--splits", default="dev,test")
    parser.add_argument("--query_tokens", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def recall(rank: int, budget: int) -> bool:
    return 0 < rank <= budget


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for split, step_type in sorted(
        {(str(row["split"]), str(row["step_type"])) for row in rows}
    ):
        group = [
            row
            for row in rows
            if str(row["split"]) == split and str(row["step_type"]) == step_type
        ]
        item: dict[str, Any] = {
            "split": split,
            "step_type": step_type,
            "steps": len(group),
            "exact_recall_at_1": statistics.fmean(
                recall(int(row["exact_rank"]), 1) for row in group
            ),
            "exact_recall_at_3": statistics.fmean(
                recall(int(row["exact_rank"]), 3) for row in group
            ),
        }
        for prototype_budget in BUDGETS:
            prefix = f"fps{prototype_budget}"
            item[f"{prefix}_recall_at_1"] = statistics.fmean(
                recall(int(row[f"{prefix}_rank"]), 1) for row in group
            )
            item[f"{prefix}_recall_at_3"] = statistics.fmean(
                recall(int(row[f"{prefix}_rank"]), 3) for row in group
            )
            item[f"{prefix}_exact_top1_agreement"] = statistics.fmean(
                row[f"{prefix}_candidates"][0] == row["exact_candidates"][0]
                for row in group
            )
            item[f"{prefix}_mean_score_gap"] = statistics.fmean(
                float(row[f"{prefix}_mean_score_gap"]) for row in group
            )
            item[f"{prefix}_safe_keep_fraction_top3"] = statistics.fmean(
                float(row[f"{prefix}_safe_keep_fraction_top3"]) for row in group
            )
            item[f"{prefix}_safe_exact_top3_recall"] = statistics.fmean(
                float(row[f"{prefix}_safe_exact_top3_recall"]) for row in group
            )
        output.append(item)
    return output


def main() -> None:
    args = parse_args()
    if min(args.candidate_limit, args.query_tokens) <= 0:
        raise ValueError("candidate_limit and query_tokens must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    profile_dir = Path(args.profile_dir)
    coreset_dir = Path(args.coreset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_summary = json.loads(
        (profile_dir / "summary.json").read_text(encoding="utf-8")
    )
    coreset_summary = json.loads(
        (coreset_dir / "summary.json").read_text(encoding="utf-8")
    )
    pair_specs = [dict(item) for item in profile_summary["pair_specs"]]
    radius_budgets = [int(item) for item in coreset_summary["radius_budgets"]]
    if tuple(radius_budgets) != BUDGETS:
        raise ValueError("coreset radius budgets do not match evaluator")
    raw = np.load(profile_dir / "raw_k.npy", mmap_mode="r")
    prototypes = np.load(coreset_dir / "prototypes.npy", mmap_mode="r")
    radii = np.load(coreset_dir / "cover_radii.npy", mmap_mode="r")
    block_ids = np.load(profile_dir / "block_ids.npy", mmap_mode="r")
    offsets = {int(block_id): offset for offset, block_id in enumerate(block_ids)}

    allowed_splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    steps = [
        row
        for row in read_jsonl(Path(args.step_queries_path))
        if str(row["split"]) in allowed_splits
    ]
    steps.sort(key=lambda row: (int(row["query_id"]), int(row["step_index"])))
    if args.max_steps > 0:
        steps = steps[: args.max_steps]
    expected = {(int(row["query_id"]), int(row["step_index"])) for row in steps}
    candidate_rows = {
        key: row
        for row in read_jsonl(Path(args.candidate_rows_path))
        if (key := (int(row["query_id"]), int(row["step_index"]))) in expected
    }
    if set(candidate_rows) != expected:
        raise ValueError("candidate rows do not cover requested steps")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=(
            resolve_dtype(args.dtype) if device.type == "cuda" else torch.float32
        ),
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    capture = QKCapture(model, sorted({int(item["layer"]) for item in pair_specs}))

    rows = []
    started = time.perf_counter()
    for index, step in enumerate(steps, start=1):
        key = (int(step["query_id"]), int(step["step_index"]))
        candidates = [
            int(item)
            for item in candidate_rows[key][args.candidate_field][
                : args.candidate_limit
            ]
        ]
        candidate_offsets = [offsets[item] for item in candidates]
        state_ids = tokenizer(step_state_text(step), add_special_tokens=False)[
            "input_ids"
        ]
        query = capture_query_ids(
            model, capture, pair_specs, state_ids, args.query_tokens, device
        )
        raw_keys = torch.from_numpy(
            np.asarray(raw[candidate_offsets], dtype=np.float32)
        ).to(device)
        prototype_keys = torch.from_numpy(
            np.asarray(prototypes[candidate_offsets], dtype=np.float32)
        ).to(device)
        candidate_radii = torch.from_numpy(
            np.asarray(radii[candidate_offsets], dtype=np.float32)
        ).to(device)
        exact_scores, _, _ = max_attention_diagnostics(query, raw_keys)
        exact_ranked = rank_ids(candidates, exact_scores.cpu().tolist())
        target = int(step["target_block_ids"][0])
        row: dict[str, Any] = {
            "query_id": key[0],
            "step_index": key[1],
            "split": str(step["split"]),
            "step_type": str(step["step_type"]),
            "target_block_id": target,
            "selection_uses_gold": False,
            "exact_candidates": exact_ranked,
            "exact_rank": target_rank(exact_ranked, target),
        }
        query_norm = query.float().norm(dim=-1)
        scale = math.sqrt(float(query.shape[-1]))
        exact_top3 = set(exact_ranked[:3])
        for budget_index, prototype_budget in enumerate(BUDGETS):
            keys = prototype_keys[:, :prototype_budget]
            scores, profile_scores, _ = max_attention_diagnostics(query, keys)
            ranked = rank_ids(candidates, scores.cpu().tolist())
            gap = (exact_scores - scores).clamp_min(0)
            bound_addition = (
                query_norm[None, :, :] * candidate_radii[:, None, :, budget_index]
            ).amax(dim=(1, 2)) / scale
            upper = scores + bound_addition
            lower_threshold = torch.topk(scores, k=3).values[-1]
            safe_ids = {
                candidates[offset]
                for offset in torch.nonzero(upper >= lower_threshold)
                .flatten()
                .cpu()
                .tolist()
            }
            prefix = f"fps{prototype_budget}"
            row[f"{prefix}_candidates"] = ranked
            row[f"{prefix}_rank"] = target_rank(ranked, target)
            row[f"{prefix}_mean_score_gap"] = float(gap.mean().item())
            row[f"{prefix}_max_score_gap"] = float(gap.max().item())
            row[f"{prefix}_safe_keep_fraction_top3"] = len(safe_ids) / len(candidates)
            row[f"{prefix}_safe_exact_top3_recall"] = len(
                exact_top3.intersection(safe_ids)
            ) / 3
        rows.append(row)
        if index % 100 == 0 or index == len(steps):
            print(json.dumps({"steps": index, "total": len(steps)}), flush=True)

    elapsed = time.perf_counter() - started
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source": "real max-QK approximation by nested FPS block coresets",
        "contains_synthetic_vectors": False,
        "selection_uses_gold": False,
        "steps": len(rows),
        "candidate_limit": args.candidate_limit,
        "query_tokens": args.query_tokens,
        "prototype_budgets": list(BUDGETS),
        "elapsed_seconds": elapsed,
        "summaries": summarize(rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    capture.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
