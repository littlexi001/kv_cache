from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.stats import betabinom
from transformers import AutoTokenizer

from analyze_state_pointer_query_manifold import pointer_token_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extrapolate real 10M state-pointer gold ranks to larger exchangeable "
            "distractor pools with a Jeffreys beta-binomial posterior predictive model."
        )
    )
    parser.add_argument("--retrieval_dynamics", required=True)
    parser.add_argument("--step_profile", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max_heads", type=int, default=8)
    parser.add_argument("--oracle_directions", type=int, default=8)
    parser.add_argument("--base_blocks", type=int, default=39062)
    parser.add_argument("--block_tokens", type=int, default=256)
    parser.add_argument("--target_tokens", default="9999872,100000000,1000000000")
    parser.add_argument("--topk", default="4,16,39,64,391,512")
    parser.add_argument("--prior_alpha", type=float, default=0.5)
    parser.add_argument("--prior_beta", type=float, default=0.5)
    return parser.parse_args()


def parse_ints(spec: str) -> list[int]:
    return [int(item.strip()) for item in spec.split(",") if item.strip()]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
    }


def predictive_survival(
    ranks: np.ndarray,
    *,
    base_blocks: int,
    target_blocks: int,
    topk: int,
    prior_alpha: float,
    prior_beta: float,
) -> np.ndarray:
    if target_blocks < base_blocks:
        raise ValueError("target blocks cannot be smaller than the observed corpus")
    additional = target_blocks - base_blocks
    thresholds = topk - ranks
    successes = ranks - 1 + prior_alpha
    failures = base_blocks - ranks + prior_beta
    output = np.zeros_like(ranks, dtype=np.float64)
    eligible = thresholds >= 0
    if additional == 0:
        output[eligible] = 1.0
        return output
    output[eligible] = betabinom.cdf(
        thresholds[eligible],
        additional,
        successes[eligible],
        failures[eligible],
    )
    return output


def main() -> None:
    args = parse_args()
    retrieval = torch.load(args.retrieval_dynamics, map_location="cpu", weights_only=False)
    step_profile = torch.load(args.step_profile, map_location="cpu", weights_only=False)
    top_ids = retrieval["top_ids"]
    if args.max_heads <= 0 or args.max_heads > top_ids.shape[1]:
        raise ValueError("max_heads must be within the retrieved head count")
    rank_by_step = {
        0: retrieval["hop1_head_ranks"][:, : args.max_heads].numpy(),
        1: retrieval["hop2_head_ranks"][:, : args.max_heads].numpy(),
    }
    metadata = retrieval["state_metadata"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    step_by_key = {
        (int(step["query_id"]), int(step["step_index"])): (index, step)
        for index, step in enumerate(step_profile["steps"])
    }
    pointer_indices = {
        key: set(
            pointer_token_indices(
                tokenizer=tokenizer,
                step=step,
                token_positions=step_profile["token_positions"][index],
            )
        )
        for key, (index, step) in step_by_key.items()
    }

    state_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for state_id, row in enumerate(metadata):
        key = (int(row["query_id"]), int(row["oracle_step_index"]))
        if int(row["state_token_index"]) in pointer_indices[key]:
            state_groups[key].append(state_id)
    expected_keys = {
        (int(step["query_id"]), int(step["step_index"]))
        for step in step_profile["steps"]
        if str(step["split"]) == "test"
    }
    if set(state_groups) != expected_keys:
        missing = sorted(expected_keys - set(state_groups))[:5]
        extra = sorted(set(state_groups) - expected_keys)[:5]
        raise RuntimeError(f"state-pointer groups mismatch: missing={missing}, extra={extra}")

    base_blocks = args.base_blocks
    maximum_rank = max(
        int(retrieval["hop1_head_ranks"].max()),
        int(retrieval["hop2_head_ranks"].max()),
    )
    if maximum_rank > base_blocks:
        raise ValueError("observed gold rank exceeds base_blocks")
    target_tokens = parse_ints(args.target_tokens)
    target_blocks = [int(tokens // args.block_tokens) for tokens in target_tokens]
    if target_blocks[0] != base_blocks:
        raise ValueError(
            f"first target has {target_blocks[0]} blocks but observed ranks use {base_blocks}"
        )
    topk_values = parse_ints(args.topk)

    rows = []
    for key, state_ids in sorted(state_groups.items()):
        query_id, step_index = key
        ranks = rank_by_step[step_index][state_ids].reshape(-1)
        ranks.sort()
        kept = ranks[: min(args.oracle_directions, len(ranks))]
        rows.append(
            {
                "query_id": query_id,
                "step_index": step_index,
                "pointer_states": len(state_ids),
                "directions": len(ranks),
                "best_rank": int(ranks[0]),
                "oracle_ranks": kept.astype(np.int64),
            }
        )

    summaries = []
    for step_index in (0, 1):
        subset = [row for row in rows if row["step_index"] == step_index]
        rank_matrix = np.stack([row["oracle_ranks"] for row in subset]).astype(np.int64)
        best_ranks = rank_matrix[:, 0]
        for tokens, blocks in zip(target_tokens, target_blocks):
            for topk in topk_values:
                probabilities = predictive_survival(
                    rank_matrix,
                    base_blocks=base_blocks,
                    target_blocks=blocks,
                    topk=topk,
                    prior_alpha=args.prior_alpha,
                    prior_beta=args.prior_beta,
                )
                oracle_best = probabilities.max(axis=1)
                independence = 1.0 - np.prod(1.0 - probabilities, axis=1)
                base_hit = best_ranks <= topk
                summaries.append(
                    {
                        "step_index": step_index,
                        "target_tokens": tokens,
                        "target_blocks": blocks,
                        "scale_vs_10m": blocks / base_blocks,
                        "topk": topk,
                        "observed_10m_best_direction_hit_rate": float(base_hit.mean()),
                        "posterior_oracle_best_survival": float(oracle_best.mean()),
                        "posterior_independence_ensemble_survival": float(
                            independence.mean()
                        ),
                        "posterior_oracle_survival_given_10m_hit": float(
                            oracle_best[base_hit].mean()
                        )
                        if bool(base_hit.any())
                        else math.nan,
                    }
                )

    step_rank_summary = {
        str(step_index): {
            "steps": sum(row["step_index"] == step_index for row in rows),
            "pointer_states_mean": mean(
                row["pointer_states"] for row in rows if row["step_index"] == step_index
            ),
            "directions_mean": mean(
                row["directions"] for row in rows if row["step_index"] == step_index
            ),
            "best_rank_quantiles": quantiles(
                row["best_rank"] for row in rows if row["step_index"] == step_index
            ),
        }
        for step_index in (0, 1)
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            serializable = dict(row)
            serializable["oracle_ranks"] = row["oracle_ranks"].tolist()
            handle.write(json.dumps(serializable, ensure_ascii=False) + "\n")
    summary = {
        "source": "real-10M state-pointer gold-rank scale law",
        "contains_synthetic_vectors": False,
        "selection_uses_gold": True,
        "selection_role": "optimistic diagnostic oracle for scale-law falsification",
        "model_assumption": (
            "new distractor blocks are exchangeable with observed real 10M distractors; "
            "per-direction exceedance rate uses a Jeffreys beta posterior and added-block "
            "counts use the beta-binomial posterior predictive"
        ),
        "dependence_warning": (
            "oracle-best directions are selected with gold; the independence ensemble "
            "ignores cross-head/block correlation and is intentionally optimistic"
        ),
        "retrieval_dynamics": args.retrieval_dynamics,
        "step_profile": args.step_profile,
        "base_blocks": base_blocks,
        "block_tokens": args.block_tokens,
        "max_heads": args.max_heads,
        "oracle_directions": args.oracle_directions,
        "steps": len(rows),
        "queries": len({row["query_id"] for row in rows}),
        "step_rank_summary": step_rank_summary,
        "scale_rows": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
