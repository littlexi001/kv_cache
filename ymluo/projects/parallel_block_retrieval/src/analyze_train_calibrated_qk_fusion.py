from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analyze_stepwise_set_utility import mcnemar_exact_p
from rerank_sparse_candidate_blocks_svd import rank_ids, target_rank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune BM25/QK fusion on train and freeze it for dev/test."
    )
    parser.add_argument("--rows_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--alphas", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def zscore(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    scale = float(array.std())
    return (array - float(array.mean())) / max(scale, 1.0e-8)


def fused_rank(row: dict[str, Any], method: str, alpha: float) -> int:
    ids = [int(item) for item in row["candidate_candidates"]]
    bm25 = zscore(row["bm25_scores"])
    qk = zscore(row[f"{method}_scores"])
    scores = alpha * bm25 + (1.0 - alpha) * qk
    return target_rank(rank_ids(ids, scores.tolist()), int(row["target_block_id"]))


def recall_at3(ranks: Sequence[int]) -> float:
    return statistics.fmean(0 < rank <= 3 for rank in ranks)


def select_alpha(
    rows: list[dict[str, Any]], method: str, alphas: Sequence[float]
) -> tuple[float, list[dict[str, float]]]:
    sweep = []
    for alpha in alphas:
        ranks = [fused_rank(row, method, alpha) for row in rows]
        reachable = [rank for rank in ranks if rank > 0]
        sweep.append(
            {
                "alpha": float(alpha),
                "recall_at_3": recall_at3(ranks),
                "mrr": (
                    statistics.fmean(1.0 / rank for rank in reachable)
                    if reachable
                    else 0.0
                ),
            }
        )
    best = max(sweep, key=lambda item: (item["recall_at_3"], item["mrr"], item["alpha"]))
    return float(best["alpha"]), sweep


def evaluate_group(
    rows: list[dict[str, Any]], method: str, alpha: float
) -> dict[str, Any]:
    baseline = [int(row["candidate_rank"]) for row in rows]
    qk = [int(row[f"{method}_rank"]) for row in rows]
    fused = [fused_rank(row, method, alpha) for row in rows]
    baseline_hit = [0 < rank <= 3 for rank in baseline]
    qk_hit = [0 < rank <= 3 for rank in qk]
    fused_hit = [0 < rank <= 3 for rank in fused]
    qk_wins = sum(right and not left for left, right in zip(baseline_hit, qk_hit, strict=True))
    qk_losses = sum(left and not right for left, right in zip(baseline_hit, qk_hit, strict=True))
    fused_wins = sum(
        right and not left for left, right in zip(baseline_hit, fused_hit, strict=True)
    )
    fused_losses = sum(
        left and not right for left, right in zip(baseline_hit, fused_hit, strict=True)
    )
    return {
        "steps": len(rows),
        "alpha": alpha,
        "bm25_recall_at_3": recall_at3(baseline),
        "qk_recall_at_3": recall_at3(qk),
        "fused_recall_at_3": recall_at3(fused),
        "bm25_or_qk_oracle_recall_at_3": statistics.fmean(
            left or right for left, right in zip(baseline_hit, qk_hit, strict=True)
        ),
        "qk_wins_losses": [qk_wins, qk_losses],
        "qk_mcnemar_p": mcnemar_exact_p(qk_wins, qk_losses),
        "fused_wins_losses": [fused_wins, fused_losses],
        "fused_mcnemar_p": mcnemar_exact_p(fused_wins, fused_losses),
    }


def main() -> None:
    args = parse_args()
    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]
    if not alphas or any(not math.isfinite(item) or not 0 <= item <= 1 for item in alphas):
        raise ValueError("alphas must be finite values in [0, 1]")
    rows = read_jsonl(Path(args.rows_path))
    step_types = sorted({str(row["step_type"]) for row in rows})
    payload: dict[str, Any] = {
        "source": "train-calibrated z-score BM25/QK fusion",
        "selection_uses_gold": False,
        "train_labels_used_for_alpha_only": True,
        "candidate_score_sources": sorted(
            {str(row.get("candidate_score_source", "unknown")) for row in rows}
        ),
        "methods": {},
    }
    for method in ("full128", "svd"):
        method_output: dict[str, Any] = {}
        for step_type in step_types:
            train = [
                row
                for row in rows
                if str(row["split"]) == "train" and str(row["step_type"]) == step_type
            ]
            if not train:
                raise ValueError(f"missing train rows for {step_type}")
            alpha, sweep = select_alpha(train, method, alphas)
            evaluations = {}
            for split in ("train", "dev", "test"):
                group = [
                    row
                    for row in rows
                    if str(row["split"]) == split
                    and str(row["step_type"]) == step_type
                ]
                if group:
                    evaluations[split] = evaluate_group(group, method, alpha)
            method_output[step_type] = {
                "selected_alpha": alpha,
                "train_sweep": sweep,
                "evaluations": evaluations,
            }
        payload["methods"][method] = method_output
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
