from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure whether raw max-QK block rankings are dominated by K norms."
    )
    parser.add_argument("--profile_dir", required=True)
    parser.add_argument("--rows_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--batch_blocks", type=int, default=32)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rank_values(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.argsort(np.argsort(array, kind="stable"), kind="stable").astype(np.float64)


def correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) <= 1:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else 0.0


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
        output.append(
            {
                "split": split,
                "step_type": step_type,
                "steps": len(group),
                "mean_score_norm_pearson": statistics.fmean(
                    row["score_norm_pearson"] for row in group
                ),
                "mean_score_norm_spearman": statistics.fmean(
                    row["score_norm_spearman"] for row in group
                ),
                "mean_qk_top1_norm_percentile": statistics.fmean(
                    row["qk_top1_norm_percentile"] for row in group
                ),
                "mean_target_norm_percentile": statistics.fmean(
                    row["target_norm_percentile"]
                    for row in group
                    if row["target_norm_percentile"] is not None
                ),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    if args.batch_blocks <= 0:
        raise ValueError("batch_blocks must be positive")
    profile_dir = Path(args.profile_dir)
    raw = np.load(profile_dir / "raw_k.npy", mmap_mode="r")
    block_ids = np.load(profile_dir / "block_ids.npy", mmap_mode="r")
    max_norms = np.zeros(len(block_ids), dtype=np.float32)
    mean_norms = np.zeros(len(block_ids), dtype=np.float32)
    for start in range(0, len(block_ids), args.batch_blocks):
        end = min(len(block_ids), start + args.batch_blocks)
        values = np.asarray(raw[start:end], dtype=np.float32)
        norms = np.linalg.norm(values, axis=-1)
        max_norms[start:end] = norms.max(axis=(1, 2))
        mean_norms[start:end] = norms.mean(axis=(1, 2))
    offset = {int(block_id): index for index, block_id in enumerate(block_ids)}

    details = []
    for row in read_jsonl(Path(args.rows_path)):
        candidates = [int(item) for item in row["candidate_candidates"]]
        norms = np.asarray([max_norms[offset[item]] for item in candidates])
        scores = np.asarray(row["full128_scores"], dtype=np.float64)
        qk_top1 = int(row["full128_candidates"][0])
        top1_offset = candidates.index(qk_top1)
        target = int(row["target_block_id"])
        target_percentile = None
        if target in candidates:
            target_offset = candidates.index(target)
            target_percentile = float((norms <= norms[target_offset]).mean())
        details.append(
            {
                "query_id": int(row["query_id"]),
                "step_index": int(row["step_index"]),
                "split": str(row["split"]),
                "step_type": str(row["step_type"]),
                "score_norm_pearson": correlation(scores, norms),
                "score_norm_spearman": correlation(
                    rank_values(scores), rank_values(norms)
                ),
                "qk_top1_norm_percentile": float(
                    (norms <= norms[top1_offset]).mean()
                ),
                "target_norm_percentile": target_percentile,
            }
        )
    payload: dict[str, Any] = {
        "source": "raw max-QK score versus candidate K-norm diagnostic",
        "selection_uses_gold": False,
        "num_profiled_blocks": len(block_ids),
        "global_max_k_norm_mean": float(max_norms.mean()),
        "global_max_k_norm_p95": float(np.percentile(max_norms, 95)),
        "global_mean_k_norm_mean": float(mean_norms.mean()),
        "summaries": summarize(details),
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
