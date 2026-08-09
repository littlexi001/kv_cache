from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Separate common K direction from useful K-mean retrieval geometry."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--profile_dir", required=True)
    parser.add_argument("--retrieval_dir", required=True)
    parser.add_argument("--ppl128", required=True)
    parser.add_argument("--ppl512", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--chunk_blocks", type=int, default=512)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def coherence(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    unit = values / np.maximum(norms, 1e-6)
    return np.linalg.norm(unit.mean(axis=1), axis=-1)


def describe(values: np.ndarray) -> dict[str, Any]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.1)),
        "p90": float(np.quantile(values, 0.9)),
        "per_profile_mean": np.mean(values, axis=0).tolist(),
    }


def random_scope_expectation(
    scope_ids: np.ndarray, metadata: list[dict[str, Any]], topk: int
) -> dict[str, float]:
    total = len(scope_ids)
    counts = {int(scope): int(np.sum(scope_ids == scope)) for scope in np.unique(scope_ids)}
    fractions = []
    any_probabilities = []
    for row in metadata:
        count = counts[int(row["book_index"])]
        fractions.append(count / total)
        miss = 1.0
        for draw in range(topk):
            miss *= (total - count - draw) / (total - draw)
        any_probabilities.append(1.0 - miss)
    return {
        "mean_same_scope_fraction": float(np.mean(fractions)),
        "mean_same_scope_any_probability": float(np.mean(any_probabilities)),
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    profile_dir = Path(args.profile_dir)
    retrieval_dir = Path(args.retrieval_dir)
    profile = json.loads((profile_dir / "summary.json").read_text(encoding="utf-8"))
    payload = torch.load(profile_dir / "basis.pt", map_location="cpu", weights_only=False)
    projected_mean = torch.einsum(
        "pd,pdr->pr", payload["mean"].float(), payload["basis"].float()
    ).numpy()
    token_indices = np.asarray(profile["token_profile_indices"], dtype=np.int64)
    sample_mean = projected_mean[token_indices]
    centered_parts = []
    raw_parts = []
    for shard in profile["shards"]:
        tokens = np.load(shard["token_path"], mmap_mode="r")
        for start in range(0, len(tokens), args.chunk_blocks):
            centered = np.asarray(tokens[start : start + args.chunk_blocks], dtype=np.float32)
            centered_parts.append(coherence(centered))
            raw_parts.append(coherence(centered + sample_mean[None, None, :, :]))
    centered_coherence = np.concatenate(centered_parts, axis=0)
    raw_coherence = np.concatenate(raw_parts, axis=0)

    retrieval_rows = read_jsonl(retrieval_dir / "rows.jsonl")
    lookup = {
        (int(row["query_id"]), int(row["prefix_tokens"]), str(row["method"])): row
        for row in retrieval_rows
    }
    overlap = []
    for suffix in (64, 128, 256, 512):
        for aggregation in ("max", "profile_rrf"):
            jaccards = {8: [], 64: [], 512: []}
            mean_any = []
            token_any = []
            for query_id in range(30):
                mean_row = lookup[
                    (query_id, suffix, f"qk_mean8_cosine_{aggregation}")
                ]
                token_row = lookup[
                    (query_id, suffix, f"qk_token8_cosine_{aggregation}")
                ]
                for topk in jaccards:
                    left = set(mean_row["top_block_ids"][:topk])
                    right = set(token_row["top_block_ids"][:topk])
                    jaccards[topk].append(len(left & right) / len(left | right))
                mean_any.append(float(mean_row["same_scope_any_at_8"]))
                token_any.append(float(token_row["same_scope_any_at_8"]))
            overlap.append(
                {
                    "state_suffix_tokens": suffix,
                    "aggregation": aggregation,
                    "mean_top8_jaccard": float(np.mean(jaccards[8])),
                    "mean_top64_jaccard": float(np.mean(jaccards[64])),
                    "mean_top512_jaccard": float(np.mean(jaccards[512])),
                    "kmean_same_scope_any_at_8": float(np.mean(mean_any)),
                    "tokenmax_same_scope_any_at_8": float(np.mean(token_any)),
                }
            )

    profile_rows = read_jsonl(retrieval_dir / "profile_rows.jsonl")
    mean_profiles = [
        row
        for row in profile_rows
        if row["family"] == "qk_mean56_cosine" and int(row["prefix_tokens"]) == 512
    ]
    by_profile = {}
    for profile_id in range(int(profile["profile_count"])):
        rows = [row for row in mean_profiles if int(row["profile_id"]) == profile_id]
        train = [row for row in rows if int(row["query_id"]) < 15]
        test = [row for row in rows if int(row["query_id"]) >= 15]
        by_profile[profile_id] = {
            "train_fraction": float(
                np.mean([float(row["same_scope_fraction_at_8"]) for row in train])
            ),
            "test_fraction": float(
                np.mean([float(row["same_scope_fraction_at_8"]) for row in test])
            ),
            "train_any": float(np.mean([float(row["same_scope_any_at_8"]) for row in train])),
            "test_any": float(np.mean([float(row["same_scope_any_at_8"]) for row in test])),
        }
    train_scores = [by_profile[index]["train_fraction"] for index in by_profile]
    test_scores = [by_profile[index]["test_fraction"] for index in by_profile]
    train_test = spearmanr(train_scores, test_scores)
    selected_profiles = sorted(
        by_profile,
        key=lambda index: (
            -by_profile[index]["train_fraction"],
            -by_profile[index]["train_any"],
            index,
        ),
    )

    ppl = {}
    for state, path in ((128, args.ppl128), (512, args.ppl512)):
        summary = json.loads(Path(path).read_text(encoding="utf-8"))
        ppl[state] = {
            row["method"]: {
                "ppl": float(row["ppl"]),
                "mean_delta_nll_vs_query_only": float(row["mean_delta_nll_vs_query_only"]),
                "delta_vs_query_only_bootstrap95": row["delta_vs_query_only_bootstrap95"],
            }
            for row in summary["quality"]
            if row["method"]
            in {
                "query_only",
                "random512",
                "bm25_e5_rrf",
                "qk_mean56_cosine_max",
                "qk_mean56_cosine_profile_rrf",
                "qk_mean8_cosine_max",
                "qk_mean8_cosine_profile_rrf",
                "qk_token8_cosine_max",
                "qk_token8_cosine_profile_rrf",
            }
        }

    scope_ids = np.load(data_dir / "base_block_scope_ids.npy", mmap_mode="r")
    metadata = read_jsonl(data_dir / "metadata.jsonl")
    retained = np.asarray(profile["retained_energy"], dtype=np.float64)
    output = {
        "source": "PG19 past-only K-mean geometry, retrieval, and reader utility",
        "protocol": {
            "past_only": True,
            "pre_rope": True,
            "svd_rank": int(profile["svd_rank"]),
            "profiles": int(profile["profile_count"]),
            "sample_token_profiles": len(token_indices),
            "contains_synthetic_vectors": False,
            "selection_uses_target": False,
        },
        "storage": {
            "mean56_bytes": int(profile["total_mean_index_bytes"]),
            "estimated_mean8_bytes": int(
                profile["total_mean_index_bytes"] * len(token_indices) / profile["profile_count"]
            ),
            "token8_bytes": int(profile["total_sample_token_index_bytes"]),
            "token8_to_mean8_ratio": float(
                profile["total_sample_token_index_bytes"]
                / (profile["total_mean_index_bytes"] * len(token_indices) / profile["profile_count"])
            ),
        },
        "svd_retained_energy": {
            "mean": float(np.mean(retained)),
            "median": float(np.median(retained)),
            "min": float(np.min(retained)),
            "max": float(np.max(retained)),
        },
        "sample_profile_directional_coherence": {
            "raw_projected_k": describe(raw_coherence),
            "global_mean_centered_projected_k": describe(centered_coherence),
            "mean_drop_after_centering": float(
                np.mean(raw_coherence) - np.mean(centered_coherence)
            ),
        },
        "random_top8_scope_expectation": random_scope_expectation(
            np.asarray(scope_ids), metadata, 8
        ),
        "kmean_vs_tokenmax_retrieval": overlap,
        "profile_stability": {
            "train_test_spearman": (
                float(train_test.statistic) if np.isfinite(train_test.statistic) else None
            ),
            "top1_train_profile": int(selected_profiles[0]),
            "top1_train_profile_test_any_at_8": by_profile[selected_profiles[0]]["test_any"],
            "top1_train_profile_test_fraction_at_8": by_profile[selected_profiles[0]][
                "test_fraction"
            ],
            "top4_train_profiles": [int(item) for item in selected_profiles[:4]],
            "top4_mean_test_any_at_8": float(
                np.mean([by_profile[item]["test_any"] for item in selected_profiles[:4]])
            ),
            "top4_mean_test_fraction_at_8": float(
                np.mean([by_profile[item]["test_fraction"] for item in selected_profiles[:4]])
            ),
        },
        "reader_ppl": ppl,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
