#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path("/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl")
PYTHON = "/home/fdong/miniconda3/envs/moe/bin/python"
BUDGETS = [256, 384, 512, 768, 1024, 1536, 2048, 3072]


def budget_candidate_args() -> list[str]:
    args: list[str] = []
    for budget in BUDGETS:
        path = ROOT / "outputs" / f"riskkv_v19_budget_sweep_b{budget}_20260711_budget_sweep_m100_m100_bDyn_pDyn"
        if not (path / "task_results.csv").exists():
            raise FileNotFoundError(f"Missing M100 budget sweep task_results.csv for B={budget}: {path}")
        args.extend(["--candidate", f"budget_b{budget}={path.relative_to(ROOT).as_posix()}"])
    return args


def train(task_encoding: str, class_weight: str, confidence: str, max_depth: int, min_leaf: int) -> None:
    conf_tag = confidence.replace(".", "")
    output_dir = (
        f"outputs/riskkv_v19_learned_budget_router_v11_m100_"
        f"preselection_{task_encoding}_{class_weight}_d{max_depth}_l{min_leaf}_conf{conf_tag}_20260711"
    )
    command = [
        PYTHON,
        "scripts/train_learned_budget_router_20260711.py",
        "--reference_dir",
        "outputs/riskkv_v19_v300_v294_no_2wiki_20260711_m100_bDyn_pDyn",
        "--full_dir",
        "outputs/riskkv_fullkv_m100_same_samples_20260710",
        "--output_dir",
        output_dir,
        *budget_candidate_args(),
        "--min_nonreference_candidates",
        str(len(BUDGETS)),
        "--quality_ratio",
        "1.0",
        "--cost_mode",
        "kv",
        "--feature_set",
        "preselection",
        "--task_encoding",
        task_encoding,
        "--model",
        "random_forest",
        "--class_weight_mode",
        class_weight,
        "--max_depth",
        str(max_depth),
        "--min_samples_leaf",
        str(min_leaf),
        "--confidence_fallback_threshold",
        confidence,
    ]
    print("TRAIN", output_dir, flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    for task_encoding in ["both", "family"]:
        for class_weight in ["none", "balanced"]:
            for max_depth, min_leaf in [(4, 8), (5, 6), (6, 4)]:
                for confidence in ["0.25", "0.35", "0.45", "0.55", "0.65", "0.75"]:
                    train(task_encoding, class_weight, confidence, max_depth, min_leaf)


if __name__ == "__main__":
    main()
