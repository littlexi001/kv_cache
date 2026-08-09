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
        matches = sorted(
            (ROOT / "outputs").glob(
                f"riskkv_v19_v*_budget_sweep_b{budget}_20260711_budget_sweep_m20_m20_bDyn_pDyn"
            )
        )
        if not matches:
            raise FileNotFoundError(f"Missing budget sweep output for B={budget}")
        args.extend(["--candidate", f"budget_b{budget}={matches[0].relative_to(ROOT).as_posix()}"])
    return args


def train(task_encoding: str, model: str, class_weight: str, threshold: str) -> None:
    thresh_tag = threshold.replace(".", "")
    output_dir = (
        f"outputs/riskkv_v19_learned_budget_router_v7_safety_ladder_"
        f"covered320_preselection_{task_encoding}_{model}_{class_weight}_p{thresh_tag}_20260711"
    )
    command = [
        PYTHON,
        "scripts/train_budget_safety_ladder_router_20260711.py",
        "--reference_dir",
        "outputs/riskkv_v19_v300_v294_no_2wiki_20260711_m100_bDyn_pDyn",
        "--output_dir",
        output_dir,
        *budget_candidate_args(),
        "--min_nonreference_candidates",
        str(len(BUDGETS)),
        "--quality_ratio",
        "1.0",
        "--task_encoding",
        task_encoding,
        "--model",
        model,
        "--class_weight_mode",
        class_weight,
        "--max_depth",
        "7",
        "--min_samples_leaf",
        "4",
        "--safe_probability_threshold",
        threshold,
    ]
    print("TRAIN", output_dir, flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    thresholds = ["0.35", "0.45", "0.55", "0.65", "0.75"]
    for task_encoding in ["both", "family"]:
        for threshold in thresholds:
            train(task_encoding, "random_forest", "balanced", threshold)
        for threshold in ["0.45", "0.55", "0.65"]:
            train(task_encoding, "extra_trees", "balanced", threshold)
        for threshold in ["0.45", "0.55", "0.65"]:
            train(task_encoding, "random_forest", "none", threshold)


if __name__ == "__main__":
    main()
