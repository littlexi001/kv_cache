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


def train(task_encoding: str, confidence: str, max_depth: int, min_samples_leaf: int) -> None:
    conf_tag = confidence.replace(".", "")
    output_dir = (
        f"outputs/riskkv_v19_learned_budget_router_v6_noweight_"
        f"covered320_preselection_{task_encoding}_conf{conf_tag}_20260711"
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
        "none",
        "--max_depth",
        str(max_depth),
        "--min_samples_leaf",
        str(min_samples_leaf),
        "--confidence_fallback_threshold",
        confidence,
    ]
    print("TRAIN", output_dir, flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    for confidence in ["0.0", "0.15", "0.25", "0.35", "0.50"]:
        train("both", confidence, max_depth=6, min_samples_leaf=3)
    for confidence in ["0.0", "0.15", "0.25", "0.35", "0.50"]:
        train("family", confidence, max_depth=5, min_samples_leaf=3)


if __name__ == "__main__":
    main()
