#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path("/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl")
PYTHON = "/home/fdong/miniconda3/envs/moe/bin/python"
BUDGET = 256


def budget_candidate_args() -> list[str]:
    matches = sorted(
        (ROOT / "outputs").glob(
            f"riskkv_v19_v*_budget_sweep_b{BUDGET}_20260711_budget_sweep_m20_m20_bDyn_pDyn"
        )
    )
    if not matches:
        raise FileNotFoundError(f"Missing budget sweep output for B={BUDGET}")
    return ["--candidate", f"budget_b{BUDGET}={matches[0].relative_to(ROOT).as_posix()}"]


def train(task_encoding: str, class_weight: str, confidence: str, max_depth: int, min_leaf: int) -> None:
    conf_tag = confidence.replace(".", "")
    output_dir = (
        f"outputs/riskkv_v19_learned_budget_router_v8_b256_gate_"
        f"covered320_preselection_{task_encoding}_{class_weight}_d{max_depth}_l{min_leaf}_conf{conf_tag}_20260711"
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
        "1",
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
            for max_depth, min_leaf in [(3, 5), (4, 4), (5, 3)]:
                for confidence in ["0.0", "0.25", "0.35", "0.45", "0.55", "0.65"]:
                    train(task_encoding, class_weight, confidence, max_depth, min_leaf)


if __name__ == "__main__":
    main()
