#!/usr/bin/env python
"""Combine strictly audited QKSieve-Robust LongBench model transfers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import qksieve_robust_contract_20260810 as contract


DEFAULT_MODELS = "llama31_8b,qwen3_4b,mistral_7b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--expected_pairs", default=160, type=int)
    parser.add_argument("--expected_tasks", default=16, type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(
    run_root: Path,
    models: tuple[str, ...],
    expected_pairs: int,
    expected_tasks: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for tag in models:
        root = run_root / tag
        summary_path = root / "paired_summary.json"
        manifest_path = root / "manifest.txt"
        if not summary_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"incomplete model evidence: {root}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("schema") != "qksieve_robust_longbench_summary_v1":
            raise AssertionError(f"{tag}: not a Robust summary")
        if summary.get("strict_pairs") != expected_pairs:
            raise AssertionError(f"{tag}: strict-pair count mismatch")
        if summary.get("tasks") != expected_tasks:
            raise AssertionError(f"{tag}: task count mismatch")
        if summary.get("full_fallback_count") != 0:
            raise AssertionError(f"{tag}: Full fallback was observed")
        if summary.get("frozen_contract") != contract.contract_payload():
            raise AssertionError(f"{tag}: frozen contract mismatch")
        full = summary["methods"]["full_kv"]
        ours = summary["methods"][contract.METHOD]
        results[tag] = {
            "strict_pairs": summary["strict_pairs"],
            "tasks": summary["tasks"],
            "full_macro": full["macro_score"],
            "qksieve_macro": ours["macro_score"],
            "quality_retention": ours["quality_retention"],
            "quality_retention_95ci": summary["bootstrap"].get(
                "quality_retention_95ci"
            ),
            "mean_prompt_tokens": ours["prompt_tokens"],
            "mean_attention_fraction": summary["attention_fraction_mean"],
            "mean_effective_quantile_samples": summary[
                "effective_sample_count_mean"
            ],
            "manifest_sha256": sha256(manifest_path),
        }
    return {
        "schema": "qksieve_robust_multimodel_summary_v1",
        "frozen_contract": contract.contract_payload(),
        "models": results,
        "minimum_quality_retention": min(
            result["quality_retention"] for result in results.values()
        ),
    }


def main() -> None:
    args = parse_args()
    models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    payload = summarize(
        args.run_root,
        models,
        args.expected_pairs,
        args.expected_tasks,
    )
    output = args.run_root / "multimodel_summary.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
