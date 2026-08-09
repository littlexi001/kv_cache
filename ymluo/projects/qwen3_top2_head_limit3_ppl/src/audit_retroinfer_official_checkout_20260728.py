#!/usr/bin/env python3
"""Audit the pinned official RetroInfer checkout without importing its kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


OFFICIAL_REPOSITORY = "https://github.com/microsoft/RetrievalAttention.git"
OFFICIAL_COMMIT = "6b1228c346836769da0ed525dadf05bb7010e96b"
WEIGHTED_FLASH_ATTN_COMMIT = "56d96228ada74d6df806b0083bf018d0d57f57e9"
CUTLASS_COMMIT = "e64a9136dd929639e5f7c969fe5af3bf7415cd4f"
REQUIRED_FILES = (
    "README.md",
    "requirements.txt",
    "config/config.py",
    "config/Llama-3.1-8B-Instruct.json",
    "model_hub/LLM.py",
    "model_hub/llama.py",
    "benchmark/longbench/longbench_run.sh",
    "benchmark/longbench/pred.py",
    "benchmark/longbench/eval.py",
    "benchmark/ruler/ruler_run.sh",
    "library/retroinfer/setup.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _git_output(checkout: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(checkout), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_checkout(checkout: Path) -> dict[str, Any]:
    checkout = checkout.resolve()
    missing = [
        relative
        for relative in REQUIRED_FILES
        if not (checkout / relative).is_file()
    ]
    commit = _git_output(checkout, "rev-parse", "HEAD")
    dirty_lines = [
        line
        for line in _git_output(
            checkout,
            "status",
            "--short",
            "--untracked-files=no",
        ).splitlines()
        if line.strip()
    ]

    readme = (
        (checkout / "README.md").read_text(encoding="utf-8")
        if not missing or "README.md" not in missing
        else ""
    )
    config_path = checkout / "config/Llama-3.1-8B-Instruct.json"
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.is_file()
        else {}
    )
    retroinfer = config.get("RetroInfer", {})
    pred_path = checkout / "benchmark/longbench/pred.py"
    pred_source = (
        pred_path.read_text(encoding="utf-8")
        if pred_path.is_file()
        else ""
    )

    checks = {
        "commit_matches": commit == OFFICIAL_COMMIT,
        "working_tree_tracked_clean": not dirty_lines,
        "required_files_present": not missing,
        "readme_identifies_retroinfer": readme.lstrip().startswith(
            "# RetroInfer"
        ),
        "readme_cites_retrievalattention_history": (
            "RetrievalAttention" in readme
        ),
        "native_retrieval_budget_is_0p018": (
            float(retroinfer.get("retrieval_budget", -1.0)) == 0.018
        ),
        "native_estimation_budget_is_0p232": (
            float(retroinfer.get("estimation_budget", -1.0)) == 0.232
        ),
        "native_cache_ratio_is_0p05": (
            float(retroinfer.get("cache_ratio", -1.0)) == 0.05
        ),
        "official_longbench_ignores_eos": "ignore_eos=True" in pred_source,
        "official_longbench_uses_hf_dataset": (
            "load_dataset('THUDM/LongBench'" in pred_source
        ),
    }
    hashes = {
        relative: _sha256(checkout / relative)
        for relative in REQUIRED_FILES
        if (checkout / relative).is_file()
    }
    return {
        "schema": "qksieve_retroinfer_official_checkout_v1",
        "complete": all(checks.values()),
        "system_identity": "RetroInfer",
        "not_system_identity": "original RetrievalAttention implementation",
        "repository": OFFICIAL_REPOSITORY,
        "commit": commit,
        "expected_commit": OFFICIAL_COMMIT,
        "dependency_pins_for_reproduction": {
            "starmys_flash_attention_weighted": (
                WEIGHTED_FLASH_ATTN_COMMIT
            ),
            "cutlass": CUTLASS_COMMIT,
        },
        "checks": checks,
        "missing_files": missing,
        "dirty_tracked_lines": dirty_lines,
        "source_sha256": hashes,
        "fairness_notes": {
            "official_native_protocol": (
                "Run unchanged official scripts for repository reproduction."
            ),
            "aligned_protocol": (
                "Separately align sample IDs, prompt truncation, chat wrapper, "
                "stop tokens, generation limits, and scoring."
            ),
            "latency_rule": (
                "Only compare complete systems on the same hardware and "
                "report CPU memory and CPU-GPU transfer."
            ),
        },
    }


def main() -> None:
    args = parse_args()
    report = audit_checkout(args.checkout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
