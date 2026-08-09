from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from verify_qksieve_frozen_evidence_20260728 import (
    FROZEN_METHOD,
    METHOD,
    SCORE_MODE,
)


REFERENCE_METHOD = "full_kv"
EXPECTED_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_squad",
    "qa_hotpot",
)
EXPECTED_LENGTH_SAMPLES = {
    4096: 10,
    8192: 10,
    16384: 10,
    32768: 10,
    65536: 5,
    131072: 5,
}
TIMING_FIELDS = ("online_tpot_seconds", "decode_tpot_seconds")
SOURCE_PATHS = (
    "src/run_sample_calibrated_ruler_20260717.py",
    "src/run_sample_calibrated_longbench_20260717.py",
    "src/run_controlled_public_kv_benchmark_v1.py",
    "src/run_head_top2_targeted_ppl_20260714.py",
    "src/variablebit_spectral_cuda_20260727.py",
    "src/qabs_cuda_kernels.py",
    "src/qksieve_query_cuda_20260728.py",
    "src/verify_qksieve_frozen_evidence_20260728.py",
    "src/summarize_qksieve_ruler_20260728.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict paired summary for the frozen 13-task RULER run."
    )
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--project_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap_resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--expected_length_samples",
        default="",
        help=(
            "Optional comma-separated LENGTH:SAMPLES protocol override. "
            "The empty default preserves the formal 4K--128K protocol."
        ),
    )
    return parser.parse_args()


def parse_expected_length_samples(value: str) -> dict[int, int]:
    if not value.strip():
        return dict(EXPECTED_LENGTH_SAMPLES)
    parsed: dict[int, int] = {}
    for item in value.split(","):
        fields = item.strip().split(":")
        if len(fields) != 2:
            raise ValueError(
                "expected_length_samples entries must be LENGTH:SAMPLES"
            )
        length, samples = (int(field) for field in fields)
        if length <= 0 or samples <= 0:
            raise ValueError("lengths and sample counts must be positive")
        if length in parsed:
            raise ValueError(f"duplicate length: {length}")
        parsed[length] = samples
    return dict(sorted(parsed.items()))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _geomean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(_mean([math.log(value) for value in values]))


def _percentile_interval(values: list[float]) -> list[float]:
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def _configured_budget(history_tokens: int) -> int:
    return min(history_tokens, 1280, max(256, math.ceil(0.06 * history_tokens)))


def _row_time(row: dict[str, str], field: str) -> float:
    generated_tokens = int(row["generated_tokens"])
    if generated_tokens <= 0:
        raise AssertionError("timing row generated no output tokens")
    if field == "online_tpot_seconds":
        return float(row["online_seconds"]) / generated_tokens
    if field == "decode_tpot_seconds":
        return float(row["decode_seconds"]) / generated_tokens
    raise KeyError(field)


def _validate_rows(
    rows: list[dict[str, str]],
    expected_length_samples: dict[int, int],
) -> dict[tuple[str, int, str], dict[str, dict[str, str]]]:
    expected_methods = {REFERENCE_METHOD, METHOD}
    expected_pairs = len(EXPECTED_TASKS) * sum(expected_length_samples.values())
    counts = Counter(row["method"] for row in rows)
    expected_counts = Counter(
        {method: expected_pairs for method in expected_methods}
    )
    if counts != expected_counts:
        raise AssertionError(
            f"method counts differ: expected={expected_counts}, got={counts}"
        )

    grouped: dict[
        tuple[str, int, str], dict[str, dict[str, str]]
    ] = defaultdict(dict)
    for row in rows:
        task = row["base_task"]
        length = int(row["requested_length"])
        sample_id = row["sample_id"]
        key = (task, length, sample_id)
        method = row["method"]
        if method in grouped[key]:
            raise AssertionError(f"duplicate method row: {key}, {method}")
        grouped[key][method] = row

    if len(grouped) != expected_pairs:
        raise AssertionError(
            f"strict pair count differs: expected={expected_pairs}, "
            f"got={len(grouped)}"
        )
    if any(set(pair) != expected_methods for pair in grouped.values()):
        raise AssertionError("one or more RULER examples are not strictly paired")

    observed_tasks = {key[0] for key in grouped}
    if observed_tasks != set(EXPECTED_TASKS):
        raise AssertionError(
            f"task set differs: expected={EXPECTED_TASKS}, "
            f"got={sorted(observed_tasks)}"
        )
    observed_lengths = {key[1] for key in grouped}
    if observed_lengths != set(expected_length_samples):
        raise AssertionError(
            f"length set differs: got={sorted(observed_lengths)}"
        )

    cell_counts = Counter((task, length) for task, length, _ in grouped)
    for task in EXPECTED_TASKS:
        for length, expected in expected_length_samples.items():
            actual = cell_counts[(task, length)]
            if actual != expected:
                raise AssertionError(
                    f"{task}@{length}: expected {expected} pairs, got {actual}"
                )

    for pair in grouped.values():
        sparse = pair[METHOD]
        full = pair[REFERENCE_METHOD]
        if sparse["configured_score_mode"] != SCORE_MODE:
            raise AssertionError(
                f"score mode drift: {sparse['configured_score_mode']}"
            )
        history_tokens = int(sparse["prefix_tokens"])
        expected_budget = _configured_budget(history_tokens)
        if int(sparse["configured_attention_tokens"]) != expected_budget:
            raise AssertionError(
                f"budget drift at history={history_tokens}: "
                f"expected={expected_budget}, "
                f"actual={sparse['configured_attention_tokens']}"
            )
        if int(sparse["suffix_tokens"]) <= 0:
            raise AssertionError("QKSieve RULER row has an empty dense suffix")
        if int(full["prompt_tokens"]) != int(sparse["prompt_tokens"]):
            raise AssertionError("Full and QKSieve prompt lengths differ")
    return grouped


def _cell_summary(
    pairs: list[dict[str, dict[str, str]]],
) -> dict[str, Any]:
    output: dict[str, Any] = {"samples": len(pairs)}
    for method in (REFERENCE_METHOD, METHOD):
        method_rows = [pair[method] for pair in pairs]
        output[method] = {
            "score": _mean([float(row["score"]) for row in method_rows]),
            **{
                field: _mean([_row_time(row, field) for row in method_rows])
                for field in TIMING_FIELDS
            },
        }
    full_score = output[REFERENCE_METHOD]["score"]
    sparse_score = output[METHOD]["score"]
    output["quality_retention"] = (
        sparse_score / full_score if full_score > 0 else None
    )
    for field in TIMING_FIELDS:
        full_time = output[REFERENCE_METHOD][field]
        sparse_time = output[METHOD][field]
        output[field.replace("_seconds", "_speedup")] = (
            full_time / sparse_time if sparse_time > 0 else None
        )
    return output


def _aggregate_cells(
    cells: list[dict[str, Any]],
) -> dict[str, Any]:
    full_macro = _mean([cell[REFERENCE_METHOD]["score"] for cell in cells])
    sparse_macro = _mean([cell[METHOD]["score"] for cell in cells])
    output: dict[str, Any] = {
        "cells": len(cells),
        "full_macro": full_macro,
        "qksieve_macro": sparse_macro,
        "quality_retention": (
            sparse_macro / full_macro if full_macro > 0 else None
        ),
    }
    for field in TIMING_FIELDS:
        speedup_field = field.replace("_seconds", "_speedup")
        output[f"geomean_{speedup_field}"] = _geomean(
            [float(cell[speedup_field]) for cell in cells]
        )
    return output


def _bootstrap(
    cell_pairs: dict[
        tuple[str, int], list[dict[str, dict[str, str]]]
    ],
    selected_cells: list[tuple[str, int]],
    resamples: int,
    rng: np.random.Generator,
) -> dict[str, list[float]]:
    if resamples <= 0:
        return {}
    quality: list[float] = []
    timing: dict[str, list[float]] = {
        field: [] for field in TIMING_FIELDS
    }
    for _ in range(resamples):
        full_scores: list[float] = []
        sparse_scores: list[float] = []
        speedups: dict[str, list[float]] = {
            field: [] for field in TIMING_FIELDS
        }
        for cell_key in selected_cells:
            pairs = cell_pairs[cell_key]
            indices = rng.integers(0, len(pairs), size=len(pairs))
            sampled = [pairs[int(index)] for index in indices]
            full_scores.append(
                _mean(
                    [
                        float(pair[REFERENCE_METHOD]["score"])
                        for pair in sampled
                    ]
                )
            )
            sparse_scores.append(
                _mean(
                    [float(pair[METHOD]["score"]) for pair in sampled]
                )
            )
            for field in TIMING_FIELDS:
                full_time = _mean(
                    [
                        _row_time(pair[REFERENCE_METHOD], field)
                        for pair in sampled
                    ]
                )
                sparse_time = _mean(
                    [_row_time(pair[METHOD], field) for pair in sampled]
                )
                speedups[field].append(full_time / sparse_time)
        full_macro = _mean(full_scores)
        sparse_macro = _mean(sparse_scores)
        quality.append(sparse_macro / full_macro)
        for field in TIMING_FIELDS:
            timing[field].append(_geomean(speedups[field]))

    output = {"quality_retention_95ci": _percentile_interval(quality)}
    for field in TIMING_FIELDS:
        output[
            f"geomean_{field.replace('_seconds', '_speedup')}_95ci"
        ] = _percentile_interval(timing[field])
    return output


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(
    rows: list[dict[str, str]],
    project_root: Path,
    bootstrap_resamples: int,
    seed: int,
    expected_length_samples: dict[int, int] | None = None,
) -> dict[str, Any]:
    expected_length_samples = dict(
        EXPECTED_LENGTH_SAMPLES
        if expected_length_samples is None
        else expected_length_samples
    )
    if not expected_length_samples:
        raise ValueError("expected_length_samples cannot be empty")
    grouped = _validate_rows(rows, expected_length_samples)
    cell_pairs: dict[
        tuple[str, int], list[dict[str, dict[str, str]]]
    ] = defaultdict(list)
    for (task, length, _), pair in grouped.items():
        cell_pairs[(task, length)].append(pair)
    for pairs in cell_pairs.values():
        pairs.sort(key=lambda pair: pair[REFERENCE_METHOD]["sample_id"])

    per_task_length = {
        f"{task}@{length}": {
            "task": task,
            "length": length,
            **_cell_summary(cell_pairs[(task, length)]),
        }
        for task in EXPECTED_TASKS
        for length in expected_length_samples
    }

    rng = np.random.default_rng(seed)
    per_length: dict[str, Any] = {}
    for length in expected_length_samples:
        keys = [(task, length) for task in EXPECTED_TASKS]
        cells = [
            per_task_length[f"{task}@{length}"] for task, _ in keys
        ]
        per_length[str(length)] = {
            **_aggregate_cells(cells),
            **_bootstrap(
                cell_pairs,
                keys,
                bootstrap_resamples,
                rng,
            ),
        }

    all_keys = [
        (task, length)
        for task in EXPECTED_TASKS
        for length in expected_length_samples
    ]
    overall = {
        **_aggregate_cells(list(per_task_length.values())),
        **_bootstrap(
            cell_pairs,
            all_keys,
            bootstrap_resamples,
            rng,
        ),
    }
    source_hashes = {}
    for relative in SOURCE_PATHS:
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        source_hashes[relative] = _sha256(path)
    return {
        "strict_pairs": len(grouped),
        "rows": len(rows),
        "tasks": len(EXPECTED_TASKS),
        "lengths": list(expected_length_samples),
        "counts": dict(Counter(row["method"] for row in rows)),
        "frozen_method": dict(FROZEN_METHOD),
        "protocol": {
            "suite": "RULER official 13-task task set",
            "prompt_wrapper": "llama3",
            "samples_per_task_length": expected_length_samples,
            "formal_protocol": (
                expected_length_samples == EXPECTED_LENGTH_SAMPLES
            ),
            "dense_prompt_suffix": True,
            "full_prediction_stored": True,
            "llama_stop_tokens": (
                "tokenizer EOS plus end_of_text/eom/eot when present"
            ),
            "quality_run_speed_metric": (
                "seconds per actually generated token; fixed-horizon "
                "same-path speed is reported separately"
            ),
            "timing_claim_policy": (
                "Quality-harness timing is diagnostic only and is not used "
                "for the paper's systems speed claims."
            ),
        },
        "source_sha256": source_hashes,
        "overall": overall,
        "per_length": per_length,
        "per_task_length": per_task_length,
    }


def main() -> None:
    args = parse_args()
    with args.input_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = summarize(
        rows,
        args.project_root.resolve(),
        args.bootstrap_resamples,
        args.seed,
        parse_expected_length_samples(args.expected_length_samples),
    )
    summary["input_csv_sha256"] = _sha256(args.input_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
