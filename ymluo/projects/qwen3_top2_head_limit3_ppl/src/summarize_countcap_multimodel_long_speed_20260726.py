from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DIRECT_SCORE_MODE = (
    "pca_int4_chunked_logscale16_sampleq_direct_qkvfused_"
    "qprojscan_qkvsplitauto"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def weighted_mean(
    rows: list[dict[str, str]],
    value_field: str,
    weight_field: str,
) -> float:
    total_weight = sum(float(row[weight_field]) for row in rows)
    return (
        sum(
            float(row[value_field]) * float(row[weight_field])
            for row in rows
        )
        / total_weight
    )


def aggregate_method(rows: list[dict[str, str]]) -> dict[str, float]:
    token_count = sum(float(row["tokens"]) for row in rows)
    step_count = sum(float(row["tokens"]) - 1.0 for row in rows)
    nll = weighted_mean(rows, "nll", "tokens")
    return {
        "cases": float(len(rows)),
        "tokens": token_count,
        "steps": step_count,
        "nll": nll,
        "ppl": math.exp(nll),
        "prefill_seconds": sum(
            float(row["dense_prompt_seconds"]) for row in rows
        ),
        "decode_seconds": sum(
            float(row["sparse_decode_seconds"]) for row in rows
        ),
        "milliseconds_per_step": (
            1000.0
            * sum(float(row["sparse_decode_seconds"]) for row in rows)
            / step_count
        ),
        "configured_attention_tokens": (
            sum(
                float(row["configured_attention_tokens_mean"])
                * (float(row["tokens"]) - 1.0)
                for row in rows
            )
            / step_count
        ),
        "actual_attention_tokens": (
            sum(
                float(row["actual_attention_tokens_mean"])
                * (float(row["tokens"]) - 1.0)
                for row in rows
            )
            / step_count
        ),
        "actual_attention_tokens_min": min(
            float(row["actual_attention_tokens_min"]) for row in rows
        ),
        "actual_attention_tokens_max": max(
            float(row["actual_attention_tokens_max"]) for row in rows
        ),
    }


def validate_frozen_protocol(
    rows: list[dict[str, str]],
    model: str,
    length: int,
) -> None:
    for row in rows:
        prefix = f"{model}/{length}/{row.get('method', '<missing>')}"
        if row.get("cache_mode") != "auto":
            raise ValueError(f"{prefix} did not use cache_mode=auto")
        if row.get("used_preallocated_cache", "").lower() != "true":
            raise ValueError(f"{prefix} did not use preallocated cache")
        if int(row.get("history_tokens", 0)) != length:
            raise ValueError(f"{prefix} has a mismatched history length")
        if row["method"] == "direct_countcap":
            if int(row.get("projection_dim", 0)) != 48:
                raise ValueError(f"{prefix} did not use PCA48")
            if row.get("score_mode") != DIRECT_SCORE_MODE:
                raise ValueError(f"{prefix} has a non-frozen score mode")


def summarize(run_root: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for path in sorted(run_root.glob("*/*/case_summary.csv")):
        relative = path.relative_to(run_root)
        model = relative.parts[0]
        length_name = relative.parts[1]
        if not length_name.startswith("length"):
            continue
        length = int(length_name.split("_", 1)[0].removeprefix("length"))
        grouped[(model, length)].extend(read_rows(path))

    output = []
    for (model, length), rows in sorted(grouped.items()):
        validate_frozen_protocol(rows, model, length)
        methods = {row["method"] for row in rows}
        if methods != {"full_attention", "direct_countcap"}:
            raise ValueError(
                f"{model}/{length} has unexpected methods {methods}"
            )
        full = aggregate_method(
            [row for row in rows if row["method"] == "full_attention"]
        )
        direct = aggregate_method(
            [row for row in rows if row["method"] == "direct_countcap"]
        )
        if full["cases"] != direct["cases"]:
            raise ValueError(f"{model}/{length} is not strictly paired")
        full_prefill_per_case = full["prefill_seconds"] / full["cases"]
        direct_prefill_per_case = direct["prefill_seconds"] / direct["cases"]
        additional_fixed_seconds = (
            direct_prefill_per_case - full_prefill_per_case
        )
        saved_seconds_per_step = (
            full["decode_seconds"] - direct["decode_seconds"]
        ) / full["steps"]
        break_even_steps = (
            max(0.0, additional_fixed_seconds / saved_seconds_per_step)
            if saved_seconds_per_step > 0.0
            else None
        )
        output.append(
            {
                "model": model,
                "history_tokens": length,
                "paired_cases": int(full["cases"]),
                "full_ppl": full["ppl"],
                "direct_ppl": direct["ppl"],
                "delta_nll": direct["nll"] - full["nll"],
                "ppl_retention": full["ppl"] / direct["ppl"],
                "configured_attention_tokens": direct[
                    "configured_attention_tokens"
                ],
                "configured_attention_ratio": (
                    direct["configured_attention_tokens"] / length
                ),
                "actual_attention_tokens": direct[
                    "actual_attention_tokens"
                ],
                "actual_attention_ratio": (
                    direct["actual_attention_tokens"] / length
                ),
                "actual_attention_tokens_min": direct[
                    "actual_attention_tokens_min"
                ],
                "actual_attention_tokens_max": direct[
                    "actual_attention_tokens_max"
                ],
                "full_milliseconds_per_step": full[
                    "milliseconds_per_step"
                ],
                "direct_milliseconds_per_step": direct[
                    "milliseconds_per_step"
                ],
                "full_prefill_seconds_per_case": full_prefill_per_case,
                "direct_prefill_seconds_per_case": direct_prefill_per_case,
                "additional_fixed_seconds_per_case": (
                    additional_fixed_seconds
                ),
                "saved_seconds_per_decode_step": saved_seconds_per_step,
                "break_even_decode_steps": break_even_steps,
                "decode_speedup": (
                    full["decode_seconds"] / direct["decode_seconds"]
                ),
                "protocol_speedup": (
                    (
                        full["prefill_seconds"]
                        + full["decode_seconds"]
                    )
                    / (
                        direct["prefill_seconds"]
                        + direct["decode_seconds"]
                    )
                ),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = summarize(args.run_root)
    expected = {
        ("llama31_8b", 64000),
        ("llama31_8b", 128000),
        ("qwen3_4b", 64000),
        ("qwen3_4b", 128000),
    }
    observed = {
        (str(row["model"]), int(row["history_tokens"])) for row in rows
    }
    if observed != expected:
        raise RuntimeError(
            f"expected model/length pairs {expected}, got {observed}"
        )
    if any(int(row["paired_cases"]) < 2 for row in rows):
        raise RuntimeError("each model/length needs at least two paired cases")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary.csv", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
