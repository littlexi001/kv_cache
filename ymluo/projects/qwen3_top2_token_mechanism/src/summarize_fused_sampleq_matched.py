from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Matched run roots, optionally written as label=path.",
    )
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    return parser.parse_args()


def load_run(specification: str) -> dict[str, Any]:
    if "=" in specification:
        label, raw_path = specification.split("=", 1)
    else:
        raw_path = specification
        label = Path(raw_path).name
    root = Path(raw_path)
    full = json.loads((root / "full" / "result.json").read_text(encoding="utf-8"))
    sparse_rows = json.loads(
        (root / "sparse" / "summary.json").read_text(encoding="utf-8")
    )
    if len(sparse_rows) != 1:
        raise ValueError(f"expected one sparse row in {root}, got {len(sparse_rows)}")
    sparse = sparse_rows[0]
    full_seconds = float(full["synchronized_model_forward_seconds"])
    sparse_seconds = float(sparse["online_seconds"])
    full_ppl = float(full["ppl"])
    sparse_ppl = float(sparse["ppl"])
    steps = int(full["query_tokens"]) + int(full["eval_tokens"]) - 1
    return {
        "label": label,
        "topic": full["topic"],
        "history_tokens": int(full["history_tokens"]),
        "eval_tokens": int(full["eval_tokens"]),
        "forward_steps": steps,
        "full_ppl": full_ppl,
        "sparse_ppl": sparse_ppl,
        "ppl_change_percent": 100.0 * (sparse_ppl / full_ppl - 1.0),
        "ppl_quality_percent": 100.0 * full_ppl / sparse_ppl,
        "delta_nll": float(sparse["nll"]) - float(full["nll"]),
        "full_prefill_seconds": float(full["prefill_seconds"]),
        "sparse_prefill_seconds": float(sparse["prefill_seconds"]),
        "full_online_seconds": full_seconds,
        "sparse_online_seconds": sparse_seconds,
        "full_ms_per_step": 1000.0 * full_seconds / steps,
        "sparse_ms_per_step": 1000.0 * sparse_seconds / steps,
        "sparse_steady_ms_per_step": 1000.0
        * float(sparse["steady_online_seconds_per_step"]),
        "online_speedup": full_seconds / sparse_seconds,
        "candidate_fraction_mean": float(sparse["candidate_fraction_mean"]),
        "attention_link_ratio": float(sparse["attention_link_ratio"]),
    }


def main() -> None:
    args = parse_args()
    rows = [load_run(specification) for specification in args.runs]
    aggregate = {
        "run_count": len(rows),
        "geometric_ppl_quality_percent": 100.0
        * (
            math.prod(
                row["full_ppl"] / row["sparse_ppl"] for row in rows
            )
            ** (1.0 / len(rows))
        ),
        "mean_delta_nll": fmean(row["delta_nll"] for row in rows),
        "mean_online_speedup": fmean(row["online_speedup"] for row in rows),
        "pooled_online_speedup": sum(row["full_online_seconds"] for row in rows)
        / sum(row["sparse_online_seconds"] for row in rows),
        "mean_candidate_fraction": fmean(
            row["candidate_fraction_mean"] for row in rows
        ),
        "mean_attention_link_ratio": fmean(row["attention_link_ratio"] for row in rows),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps({"rows": rows, "aggregate": aggregate}, indent=2),
        encoding="utf-8",
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": rows, "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
