from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


TOPICS = ("medicine", "politics", "computer", "space")
METHOD_PATTERNS = {
    "full": "full_{topic}.json",
    "pca64_candidate0.06": "pca_r64_c006_{topic}.json",
    "qkmetric64_candidate0.06": "qkmetric_r64_c006_{topic}.json",
    "qkmetric48_candidate0.06": "qkmetric_r48_c006_{topic}.json",
    "qkmetric64_candidate0.04": "qkmetric_r64_c004_{topic}.json",
}


def read_method_row(path: Path, topic: str) -> dict[str, Any]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    row = next((item for item in rows if item.get("topic") == topic), rows[0])
    online = row.get("online_seconds", row.get("mean_online_seconds"))
    if online is None:
        raise ValueError(f"{path} has no online timing")
    return {
        "topic": topic,
        "nll": float(row["nll"]),
        "ppl": float(row["ppl"]),
        "online_seconds": float(online),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    nll = statistics.fmean(row["nll"] for row in rows)
    return {
        "macro_nll": nll,
        "geometric_ppl": math.exp(nll),
        "mean_online_seconds": statistics.fmean(
            row["online_seconds"] for row in rows
        ),
    }


def selected_retrieval(path: Path, names: tuple[str, ...]) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))["retrieval"]
    return {name: report[name] for name in names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    per_topic: dict[str, dict[str, Any]] = {}
    aggregates: dict[str, dict[str, float]] = {}
    for method, pattern in METHOD_PATTERNS.items():
        rows = [
            read_method_row(args.artifact_dir / pattern.format(topic=topic), topic)
            for topic in TOPICS
        ]
        per_topic[method] = {row["topic"]: row for row in rows}
        aggregates[method] = aggregate(rows)

    full = aggregates["full"]
    for row in aggregates.values():
        row["quality_retention_percent"] = (
            full["geometric_ppl"] / row["geometric_ppl"] * 100.0
        )
        row["speedup_vs_full"] = (
            full["mean_online_seconds"] / row["mean_online_seconds"]
        )

    repeated = []
    for path in sorted(args.artifact_dir.glob("qkmetric_runtime_repeat_gpu*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        repeated.append({row["method"]: row for row in report["macro"]})
    subsystem_methods = (
        "full_sdpa",
        "fixed_pca64_candidate0.06",
        "qkmetric_pca64_candidate0.06",
        "qkmetric_pca48_candidate0.06",
        "qkmetric_pca64_candidate0.04",
    )
    subsystem = {
        method: {
            "pipeline_ms_median": statistics.median(
                repeat[method]["pipeline_ms"] for repeat in repeated
            ),
            "speedup_median": statistics.median(
                repeat[method]["speedup_vs_full_sdpa"] for repeat in repeated
            ),
            "state_ratio_vs_full_kv": repeated[0][method][
                "state_ratio_vs_full_kv"
            ],
            "repeat_count": len(repeated),
        }
        for method in subsystem_methods
    }

    payload = {
        "topics": TOPICS,
        "per_topic": per_topic,
        "aggregate": aggregates,
        "subsystem_128k_layer16": subsystem,
        "retrieval": {
            "llama_multilayer": selected_retrieval(
                args.artifact_dir / "qkmetric_rotation_precision_llama.json",
                (
                    "qkmetric_r48_identity_int4_candidate0.06",
                    "qkmetric_r64_identity_int4_candidate0.04",
                    "qkmetric_r64_identity_int4_candidate0.06",
                ),
            ),
            "qwen128_layer16": selected_retrieval(
                args.artifact_dir / "qkmetric_rotation_precision_qwen128.json",
                (
                    "qkmetric_r48_identity_int4_candidate0.06",
                    "qkmetric_r64_identity_int4_candidate0.04",
                    "qkmetric_r64_identity_int4_candidate0.06",
                ),
            ),
        },
        "negative_experiments": {
            "extreme_weighted": {
                "llama": str(
                    args.artifact_dir / "extreme_weighted_qkmetric_llama.json"
                ),
                "qwen128": str(
                    args.artifact_dir / "extreme_weighted_qkmetric_qwen128.json"
                ),
            },
            "spectral_layer_gate": {
                "llama": str(
                    args.artifact_dir / "qk_spectral_layer_gate_llama.json"
                ),
                "qwen128": str(
                    args.artifact_dir / "qk_spectral_layer_gate_qwen128.json"
                ),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregates, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
