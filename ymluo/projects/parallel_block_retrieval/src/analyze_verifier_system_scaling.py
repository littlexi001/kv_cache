from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STAGES = (
    "bridge",
    "answer_generation",
    "verifier",
    "model_total",
    "estimated_online_total",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize strongest-verifier same-request GPU scaling."
    )
    parser.add_argument("--summary_paths", required=True)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries: list[dict[str, Any]] = [
        json.loads(Path(item.strip()).read_text(encoding="utf-8"))
        for item in args.summary_paths.split(",")
        if item.strip()
    ]
    summaries.sort(key=lambda item: int(item["world_size"]))
    baseline = next(item for item in summaries if int(item["world_size"]) == 1)
    rows = []
    for summary in summaries:
        world_size = int(summary["world_size"])
        stage_rows = {}
        for stage in STAGES:
            mean = float(summary[stage]["mean"])
            baseline_mean = float(baseline[stage]["mean"])
            speedup = baseline_mean / mean
            stage_rows[stage] = {
                "mean_seconds": mean,
                "median_seconds": float(summary[stage]["median"]),
                "p95_seconds": float(summary[stage]["p95"]),
                "speedup_vs_1gpu": speedup,
                "parallel_efficiency": speedup / world_size,
            }
        rows.append(
            {
                "world_size": world_size,
                "queries": int(summary["queries"]),
                "answer_accuracy": float(summary["answer_accuracy"]),
                "bridge_replay_match_rate": float(
                    summary["bridge_replay_match_rate"]
                ),
                "mean_gpu_seconds_per_query": world_size
                * float(summary["estimated_online_total"]["mean"]),
                "stages": stage_rows,
            }
        )
    payload = {
        "source": "same queries, frozen routing trace, model, prompts, and branches",
        "strict_per_request_wall_clock": True,
        "model_loading_included": False,
        "scaling": rows,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
