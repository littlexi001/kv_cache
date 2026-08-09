from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize strict same-query GPU scaling runs.")
    parser.add_argument("--summary_paths", required=True)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(Path(item.strip()).read_text(encoding="utf-8"))
        for item in args.summary_paths.split(",")
        if item.strip()
    ]
    rows.sort(key=lambda item: int(item["world_size"]))
    baseline = next(item for item in rows if int(item["world_size"]) == 1)
    baseline_wall = float(baseline["mean_wall_seconds"])
    scaling = []
    for row in rows:
        world_size = int(row["world_size"])
        wall = float(row["mean_wall_seconds"])
        speedup = baseline_wall / wall
        scaling.append(
            {
                "world_size": world_size,
                "mean_wall_seconds": wall,
                "median_wall_seconds": float(row["median_wall_seconds"]),
                "speedup_vs_1gpu": speedup,
                "parallel_efficiency": speedup / world_size,
                "generated_tokens_per_second": float(
                    row["mean_generated_tokens_per_second"]
                ),
            }
        )
    payload = {
        "source": "same queries, branch set, model, prompts, and decode budget",
        "strict_wall_clock": True,
        "scaling": scaling,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
