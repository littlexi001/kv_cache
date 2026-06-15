"""Summarize a run's JSONL metrics without sending a full server log."""

import argparse
import json
from pathlib import Path


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records in {path}")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--tail", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config_path = args.run_dir / "runtime_config.json"
    records = read_jsonl(args.run_dir / "train_metrics.jsonl")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    tail = records[-min(args.tail, len(records)) :]
    numeric_keys = sorted(
        key for key, value in tail[-1].items() if isinstance(value, (int, float)) and key not in {"step", "time"}
    )
    summary = {
        "run_dir": str(args.run_dir),
        "architecture": config.get("architecture"),
        "architecture_version": config.get("architecture_version"),
        "setting": {
            key: config.get(key)
            for key in (
                "router_input",
                "effective_router_input",
                "center_router_input",
                "router_normalization",
                "num_experts",
                "expert_intermediate_size",
                "head_expert_intermediate_size",
                "head_expert_output_size",
                "head_expert_aggregation",
                "local_window",
                "sink_tokens",
                "seq_len",
                "global_batch_size",
            )
        },
        "first": records[0],
        "last": records[-1],
        "tail_window": len(tail),
        "tail_mean": {
            key: sum(float(record[key]) for record in tail if key in record) / sum(key in record for record in tail)
            for key in numeric_keys
            if any(key in record for record in tail)
        },
    }
    output_path = args.output or args.run_dir / "metrics_summary.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=True, indent=2)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
