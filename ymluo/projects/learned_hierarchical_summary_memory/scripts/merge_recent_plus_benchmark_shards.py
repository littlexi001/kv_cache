from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any


BASE = Path("/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_recent_plus_bench_m4_parallel_20260706")
SHARDS = ["longbench_exact", "longbench_summary", "ruler_4k8k", "ruler_16k"]
OUT = BASE / "merged"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["benchmark"], row["task"], row["method"]), []).append(row)
        groups.setdefault(("__overall__", "__overall__", row["method"]), []).append(row)

    full_tokens: dict[tuple[str, str], float] = {}
    full_seconds: dict[tuple[str, str], float] = {}
    for (bench, task, method), items in groups.items():
        if method == "full_raw":
            full_tokens[(bench, task)] = statistics.mean(float(item["prompt_tokens"]) for item in items)
            full_seconds[(bench, task)] = statistics.mean(float(item["seconds"]) for item in items)

    out: list[dict[str, Any]] = []
    for (bench, task, method), items in sorted(groups.items()):
        key = (bench, task)
        avg_tokens = statistics.mean(float(item["prompt_tokens"]) for item in items)
        avg_seconds = statistics.mean(float(item["seconds"]) for item in items)
        ft = full_tokens.get(key, avg_tokens)
        fs = full_seconds.get(key, avg_seconds)
        out.append(
            {
                "benchmark": bench,
                "task": task,
                "method": method,
                "samples": len(items),
                "exact_accuracy": statistics.mean(float(item["exact_correct"]) for item in items),
                "avg_rouge_l": statistics.mean(float(item["rouge_l"]) for item in items),
                "avg_score": statistics.mean(float(item["score"]) for item in items),
                "avg_prompt_tokens": avg_tokens,
                "token_ratio_vs_full_raw": avg_tokens / ft if ft else 0.0,
                "avg_seconds": avg_seconds,
                "speedup_vs_full_raw": fs / avg_seconds if avg_seconds else 0.0,
            }
        )
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    configs: list[dict[str, Any]] = []
    for shard in SHARDS:
        shard_dir = BASE / shard
        rows.extend(read_csv(shard_dir / "trials.csv"))
        configs.append(json.loads((shard_dir / "summary.json").read_text(encoding="utf-8"))["config"])

    first = dict(configs[0])
    first["output_dir"] = str(OUT)
    first["longbench_tasks"] = [
        "hotpotqa",
        "2wikimqa",
        "musique",
        "passage_retrieval_en",
        "passage_count",
        "qasper",
        "gov_report",
        "multi_news",
    ]
    first["ruler_tasks"] = [
        "niah_single_1",
        "niah_single_2",
        "niah_multikey_1",
        "niah_multiquery",
        "niah_multivalue",
        "vt",
        "cwe",
        "fwe",
    ]
    first["ruler_context_lengths"] = [4096, 8192, 16384]

    summary = summarize(rows)
    write_csv(OUT / "trials.csv", rows)
    write_csv(OUT / "summary.csv", summary)
    (OUT / "summary.json").write_text(
        json.dumps(
            {
                "config": first,
                "shards": SHARDS,
                "num_trials": len(rows),
                "num_cases": len({(row["benchmark"], row["task"], row["case_id"]) for row in rows}),
                "summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"merged_trials={len(rows)}")
    print(f"merged_cases={len({(row['benchmark'], row['task'], row['case_id']) for row in rows})}")
    print(OUT)


if __name__ == "__main__":
    main()
